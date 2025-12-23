# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict, List, Optional, Union, no_type_check
import asyncio

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj, MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

try:
    from fluxon_py import FluxonKvClientConfig, new_store

    _fluxon_import_error: Optional[BaseException] = None
except Exception as e:
    FluxonKvClientConfig = None  # type: ignore[assignment]
    new_store = None  # type: ignore[assignment]
    _fluxon_import_error = e
    logger.warning(
        "FluxonConnector disabled because `fluxon_py` import failed: %s. "
        "This is expected if Fluxon is not installed; LMCache will raise only when "
        "a `fluxon://` remote_url is used.",
        e,
    )

_FIELD_META = "lmcache_meta"
_FIELD_DATA = "lmcache_data"


class FluxonConnector(RemoteConnector):
    """LMCache remote connector backed by Fluxon KV (via `fluxon_py`)."""

    def __init__(
        self,
        config_path: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ):
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        if _fluxon_import_error is not None:
            raise RuntimeError(
                "FluxonConnector requires the Fluxon Python package. "
                "Install it (for example): `pip install -e ./fluxon` "
                "or ensure `fluxon_py` is importable. "
                f"Import error: {_fluxon_import_error}"
            )
        assert FluxonKvClientConfig is not None
        assert new_store is not None

        cfg = FluxonKvClientConfig.from_file(config_path)
        logger.info("FluxonConnector using config file: %s", config_path)

        store_result = new_store(cfg)
        if not getattr(store_result, "is_ok")():
            err = store_result.unwrap_error()
            raise RuntimeError(f"Failed to initialize Fluxon KV store: {err}")

        self.store = store_result.unwrap()
        logger.info("FluxonConnector initialized Fluxon KV store successfully")

    def _key_to_str(self, key: CacheEngineKey) -> str:
        return key.to_string()

    @staticmethod
    def _unwrap_result(res: Any, *, op: str) -> Any:
        if not getattr(res, "is_ok")():
            err = res.unwrap_error()
            raise RuntimeError(f"Fluxon {op} failed: {err}")
        return res.unwrap()

    def _exists_sync(self, key_str: str) -> bool:
        res = self.store.is_exist(key_str)
        return bool(self._unwrap_result(res, op="is_exist"))

    async def exists(self, key: CacheEngineKey) -> bool:
        return await asyncio.to_thread(self._exists_sync, self._key_to_str(key))

    def exists_sync(self, key: CacheEngineKey) -> bool:
        future = asyncio.run_coroutine_threadsafe(self.exists(key), self.loop)
        try:
            return bool(future.result())
        except Exception as e:
            logger.warning("FluxonConnector.exists_sync failed for key %s: %s", key, e)
            return False

    def _get_value(self, key_str: str) -> Dict[str, Any]:
        res = self.store.get(key_str)
        fut = self._unwrap_result(res, op="get")
        wait_res = fut.wait()
        holder = self._unwrap_result(wait_res, op="get.wait")
        access_res = holder.access()
        value = self._unwrap_result(access_res, op="memholder.access")
        if not isinstance(value, dict):
            raise RuntimeError(
                f"Fluxon get returned non-dict value for key {key_str}: {type(value)}"
            )

        # Keep `holder` alive by attaching it to the returned dict.
        # The underlying KV payload may be backed by shared memory managed by
        # the MemHolder; without keeping it alive, dlpack views can dangle.
        value["_lmcache_fluxon_memholder"] = holder
        return value

    def _build_memory_obj_from_value(
        self, key_str: str, value: Dict[str, Any]
    ) -> Optional[MemoryObj]:
        meta_bytes = value.get(_FIELD_META)
        data_obj = value.get(_FIELD_DATA)
        if not isinstance(meta_bytes, (bytes, bytearray, memoryview)):
            logger.warning(
                "FluxonConnector.get: missing/invalid meta field for key %s: %s",
                key_str,
                type(meta_bytes),
            )
            return None

        metadata = RemoteMetadata.deserialize(memoryview(meta_bytes))
        if metadata.dtype is None:
            logger.warning(
                "FluxonConnector.get: invalid dtype in metadata for key %s", key_str
            )
            return None

        holder = value.get("_lmcache_fluxon_memholder")

        if not hasattr(data_obj, "__dlpack__"):
            raise RuntimeError(
                f"FluxonConnector.get requires dlpack-backed payload for key {key_str}, "
                f"but got type={type(data_obj)}"
            )

        typed = torch.utils.dlpack.from_dlpack(data_obj)
        raw_uint8 = typed.view(torch.uint8).reshape(-1)

        expected_bytes = int(metadata.length)
        if raw_uint8.numel() < expected_bytes:
            logger.warning(
                "FluxonConnector.get: data length too short for key %s: got=%d need=%d",
                key_str,
                raw_uint8.numel(),
                expected_bytes,
            )
            return None

        if raw_uint8.numel() > expected_bytes:
            raw_uint8 = raw_uint8[:expected_bytes]

        meta = MemoryObjMetadata(
            shape=metadata.shape,
            dtype=metadata.dtype,
            address=int(raw_uint8.data_ptr()),
            phy_size=int(raw_uint8.numel()),
            ref_count=1,
            pin_count=0,
            fmt=metadata.fmt,
        )
        memory_obj = TensorMemoryObj(raw_uint8, meta, parent_allocator=None)
        # Keep Fluxon memholder alive for the lifetime of the returned MemoryObj,
        # otherwise dlpack views can dangle after this function returns.
        setattr(memory_obj, "_fluxon_memholder", holder)
        return memory_obj

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = self._key_to_str(key)
        value = await asyncio.to_thread(self._get_value, key_str)
        return self._build_memory_obj_from_value(key_str, value)

    def _put_payload(self, memory_obj: MemoryObj) -> torch.Tensor:
        raw = getattr(memory_obj, "raw_data", None)
        if isinstance(raw, torch.Tensor) and raw.device.type == "cpu":
            return raw.view(torch.uint8).reshape(-1)
        t = getattr(memory_obj, "tensor", None)
        if isinstance(t, torch.Tensor) and t.device.type == "cpu":
            return t.view(torch.uint8).reshape(-1)

        kv_bytes = memory_obj.byte_array
        buf = memoryview(kv_bytes) if not isinstance(kv_bytes, memoryview) else kv_bytes
        return torch.frombuffer(buf, dtype=torch.uint8)

    def _put_sync(self, key_str: str, memory_obj: MemoryObj) -> None:
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        metadata_bytes = RemoteMetadata(
            len(memory_obj.byte_array), kv_shape, kv_dtype, memory_format
        ).serialize()

        payload: Dict[str, Union[bytes, torch.Tensor]] = {
            _FIELD_META: metadata_bytes,
            _FIELD_DATA: self._put_payload(memory_obj),
        }

        res = self.store.put(key_str, payload)
        fut = self._unwrap_result(res, op="put")
        wait_res = fut.wait()
        _ = self._unwrap_result(wait_res, op="put.wait")

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        await asyncio.to_thread(self._put_sync, self._key_to_str(key), memory_obj)

    @no_type_check
    async def list(self) -> List[str]:
        # Fluxon KV API does not expose listing keys in the unified client.
        return []

    async def close(self):
        try:
            res = self.store.close()
            _ = self._unwrap_result(res, op="close")
            logger.info("Closed Fluxon KV store connection")
        except Exception as e:
            logger.warning("FluxonConnector.close failed: %s", e)
