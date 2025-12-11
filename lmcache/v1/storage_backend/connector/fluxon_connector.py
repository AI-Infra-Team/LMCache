# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional, no_type_check
import asyncio

# Third Party
from kvcache_api_layer.config import FluxonKvClientConfig
from kvcache_api_layer.kvclient import new_store

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

METADATA_BYTES_LEN = 28


class FluxonConnector(RemoteConnector):
    """LMCache remote connector backed by Fluxon KV."""

    def __init__(
        self,
        config_path: Optional[str],
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
    ):
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend

        # Load Fluxon config from yaml file.
        # If config_path is None or empty, fallback to FluxonKvClientConfig.from_file().
        if config_path is not None and config_path != "":
            cfg = FluxonKvClientConfig.from_file(config_path)
            logger.info("FluxonConnector using config file: %s", config_path)
        else:
            cfg = FluxonKvClientConfig.from_file()
            logger.info("FluxonConnector using default config file path")

        store_result = new_store(cfg)
        err = store_result.error()
        if err is not None:
            raise RuntimeError(f"Failed to initialize Fluxon KV store: {err}")

        store = store_result.success()
        if store is None:
            raise RuntimeError("new_store returned success=None for Fluxon KV store")

        self.store = store
        logger.info("FluxonConnector initialized Fluxon KV store successfully")

    def _key_to_str(self, key: CacheEngineKey) -> str:
        return key.to_string()

    def _exists_sync(self, key_str: str) -> bool:
        """Blocking existence check using FluxonKVCacheStore."""
        logger.info("Fluxon is_exist for key %s start", key_str)
        result = self.store.is_exist(key_str)
        err = getattr(result, "error", None)() if hasattr(result, "error") else None
        if err is not None:
            logger.warning("Fluxon is_exist error for key %s: %s", key_str, err)
            return False

        future = (
            result.success() if hasattr(result, "success") else None  # type: ignore[call-arg]
        )
        if future is None:
            logger.warning("Fluxon is_exist success() returned None for key %s", key_str)
            return False

        wait_result = future.wait()
        wait_err = (
            wait_result.error()
            if hasattr(wait_result, "error")
            else None  # type: ignore[call-arg]
        )
        if wait_err is not None:
            logger.warning(
                "Fluxon is_exist future error for key %s: %s", key_str, wait_err
            )
            return False

        exists_val = (
            wait_result.success()
            if hasattr(wait_result, "success")
            else None  # type: ignore[call-arg]
        )
        logger.info("Fluxon is_exist for key %s done, is_exist: %s", key_str, exists_val)
        return bool(exists_val)

    async def exists(self, key: CacheEngineKey) -> bool:
        key_str = self._key_to_str(key)
        return await asyncio.to_thread(self._exists_sync, key_str)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        future = asyncio.run_coroutine_threadsafe(self.exists(key), self.loop)
        try:
            return bool(future.result())
        except Exception as e:
            logger.warning("FluxonConnector.exists_sync failed for key %s: %s", key, e)
            return False

    def _get_value_bytes(self, key_str: str) -> Optional[bytes]:
        """Blocking get that returns raw value bytes (metadata + payload)."""
        logger.info("Fluxon get for key %s start", key_str)
        result = self.store.get(key_str)
        err = getattr(result, "error", None)() if hasattr(result, "error") else None
        if err is not None:
            logger.warning("Fluxon get error for key %s: %s", key_str, err)
            return None

        future = (
            result.success() if hasattr(result, "success") else None  # type: ignore[call-arg]
        )
        if future is None:
            logger.warning("Fluxon get success() returned None for key %s", key_str)
            return None

        wait_result = future.wait()
        wait_err = (
            wait_result.error()
            if hasattr(wait_result, "error")
            else None  # type: ignore[call-arg]
        )
        if wait_err is not None:
            logger.warning(
                "Fluxon get future error for key %s: %s", key_str, wait_err
            )
            return None

        holder = (
            wait_result.success()
            if hasattr(wait_result, "success")
            else None  # type: ignore[call-arg]
        )
        if holder is None:
            logger.warning("Fluxon get future success() is None for key %s", key_str)
            return None

        bytes_result = holder.bytes()
        bytes_err = (
            bytes_result.error()
            if hasattr(bytes_result, "error")
            else None  # type: ignore[call-arg]
        )
        if bytes_err is not None:
            logger.warning(
                "Fluxon MemHolder.bytes() error for key %s: %s", key_str, bytes_err
            )
            return None

        value = (
            bytes_result.success()
            if hasattr(bytes_result, "success")
            else None  # type: ignore[call-arg]
        )
        if value is None:
            logger.warning(
                "Fluxon MemHolder.bytes().success() returned None for key %s", key_str
            )
            return None

        return value

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = self._key_to_str(key)
        value = await asyncio.to_thread(self._get_value_bytes, key_str)
        if value is None:
            return None

        if len(value) < METADATA_BYTES_LEN:
            logger.warning(
                "FluxonConnector.get: value for key %s is shorter than metadata size",
                key_str,
            )
            return None

        view = memoryview(value)
        metadata_bytes = view[:METADATA_BYTES_LEN]
        metadata = RemoteMetadata.deserialize(metadata_bytes)

        logger.info(f"Fluxon get for key {key_str} deserialized metadata: {metadata}")

        memory_obj = self.local_cpu_backend.allocate(
            metadata.shape,
            metadata.dtype,
            metadata.fmt,
        )
        if memory_obj is None:
            logger.warning(
                "FluxonConnector.get: failed to allocate memory for key %s", key_str
            )
            return None

        buffer = memory_obj.byte_array
        data_view = view[METADATA_BYTES_LEN : METADATA_BYTES_LEN + metadata.length]

        if isinstance(buffer, (bytearray, bytes)):
            buffer[: metadata.length] = data_view.tobytes()
        elif isinstance(buffer, memoryview):
            # See LMCache's PR#622 https://github.com/LMCache/LMCache/pull/662
            if buffer.format == '<B':
                mv = buffer.cast('B') 
            else:
                mv = buffer
            mv[: metadata.length] = data_view
        else:
            mv = memoryview(buffer)
            if mv.format == '<B':
                mv = mv.cast('B')
            mv[: metadata.length] = data_view
        
        logger.info(f"Fluxon get for key %s done", key_str)

        return memory_obj

    def _put_sync(self, key_str: str, memory_obj: MemoryObj) -> None:
        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        metadata_bytes = RemoteMetadata(
            len(kv_bytes), kv_shape, kv_dtype, memory_format
        ).serialize()

        payload = metadata_bytes + kv_bytes

        logger.info("Fluxon put for key %s start", key_str)
        result = self.store.put(key_str, payload)
        err = getattr(result, "error", None)() if hasattr(result, "error") else None
        if err is not None:
            raise RuntimeError(f"Fluxon put error for key {key_str}: {err}")

        future = (
            result.success() if hasattr(result, "success") else None  # type: ignore[call-arg]
        )
        if future is None:
            raise RuntimeError(
                f"Fluxon put success() returned None for key {key_str}"
            )

        wait_result = future.wait()
        wait_err = (
            wait_result.error()
            if hasattr(wait_result, "error")
            else None  # type: ignore[call-arg]
        )
        if wait_err is not None:
            raise RuntimeError(
                f"Fluxon put future error for key {key_str}: {wait_err}"
            )
        
        logger.info(f"Fluxon put for key %s done", key_str)

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        key_str = self._key_to_str(key)
        await asyncio.to_thread(self._put_sync, key_str, memory_obj)

    # For now we do not expose batched optimizations; RemoteBackend will
    # fall back to per-key put/get when support_batched_* is False.

    @no_type_check
    async def list(self) -> List[str]:
        # Fluxon KV API does not expose listing keys in the unified client;
        # return an empty list to satisfy interface.
        return []

    async def close(self):
        result = self.store.close()
        err = getattr(result, "error", None)() if hasattr(result, "error") else None
        if err is not None:
            logger.warning("FluxonConnector.close error: %s", err)
        else:
            logger.info("Closed Fluxon KV store connection")
