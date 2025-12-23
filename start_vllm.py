#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
import json
import getpass
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


REPO_DIR = Path(__file__).resolve().parent


def die(msg: str, code: int = 2) -> "None":
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def warn(msg: str) -> None:
    print(f"warn: {msg}", file=sys.stderr)


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        die("PyYAML is not installed; cannot use --config")
    try:
        data = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        die(f"config not found: {path}")
    except Exception as e:
        die(f"failed to read config {path}: {e}")
    if data is None:
        return {}
    if not isinstance(data, dict):
        die(f"config root must be a mapping/dict: {path}")
    return data


def to_int(value: Any, field: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        die(f"{field} must be an int, got: {value!r}")


def to_bool(value: Any, field: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    die(f"{field} must be a bool, got: {value!r}")


def find_binary(name: str) -> str:
    from shutil import which

    path = which(name)
    if not path:
        die(f"{name} not found in PATH")
    return path


def run_help_text(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return proc.stdout or ""
    except Exception:
        return ""


def supports_flag(help_text: str, flag: str) -> bool:
    return flag in help_text


@dataclass
class RayConfig:
    mode: str = "none"  # none|head|worker
    address: str | None = None
    head_host: str | None = None
    port: int = 6379
    dashboard_host: str = "0.0.0.0"
    temp_dir: str | None = None


@dataclass
class MooncakeConfig:
    start: bool = False
    master_rpc_address: str = "0.0.0.0"
    master_rpc_port: int = 50051
    global_file_segment_size_gb: int = 32
    default_kv_lease_ttl_ms: int = 86400000
    server_local_hostname: str | None = None
    server_metadata_server: str | None = None
    server_master_server_addr: str | None = None
    server_global_segment_size_gb: int = 32
    server_local_buffer_size_gb: int = 4
    server_protocol: str = "tcp"
    server_rdma_devices: str = ""


@dataclass
class LaunchConfig:
    lmcache_config_file: str | None = None
    model: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    tp: int | None = None
    pp: int | None = None
    pyhashseed: str = "0"
    vllm_bin: str = "vllm"
    extra_vllm_args: list[str] = None  # type: ignore[assignment]
    env: dict[str, str] = None  # type: ignore[assignment]
    kv_transfer_config: dict[str, Any] | str | None = None
    ray: RayConfig = None  # type: ignore[assignment]
    mooncake: MooncakeConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.extra_vllm_args is None:
            self.extra_vllm_args = []
        if self.env is None:
            self.env = {}
        if self.kv_transfer_config is None:
            self.kv_transfer_config = {
                "kv_connector": "LMCacheConnectorV1",
                "kv_role": "kv_both",
            }
        if self.ray is None:
            self.ray = RayConfig()
        if self.mooncake is None:
            self.mooncake = MooncakeConfig()


def config_from_dict(d: dict[str, Any]) -> LaunchConfig:
    cfg = LaunchConfig()

    cfg.lmcache_config_file = d.get("lmcache_config_file", cfg.lmcache_config_file)
    cfg.model = d.get("model", cfg.model)
    cfg.host = str(d.get("host", cfg.host))
    cfg.port = to_int(d.get("port", cfg.port), "port") or cfg.port
    cfg.tp = to_int(d.get("tp", cfg.tp), "tp")
    cfg.pp = to_int(d.get("pp", cfg.pp), "pp")
    cfg.pyhashseed = str(d.get("pyhashseed", cfg.pyhashseed))
    cfg.vllm_bin = str(d.get("vllm_bin", cfg.vllm_bin))

    env = d.get("env", {}) or {}
    if not isinstance(env, dict):
        die("env must be a mapping/dict of string->string")
    cfg.env = {str(k): str(v) for k, v in env.items()}

    kv = d.get("kv_transfer_config", cfg.kv_transfer_config)
    if kv is None:
        cfg.kv_transfer_config = None
    elif isinstance(kv, dict):
        cfg.kv_transfer_config = kv
    elif isinstance(kv, str):
        cfg.kv_transfer_config = kv
    else:
        die("kv_transfer_config must be a dict, string, or null")

    extra = d.get("extra_vllm_args", [])
    if extra is None:
        extra = []
    if isinstance(extra, str):
        cfg.extra_vllm_args = shlex.split(extra)
    elif isinstance(extra, list):
        cfg.extra_vllm_args = [str(x) for x in extra]
    else:
        die("extra_vllm_args must be a list of strings (or a single string)")

    ray_d = d.get("ray", {}) or {}
    if not isinstance(ray_d, dict):
        die("ray must be a dict")
    cfg.ray = RayConfig(
        mode=str(ray_d.get("mode", cfg.ray.mode)),
        address=ray_d.get("address", cfg.ray.address),
        head_host=ray_d.get("head_host", cfg.ray.head_host),
        port=to_int(ray_d.get("port", cfg.ray.port), "ray.port") or cfg.ray.port,
        dashboard_host=str(ray_d.get("dashboard_host", cfg.ray.dashboard_host)),
        temp_dir=ray_d.get("temp_dir", cfg.ray.temp_dir),
    )

    mc_d = d.get("mooncake", {}) or {}
    if not isinstance(mc_d, dict):
        die("mooncake must be a dict")
    cfg.mooncake = MooncakeConfig(
        start=to_bool(mc_d.get("start", cfg.mooncake.start), "mooncake.start")
        or False,
        master_rpc_address=str(
            mc_d.get("master_rpc_address", cfg.mooncake.master_rpc_address)
        ),
        master_rpc_port=to_int(
            mc_d.get("master_rpc_port", cfg.mooncake.master_rpc_port),
            "mooncake.master_rpc_port",
        )
        or cfg.mooncake.master_rpc_port,
        global_file_segment_size_gb=to_int(
            mc_d.get(
                "global_file_segment_size_gb",
                cfg.mooncake.global_file_segment_size_gb,
            ),
            "mooncake.global_file_segment_size_gb",
        )
        or cfg.mooncake.global_file_segment_size_gb,
        default_kv_lease_ttl_ms=to_int(
            mc_d.get("default_kv_lease_ttl_ms", cfg.mooncake.default_kv_lease_ttl_ms),
            "mooncake.default_kv_lease_ttl_ms",
        )
        or cfg.mooncake.default_kv_lease_ttl_ms,
        server_local_hostname=mc_d.get("server_local_hostname"),
        server_metadata_server=mc_d.get("server_metadata_server"),
        server_master_server_addr=mc_d.get("server_master_server_addr"),
        server_global_segment_size_gb=to_int(
            mc_d.get(
                "server_global_segment_size_gb",
                cfg.mooncake.server_global_segment_size_gb,
            ),
            "mooncake.server_global_segment_size_gb",
        )
        or cfg.mooncake.server_global_segment_size_gb,
        server_local_buffer_size_gb=to_int(
            mc_d.get(
                "server_local_buffer_size_gb",
                cfg.mooncake.server_local_buffer_size_gb,
            ),
            "mooncake.server_local_buffer_size_gb",
        )
        or cfg.mooncake.server_local_buffer_size_gb,
        server_protocol=str(
            mc_d.get("server_protocol", cfg.mooncake.server_protocol)
        ),
        server_rdma_devices=str(
            mc_d.get("server_rdma_devices", cfg.mooncake.server_rdma_devices)
        ),
    )

    return cfg


def resolve_lmcache_config(cfg: LaunchConfig, *, base_dir: Path) -> str:
    if not cfg.lmcache_config_file:
        die("missing lmcache_config_file (set it to a valid LMCache YAML path)")

    raw = os.path.expandvars(os.path.expanduser(cfg.lmcache_config_file))
    p = Path(raw)
    if p.is_absolute():
        if not p.exists():
            die(f"LMCache config not found: {p}")
        return str(p)

    # For convenience, accept relative paths that are either:
    # 1) relative to the launcher config directory (base_dir), or
    # 2) relative to the repo root (REPO_DIR).
    candidates = [
        (base_dir / p).resolve(),
        (REPO_DIR / p).resolve(),
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    die(
        "LMCache config not found (tried: "
        + ", ".join(str(c) for c in candidates)
        + f") for lmcache_config_file={raw!r}"
    )


class ProcessGroup:
    def __init__(self) -> None:
        self.procs: list[subprocess.Popen[str]] = []

    def start(self, args: list[str], *, env: dict[str, str] | None = None) -> None:
        proc = subprocess.Popen(args, env=env)
        self.procs.append(proc)

    def terminate_all(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        for p in self.procs:
            try:
                p.wait(timeout=5)
            except Exception:
                pass
        for p in self.procs:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass


def start_mooncake_if_needed(cfg: LaunchConfig, group: ProcessGroup) -> None:
    if not cfg.mooncake.start:
        return

    master_bin = "mooncake_master"
    find_binary(master_bin)
    group.start(
        [
            master_bin,
            "--rpc_address",
            cfg.mooncake.master_rpc_address,
            "--rpc_port",
            str(cfg.mooncake.master_rpc_port),
            "--global_file_segment_size",
            str(cfg.mooncake.global_file_segment_size_gb * 1024 * 1024 * 1024),
            "--default_kv_lease_ttl",
            str(cfg.mooncake.default_kv_lease_ttl_ms),
        ]
    )

    server_py = str((REPO_DIR / "mooncake_server.py").resolve())
    find_binary("python3")
    server_args = ["python3", server_py]
    if cfg.mooncake.server_local_hostname:
        server_args += ["--local-hostname", cfg.mooncake.server_local_hostname]
    if cfg.mooncake.server_metadata_server:
        server_args += ["--metadata-server", cfg.mooncake.server_metadata_server]
    if cfg.mooncake.server_master_server_addr:
        server_args += ["--master-server-addr", cfg.mooncake.server_master_server_addr]
    server_args += [
        "--global-segment-size-gb",
        str(cfg.mooncake.server_global_segment_size_gb),
        "--local-buffer-size-gb",
        str(cfg.mooncake.server_local_buffer_size_gb),
        "--protocol",
        cfg.mooncake.server_protocol,
        "--rdma-devices",
        cfg.mooncake.server_rdma_devices,
    ]
    group.start(server_args)


def start_ray_if_needed(cfg: LaunchConfig) -> None:
    mode = cfg.ray.mode
    if mode == "none" and not cfg.ray.address:
        return

    find_binary("ray")

    ray_help = run_help_text(["ray", "start", "--help"])
    disable_usage_stats_flag = (
        ["--disable-usage-stats"] if supports_flag(ray_help, "--disable-usage-stats") else []
    )

    ray_temp_dir: str
    if cfg.ray.temp_dir:
        ray_temp_dir = os.path.expandvars(os.path.expanduser(str(cfg.ray.temp_dir)))
    else:
        user = os.getenv("USER") or getpass.getuser() or str(os.getuid())
        ray_temp_dir = f"/tmp/ray-{user}-lmcache-{cfg.ray.port}"
    Path(ray_temp_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RAY_TMPDIR", ray_temp_dir)

    temp_dir_flag = ["--temp-dir", ray_temp_dir] if supports_flag(ray_help, "--temp-dir") else []

    if mode == "head":
        args = [
            "ray",
            "start",
            "--head",
            "--port",
            str(cfg.ray.port),
            "--dashboard-host",
            cfg.ray.dashboard_host,
            *disable_usage_stats_flag,
            *temp_dir_flag,
        ]
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            out = proc.stdout or ""
            m = re.search(r"already running at ([0-9.]+:[0-9]+)", out)
            if m:
                existing = m.group(1)
                warn(
                    f"Ray head already running at {existing}; reusing existing cluster "
                    f"(stop it or change `ray.port` if you want a fresh cluster)"
                )
                cfg.ray.address = cfg.ray.address or existing
                return

            # Another common case: a head is already running on this port, but
            # Ray asserts due to session mismatch when attempting to start a new
            # head. In this case, proceed by reusing the existing cluster.
            if "does not match persisted value" in out:
                existing = cfg.ray.address or f"127.0.0.1:{cfg.ray.port}"
                warn(
                    f"Ray head seems to already be running on {existing}; reusing existing cluster "
                    f"(stop it or change `ray.port` if you want a fresh cluster)"
                )
                cfg.ray.address = existing
                return

            die(f"failed to start Ray head:\n{out.strip()}")
        else:
            # Use explicit address if provided; otherwise default to local loopback.
            cfg.ray.address = cfg.ray.address or f"127.0.0.1:{cfg.ray.port}"
    elif mode == "worker":
        if not cfg.ray.address:
            if not cfg.ray.head_host:
                die("ray.mode=worker requires ray.address=<ip:port> or ray.head_host=<ip/host>")
            cfg.ray.address = f"{cfg.ray.head_host}:{cfg.ray.port}"
        args = ["ray", "start", "--address", cfg.ray.address, *disable_usage_stats_flag, *temp_dir_flag]
        subprocess.run(args, check=True)
        print(f"Ray worker started and connected to {cfg.ray.address}")
        raise SystemExit(0)
    elif mode == "none":
        # Use existing cluster at ray.address.
        if not cfg.ray.address:
            die("ray.address is required when ray.mode=none but Ray is enabled")
    else:
        die("ray.mode must be one of: none|head|worker")


def build_vllm_cmd(cfg: LaunchConfig) -> list[str]:
    if not cfg.model:
        die("missing model (set `model` in config or pass --model)")

    vllm_bin = cfg.vllm_bin
    find_binary(vllm_bin)
    help_text = run_help_text([vllm_bin, "serve", "--help"])

    model_arg = os.path.expandvars(os.path.expanduser(str(cfg.model)))
    model_path = Path(model_arg)
    # Fail fast for local paths (helps avoid long HF stack traces when a path
    # doesn't exist or is not readable on this node).
    looks_like_path = model_arg.startswith(("/", "./", "../", "~")) or model_path.exists()
    if looks_like_path:
        if not model_path.exists():
            die(
                f"model path not found: {model_path}\n"
                "If you meant a HuggingFace repo id, use 'namespace/repo_name' instead of a filesystem path.\n"
                "For Ray multi-node, the model path must exist on every node (head + workers) at the same path."
            )
        if model_path.is_dir():
            if not (model_path / "config.json").exists():
                die(
                    f"model directory missing config.json: {model_path}\n"
                    "Expected a HuggingFace-style local model directory.\n"
                    "For Ray multi-node, ensure the directory is mounted/copied to every node."
                )

    cmd: list[str] = [
        vllm_bin,
        "serve",
        model_arg,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
    ]

    if cfg.kv_transfer_config is not None:
        if isinstance(cfg.kv_transfer_config, dict):
            kv_str = json.dumps(cfg.kv_transfer_config, separators=(",", ":"))
        else:
            kv_str = cfg.kv_transfer_config
        cmd += ["--kv-transfer-config", kv_str]

    if cfg.tp is not None:
        cmd += ["--tensor-parallel-size", str(cfg.tp)]
    if cfg.pp is not None:
        cmd += ["--pipeline-parallel-size", str(cfg.pp)]

    if supports_flag(help_text, "--disable-hybrid-kv-cache-manager"):
        cmd += ["--disable-hybrid-kv-cache-manager"]

    if cfg.ray.address:
        if supports_flag(help_text, "--distributed-executor-backend"):
            cmd += ["--distributed-executor-backend", "ray"]
        if supports_flag(help_text, "--ray-address"):
            cmd += ["--ray-address", cfg.ray.address]
        else:
            os.environ["RAY_ADDRESS"] = cfg.ray.address

    cmd += cfg.extra_vllm_args
    return cmd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified vLLM launcher for LMCache (configured via YAML)."
    )
    p.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="YAML config file",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).expanduser()
    cfg = config_from_dict(load_yaml(config_path))

    config_dir = config_path.resolve().parent
    lmcache_config = resolve_lmcache_config(cfg, base_dir=config_dir)
    os.environ["LMCACHE_CONFIG_FILE"] = lmcache_config
    os.environ.setdefault("PYTHONHASHSEED", cfg.pyhashseed)

    for k, v in cfg.env.items():
        os.environ[str(k)] = os.path.expandvars(os.path.expanduser(str(v)))

    if cfg.tp and cfg.tp > 1:
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "1")

    # Ray start (optional).
    start_ray_if_needed(cfg)

    # Mooncake start (optional).
    group = ProcessGroup()
    start_mooncake_if_needed(cfg, group)

    cmd = build_vllm_cmd(cfg)
    print(f"LMCACHE_CONFIG_FILE={os.getenv('LMCACHE_CONFIG_FILE', '<unset>')}")
    print("Running:", " ".join(shlex.quote(c) for c in cmd))

    vllm_proc = subprocess.Popen(cmd)

    def _handle_signal(_sig: int, _frame: Any) -> None:
        if vllm_proc.poll() is None:
            try:
                vllm_proc.terminate()
            except Exception:
                pass

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    rc = vllm_proc.wait()
    group.terminate_all()
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
