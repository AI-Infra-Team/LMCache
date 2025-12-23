## How to run LMCache-Fluxon

1. Ensure that Fluxon is cloned and installed in the machine.
2. Run `pip install -e .` to install LMCache-Fluxon
3. Start Fluxon/Mooncake storage service (if needed).
   - Mooncake: `./start_mooncake.sh` (master) and `python3 mooncake_server.py` (server).
4. Ensure that vLLM is installed.
5. Create a launcher config (copy from `start_vllm.example.yaml`) and edit at least `model` (and `tp/pp` if needed).
   - If you want Ray multi-node: start from `configs/ray/head_redis.yaml` + `configs/ray/worker_redis.yaml` (or `*_valkey.yaml`, `*_mooncake.yaml`).
   - `lmcache_config_file` is required and should point to an LMCache YAML (e.g. `configs/lmcache/redis.yaml`) that sets `remote_url`.
6. Start vLLM with LMCache: `./start_vllm.py -c <YOUR_CONFIG.yaml>`
7. Wait for vLLM starting.
8. Run `ttft-estimator.py` for test.

Notes:
- Multi-GPU (single node): set `CUDA_VISIBLE_DEVICES=0,1,...` and set `tp: <GPU_COUNT>` in the launcher config.
- Multi-node (Ray): set `ray.mode: head` in the head config; in worker configs set `ray.mode: worker` and either `ray.address: <head_ip:port>` or `ray.head_host: <head_ip>` + `ray.port: <port>`.
- If the machine is shared by multiple users, consider setting `ray.temp_dir` in the launcher config to avoid `/tmp/ray` permission conflicts (default is `/tmp/ray-$USER-lmcache-$PORT`).

Quick Ray test (2 nodes):
1. Head node:
   - In config: `ray.mode: head`
   - Run: `./start_vllm.py -c <HEAD_CONFIG.yaml>`
2. Worker node(s):
   - In config: `ray.mode: worker` and `ray.address: <HEAD_IP>:6379`
   - Run: `./start_vllm.py -c <WORKER_CONFIG.yaml>`
