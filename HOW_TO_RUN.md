## How to run LMCache-Fluxon

1. Ensure that Fluxon is cloned and installed in the machine.
2. Run `pip install -e .` to install LMCache-Fluxon
3. Start Fluxon/Mooncake backend, you can use `start_mooncake.sh` to run mooncake master and `mooncake_server.py` to run mooncake server (client with memory).
4. Ensure that vLLM is installed.
5. Run `start_vllm_with_fluxon.sh` or `start_vllm_with_mooncake.sh`, depends on which backend you would like to use.
6. Wait for vLLM starting.
7. Run `ttft-estimator.py` for test.