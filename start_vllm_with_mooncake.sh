#!/bin/bash
# vllm is needed
export LMCACHE_CONFIG_FILE=/home/hzx/LMCache/lmcache_mooncake.yaml
vllm serve /home/hzx/vllm/models/Qwen3-1.7B  \
    --port 8000 \
    --disable-hybrid-kv-cache-manager \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'