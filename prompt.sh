curl http://127.0.0.1:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "/home/hzx/vllm/models/Qwen3-1.7B",
  "prompt": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts",
  "max_tokens": 100,
  "temperature": 0
}'
