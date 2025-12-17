"""
Startup (Example with vLLM): 
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --max-model-len 130000 --port 8000 \
    --load-format dummy \
    --max-num-seqs 1

Example Usage (Run Benchmark): 
python ttft-estimator.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --backend no-cache \
    --context-lengths 2000,5000,10000,20000 \
    --csv-file results.csv

Example Usage (Run Benchmark with different backend label): 
python ttft-estimator.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --backend lmcache-gpu \
    --context-lengths 2000,5000,10000,20000 \
    --csv-file results.csv

Example Usage (Plot results):
python ttft-estimator.py --plot-only --csv-file results.csv
"""

# Standard
import argparse
import time
import csv
import os
from collections import defaultdict

# Third Party
from openai import OpenAI
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

args = None
tokenizer = None
# the number of tokens in 10,000 "hi"s
hi_multiplier = None
context_length_ttfts = []
client = None # Will be initialized after args

def query_and_measure_ttft(prompt):
    start = time.perf_counter()
    ttft = None

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=args.model,
        temperature=0.7,
        stream=True,
        max_completion_tokens=5,
    )

    for chunk in chat_completion:
        chunk_message = chunk.choices[0].delta.content
        if chunk_message is not None:
            if ttft is None:
                ttft = time.perf_counter()
            # print(chunk_message, end="", flush=True) # Optional: comment out to reduce noise

    # print("\n")
    if ttft is None:
        return 0 # Handle error or timeout
    return ttft - start


def append_to_csv():
    """Appends the current run's results to the CSV file."""
    file_exists = os.path.isfile(args.csv_file)
    
    print(f"Saving results to {args.csv_file}...")
    with open(args.csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Write header if file is new
        if not file_exists:
            writer.writerow(['backend', 'context_length', 'ttft'])
        
        for length, ttft in context_length_ttfts:
            writer.writerow([args.backend, length, ttft])
    print("Save complete.")


def plot_from_csv():
    """Reads the CSV file and plots comparison lines for each backend."""
    if not os.path.isfile(args.csv_file):
        print(f"Error: File {args.csv_file} not found.")
        return

    data = defaultdict(list)
    
    # Read CSV
    with open(args.csv_file, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            backend = row['backend']
            ctx_len = int(row['context_length'])
            ttft = float(row['ttft'])
            data[backend].append((ctx_len, ttft))

    # Plotting
    plt.figure(figsize=(10, 6))
    
    for backend, points in data.items():
        # Sort points by context length
        points.sort(key=lambda x: x[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        plt.plot(xs, ys, marker='o', label=backend)

    plt.xlabel("Context Length")
    plt.ylabel("TTFT (s)")
    plt.title("TTFT vs Context Length by Backend")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    output_img = "ttft_comparison.png"
    plt.savefig(output_img)
    print(f"Comparison plot saved to {output_img}")


def run_benchmark():
    global hi_multiplier
    
    # Initialize tokenizer and calculate multiplier
    # We only do this in benchmark mode
    global tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    set_hi_multiplier()

    # Warm up
    print("Starting warm up...")
    warm_up_prompt = "bye" * 50
    query_and_measure_ttft(warm_up_prompt)
    print("Warm up complete")

    # Run Benchmark
    for i, context_length in enumerate(
        map(int, (s.strip() for s in args.context_lengths.split(",")))
    ):
        number_of_his = context_length * hi_multiplier // 10_000
        # break the prefix with the enumeration to avoid unintended cache hits if logic allows,
        # but for LMCache testing you might actually want hits? 
        # Kept original logic: prefix changes slightly with `i`
        prompt = f"{i}" + "hi" * number_of_his
        
        ttft = query_and_measure_ttft(prompt)
        print(f"Backend: {args.backend} | Context length: {context_length} | TTFT: {ttft:.4f}s")
        context_length_ttfts.append((context_length, ttft))
    
    # Save to CSV
    append_to_csv()


def set_hi_multiplier():
    global hi_multiplier
    prompt = "hi" * 10000
    hi_multiplier = len(tokenizer.encode(prompt))
    print(f'number tokens in 10,000 "hi\'s": {hi_multiplier}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, help="Required for benchmark run")
    parser.add_argument("--host", type=str, required=False, default="localhost")
    parser.add_argument("--port", type=str, required=False, default="8000")
    parser.add_argument("--context-lengths", type=str, required=False, default="1024")
    
    # New arguments
    parser.add_argument("--backend", type=str, default="default", help="Name of the backend for this run (e.g. 'no-cache', 'lmcache')")
    parser.add_argument("--csv-file", type=str, default="ttft_results.csv", help="Path to CSV file for storing/reading data")
    parser.add_argument("--plot-only", action="store_true", help="Skip benchmark and just plot existing data from CSV")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.plot_only:
        plot_from_csv()
    else:
        if not args.model:
            print("Error: --model argument is required for benchmarking.")
            exit(1)
            
        # Initialize client here to respect host/port args
        base_url = f"http://{args.host}:{args.port}/v1"
        client = OpenAI(api_key="dummy-key", base_url=base_url)
        
        run_benchmark()