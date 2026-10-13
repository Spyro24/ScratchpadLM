"""Benchmark script for ScratchpadLM.

Measures training speed, generation speed, and router performance.
Stdlib only.
"""

import time
import random
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinylm import (
    TinyLM, Router, RouterConfig, SpecializedLM, SpecializedConfig,
    train, train_parallel, generate, generate_router_training_data,
    train_router, save_model, load_model, count_params
)
from tokenizer import BPETokenizer


def create_test_corpus(size_kb=10):
    """Create a test corpus for benchmarking."""
    # Create diverse text with code, math, and prose patterns
    patterns = [
        "the quick brown fox jumps over the lazy dog. " * 50,
        "def function(args): return args + 1\n" * 100,
        "class MyClass: def method(self): return self.value\n" * 50,
        "import numpy as np\nimport torch\n" * 30,
        "for i in range(100): print(i)\n" * 30,
        "if x > 0: return x * 2\n" * 50,
        "math calculate solve prove logic theorem\n" * 50,
        "1 + 1 = 2\n" * 100,
        "write summarize explain story article text\n" * 50,
    ]
    
    corpus = " ".join(patterns)
    # Scale to target size
    target_chars = size_kb * 1024
    if len(corpus) < target_chars:
        corpus = corpus * (target_chars // len(corpus) + 1)
    return corpus[:target_chars]


def bench_tokenizer(corpus, vocab_size=2000):
    """Benchmark tokenizer training and encoding/decoding."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: Tokenizer")
    print(f"{'='*60}")
    
    # Train
    tok = BPETokenizer()
    start = time.time()
    tok.train(corpus, vocab_size=vocab_size)
    train_time = time.time() - start
    print(f"  Train: {train_time:.3f}s ({len(corpus)/train_time/1024:.1f} KB/s)")
    
    # Encode
    start = time.time()
    ids = tok.encode(corpus)
    encode_time = time.time() - start
    print(f"  Encode: {encode_time:.3f}s ({len(corpus)/encode_time/1024:.1f} KB/s)")
    print(f"  Tokens: {len(ids):,}")
    
    # Decode
    start = time.time()
    decoded = tok.decode(ids)
    decode_time = time.time() - start
    print(f"  Decode: {decode_time:.3f}s ({len(decoded)/decode_time/1024:.1f} KB/s)")
    
    return {
        "train_time": train_time,
        "encode_time": encode_time,
        "decode_time": decode_time,
        "vocab_size": len(tok.vocab),
        "num_tokens": len(ids)
    }


def bench_tinylm_train(model, ids, steps=100, ctx=50):
    """Benchmark TinyLM training."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: TinyLM Training")
    print(f"{'='*60}")
    
    start = time.time()
    train(model, ids, steps=steps, lr=0.05, batch_size=16, n_negatives=256)
    elapsed = time.time() - start
    
    tokens_processed = steps * 16 * model.ctx  # approx
    tok_per_sec = tokens_processed / elapsed
    
    print(f"  Steps: {steps}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Tokens/sec: {tok_per_sec:,.0f}")
    print(f"  Steps/sec: {steps/elapsed:.1f}")
    
    return {
        "elapsed": elapsed,
        "steps": steps,
        "tokens_per_sec": tok_per_sec,
        "steps_per_sec": steps/elapsed
    }


def bench_tinylm_parallel(model, ids, steps=100, workers=4):
    """Benchmark parallel TinyLM training."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: TinyLM Parallel Training (workers={workers})")
    print(f"{'='*60}")
    
    start = time.time()
    train_parallel(model, ids, steps=steps, lr=0.05, batch_size=16, 
                   n_negatives=256, workers=workers)
    elapsed = time.time() - start
    
    tokens_processed = steps * 16 * model.ctx
    tok_per_sec = tokens_processed / elapsed
    
    print(f"  Steps: {steps}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Tokens/sec: {tok_per_sec:,.0f}")
    print(f"  Steps/sec: {steps/elapsed:.1f}")
    
    return {
        "elapsed": elapsed,
        "steps": steps,
        "tokens_per_sec": tok_per_sec,
        "steps_per_sec": steps/elapsed
    }


def bench_generation(model, tok, prompts, n_tokens=50):
    """Benchmark text generation."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: Generation")
    print(f"{'='*60}")
    
    total_tokens = 0
    total_time = 0
    
    for prompt in prompts:
        start = time.time()
        text = generate(model, tok, prompt, n_tokens=n_tokens, temperature=0.8)
        elapsed = time.time() - start
        
        actual_tokens = len(tok.encode(text)) - len(tok.encode(prompt))
        total_tokens += actual_tokens
        total_time += elapsed
        
        print(f"  '{prompt[:40]}...' -> {actual_tokens} tokens in {elapsed:.3f}s ({actual_tokens/elapsed:.1f} tok/s)")
    
    avg_tok_per_sec = total_tokens / total_time if total_time > 0 else 0
    print(f"\n  Average: {avg_tok_per_sec:.1f} tok/s")
    
    return {
        "total_tokens": total_tokens,
        "total_time": total_time,
        "avg_tokens_per_sec": avg_tok_per_sec
    }


def bench_router_train(corpus, tokenizer, vocab_size, ctx=128, steps=50):
    """Benchmark router training."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: Router Training")
    print(f"{'='*60}")
    
    # Generate training data
    start = time.time()
    examples = generate_router_training_data(corpus, tokenizer, vocab_size, ctx)
    data_time = time.time() - start
    print(f"  Data generation: {data_time:.3f}s ({len(examples)} examples)")
    
    # Create router
    config = RouterConfig(
        vocab_size=vocab_size,
        ctx=ctx,
        emb=64,
        hidden=128,
        n_layers=2,
        d_ff=512
    )
    router = Router(config, seed=42)
    print(f"  Router params: {count_params(router):,}")
    
    # Train
    start = time.time()
    train_router(router, tokenizer, examples, steps=steps, lr=0.05, 
                 batch_size=16, n_negatives=256)
    elapsed = time.time() - start
    
    print(f"  Training: {elapsed:.3f}s ({steps} steps)")
    print(f"  Steps/sec: {steps/elapsed:.1f}")
    
    return {
        "data_time": data_time,
        "train_time": elapsed,
        "steps": steps,
        "examples": len(examples),
        "steps_per_sec": steps/elapsed
    }


def bench_router_inference(router, tokenizer, prompts):
    """Benchmark router inference (split/route)."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: Router Inference")
    print(f"{'='*60}")
    
    total_time = 0
    for prompt in prompts:
        start = time.time()
        routes = router.split(prompt, tokenizer)
        elapsed = time.time() - start
        total_time += elapsed
        print(f"  '{prompt[:40]}...' -> {len(routes)} routes in {elapsed*1000:.1f}ms")
    
    avg_ms = (total_time / len(prompts)) * 1000
    print(f"\n  Average: {avg_ms:.1f}ms per prompt")
    
    return {
        "total_time": total_time,
        "avg_ms": avg_ms
    }


def bench_save_load(model, path="/tmp/bench_model.tlm"):
    """Benchmark model save/load."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: Save/Load")
    print(f"{'='*60}")
    
    # Save
    start = time.time()
    save_model(model, path)
    save_time = time.time() - start
    
    # Get file size
    size_kb = os.path.getsize(path) / 1024
    print(f"  Save: {save_time:.3f}s ({size_kb:.1f} KB)")
    
    # Load
    start = time.time()
    loaded = load_model(path)
    load_time = time.time() - start
    print(f"  Load: {load_time:.3f}s")
    
    return {
        "save_time": save_time,
        "load_time": load_time,
        "size_kb": size_kb
    }


def bench_count_params(model):
    """Benchmark parameter counting."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: Parameter Counting")
    print(f"{'='*60}")
    
    start = time.time()
    n = count_params(model)
    elapsed = time.time() - start
    
    print(f"  Params: {n:,}")
    print(f"  Time: {elapsed*1000:.2f}ms")
    
    return {"params": n, "time_ms": elapsed*1000}


def run_all_benchmarks():
    """Run all benchmarks and print summary."""
    print("="*60)
    print("ScratchpadLM Benchmark Suite")
    print("="*60)
    
    # Create test corpus
    print("Creating test corpus...")
    corpus = create_test_corpus(10)  # 10KB
    print(f"Corpus size: {len(corpus):,} chars")
    
    # Create tokenizer (fresh, don't load existing)
    import os
    if os.path.exists("bpe_merges.txt"):
        os.remove("bpe_merges.txt")
    
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=2000)
    
    # Test prompts
    prompts = [
        "The quick brown fox",
        "def hello_world():",
        "Calculate 2 + 2 =",
        "Once upon a time",
    ]
    
    results = {}
    
    # Benchmark tokenizer
    results["tokenizer"] = bench_tokenizer(corpus)
    
    # Create TinyLM (smaller for faster benchmarks)
    model = TinyLM(vocab_size=len(tokenizer.vocab), ctx=32, emb=32, hidden=64)
    ids = tokenizer.encode(corpus)
    
    # Benchmark TinyLM training (very fast)
    results["tinylm_train"] = bench_tinylm_train(model, ids, steps=5)
    
    # Benchmark generation
    results["generation"] = bench_generation(model, tokenizer, prompts, n_tokens=10)
    
    # Benchmark save/load
    results["save_load"] = bench_save_load(model)
    
    # Benchmark param counting
    results["count_params"] = bench_count_params(model)
    
    # Benchmark router (skip training, just test inference with a pre-trained or tiny router)
    print("\nCreating router for inference test...")
    router_config = RouterConfig(vocab_size=2000, ctx=32, emb=32, hidden=64, n_layers=2, d_ff=128)
    router = Router(router_config, seed=42)
    # Quick train just for inference test (1 step)
    examples = generate_router_training_data(corpus, tokenizer, 2000, 32)
    train_router(router, tokenizer, examples, steps=1, lr=0.05, batch_size=4, n_negatives=32)
    results["router_inference"] = bench_router_inference(router, tokenizer, prompts)
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    for name, res in results.items():
        print(f"\n{name.upper()}:")
        for k, v in res.items():
            print(f"  {k}: {v}")
    
    return results


if __name__ == "__main__":
    import time
    import math
    import os
    
    # Run benchmarks
    results = run_all_benchmarks()
    
    # Save results
    import json
    with open("benchmark_results.json", "w") as f:
        # Convert any non-serializable objects
        serializable = {}
        for k, v in results.items():
            serializable[k] = {kk: (float(vv) if isinstance(vv, (int, float)) else str(vv)) 
                              for kk, vv in v.items()}
        json.dump(serializable, f, indent=2)
    print("\nResults saved to benchmark_results.json")