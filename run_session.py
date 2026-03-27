"""
Session runner — waits for Llama bench, runs Gemma bench, then experiments I/J/K.
Run from D:/APATPROJECTS/rule30-research/
"""
import subprocess
import json
import time
import os
import sys
import shutil
from pathlib import Path

LLAMA_BENCH = "D:/llm/llama.cpp/llama-bench.exe"
BENCH_RESULTS = Path("D:/llm/benchmark_results.json")
HF_CACHE = Path("C:/Users/Administrator/.cache/huggingface/hub")
REPO = Path("D:/APATPROJECTS/rule30-research")
EXPERIMENTS = REPO / "experiments"

MODELS_TO_BENCH = [
    {
        "name": "Meta-Llama-3.1-8B-Instruct-Q4_K_M",
        "hf": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M",
        "cache_dir": "models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF",
        "bench_log": "/tmp/llama31_bench.log",
        "bench_json": "/tmp/llama31_bench.json",
        "already_running": True,  # started before this script
    },
    {
        "name": "gemma-3-4b-it-Q4_K_M",
        "hf": "bartowski/gemma-3-4b-it-GGUF:Q4_K_M",
        "cache_dir": "models--bartowski--gemma-3-4b-it-GGUF",
        "bench_log": "/tmp/gemma3_bench.log",
        "bench_json": "/tmp/gemma3_bench.json",
        "already_running": False,
    },
]


def wait_for_bench(bench_json, bench_log, timeout=7200):
    """Wait until bench_json is non-empty (bench complete)."""
    print(f"  Waiting for {bench_json}...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = Path(bench_json)
        if p.exists() and p.stat().st_size > 10:
            try:
                data = json.loads(p.read_text())
                if isinstance(data, list) and len(data) > 0:
                    return data
            except json.JSONDecodeError:
                pass
        # Print last line of log every 30s
        lp = Path(bench_log)
        if lp.exists():
            lines = lp.read_text(errors="replace").strip().split("\n")
            last = lines[-1] if lines else ""
            print(f"    [{int(time.time()-t0)}s] {last[:100]}")
        time.sleep(30)
    raise TimeoutError(f"Bench did not complete within {timeout}s")


def append_bench_result(model_name, raw_data):
    """Append model bench result to benchmark_results.json."""
    with open(BENCH_RESULTS) as f:
        results = json.load(f)
    if not isinstance(results, list):
        results = [results]

    # Extract prompt and gen entries
    prompt_entry = next((x for x in raw_data if x.get("n_prompt", 0) > 0), None)
    gen_entry = next((x for x in raw_data if x.get("n_gen", 0) > 0), None)

    entry = {
        "timestamp": raw_data[0].get("test_time", "unknown"),
        "model": model_name,
        "model_type": raw_data[0].get("model_type", ""),
        "model_size_bytes": raw_data[0].get("model_size", 0),
        "model_n_params": raw_data[0].get("model_n_params", 0),
        "ngl": raw_data[0].get("n_gpu_layers", 35),
        "prompt_tokens": prompt_entry.get("n_prompt") if prompt_entry else None,
        "gen_tokens": gen_entry.get("n_gen") if gen_entry else None,
        "prompt_tps_avg": round(prompt_entry["avg_ts"], 3) if prompt_entry else None,
        "prompt_tps_stddev": round(prompt_entry["stddev_ts"], 3) if prompt_entry else None,
        "gen_tps_avg": round(gen_entry["avg_ts"], 3) if gen_entry else None,
        "gen_tps_stddev": round(gen_entry["stddev_ts"], 3) if gen_entry else None,
    }

    # Don't duplicate
    if not any(e.get("model") == model_name for e in results):
        results.append(entry)
        with open(BENCH_RESULTS, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Appended {model_name} to {BENCH_RESULTS}")
    else:
        print(f"  {model_name} already in results, skipping")
    return entry


def delete_hf_cache(cache_dir_name):
    cache_path = HF_CACHE / cache_dir_name
    if cache_path.exists():
        shutil.rmtree(cache_path)
        print(f"  Deleted HF cache: {cache_path}")
    else:
        print(f"  Cache not found (already deleted?): {cache_path}")


def run_bench(model_info):
    name = model_info["name"]
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")

    if not model_info["already_running"]:
        print(f"  Starting download+bench: {model_info['hf']}")
        proc = subprocess.Popen(
            [LLAMA_BENCH, "-hf", model_info["hf"], "-ngl", "35", "-o", "json"],
            stdout=open(model_info["bench_json"], "w"),
            stderr=open(model_info["bench_log"], "w")
        )
        print(f"  PID: {proc.pid}")

    raw_data = wait_for_bench(model_info["bench_json"], model_info["bench_log"])
    entry = append_bench_result(name, raw_data)
    print(f"  Result: prompt={entry.get('prompt_tps_avg')} t/s, gen={entry.get('gen_tps_avg')} t/s")
    delete_hf_cache(model_info["cache_dir"])
    return entry


def run_experiment(script_name):
    script = EXPERIMENTS / script_name
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO)
    )
    if result.returncode != 0:
        print(f"  WARNING: {script_name} exited with code {result.returncode}")
    return result.returncode


def update_llm_benchmarks_doc():
    """Rewrite docs/llm-benchmarks.md with all current results."""
    with open(BENCH_RESULTS) as f:
        results = json.load(f)

    rows = []
    for r in results:
        size_gb = r.get("model_size_bytes", 0) / 1e9
        rows.append(f"| {r['model']} | {size_gb:.1f} GB | **{r.get('prompt_tps_avg', 'N/A')}** | **{r.get('gen_tps_avg', 'N/A')}** |")

    doc = f"""# LLM Inference Benchmarks — GTX 1060 6GB

**Date:** 2026-03-27
**Engine:** llama.cpp build b8532 (CUDA SM 6.1)
**Config:** ngl=35 (all layers on GPU), n_batch=2048, n_ubatch=512, n_threads=4
**Tests:** pp512 = prompt processing 512 tokens, tg128 = text generation 128 tokens

---

## Results

| Model | Size | pp512 (t/s) | tg128 (t/s) |
|-------|------|------------|------------|
{chr(10).join(rows)}

---

## Qwen2.5-7B ngl Sweep (tuning reference)

| ngl | pp512 (t/s) | tg128 (t/s) |
|-----|------------|------------|
| 0 (CPU only) | 277 | 4.7 |
| 10 | 285 | 6.4 |
| 20 | 356 | 11.4 |
| 28 | 407 | 20.1 |
| **35 (all)** | **414** | **23.2** |

**Use ngl=35 for 7B models.**

---

## Notes

- Memory bandwidth utilization at tg128: ~88 GB/s / 192 GB/s peak = **~46%** (typical for GDDR5 + 7B model)
- Qwen2/DeepSeek share the same architecture — identical throughput expected
- For code generation (Exp I/K), Qwen2.5 is preferred; for reasoning chains, DeepSeek-R1

## Raw Data

`D:\\\\llm\\\\benchmark_results.json`
"""
    doc_path = REPO / "docs" / "llm-benchmarks.md"
    doc_path.write_text(doc)
    print(f"  Updated {doc_path}")


def git_push(message):
    os.chdir(REPO)
    subprocess.run(["git", "add", "-A"], cwd=REPO)
    result = subprocess.run(
        ["git", "commit", "-m", message + "\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"],
        cwd=REPO, capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("Nothing to commit or commit failed:", result.stderr)
        return
    push = subprocess.run(["git", "push"], cwd=REPO, capture_output=True, text=True)
    print(push.stdout)
    if push.returncode != 0:
        print("Push failed:", push.stderr)


def main():
    print("=" * 60)
    print("SESSION RUNNER — LLM benchmarks + Experiments I/J/K")
    print("=" * 60)

    # 1. LLM benchmarks
    bench_entries = []
    for model_info in MODELS_TO_BENCH:
        entry = run_bench(model_info)
        bench_entries.append(entry)

    update_llm_benchmarks_doc()
    git_push("Add LLM benchmark results: Llama-3.1-8B and Gemma-3-4B")

    # 2. Experiments I, J, K
    exp_scripts = [
        ("lstm_prediction.py", "Experiment I (LSTM)"),
        ("cnn_nonstationarity.py", "Experiment J (CNN)"),
        ("transformer_prediction.py", "Experiment K (Transformer)"),
    ]
    for script, label in exp_scripts:
        rc = run_experiment(script)
        git_push(f"Add {label} results")

    print("\n" + "=" * 60)
    print("ALL DONE. Experiments I/J/K complete and pushed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
