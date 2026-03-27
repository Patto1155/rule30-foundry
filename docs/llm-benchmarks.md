# LLM Inference Benchmarks — GTX 1060 6GB

**Date:** 2026-03-27
**Engine:** llama.cpp build b8532 (CUDA SM 6.1)
**Config:** ngl=35 (all layers on GPU), n_batch=2048, n_ubatch=512, n_threads=4
**Tests:** pp512 = prompt processing 512 tokens, tg128 = text generation 128 tokens

---

## Results

| Model | Size | pp512 (t/s) | tg128 (t/s) | Notes |
|-------|------|------------|------------|-------|
| Qwen2.5-7B-Instruct Q4_K_M | 4.4 GB | **414** | **23.2** | Best general 7B |
| DeepSeek-R1-Distill-Qwen-7B Q4_K_M | 4.4 GB | **406** | **24.0** | Best reasoning 7B |

Both models fit fully in 6 GB VRAM at Q4_K_M quantization.
Performance is nearly identical — both are the same Qwen2 architecture at 7B parameters.

---

## Qwen2.5-7B ngl Sweep (tuning reference)

| ngl | pp512 (t/s) | tg128 (t/s) |
|-----|------------|------------|
| 0 (CPU only) | 277 | 4.7 |
| 10 | 285 | 6.4 |
| 20 | 356 | 11.4 |
| 28 | 407 | 20.1 |
| **35 (all)** | **414** | **23.2** |

**Use ngl=35 for 7B models.** The performance curve is steep up to ngl=28, then plateaus — all 35 transformer layers fit comfortably.

---

## Benchmarks Still To Run

- Llama-3.1-8B-Instruct Q4_K_M — `-hf bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M`
- Gemma-3-4B-Instruct Q4_K_M — `-hf bartowski/gemma-3-4b-it-GGUF:Q4_K_M`

---

## Raw Data

- Qwen2.5-7B: `D:\llm\benchmark_results.json`
- DeepSeek-R1: results inline above (raw JSON archived locally)

---

## Notes

- Memory bandwidth utilization at tg128: ~88 GB/s / 192 GB/s peak = **~46%** (typical for GDDR5 + 7B model)
- Both models are equally fast because DeepSeek-R1-Distill-Qwen-7B uses the same Qwen2 architecture with identical weight dimensions
- For code generation (Experiment I/K), Qwen2.5 is the better choice; for reasoning chains, DeepSeek-R1
