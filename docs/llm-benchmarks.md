# LLM Inference Benchmarks — GTX 1060 6GB

**Date:** 2026-03-27
**Engine:** llama.cpp build b8532 (CUDA SM 6.1)
**Config:** ngl=35 (all layers on GPU), n_batch=2048, n_ubatch=512, n_threads=4
**Tests:** pp512 = prompt processing 512 tokens, tg128 = text generation 128 tokens

---

## Results

| Model | Size | pp512 (t/s) | tg128 (t/s) |
|-------|------|------------|------------|
| Qwen2.5-7B-Instruct-Q4_K_M | 4.7 GB | **413.631** | **23.176** |
| DeepSeek-R1-Distill-Qwen-7B-Q4_K_M | 4.7 GB | **405.97** | **24.04** |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M | 4.9 GB | **362.842** | **22.473** |
| Qwen2.5-3B-Instruct-Q4_K_M | 1.9 GB | **394.521** | **19.102** |

---

## Qwen2.5-7B ngl Sweep (tuning reference)

| ngl | pp512 (t/s) | tg128 (t/s) |
|-----|------------|------------|
| 0 (CPU only) | 277 | 4.7 |
| 10 | 285 | 6.4 |
| 20 | 356 | 11.4 |
| 28 | 407 | 20.1 |
| **35 (all)** | **414** | **23.2** |

**Use ngl=35 for 7B models.** The curve is steep up to ngl=28, then plateaus — all transformer layers fit in 6 GB VRAM.

---

## Observations

- **Qwen2 architecture** (Qwen2.5-7B and DeepSeek-R1-Distill-Qwen-7B): nearly identical throughput because both use the same 7B Qwen2 weight layout
- **Llama 3.1 8B** (4.9 GB): ~12% slower prompt vs Qwen 7B, same generation — slightly larger model fills VRAM more tightly
- **Qwen2.5-3B** (1.9 GB): prompt within 5% of 7B models (bandwidth-limited), generation ~19 t/s — fits 3B in ~1.9 GB VRAM, leaving 4 GB for KV cache
- Memory bandwidth utilization at tg128: ~88 GB/s / 192 GB/s = **~46%** (GDDR5, typical for single-GPU LLM inference)

## Notes

- Gemma-3-4B and Phi-4-mini skipped — gated repos require HF token (Google/Microsoft ToS). Substitute: Qwen2.5-3B.
- For code/reasoning experiments: Qwen2.5-7B and DeepSeek-R1-7B both suitable; DeepSeek-R1 has reasoning chain training

## Raw Data

`D:\llmenchmark_results.json`
