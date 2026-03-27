# GPU Benchmark — GTX 1060 6GB (Pascal SM 6.1)

**Date:** 2026-03-27
**Machine:** Windows 10, Intel i5-7600K @ 3.8GHz, 16GB RAM
**GPU:** NVIDIA GeForce GTX 1060 6GB
**CUDA SM:** 6.1 (Pascal)

---

## Hardware Specs

| Property            | Value                |
|---------------------|----------------------|
| GPU                 | GTX 1060 6GB         |
| Compute Capability  | SM 6.1 (Pascal)      |
| VRAM                | 6 GB GDDR5           |
| Peak FP32           | ~4.0 TFLOPS          |
| Memory Bandwidth    | 192 GB/s             |
| CUDA Cores          | 1280                 |
| CPU                 | i5-7600K @ 3.8GHz    |
| RAM                 | 16 GB                |

---

## LLM Inference — llama.cpp (build b8532, CUDA)

Model: **Qwen2.5-7B-Instruct Q4_K_M** (4.4 GB, 7.6B params)

| GPU Layers (ngl) | Prompt (pp512) | Generate (tg128) | Notes              |
|------------------|---------------|------------------|--------------------|
| 0 (CPU only)     | 277 t/s       | 4.7 t/s          | GPU idle           |
| 10               | 285 t/s       | 6.4 t/s          |                    |
| 20               | 356 t/s       | 11.4 t/s         |                    |
| 28               | 407 t/s       | 20.1 t/s         |                    |
| **35 (all)**     | **414 t/s**   | **23.2 t/s**     | **Optimal config** |

**Best config: `ngl=35`** — all layers fit in 6 GB VRAM.
Memory bandwidth utilization at peak: ~88 GB/s / 192 GB/s = **~46%**
(LLM inference is memory-bound; 46% is typical for GDDR5-class GPUs.)

---

## Rule 30 GPU Simulation — CuPy CUDA Kernel

Kernel: bit-packed uint64 tape, CUDA RawKernel, 1 thread per 64-cell word.

| Config              | Value             |
|---------------------|-------------------|
| Tape width          | 21,000,000 cells  |
| Steps computed      | 10,000,000        |
| Throughput          | **27,500 steps/s** |
| Cell throughput     | **579 Gcells/s**  |
| Center col verified | fraction_ones = 0.500222 ✓ |

Estimated effective bitwise ops: ~3 ops/cell × 579 Gcells/s ≈ **1.7 TOPS** (bitwise).

---

## Effective Compute Budget

| Task              | Throughput     | GPU Utilization |
|-------------------|---------------|-----------------|
| Rule 30 (CuPy)    | 579 Gcells/s  | High            |
| LLM prompt proc.  | 414 t/s       | High            |
| LLM generation    | 23.2 t/s      | Memory-bound    |
| 10M-step sim      | ~6 min total  | —               |

---

## Best Practices for GTX 1060 (SM 6.1)

**Inference engines (compatible):**
- `llama.cpp` / `llama-server` — primary recommendation, CUDA SM 6.1 support
- `ExLlamaV2` / `TabbyAPI` — best efficiency with EXL2 quants
- `LM Studio` — GUI wrapper over llama.cpp

**Not compatible:** vLLM, SGLang, TRT-LLM (require SM 7.0+)

**Model size:** 6 GB VRAM fits 7–8B models at Q4 quantization (~4.5–5 GB).

**Optimal llama.cpp flags for 7B Q4:**
```
-ngl 35 -n_batch 2048 -n_ubatch 512 -t 4
```

**Python GPU (CuPy):**
- Install: `pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12`
- No CUDA toolkit needed (uses pip-installed nvrtc)
- `cp.RawKernel` for custom CUDA kernels (e.g. Rule 30 simulation)

**Avoid:**
- Ollama: adds overhead, less control over ngl and quantization
- vLLM/SGLang: SM 7.0+ only
- Atomic ops in CuPy kernels for packed-bit extraction (race conditions observed)

---

## Compute Cost Model

To estimate run time for new experiments:

- **Memory-bound ops** (reads/writes, no computation): `size_GB / 192 GB/s`
- **LLM codegen** at 23 t/s: 1000 tokens ≈ 43 seconds
- **Rule 30 simulation**: 1M steps on 21M-cell tape ≈ 36 seconds
- **NumPy CPU experiments** on 10M bits: typically 1–15 minutes depending on order

---

*Generated from empirical benchmark runs. See `data/center_col_10M_results.json` and `D:\llm\benchmark_results.json` for raw data.*
