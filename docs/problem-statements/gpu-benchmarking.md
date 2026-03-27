# GPU Benchmarking As A Research Primitive

- Date: 2026-03-27
- Owner: Local workstation (GTX 1060 6GB)

## Goal

Treat the GPU as a measurable, schedulable research resource.

Primary outputs:

1. A reproducible benchmark for this specific machine (so "GPU hours" becomes a concrete budget).
2. A short public summary (README) that makes the project's compute capacity legible.
3. A set of operating best-practices (so time is spent on useful runs, not avoidable overhead).

Non-goals:

- Uploading large artifacts to the public repo.
- Chasing synthetic FLOPS numbers with no connection to this project's kernels/workloads.

## Metrics To Track (Minimum Set)

Record these in a dated experiment log and keep a rolling "best known" in the README.

- Rule 30 throughput:
  - `cells`, `steps`, `steps/s`, `Gcells/s`, `VRAM MB`, `fraction_ones`, verification pass/fail
- Linear algebra throughput (optional but useful):
  - FP32 GEMM: `N`, elapsed, effective TFLOP/s (CuPy `matmul`)
- LLM inference throughput (optional, separate report under `D:\llm\`):
  - `llama-bench` prompt processing and token generation throughput across `ngl` settings

## Artifact Policy (Repo Hygiene)

Public repo:

- Commit and push:
  - Markdown logs under `docs/experiment-logs/`
  - Small JSON summaries (only if small and stable)
  - Code changes
- Do NOT commit:
  - Large binaries (`.bin`, `.gguf`), large `.csv`, `.npy` outputs

Local machine (not committed):

- Keep large artifacts under `data/` and `D:\llm\` and reference them from logs with:
  - absolute path
  - file size
  - SHA256 (or at least modification time + size if hashing is too slow)

## Benchmark Procedure (Suggested)

1. Confirm environment:
  - Windows + NVIDIA driver OK
  - CuPy can compile kernels (NVRTC present)
  - `tqdm` installed for long runs

2. Rule 30 "real workload" benchmark:
  - Generate center column with correct tape width:
    - requirement: `cells >= 2*steps + 1` (light cone)
  - Record throughput for:
    - no extraction (pure stepping)
    - extraction enabled (if it materially changes performance)

3. Optional: CuPy FP32 GEMM microbench:
  - Run one or two matrix sizes that fit in VRAM (e.g. N=4096, 8192)
  - Report effective TFLOP/s and include exact code/config in the log

4. Optional: `llama.cpp` benchmark:
  - Prefer `llama-bench` and record:
    - prompt processing tokens/s
    - generation tokens/s
    - `ngl`, context, batch size

## Best Practices (Operational)

- Prefer `llama.cpp` tooling for local LLM benchmarking/control (`llama-bench`, `llama-server`) over higher-level wrappers when you care about reproducible performance numbers.
- Always show progress for long runs (tqdm/progress output).
- Push small results incrementally: each completed experiment/benchmark should have a log and a push, not an end-of-day dump.
- Keep "GPU hours" useful:
  - preflight small runs to validate correctness
  - batch/avoid per-step CPU sync in GPU kernels
  - record parameters so results remain interpretable later

## Definition Of Done

- A dated benchmark log exists under `docs/experiment-logs/` with the minimum metrics.
- README has a short "GPU benchmark" summary and a pointer to the full log.
- Large artifacts remain local, with references recorded in logs.

