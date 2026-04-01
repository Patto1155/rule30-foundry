#!/usr/bin/env python
"""Extended Rule 30 simulation — 100M or 200M steps (GPU).

Default target: 100M steps (~9 hours).
For 200M steps pass --200m (~35 hours — genuine overnight + day run).

Tape width = 2 x steps + 1 to prevent edge effects.
Saves center column as packed bits (bitorder='little') to match prior runs.

Telemetry: run alongside this script in a separate terminal:
  nvidia-smi dmon -s pucvmet -d 5 -f data\\gpu_telemetry.csv

Progress: tail -f data\\sim_progress.log
"""
import sys, os, json, time, datetime
import numpy as np
from pathlib import Path

try:
    import cupy as cp
    GPU = True
except ImportError:
    sys.exit("CuPy required for this experiment.")

# ── Configuration ────────────────────────────────────────────────────────────
if   "--200m" in sys.argv:
    N_STEPS = 200_000_000
elif "--test"  in sys.argv:
    N_STEPS = 500_000
else:
    N_STEPS = 100_000_000

N_CELLS_MIN = 2 * N_STEPS + 1
n_words     = (N_CELLS_MIN + 63) // 64
n_cells     = n_words * 64
center_word = n_words // 2
center_bit  = 32

SUFFIX      = "200m" if N_STEPS == 200_000_000 else ("test" if N_STEPS < 1_000_000 else "100m")
OUT_BIN     = Path(fr"D:\APATPROJECTS\rule30-research\data\center_col_{SUFFIX}.bin")
OUT_JSON    = Path(fr"D:\APATPROJECTS\rule30-research\data\center_col_{SUFFIX}_results.json")
PROG_LOG    = Path(r"D:\APATPROJECTS\rule30-research\data\sim_progress.log")

LOG_INTERVAL = max(1, N_STEPS // 1000)   # log every 0.1%
# ─────────────────────────────────────────────────────────────────────────────

KERNEL_SRC = r"""
extern "C" __global__
void rule30_center(
    const unsigned long long* tape,
    unsigned long long*       out,
    int                       n_words,
    int                       center_word,
    int                       center_bit,
    unsigned char*            center_col,
    long long                 step
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_words) return;

    unsigned long long c    = tape[idx];
    unsigned long long prev = (idx > 0)          ? tape[idx - 1] : 0ULL;
    unsigned long long next = (idx < n_words - 1) ? tape[idx + 1] : 0ULL;

    if (idx == center_word)
        center_col[step] = (unsigned char)((c >> center_bit) & 1ULL);

    unsigned long long L = (c >> 1) | (prev << 63);
    unsigned long long R = (c << 1) | (next >> 63);
    out[idx] = L ^ (c | R);
}
"""


def log(msg):
    ts   = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(PROG_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    PROG_LOG.parent.mkdir(parents=True, exist_ok=True)
    open(PROG_LOG, "w").close()

    log("=" * 60)
    log(f"Rule 30 Extended Simulation — {N_STEPS:,} steps")
    log("=" * 60)
    log(f"Tape:       {n_cells:,} cells = {n_words:,} uint64 words")
    log(f"Tape VRAM:  {n_words * 8 * 2 / 1024**2:.1f} MB (double-buffered)")
    log(f"Center col: {N_STEPS / 1024**2:.1f} MB on GPU")
    log(f"Center:     word {center_word}, bit {center_bit} -> cell {center_word*64+center_bit:,}")
    log(f"Output:     {OUT_BIN}")
    log(f"Telemetry:  nvidia-smi dmon -s pucvmet -d 5 -f data\\gpu_telemetry.csv")
    log("")

    kernel = cp.RawKernel(KERNEL_SRC, "rule30_center")

    # Allocate ------------------------------------------------------------
    tape_a = cp.zeros(n_words, dtype=cp.uint64)
    tape_b = cp.zeros(n_words, dtype=cp.uint64)
    tape_a[center_word] = cp.uint64(1 << center_bit)
    center_gpu = cp.zeros(N_STEPS, dtype=cp.uint8)

    block  = 256
    grid   = (n_words + block - 1) // block

    # Known first 20 bits for verification
    EXPECTED_20 = [1,1,0,1,1,1,0,0,1,1,0,0,0,1,0,1,1,0,0,1]

    log("Starting simulation ...")
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    t_last_log = t0

    cur, nxt = tape_a, tape_b
    for step in range(N_STEPS):
        kernel((grid,), (block,),
               (cur, nxt, np.int32(n_words),
                np.int32(center_word), np.int32(center_bit),
                center_gpu, np.int64(step)))
        cur, nxt = nxt, cur

        if step % LOG_INTERVAL == 0 and step > 0:
            now      = time.perf_counter()
            elapsed  = now - t0
            sps      = step / elapsed
            eta_s    = (N_STEPS - step) / sps if sps > 0 else 0
            eta_str  = str(datetime.timedelta(seconds=int(eta_s)))
            pct      = 100.0 * step / N_STEPS
            log(f"  {pct:5.1f}%  step {step:,}/{N_STEPS:,}  "
                f"{sps:,.0f} steps/s  ETA {eta_str}")
            t_last_log = now

    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0
    log(f"\nSimulation done in {elapsed:.1f}s ({elapsed/3600:.2f}h)")

    # Transfer and verify ------------------------------------------------
    log("Transferring center column ...")
    center_bits = cp.asnumpy(center_gpu)
    fraction    = float(center_bits.mean())
    actual_20   = center_bits[:20].tolist()
    verified    = actual_20 == EXPECTED_20

    log(f"  fraction_ones = {fraction:.8f}  (expected ~0.5)")
    log(f"  First 20 bits: {actual_20}")
    log(f"  Expected:      {EXPECTED_20}")
    log(f"  Verification:  {'PASS PASS' if verified else 'FAIL FAIL — data may be corrupt!'}")

    # Save binary --------------------------------------------------------
    log(f"Packing and saving to {OUT_BIN} ...")
    packed = np.packbits(center_bits, bitorder="little")
    OUT_BIN.parent.mkdir(parents=True, exist_ok=True)
    with open(str(OUT_BIN), "wb") as f:
        f.write(packed.tobytes())
    log(f"  Saved {OUT_BIN.stat().st_size:,} bytes ({N_STEPS:,} bits)")

    # Results JSON -------------------------------------------------------
    results = {
        "n_steps":           N_STEPS,
        "n_cells":           n_cells,
        "n_words":           n_words,
        "center_cell":       center_word * 64 + center_bit,
        "elapsed_seconds":   round(elapsed, 2),
        "steps_per_second":  round(N_STEPS / elapsed, 1),
        "gcells_per_second": round(n_cells * N_STEPS / elapsed / 1e9, 3),
        "fraction_ones":     round(fraction, 8),
        "verification_passed": verified,
        "first_20_bits":     actual_20,
        "output_file":       str(OUT_BIN),
        "output_bytes":      int(OUT_BIN.stat().st_size),
    }
    with open(str(OUT_JSON), "w") as f:
        json.dump(results, f, indent=2)
    log(f"JSON  -> {OUT_JSON}")

    log(f"\n{'='*60}")
    log(f"DONE — {N_STEPS:,} steps in {elapsed/3600:.2f}h")
    log(f"  {results['steps_per_second']:,.0f} steps/s  |  "
        f"{results['gcells_per_second']:.1f} Gcells/s")
    log(f"  fraction_ones = {fraction:.8f}")
    log(f"  Verification: {'PASS' if verified else 'FAIL'}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
