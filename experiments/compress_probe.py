#!/usr/bin/env python
"""Experiment N — Compression-Based Irreducibility Probe.

Directly attacks Prize Problem 2: if no compressor finds structure at any scale,
no algorithm can compute the sequence faster than step-by-step simulation.

Three angles:
  1. Multi-scale sweep: gzip/bz2 vs random baseline at scales 1K–46M bits
  2. Sliding window: does compression ratio vary with position? (non-stationarity)
  3. Run-length distribution: compare to geometric(0.5) expected from Bernoulli process
"""
import sys, gzip, bz2, json, time, datetime
import numpy as np
from pathlib import Path

DATA_FILE = Path(r"D:\APATPROJECTS\rule30-research\data\center_col_46M.bin")
OUT_JSON  = Path(r"D:\APATPROJECTS\rule30-research\data\compress_probe.json")
LOG_FILE  = Path(r"D:\APATPROJECTS\rule30-research\docs\experiment-logs\N_compress_probe.md")
PLOT_FILE = Path(r"D:\APATPROJECTS\rule30-research\docs\plots\compress_probe.png")
PROG_LOG  = Path(r"D:\APATPROJECTS\rule30-research\data\N_progress.log")

TOTAL_BYTES = 5_750_000   # 46M bits packed

TEST = "--test" in sys.argv

# Scales in BYTES (raw packed file)
if TEST:
    TEST_BYTE_SIZES  = [128, 512, 2048, 8192]
    N_WINDOWS        = 4
    WINDOW_BYTES     = 8_192
    RL_SAMPLE_BYTES  = 8_192
else:
    TEST_BYTE_SIZES  = [2**k for k in range(7, 24)]   # 128B -> 8MB (covers full file)
    TEST_BYTE_SIZES  = [s for s in TEST_BYTE_SIZES if s <= TOTAL_BYTES] + [TOTAL_BYTES]
    N_WINDOWS        = 20
    WINDOW_BYTES     = TOTAL_BYTES // N_WINDOWS          # ~287K bytes = ~2.3M bits
    RL_SAMPLE_BYTES  = min(1_250_000, TOTAL_BYTES)       # 10M bits


def log(msg):
    ts   = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(PROG_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ratio(data: bytes, comp: str) -> float:
    """Compressed size / original size."""
    orig = len(data)
    if orig == 0:
        return 1.0
    if comp == "gzip":
        c = gzip.compress(data, compresslevel=9)
    elif comp == "bz2":
        c = bz2.compress(data, compresslevel=9)
    else:
        raise ValueError(comp)
    return len(c) / orig


def main():
    PROG_LOG.parent.mkdir(parents=True, exist_ok=True)
    open(PROG_LOG, "w").close()
    log("=== Experiment N — Compression-Based Irreducibility Probe ===")
    log(f"TEST={TEST}")
    t0 = time.perf_counter()

    rng = np.random.default_rng(42)

    # Load raw bytes (no unpack — compress the packed representation directly)
    log(f"Loading {DATA_FILE.name} ...")
    raw_all = DATA_FILE.read_bytes()[:TOTAL_BYTES]
    log(f"  Loaded {len(raw_all):,} bytes ({len(raw_all)*8:,} bits)")

    # ── Part 1: Multi-scale compression sweep ───────────────────────────
    log("\n--- Part 1: Multi-scale sweep ---")
    scale_results = []

    for nbytes in TEST_BYTE_SIZES:
        r30_data   = raw_all[:nbytes]
        rand_bytes = rng.integers(0, 256, size=nbytes, dtype=np.uint8).tobytes()

        row = {"n_bytes": nbytes, "n_bits": nbytes * 8}
        for comp in ["gzip", "bz2"]:
            rr = ratio(r30_data,   comp)
            rb = ratio(rand_bytes, comp)
            row[f"r30_{comp}"]       = round(rr, 6)
            row[f"rand_{comp}"]      = round(rb, 6)
            row[f"vs_random_{comp}"] = round(rr / rb if rb > 0 else 1.0, 6)
        scale_results.append(row)
        log(f"  {nbytes:>9,} B ({nbytes*8/1e6:6.2f}Mbit): "
            f"gzip r30={row['r30_gzip']:.4f} rand={row['rand_gzip']:.4f} "
            f"ratio={row['vs_random_gzip']:.4f}")

    # ── Part 2: Sliding window non-stationarity ──────────────────────────
    log(f"\n--- Part 2: Sliding window ({N_WINDOWS} x {WINDOW_BYTES:,} bytes) ---")
    window_results = []
    for i in range(N_WINDOWS):
        start = i * WINDOW_BYTES
        end   = start + WINDOW_BYTES
        if end > TOTAL_BYTES:
            break
        chunk = raw_all[start:end]
        r = ratio(chunk, "gzip")
        window_results.append({"start_byte": start, "gzip": round(r, 6)})
        log(f"  Window {i+1:>2}/{N_WINDOWS}  byte [{start:>9,}–{end:>9,}]: gzip={r:.6f}")

    w_ratios = [w["gzip"] for w in window_results]
    w_mean   = float(np.mean(w_ratios))
    w_std    = float(np.std(w_ratios))
    log(f"  Window gzip: mean={w_mean:.6f}, std={w_std:.6f}  "
        f"({'stationary' if w_std < 0.001 else 'NON-STATIONARY signal'})")

    # ── Part 3: Run-length distribution ─────────────────────────────────
    log(f"\n--- Part 3: Run-length distribution ({RL_SAMPLE_BYTES*8:,} bits) ---")
    rl_raw  = raw_all[:RL_SAMPLE_BYTES]
    rl_bits = np.unpackbits(np.frombuffer(rl_raw, dtype=np.uint8), bitorder="little")
    n_rl    = len(rl_bits)

    transitions = np.where(np.diff(rl_bits.astype(np.int8)) != 0)[0] + 1
    run_lengths = np.diff(np.concatenate([[0], transitions, [n_rl]]))

    mean_rl   = float(run_lengths.mean())
    median_rl = float(np.median(run_lengths))
    total_runs = len(run_lengths)

    # Histogram k=1..6, 7+
    rl_hist = {str(k): int(np.sum(run_lengths == k)) for k in range(1, 7)}
    rl_hist["7+"] = int(np.sum(run_lengths >= 7))
    # Expected counts under geometric(0.5): P(run=k) = (0.5)^k
    exp_hist = {str(k): int(round(0.5**k * total_runs)) for k in range(1, 7)}
    exp_hist["7+"] = int(round(0.5**6 * total_runs))   # ≈ P(run≥7)xtotal (approx)

    log(f"  Runs: {total_runs:,}  mean_length={mean_rl:.4f}  (geometric(0.5) expects 2.0)")
    log(f"  Obs:  {rl_hist}")
    log(f"  Exp:  {exp_hist}")

    elapsed = time.perf_counter() - t0
    log(f"\nDone in {elapsed:.1f}s")

    # Save JSON ----------------------------------------------------------
    last = scale_results[-1] if scale_results else {}
    results = {
        "total_bytes": TOTAL_BYTES,
        "test_mode": TEST,
        "scale_sweep": scale_results,
        "sliding_window": window_results,
        "window_gzip_mean": round(w_mean, 6),
        "window_gzip_std":  round(w_std,  6),
        "run_length_mean":   round(mean_rl, 6),
        "run_length_median": round(median_rl, 1),
        "run_length_hist":   rl_hist,
        "run_length_expected_geometric": exp_hist,
        "elapsed_s": round(elapsed, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(str(OUT_JSON), "w") as f:
        json.dump(results, f, indent=2)
    log(f"JSON  -> {OUT_JSON}")

    # Plot ---------------------------------------------------------------
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Experiment N — Compression-Based Irreducibility Probe\n"
                     "Rule 30 center column vs random baseline",
                     fontsize=12, fontweight="bold")

        # 1: compression ratio vs scale
        ax = axes[0]
        sizes_mb = [r["n_bits"] / 1e6 for r in scale_results]
        r30_g    = [r["r30_gzip"]  for r in scale_results]
        rand_g   = [r["rand_gzip"] for r in scale_results]
        ax.semilogx(sizes_mb, r30_g,  "b-o", ms=5, lw=1.5, label="Rule 30")
        ax.semilogx(sizes_mb, rand_g, "r--s", ms=5, lw=1.5, label="Random")
        ax.axhline(1.0, color="gray", lw=1, ls=":", label="Incompressible (1.0)")
        ax.set_xlabel("Sequence (Mbits)"); ax.set_ylabel("gzip compression ratio")
        ax.set_title("Compression ratio vs scale\n(below random = structure found)")
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_ylim(0.85, 1.15)

        # 2: sliding window
        ax = axes[1]
        w_starts = [w["start_byte"] * 8 / 1e6 for w in window_results]
        ax.plot(w_starts, w_ratios, "g-o", ms=5)
        ax.axhline(w_mean, color="gray", ls="--", lw=1, label=f"mean={w_mean:.4f}")
        ax.fill_between(w_starts,
                        [w_mean - 2*w_std]*len(w_starts),
                        [w_mean + 2*w_std]*len(w_starts),
                        alpha=0.15, color="green", label="+-2sigma")
        ax.set_xlabel("Window start (Mbit)"); ax.set_ylabel("gzip ratio")
        ax.set_title(f"Sliding window ({WINDOW_BYTES*8//1000}K-bit windows)\nstd={w_std:.6f}")
        ax.legend(); ax.grid(True, alpha=0.3)

        # 3: run-length distribution
        ax = axes[2]
        ks     = list(range(1, 7))
        obs_c  = [rl_hist[str(k)] for k in ks]
        exp_c  = [exp_hist[str(k)] for k in ks]
        x_pos  = np.arange(len(ks))
        w_bar  = 0.35
        ax.bar(x_pos - w_bar/2, obs_c, w_bar, label="Rule 30", color="steelblue")
        ax.bar(x_pos + w_bar/2, exp_c, w_bar, label="Geometric(0.5)", color="tomato", alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"len={k}" for k in ks])
        ax.set_ylabel("Count")
        ax.set_title(f"Run-length distribution\nmean={mean_rl:.4f} (expected 2.0)")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(PLOT_FILE), dpi=150, bbox_inches="tight"); plt.close()
        log(f"Plot  -> {PLOT_FILE}")
    except Exception as e:
        log(f"Plot skipped: {e}")

    # Experiment log -----------------------------------------------------
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(str(LOG_FILE), "w", encoding="utf-8") as f:
        f.write(f"""# Experiment Log — Compression-Based Irreducibility Probe

- Date: {datetime.date.today().isoformat()}
- Title: Multi-scale gzip/bz2 compression vs random baseline + sliding window + run-length analysis
- Goal: Attack Prize Problem 2 — if incompressible at all scales, no faster algorithm exists
- Data: center_col_46M.bin ({TOTAL_BYTES:,} bytes = 46M bits)
- Method:
  1. Compress raw bytes at scales 128B–{TOTAL_BYTES//1000}KB; compare to random baseline
  2. Sliding window: {N_WINDOWS} windows x {WINDOW_BYTES:,} bytes, measure gzip variation
  3. Run-length histogram vs geometric(0.5) expected from Bernoulli process
- Results (at full {TOTAL_BYTES:,} bytes):
  - gzip: r30={last.get('r30_gzip','?')}, random={last.get('rand_gzip','?')}, ratio={last.get('vs_random_gzip','?')}
  - bz2:  r30={last.get('r30_bz2','?')}, random={last.get('rand_bz2','?')}, ratio={last.get('vs_random_bz2','?')}
  - Window gzip std: {w_std:.6f}
  - Run-length mean: {mean_rl:.4f} (geometric(0.5) expects 2.0)
- Interpretation:
  {"Rule 30 compresses at roughly same ratio as random -> no structure detectable by dictionary-based compression -> supports Prize Problem 2 irreducibility." if abs(last.get('vs_random_gzip',1.0) - 1.0) < 0.01 else "Rule 30 compresses DIFFERENTLY from random — structure detected."}
  {"Window std < 0.001 -> compression ratio is stationary across the full sequence." if w_std < 0.001 else f"Window std = {w_std:.4f} -> possible non-stationarity."}
  {"Run-length mean ≈ 2.0 -> consistent with Bernoulli(0.5) at local scale." if abs(mean_rl - 2.0) < 0.1 else f"Run-length mean = {mean_rl:.3f} deviates from 2.0."}
- Elapsed: {elapsed:.0f}s
""")
    log(f"Log   -> {LOG_FILE}")


if __name__ == "__main__":
    main()
