#!/usr/bin/env python
"""Experiment N — Motif Mining and Grammar Compression (CPU).

Tests whether the Rule 30 center column contains any compressible / repeating
structure, using three complementary approaches:

  1. Compression ratio  — zlib/lzma on sliding chunks vs random baseline
  2. K-mer frequency    — count all k-bit words, test deviation from uniform
  3. Re-Pair grammar    — a simplified digram replacement to estimate kolmogorov-
                          complexity proxy (compression ratio, grammar size)

If ANY approach finds compression ratio < 1.0 relative to random, or k-mer
distribution significantly non-uniform, that is evidence against irreducibility.

Visualization outputs:
  1. Compression ratio vs chunk size (Rule 30 vs random)
  2. K-mer frequency rank plots (k=8, 12, 16) — should be flat if uniform
  3. Top deviating k-mers (observed vs expected count)
"""

import os
import sys
import time
import json
import zlib
import lzma
import datetime
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from scipy.stats import chi2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE   = Path(r"D:\APATPROJECTS\rule30-research\data\center_col_46M.bin")
OUT_JSON    = Path(r"D:\APATPROJECTS\rule30-research\data\motif_mining.json")
PLOT_FILE   = Path(r"D:\APATPROJECTS\rule30-research\docs\plots\motif_mining.png")
LOG_FILE    = Path(r"D:\APATPROJECTS\rule30-research\docs\experiment-logs\N_motif_mining.md")

TOTAL_BITS   = 46_000_000
ANALYSIS_BITS = 10_000_000   # use first 10M for most tests (fast enough)
KMER_SIZES   = [8, 12, 16]   # k values for k-mer frequency analysis
# Chunk sizes for compression ratio test (in bits)
CHUNK_SIZES  = [1_000, 4_000, 16_000, 64_000, 256_000, 1_000_000]
N_CHUNKS     = 20             # average over N_CHUNKS chunks per size

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_bits(path: Path, n_bits: int) -> np.ndarray:
    n_bytes = (n_bits + 7) // 8
    raw = np.fromfile(str(path), dtype=np.uint8, count=n_bytes)
    return np.unpackbits(raw, bitorder='little')[:n_bits]


def bits_to_bytes_packed(bits: np.ndarray) -> bytes:
    """Pack bits back to bytes (MSB first, zero-pad to byte boundary)."""
    padded = np.zeros(((len(bits) + 7) // 8) * 8, dtype=np.uint8)
    padded[:len(bits)] = bits
    return np.packbits(padded).tobytes()


def compression_ratio(data_bytes: bytes, method: str = "zlib") -> float:
    """Return compressed_size / original_size."""
    if method == "zlib":
        compressed = zlib.compress(data_bytes, level=9)
    elif method == "lzma":
        compressed = lzma.compress(data_bytes,
                                   format=lzma.FORMAT_RAW,
                                   filters=[{"id": lzma.FILTER_LZMA2,
                                             "preset": 6}])
    else:
        raise ValueError(f"Unknown method: {method}")
    return len(compressed) / len(data_bytes)


def kmer_uniformity_test(bits: np.ndarray, k: int) -> dict:
    """
    Count all k-bit words in bits, run chi-squared test against uniform distribution.
    Returns dict with chi2 stat, p-value, top deviating k-mers.
    """
    n = len(bits)
    n_kmers = n - k + 1
    expected_count = n_kmers / (2 ** k)

    # Build k-mer integers efficiently using sliding window
    # Represent each k-mer as an integer (bit 0 = leftmost)
    if k > 24:
        # Use string approach for large k to avoid memory issues
        counts = Counter()
        for i in range(0, n_kmers, max(1, n_kmers // 5_000_000)):
            w = tuple(bits[i:i+k])
            counts[w] += 1
        observed = np.array(list(counts.values()), dtype=np.float64)
    else:
        # Efficient: roll a bitmask window
        window = np.uint32(0)
        mask   = np.uint32((1 << k) - 1)
        counts = np.zeros(1 << k, dtype=np.int64)

        # Prime the window
        for i in range(min(k, n)):
            window = ((window << np.uint32(1)) | np.uint32(int(bits[i]))) & mask

        for i in range(k, n):
            counts[int(window)] += 1
            window = ((window << np.uint32(1)) | np.uint32(int(bits[i]))) & mask
        counts[int(window)] += 1  # last window

        observed = counts.astype(np.float64)

    # Chi-squared test
    # Only include cells with expected >= 5 for validity
    total_cells = 2 ** k
    if expected_count >= 5:
        chi2_stat = float(np.sum((observed - expected_count) ** 2 / expected_count))
        df = total_cells - 1
        p_value = float(1 - chi2.cdf(chi2_stat, df))
    else:
        # Too many cells, too few samples — report raw deviation instead
        chi2_stat = float(np.sum((observed - expected_count) ** 2 / max(expected_count, 1)))
        df = total_cells - 1
        p_value = float(1 - chi2.cdf(chi2_stat, df)) if expected_count >= 1 else None

    # Top deviating k-mers
    if k <= 16:
        deviations = np.abs(observed - expected_count)
        top_idx = np.argsort(deviations)[-10:][::-1]
        top_kmers = [
            {
                "kmer": format(int(i), f"0{k}b"),
                "count": int(observed[i]),
                "expected": round(expected_count, 2),
                "deviation": round(float(deviations[i]), 2),
                "z": round(float((observed[i] - expected_count) /
                                 np.sqrt(expected_count)), 4),
            }
            for i in top_idx
        ]
    else:
        top_kmers = []

    return {
        "k":            k,
        "n_distinct":   int(np.sum(observed > 0)),
        "total_cells":  total_cells,
        "expected_per_cell": round(expected_count, 2),
        "chi2_stat":    round(chi2_stat, 4),
        "df":           df,
        "p_value":      round(p_value, 6) if p_value is not None else None,
        "uniform":      p_value > 0.01 if p_value is not None else None,
        "top_kmers":    top_kmers,
    }


# ---------------------------------------------------------------------------
# Simplified Re-Pair compression proxy
# ---------------------------------------------------------------------------

def repear_compression_ratio(bits: np.ndarray, max_rounds: int = 50) -> dict:
    """
    Approximate Re-Pair: repeatedly replace the most frequent digram.
    Returns grammar size / input size as proxy for Kolmogorov complexity.
    Works on a sample to keep runtime under 60s.
    """
    # Work on byte-level for speed (treat bytes as symbols)
    n_sample = min(500_000, len(bits))
    n_bytes  = n_sample // 8
    # Pack bits to bytes
    padded = np.zeros(((n_sample + 7) // 8) * 8, dtype=np.uint8)
    padded[:n_sample] = bits[:n_sample]
    sequence = list(np.packbits(padded[:n_bytes * 8]).tolist())

    original_len = len(sequence)
    grammar_rules = 0
    next_symbol = 256  # new symbol IDs start here

    for _ in range(max_rounds):
        if len(sequence) < 2:
            break
        # Count digrams
        digrams = Counter(zip(sequence[:-1], sequence[1:]))
        if not digrams:
            break
        most_common_digram, freq = digrams.most_common(1)[0]
        if freq < 2:
            break  # no more repeated digrams
        # Replace all non-overlapping occurrences
        new_seq = []
        i = 0
        while i < len(sequence) - 1:
            if sequence[i] == most_common_digram[0] and sequence[i+1] == most_common_digram[1]:
                new_seq.append(next_symbol)
                i += 2
            else:
                new_seq.append(sequence[i])
                i += 1
        if i == len(sequence) - 1:
            new_seq.append(sequence[-1])
        sequence = new_seq
        grammar_rules += 1
        next_symbol += 1

    grammar_ratio = (len(sequence) + grammar_rules) / original_len
    return {
        "input_bytes":    original_len,
        "output_symbols": len(sequence),
        "grammar_rules":  grammar_rules,
        "grammar_ratio":  round(grammar_ratio, 6),
        "rounds_run":     min(max_rounds, grammar_rules),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Experiment N — Motif Mining and Grammar Compression")
    print("=" * 65)

    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found. Run the simulation first.")
        sys.exit(1)

    t0 = time.perf_counter()
    print(f"Loading {ANALYSIS_BITS//1_000_000}M bits …")
    bits = load_bits(DATA_FILE, ANALYSIS_BITS)
    print(f"  Loaded. fraction_ones={bits.mean():.6f}")

    rng = np.random.default_rng(seed=42)
    random_bits = rng.integers(0, 2, size=ANALYSIS_BITS, dtype=np.uint8)

    results = {}

    # -----------------------------------------------------------------------
    # 1. Compression ratio vs chunk size
    # -----------------------------------------------------------------------
    print("\n--- Part 1: Compression Ratio ---")
    comp_results = []
    for method in ["zlib", "lzma"]:
        print(f"  Method: {method}")
        for chunk_bits in CHUNK_SIZES:
            chunk_bytes_n = chunk_bits // 8
            r30_ratios = []
            rnd_ratios = []
            for trial in range(N_CHUNKS):
                start = trial * chunk_bits
                if start + chunk_bits > ANALYSIS_BITS:
                    break
                r30_chunk = bits_to_bytes_packed(bits[start : start + chunk_bits])
                rnd_chunk = bits_to_bytes_packed(random_bits[start : start + chunk_bits])
                r30_ratios.append(compression_ratio(r30_chunk, method))
                rnd_ratios.append(compression_ratio(rnd_chunk, method))
            entry = {
                "method":        method,
                "chunk_bits":    chunk_bits,
                "rule30_ratio":  round(float(np.mean(r30_ratios)), 5),
                "random_ratio":  round(float(np.mean(rnd_ratios)), 5),
                "diff":          round(float(np.mean(r30_ratios)) - float(np.mean(rnd_ratios)), 5),
            }
            comp_results.append(entry)
            print(f"    chunk={chunk_bits//1000}K bits  "
                  f"rule30={entry['rule30_ratio']:.4f}  "
                  f"random={entry['random_ratio']:.4f}  "
                  f"diff={entry['diff']:+.5f}")
    results["compression"] = comp_results

    # -----------------------------------------------------------------------
    # 2. K-mer frequency test
    # -----------------------------------------------------------------------
    print("\n--- Part 2: K-mer Frequency Tests ---")
    kmer_results = []
    for k in KMER_SIZES:
        print(f"  k={k}  ({2**k} possible k-mers) …", end="", flush=True)
        t_k = time.perf_counter()
        # Use first 5M bits for k-mer test
        kmer_bits = min(5_000_000, ANALYSIS_BITS)
        kr = kmer_uniformity_test(bits[:kmer_bits], k)
        kmer_results.append(kr)
        status = "UNIFORM" if kr["uniform"] else f"NON-UNIFORM (p={kr['p_value']:.4f})"
        print(f"  chi2={kr['chi2_stat']:.1f}  df={kr['df']}  p={kr['p_value']}  → {status}  [{time.perf_counter()-t_k:.1f}s]")
    results["kmer"] = kmer_results

    # -----------------------------------------------------------------------
    # 3. Re-Pair grammar compression
    # -----------------------------------------------------------------------
    print("\n--- Part 3: Re-Pair Grammar Compression ---")
    print("  Rule 30 …", end="", flush=True)
    t_rp = time.perf_counter()
    rp_r30 = repear_compression_ratio(bits)
    print(f"  ratio={rp_r30['grammar_ratio']:.5f}  rules={rp_r30['grammar_rules']}  [{time.perf_counter()-t_rp:.1f}s]")

    print("  Random  …", end="", flush=True)
    t_rp2 = time.perf_counter()
    rp_rnd = repear_compression_ratio(random_bits)
    print(f"  ratio={rp_rnd['grammar_ratio']:.5f}  rules={rp_rnd['grammar_rules']}  [{time.perf_counter()-t_rp2:.1f}s]")

    results["repair"] = {
        "rule30": rp_r30,
        "random": rp_rnd,
        "ratio_diff": round(rp_r30["grammar_ratio"] - rp_rnd["grammar_ratio"], 6),
    }

    # -----------------------------------------------------------------------
    # 4. Overall verdict
    # -----------------------------------------------------------------------
    print("\n--- Summary ---")
    compressible = any(e["diff"] < -0.01 for e in comp_results)
    nonuniform_kmer = any(not kr["uniform"] for kr in kmer_results if kr["uniform"] is not None)
    repear_diff = results["repair"]["ratio_diff"]
    compressible_repear = repear_diff < -0.01

    flags = []
    if compressible:          flags.append("zlib/lzma compression below random baseline")
    if nonuniform_kmer:       flags.append("non-uniform k-mer distribution detected")
    if compressible_repear:   flags.append("Re-Pair grammar ratio below random baseline")

    if flags:
        verdict = "STRUCTURE DETECTED: " + "; ".join(flags)
    else:
        verdict = ("No compressible structure found. "
                   "Compression ratios match random baseline. "
                   "K-mer distributions uniform. "
                   "Grammar compression no better than random. "
                   "Consistent with computational irreducibility.")
    print(f"VERDICT: {verdict}")

    elapsed = time.perf_counter() - t0

    # Save JSON
    results["verdict"] = verdict
    results["elapsed_s"] = round(elapsed, 1)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(str(OUT_JSON), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON → {OUT_JSON}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    print("Generating plots …")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Experiment N — Motif Mining & Grammar Compression\n"
                     "Rule 30 Center Column (10M bits)",
                     fontsize=13, fontweight="bold")

        # Panel 1: compression ratio vs chunk size (zlib only)
        ax = axes[0]
        zlib_entries = [e for e in comp_results if e["method"] == "zlib"]
        xs = [e["chunk_bits"] / 1000 for e in zlib_entries]
        r30_ys = [e["rule30_ratio"] for e in zlib_entries]
        rnd_ys = [e["random_ratio"] for e in zlib_entries]
        ax.plot(xs, r30_ys, "o-", color="#2196F3", lw=2, label="Rule 30")
        ax.plot(xs, rnd_ys, "s--", color="#FF5722", lw=2, label="Random baseline")
        ax.set_xscale("log")
        ax.set_xlabel("Chunk size (Kbits, log scale)")
        ax.set_ylabel("Compression ratio (zlib, lower = more compressible)")
        ax.set_title("Compression ratio vs chunk size\nshould overlap if random")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(1.0, color="gray", lw=0.8, ls=":")

        # Panel 2: k-mer frequency deviation for k=8
        ax = axes[1]
        kr8 = next((kr for kr in kmer_results if kr["k"] == 8), None)
        if kr8 and kr8["top_kmers"]:
            kmers   = [d["kmer"] for d in kr8["top_kmers"]]
            zscores = [d["z"] for d in kr8["top_kmers"]]
            colors  = ["#F44336" if abs(z) > 3 else "#2196F3" for z in zscores]
            ax.bar(range(len(kmers)), zscores, color=colors)
            ax.set_xticks(range(len(kmers)))
            ax.set_xticklabels(kmers, rotation=90, fontsize=7)
            ax.axhline( 3, color="red", lw=1, ls="--", label="|z|=3")
            ax.axhline(-3, color="red", lw=1, ls="--")
            ax.set_xlabel("8-mer (binary)")
            ax.set_ylabel("Z-score (observed vs expected)")
            ax.set_title(f"Top 10 deviating 8-mers\n"
                         f"chi2={kr8['chi2_stat']:.1f}, p={kr8['p_value']:.4f}")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "k=8 data unavailable", ha="center", transform=ax.transAxes)
        ax.grid(True, alpha=0.3)

        # Panel 3: compression ratio comparison across methods + Re-Pair
        ax = axes[2]
        methods = ["zlib (1K)", "zlib (64K)", "zlib (1M)", "lzma (64K)", "Re-Pair"]
        r30_vals = [
            next((e["rule30_ratio"] for e in comp_results
                  if e["method"]=="zlib" and e["chunk_bits"]==1000), None),
            next((e["rule30_ratio"] for e in comp_results
                  if e["method"]=="zlib" and e["chunk_bits"]==64000), None),
            next((e["rule30_ratio"] for e in comp_results
                  if e["method"]=="zlib" and e["chunk_bits"]==1000000), None),
            next((e["rule30_ratio"] for e in comp_results
                  if e["method"]=="lzma" and e["chunk_bits"]==64000), None),
            results["repair"]["rule30"]["grammar_ratio"],
        ]
        rnd_vals = [
            next((e["random_ratio"] for e in comp_results
                  if e["method"]=="zlib" and e["chunk_bits"]==1000), None),
            next((e["random_ratio"] for e in comp_results
                  if e["method"]=="zlib" and e["chunk_bits"]==64000), None),
            next((e["random_ratio"] for e in comp_results
                  if e["method"]=="zlib" and e["chunk_bits"]==1000000), None),
            next((e["random_ratio"] for e in comp_results
                  if e["method"]=="lzma" and e["chunk_bits"]==64000), None),
            results["repair"]["random"]["grammar_ratio"],
        ]
        x = np.arange(len(methods))
        width = 0.35
        ax.bar(x - width/2, [v or 0 for v in r30_vals], width, color="#2196F3",
               alpha=0.8, label="Rule 30")
        ax.bar(x + width/2, [v or 0 for v in rnd_vals], width, color="#FF5722",
               alpha=0.8, label="Random")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Compression ratio")
        ax.set_title("Rule 30 vs Random across compression methods\nbars should be same height if no structure")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(PLOT_FILE), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot → {PLOT_FILE}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    # Experiment log
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(str(LOG_FILE), "w", encoding="utf-8") as f:
        f.write(f"""# Experiment Log

- Date: {datetime.date.today().isoformat()}
- Title: Motif Mining and Grammar Compression
- Goal: Search for compressible / repeating structure in Rule 30 center column using compression algorithms and k-mer statistics
- Setup: {ANALYSIS_BITS:,}-bit center column; zlib + lzma compression at {len(CHUNK_SIZES)} chunk sizes; k-mer chi-squared tests for k={KMER_SIZES}; Re-Pair grammar compression on 500K-bit sample
- Method:
  1. Compression ratio: compress Rule 30 chunks vs equally-sized random chunks at multiple scales
  2. K-mer frequency: chi-squared uniformity test for k-bit words, flag non-uniform distributions
  3. Re-Pair: repeatedly replace most-frequent digram; measure grammar size / input size
- Result:
  Compression ratios:
""")
        for e in comp_results:
            f.write(f"    {e['method']} @ {e['chunk_bits']//1000}K bits: "
                    f"rule30={e['rule30_ratio']:.5f}  random={e['random_ratio']:.5f}  "
                    f"diff={e['diff']:+.5f}\n")
        f.write(f"""  K-mer tests:\n""")
        for kr in kmer_results:
            f.write(f"    k={kr['k']}: chi2={kr['chi2_stat']:.1f}  p={kr['p_value']}  "
                    f"uniform={kr['uniform']}\n")
        f.write(f"""  Re-Pair: rule30={rp_r30['grammar_ratio']:.5f}  random={rp_rnd['grammar_ratio']:.5f}  diff={results['repair']['ratio_diff']:+.5f}
- Interpretation: {verdict}
- Verification: Three-panel plot saved to docs/plots/. Compression ratios should overlap Rule 30 and random if no structure.
- Elapsed: {elapsed:.0f}s
""")
    print(f"Log  → {LOG_FILE}")
    print(f"\nTotal: {elapsed:.1f}s  Done.")


if __name__ == "__main__":
    main()
