#!/usr/bin/env python
"""Full-Range FFT Autocorrelation — all lags 1 to 23M.

Resolves the period-42795 candidate from the windowed period search.
The windowed search found z=4.66 in a local window; the FFT gives the
true autocorrelation across the entire 46M-bit sequence.

Key question: does lag 42,795 stand out from noise in the full sequence?
If not, the candidate is a windowing artifact and can be closed.
"""
import sys, json, time, datetime
import numpy as np
from pathlib import Path

DATA_FILE = Path(r"D:\APATPROJECTS\rule30-research\data\center_col_46M.bin")
OUT_JSON  = Path(r"D:\APATPROJECTS\rule30-research\data\fft_autocorr.json")
LOG_FILE  = Path(r"D:\APATPROJECTS\rule30-research\docs\experiment-logs\fft_autocorr.md")
PLOT_FILE = Path(r"D:\APATPROJECTS\rule30-research\docs\plots\fft_autocorr.png")
PROG_LOG  = Path(r"D:\APATPROJECTS\rule30-research\data\fft_autocorr.progress.log")

TOTAL_BITS       = 46_000_000
N_TOP            = 200
PERIOD_CANDIDATE = 42_795   # from period_search experiment (z=4.66, match=52.33%)
TEST             = "--test" in sys.argv


def log(msg):
    ts  = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(PROG_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    PROG_LOG.parent.mkdir(parents=True, exist_ok=True)
    open(PROG_LOG, "w").close()
    log("=== Full-Range FFT Autocorrelation ===")
    log(f"TEST={TEST}")
    t0 = time.perf_counter()

    # Check cuFFT separately — CuPy kernels may work without cuFFT DLL
    try:
        import cupy as cp
        cp.fft.rfft(cp.zeros(4, dtype=cp.float32))  # probe cuFFT
        GPU = True
        log("CuPy + cuFFT available — FFT on GPU")
    except Exception:
        GPU = False
        log("cuFFT not available — FFT on CPU (NumPy, still fast)")

    # Load ---------------------------------------------------------------
    n_bits = 1_000_000 if TEST else TOTAL_BITS
    n_bytes = (n_bits + 7) // 8
    log(f"Loading {n_bits:,} bits from {DATA_FILE.name} ...")
    raw  = np.fromfile(str(DATA_FILE), dtype=np.uint8, count=n_bytes)
    bits = np.unpackbits(raw, bitorder="little")[:n_bits]
    log(f"  fraction_ones={bits.mean():.8f}")

    # Convert 0/1 -> +-1 (zero-mean approximation)
    x   = bits.astype(np.float32) * 2.0 - 1.0
    N   = len(x)
    sigma = 1.0 / np.sqrt(N)
    log(f"  N={N:,}  |  1sigma={sigma:.8f}  |  5sigma={5*sigma:.8f}")

    # FFT autocorrelation ------------------------------------------------
    log("Computing FFT autocorrelation ...")
    t1 = time.perf_counter()

    if GPU:
        xg = cp.asarray(x)
        F  = cp.fft.rfft(xg, n=N)
        P  = (F * cp.conj(F)).real        # power spectrum (real)
        ag = cp.fft.irfft(P, n=N)
        a  = cp.asnumpy(ag) / N           # unnormalised circular autocorr
        del xg, F, P, ag
        cp.get_default_memory_pool().free_all_blocks()
    else:
        F = np.fft.rfft(x, n=N)
        P = (F * np.conj(F)).real
        a = np.fft.irfft(P, n=N) / N

    var_est = float(a[0])
    a_norm  = a / var_est if var_est > 0 else a
    log(f"  FFT done in {time.perf_counter()-t1:.1f}s  |  var(lag-0)={var_est:.8f}")

    # Period candidate ---------------------------------------------------
    cand_r = float(a_norm[PERIOD_CANDIDATE]) if PERIOD_CANDIDATE < N else 0.0
    cand_z = abs(cand_r) / sigma
    log(f"\n  Lag {PERIOD_CANDIDATE}: r={cand_r:+.8f}, z={cand_z:.2f}sigma  "
        f"({'SIGNIFICANT' if cand_z > 5 else 'not significant (candidate closed)'})")

    # Top N lags ---------------------------------------------------------
    half = N // 2
    abs_a = np.abs(a_norm[1:half])       # lags 1 .. N/2-1
    top_i = np.argpartition(abs_a, -N_TOP)[-N_TOP:]
    top_i = top_i[np.argsort(abs_a[top_i])[::-1]]
    top_lags = (top_i + 1).tolist()
    top_vals = abs_a[top_i].tolist()
    log(f"  Max |r|={top_vals[0]:.8f} at lag {top_lags[0]:,}")
    log(f"  Top 5: {[(top_lags[k], round(top_vals[k],8)) for k in range(min(5,len(top_lags)))]}")

    n_3s = int(np.sum(abs_a > 3 * sigma))
    n_5s = int(np.sum(abs_a > 5 * sigma))
    exp_3s = int(0.0027  * half)
    exp_5s = int(5.7e-7  * half)
    log(f"  Lags >3sigma: {n_3s:,}  (expected ~{exp_3s:,} by chance)")
    log(f"  Lags >5sigma: {n_5s:,}  (expected ~{exp_5s:,} by chance)")

    elapsed = time.perf_counter() - t0
    log(f"\nDone in {elapsed:.1f}s")

    # Save JSON ----------------------------------------------------------
    result = {
        "n_bits": N,
        "sigma_1": round(sigma, 10),
        "period_candidate_lag":  PERIOD_CANDIDATE,
        "period_candidate_r":    round(cand_r, 10),
        "period_candidate_z":    round(cand_z, 4),
        "period_candidate_significant": bool(cand_z > 5.0),
        "max_abs_r":   round(top_vals[0], 10),
        "max_abs_lag": int(top_lags[0]),
        "n_sig_3sigma": n_3s,
        "n_sig_5sigma": n_5s,
        "top_lags": top_lags,
        "top_abs_r": [round(v, 10) for v in top_vals],
        "elapsed_s": round(elapsed, 2),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(str(OUT_JSON), "w") as f:
        json.dump(result, f, indent=2)
    log(f"JSON  -> {OUT_JSON}")

    # Plot ---------------------------------------------------------------
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"FFT Autocorrelation — Rule 30 center column ({N:,} bits)\n"
            f"Max |r|={top_vals[0]:.6f} @ lag {top_lags[0]:,}  |  "
            f"Candidate lag {PERIOD_CANDIDATE}: z={cand_z:.2f}sigma",
            fontsize=11, fontweight="bold")

        # 1: distribution of |r|
        ax = axes[0]
        ax.hist(abs_a, bins=300, density=True, color="steelblue", alpha=0.8)
        for mult, col, lab in [(3,"orange","3sigma"),(5,"red","5sigma")]:
            ax.axvline(mult*sigma, color=col, lw=1.5, ls="--", label=f"{lab}={mult*sigma:.5f}")
        ax.set_xlabel("|r| at each lag"); ax.set_ylabel("Density")
        ax.set_title(f"Distribution of |r| over {half:,} lags\n(Rayleigh expected under H₀)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # 2: top N lags scatter
        ax = axes[1]
        ax.scatter(top_lags, top_vals, s=6, color="tomato", alpha=0.7, label=f"Top {N_TOP}")
        cand_in_top = PERIOD_CANDIDATE in top_lags
        cy = top_vals[top_lags.index(PERIOD_CANDIDATE)] if cand_in_top else abs(a_norm[PERIOD_CANDIDATE])
        ax.scatter([PERIOD_CANDIDATE], [cy], s=120, color="gold", zorder=6,
                   marker="*", label=f"Lag {PERIOD_CANDIDATE}")
        for mult, col in [(5,"red"),(3,"orange")]:
            ax.axhline(mult*sigma, color=col, lw=1, ls="--", label=f"{mult}sigma")
        ax.set_xlabel("Lag"); ax.set_ylabel("|r|")
        ax.set_title(f"Top {N_TOP} lags  (gold★ = period candidate)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # 3: zoom +-5000 around candidate
        z0 = max(1, PERIOD_CANDIDATE - 5000)
        z1 = min(half, PERIOD_CANDIDATE + 5000)
        ax = axes[2]
        ax.plot(np.arange(z0, z1), np.abs(a_norm[z0:z1]), lw=0.5, color="steelblue")
        ax.axvline(PERIOD_CANDIDATE, color="gold", lw=2, ls="--",
                   label=f"Lag {PERIOD_CANDIDATE} (z={cand_z:.1f}sigma)")
        for mult, col in [(5,"red"),(3,"orange")]:
            ax.axhline(mult*sigma, color=col, lw=1, ls="--", label=f"{mult}sigma")
        ax.set_xlabel("Lag"); ax.set_ylabel("|r|")
        ax.set_title(f"Zoom lags {z0:,}–{z1:,}"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(PLOT_FILE), dpi=150, bbox_inches="tight"); plt.close()
        log(f"Plot  -> {PLOT_FILE}")
    except Exception as e:
        log(f"Plot skipped: {e}")

    # Experiment log -----------------------------------------------------
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(str(LOG_FILE), "w", encoding="utf-8") as f:
        f.write(f"""# Experiment Log — Full-Range FFT Autocorrelation

- Date: {datetime.date.today().isoformat()}
- Title: FFT Autocorrelation, all lags 1–{half:,}
- Goal: Definitively resolve the period-42795 candidate (windowed search z=4.66, match 52.33%)
- Data: center_col_46M.bin ({N:,} bits)
- Method: Convert bits -> +-1; rfft; |F|²; irfft; normalise by lag-0 variance.
  Under H₀: each lag r ~ N(0, 1/N), 1sigma={sigma:.8f}, 5sigma={5*sigma:.8f}.
- Result:
  - Period candidate lag {PERIOD_CANDIDATE}: r={cand_r:+.8f}, z={cand_z:.2f}sigma
  - Max |r| overall: {top_vals[0]:.8f} at lag {top_lags[0]:,}
  - Lags >3sigma: {n_3s:,} observed vs ~{exp_3s:,} expected
  - Lags >5sigma: {n_5s:,} observed vs ~{exp_5s:,} expected
- Interpretation:
  {"Lag " + str(PERIOD_CANDIDATE) + " IS significant (z>" + "5sigma). This may be a genuine structural feature — run period_search_extended.py on 200M steps to confirm." if cand_z > 5 else "Lag " + str(PERIOD_CANDIDATE) + " is NOT significant (z=" + f"{cand_z:.2f}sigma). The windowed period search result was a statistical artifact of the limited window. No periodic structure found across all 23M lags."}
- Elapsed: {elapsed:.0f}s
""")
    log(f"Log   -> {LOG_FILE}")


if __name__ == "__main__":
    main()
