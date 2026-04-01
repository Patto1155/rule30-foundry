#!/usr/bin/env python
"""Experiment O — Column Mutual Information and Transfer Entropy.

First 2D structural analysis of the Rule 30 spacetime diagram.
All experiments A–N only analysed the 1D center column projection.

Key outside-the-box finding to look for:
  TE(right+d -> center) != TE(left-d -> center)

Rule 30 is NOT left-right symmetric:
  c'(i) = c(i-1) XOR (c(i) OR c(i+1))
  Left neighbour enters via XOR directly.
  Right neighbour enters via OR-then-XOR.
These are different operations -> information should flow asymmetrically.
If TE asymmetry is measured, this is the first empirical quantification
of Rule 30's directional information geometry.

Important caveat:
  A large one-step TE at d=1 can be a direct consequence of the local update rule.
  This script therefore computes surrogate baselines by time-shifting the source
  column so the reported excess over baseline is explicit.
"""
import sys, json, time, datetime
import numpy as np
from pathlib import Path

try:
    import cupy as cp
    GPU = True
except ImportError:
    cp  = None
    GPU = False

# ── Config ────────────────────────────────────────────────────────────────────
TEST        = "--test" in sys.argv
N_SIM_STEPS = 50_000   if TEST else 500_000
STRIP_WIDTH = 64       if TEST else 512
CHUNK_SIZE  = 5_000    if TEST else 10_000
BURN_IN     = STRIP_WIDTH + 500    # skip startup phase (pattern hasn't filled strip)
DISTANCES   = [1,2,4,8,16,32,64]  if TEST else [1,2,4,8,16,32,64,128,256,512]
SURROGATE_SHIFT = 257 if TEST else 9973

OUT_JSON  = Path(r"D:\APATPROJECTS\rule30-research\data\column_mi.json")
LOG_FILE  = Path(r"D:\APATPROJECTS\rule30-research\docs\experiment-logs\O_column_mi.md")
PLOT_FILE = Path(r"D:\APATPROJECTS\rule30-research\docs\plots\column_mi.png")
PROG_LOG  = Path(r"D:\APATPROJECTS\rule30-research\data\O_progress.log")

# ── GPU kernel ────────────────────────────────────────────────────────────────
KERNEL_SRC = r"""
extern "C" __global__
void rule30_step(
    const unsigned long long* tape,
    unsigned long long*       out,
    int                       n_words
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_words) return;
    unsigned long long c    = tape[idx];
    unsigned long long prev = (idx > 0)          ? tape[idx - 1] : 0ULL;
    unsigned long long next = (idx < n_words - 1) ? tape[idx + 1] : 0ULL;
    unsigned long long L    = (c << 1) | (prev >> 63);
    unsigned long long R    = (c >> 1) | (next << 63);
    out[idx] = L ^ (c | R);
}
"""


def log(msg):
    ts   = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(PROG_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ── Information-theory functions ──────────────────────────────────────────────

def mi_from_counts(counts4: np.ndarray) -> float:
    """MI(X;Y) from 4-bin joint counts [p00, p01, p10, p11] (X*2+Y indexing)."""
    n = counts4.sum()
    if n == 0:
        return 0.0
    p  = counts4.astype(np.float64) / n
    px = np.array([p[0] + p[1], p[2] + p[3]])   # marginal X
    py = np.array([p[0] + p[2], p[1] + p[3]])   # marginal Y
    def h(ps):
        q = ps[ps > 1e-15]
        return float(-np.sum(q * np.log2(q)))
    return max(0.0, h(px) + h(py) - h(p))


def te_from_counts(counts8: np.ndarray) -> float:
    """TE(X->Y) from 8-bin counts indexed as yt1 + 2*yt + 4*xt.

    TE = H(Y_{t+1} | Y_t) - H(Y_{t+1} | Y_t, X_t)
       = I(Y_{t+1} ; X_t | Y_t)
    """
    n = counts8.sum()
    if n == 0:
        return 0.0
    p8 = counts8.astype(np.float64) / n
    # Reshape to [xt=0/1, yt=0/1, yt1=0/1] — bit positions: xt=bit2, yt=bit1, yt1=bit0
    p  = p8.reshape(2, 2, 2)   # [xt, yt, yt1]

    def h(arr):
        f = arr.flatten(); f = f[f > 1e-15]
        return float(-np.sum(f * np.log2(f)))

    p_yt_yt1  = p.sum(axis=0)          # [yt, yt1] — marginalise over xt
    p_yt      = p_yt_yt1.sum(axis=1)   # [yt]
    H_yt1_yt  = h(p_yt_yt1)
    H_yt      = h(p_yt)
    H_yt1_given_yt = H_yt1_yt - H_yt  # = H(Y_{t+1} | Y_t)

    p_xt_yt   = p.sum(axis=2)          # [xt, yt] — marginalise over yt1
    H_full    = h(p)                   # H(Y_{t+1}, Y_t, X_t)
    H_xt_yt   = h(p_xt_yt)            # H(X_t, Y_t)
    H_yt1_given_xt_yt = H_full - H_xt_yt  # = H(Y_{t+1} | X_t, Y_t)

    return max(0.0, H_yt1_given_yt - H_yt1_given_xt_yt)


def naive_center_bits(n_steps: int) -> list[int]:
    n_cells = 2 * n_steps + 1
    center = n_cells // 2
    row = np.zeros(n_cells, dtype=np.uint8)
    row[center] = 1
    out = []
    for _ in range(n_steps):
        out.append(int(row[center]))
        nxt = np.zeros_like(row)
        for i in range(n_cells):
            left = row[i - 1] if i > 0 else 0
            center_bit = row[i]
            right = row[i + 1] if i + 1 < n_cells else 0
            nxt[i] = left ^ (center_bit | right)
        row = nxt
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    PROG_LOG.parent.mkdir(parents=True, exist_ok=True)
    open(PROG_LOG, "w").close()
    log("=== Experiment O — Column MI and Transfer Entropy ===")
    log(f"TEST={TEST}  GPU={GPU}  N_SIM_STEPS={N_SIM_STEPS:,}  "
        f"STRIP_WIDTH={STRIP_WIDTH}  BURN_IN={BURN_IN}")
    t0 = time.perf_counter()

    # ── Tape setup ────────────────────────────────────────────────────────
    N_CELLS_MIN = 2 * N_SIM_STEPS + 1
    n_words     = (N_CELLS_MIN + 63) // 64
    n_cells     = n_words * 64
    center_word = n_words // 2
    center_bit  = 32
    center_cell = center_word * 64 + center_bit

    strip_left  = center_cell - STRIP_WIDTH
    strip_right = center_cell + STRIP_WIDTH
    w_start     = strip_left  // 64
    w_end       = strip_right // 64
    n_sw        = w_end - w_start + 1    # strip words
    n_cols      = 2 * STRIP_WIDTH + 1    # number of strip columns
    center_idx  = STRIP_WIDTH            # index of center in strip

    # Precompute per-column: which local strip word and which bit
    col_positions   = np.arange(strip_left, strip_right + 1)
    col_word_local  = (col_positions // 64 - w_start).astype(np.intp)
    col_bit_shift   = (col_positions % 64).astype(np.uint64)

    log(f"Tape:   {n_cells:,} cells, {n_words:,} words, VRAM~{n_words*8*2/1024**2:.0f}MB")
    log(f"Strip:  cells {strip_left:,}–{strip_right:,}  ({n_cols} cols, {n_sw} words)")
    log(f"Center: word {center_word}, bit {center_bit}, cell {center_cell:,}")

    if not GPU:
        log("WARNING: No CuPy — falling back to slow CPU simulation")

    # ── Accumulators ─────────────────────────────────────────────────────
    n_d = len(DISTANCES)
    mi_right_c  = np.zeros((n_d, 4), dtype=np.int64)  # MI(center, right+d)
    mi_left_c   = np.zeros((n_d, 4), dtype=np.int64)  # MI(center, left-d)
    mi_right_s  = np.zeros((n_d, 4), dtype=np.int64)  # surrogate MI
    mi_left_s   = np.zeros((n_d, 4), dtype=np.int64)
    te_right_c  = np.zeros((n_d, 8), dtype=np.int64)  # TE(right+d -> center)
    te_left_c   = np.zeros((n_d, 8), dtype=np.int64)  # TE(left-d  -> center)
    te_right_s  = np.zeros((n_d, 8), dtype=np.int64)  # surrogate TE
    te_left_s   = np.zeros((n_d, 8), dtype=np.int64)
    n_samples   = 0

    # ── GPU simulation ────────────────────────────────────────────────────
    if GPU:
        kernel  = cp.RawKernel(KERNEL_SRC, "rule30_step")
        tape_a  = cp.zeros(n_words, dtype=cp.uint64)
        tape_b  = cp.zeros(n_words, dtype=cp.uint64)
        tape_a[center_word] = cp.uint64(1 << center_bit)
        w_idx_gpu = cp.asarray(np.arange(w_start, w_end + 1, dtype=np.int32))
        strip_buf = cp.zeros((CHUNK_SIZE, n_sw), dtype=cp.uint64)
        block = 256
        grid  = (n_words + block - 1) // block
    else:
        tape_a = np.zeros(n_words, dtype=np.uint64)
        tape_b = np.zeros(n_words, dtype=np.uint64)
        tape_a[center_word] = np.uint64(1 << center_bit)
        strip_buf_cpu = np.zeros((CHUNK_SIZE, n_sw), dtype=np.uint64)

    # Verify first 20 center bits after quick pre-run
    verify_len = 96
    EXPECTED = naive_center_bits(verify_len)
    verify_buf  = []

    n_chunks = (N_SIM_STEPS + CHUNK_SIZE - 1) // CHUNK_SIZE
    log(f"\nRunning {N_SIM_STEPS:,} steps in {n_chunks} chunks of {CHUNK_SIZE:,} ...")
    log(f"Burn-in: first {BURN_IN} steps excluded from MI/TE accumulation")

    t_sim_start = time.perf_counter()

    for chunk_idx in range(n_chunks):
        step_start = chunk_idx * CHUNK_SIZE
        step_end   = min(step_start + CHUNK_SIZE, N_SIM_STEPS)
        chunk_len  = step_end - step_start

        # Run chunk_len simulation steps, gather strip words
        if GPU:
            cur, nxt = tape_a, tape_b
            for s in range(chunk_len):
                strip_buf[s] = cur[w_idx_gpu]          # record before advancing
                kernel((grid,), (block,), (cur, nxt, np.int32(n_words)))
                cur, nxt = nxt, cur
            tape_a, tape_b = cur, nxt
            strip_data = cp.asnumpy(strip_buf[:chunk_len])   # [chunk_len, n_sw] uint64
        else:
            # CPU fallback (slow)
            def cpu_step(t):
                L = (t << np.uint64(1))
                L[1:] |= (t[:-1] >> np.uint64(63))
                R = (t >> np.uint64(1))
                R[:-1] |= (t[1:] << np.uint64(63))
                return L ^ (t | R)
            cur = tape_a
            for s in range(chunk_len):
                strip_buf_cpu[s] = cur[w_start:w_end + 1]   # record before advancing
                cur = cpu_step(cur)
            tape_a = cur
            strip_data = strip_buf_cpu[:chunk_len].copy()

        # ── Bit extraction ────────────────────────────────────────────────
        # bits: [chunk_len, n_cols] uint8
        bits = ((strip_data[:, col_word_local] >> col_bit_shift) & 1).astype(np.uint8)

        # Verify first 20 center bits (steps 0-19)
        if len(verify_buf) < verify_len:
            need = verify_len - len(verify_buf)
            verify_buf.extend(bits[:need, center_idx].tolist())

        # Skip burn-in steps
        global_steps = np.arange(step_start, step_end)
        active_mask  = global_steps >= BURN_IN
        if not active_mask.any():
            continue
        bits_active = bits[active_mask]

        # ── Accumulate MI and TE ──────────────────────────────────────────
        c0 = bits_active[:, center_idx]
        for i, d in enumerate(DISTANCES):
            if center_idx + d >= n_cols or center_idx - d < 0:
                continue
            cr = bits_active[:, center_idx + d]   # right column
            cl = bits_active[:, center_idx - d]   # left column
            shift = SURROGATE_SHIFT % len(bits_active)
            if shift == 0 and len(bits_active) > 1:
                shift = 1
            cr_s = np.roll(cr, shift)
            cl_s = np.roll(cl, shift)

            # MI(center, right+d): joint index = c0*2 + cr
            j_r = c0.astype(np.int32) * 2 + cr
            mi_right_c[i] += np.bincount(j_r, minlength=4)
            j_l = c0.astype(np.int32) * 2 + cl
            mi_left_c[i]  += np.bincount(j_l, minlength=4)
            j_r_s = c0.astype(np.int32) * 2 + cr_s
            j_l_s = c0.astype(np.int32) * 2 + cl_s
            mi_right_s[i] += np.bincount(j_r_s, minlength=4)
            mi_left_s[i]  += np.bincount(j_l_s, minlength=4)

            # TE: need consecutive pairs (t, t+1)
            if len(bits_active) > 1:
                yt1  = bits_active[1:, center_idx]
                yt   = bits_active[:-1, center_idx]
                xtr  = bits_active[:-1, center_idx + d]
                xtl  = bits_active[:-1, center_idx - d]
                xtr_s = np.roll(xtr, shift)
                xtl_s = np.roll(xtl, shift)
                te_r = (yt1.astype(np.int32) + 2 * yt + 4 * xtr).astype(np.intp)
                te_l = (yt1.astype(np.int32) + 2 * yt + 4 * xtl).astype(np.intp)
                te_r_s = (yt1.astype(np.int32) + 2 * yt + 4 * xtr_s).astype(np.intp)
                te_l_s = (yt1.astype(np.int32) + 2 * yt + 4 * xtl_s).astype(np.intp)
                te_right_c[i] += np.bincount(te_r, minlength=8)
                te_left_c[i]  += np.bincount(te_l, minlength=8)
                te_right_s[i] += np.bincount(te_r_s, minlength=8)
                te_left_s[i]  += np.bincount(te_l_s, minlength=8)

        n_samples += len(bits_active)

        # Progress log every 10 chunks
        if chunk_idx % 10 == 0 or chunk_idx == n_chunks - 1:
            elapsed_sim = time.perf_counter() - t_sim_start
            sps = step_end / elapsed_sim if elapsed_sim > 0 else 0
            pct = 100.0 * step_end / N_SIM_STEPS
            eta = (N_SIM_STEPS - step_end) / sps if sps > 0 else 0
            log(f"  Chunk {chunk_idx+1}/{n_chunks}  {pct:.1f}%  "
                f"{sps:,.0f} steps/s  ETA {datetime.timedelta(seconds=int(eta))}  "
                f"n_samples={n_samples:,}")

    # Verify
    actual_verify = verify_buf[:verify_len]
    verified  = actual_verify == EXPECTED
    log(f"\nVerification: first {verify_len} center bits match naive Rule 30 = {verified}")
    log(f"  {'PASS PASS' if verified else 'FAIL FAIL — strip extraction may be wrong!'}")

    # ── Compute MI and TE ─────────────────────────────────────────────────
    log("\nComputing MI and TE ...")
    mi_right, mi_left, te_right, te_left, asym = [], [], [], [], []
    mi_right_base, mi_left_base = [], []
    te_right_base, te_left_base = [], []
    te_right_excess, te_left_excess = [], []
    for i, d in enumerate(DISTANCES):
        mir = mi_from_counts(mi_right_c[i])
        mil = mi_from_counts(mi_left_c[i])
        mir_s = mi_from_counts(mi_right_s[i])
        mil_s = mi_from_counts(mi_left_s[i])
        ter = te_from_counts(te_right_c[i])
        tel = te_from_counts(te_left_c[i])
        ter_s = te_from_counts(te_right_s[i])
        tel_s = te_from_counts(te_left_s[i])
        mi_right.append(round(mir, 8))
        mi_left.append(round(mil, 8))
        mi_right_base.append(round(mir_s, 8))
        mi_left_base.append(round(mil_s, 8))
        te_right.append(round(ter, 8))
        te_left.append(round(tel, 8))
        te_right_base.append(round(ter_s, 8))
        te_left_base.append(round(tel_s, 8))
        te_right_excess.append(round(ter - ter_s, 8))
        te_left_excess.append(round(tel - tel_s, 8))
        asym.append(round(ter - tel, 8))
        log(
            f"  d={d:>4}: "
            f"MI_right={mir:.6f} (base {mir_s:.6f})  "
            f"MI_left={mil:.6f} (base {mil_s:.6f})  "
            f"TE_right={ter:.6f} (base {ter_s:.6f})  "
            f"TE_left={tel:.6f} (base {tel_s:.6f})  "
            f"asym={ter-tel:+.6f}"
        )

    max_asym    = max(abs(a) for a in asym)
    asym_nonzero = sum(1 for a in asym if abs(a) > 1e-6)
    log(f"\n  Max |TE asymmetry|: {max_asym:.8f}")
    log(f"  Distances with nonzero asymmetry: {asym_nonzero}/{len(DISTANCES)}")

    elapsed = time.perf_counter() - t0
    log(f"\nTotal time: {elapsed:.1f}s")

    # ── Save JSON ─────────────────────────────────────────────────────────
    result = {
        "n_sim_steps":   N_SIM_STEPS,
        "strip_width":   STRIP_WIDTH,
        "burn_in":       BURN_IN,
        "n_samples":     n_samples,
        "distances":     DISTANCES,
        "mi_right":      mi_right,
        "mi_left":       mi_left,
        "mi_right_surrogate": mi_right_base,
        "mi_left_surrogate": mi_left_base,
        "te_right":      te_right,
        "te_left":       te_left,
        "te_right_surrogate": te_right_base,
        "te_left_surrogate": te_left_base,
        "te_right_excess": te_right_excess,
        "te_left_excess": te_left_excess,
        "te_asymmetry":  asym,
        "max_te_asymmetry": round(max_asym, 10),
        "surrogate_shift": SURROGATE_SHIFT,
        "verification_passed": verified,
        "elapsed_s":     round(elapsed, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(str(OUT_JSON), "w") as f:
        json.dump(result, f, indent=2)
    log(f"JSON  -> {OUT_JSON}")

    # ── Plot ──────────────────────────────────────────────────────────────
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"Experiment O — Column MI and Transfer Entropy\n"
            f"Rule 30, {N_SIM_STEPS:,} steps, strip +-{STRIP_WIDTH}, "
            f"{n_samples:,} samples post burn-in",
            fontsize=11, fontweight="bold")

        ds = np.array(DISTANCES)

        # 1: MI decay
        ax = axes[0]
        ax.semilogx(ds, mi_right, "b-o", ms=6, lw=1.5, label="MI(center, right+d)")
        ax.semilogx(ds, mi_left,  "r--s", ms=6, lw=1.5, label="MI(center, left-d)")
        ax.semilogx(ds, mi_right_base, "b:", lw=1.2, label="MI right surrogate")
        ax.semilogx(ds, mi_left_base,  "r:", lw=1.2, label="MI left surrogate")
        ax.set_xlabel("Column separation d"); ax.set_ylabel("MI (bits)")
        ax.set_title("Mutual information vs distance\nwith time-shifted surrogate baselines")
        ax.legend(); ax.grid(True, alpha=0.3)

        # 2: Transfer entropy
        ax = axes[1]
        ax.semilogx(ds, te_right, "b-o",  ms=6, lw=1.5, label="TE(right+d -> center)")
        ax.semilogx(ds, te_left,  "r--s", ms=6, lw=1.5, label="TE(left-d -> center)")
        ax.semilogx(ds, te_right_base, "b:", lw=1.2, label="TE right surrogate")
        ax.semilogx(ds, te_left_base,  "r:", lw=1.2, label="TE left surrogate")
        ax.set_xlabel("Column separation d"); ax.set_ylabel("TE (bits)")
        ax.set_title("Transfer entropy vs distance\nraw vs surrogate baseline")
        ax.legend(); ax.grid(True, alpha=0.3)

        # 3: TE asymmetry
        ax = axes[2]
        colors = ["tomato" if a > 0 else "steelblue" for a in asym]
        ax.bar(range(len(DISTANCES)), asym, color=colors)
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(range(len(DISTANCES)))
        ax.set_xticklabels([str(d) for d in DISTANCES], rotation=45)
        ax.set_xlabel("Distance d"); ax.set_ylabel("TE(right->center) - TE(left->center)")
        ax.set_title(f"TE asymmetry (Rule 30 left-right asymmetry)\n"
                     f"max|asym|={max_asym:.6f}")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(PLOT_FILE), dpi=150, bbox_inches="tight"); plt.close()
        log(f"Plot  -> {PLOT_FILE}")
    except Exception as e:
        log(f"Plot skipped: {e}")

    # ── Experiment log ────────────────────────────────────────────────────
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(str(LOG_FILE), "w", encoding="utf-8") as f:
        f.write(f"""# Experiment Log — Column Mutual Information and Transfer Entropy

- Date: {datetime.date.today().isoformat()}
- Title: Column MI and Transfer Entropy — first 2D analysis
- Goal: Detect spatial structure and directional information flow invisible to 1D analysis
- Setup: Fresh {N_SIM_STEPS:,}-step simulation, strip center+-{STRIP_WIDTH} ({n_cols} cols), burn-in={BURN_IN}, GPU={GPU}
- Method:
  - MI(center, right+d) and MI(center, left-d) for d in {DISTANCES}
  - TE(right+d->center) and TE(left-d->center): H(Y_{{t+1}}|Y_t) - H(Y_{{t+1}}|Y_t,X_t)
  - Surrogate control: time-shift the source column by {SURROGATE_SHIFT} samples to estimate a same-marginal noise floor
- Results:
  - n_samples (post burn-in): {n_samples:,}
  - Verification: {'PASS' if verified else 'FAIL'}
  - Max |TE asymmetry|: {max_asym:.8f} bits
  - Distances with nonzero asymmetry: {asym_nonzero}/{len(DISTANCES)}
  - MI at d=1: right={mi_right[0]:.6f} (surrogate {mi_right_base[0]:.6f}), left={mi_left[0]:.6f} (surrogate {mi_left_base[0]:.6f})
  - TE at d=1: right={te_right[0]:.6f} (surrogate {te_right_base[0]:.6f}), left={te_left[0]:.6f} (surrogate {te_left_base[0]:.6f}), asym={asym[0]:+.6f}
- Interpretation:
  {"The dominant d=1 TE signal survives the surrogate control and is therefore not just a finite-sample artifact." if max(te_left_excess[0], te_right_excess[0]) > 0.1 else "The d=1 TE signal does not clearly exceed the surrogate control."}
  {"Beyond d=1, MI and TE mostly sit near the surrogate baseline, so this experiment does not show strong long-range cross-column dependence at the tested distances." if max([abs(te_right_excess[i]) for i in range(1, len(te_right_excess))] + [abs(te_left_excess[i]) for i in range(1, len(te_left_excess))]) < 0.01 else "Some nontrivial above-surrogate dependence persists beyond d=1 and deserves follow-up."}
  A large one-step TE at d=1 is still partly a local-rule fact, not automatically a deep global-structure result.
- Elapsed: {elapsed:.0f}s
""")
    log(f"Log   -> {LOG_FILE}")


if __name__ == "__main__":
    main()
