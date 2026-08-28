"""Is the settled-word stream actually generic? Test it instead of assuming it.

The power analysis in 2026-08-19-period-doubling-criterion.md rests on ONE
modelling assumption: past d ~ 403 the settled words w_d behave like independent
uniform 16-bit words, so a consecutive-word collision is a ~2^-16 per-diagonal
event. Everything downstream -- "the old evidence had no power", "expected first
failure at d ~ 1.3x10^5", "reaching 10^6 clean is a ~2^-11 coincidence" --
inherits that assumption. It was asserted, never measured. This measures it.

Sample. Words are taken from the O(1) pattern map over the generic regime
403 < d < 53205 (the range between the last early zero word and the first real
collision, so no ambiguous step is ever taken). n ~ 5.3x10^4 words, validated
bit-exact against simulation wherever simulation exists (d < 12000).

Statistics, each against a matched uniform-random control (8 seeds):
  - per-bit balance          16 marginals, each ~1/2 under genericity
  - parity balance           the quantity Lemma B actually depends on
  - all-pairs collisions     the direct test of the 2^-16 rate, ~1.4x10^9 pairs
  - distinct words           birthday/coupon statistic
  - chi-square, 256 bins     high byte and low byte
  - lag-k collision rate     k = 1..64, tests serial structure

Also checks a structural fact that bounds what "generic" can mean: the state map
(u, v) -> (v, w) is INVERTIBLE wherever v != 0, since u[t] = w[t+1] XOR
(v[t] OR w[t]) recovers u from (v, w). So the orbit is purely periodic and
visits states without replacement -- negligible for n << 2^32, but it means the
word stream can never be truly i.i.d., only statistically indistinguishable.

Run:  python experiments/word_genericity.py --pretty
Exits non-zero only if the sample fails to validate against simulation, NOT on
a statistical verdict -- the verdict is reported, not gated.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagonal_recursion import pattern_map_step  # noqa: E402
from period_doubling import PERIOD, words_from_simulation  # noqa: E402

GENERIC_START = 403          # first odd-parity word; end of the structured regime
MASK = (1 << PERIOD) - 1
SPACE = 1 << PERIOD


def invertibility_check(samples: int = 200_000, seed: int = 0) -> dict:
    """u[t] = w[t+1] XOR (v[t] OR w[t]) should recover u from (v, w)."""
    rng = np.random.default_rng(seed)
    tested = bad = skipped = 0
    for _ in range(samples):
        u = int(rng.integers(0, SPACE))
        v = int(rng.integers(0, SPACE))
        if v == 0:
            skipped += 1
            continue
        w, flag = pattern_map_step(u, v, PERIOD)
        if flag != "ok":
            skipped += 1
            continue
        rec = 0
        for t in range(PERIOD):
            bit = ((w >> ((t + 1) % PERIOD)) & 1) ^ (((v >> t) & 1) | ((w >> t) & 1))
            if bit:
                rec |= 1 << t
        tested += 1
        if rec != u:
            bad += 1
    return {"tested": tested, "skipped": skipped, "failures": bad,
            "invertible": bad == 0}


def words_from_map(seed_words, start_d: int, limit_d: int) -> tuple[np.ndarray, dict]:
    """Iterate the map from simulation-seeded words, stopping at the first zero."""
    u, v = seed_words
    out = []
    d = start_d
    stopped = None
    while d < limit_d:
        if v == 0:
            stopped = {"reason": "zero_word", "d": d - 1}
            break
        if u == v:
            stopped = {"reason": "collision", "d_lo": d - 2, "d_hi": d - 1}
            break
        w, flag = pattern_map_step(u, v, PERIOD)
        if flag != "ok":
            stopped = {"reason": flag, "d": d}
            break
        out.append(w)
        u, v = v, w
        d += 1
    return np.array(out, dtype=np.int64), {"stopped": stopped, "first_d": start_d,
                                           "count": len(out)}


def stats(words: np.ndarray) -> dict:
    n = int(words.size)
    bits = ((words[:, None] >> np.arange(PERIOD)[None, :]) & 1)
    bit_ones = bits.sum(axis=0).tolist()
    parity = int(bits.sum(axis=1).astype(np.int64).__and__(1).sum())

    counts = np.bincount(words, minlength=SPACE)
    distinct = int((counts > 0).sum())
    # unordered equal pairs
    coll = int((counts * (counts - 1) // 2).sum())

    hi = np.bincount(words >> 8, minlength=256)
    lo = np.bincount(words & 0xFF, minlength=256)
    exp_bin = n / 256.0
    chi_hi = float(((hi - exp_bin) ** 2 / exp_bin).sum())
    chi_lo = float(((lo - exp_bin) ** 2 / exp_bin).sum())

    lag = {}
    for k in range(1, 65):
        lag[k] = int((words[:-k] == words[k:]).sum())

    return {"n": n, "bit_ones": bit_ones, "parity_odd": parity,
            "distinct": distinct, "pair_collisions": coll,
            "chi2_high_byte": chi_hi, "chi2_low_byte": chi_lo,
            "lag_collisions": lag}


def expectations(n: int) -> dict:
    return {
        "bit_ones": n / 2.0,
        "bit_ones_sd": math.sqrt(n) / 2.0,
        "parity_odd": n / 2.0,
        "parity_sd": math.sqrt(n) / 2.0,
        "distinct": SPACE * (1.0 - (1.0 - 1.0 / SPACE) ** n),
        "pair_collisions": n * (n - 1) / 2.0 / SPACE,
        "pair_collisions_sd": math.sqrt(n * (n - 1) / 2.0 / SPACE),
        "chi2_df": 255,
        "chi2_sd": math.sqrt(2 * 255),
        "lag": n / SPACE,
    }


def zscores(s: dict, e: dict) -> dict:
    n = s["n"]
    worst_bit = max(range(PERIOD),
                    key=lambda i: abs(s["bit_ones"][i] - e["bit_ones"]))
    lag_tot = sum(s["lag_collisions"].values())
    lag_exp = sum(e["lag"] for _ in s["lag_collisions"])
    return {
        "bit_z": [round((c - e["bit_ones"]) / e["bit_ones_sd"], 3)
                  for c in s["bit_ones"]],
        "worst_bit": worst_bit,
        "worst_bit_z": round((s["bit_ones"][worst_bit] - e["bit_ones"])
                             / e["bit_ones_sd"], 3),
        "parity_z": round((s["parity_odd"] - e["parity_odd"]) / e["parity_sd"], 3),
        "collisions_z": round((s["pair_collisions"] - e["pair_collisions"])
                              / e["pair_collisions_sd"], 3),
        "distinct_delta": round(s["distinct"] - e["distinct"], 1),
        "chi2_high_z": round((s["chi2_high_byte"] - 255) / e["chi2_sd"], 3),
        "chi2_low_z": round((s["chi2_low_byte"] - 255) / e["chi2_sd"], 3),
        "lag_total": lag_tot,
        "lag_total_expected": round(lag_exp, 2),
        "observed_collision_rate": s["pair_collisions"] / (n * (n - 1) / 2.0),
        "modelled_collision_rate": 1.0 / SPACE,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=26000)
    p.add_argument("--diagonals", type=int, default=12000)
    p.add_argument("--tail", type=int, default=4096)
    p.add_argument("--limit-d", type=int, default=53205)
    p.add_argument("--controls", type=int, default=200)
    p.add_argument("--pretty", action="store_true")
    p.add_argument("--out")
    a = p.parse_args(argv)

    t0 = time.time()
    sim = words_from_simulation(a.steps, a.diagonals, a.tail)

    start = GENERIC_START
    seed_words = (sim[start - 2], sim[start - 1])
    words, meta = words_from_map(seed_words, start, a.limit_d)

    # Validate the map-generated sample against simulation where both exist.
    mism = 0
    for i, w in enumerate(words):
        d = start + i
        if d < len(sim) and sim[d] is not None and int(w) != sim[d]:
            mism += 1
    overlap = sum(1 for i in range(len(words))
                  if start + i < len(sim) and sim[start + i] is not None)

    s = stats(words)
    e = expectations(s["n"])
    z = zscores(s, e)

    rng = np.random.default_rng(30)
    controls = []
    for _ in range(a.controls):
        cw = rng.integers(0, SPACE, size=s["n"], dtype=np.int64)
        cs = stats(cw)
        controls.append(zscores(cs, expectations(cs["n"])))

    verdict_keys = ["worst_bit_z", "parity_z", "collisions_z",
                    "chi2_high_z", "chi2_low_z"]

    def spread(key):
        v = [c[key] for c in controls]
        return {"min": min(v), "max": max(v)}

    def empirical_p(key):
        """Two-sided p: fraction of controls at least as extreme as observed.

        Uses the control distribution itself rather than a normal
        approximation, so it stays honest for chi2 and the birthday
        statistic. (k+1)/(m+1) is the standard unbiased small-sample form.
        """
        v = [abs(c[key]) for c in controls]
        k = sum(1 for x in v if x >= abs(z[key]))
        return round((k + 1) / (len(v) + 1), 4)

    control_band = {k: spread(k) for k in verdict_keys}
    p_values = {k: empirical_p(k) for k in verdict_keys}
    outside = [k for k in verdict_keys
               if not (control_band[k]["min"] <= z[k] <= control_band[k]["max"])]
    # Bonferroni over the 5 statistics actually examined.
    significant = [k for k in verdict_keys if p_values[k] < 0.05 / len(verdict_keys)]
    extreme = [k for k in verdict_keys if abs(z[k]) > 3.0]

    out = {
        "artifact_type": "rule30.word_genericity",
        "artifact_version": 1,
        "params": {"generic_start": GENERIC_START, "limit_d": a.limit_d,
                   "steps": a.steps, "diagonals": a.diagonals,
                   "controls": a.controls, "period": PERIOD},
        "sample": {**meta, "validated_against_simulation": overlap,
                   "mismatches": mism},
        "invertibility": invertibility_check(),
        "observed": s,
        "expected": e,
        "z": z,
        "control_band": control_band,
        "empirical_p": p_values,
        "outside_control_band": outside,
        "significant_after_bonferroni": significant,
        "abs_z_over_3": extreme,
        "elapsed_s": round(time.time() - t0, 3),
        "ok": mism == 0,
    }

    if a.pretty:
        w = sys.stderr
        print(f"sample     : n={s['n']} over {start} < d < {a.limit_d}, "
              f"stopped={meta['stopped']}", file=w)
        print(f"validated  : {overlap} words vs simulation, {mism} mismatches",
              file=w)
        inv = out["invertibility"]
        print(f"invertible : {inv['invertible']} "
              f"({inv['tested']} random (u,v), {inv['failures']} failures)", file=w)
        print("", file=w)
        print(f"{'statistic':<22}{'observed':>14}{'expected':>14}{'z':>9}"
              f"{'control band':>20}", file=w)
        rows = [
            ("worst bit balance", s["bit_ones"][z["worst_bit"]], e["bit_ones"],
             z["worst_bit_z"], "worst_bit_z"),
            ("parity odd", s["parity_odd"], e["parity_odd"],
             z["parity_z"], "parity_z"),
            ("all-pairs collisions", s["pair_collisions"], e["pair_collisions"],
             z["collisions_z"], "collisions_z"),
            ("chi2 high byte", s["chi2_high_byte"], 255.0,
             z["chi2_high_z"], "chi2_high_z"),
            ("chi2 low byte", s["chi2_low_byte"], 255.0,
             z["chi2_low_z"], "chi2_low_z"),
        ]
        for label, obs, exp, zz, key in rows:
            band = control_band[key]
            band_str = "[%+.2f, %+.2f]" % (band["min"], band["max"])
            print(f"{label:<22}{obs:>14.1f}{exp:>14.1f}{zz:>9.2f}{band_str:>20}",
                  file=w)
        print("", file=w)
        print(f"distinct words   : {s['distinct']} vs {e['distinct']:.0f} expected "
              f"({z['distinct_delta']:+.0f})", file=w)
        print(f"lag 1..64 collis.: {z['lag_total']} vs "
              f"{z['lag_total_expected']:.1f} expected", file=w)
        print(f"collision rate   : {z['observed_collision_rate']:.3e} vs "
              f"modelled 2^-16 = {z['modelled_collision_rate']:.3e}", file=w)
        print("", file=w)
        print("empirical two-sided p (vs %d controls):" % a.controls, file=w)
        for k in verdict_keys:
            print(f"  {k:<16} z={z[k]:+7.2f}  p={p_values[k]:.4f}", file=w)
        print("", file=w)
        print(f"significant @ Bonferroni 0.05/5 : {significant or 'NONE'}", file=w)
        print(f"|z| > 3                         : {extreme or 'none'}", file=w)

    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str),
                               encoding="utf-8")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
