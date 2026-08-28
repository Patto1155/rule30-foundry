#!/usr/bin/env python
"""ca_lab — a thin CLI over the verified CA + coarse-grain machinery.

Purpose: drive brute-force CA exploration in ONE command that returns JSON, so an
agent (or a human) can sweep rules / shears / scales without writing a script each
time. Everything underneath is the verified, GPU-accelerated stack:

  * fields   -> experiments/eca_sim.simulate_spacetime_rule (arbitrary Wolfram rule,
                byte-verified vs naive; fused GPU kernel)
  * closure  -> experiments/coarse_grain_fast.enumerate_b2_fast (GPU, 61x; exact
                match to the reference enumerate_b2)

Subcommands (all print JSON to stdout; add --pretty for a human table on stderr):

  sweep    coarse-grain closure across rules x shears (the main exploration tool)
  closure  single (rule, shear) closure
  sim      simulate a rule, report field statistics (density, col/row entropy)

Examples
--------
  python ca_lab.py sweep --rules 30,45,90,110 --shears 0,0.25,1 --steps 1200 --null
  python ca_lab.py closure --rule 30 --shear 0 --steps 1600
  python ca_lab.py sim --rule 110 --steps 400 --width 400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))
from eca_sim import simulate_spacetime_rule, GPU_AVAILABLE  # noqa: E402
from coarse_grain_fast import enumerate_b2_fast  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _rule_field(rule: int, n_steps: int, width: int, seed: int) -> np.ndarray:
    """Random-IC interior window of a rule's spacetime (open boundary trimmed)."""
    rng = np.random.default_rng(seed)
    margin = n_steps + 32
    n_cells = width + 2 * margin
    row = rng.integers(0, 2, size=n_cells, dtype=np.uint8)
    st = simulate_spacetime_rule(row, n_steps, rule, gpu=GPU_AVAILABLE)
    return st[:, margin:margin + width].copy()


def _parse_floats(s: str):
    return [float(x) for x in s.split(",") if x != ""]


def _parse_ints(s: str):
    return [int(x) for x in s.split(",") if x != ""]


def _best_over_shears(field, shears, r, h_min):
    best = {"excess": -1.0, "closure": 0.0, "shear": None}
    per_shear = {}
    for sigma in shears:
        g = enumerate_b2_fast(field, sigma, r, h_min, gpu=GPU_AVAILABLE)
        per_shear[str(sigma)] = {"excess": round(g["excess"], 5),
                                 "closure": round(g["closure"], 5)}
        if g["excess"] > best["excess"]:
            best = {"excess": round(g["excess"], 5), "closure": round(g["closure"], 5),
                    "shear": sigma, "pi": g.get("pi")}
    return best, per_shear


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #

def cmd_sweep(args) -> dict:
    rules = _parse_ints(args.rules)
    shears = _parse_floats(args.shears)
    t0 = time.perf_counter()
    results = {}
    for rule in rules:
        field = _rule_field(rule, args.steps, args.width, args.seed)
        best, per_shear = _best_over_shears(field, shears, args.r, args.hmin)
        results[str(rule)] = {"density": round(float(field.mean()), 5),
                              "best": best, "per_shear": per_shear}
    if args.null:
        rng = np.random.default_rng(args.seed + 992)
        null = rng.integers(0, 2, size=(args.steps, args.width), dtype=np.uint8)
        nb, nps = _best_over_shears(null, shears, args.r, args.hmin)
        results["iid"] = {"best": nb, "per_shear": nps}
    return {"cmd": "sweep",
            "params": {"rules": rules, "shears": shears, "b": 2, "r": args.r,
                       "hmin": args.hmin, "steps": args.steps, "width": args.width,
                       "seed": args.seed, "gpu": GPU_AVAILABLE},
            "results": results, "elapsed_s": round(time.perf_counter() - t0, 2)}


def cmd_closure(args) -> dict:
    t0 = time.perf_counter()
    field = _rule_field(args.rule, args.steps, args.width, args.seed)
    g = enumerate_b2_fast(field, args.shear, args.r, args.hmin, gpu=GPU_AVAILABLE)
    return {"cmd": "closure",
            "params": {"rule": args.rule, "shear": args.shear, "b": 2, "r": args.r,
                       "hmin": args.hmin, "steps": args.steps, "width": args.width,
                       "seed": args.seed, "gpu": GPU_AVAILABLE},
            "density": round(float(field.mean()), 5),
            "excess": round(g["excess"], 5), "closure": round(g["closure"], 5),
            "baseline": round(g.get("baseline", float("nan")), 5),
            "pi": g.get("pi"), "elapsed_s": round(time.perf_counter() - t0, 2)}


def _to_np(a):
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)


def cmd_search(args) -> dict:
    from coarse_grain_bk import prep_blocks, search_projection, closure_batch
    t0 = time.perf_counter()
    field = _rule_field(args.rule, args.steps, args.width, args.seed)
    prep = prep_blocks(field, args.b, args.shear, args.r, gpu=GPU_AVAILABLE,
                       m_max=args.msearch, seed=0)
    res = search_projection(prep, budget=args.budget, pop=args.pop, elite=args.elite,
                            seed=args.sseed, h_min=args.hmin, b=args.b)
    # re-evaluate the best projection on the FULL transition set (no subsample)
    prep_full = prep_blocks(field, args.b, args.shear, args.r, gpu=GPU_AVAILABLE)
    cb = closure_batch(prep_full, np.array(res["best_pi"], dtype=np.int64)[None, :], h_min=0.0)
    full_clo = float(_to_np(cb["closure"])[0])
    full_exc = float(_to_np(cb["excess"])[0])
    return {"cmd": "search",
            "params": {"rule": args.rule, "b": args.b, "shear": args.shear, "r": args.r,
                       "hmin": args.hmin, "steps": args.steps, "width": args.width,
                       "budget": args.budget, "msearch": args.msearch,
                       "seed": args.seed, "sseed": args.sseed, "gpu": GPU_AVAILABLE},
            "search_closure": round(res["best_closure"], 5),
            "search_excess": round(res["best_excess"], 5),
            "full_closure": round(full_clo, 5),
            "full_excess": round(full_exc, 5),
            "M_search": prep["M"], "M_full": prep_full["M"],
            "evals": res["evals"], "generations": res["generations"],
            "elapsed_s": round(time.perf_counter() - t0, 2)}


def cmd_sim(args) -> dict:
    t0 = time.perf_counter()
    field = _rule_field(args.rule, args.steps, args.width, args.seed)
    col_density = field.mean(axis=0)            # per-column 1-density
    p = field.mean()
    # Shannon entropy of the global bit (bits) and mean per-column entropy.
    def H(q):
        q = np.clip(q, 1e-12, 1 - 1e-12)
        return float(-q * np.log2(q) - (1 - q) * np.log2(1 - q))
    return {"cmd": "sim",
            "params": {"rule": args.rule, "steps": args.steps, "width": args.width,
                       "seed": args.seed, "gpu": GPU_AVAILABLE},
            "shape": list(field.shape),
            "density": round(float(p), 5),
            "global_bit_entropy": round(H(p), 5),
            "mean_col_entropy": round(float(np.mean([H(c) for c in col_density])), 5),
            "elapsed_s": round(time.perf_counter() - t0, 2)}


# --------------------------------------------------------------------------- #
# pretty printers (stderr; stdout stays pure JSON)
# --------------------------------------------------------------------------- #

def _pretty(out: dict):
    c = out["cmd"]
    if c == "sweep":
        print(f"sweep  steps={out['params']['steps']} width={out['params']['width']} "
              f"r={out['params']['r']} hmin={out['params']['hmin']}  "
              f"({out['elapsed_s']}s, gpu={out['params']['gpu']})", file=sys.stderr)
        print(f"  {'rule':>5} | {'excess':>8} {'closure':>8} {'shear':>5}", file=sys.stderr)
        for k, v in out["results"].items():
            b = v["best"]
            print(f"  {k:>5} | {b['excess']:>+8.4f} {b['closure']:>8.4f} "
                  f"{str(b['shear']):>5}", file=sys.stderr)
    elif c == "closure":
        print(f"closure rule={out['params']['rule']} shear={out['params']['shear']}: "
              f"excess {out['excess']:+.4f}  closure {out['closure']:.4f}  "
              f"({out['elapsed_s']}s)", file=sys.stderr)
    elif c == "search":
        print(f"search rule={out['params']['rule']} b={out['params']['b']} "
              f"shear={out['params']['shear']}: search_clo {out['search_closure']:.4f} "
              f"-> full_clo {out['full_closure']:.4f} (excess {out['full_excess']:+.4f})  "
              f"evals={out['evals']} M {out['M_search']}->{out['M_full']}  "
              f"({out['elapsed_s']}s)", file=sys.stderr)
    elif c == "sim":
        print(f"sim rule={out['params']['rule']} {out['shape']}  "
              f"density={out['density']}  col_entropy={out['mean_col_entropy']}  "
              f"({out['elapsed_s']}s)", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    sub_common = argparse.ArgumentParser(add_help=False)
    sub_common.add_argument(
        "--pretty",
        action="store_true",
        default=argparse.SUPPRESS,
        help="also print a human table to stderr",
    )

    p = argparse.ArgumentParser(description="CA coarse-grain exploration CLI (JSON out).")
    p.add_argument(
        "--pretty",
        action="store_true",
        help="also print a human table to stderr",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sweep", parents=[sub_common],
                       help="coarse-grain closure across rules x shears")
    s.add_argument("--rules", default="30,45,90,110")
    s.add_argument("--shears", default="0,0.25,1")
    s.add_argument("--steps", type=int, default=1200)
    s.add_argument("--width", type=int, default=1200)
    s.add_argument("--r", type=int, default=1)
    s.add_argument("--hmin", type=float, default=0.85)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--null", action="store_true", help="also evaluate an i.i.d. null")
    s.set_defaults(func=cmd_sweep)

    c = sub.add_parser("closure", parents=[sub_common], help="single (rule, shear) closure")
    c.add_argument("--rule", type=int, required=True)
    c.add_argument("--shear", type=float, default=0.0)
    c.add_argument("--steps", type=int, default=1200)
    c.add_argument("--width", type=int, default=1200)
    c.add_argument("--r", type=int, default=1)
    c.add_argument("--hmin", type=float, default=0.85)
    c.add_argument("--seed", type=int, default=7)
    c.set_defaults(func=cmd_closure)

    sr = sub.add_parser("search", parents=[sub_common],
                        help="search for the closure-maximizing projection at given b (b>=3)")
    sr.add_argument("--rule", type=int, required=True)
    sr.add_argument("--b", type=int, default=3)
    sr.add_argument("--shear", type=float, default=0.0)
    sr.add_argument("--r", type=int, default=1)
    sr.add_argument("--hmin", type=float, default=0.85)
    sr.add_argument("--steps", type=int, default=1200)
    sr.add_argument("--width", type=int, default=1200)
    sr.add_argument("--budget", type=int, default=50000, help="total projections scored")
    sr.add_argument("--msearch", type=int, default=60000, help="transitions used during search")
    sr.add_argument("--pop", type=int, default=256)
    sr.add_argument("--elite", type=int, default=24)
    sr.add_argument("--seed", type=int, default=7, help="field IC seed")
    sr.add_argument("--sseed", type=int, default=1, help="search RNG seed")
    sr.set_defaults(func=cmd_search)

    m = sub.add_parser("sim", parents=[sub_common], help="simulate a rule, report field statistics")
    m.add_argument("--rule", type=int, required=True)
    m.add_argument("--steps", type=int, default=400)
    m.add_argument("--width", type=int, default=400)
    m.add_argument("--seed", type=int, default=7)
    m.set_defaults(func=cmd_sim)
    return p


def main():
    args = build_parser().parse_args()
    out = args.func(args)
    if args.pretty:
        _pretty(out)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
