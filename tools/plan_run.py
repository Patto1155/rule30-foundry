#!/usr/bin/env python
"""Cost and feasibility model for Rule 30 simulation runs.

Everything here is calibrated from runs recorded in this repo, not from
vendor specifications:

    data/center_col_10M_results.json   21M cells x 10M steps
                                       362.94 s  ->  578.6 Gcells/s, 14.5 MB
    data/center_col_46M_results.json   93M cells x 46M steps
                                       7106.15 s ->  602.0 Gcells/s, 66.0 MB

Two facts follow, and both contradict the standing plan.

**The kernel is memory-bandwidth bound, not compute bound.** 590 Gcells/s over
64 cells per uint64 word is ~9.2e9 word-updates/s; each reads three words and
writes one, so ~295 GB/s of traffic against the GTX 1060's 192 GB/s peak -
i.e. it is running at the bandwidth roof with cache reuse doing the rest.
Choose rented hardware by **memory bandwidth**, not by FLOPS or VRAM.

**VRAM is not the constraint, anywhere reachable.** The 46M run used **66 MB**
of a 6 GB card. `docs/handover/2026-08-30-next-session-plan.md` says of the
period search: "The binding constraint is VRAM ... a 24 GB card quadruples the
reachable horizon over the local 6 GB GTX 1060." That is wrong. Cost is
quadratic in the horizon and memory is linear, so time runs out first by an
enormous margin: 1e9 steps still fits a 6 GB card and would take 39 days on
it. Renting for VRAM buys nothing; renting for bandwidth buys time.

Usage:
    python tools/plan_run.py --steps 100000000
    python tools/plan_run.py --steps 100000000 --bandwidth 1555 --price 1.50
    python tools/plan_run.py --table
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gpu"))

import tape_geometry as tg  # noqa: E402

# Calibration point: measured on the GTX 1060 in this repo.
BASELINE_GCELLS_PER_S = 590.0
BASELINE_BANDWIDTH_GBS = 192.0          # GTX 1060 6GB, GDDR5 192-bit

# Memory bandwidth in GB/s. The only spec that predicts throughput here.
CARDS = {
    "gtx1060":  ("GTX 1060 6GB (this repo's baseline)", 192.0),
    "rtx2050":  ("RTX 2050 laptop", 112.0),
    "rtx3090":  ("RTX 3090", 936.0),
    "rtx4090":  ("RTX 4090", 1008.0),
    "l40s":     ("L40S", 864.0),
    "a100-40":  ("A100 40GB", 1555.0),
    "a100-80":  ("A100 80GB", 2039.0),
    "h100":     ("H100 SXM", 3350.0),
}


def throughput(bandwidth_gbs: float) -> float:
    """Gcells/s, scaled linearly from the measured baseline."""
    return BASELINE_GCELLS_PER_S * bandwidth_gbs / BASELINE_BANDWIDTH_GBS


def plan(n_steps: int, bandwidth_gbs: float) -> dict:
    cells = tg.min_cells_for_steps(n_steps)
    geom = tg.describe(cells, n_steps)
    cell_updates = cells * n_steps
    rate = throughput(bandwidth_gbs) * 1e9
    seconds = cell_updates / rate
    vram = geom["tape_bytes"] + geom["center_buffer_bytes"]
    return {
        "n_steps": n_steps,
        "n_cells": cells,
        "cell_updates": cell_updates,
        "gcells_per_s": round(throughput(bandwidth_gbs), 1),
        "seconds": seconds,
        "hours": seconds / 3600.0,
        "vram_bytes": vram,
        "vram_mb": vram / 1024 / 1024,
        "tape_mb": geom["tape_bytes"] / 1024 / 1024,
        "center_mb": geom["center_buffer_bytes"] / 1024 / 1024,
        "output_mb": n_steps / 8 / 1e6,
    }


def _fmt_time(hours: float) -> str:
    if hours < 1:
        return f"{hours * 60:.0f} min"
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} d"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, help="target center-column length")
    ap.add_argument("--bandwidth", type=float, default=None,
                    help="memory bandwidth in GB/s")
    ap.add_argument("--card", choices=sorted(CARDS), default=None)
    ap.add_argument("--price", type=float, default=None,
                    help="USD per hour, to cost the run")
    ap.add_argument("--table", action="store_true",
                    help="horizon x hardware grid")
    args = ap.parse_args()

    if args.table:
        horizons = [10_000_000, 46_000_000, 100_000_000, 300_000_000,
                    1_000_000_000]
        names = ["gtx1060", "rtx4090", "a100-40", "h100"]
        print(f"{'steps':>14} {'cells':>14} {'VRAM':>9}  " +
              "  ".join(f"{CARDS[n][0].split('(')[0].strip():>12}" for n in names))
        print("-" * 96)
        for h in horizons:
            base = plan(h, CARDS["gtx1060"][1])
            row = (f"{h:>14,} {base['n_cells']:>14,} "
                   f"{base['vram_mb']:>7.0f} MB  ")
            row += "  ".join(
                f"{_fmt_time(plan(h, CARDS[n][1])['hours']):>12}" for n in names)
            print(row)
        print()
        print("VRAM is the same on every card - it is a property of the run, "
              "not the hardware.")
        print("Nothing in this table comes close to filling a 6 GB card. "
              "Time is the only constraint.")
        return 0

    if not args.steps:
        ap.error("--steps is required unless --table")

    bw = args.bandwidth
    label = f"{bw:.0f} GB/s"
    if args.card:
        label, bw = CARDS[args.card]
    if bw is None:
        label, bw = CARDS["gtx1060"]

    p = plan(args.steps, bw)
    print(f"target            {p['n_steps']:,} center bits "
          f"({p['output_mb']:.1f} MB packed output)")
    print(f"tape              {p['n_cells']:,} cells "
          f"(light cone exact for every step)")
    print(f"cell updates      {p['cell_updates']:.3e}")
    print(f"hardware          {label}  ->  {p['gcells_per_s']:,.0f} Gcells/s "
          f"(scaled from a measured 590 Gcells/s at 192 GB/s)")
    print(f"wall clock        {_fmt_time(p['hours'])}")
    print(f"VRAM              {p['vram_mb']:.0f} MB "
          f"({p['tape_mb']:.0f} MB tape + {p['center_mb']:.0f} MB center buffer)")
    if args.price:
        print(f"cost              ${p['hours'] * args.price:,.2f} "
              f"at ${args.price:.2f}/h")
    if p["vram_mb"] < 6000:
        print("\nnote: fits a 6 GB card. Renting for VRAM buys nothing here; "
              "rent for bandwidth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
