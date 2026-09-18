#!/usr/bin/env python
"""Shard a DFAO card space across cores and run jev_search on each shard.

The per-shard runner (tools/jev_search.py) is sequential by design: one solve
at a time, so its verified-implication pruning stays sound. Throughput comes
from running independent shards side by side, not from loosening that.

Sharding is by prefix length n. Cards sharing an n are exactly the cards that
prune each other (monotonicity in states at fixed n), so splitting on n keeps
every implication inside one shard and loses no pruning.

See docs/JEV_SEARCH.md. This driver schedules; it certifies nothing.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A frontier card claims an exact exclusion, so it needs a matched random null
# at the same (n, states, base, direction); load_plan enforces this.
CONTROLS = [
    ("control-tm64-s2", "thue-morse", 64, 2, "calibration", "SAT"),
    ("control-center16-s4", "center", 16, 4, "calibration", "UNSAT"),
]


def card(cid, sequence, n, states, role, expected=None, seed=30, base=2, direction="msd"):
    c = {"id": cid, "sequence": sequence, "n": n, "states": states, "base": base,
         "direction": direction, "role": role,
         "seed": seed if sequence == "random" else 0,
         "hypothesis": f"{sequence} prefix n={n} admits a {states}-state base-{base} {direction} DFAO",
         "if_sat": f"a {states}-state DFAO reproduces the first {n} bits; no exclusion at this size",
         "if_unsat": f"no {states}-state DFAO reproduces the first {n} bits; finite exclusion at n={n}"}
    if expected:
        c["expected"] = expected
    return c


def shard_plan(goal, n, states, seed):
    cards = [card(cid, seq, cn, cs, role, exp) for cid, seq, cn, cs, role, exp in CONTROLS]
    # A control already pins its own (sequence, n, states); load_plan rejects a
    # second card on that instance, so a shard overlapping a control skips it
    # rather than restating a calibration as a frontier question.
    taken = {(seq, cn, cs) for _, seq, cn, cs, _, _ in CONTROLS}
    # Center cards first, then their nulls. The fixed policy walks the plan in
    # order, and a matched null runs 3-4x longer than the center card it
    # controls (n=64: 493 s vs 123 s at s=12), so interleaving lets a control
    # eat the wall budget and strand the frontier card behind it. The null is
    # still required for admissibility; it just need not run first.
    centers, nulls = [], []
    for s in states:
        if ("center", n, s) in taken:
            continue
        centers.append(card(f"center{n}-s{s}", "center", n, s, "frontier"))
        nulls.append(card(f"random{n}-s{s}", "random", n, s, "frontier", seed=seed))
    return {"scope": "frontier", "goal": goal, "cards": cards + centers + nulls}


def run_shard(plan_path, out, args):
    argv = [sys.executable, str(ROOT / "tools/jev_search.py"), str(plan_path),
            "--policy", args.policy, "--out", str(out),
            "--max-steps", str(args.max_steps), "--seconds", str(args.seconds),
            "--solve-seconds", str(args.solve_seconds),
            "--check-seconds", str(args.check_seconds), "--seed", str(args.seed)]
    if args.policy == "jev":
        argv += ["--jev-provider", args.jev_provider]
    p = subprocess.run(argv, capture_output=True, text=True, cwd=ROOT)
    return p.returncode


def prune_proofs(out):
    """Drop regenerable CNF/DRAT bulk; keep result.json, witnesses and hashes.

    A verified UNSAT proof at n=52 is ~112 MB, so a sweep fills the disk long
    before it exhausts the card space. Hashes stay in result.json, and the CNF
    is a pure function of the card, so a run stays re-checkable from the plan.
    Drop this flag when a run is meant to be a retained ledger artifact.
    """
    freed = 0
    for pattern in ("*/instance.drat", "*/instance.cnf"):
        for f in out.glob(pattern):
            freed += f.stat().st_size
            f.unlink()
    return freed


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="campaign directory (must not exist)")
    p.add_argument("--goal", required=True)
    p.add_argument("--n", required=True, help="comma-separated prefix lengths, one shard each")
    p.add_argument("--states", required=True, help="comma-separated state counts")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--policy", choices=("fixed", "random", "jev"), default="fixed")
    p.add_argument("--jev-provider", choices=("auto", "typesafe", "openrouter"), default="auto")
    p.add_argument("--seconds", type=float, default=1800)
    p.add_argument("--solve-seconds", type=float, default=60)
    # drat-trim is the binding cost at the frontier, not the solver: an n=52
    # refutation solved in 18 s and took 21 s to check. Keep this well above
    # --solve-seconds or shards halt with stop_reason=verification-failed.
    p.add_argument("--check-seconds", type=float, default=300)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=30)
    p.add_argument("--keep-proofs", action="store_true")
    args = p.parse_args()

    if args.check_seconds < args.solve_seconds:
        print("jev_campaign: --check-seconds below --solve-seconds invites "
              "verification-failed halts on large refutations", file=sys.stderr)
    ns = [int(x) for x in args.n.split(",")]
    states = [int(x) for x in args.states.split(",")]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plans = out / "plans"
    plans.mkdir()

    jobs = []
    for n in ns:
        path = plans / f"n{n}.json"
        path.write_text(json.dumps(shard_plan(args.goal, n, states, args.seed), indent=2) + "\n")
        jobs.append((n, path))

    codes = {}
    with futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(run_shard, path, out / f"n{n}", args): n for n, path in jobs}
        for f in futures.as_completed(futs):
            n = futs[f]
            codes[n] = f.result()
            print(f"n={n} exit={codes[n]}", file=sys.stderr, flush=True)

    freed = 0
    shards = {}
    for n, _ in jobs:
        summary = out / f"n{n}" / "summary.json"
        if summary.exists():
            s = json.loads(summary.read_text())
            shards[n] = {"stop_reason": s["stop_reason"], "attempted": s["attempted"],
                         "inferred": len(s["inferred"]), "elapsed_s": s["elapsed_s"],
                         "jev_calls": s["jev_calls"], "exit": codes[n]}
        else:
            shards[n] = {"stop_reason": "no-summary", "exit": codes[n]}
        if not args.keep_proofs:
            freed += prune_proofs(out / f"n{n}")

    report = {"goal": args.goal, "policy": args.policy, "shards": shards,
              "proof_bytes_pruned": freed,
              "solved": sum(v.get("attempted", 0) for v in shards.values()),
              "inferred": sum(v.get("inferred", 0) for v in shards.values()),
              "clean": [n for n, v in shards.items() if v["exit"] == 0],
              "halted": [n for n, v in shards.items() if v["exit"] != 0],
              "claim_limit": "Finite DFAO instances only. Scheduling report; "
                             "verification lives in each shard's result.json."}
    (out / "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 1 if report["halted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
