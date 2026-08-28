"""Does the pattern-map orbit ever enter a cycle? (Testing the cycle-certificate route.)

Lemma C says the state map F(u,v) = (v,w) is injective wherever v != 0. The
tempting consequence: the orbit is eventually periodic on a finite state space,
so Floyd cycle detection would find its cycle, and a cycle containing no
odd-parity collision would prove period-16 forever -- a finite certificate,
cheaper than a 10^6 walk or a 2^32 bitmap.

This module tests the premise, which turns out to be false.

THE PROBLEM. F is a PARTIAL map. It is undefined at the 2^16 states with v = 0,
because there the one-period composite is affine and TWO period-16 words satisfy
the recursion (verified in zero_word_regression.py gate 3: at d=399 both 0x9f60
and 0x609f are valid, and only simulation picks 0x9f60). So (w_{d-2}, w_{d-1})
is not a complete state -- information from outside the 32 bits enters at every
zero word. Floyd needs a total function; the orbit instead EXITS the
deterministic region.

WHAT IS MEASURED HERE.

  1. Floyd on the real seed orbit. Does it find a cycle, or terminate?
  2. Trajectory-length survey from random start states: F is a partial injection
     with 2^16 terminal states out of 2^32, so a trajectory should survive n
     steps with probability ~(1 - 2^-16)^n -- geometric, mean 2^16. Measured
     against that prediction.
  3. The implied probability that any trajectory survives long enough for a
     cycle to close.

Consequence if the survey confirms the geometric law: the cycle route is not
merely expensive, it is VACUOUS -- the orbit provably keeps leaving the region
where the cycle argument applies. The same objection kills the 2^32 bitmap,
which also assumes a total function.

Run:  python experiments/orbit_cycle_structure.py --pretty
Exits non-zero if the vectorized map disagrees with pattern_map_step.
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

SPACE = 1 << PERIOD
STATES = 1 << (2 * PERIOD)


def f_vec(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized pattern_map_step for v != 0.

    Since v != 0 there is a reset position (v[t] = 1 forces x -> NOT u[t]
    independently of x), so the one-period composite is constant and iterating
    two full periods from any start converges. Bits are recorded during the
    second period, matching pattern_map_step's seq[t] convention exactly.
    """
    x = np.zeros_like(u)
    w = np.zeros_like(u)
    for t in range(2 * PERIOD):
        tt = t & (PERIOD - 1)
        if t >= PERIOD:
            w |= x << tt
        ut = (u >> tt) & 1
        vt = (v >> tt) & 1
        x = ut ^ (vt | x)
    return w


def gate_vec_matches_reference(samples: int = 20000, seed: int = 0) -> dict:
    """The fast map must agree bit-exactly with the trusted scalar one."""
    rng = np.random.default_rng(seed)
    u = rng.integers(0, SPACE, size=samples, dtype=np.int64)
    v = rng.integers(1, SPACE, size=samples, dtype=np.int64)
    fast = f_vec(u, v)
    bad = 0
    for i in range(samples):
        ref, flag = pattern_map_step(int(u[i]), int(v[i]), PERIOD)
        if flag != "ok" or int(fast[i]) != ref:
            bad += 1
    return {"tested": samples, "mismatches": bad, "ok": bad == 0}


def step(state):
    """One step of F. Returns None where F is undefined (v == 0)."""
    u, v = state
    if v == 0:
        return None
    w, flag = pattern_map_step(u, v, PERIOD)
    if flag != "ok":
        return None
    return (v, w)


def floyd(start, max_steps: int) -> dict:
    """Floyd tortoise-and-hare on the real orbit, halting where F is undefined."""
    tortoise = step(start)
    hare = step(step(start)) if tortoise is not None else None
    steps = 1
    while tortoise is not None and hare is not None and tortoise != hare:
        if steps >= max_steps:
            return {"outcome": "step_cap", "steps": steps}
        tortoise = step(tortoise)
        hare = step(hare)
        if hare is not None:
            hare = step(hare)
        steps += 1
    if tortoise is None or hare is None:
        # Recount deterministically: how far does the plain orbit get?
        s, n = start, 0
        while True:
            nxt = step(s)
            if nxt is None:
                return {"outcome": "left_deterministic_region", "steps": n,
                        "final_state": list(s), "reason": "v_is_zero"}
            s = nxt
            n += 1
            if n > max_steps:
                return {"outcome": "step_cap", "steps": n}
    # meeting point found -> locate mu and lambda
    mu = 0
    tortoise = start
    while tortoise != hare:
        tortoise, hare = step(tortoise), step(hare)
        mu += 1
    lam, h = 1, step(hare)
    while hare != h:
        h = step(h)
        lam += 1
    return {"outcome": "cycle", "mu": mu, "lambda": lam}


def survey(n_traj: int, cap: int, seed: int = 30) -> dict:
    """Trajectory lengths from random starts, in lockstep, vs the geometric law."""
    rng = np.random.default_rng(seed)
    u = rng.integers(0, SPACE, size=n_traj, dtype=np.int64)
    v = rng.integers(1, SPACE, size=n_traj, dtype=np.int64)
    alive = np.ones(n_traj, dtype=bool)
    length = np.zeros(n_traj, dtype=np.int64)
    checkpoints, marks = [], [1 << k for k in range(8, 21)]
    for n in range(1, cap + 1):
        w = f_vec(u, v)
        dead_now = alive & (w == 0)
        length[dead_now] = n
        alive &= ~dead_now
        if not alive.any():
            break
        u, v = v, w
        if n in marks:
            checkpoints.append({"steps": n,
                                "survivors": int(alive.sum()),
                                "predicted": round(n_traj * (1 - 1 / SPACE) ** n, 1)})
    finished = length > 0
    return {
        "n_trajectories": n_traj,
        "cap": cap,
        "terminated": int(finished.sum()),
        "still_alive_at_cap": int(alive.sum()),
        "mean_length": float(length[finished].mean()) if finished.any() else None,
        "median_length": float(np.median(length[finished])) if finished.any() else None,
        "predicted_mean": SPACE,
        "survival_checkpoints": checkpoints,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=26000)
    p.add_argument("--diagonals", type=int, default=12000)
    p.add_argument("--tail", type=int, default=4096)
    p.add_argument("--trajectories", type=int, default=4096)
    p.add_argument("--cap", type=int, default=1 << 17)
    p.add_argument("--max-steps", type=int, default=400_000)
    p.add_argument("--pretty", action="store_true")
    p.add_argument("--out")
    a = p.parse_args(argv)

    t0 = time.time()
    gate = gate_vec_matches_reference()

    sim = words_from_simulation(a.steps, a.diagonals, a.tail)
    start = (sim[401], sim[402])          # first state past the last early zero
    fl = floyd(start, a.max_steps)

    sv = survey(a.trajectories, a.cap)

    # For a cycle to close, a trajectory must dodge every terminal state for
    # ~half the state space (expected cycle length of a random permutation).
    expected_cycle = STATES / 2
    log10_p = expected_cycle * math.log10(1 - 1 / SPACE)

    out = {
        "artifact_type": "rule30.orbit_cycle_structure",
        "artifact_version": 1,
        "params": {"period": PERIOD, "state_space": STATES,
                   "terminal_states": SPACE, "trajectories": a.trajectories,
                   "cap": a.cap},
        "gate_vec_matches_reference": gate,
        "floyd_on_seed_orbit": fl,
        "random_start_survey": sv,
        "cycle_feasibility": {
            "expected_cycle_length_random_permutation": expected_cycle,
            "log10_prob_trajectory_survives_that_long": round(log10_p, 1),
        },
        "elapsed_s": round(time.time() - t0, 3),
        "ok": gate["ok"],
    }

    if a.pretty:
        w = sys.stderr
        print(f"vec vs reference : {'PASS' if gate['ok'] else 'FAIL'} "
              f"({gate['tested']} pairs, {gate['mismatches']} mismatches)", file=w)
        print("", file=w)
        print(f"Floyd on the real seed orbit (start d=403):", file=w)
        print(f"  outcome : {fl['outcome']}", file=w)
        for k in ("steps", "mu", "lambda", "reason"):
            if k in fl:
                print(f"  {k:<7} : {fl[k]}", file=w)
        print("", file=w)
        print(f"Random-start survey ({sv['n_trajectories']} trajectories, "
              f"cap {sv['cap']}):", file=w)
        print(f"  terminated       : {sv['terminated']} "
              f"(still alive at cap: {sv['still_alive_at_cap']})", file=w)
        print(f"  mean length      : {sv['mean_length']:.0f} "
              f"vs predicted 2^16 = {sv['predicted_mean']}", file=w)
        print(f"  {'steps':>8} {'survivors':>10} {'geometric':>10}", file=w)
        for c in sv["survival_checkpoints"]:
            print(f"  {c['steps']:>8} {c['survivors']:>10} {c['predicted']:>10.1f}",
                  file=w)
        print("", file=w)
        cf = out["cycle_feasibility"]
        print(f"For a cycle to close, a trajectory must survive "
              f"~{cf['expected_cycle_length_random_permutation']:.3g} steps.", file=w)
        print(f"  P[that happens] ~ 10^{cf['log10_prob_trajectory_survives_that_long']}",
              file=w)

    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str),
                               encoding="utf-8")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
