#!/usr/bin/env python
"""Prize-facing Rule 30 center-column lab.

This CLI keeps exact center-column work separate from broad statistical tests.
Every command prints JSON to stdout and aims to emit either a candidate shortcut
artifact or a checkable finite obstruction.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments"))


ARTIFACT_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def today_slug() -> str:
    return datetime.now().date().isoformat()


def command_string() -> str:
    return "python prize_lab.py " + " ".join(shlex.quote(arg) for arg in sys.argv[1:])


def git_context() -> dict:
    def run_git(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        value = proc.stdout.strip()
        return value or None

    return {
        "git_branch": run_git("branch", "--show-current"),
        "git_head": run_git("rev-parse", "HEAD"),
    }


def run_context(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "command": command_string(),
        **git_context(),
    }


def sha256_ascii(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_csv_ints(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            stop = int(right)
            step = 1 if start <= stop else -1
            out.extend(range(start, stop + step, step))
        else:
            out.append(int(part))
    if not out:
        raise ValueError(f"empty integer list: {value!r}")
    return out


def parse_csv_strings(value: str, *, allowed: set[str] | None = None) -> list[str]:
    out = [part.strip() for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError(f"empty string list: {value!r}")
    if allowed is not None:
        bad = [item for item in out if item not in allowed]
        if bad:
            raise ValueError(f"unsupported values {bad}; allowed={sorted(allowed)}")
    return out


def safe_name_part(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def default_frontier_dir(run_date: str) -> Path:
    return ROOT / "data" / "prize" / f"{run_date}-frontier"


def write_json(path: Path, out: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "?:??"
    seconds_i = int(seconds)
    minutes, sec = divmod(seconds_i, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:d}:{sec:02d}"


class ProgressBar:
    """Small stderr-only progress bar that preserves JSON stdout."""

    def __init__(self, label: str, total: int, *, enabled: bool = True, width: int = 28) -> None:
        self.label = label
        self.total = max(0, int(total))
        self.enabled = enabled and self.total > 0
        self.width = width
        self.start = time.perf_counter()
        self.last_print = 0.0
        self.current = 0
        if self.enabled:
            self.update(0, force=True)

    def update(self, current: int, *, force: bool = False) -> None:
        if not self.enabled:
            return
        self.current = max(0, min(int(current), self.total))
        now = time.perf_counter()
        if not force and self.current < self.total and now - self.last_print < 0.25:
            return
        self.last_print = now
        frac = self.current / self.total if self.total else 1.0
        filled = min(self.width, int(round(frac * self.width)))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = now - self.start
        rate = self.current / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.current) / rate if rate > 0 else None
        line = (
            f"\r{self.label} [{bar}] "
            f"{self.current}/{self.total} {frac * 100:5.1f}% "
            f"{rate:,.0f}/s ETA {format_duration(eta)}"
        )
        sys.stderr.write(line)
        sys.stderr.flush()

    def advance(self, amount: int = 1, *, force: bool = False) -> None:
        self.update(self.current + amount, force=force)

    def finish(self, *, message: str = "done", complete: bool = True) -> None:
        if not self.enabled:
            return
        self.update(self.total if complete else self.current, force=True)
        elapsed = format_duration(time.perf_counter() - self.start)
        sys.stderr.write(f" {message} in {elapsed}\n")
        sys.stderr.flush()


def _rule30_step_int(state: int, mask: int) -> int:
    """One open-boundary Rule 30 step on an integer bitset."""
    return ((state << 1) ^ (state | (state >> 1))) & mask


def center_bits_int(n_steps: int, *, margin: int = 0) -> list[int]:
    """Exact center column bits for a single black cell, times 0..n_steps."""
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    width = 2 * (n_steps + margin) + 1
    center = n_steps + margin
    state = 1 << center
    mask = (1 << width) - 1
    bits: list[int] = []
    for _ in range(n_steps + 1):
        bits.append((state >> center) & 1)
        state = _rule30_step_int(state, mask)
    return bits


def center_bits_reference(n_steps: int, *, margin: int = 64, naive: bool = False) -> list[int]:
    """Reference center bits through existing array/packed helpers."""
    from rule30_open_utils import (
        make_single_spike_row,
        simulate_center_columns_batch,
        simulate_naive_center_columns,
    )

    center = n_steps + margin
    n_cells = 2 * center + 1
    row = make_single_spike_row(n_cells, center)
    if naive:
        cols = simulate_naive_center_columns(row, n_steps, center)[0]
    else:
        cols = simulate_center_columns_batch(row, n_steps, center, gpu=False)[0]
    return [int(x) for x in cols.tolist()]


def verify_center_engine(n_steps: int = 160) -> dict:
    """Cross-check integer, packed CPU, and naive center-column engines."""
    int_bits = center_bits_int(n_steps, margin=64)
    packed_bits = center_bits_reference(n_steps, margin=64, naive=False)
    naive_bits = center_bits_reference(n_steps, margin=64, naive=True)
    ok = int_bits == packed_bits == naive_bits
    if not ok:
        first = next(
            i
            for i, (a, b, c) in enumerate(zip(int_bits, packed_bits, naive_bits))
            if not (a == b == c)
        )
        raise RuntimeError(f"center engine mismatch at step {first}")
    return {
        "checked_steps": n_steps,
        "engines": ["python_int", "packed_cpu", "naive_array"],
        "ok": True,
    }


def bits_ascii(bits: Iterable[int]) -> str:
    return "".join("1" if int(b) else "0" for b in bits)


def sequence_bits(kind: str, n_bits: int, *, margin: int = 0, seed: int = 0) -> list[int]:
    """Generate a named binary sequence prefix, including index 0."""
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    if kind == "center":
        return center_bits_int(n_bits - 1, margin=margin)
    if kind == "thue-morse":
        return [n.bit_count() & 1 for n in range(n_bits)]
    if kind == "random":
        rng = random.Random(seed)
        return [rng.randrange(2) for _ in range(n_bits)]
    raise ValueError(f"unknown sequence kind: {kind}")


def bit_artifact(bits: list[int], params: dict) -> dict:
    s = bits_ascii(bits)
    return {
        "artifact_type": "rule30.center_prefix",
        "artifact_version": ARTIFACT_VERSION,
        "time_zero_included": True,
        "params": params,
        "n_bits": len(bits),
        "ones": int(sum(bits)),
        "zeros": int(len(bits) - sum(bits)),
        "density": float(sum(bits) / len(bits)) if bits else 0.0,
        "sha256_bits_ascii": hashlib.sha256(s.encode("ascii")).hexdigest(),
        "bits": s,
    }


def write_json_if_requested(out: dict, path: str | None) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def cmd_center(args: argparse.Namespace) -> dict:
    t0 = time.perf_counter()
    if args.verify:
        verification = verify_center_engine(min(args.steps, args.verify_steps))
    else:
        verification = None
    bits = center_bits_int(args.steps, margin=args.margin)
    out = bit_artifact(
        bits,
        {
            "steps": args.steps,
            "margin": args.margin,
            "engine": "python_int",
            "verification": verification,
        },
    )
    if not args.include_bits:
        out.pop("bits")
    out["elapsed_s"] = round(time.perf_counter() - t0, 6)
    write_json_if_requested(out, args.out)
    return out


def berlekamp_massey(bits: list[int]) -> tuple[int, list[int]]:
    """Return GF(2) linear complexity and connection polynomial."""
    c = [1]
    b = [1]
    L = 0
    m = 1
    for n in range(len(bits)):
        discrepancy = bits[n]
        for i in range(1, L + 1):
            discrepancy ^= c[i] & bits[n - i]
        if discrepancy == 0:
            m += 1
            continue
        old_c = c[:]
        if len(c) < len(b) + m:
            c.extend([0] * (len(b) + m - len(c)))
        for j, bj in enumerate(b):
            c[j + m] ^= bj
        if 2 * L <= n:
            L = n + 1 - L
            b = old_c
            m = 1
        else:
            m += 1
    return L, c[: L + 1]


def recurrence_taps(poly: list[int]) -> list[int]:
    return [i for i in range(1, len(poly)) if poly[i]]


def recurrence_first_mismatch(bits: list[int], taps: list[int], start: int) -> int | None:
    for n in range(start, len(bits)):
        pred = 0
        for tap in taps:
            pred ^= bits[n - tap]
        if pred != bits[n]:
            return n
    return None


def cmd_recurrence(args: argparse.Namespace) -> dict:
    t0 = time.perf_counter()
    total_steps = args.train_bits + args.holdout_bits - 1
    bits = center_bits_int(total_steps, margin=args.margin)
    train = bits[: args.train_bits]
    L, poly = berlekamp_massey(train)
    taps = recurrence_taps(poly)
    mismatch = recurrence_first_mismatch(bits, taps, max(L, args.train_bits))
    claim_level = "Proof candidate" if mismatch is None and L <= args.max_report_order else "Certificate"
    out = {
        "artifact_type": "rule30.gf2_linear_recurrence_search",
        "artifact_version": ARTIFACT_VERSION,
        "params": {
            "train_bits": args.train_bits,
            "holdout_bits": args.holdout_bits,
            "margin": args.margin,
        },
        "claim_level": claim_level,
        "linear_complexity_train_prefix": L,
        "no_recurrence_below_order": L,
        "connection_polynomial_taps": taps if L <= args.max_report_order else [],
        "candidate_order": L,
        "holdout_checked": args.holdout_bits,
        "holdout_passed": mismatch is None,
        "first_holdout_mismatch": mismatch,
        "sha256_train_bits_ascii": hashlib.sha256(bits_ascii(train).encode("ascii")).hexdigest(),
        "elapsed_s": round(time.perf_counter() - t0, 6),
    }
    write_json_if_requested(out, args.out)
    return out


def kernel_signatures(
    bits: list[int],
    *,
    base: int,
    depth: int,
    sample_len: int,
) -> dict:
    """Finite k-kernel signatures for LSB-first automatic shortcut lower bounds."""
    all_sigs: set[str] = set()
    per_depth = []
    max_index = len(bits) - 1
    for d in range(depth + 1):
        stride = base**d
        sigs: dict[str, list[int]] = {}
        eligible = 0
        for residue in range(stride):
            last = residue + (sample_len - 1) * stride
            if last > max_index:
                continue
            eligible += 1
            sig = bits_ascii(bits[residue + i * stride] for i in range(sample_len))
            digest = hashlib.sha256(sig.encode("ascii")).hexdigest()
            sigs.setdefault(digest, []).append(residue)
            all_sigs.add(digest)
        per_depth.append(
            {
                "depth": d,
                "stride": stride,
                "eligible_residues": eligible,
                "distinct_signatures": len(sigs),
                "collisions": sum(max(0, len(v) - 1) for v in sigs.values()),
            }
        )
    return {
        "per_depth": per_depth,
        "distinct_signatures_all_depths": len(all_sigs),
        "lsd_first_state_lower_bound": len(all_sigs),
    }


def cmd_kernel(args: argparse.Namespace) -> dict:
    t0 = time.perf_counter()
    bits = center_bits_int(args.steps, margin=args.margin)
    sigs = kernel_signatures(bits, base=args.base, depth=args.depth, sample_len=args.sample_len)
    out = {
        "artifact_type": "rule30.finite_kernel_lower_bound",
        "artifact_version": ARTIFACT_VERSION,
        "params": {
            "steps": args.steps,
            "base": args.base,
            "depth": args.depth,
            "sample_len": args.sample_len,
            "margin": args.margin,
            "direction": "lsd",
        },
        "claim_level": "Certificate",
        **sigs,
        "sha256_bits_ascii": hashlib.sha256(bits_ascii(bits).encode("ascii")).hexdigest(),
        "elapsed_s": round(time.perf_counter() - t0, 6),
    }
    write_json_if_requested(out, args.out)
    return out


def digits_for(n: int, base: int, direction: str) -> tuple[int, ...]:
    if n == 0:
        digits = [0]
    else:
        digits = []
        x = n
        while x:
            digits.append(x % base)
            x //= base
        digits.reverse()
    if direction == "lsd":
        digits.reverse()
    return tuple(digits)


def build_prefix_trie(words: list[tuple[int, ...]]) -> tuple[list[dict[int, int]], list[int]]:
    children: list[dict[int, int]] = [{}]
    terminals: list[int] = []
    for word in words:
        node = 0
        for digit in word:
            nxt = children[node].get(digit)
            if nxt is None:
                nxt = len(children)
                children[node][digit] = nxt
                children.append({})
            node = nxt
        terminals.append(node)
    return children, terminals


class CnfBuilder:
    def __init__(self) -> None:
        self.next_var = 1
        self.names: dict[int, str] = {}
        self.clauses: list[list[int]] = []

    def var(self, name: str) -> int:
        v = self.next_var
        self.next_var += 1
        self.names[v] = name
        return v

    def add(self, *lits: int) -> None:
        self.clauses.append(list(lits))

    def exactly_one(self, vars_: list[int]) -> None:
        self.add(*vars_)
        for i in range(len(vars_)):
            for j in range(i + 1, len(vars_)):
                self.add(-vars_[i], -vars_[j])

    def dimacs(self, metadata: dict) -> str:
        lines = [
            "c rule30-foundry prize_lab DFAO finite-prefix SAT encoding",
            f"c metadata {json.dumps(metadata, sort_keys=True)}",
        ]
        for v in sorted(self.names):
            lines.append(f"c var {v} {self.names[v]}")
        lines.append(f"p cnf {self.next_var - 1} {len(self.clauses)}")
        lines.extend(" ".join(str(x) for x in clause) + " 0" for clause in self.clauses)
        return "\n".join(lines) + "\n"


def dfao_sat_cnf(bits: list[int], *, states: int, base: int, direction: str) -> tuple[str, dict]:
    words = [digits_for(n, base, direction) for n in range(len(bits))]
    trie, terminals = build_prefix_trie(words)
    cnf = CnfBuilder()
    x = [[cnf.var(f"x_node{node}_state{state}") for state in range(states)] for node in range(len(trie))]
    trans = [
        [
            [cnf.var(f"t_state{state}_digit{digit}_state{dst}") for dst in range(states)]
            for digit in range(base)
        ]
        for state in range(states)
    ]
    out = [cnf.var(f"out_state{state}") for state in range(states)]

    cnf.add(x[0][0])
    for node_vars in x:
        cnf.exactly_one(node_vars)
    for state in range(states):
        for digit in range(base):
            cnf.exactly_one(trans[state][digit])

    for parent, edges in enumerate(trie):
        for digit, child in edges.items():
            for state in range(states):
                for dst in range(states):
                    cnf.add(-x[parent][state], -trans[state][digit][dst], x[child][dst])

    for n, node in enumerate(terminals):
        bit = bits[n]
        for state in range(states):
            cnf.add(-x[node][state], out[state] if bit else -out[state])

    metadata = {
        "artifact_type": "rule30.dfao_sat_dimacs",
        "artifact_version": ARTIFACT_VERSION,
        "n_bits": len(bits),
        "states": states,
        "base": base,
        "direction": direction,
        "time_zero_included": True,
        "trie_nodes": len(trie),
        "variables": cnf.next_var - 1,
        "clauses": len(cnf.clauses),
        "sha256_bits_ascii": hashlib.sha256(bits_ascii(bits).encode("ascii")).hexdigest(),
    }
    return cnf.dimacs(metadata), metadata


def transition_table(flat: tuple[int, ...], states: int, base: int) -> list[list[int]]:
    return [list(flat[state * base:(state + 1) * base]) for state in range(states)]


def dfao_terminal_state(transitions: list[list[int]], word: tuple[int, ...], initial_state: int = 0) -> int:
    state = initial_state
    for digit in word:
        state = transitions[state][digit]
    return state


def dfao_outputs_if_fit(
    bits: list[int],
    words: list[tuple[int, ...]],
    transitions: list[list[int]],
    states: int,
) -> list[int] | None:
    outputs: list[int | None] = [None] * states
    for n, bit in enumerate(bits):
        state = dfao_terminal_state(transitions, words[n])
        prior = outputs[state]
        if prior is None:
            outputs[state] = bit
        elif prior != bit:
            return None
    return [0 if bit is None else int(bit) for bit in outputs]


def exhaustive_dfao_search(
    bits: list[int],
    *,
    max_states: int,
    base: int,
    direction: str,
    max_transitions: int,
    progress: bool = False,
    progress_label: str = "dfao-search",
) -> dict:
    """Exhaustively search small DFAO transition tables for a finite prefix fit."""
    words = [digits_for(n, base, direction) for n in range(len(bits))]
    per_state = []
    for states in range(1, max_states + 1):
        transition_count = states ** (states * base)
        if transition_count > max_transitions:
            per_state.append(
                {
                    "states": states,
                    "transition_tables": transition_count,
                    "fit_found": None,
                    "skipped": "transition_count_exceeds_limit",
                }
            )
            return {
                "found": False,
                "complete": False,
                "searched_states": states - 1,
                "lower_bound_states": states,
                "per_state": per_state,
                "candidate": None,
            }
        tested = 0
        bar = ProgressBar(
            f"{progress_label} s={states}",
            transition_count,
            enabled=progress,
        )
        for flat in itertools.product(range(states), repeat=states * base):
            tested += 1
            if tested == 1 or tested % 4096 == 0:
                bar.update(tested)
            transitions = transition_table(flat, states, base)
            outputs = dfao_outputs_if_fit(bits, words, transitions, states)
            if outputs is not None:
                bar.finish(message="found", complete=False)
                candidate = {
                    "artifact_type": "rule30.dfao_candidate",
                    "artifact_version": ARTIFACT_VERSION,
                    "states": states,
                    "base": base,
                    "direction": direction,
                    "initial_state": 0,
                    "transitions": transitions,
                    "outputs": outputs,
                }
                per_state.append(
                    {
                        "states": states,
                        "transition_tables": transition_count,
                        "tested": tested,
                        "fit_found": True,
                    }
                )
                return {
                    "found": True,
                    "complete": True,
                    "searched_states": states,
                    "lower_bound_states": states,
                    "per_state": per_state,
                    "candidate": candidate,
                }
        per_state.append(
            {
                "states": states,
                "transition_tables": transition_count,
                "tested": tested,
                "fit_found": False,
            }
        )
        bar.finish(message="no fit")
    return {
        "found": False,
        "complete": True,
        "searched_states": max_states,
        "lower_bound_states": max_states + 1,
        "per_state": per_state,
        "candidate": None,
    }


def cmd_dfao_search(args: argparse.Namespace) -> dict:
    t0 = time.perf_counter()
    bits = sequence_bits(args.sequence, args.bits, margin=args.margin, seed=args.seed)
    search = exhaustive_dfao_search(
        bits,
        max_states=args.max_states,
        base=args.base,
        direction=args.direction,
        max_transitions=args.max_transitions,
        progress=args.progress,
        progress_label=f"dfao-search {args.sequence} b{args.base} {args.direction} n{args.bits}",
    )
    out = {
        "artifact_type": "rule30.dfao_exhaustive_search",
        "artifact_version": ARTIFACT_VERSION,
        "params": {
            "sequence": args.sequence,
            "bits": args.bits,
            "max_states": args.max_states,
            "base": args.base,
            "direction": args.direction,
            "margin": args.margin,
            "seed": args.seed if args.sequence == "random" else None,
            "max_transitions": args.max_transitions,
        },
        "claim_level": "Certificate",
        "sha256_bits_ascii": hashlib.sha256(bits_ascii(bits).encode("ascii")).hexdigest(),
        **search,
        "elapsed_s": round(time.perf_counter() - t0, 6),
    }
    write_json_if_requested(out, args.out)
    return out


def cmd_dfao_sat(args: argparse.Namespace) -> dict:
    t0 = time.perf_counter()
    bits = center_bits_int(args.bits - 1, margin=args.margin)
    dimacs, metadata = dfao_sat_cnf(bits, states=args.states, base=args.base, direction=args.direction)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(dimacs, encoding="ascii")
    out = {
        **metadata,
        "out": args.out,
        "claim_level": "Certificate",
        "meaning": "UNSAT proves no DFAO in this finite class matches the tested prefix; SAT gives a candidate to decode and verify.",
        "elapsed_s": round(time.perf_counter() - t0, 6),
    }
    return out


def run_dfao(candidate: dict, n: int) -> int:
    base = int(candidate["base"])
    direction = candidate.get("direction", "msd")
    state = int(candidate.get("initial_state", 0))
    transitions = candidate["transitions"]
    outputs = candidate["outputs"]
    for digit in digits_for(n, base, direction):
        state = int(transitions[state][digit])
    return int(outputs[state])


def cmd_check_dfao(args: argparse.Namespace) -> dict:
    t0 = time.perf_counter()
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    if isinstance(candidate.get("candidate"), dict):
        candidate = candidate["candidate"]
    bits = sequence_bits(args.sequence, args.steps + 1, margin=args.margin, seed=args.seed)
    first_mismatch = None
    for n, bit in enumerate(bits):
        got = run_dfao(candidate, n)
        if got != bit:
            first_mismatch = {"n": n, "expected": bit, "got": got}
            break
    out = {
        "artifact_type": "rule30.dfao_candidate_check",
        "artifact_version": ARTIFACT_VERSION,
        "candidate": args.candidate,
        "sequence": args.sequence,
        "checked_bits": len(bits),
        "ok": first_mismatch is None,
        "first_mismatch": first_mismatch,
        "sha256_bits_ascii": hashlib.sha256(bits_ascii(bits).encode("ascii")).hexdigest(),
        "elapsed_s": round(time.perf_counter() - t0, 6),
    }
    write_json_if_requested(out, args.out)
    return out


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def sequence_sha(sequence: str, n_bits: int, *, margin: int = 0, seed: int = 0) -> str:
    return sha256_ascii(bits_ascii(sequence_bits(sequence, n_bits, margin=margin, seed=seed)))


def dfao_frontier_filename(sequence: str, bits: int, base: int, direction: str, states: int, seed: int) -> str:
    seed_part = f"_seed{seed}" if sequence == "random" else ""
    return f"dfao_{safe_name_part(sequence)}{seed_part}_b{base}_{direction}_n{bits}_s{states}.json"


def sat_frontier_stem(sequence: str, bits: int, base: int, direction: str, states: int, seed: int) -> str:
    seed_part = f"_seed{seed}" if sequence == "random" else ""
    return f"dfao_sat_{safe_name_part(sequence)}{seed_part}_b{base}_{direction}_n{bits}_s{states}"


def artifact_is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(data.get("complete") or data.get("found"))


def manifest_base(command_name: str, run_id: str, out_dir: Path, started_at: str) -> dict:
    return {
        "artifact_type": "rule30.prize_frontier_manifest",
        "artifact_version": ARTIFACT_VERSION,
        "command_name": command_name,
        "run_id": run_id,
        "created_at": started_at,
        "command": command_string(),
        **git_context(),
        "out_dir": relative_to_root(out_dir),
        "runs": [
            {
                "command_name": command_name,
                "run_id": run_id,
                "created_at": started_at,
                "command": command_string(),
            }
        ],
        "entries": [],
    }


def load_or_create_manifest(manifest_path: Path, command_name: str, run_id: str, out_dir: Path, started_at: str) -> dict:
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("artifact_type") != "rule30.prize_frontier_manifest":
            raise ValueError(f"refusing to merge unknown manifest type at {manifest_path}")
        manifest.setdefault("entries", [])
        manifest.setdefault("runs", [])
        manifest["command_name"] = "mixed-frontier"
        manifest["updated_at"] = started_at
        manifest["out_dir"] = relative_to_root(out_dir)
        manifest["runs"].append(
            {
                "command_name": command_name,
                "run_id": run_id,
                "created_at": started_at,
                "command": command_string(),
            }
        )
        return manifest
    return manifest_base(command_name, run_id, out_dir, started_at)


def write_manifest(path: Path, manifest: dict) -> None:
    entries = manifest["entries"]
    manifest["entry_count"] = len(entries)
    manifest["completed"] = sum(1 for entry in entries if not entry.get("skipped"))
    manifest["skipped"] = sum(1 for entry in entries if entry.get("skipped"))
    manifest["task_count"] = len(entries)
    write_json(path, manifest)


def upsert_manifest_entry(manifest: dict, entry: dict) -> None:
    key = (entry.get("kind"), entry.get("path"))
    for i, old in enumerate(manifest["entries"]):
        if (old.get("kind"), old.get("path")) == key:
            manifest["entries"][i] = entry
            return
    manifest["entries"].append(entry)


def verify_candidate_on_bits(candidate: dict, bits: list[int]) -> dict:
    for n, bit in enumerate(bits):
        got = run_dfao(candidate, n)
        if got != bit:
            return {"ok": False, "first_mismatch": {"n": n, "expected": bit, "got": got}}
    return {"ok": True, "first_mismatch": None}


def dfao_frontier_tasks(args: argparse.Namespace) -> list[dict]:
    sequences = parse_csv_strings(args.sequences, allowed={"center", "thue-morse", "random"})
    bits_values = parse_csv_ints(args.bits)
    bases = parse_csv_ints(args.bases)
    directions = parse_csv_strings(args.directions, allowed={"msd", "lsd"})
    state_values = parse_csv_ints(args.max_states)
    tasks = []
    for sequence, bits, base, direction, states in itertools.product(
        sequences, bits_values, bases, directions, state_values
    ):
        tasks.append(
            {
                "sequence": sequence,
                "bits": bits,
                "base": base,
                "direction": direction,
                "states": states,
                "seed": args.seed if sequence == "random" else 0,
                "margin": args.margin,
                "max_transitions": args.max_transitions,
            }
        )
    return tasks


def cmd_dfao_frontier(args: argparse.Namespace) -> dict:
    tasks = dfao_frontier_tasks(args)
    run_date = args.run_date or today_slug()
    out_dir = Path(args.out_dir) if args.out_dir else default_frontier_dir(run_date)
    run_id = args.run_id or f"dfao-frontier-{run_date}-{uuid.uuid4().hex[:8]}"

    if args.dry_run:
        return {
            "artifact_type": "rule30.dfao_frontier_dry_run",
            "artifact_version": ARTIFACT_VERSION,
            "run_id": run_id,
            "out_dir": relative_to_root(out_dir),
            "tasks": tasks,
            "task_count": len(tasks),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = load_or_create_manifest(manifest_path, "dfao-frontier", run_id, out_dir, utc_now_iso())
    completed = 0
    skipped = 0
    task_bar = ProgressBar("dfao-frontier tasks", len(tasks), enabled=args.progress)
    for task_index, task in enumerate(tasks, start=1):
        artifact_path = out_dir / dfao_frontier_filename(
            task["sequence"],
            task["bits"],
            task["base"],
            task["direction"],
            task["states"],
            task["seed"],
        )
        if artifact_path.exists() and not args.force and artifact_is_complete(artifact_path):
            skipped += 1
            loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
            upsert_manifest_entry(
                manifest,
                {
                    "kind": "dfao_frontier",
                    "path": relative_to_root(artifact_path),
                    "skipped": True,
                    "complete": loaded.get("complete"),
                    "found": loaded.get("found"),
                    "sha256_file": sha256_file(artifact_path),
                    **task,
                }
            )
            write_manifest(manifest_path, manifest)
            task_bar.update(task_index)
            continue

        t0 = time.perf_counter()
        bits = sequence_bits(task["sequence"], task["bits"], margin=task["margin"], seed=task["seed"])
        progress_label = (
            f"dfao-frontier {task['sequence']} b{task['base']} "
            f"{task['direction']} n{task['bits']}"
        )
        search = exhaustive_dfao_search(
            bits,
            max_states=task["states"],
            base=task["base"],
            direction=task["direction"],
            max_transitions=task["max_transitions"],
            progress=args.progress,
            progress_label=progress_label,
        )
        candidate_check = None
        if search.get("candidate"):
            candidate_check = verify_candidate_on_bits(search["candidate"], bits)
        positive_control = None
        if task["sequence"] == "thue-morse" and task["base"] == 2 and task["direction"] == "msd":
            positive_control = {
                "expected_states": 2,
                "passed": bool(search.get("found") and search.get("candidate", {}).get("states") <= 2),
            }

        artifact = {
            "artifact_type": "rule30.dfao_frontier_search",
            "artifact_version": ARTIFACT_VERSION,
            **run_context(run_id),
            "sequence": task["sequence"],
            "bits": task["bits"],
            "base": task["base"],
            "direction": task["direction"],
            "states": task["states"],
            "margin": task["margin"],
            "seed": task["seed"] if task["sequence"] == "random" else None,
            "max_transitions": task["max_transitions"],
            "complete": search["complete"],
            "found": search["found"],
            "lower_bound_states": search["lower_bound_states"],
            "claim_level": "Certificate" if search["complete"] else "Observation",
            "sha256_bits_ascii": sha256_ascii(bits_ascii(bits)),
            "candidate_check": candidate_check,
            "positive_control": positive_control,
            "elapsed_s": round(time.perf_counter() - t0, 6),
            **search,
        }
        write_json(artifact_path, artifact)
        completed += 1
        upsert_manifest_entry(
            manifest,
            {
                "kind": "dfao_frontier",
                "path": relative_to_root(artifact_path),
                "skipped": False,
                "complete": artifact["complete"],
                "found": artifact["found"],
                "sha256_file": sha256_file(artifact_path),
                **task,
            }
        )
        write_manifest(manifest_path, manifest)
        task_bar.update(task_index)

    write_manifest(manifest_path, manifest)
    task_bar.finish()
    return manifest


def solver_command(solver_cmd: str, cnf_path: Path) -> str:
    if "{cnf}" in solver_cmd:
        return solver_cmd.format(cnf=str(cnf_path))
    return f"{solver_cmd} {shlex.quote(str(cnf_path))}"


def parse_solver_status(text: str) -> str:
    upper = text.upper()
    if "UNSATISFIABLE" in upper or "\nS UNSAT" in upper:
        return "UNSAT"
    if "SATISFIABLE" in upper or "\nS SAT" in upper:
        return "SAT"
    return "UNKNOWN"


def run_sat_solver(solver_cmd: str, cnf_path: Path, timeout_s: int) -> dict:
    cmd = solver_command(solver_cmd, cnf_path)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        status = parse_solver_status(combined)
        return {
            "solver_command": cmd,
            "solver_status": status,
            "solver_exit_code": proc.returncode,
            "solver_runtime_s": round(time.perf_counter() - t0, 6),
            "solver_timed_out": False,
            "solver_output_excerpt": combined[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        combined = ((exc.stdout or "") if isinstance(exc.stdout, str) else "") + "\n"
        combined += (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        return {
            "solver_command": cmd,
            "solver_status": "UNKNOWN",
            "solver_exit_code": None,
            "solver_runtime_s": round(time.perf_counter() - t0, 6),
            "solver_timed_out": True,
            "solver_output_excerpt": combined[-4000:],
        }


def sat_frontier_tasks(args: argparse.Namespace) -> list[dict]:
    sequences = parse_csv_strings(args.sequences, allowed={"center", "thue-morse", "random"})
    bits_values = parse_csv_ints(args.bits)
    bases = parse_csv_ints(args.bases)
    directions = parse_csv_strings(args.directions, allowed={"msd", "lsd"})
    state_values = parse_csv_ints(args.states)
    tasks = []
    for sequence, bits, base, direction, states in itertools.product(
        sequences, bits_values, bases, directions, state_values
    ):
        tasks.append(
            {
                "sequence": sequence,
                "bits": bits,
                "base": base,
                "direction": direction,
                "states": states,
                "seed": args.seed if sequence == "random" else 0,
                "margin": args.margin,
            }
        )
    return tasks


def cnf_counts(path: Path) -> dict:
    variables = None
    declared_clauses = None
    actual_clauses = 0
    with path.open("r", encoding="ascii") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("c"):
                continue
            if stripped.startswith("p "):
                parts = stripped.split()
                if len(parts) != 4 or parts[1] != "cnf":
                    raise ValueError(f"unsupported DIMACS header: {stripped}")
                variables = int(parts[2])
                declared_clauses = int(parts[3])
                continue
            actual_clauses += 1
    if variables is None or declared_clauses is None:
        raise ValueError(f"missing DIMACS header in {path}")
    return {
        "variables": variables,
        "clauses": declared_clauses,
        "actual_clause_lines": actual_clauses,
        "header_matches_body": declared_clauses == actual_clauses,
    }


def cmd_dfao_sat_frontier(args: argparse.Namespace) -> dict:
    tasks = sat_frontier_tasks(args)
    run_date = args.run_date or today_slug()
    out_dir = Path(args.out_dir) if args.out_dir else default_frontier_dir(run_date)
    run_id = args.run_id or f"dfao-sat-frontier-{run_date}-{uuid.uuid4().hex[:8]}"

    if args.dry_run:
        return {
            "artifact_type": "rule30.dfao_sat_frontier_dry_run",
            "artifact_version": ARTIFACT_VERSION,
            "run_id": run_id,
            "out_dir": relative_to_root(out_dir),
            "tasks": tasks,
            "task_count": len(tasks),
            "solver_cmd": args.solver_cmd,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = load_or_create_manifest(manifest_path, "dfao-sat-frontier", run_id, out_dir, utc_now_iso())
    completed = 0
    skipped = 0
    task_bar = ProgressBar("dfao-sat-frontier tasks", len(tasks), enabled=args.progress)
    for task_index, task in enumerate(tasks, start=1):
        stem = sat_frontier_stem(
            task["sequence"],
            task["bits"],
            task["base"],
            task["direction"],
            task["states"],
            task["seed"],
        )
        cnf_path = out_dir / f"{stem}.cnf"
        metadata_path = out_dir / f"{stem}.json"
        if (
            metadata_path.exists()
            and cnf_path.exists()
            and not args.force
        ):
            skipped += 1
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            upsert_manifest_entry(
                manifest,
                {
                    "kind": "dfao_sat_frontier",
                    "path": relative_to_root(metadata_path),
                    "cnf_path": relative_to_root(cnf_path),
                    "skipped": True,
                    "complete": loaded.get("complete"),
                    "found": loaded.get("found"),
                    "sha256_file": sha256_file(metadata_path),
                    **task,
                }
            )
            write_manifest(manifest_path, manifest)
            task_bar.update(task_index)
            continue

        t0 = time.perf_counter()
        if args.progress:
            sys.stderr.write(
                f"\nwriting CNF {stem} "
                f"(sequence={task['sequence']} bits={task['bits']} states={task['states']})\n"
            )
            sys.stderr.flush()
        bits = sequence_bits(task["sequence"], task["bits"], margin=task["margin"], seed=task["seed"])
        dimacs, cnf_metadata = dfao_sat_cnf(
            bits,
            states=task["states"],
            base=task["base"],
            direction=task["direction"],
        )
        cnf_path.write_text(dimacs, encoding="ascii")
        solver_result = None
        found = None
        lower_bound_states = None
        complete = False
        if args.solver_cmd:
            solver_result = run_sat_solver(args.solver_cmd, cnf_path, args.solver_timeout_s)
            if solver_result["solver_status"] == "SAT":
                found = True
                complete = True
            elif solver_result["solver_status"] == "UNSAT":
                found = False
                complete = True
                lower_bound_states = task["states"] + 1

        counts = cnf_counts(cnf_path)
        metadata = {
            "artifact_type": "rule30.dfao_sat_frontier",
            "artifact_version": ARTIFACT_VERSION,
            **run_context(run_id),
            "sequence": task["sequence"],
            "bits": task["bits"],
            "base": task["base"],
            "direction": task["direction"],
            "states": task["states"],
            "margin": task["margin"],
            "seed": task["seed"] if task["sequence"] == "random" else None,
            "complete": complete,
            "found": found,
            "lower_bound_states": lower_bound_states,
            "claim_level": "Certificate" if complete else "Observation",
            "sha256_bits_ascii": cnf_metadata["sha256_bits_ascii"],
            "variables": cnf_metadata["variables"],
            "clauses": cnf_metadata["clauses"],
            "cnf_path": relative_to_root(cnf_path),
            "metadata_path": relative_to_root(metadata_path),
            "cnf_sha256": sha256_file(cnf_path),
            "cnf_counts": counts,
            "cnf_emitted": True,
            "solver": solver_result,
            "meaning": "UNSAT proves no at-most-S-state DFAO in this finite class matches the tested prefix; SAT/UNKNOWN need candidate decoding or more solving.",
            "elapsed_s": round(time.perf_counter() - t0, 6),
        }
        write_json(metadata_path, metadata)
        completed += 1
        upsert_manifest_entry(
            manifest,
            {
                "kind": "dfao_sat_frontier",
                "path": relative_to_root(metadata_path),
                "cnf_path": relative_to_root(cnf_path),
                "skipped": False,
                "complete": complete,
                "found": found,
                "sha256_file": sha256_file(metadata_path),
                **task,
            }
        )
        write_manifest(manifest_path, manifest)
        task_bar.update(task_index)

    write_manifest(manifest_path, manifest)
    task_bar.finish()
    return manifest


def eca_bit(rule: int, left: int, center: int, right: int) -> int:
    return (rule >> ((left << 2) | (center << 1) | right)) & 1


def cone_output_window(base_row: tuple[int, ...], *, depth: int, width: int, rule: int) -> tuple[int, ...]:
    row = list(base_row)
    centers: list[int] = []
    first_time = depth - width + 1
    for t in range(depth + 1):
        if t >= first_time:
            centers.append(row[len(row) // 2])
        if t < depth:
            row = [eca_bit(rule, row[i], row[i + 1], row[i + 2]) for i in range(len(row) - 2)]
    return tuple(centers)


def bits_tuple(value: int, width: int) -> tuple[int, ...]:
    return tuple((value >> shift) & 1 for shift in range(width - 1, -1, -1))


def encode_bits(bits: Iterable[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def tuple_bits_ascii(bits: Iterable[int]) -> str:
    return "".join("1" if bit else "0" for bit in bits)


@lru_cache(maxsize=None)
def cone_summary_map(
    *,
    rule: int,
    depth: int,
    summary_width: int,
    core_bits: tuple[int, ...],
) -> tuple[int, ...]:
    domain = 1 << (2 * summary_width)
    out: list[int] = []
    for boundary in range(domain):
        boundary_bits = bits_tuple(boundary, 2 * summary_width)
        left = boundary_bits[:summary_width]
        right = boundary_bits[summary_width:]
        row = left + core_bits + right
        out.append(encode_bits(cone_output_window(row, depth=depth, width=summary_width, rule=rule)))
    return tuple(out)


def cone_summary_stats(rule: int, depth: int, summary_width: int, *, progress: bool = False) -> dict:
    base_width = 2 * depth + 1
    core_width = base_width - 2 * summary_width
    if core_width < 0:
        return {
            "rule": rule,
            "depth": depth,
            "summary_width": summary_width,
            "skipped": "summary_width_exceeds_cone_base",
        }

    summary_ids: dict[tuple[int, ...], int] = {}
    first_core: dict[tuple[int, ...], str] = {}
    collision_example = None
    core_count = 1 << core_width
    core_bar = ProgressBar(
        f"cone rule={rule} d={depth} w={summary_width} summaries",
        core_count,
        enabled=progress,
    )
    for core_value in range(core_count):
        if core_value == 0 or core_value % 1024 == 0:
            core_bar.update(core_value)
        core = bits_tuple(core_value, core_width)
        summary = cone_summary_map(rule=rule, depth=depth, summary_width=summary_width, core_bits=core)
        if summary not in summary_ids:
            summary_ids[summary] = len(summary_ids)
            first_core[summary] = tuple_bits_ascii(core)
        elif collision_example is None:
            collision_example = {
                "core_a": first_core[summary],
                "core_b": tuple_bits_ascii(core),
            }
    core_bar.finish()

    composition = cone_composition_check(rule, depth, summary_width, progress=progress)
    return {
        "rule": rule,
        "depth": depth,
        "summary_width": summary_width,
        "boundary_bits": 2 * summary_width,
        "output_window_bits": summary_width,
        "cone_base_bits": base_width,
        "core_bits": core_width,
        "core_count": core_count,
        "map_domain_size": 1 << (2 * summary_width),
        "distinct_summaries": len(summary_ids),
        "summary_collisions": core_count - len(summary_ids),
        "summary_collision_example": collision_example,
        "composition": composition,
    }


def cone_composition_check(rule: int, depth: int, summary_width: int, *, progress: bool = False) -> dict:
    child_depth = depth - 1
    if child_depth < 1:
        return {"checked": False, "reason": "depth_too_small"}
    child_base = 2 * child_depth + 1
    if child_base - 2 * summary_width < 0:
        return {"checked": False, "reason": "summary_width_exceeds_child_cone_base"}

    summary_ids: dict[tuple[int, ...], int] = {}

    def summary_id(child_row: tuple[int, ...]) -> int:
        core = child_row[summary_width: len(child_row) - summary_width]
        summary = cone_summary_map(
            rule=rule,
            depth=child_depth,
            summary_width=summary_width,
            core_bits=core,
        )
        if summary not in summary_ids:
            summary_ids[summary] = len(summary_ids)
        return summary_ids[summary]

    seen: dict[tuple[int, int, int], tuple[int, str]] = {}
    collision_count = 0
    first_collision = None
    base_width = 2 * depth + 1
    row_count = 1 << base_width
    composition_bar = ProgressBar(
        f"cone rule={rule} d={depth} w={summary_width} compose",
        row_count,
        enabled=progress,
    )
    for row_value in range(row_count):
        if row_value == 0 or row_value % 1024 == 0:
            composition_bar.update(row_value)
        row = bits_tuple(row_value, base_width)
        key = (
            summary_id(row[:-2]),
            summary_id(row[1:-1]),
            summary_id(row[2:]),
        )
        parent_output = encode_bits(cone_output_window(row, depth=depth, width=summary_width, rule=rule))
        row_s = tuple_bits_ascii(row)
        prior = seen.get(key)
        if prior is None:
            seen[key] = (parent_output, row_s)
        elif prior[0] != parent_output:
            collision_count += 1
            if first_collision is None:
                first_collision = {
                    "child_summary_key": key,
                    "row_a": prior[1],
                    "row_b": row_s,
                    "output_a": prior[0],
                    "output_b": parent_output,
                }
    composition_bar.finish()

    return {
        "checked": True,
        "success": collision_count == 0,
        "composition_keys": len(seen),
        "composition_collisions": collision_count,
        "collision_example": first_collision,
        "child_distinct_summaries": len(summary_ids),
    }


def cmd_cone_summary(args: argparse.Namespace) -> dict:
    t0 = time.perf_counter()
    depths = parse_csv_ints(args.depths)
    widths = parse_csv_ints(args.summary_widths)
    rules = parse_csv_ints(args.rules)
    random_rule = None
    if args.include_random_rule:
        random_rule = random.Random(args.random_rule_seed).randrange(256)
        rules.append(random_rule)
    rules = list(dict.fromkeys(rules))

    run_id = args.run_id or f"cone-summary-{today_slug()}-{uuid.uuid4().hex[:8]}"
    results = []
    cases = list(itertools.product(rules, depths, widths))
    case_bar = ProgressBar("cone-summary cases", len(cases), enabled=args.progress)
    for case_index, (rule, depth, width) in enumerate(cases, start=1):
        if width <= 0:
            raise ValueError("summary widths must be positive")
        if width > depth + 1:
            results.append(
                {
                    "rule": rule,
                    "depth": depth,
                    "summary_width": width,
                    "skipped": "summary_width_exceeds_output_times",
                }
            )
            case_bar.update(case_index)
            continue
        results.append(cone_summary_stats(rule, depth, width, progress=args.progress))
        case_bar.update(case_index)
    case_bar.finish()

    out = {
        "artifact_type": "rule30.cone_summary",
        "artifact_version": ARTIFACT_VERSION,
        **run_context(run_id),
        "claim_level": "Observation",
        "complete": True,
        "found": None,
        "lower_bound_states": None,
        "hypothesis": "Small exact cone summaries might compose into a shortcut for n -> center_bit(n).",
        "prediction": "If useful summaries exist, distinct-summary growth and composition collisions should stay controlled as depth grows.",
        "method": "Enumerate exact maps from boundary bits to final center output windows for finite triangular ECA cones.",
        "positive_controls": "Rule 90/shift-like rules can be added via --rules; this default run keeps Rule 30 against chaotic/null controls.",
        "null_controls": {"rules": [r for r in rules if r != 30], "random_rule_seed": args.random_rule_seed, "random_rule": random_rule},
        "depths": depths,
        "summary_widths": widths,
        "rules": rules,
        "results": results,
        "interpretation": "Finite obstruction/observation only; composition collisions rule out this exact summary width, not all possible shortcuts.",
        "next_promotion_step": "Promote only if a summary family has low growth and no composition collisions across larger depths.",
        "elapsed_s": round(time.perf_counter() - t0, 6),
    }
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        write_json(out_path, out)
        if out_path.parent.name.endswith("-frontier"):
            manifest_path = out_path.parent / "manifest.json"
            manifest = load_or_create_manifest(manifest_path, "cone-summary", run_id, out_path.parent, utc_now_iso())
            upsert_manifest_entry(
                manifest,
                {
                    "kind": "cone_summary",
                    "path": relative_to_root(out_path),
                    "skipped": False,
                    "complete": True,
                    "found": None,
                    "sha256_file": sha256_file(out_path),
                    "depths": depths,
                    "summary_widths": widths,
                    "rules": rules,
                },
            )
            write_manifest(manifest_path, manifest)
    return out


def artifact_paths_from_input(path: Path) -> list[Path]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("artifact_type") == "rule30.prize_frontier_manifest":
        paths = []
        for entry in data.get("entries", []):
            entry_path = Path(entry["path"])
            if not entry_path.is_absolute():
                entry_path = ROOT / entry_path
            paths.append(entry_path)
        return paths
    return [path]


def verify_sequence_hash(data: dict) -> list[str]:
    errors = []
    sequence = data.get("sequence")
    bits = data.get("bits")
    expected = data.get("sha256_bits_ascii")
    if not sequence or bits is None or not expected:
        return errors
    seed = int(data.get("seed") or 0)
    margin = int(data.get("margin") or 0)
    actual = sequence_sha(sequence, int(bits), margin=margin, seed=seed)
    if actual != expected:
        errors.append(f"sha256_bits_ascii mismatch for {sequence} n={bits}: {actual} != {expected}")
    return errors


def verify_json_artifact(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = verify_sequence_hash(data)

    if data.get("found") and isinstance(data.get("candidate"), dict):
        bits = sequence_bits(
            data["sequence"],
            int(data["bits"]),
            margin=int(data.get("margin") or 0),
            seed=int(data.get("seed") or 0),
        )
        check = verify_candidate_on_bits(data["candidate"], bits)
        if not check["ok"]:
            errors.append(f"candidate mismatch: {check['first_mismatch']}")

    if data.get("sequence") == "thue-morse" and data.get("base") == 2 and data.get("direction") == "msd":
        if data.get("found") is not True:
            errors.append("Thue-Morse positive control did not find a DFAO")
        candidate = data.get("candidate") or {}
        if candidate and int(candidate.get("states", 999)) > 2:
            errors.append("Thue-Morse positive control exceeded expected 2 states")

    if data.get("artifact_type") == "rule30.dfao_sat_frontier":
        cnf_path = Path(data["cnf_path"])
        if not cnf_path.is_absolute():
            cnf_path = ROOT / cnf_path
        if not cnf_path.exists():
            errors.append(f"missing CNF path: {cnf_path}")
        else:
            counts = cnf_counts(cnf_path)
            if counts["variables"] != int(data["variables"]):
                errors.append(f"variable count mismatch: {counts['variables']} != {data['variables']}")
            if counts["clauses"] != int(data["clauses"]):
                errors.append(f"clause count mismatch: {counts['clauses']} != {data['clauses']}")
            if not counts["header_matches_body"]:
                errors.append("DIMACS clause header does not match clause lines")
            expected_cnf_sha = data.get("cnf_sha256")
            if expected_cnf_sha and sha256_file(cnf_path) != expected_cnf_sha:
                errors.append("CNF sha256 mismatch")

    return {
        "path": relative_to_root(path),
        "artifact_type": data.get("artifact_type"),
        "ok": not errors,
        "errors": errors,
    }


def cmd_verify_artifacts(args: argparse.Namespace) -> dict:
    t0 = time.perf_counter()
    input_path = Path(args.path)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    paths = artifact_paths_from_input(input_path)
    checks = [verify_json_artifact(path) for path in paths]
    ok = all(check["ok"] for check in checks)
    out = {
        "artifact_type": "rule30.prize_artifact_verification",
        "artifact_version": ARTIFACT_VERSION,
        "input": relative_to_root(input_path),
        "checked": len(checks),
        "ok": ok,
        "checks": checks,
        "elapsed_s": round(time.perf_counter() - t0, 6),
    }
    write_json_if_requested(out, args.out)
    return out


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", help="optional artifact path")

    p = argparse.ArgumentParser(description="Rule 30 prize-facing center-column lab")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("center", parents=[common], help="emit exact center-column prefix")
    c.add_argument("--steps", type=int, default=256, help="last time index; emits steps+1 bits")
    c.add_argument("--margin", type=int, default=0)
    c.add_argument("--include-bits", action=argparse.BooleanOptionalAction, default=True)
    c.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    c.add_argument("--verify-steps", type=int, default=160)
    c.set_defaults(func=cmd_center)

    r = sub.add_parser("recurrence", parents=[common], help="GF(2) linear recurrence search")
    r.add_argument("--train-bits", type=int, default=512)
    r.add_argument("--holdout-bits", type=int, default=128)
    r.add_argument("--margin", type=int, default=0)
    r.add_argument("--max-report-order", type=int, default=128)
    r.set_defaults(func=cmd_recurrence)

    k = sub.add_parser("kernel", parents=[common], help="finite k-kernel lower bound for LSB-first DFAO shortcuts")
    k.add_argument("--steps", type=int, default=2048)
    k.add_argument("--base", type=int, default=2)
    k.add_argument("--depth", type=int, default=6)
    k.add_argument("--sample-len", type=int, default=32)
    k.add_argument("--margin", type=int, default=0)
    k.set_defaults(func=cmd_kernel)

    s = sub.add_parser("dfao-sat", help="emit DIMACS for finite-prefix DFAO existence")
    s.add_argument("--bits", type=int, default=64, help="prefix length including time 0")
    s.add_argument("--states", type=int, required=True)
    s.add_argument("--base", type=int, default=2)
    s.add_argument("--direction", choices=["msd", "lsd"], default="msd")
    s.add_argument("--margin", type=int, default=0)
    s.add_argument("--out", required=True, help="DIMACS output path")
    s.set_defaults(func=cmd_dfao_sat)

    f = sub.add_parser("dfao-frontier", help="batch exhaustive finite-prefix DFAO searches")
    f.add_argument("--sequences", default="center,random,thue-morse")
    f.add_argument("--bits", default="128", help="comma/range prefix lengths including index 0")
    f.add_argument("--bases", default="2")
    f.add_argument("--directions", default="msd,lsd")
    f.add_argument("--max-states", default="5", help="comma/range state bounds")
    f.add_argument("--seed", type=int, default=30, help="seed for random baseline")
    f.add_argument("--margin", type=int, default=0)
    f.add_argument("--max-transitions", type=int, default=10_000_000)
    f.add_argument("--out-dir")
    f.add_argument("--run-date")
    f.add_argument("--run-id")
    f.add_argument("--dry-run", action="store_true")
    f.add_argument("--force", action="store_true")
    f.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    f.set_defaults(func=cmd_dfao_frontier)

    sf = sub.add_parser("dfao-sat-frontier", help="batch DFAO SAT CNF + sidecar generation")
    sf.add_argument("--sequences", default="center")
    sf.add_argument("--bits", default="128,256", help="comma/range prefix lengths including index 0")
    sf.add_argument("--states", default="5-8", help="comma/range state counts")
    sf.add_argument("--bases", default="2,4")
    sf.add_argument("--directions", default="msd,lsd")
    sf.add_argument("--seed", type=int, default=30, help="seed for random baseline")
    sf.add_argument("--margin", type=int, default=0)
    sf.add_argument("--solver-cmd", help="optional SAT solver command; use {cnf} placeholder or CNF path is appended")
    sf.add_argument("--solver-timeout-s", type=int, default=600)
    sf.add_argument("--out-dir")
    sf.add_argument("--run-date")
    sf.add_argument("--run-id")
    sf.add_argument("--dry-run", action="store_true")
    sf.add_argument("--force", action="store_true")
    sf.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    sf.set_defaults(func=cmd_dfao_sat_frontier)

    x = sub.add_parser("dfao-search", parents=[common], help="exhaustively search small finite-prefix DFAO")
    x.add_argument("--sequence", choices=["center", "thue-morse", "random"], default="center")
    x.add_argument("--bits", type=int, default=128, help="prefix length including index 0")
    x.add_argument("--max-states", type=int, default=4)
    x.add_argument("--base", type=int, default=2)
    x.add_argument("--direction", choices=["msd", "lsd"], default="msd")
    x.add_argument("--margin", type=int, default=0)
    x.add_argument("--seed", type=int, default=0)
    x.add_argument("--max-transitions", type=int, default=2_000_000)
    x.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    x.set_defaults(func=cmd_dfao_search)

    d = sub.add_parser("check-dfao", parents=[common], help="verify a DFAO candidate JSON")
    d.add_argument("--candidate", required=True)
    d.add_argument("--sequence", choices=["center", "thue-morse", "random"], default="center")
    d.add_argument("--steps", type=int, default=256)
    d.add_argument("--margin", type=int, default=0)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=cmd_check_dfao)

    cs = sub.add_parser("cone-summary", parents=[common], help="enumerate finite light-cone boundary summaries")
    cs.add_argument("--depths", default="4,5,6,7,8")
    cs.add_argument("--summary-widths", default="1,2,3,4")
    cs.add_argument("--rules", default="30,45")
    cs.add_argument("--include-random-rule", action=argparse.BooleanOptionalAction, default=True)
    cs.add_argument("--random-rule-seed", type=int, default=30)
    cs.add_argument("--run-id")
    cs.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    cs.set_defaults(func=cmd_cone_summary)

    v = sub.add_parser("verify-artifacts", parents=[common], help="verify saved frontier artifacts or manifest")
    v.add_argument("path")
    v.set_defaults(func=cmd_verify_artifacts)

    return p


def main() -> None:
    args = build_parser().parse_args()
    out = args.func(args)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
