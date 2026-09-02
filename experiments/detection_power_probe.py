#!/usr/bin/env python
"""How much structure must exist before the I-L model suite can see it?

Why this exists
---------------
`experiments/rerun_il_bitorder.py` re-ran experiments I-L on the corrected bit
order and added positive controls the originals never had. One control failed
loudly:

    periodic31   LSTM 100.00% acc, Transformer BPT 0.000130   -> found
    lfsr_3_5     Transformer BPT 0.000009, 100.00% acc        -> found
    lfsr_13_27   LSTM BPT 1.000165, Transformer BPT 1.001907  -> NOT FOUND

`lfsr_13_27` is ``s[i] = s[i-13] XOR s[i-27]``: fully determined by 27 bits,
well inside the 64-bit context window, and every model in the suite missed it
completely. That matters for how experiments I-L should be read, because Rule
30 is **left-permutive** - its update is XOR-like - and Experiment S measured
linear complexity ``L(n) = n/2``. The structure most plausibly present in the
center column is exactly the kind this suite is demonstrably blind to.

Before that becomes a claim, one alternative has to be excluded: maybe the
suite failed only because the re-run's budget was small (200k training bits,
2 epochs, hidden <= 64). This script escalates capacity, data and training
time on `lfsr_13_27` specifically, to separate

    "budget-limited"       -> more compute finds it, and I-L at their original
                              GPU scale may well have had the power
    "architecture-limited" -> it stays invisible, and the I-L negatives cannot
                              exclude XOR-type structure at any scale tested

Also fixes a control gap
------------------------
`periodic31` is **not** a positive control for the CNN decile probe
(Experiment J). That probe asks which tenth of the stream a window came from,
and a periodic stream is genuinely stationary - 10% is the *correct* answer,
not a failure. So the re-run had no evidence that the probe can detect
non-stationarity at all. `drift` is a stream whose bias ramps from 0.30 to
0.70 across its length: blatantly non-stationary, and a probe that misses it
cannot support Experiment J's "stationary sequence" verdict.

Usage
-----
    python experiments/detection_power_probe.py --out data/prize/probe.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from prize_lab import git_context  # noqa: E402
from rerun_il_bitorder import (  # noqa: E402
    CausalTransformer, LSTMPredictor, _next_bit_task, cnn_task,
)

ARTIFACT_TYPE = "rule30.detection_power_probe"
ARTIFACT_VERSION = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def lfsr(n_bits: int, taps: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = max(taps)
    out = np.empty(n_bits, dtype=np.uint8)
    out[:order] = rng.integers(0, 2, size=order, dtype=np.uint8)
    a, b = taps
    for i in range(order, n_bits):
        out[i] = out[i - a] ^ out[i - b]
    return out


def drifting(n_bits: int, seed: int, lo: float = 0.30, hi: float = 0.70):
    """Bias ramps lo -> hi across the stream. Blatantly non-stationary."""
    rng = np.random.default_rng(seed)
    p = np.linspace(lo, hi, n_bits)
    return (rng.random(n_bits) < p).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bits", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=30)
    ap.add_argument("--bitstream", type=Path,
                    default=ROOT / "data" / "center_col_10M.bin")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    torch.set_num_threads(4)
    log(f"device {DEVICE}, cuda_available={torch.cuda.is_available()}")

    started = time.time()
    bits = lfsr(args.bits, (13, 27), args.seed)

    # Escalating budgets on the control the suite failed.
    # Escalating budget on the control the suite failed. The rungs separate
    # three explanations rather than just spending compute:
    #   capacity   - wider model, same data
    #   data       - more data and more epochs
    #   visibility - a SHORT context, where both taps (13 and 27) sit well
    #                inside the window and there are 32 positions to consider
    #                rather than 64. If a short-context model with ample
    #                epochs still fails, the limit is optimisation, not
    #                whether the model can see the relevant bits.
    ladder = [
        {"label": "lstm as re-run (h=64, c=64, 200k, 2ep)", "kind": "lstm",
         "hidden": 64, "seq_len": 64, "train_bits": 200_000, "epochs": 2},
        {"label": "lstm wider + more data (h=128, c=64, 300k, 3ep)",
         "kind": "lstm", "hidden": 128, "seq_len": 64,
         "train_bits": 300_000, "epochs": 3},
        {"label": "lstm short ctx (h=128, c=32, 300k, 3ep)", "kind": "lstm",
         "hidden": 128, "seq_len": 32, "train_bits": 300_000, "epochs": 3},
        {"label": "tfmr as re-run (d=64, c=64, 200k, 2ep)", "kind": "tfmr",
         "d_model": 64, "seq_len": 64, "train_bits": 200_000, "epochs": 2},
        {"label": "tfmr wider + more data (d=128, c=64, 300k, 4ep)",
         "kind": "tfmr", "d_model": 128, "seq_len": 64,
         "train_bits": 300_000, "epochs": 4},
        {"label": "tfmr short ctx (d=128, c=32, 300k, 6ep)", "kind": "tfmr",
         "d_model": 128, "seq_len": 32, "train_bits": 300_000, "epochs": 6},
    ]

    log("\nlfsr_13_27 (s[i] = s[i-13] XOR s[i-27]) at escalating budget:")
    rungs = []
    for cfg in ladder:
        t0 = time.time()
        if cfg["kind"] == "lstm":
            r = _next_bit_task(
                bits, lambda c=cfg: LSTMPredictor(c["hidden"]), as_int=False,
                seed=args.seed, seq_len=cfg["seq_len"],
                train_bits=cfg["train_bits"], test_bits=100_000,
                train_stride=4, test_stride=8, epochs=cfg["epochs"], batch=256)
        else:
            r = _next_bit_task(
                bits,
                lambda c=cfg: CausalTransformer(d_model=c["d_model"],
                                                max_len=c["seq_len"] + 8),
                as_int=True, seed=args.seed, seq_len=cfg["seq_len"],
                train_bits=cfg["train_bits"], test_bits=100_000,
                train_stride=max(1, cfg["seq_len"] // 4),
                test_stride=max(1, cfg["seq_len"] // 2),
                epochs=cfg["epochs"], batch=128)
        r["label"] = cfg["label"]
        r["config"] = cfg
        r["elapsed_s"] = round(time.time() - t0, 1)
        r["learned"] = r["bpt"] < 0.95
        rungs.append(r)
        log(f"  {cfg['label']:<44s} BPT {r['bpt']:.6f}  acc {r['accuracy']:.4f}"
            f"  {'LEARNED' if r['learned'] else 'not learned'}  "
            f"({r['elapsed_s']}s, {r['params']:,} params)")

    # CNN control gap: a genuinely non-stationary stream.
    log("\nCNN decile probe (Experiment J) against a non-stationary control:")
    cnn_cfg = dict(cnn_window=512, cnn_train=10_000, cnn_test=3_000,
                   cnn_epochs=2, cnn_classes=10, batch=256)
    cnn_results = {}
    for name, stream in (("drift_0.30_to_0.70", drifting(args.bits, args.seed)),
                         ("random_stationary", np.random.default_rng(
                             args.seed).integers(0, 2, size=args.bits,
                                                 dtype=np.uint8))):
        t0 = time.time()
        r = cnn_task(stream, cnn_cfg, args.seed)
        r["elapsed_s"] = round(time.time() - t0, 1)
        cnn_results[name] = r
        log(f"  {name:<24s} acc {r['accuracy']:.4f} "
            f"(chance {r['chance']:.4f})  ({r['elapsed_s']}s)")

    any_learned = any(r["learned"] for r in rungs)
    drift_acc = cnn_results["drift_0.30_to_0.70"]["accuracy"]
    cnn_has_power = drift_acc > 0.15

    artifact = {
        "artifact_type": ARTIFACT_TYPE,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - started, 1),
        "question": (
            "Is the I-L suite's failure on a 27-bit LFSR budget-limited or "
            "architecture-limited, and can its CNN probe detect "
            "non-stationarity at all?"),
        "device": str(DEVICE),
        "lfsr_13_27_ladder": rungs,
        "lfsr_verdict": (
            "budget-limited: more capacity or data finds it" if any_learned
            else "architecture-limited at every budget tested here: the suite "
                 "cannot see XOR structure at lag 27"),
        "cnn_control": cnn_results,
        "cnn_verdict": (
            "the decile probe detects non-stationarity" if cnn_has_power
            else "the decile probe does NOT detect a bias ramp of 0.30->0.70, "
                 "so Experiment J's 'stationary sequence' verdict is not "
                 "supported by it"),
        "note": (
            "periodic31 is not a positive control for the decile probe: a "
            "periodic stream is genuinely stationary, so 10% is the correct "
            "answer there, not a failure."),
        "git": git_context(),
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(artifact, indent=2) + "\n",
                            encoding="utf-8", newline="")
        log(f"\nwrote {args.out}")
    else:
        print(json.dumps(artifact, indent=2))

    log(f"\nlfsr: {artifact['lfsr_verdict']}")
    log(f"cnn : {artifact['cnn_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
