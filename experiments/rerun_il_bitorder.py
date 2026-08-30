#!/usr/bin/env python
"""Re-run experiments I-L on the CORRECT bit order, as a paired comparison.

The debt
--------
`gpu/rule30_sim.py` writes center-column dumps LSB-first. Four scripts decoded
them with NumPy's MSB-first default, so README experiments I (LSTM), J (CNN),
K (Transformer) and L (ML scaling) were computed on the center column with
every consecutive 8-bit block reversed - 499,516 of 1,000,000 bit positions
differ, while the bit mean is identical. All four are Retracted.

Why this is not a straight re-run
---------------------------------
The originals used a GPU and budgets to match (5-7M training bits, contexts to
1024, hidden sizes to 256). This machine has no GPU, so reproducing their
absolute numbers is not possible, and comparing a small CPU run against a large
GPU run would confound bit order with scale - the one thing the re-run exists
to separate.

So the design is **paired**: every configuration is trained on both streams at
*identical* budget, seed and architecture.

    center_true      LSB-first decode - the actual center column
    center_reversed  MSB-first decode - exactly what I-L saw

The question "did the bit-order bug change the conclusion?" is then answered by
an internally controlled experiment rather than by a scale-mismatched
comparison against the recorded numbers.

Controls the originals did not have
-----------------------------------
I-L are all "the model learned nothing" results. That is only evidence about
Rule 30 if the pipeline can learn *something* - otherwise BPT ~ 1.000 is
equally consistent with a broken training loop, and nothing in the original
experiments distinguishes those. Three extra streams run at the same budget:

    periodic31  period 31, well inside the 64-bit context - trivially
                learnable, and a pipeline that misses it is simply broken
    lfsr_3_5    s[i] = s[i-3] XOR s[i-5] - short-lag XOR
    lfsr_13_27  s[i] = s[i-13] XOR s[i-27] - long-lag XOR, the hard control,
                and the relevant one: Rule 30 is left-permutive so its update
                is XOR-like, and Experiment S found no short LFSR. A model
                that cannot learn a 27-bit LFSR cannot be trusted to rule one
                out. Reported honestly either way.
    random      i.i.d. fair coin - the floor, must give BPT ~ 1.000

Reading the result
------------------
- `center_true` ~ `center_reversed`  ->  the bug did not change the conclusion.
- `periodic`/`lfsr` well below 1.0    ->  the pipeline has detection power.
- `center_true` ~ `random` ~ 1.0      ->  no exploitable structure at this budget.

Usage
-----
    python experiments/rerun_il_bitorder.py --out data/prize/rerun-il.json
    python experiments/rerun_il_bitorder.py --smoke      # tiny, ~1 min
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
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from prize_lab import git_context  # noqa: E402

ARTIFACT_TYPE = "rule30.rerun_il_bitorder"
ARTIFACT_VERSION = 1
LN2 = math.log(2.0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# streams
# ---------------------------------------------------------------------------


def build_streams(path: Path, n_bits: int, seed: int = 30) -> dict[str, np.ndarray]:
    """The two decodes of the same bytes, plus three controls."""
    n_bytes = (n_bits + 7) // 8
    raw = np.fromfile(str(path), dtype=np.uint8, count=n_bytes)

    true = np.unpackbits(raw, bitorder="little")[:n_bits]
    reversed_ = np.unpackbits(raw, bitorder="big")[:n_bits]

    # The two decodes must differ on ~50% of positions and share a bit count.
    ndiff = int(np.count_nonzero(true != reversed_))
    assert 0.45 < ndiff / n_bits < 0.55, f"decodes differ on {ndiff/n_bits:.3%}"
    assert true.sum() == reversed_.sum(), "per-byte reversal must preserve ones"

    rng = np.random.default_rng(seed)
    random = rng.integers(0, 2, size=n_bits, dtype=np.uint8)

    # Positive controls, as a difficulty ladder rather than one data point.
    # period 31 sits well inside the 64-bit context window, so the last 31
    # bits determine the next one outright: a pipeline that cannot learn this
    # is broken, and any null result from it is uninterpretable.
    periodic = np.resize(rng.integers(0, 2, size=31, dtype=np.uint8), n_bits)

    def lfsr_stream(taps: tuple[int, int]) -> np.ndarray:
        order = max(taps)
        out = np.empty(n_bits, dtype=np.uint8)
        out[:order] = rng.integers(0, 2, size=order, dtype=np.uint8)
        a, b = taps
        for i in range(order, n_bits):
            out[i] = out[i - a] ^ out[i - b]
        return out

    return {
        "center_true": true,
        "center_reversed": reversed_,
        "periodic31": periodic,        # trivial memorisation
        "lfsr_3_5": lfsr_stream((3, 5)),    # short-lag XOR
        "lfsr_13_27": lfsr_stream((13, 27)),  # long-lag XOR, the hard control
        "random": random,              # the floor
    }


# ---------------------------------------------------------------------------
# models (shapes follow the original scripts)
# ---------------------------------------------------------------------------


class LSTMPredictor(nn.Module):
    def __init__(self, hidden_size: int, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, n_layers, batch_first=True,
                            dropout=0.1 if n_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.head(out[:, -1, :]).squeeze(-1)


class BitEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        self.embed = nn.Embedding(2, d_model)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.embed(x) + self.pe[:x.size(1)]


class CausalTransformer(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_layers=2, ffn_dim=256,
                 max_len=2048):
        super().__init__()
        self.embedding = BitEmbedding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        mask = nn.Transformer.generate_square_subsequent_mask(
            x.size(1), device=x.device)
        out = self.encoder(self.embedding(x), mask=mask, is_causal=True)
        return self.head(out).squeeze(-1)


class CNN1D(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes))

    def forward(self, x):
        return self.net(x.unsqueeze(1))


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------


def sequences(bits: np.ndarray, seq_len: int, stride: int):
    win = np.lib.stride_tricks.sliding_window_view(bits[:-1], seq_len)[::stride]
    y = bits[seq_len::stride]
    n = min(len(win), len(y))
    return win[:n], y[:n]


def _next_bit_task(bits, model_fn, *, seq_len, train_bits, test_bits,
                   train_stride, test_stride, epochs, batch, as_int, seed):
    torch.manual_seed(seed)
    train, test = bits[:train_bits], bits[train_bits:train_bits + test_bits]
    Xtr, ytr = sequences(train, seq_len, train_stride)
    Xte, yte = sequences(test, seq_len, test_stride)

    dt = torch.long if as_int else torch.float32
    Xtr = torch.tensor(np.ascontiguousarray(Xtr), dtype=dt)
    Xte = torch.tensor(np.ascontiguousarray(Xte), dtype=dt)
    ytr = torch.tensor(ytr, dtype=torch.float32)
    yte = torch.tensor(yte, dtype=torch.float32)

    model = model_fn().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            logit = model(Xtr[idx].to(DEVICE))
            if logit.dim() == 2:                 # transformer: last position
                logit = logit[:, -1]
            loss = crit(logit, ytr[idx].to(DEVICE))
            loss.backward()
            opt.step()

    model.eval()
    total, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for i in range(0, len(Xte), batch * 4):
            xb = Xte[i:i + batch * 4].to(DEVICE)
            yb = yte[i:i + batch * 4].to(DEVICE)
            logit = model(xb)
            if logit.dim() == 2:
                logit = logit[:, -1]
            total += crit(logit, yb).item() * len(xb)
            correct += ((logit > 0).float() == yb).sum().item()
            n += len(xb)

    return {"bpt": total / n / LN2, "accuracy": correct / n,
            "test_predictions": n,
            "params": sum(p.numel() for p in model.parameters())}


def lstm_task(bits, hidden, cfg, seed):
    return _next_bit_task(
        bits, lambda: LSTMPredictor(hidden), as_int=False, seed=seed,
        seq_len=cfg["seq_len"], train_bits=cfg["train_bits"],
        test_bits=cfg["test_bits"], train_stride=cfg["train_stride"],
        test_stride=cfg["test_stride"], epochs=cfg["epochs"],
        batch=cfg["batch"])


def transformer_task(bits, context, cfg, seed):
    return _next_bit_task(
        bits, lambda: CausalTransformer(max_len=context + 8), as_int=True,
        seed=seed, seq_len=context, train_bits=cfg["train_bits"],
        test_bits=cfg["test_bits"], train_stride=max(1, context // 4),
        test_stride=max(1, context // 2), epochs=cfg["epochs"],
        batch=cfg["tf_batch"])


def cnn_task(bits, cfg, seed):
    """Experiment J: can a CNN tell which decile of the stream a window is from?

    Chance is 1/n_classes. The original reported 10.15% against a 10% floor.
    """
    torch.manual_seed(seed)
    n_classes, window = cfg["cnn_classes"], cfg["cnn_window"]
    rng = np.random.default_rng(seed)
    total = len(bits) - window
    decile = total // n_classes

    def build(n):
        X = np.empty((n, window), dtype=np.float32)
        y = np.empty(n, dtype=np.int64)
        for i in range(n):
            lab = int(rng.integers(0, n_classes))
            start = int(rng.integers(lab * decile, (lab + 1) * decile))
            X[i] = bits[start:start + window]
            y[i] = lab
        return torch.tensor(X), torch.tensor(y)

    Xtr, ytr = build(cfg["cnn_train"])
    Xte, yte = build(cfg["cnn_test"])

    model = CNN1D(n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    model.train()
    for _ in range(cfg["cnn_epochs"]):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            opt.zero_grad()
            loss = crit(model(Xtr[idx].to(DEVICE)), ytr[idx].to(DEVICE))
            loss.backward()
            opt.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(Xte), cfg["batch"] * 4):
            pred = model(Xte[i:i + cfg["batch"] * 4].to(DEVICE)).argmax(1).cpu()
            correct += (pred == yte[i:i + cfg["batch"] * 4]).sum().item()

    return {"accuracy": correct / len(Xte), "chance": 1.0 / n_classes,
            "test_windows": len(Xte),
            "params": sum(p.numel() for p in model.parameters())}


# ---------------------------------------------------------------------------


SMOKE = dict(seq_len=32, train_bits=20_000, test_bits=10_000, train_stride=8,
             test_stride=16, epochs=1, batch=256, tf_batch=128,
             cnn_window=256, cnn_train=1_000, cnn_test=500, cnn_epochs=1,
             cnn_classes=10, lstm_hidden=[32], contexts=[32])

FULL = dict(seq_len=64, train_bits=200_000, test_bits=100_000, train_stride=4,
            test_stride=8, epochs=2, batch=256, tf_batch=128,
            cnn_window=512, cnn_train=10_000, cnn_test=3_000, cnn_epochs=2,
            cnn_classes=10, lstm_hidden=[32, 64], contexts=[64])

# What the retracted README rows recorded, for side-by-side reading only.
RECORDED = {
    "I": "LSTM hidden 32-256: BPT = 1.000001 at all sizes",
    "J": "CNN decile probe: 10.15% accuracy against a 10% floor",
    "K": "Transformer context 64-1024: BPT flat at ~1.000",
    "L": "ML scaling: BPT range < 0.001 across d_model 32-256, n_data 0.5-7M",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bitstream", type=Path,
                    default=ROOT / "data" / "center_col_10M.bin")
    ap.add_argument("--bits", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=30)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if not args.bitstream.exists():
        raise SystemExit(f"bitstream not found: {args.bitstream}")

    cfg = dict(SMOKE if args.smoke else FULL)
    torch.set_num_threads(4)
    log(f"device {DEVICE}, torch {torch.__version__}, "
        f"cuda_available={torch.cuda.is_available()}")

    streams = build_streams(args.bitstream, args.bits, seed=args.seed)
    log(f"streams: {', '.join(streams)}  ({args.bits:,} bits each)")

    started = time.time()
    results: dict[str, dict] = {}

    for name, bits in streams.items():
        entry: dict = {}
        for hidden in cfg["lstm_hidden"]:
            t0 = time.time()
            r = lstm_task(bits, hidden, cfg, args.seed)
            r["elapsed_s"] = round(time.time() - t0, 1)
            entry[f"lstm_h{hidden}"] = r
            log(f"  {name:16s} lstm h={hidden:<4d} BPT {r['bpt']:.6f}  "
                f"acc {r['accuracy']:.4f}  ({r['elapsed_s']}s)")
        for ctx in cfg["contexts"]:
            t0 = time.time()
            r = transformer_task(bits, ctx, cfg, args.seed)
            r["elapsed_s"] = round(time.time() - t0, 1)
            entry[f"transformer_c{ctx}"] = r
            log(f"  {name:16s} tfmr  c={ctx:<4d} BPT {r['bpt']:.6f}  "
                f"acc {r['accuracy']:.4f}  ({r['elapsed_s']}s)")
        t0 = time.time()
        r = cnn_task(bits, cfg, args.seed)
        r["elapsed_s"] = round(time.time() - t0, 1)
        entry["cnn_decile"] = r
        log(f"  {name:16s} cnn         acc {r['accuracy']:.4f} "
            f"(chance {r['chance']:.4f})  ({r['elapsed_s']}s)")
        results[name] = entry

    # The comparison the re-run exists to make.
    deltas = {}
    for key in results["center_true"]:
        a = results["center_true"][key]
        b = results["center_reversed"][key]
        metric = "bpt" if "bpt" in a else "accuracy"
        deltas[key] = {
            "metric": metric,
            "center_true": a[metric],
            "center_reversed": b[metric],
            "abs_difference": abs(a[metric] - b[metric]),
        }

    controls = ("periodic31", "lfsr_3_5", "lfsr_13_27", "random")
    detection = {
        k: {c: results[c][k].get("bpt", results[c][k].get("accuracy"))
            for c in controls}
        for k in results["center_true"]
    }

    artifact = {
        "artifact_type": ARTIFACT_TYPE,
        "artifact_version": ARTIFACT_VERSION,
        "claim_level": "Robust observation (paired, controlled, reduced scale)",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - started, 1),
        "question": (
            "Does correcting the LSB/MSB bit order change the I-L conclusions?"),
        "method": {
            "design": "paired: identical budget, seed and architecture on both "
                      "decodes of the same bytes",
            "why_not_a_direct_reproduction": (
                "The originals ran on a GPU with 5-7M training bits and "
                "contexts to 1024. This machine has no GPU, so comparing "
                "against their absolute numbers would confound bit order with "
                "scale - the one thing this re-run must separate."),
            "positive_controls": (
                "periodic and lfsr streams at the same budget. I-L are all "
                "'the model learned nothing' results, which say something "
                "about Rule 30 only if the pipeline can learn something. The "
                "originals had no such control."),
            "scale_is_reduced": True,
            "device": str(DEVICE),
        },
        "params": {**cfg, "bits": args.bits, "seed": args.seed},
        "recorded_originals": RECORDED,
        "results": results,
        "bitorder_comparison": deltas,
        "detection_power": detection,
        "git": git_context(),
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(artifact, indent=2) + "\n",
                            encoding="utf-8", newline="")
        log(f"\nwrote {args.out}")
    else:
        print(json.dumps(artifact, indent=2))

    worst = max(d["abs_difference"] for d in deltas.values())
    log(f"\nlargest true-vs-reversed difference across all configs: {worst:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
