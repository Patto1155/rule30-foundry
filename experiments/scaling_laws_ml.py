"""
Experiment L — ML Scaling Laws
Measures BPT vs (1) model parameter count and (2) training data size,
using a fixed Transformer architecture (best from Exp K).
Produces two scaling curves:
  - BPT vs log(n_params):  d_model = 32, 64, 128, 256  at fixed data + context
  - BPT vs log(n_data):    data = 500K, 1M, 2M, 5M, 7M at fixed d_model + context
If BPT is flat on both axes: computational irreducibility is robust to scale.
If BPT falls: structure found that is accessible at sufficient capacity/data.
"""
import numpy as np
import torch
import torch.nn as nn
import time
import json
import math
import sys
from pathlib import Path
from tqdm import tqdm

REPO = Path(__file__).parent.parent
DATA_FILE = REPO / "data" / "center_col_10M.bin"
LOG_FILE = REPO / "docs" / "experiment-logs" / "L_scaling_laws_ml.md"
RESULTS_FILE = REPO / "data" / "scaling_laws_ml.json"

# Fixed hyperparameters (best from Exp K: context=512 was last successful)
CONTEXT_LEN = 256      # large enough to test memory, small enough to fit GPU
EPOCHS = 3
FIXED_D_MODEL = 64     # used in data-scaling sweep
FIXED_TRAIN_BITS = 5_000_000  # used in model-scaling sweep
TEST_BITS_START = 7_000_000
TEST_BITS = 3_000_000
N_HEADS = 4
N_LAYERS = 2
FFN_DIM_MULT = 2  # ffn_dim = d_model * FFN_DIM_MULT

# Sweep 1: model size (d_model) at fixed data
MODEL_SIZES = [32, 64, 128, 256]
BATCH_SIZES_MODEL = {32: 256, 64: 128, 128: 64, 256: 32}

# Sweep 2: training data size at fixed model
DATA_SIZES = [500_000, 1_000_000, 2_000_000, 5_000_000, 7_000_000]
BATCH_SIZE_DATA = 64  # fixed for data sweep

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def load_bits():
    data = np.frombuffer(open(DATA_FILE, "rb").read(), dtype=np.uint8)
    return np.unpackbits(data)[:10_000_000]


class BitEmbedding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.embed = nn.Embedding(2, d_model)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.embed(x) + self.pe[:x.size(1)]


class CausalTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, ffn_dim, max_len):
        super().__init__()
        self.embedding = BitEmbedding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        T = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        emb = self.embedding(x)
        out = self.encoder(emb, mask=mask, is_causal=True)
        return self.head(out).squeeze(-1)


def train_eval(d_model, n_params, train_bits, test_bits, batch_size, label):
    """Train a transformer and return BPT on test set."""
    t0 = time.time()
    stride = max(1, CONTEXT_LEN // 4)
    train_starts = np.arange(0, len(train_bits) - CONTEXT_LEN, stride)
    test_starts = np.arange(0, len(test_bits) - CONTEXT_LEN, 4)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ffn_dim = d_model * FFN_DIM_MULT
    n_heads_actual = max(1, min(N_HEADS, d_model // 16))  # ensure d_model divisible by n_heads
    model = CausalTransformer(d_model, n_heads_actual, N_LAYERS, ffn_dim, CONTEXT_LEN + 1).to(device)
    n_params_actual = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    print(f"  {label}: d_model={d_model}, params={n_params_actual:,}, train_seqs={len(train_starts):,}, batch={batch_size}")

    for epoch in range(EPOCHS):
        model.train()
        perm = np.random.permutation(len(train_starts))
        total_loss = 0.0; n_batches = 0
        bar = tqdm(range(0, len(perm), batch_size), desc=f"    Epoch {epoch+1}/{EPOCHS}", leave=False)
        for i in bar:
            batch_starts = train_starts[perm[i:i+batch_size]]
            xb = torch.tensor(np.stack([train_bits[s:s+CONTEXT_LEN] for s in batch_starts]).astype(np.int64), device=device)
            yb = torch.tensor(np.stack([train_bits[s+1:s+CONTEXT_LEN+1] for s in batch_starts]).astype(np.float32), device=device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); n_batches += 1
            bar.set_postfix(loss=f"{total_loss/n_batches:.4f}")
        print(f"    Epoch {epoch+1} loss: {total_loss/n_batches:.6f}")

    model.eval()
    total_loss = 0.0; correct = 0; n_eval = 0
    with torch.no_grad():
        for i in range(0, len(test_starts), batch_size * 2):
            batch_starts = test_starts[i:i+batch_size*2]
            xb = torch.tensor(np.stack([test_bits[s:s+CONTEXT_LEN] for s in batch_starts]).astype(np.int64), device=device)
            yb_last = torch.tensor([test_bits[s+CONTEXT_LEN] for s in batch_starts], dtype=torch.float32, device=device)
            logits = model(xb)
            loss = criterion(logits[:, -1], yb_last)
            total_loss += loss.item() * len(batch_starts)
            correct += ((logits[:, -1] > 0).float() == yb_last).sum().item()
            n_eval += len(batch_starts)

    bpt = (total_loss / n_eval) / math.log(2)
    accuracy = correct / n_eval
    elapsed = time.time() - t0
    print(f"    BPT={bpt:.6f}, accuracy={accuracy*100:.4f}%, time={elapsed:.0f}s")

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"d_model": d_model, "n_params": n_params_actual, "bpt": bpt,
            "accuracy": accuracy, "time_s": round(elapsed, 1)}


def main():
    print("=" * 60)
    print("Experiment L — ML Scaling Laws (model size + data size)")
    print("=" * 60)

    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found.")
        sys.exit(1)

    all_bits = load_bits()
    print(f"Loaded {len(all_bits):,} bits")
    test_bits = all_bits[TEST_BITS_START:TEST_BITS_START + TEST_BITS]

    # === Sweep 1: Model Size ===
    print(f"\n{'='*50}")
    print(f"SWEEP 1: Model size (d_model sweep, context={CONTEXT_LEN}, train={FIXED_TRAIN_BITS:,})")
    print(f"{'='*50}")
    train_bits_fixed = all_bits[:FIXED_TRAIN_BITS]
    model_sweep = []
    for d_model in MODEL_SIZES:
        r = train_eval(d_model, None, train_bits_fixed, test_bits,
                       BATCH_SIZES_MODEL.get(d_model, 32), f"d_model={d_model}")
        model_sweep.append(r)

    # === Sweep 2: Data Size ===
    print(f"\n{'='*50}")
    print(f"SWEEP 2: Data size (d_model={FIXED_D_MODEL}, context={CONTEXT_LEN})")
    print(f"{'='*50}")
    data_sweep = []
    for n_data in DATA_SIZES:
        train_bits_var = all_bits[:n_data]
        r = train_eval(FIXED_D_MODEL, None, train_bits_var, test_bits,
                       BATCH_SIZE_DATA, f"n_data={n_data:,}")
        r["n_data"] = n_data
        data_sweep.append(r)

    # === Summary ===
    print("\n" + "=" * 60)
    print("SUMMARY — ML Scaling Laws")
    print("=" * 60)
    print("\nSweep 1 (model size):")
    print(f"  {'d_model':>8}  {'params':>10}  {'BPT':>10}  {'time':>8}")
    for r in model_sweep:
        print(f"  {r['d_model']:>8}  {r['n_params']:>10,}  {r['bpt']:>10.6f}  {r['time_s']:>7.0f}s")

    print("\nSweep 2 (data size):")
    print(f"  {'n_data':>10}  {'BPT':>10}  {'time':>8}")
    for r in data_sweep:
        print(f"  {r['n_data']:>10,}  {r['bpt']:>10.6f}  {r['time_s']:>7.0f}s")

    model_bpts = [r["bpt"] for r in model_sweep]
    data_bpts = [r["bpt"] for r in data_sweep]
    model_range = max(model_bpts) - min(model_bpts)
    data_range = max(data_bpts) - min(data_bpts)

    if model_range > 0.01 or data_range > 0.01:
        conclusion = "SIGNIFICANT scaling: BPT varies meaningfully with model size or data size — structure accessible at scale"
    else:
        conclusion = (f"Flat scaling: BPT range = {model_range:.6f} (model), {data_range:.6f} (data). "
                      "No improvement with scale — computational irreducibility is robust")

    print(f"\n  CONCLUSION: {conclusion}")

    results = {"model_sweep": model_sweep, "data_sweep": data_sweep,
               "conclusion": conclusion, "model_bpt_range": model_range, "data_bpt_range": data_range}
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    log_content = f"""# Experiment Log

- Date: 2026-03-27
- Title: ML Scaling Laws (model size and data size)
- Goal: Test if BPT improves with larger model or more training data (proper scaling law analysis)
- Setup: context_len={CONTEXT_LEN}, n_heads={N_HEADS}, n_layers={N_LAYERS}, epochs={EPOCHS}
  - Model sweep: d_model={MODEL_SIZES} at {FIXED_TRAIN_BITS:,} training bits
  - Data sweep: d_model={FIXED_D_MODEL} at n_data={DATA_SIZES}
- Method: Train Transformer predictor, measure BPT on held-out 3M bits at each scale point
- Result:
  - Model sweep BPT range: {model_range:.6f}
  - Data sweep BPT range: {data_range:.6f}
"""
    for r in model_sweep:
        log_content += f"  - d_model={r['d_model']}: BPT={r['bpt']:.6f}, params={r['n_params']:,}\n"
    log_content += "  Data sweep:\n"
    for r in data_sweep:
        log_content += f"  - n_data={r['n_data']:,}: BPT={r['bpt']:.6f}\n"
    log_content += f"- Interpretation: {conclusion}\n"
    log_content += "- Next Step: If flat, the irreducibility evidence is now complete across architectures, scales, and data sizes\n"

    with open(LOG_FILE, "w") as f:
        f.write(log_content)
    print(f"\nResults saved. Log: {LOG_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()
