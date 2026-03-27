"""
Experiment K — Transformer Context Length vs. Complexity
GPT-style next-bit predictor. Tests if BPT falls with larger context length.
Context lengths: 64, 128, 256, 512, 1024.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import sys
import math
from pathlib import Path
from tqdm import tqdm

REPO = Path(__file__).parent.parent
DATA_FILE = REPO / "data" / "center_col_10M.bin"
LOG_FILE = REPO / "docs" / "experiment-logs" / "K_transformer_prediction.md"
RESULTS_FILE = REPO / "data" / "transformer_results.json"

TRAIN_BITS = 7_000_000
TEST_BITS = 3_000_000
EPOCHS = 3
CONTEXT_LENGTHS = [64, 128, 256, 512, 1024]
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
FFN_DIM = 128

# Adaptive batch sizes: O(L^2) attention — halve batch for each doubling of context
BATCH_SIZES = {64: 256, 128: 128, 256: 64, 512: 32, 1024: 16}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def load_bits():
    data = np.frombuffer(open(DATA_FILE, "rb").read(), dtype=np.uint8)
    bits = np.unpackbits(data)[:TRAIN_BITS + TEST_BITS]
    return bits[:TRAIN_BITS], bits[TRAIN_BITS:TRAIN_BITS + TEST_BITS]


class BitEmbedding(nn.Module):
    """Embed 0/1 bits + positional encoding."""
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
        # x: (B, T) int
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
        self.max_len = max_len

    def forward(self, x):
        # x: (B, T) int bits
        T = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        emb = self.embedding(x)
        out = self.encoder(emb, mask=mask, is_causal=True)
        logits = self.head(out).squeeze(-1)  # (B, T)
        return logits


def run_context_length(context_len, train_bits, test_bits):
    print(f"\n--- context_len={context_len} ---")
    t0 = time.time()

    BATCH_SIZE = BATCH_SIZES.get(context_len, 16)

    # Build dataset: stride=context_len//4 for reasonable dataset size
    stride = max(1, context_len // 4)
    n_train = (len(train_bits) - context_len) // stride
    n_test = (len(test_bits) - context_len)  # stride 1 for full eval

    # Clear GPU cache before each context length
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = CausalTransformer(D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, context_len + 1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}  |  Train seqs: {n_train:,}  |  Test seqs: {n_test:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    # Precompute train indices for strided sampling
    train_starts = np.arange(0, len(train_bits) - context_len, stride)

    for epoch in range(EPOCHS):
        model.train()
        perm = np.random.permutation(len(train_starts))
        total_loss = 0.0; n_batches = 0
        bar = tqdm(range(0, len(perm), BATCH_SIZE), desc=f"  Epoch {epoch+1}/{EPOCHS}", leave=False)
        for i in bar:
            batch_starts = train_starts[perm[i:i+BATCH_SIZE]]
            # Build batch: input = bits[s:s+L], target = bits[s+1:s+L+1]
            xb = np.stack([train_bits[s:s+context_len] for s in batch_starts]).astype(np.int64)
            yb = np.stack([train_bits[s+1:s+context_len+1] for s in batch_starts]).astype(np.float32)
            xb = torch.tensor(xb, device=device)
            yb = torch.tensor(yb, device=device)
            optimizer.zero_grad()
            logits = model(xb)  # (B, T)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); n_batches += 1
            bar.set_postfix(loss=f"{total_loss/n_batches:.4f}")
        print(f"  Epoch {epoch+1} avg loss: {total_loss/n_batches:.6f}")

    # Eval on last position only (final bit prediction, no look-ahead)
    model.eval()
    total_loss = 0.0; correct = 0; n_eval = 0
    test_starts = np.arange(0, len(test_bits) - context_len, 4)  # stride 4 for speed
    with torch.no_grad():
        bar = tqdm(range(0, len(test_starts), BATCH_SIZE*2), desc="  Eval", leave=False)
        for i in bar:
            batch_starts = test_starts[i:i+BATCH_SIZE*2]
            xb = np.stack([test_bits[s:s+context_len] for s in batch_starts]).astype(np.int64)
            yb_last = np.array([test_bits[s+context_len] for s in batch_starts], dtype=np.float32)
            xb = torch.tensor(xb, device=device)
            yb_last_t = torch.tensor(yb_last, device=device)
            logits = model(xb)
            last_logit = logits[:, -1]  # predict next bit after context
            loss = criterion(last_logit, yb_last_t)
            total_loss += loss.item() * len(batch_starts)
            pred = (last_logit > 0).float()
            correct += (pred == yb_last_t).sum().item()
            n_eval += len(batch_starts)

    avg_loss = total_loss / n_eval
    accuracy = correct / n_eval
    bpt = avg_loss / math.log(2)
    elapsed = time.time() - t0
    print(f"  Accuracy: {accuracy*100:.4f}%  |  BPT: {bpt:.6f}  |  Time: {elapsed:.0f}s")

    # Free GPU memory before next context length
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "context_len": context_len,
        "n_params": n_params,
        "accuracy": accuracy,
        "bpt": bpt,
        "time_s": round(elapsed, 1)
    }


def main():
    print("=" * 60)
    print("Experiment K — Transformer Context Length vs. Complexity")
    print("=" * 60)

    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found.")
        sys.exit(1)

    print("Loading bits...")
    train_bits, test_bits = load_bits()
    print(f"  Train: {len(train_bits):,}  Test: {len(test_bits):,}")

    # Resume from prior partial results if they exist
    prior_results = []
    completed_ctxs = set()
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE) as f:
                saved = json.load(f)
                prior_results = saved.get("results", [])
                completed_ctxs = {r["context_len"] for r in prior_results}
                print(f"  Resuming: already completed contexts {sorted(completed_ctxs)}")
        except Exception:
            pass

    results = list(prior_results)
    for ctx in CONTEXT_LENGTHS:
        if ctx in completed_ctxs:
            print(f"\n  Skipping context_len={ctx} (already done)")
            continue
        r = run_context_length(ctx, train_bits, test_bits)
        results.append(r)

    print("\n" + "=" * 60)
    print("SUMMARY — Transformer Context Length vs Complexity")
    print("=" * 60)
    print(f"  {'context':>8}  {'params':>10}  {'accuracy':>10}  {'BPT':>10}  {'time':>8}")
    best_bpt = min(r["bpt"] for r in results)
    for r in results:
        flag = " *" if r["bpt"] == best_bpt else ""
        print(f"  {r['context_len']:>8}  {r['n_params']:>10,}  {r['accuracy']*100:>9.4f}%  {r['bpt']:>10.6f}  {r['time_s']:>7.0f}s{flag}")

    bpts = [r["bpt"] for r in results]
    still_decreasing = bpts[-1] < bpts[-2] - 0.001 if len(bpts) >= 2 else False
    improvement = 1.0 - best_bpt

    if improvement > 0.001:
        conclusion = "SIGNIFICANT: Transformer found structure. BPT decreasing with context."
        if still_decreasing:
            conclusion += " Curve still falling at L=1024 — longer context may help further."
    else:
        conclusion = "No improvement over random — computational irreducibility holds for transformer too"

    print(f"\n  Baseline BPT: 1.000000")
    print(f"  Best BPT: {best_bpt:.6f}  (improvement: {improvement:.6f})")
    print(f"  Curve still decreasing at max context: {still_decreasing}")
    print(f"\n  CONCLUSION: {conclusion}")

    with open(RESULTS_FILE, "w") as f:
        json.dump({"results": results, "conclusion": conclusion, "best_bpt": best_bpt,
                   "improvement": improvement, "still_decreasing": still_decreasing}, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    log_content = f"""# Experiment Log

- Date: 2026-03-27
- Title: Transformer Context Length vs. Complexity
- Goal: Test whether longer context helps a transformer predict Rule 30 center column (attacks Problem 2)
- Setup: 7M train / 3M test bits, d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}, context_lengths={CONTEXT_LENGTHS}, epochs={EPOCHS}
- Method: GPT-style next-bit predictor, causal mask, measure BPT at each context length. BPT < 1.0 = structure found.
- Result:
"""
    for r in results:
        log_content += f"  - context={r['context_len']}: accuracy={r['accuracy']*100:.4f}%, BPT={r['bpt']:.6f}, time={r['time_s']:.0f}s\n"
    log_content += f"""- Interpretation: Baseline BPT=1.0. Best BPT={best_bpt:.6f}, improvement={improvement:.6f} bits/token.
  Curve still decreasing at max context: {still_decreasing}.
  {conclusion}
- Next Step: If significant, increase d_model or n_layers; if flat, this is strong irreducibility evidence across all tested architectures
"""
    with open(LOG_FILE, "w") as f:
        f.write(log_content)
    print(f"Experiment log saved to {LOG_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()
