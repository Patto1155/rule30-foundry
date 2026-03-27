"""
Experiment I — LSTM Prediction Scaling Law
Trains LSTM next-bit predictors of varying hidden size on 5M Rule 30 center column bits,
evaluates on held-out 5M bits. Measures bits-per-token and accuracy vs hidden size.
"""
import numpy as np
import torch
import torch.nn as nn
import time
import json
import sys
import os
from pathlib import Path
from tqdm import tqdm

REPO = Path(__file__).parent.parent
DATA_FILE = REPO / "data" / "center_col_10M.bin"
LOG_FILE = REPO / "docs" / "experiment-logs" / "I_lstm_prediction.md"
RESULTS_FILE = REPO / "data" / "lstm_results.json"

SEQ_LEN = 128
BATCH_SIZE = 1024
EPOCHS = 3
HIDDEN_SIZES = [32, 64, 128, 256]
N_LAYERS = 2
TRAIN_BITS = 5_000_000
TEST_BITS = 5_000_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def load_bits():
    data = np.frombuffer(open(DATA_FILE, "rb").read(), dtype=np.uint8)
    # Unpack bits
    bits = np.unpackbits(data)[:TRAIN_BITS + TEST_BITS]
    train = bits[:TRAIN_BITS]
    test = bits[TRAIN_BITS:TRAIN_BITS + TEST_BITS]
    return train, test


def make_sequences(bits, seq_len, stride=1):
    """Return (N, seq_len) input and (N,) target tensors."""
    n = (len(bits) - seq_len) // stride
    X = np.lib.stride_tricks.sliding_window_view(bits[:-1], seq_len)[::stride]
    y = bits[seq_len::stride]
    X = X[:n]
    y = y[:n]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class LSTMPredictor(nn.Module):
    def __init__(self, hidden_size, n_layers):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, n_layers, batch_first=True, dropout=0.1 if n_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T)
        x = x.unsqueeze(-1)  # (B, T, 1)
        out, _ = self.lstm(x)
        logit = self.head(out[:, -1, :]).squeeze(-1)  # (B,)
        return logit


def train_and_eval(hidden_size, train_bits, test_bits):
    print(f"\n--- hidden_size={hidden_size} ---")
    t0 = time.time()

    # Build datasets with stride=4 for speed
    stride = 4
    X_train, y_train = make_sequences(train_bits, SEQ_LEN, stride=stride)
    X_test, y_test = make_sequences(test_bits, SEQ_LEN, stride=1)  # full test eval

    model = LSTMPredictor(hidden_size, N_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    n_train = len(X_train)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}  |  Train seqs: {n_train:,}  |  Test seqs: {len(X_test):,}")

    # Training
    model.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(n_train)
        total_loss = 0.0
        n_batches = 0
        bar = tqdm(range(0, n_train, BATCH_SIZE), desc=f"  Epoch {epoch+1}/{EPOCHS}", leave=False)
        for i in bar:
            idx = perm[i:i+BATCH_SIZE]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)
            optimizer.zero_grad()
            logit = model(xb)
            loss = criterion(logit, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            bar.set_postfix(loss=f"{total_loss/n_batches:.4f}")
        print(f"  Epoch {epoch+1} avg loss: {total_loss/n_batches:.6f}")

    # Evaluation
    model.eval()
    total_loss = 0.0
    correct = 0
    n_test = len(X_test)
    with torch.no_grad():
        bar = tqdm(range(0, n_test, BATCH_SIZE*4), desc="  Eval", leave=False)
        for i in bar:
            xb = X_test[i:i+BATCH_SIZE*4].to(device)
            yb = y_test[i:i+BATCH_SIZE*4].to(device)
            logit = model(xb)
            loss = criterion(logit, yb)
            total_loss += loss.item() * len(xb)
            pred = (logit > 0).float()
            correct += (pred == yb).sum().item()

    avg_loss = total_loss / n_test
    accuracy = correct / n_test
    bpt = avg_loss / np.log(2)  # convert nats to bits

    elapsed = time.time() - t0
    print(f"  Accuracy: {accuracy*100:.4f}%  |  BPT: {bpt:.6f}  |  Time: {elapsed:.0f}s")
    return {
        "hidden_size": hidden_size,
        "n_params": n_params,
        "accuracy": accuracy,
        "bpt": bpt,
        "avg_loss_nats": avg_loss,
        "time_s": round(elapsed, 1)
    }


def main():
    print("=" * 60)
    print("Experiment I — LSTM Prediction Scaling Law")
    print("=" * 60)

    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found. Run gpu/rule30_sim.py first.")
        sys.exit(1)

    print(f"Loading {DATA_FILE}...")
    train_bits, test_bits = load_bits()
    print(f"  Train: {len(train_bits):,} bits, Test: {len(test_bits):,} bits")
    print(f"  Train 1s fraction: {train_bits.mean():.6f}  (expected ~0.500)")

    results = []
    for hs in HIDDEN_SIZES:
        r = train_and_eval(hs, train_bits, test_bits)
        results.append(r)

    print("\n" + "=" * 60)
    print("SUMMARY — LSTM Prediction Scaling Law")
    print("=" * 60)
    print(f"  {'hidden':>8}  {'params':>10}  {'accuracy':>10}  {'BPT':>10}  {'time':>8}")
    best_bpt = min(r["bpt"] for r in results)
    baseline_bpt = 1.0
    for r in results:
        flag = " *" if r["bpt"] == best_bpt else ""
        print(f"  {r['hidden_size']:>8}  {r['n_params']:>10,}  {r['accuracy']*100:>9.4f}%  {r['bpt']:>10.6f}  {r['time_s']:>7.0f}s{flag}")

    improvement = baseline_bpt - best_bpt
    print(f"\n  Baseline BPT (fair coin): 1.000000")
    print(f"  Best BPT:                 {best_bpt:.6f}")
    print(f"  Improvement:              {improvement:.6f} bits/token")

    if improvement > 0.001:
        conclusion = "SIGNIFICANT: LSTM found exploitable structure (non-Markov memory)"
    else:
        conclusion = "No significant improvement over random — supports computational irreducibility"
    print(f"\n  CONCLUSION: {conclusion}")

    with open(RESULTS_FILE, "w") as f:
        json.dump({"results": results, "conclusion": conclusion, "best_bpt": best_bpt,
                   "baseline_bpt": 1.0, "improvement": improvement}, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    log_content = f"""# Experiment Log

- Date: 2026-03-27
- Title: LSTM Prediction Scaling Law
- Goal: Test whether LSTM with non-linear memory can predict Rule 30 center column better than Markov (Exp H)
- Setup: 5M train / 5M test bits, seq_len={SEQ_LEN}, hidden_sizes={HIDDEN_SIZES}, n_layers={N_LAYERS}, epochs={EPOCHS}
- Method: Train LSTM next-bit predictor, measure bits-per-token (BPT) and accuracy on held-out bits. BPT < 1.0 = structure found.
- Result:
"""
    for r in results:
        log_content += f"  - hidden={r['hidden_size']}: accuracy={r['accuracy']*100:.4f}%, BPT={r['bpt']:.6f}, time={r['time_s']:.0f}s\n"
    log_content += f"""- Interpretation: Baseline BPT=1.0 (fair coin). Best BPT={best_bpt:.6f}, improvement={improvement:.6f} bits/token.
  {conclusion}
- Next Step: Run Experiment K (Transformer, larger context) for non-linear + long-range test
"""
    with open(LOG_FILE, "w") as f:
        f.write(log_content)
    print(f"Experiment log saved to {LOG_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()
