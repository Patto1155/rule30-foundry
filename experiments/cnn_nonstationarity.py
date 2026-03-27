"""
Experiment J — CNN Non-stationarity Probe
Trains a 1D CNN to classify which temporal decile (10-class) a 512-bit window came from.
If accuracy > 10% (chance), the sequence is non-stationary.
"""
import numpy as np
import torch
import torch.nn as nn
import time
import sys
from pathlib import Path
from tqdm import tqdm

REPO = Path(__file__).parent.parent
DATA_FILE = REPO / "data" / "center_col_10M.bin"
LOG_FILE = REPO / "docs" / "experiment-logs" / "J_cnn_nonstationarity.md"

WINDOW_SIZE = 512
N_CLASSES = 10
TRAIN_WINDOWS = 200_000
TEST_WINDOWS = 40_000
BATCH_SIZE = 512
EPOCHS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def load_bits():
    data = np.frombuffer(open(DATA_FILE, "rb").read(), dtype=np.uint8)
    bits = np.unpackbits(data)[:10_000_000]
    return bits


def make_dataset(bits, n_windows, window_size, n_classes):
    """Sample random windows and assign decile labels."""
    total = len(bits) - window_size
    decile_size = total // n_classes
    rng = np.random.default_rng(42)
    X = np.empty((n_windows, window_size), dtype=np.float32)
    y = np.empty(n_windows, dtype=np.int64)
    for i in range(n_windows):
        label = rng.integers(0, n_classes)
        start = rng.integers(label * decile_size, (label + 1) * decile_size)
        X[i] = bits[start:start + window_size].astype(np.float32)
        y[i] = label
    return torch.tensor(X), torch.tensor(y)


class CNN1D(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1))


def main():
    print("=" * 60)
    print("Experiment J — CNN Non-stationarity Probe")
    print("=" * 60)

    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found.")
        sys.exit(1)

    print("Loading bits...")
    bits = load_bits()
    print(f"  Total bits: {len(bits):,}")

    print("Building datasets...")
    X_train, y_train = make_dataset(bits, TRAIN_WINDOWS, WINDOW_SIZE, N_CLASSES)
    X_test, y_test = make_dataset(bits, TEST_WINDOWS, WINDOW_SIZE, N_CLASSES)
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

    model = CNN1D(N_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  CNN params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    t0 = time.time()
    n_train = len(X_train)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0; n_batches = 0
        bar = tqdm(range(0, n_train, BATCH_SIZE), desc=f"  Epoch {epoch+1}/{EPOCHS}", leave=False)
        for i in bar:
            idx = perm[i:i+BATCH_SIZE]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); n_batches += 1
            bar.set_postfix(loss=f"{total_loss/n_batches:.4f}")
        scheduler.step()
        print(f"  Epoch {epoch+1} avg loss: {total_loss/n_batches:.4f}")

    # Eval
    model.eval()
    correct = 0
    n_test = len(X_test)
    per_class_correct = np.zeros(N_CLASSES)
    per_class_total = np.zeros(N_CLASSES)
    with torch.no_grad():
        for i in range(0, n_test, BATCH_SIZE):
            xb = X_test[i:i+BATCH_SIZE].to(device)
            yb = y_test[i:i+BATCH_SIZE]
            out = model(xb)
            pred = out.argmax(dim=1).cpu()
            correct += (pred == yb).sum().item()
            for c in range(N_CLASSES):
                mask = yb == c
                per_class_correct[c] += (pred[mask] == yb[mask]).sum().item()
                per_class_total[c] += mask.sum().item()

    accuracy = correct / n_test
    elapsed = time.time() - t0
    chance = 1.0 / N_CLASSES
    significant = accuracy > (chance + 0.05)  # 5pp above chance

    print("\n" + "=" * 60)
    print("SUMMARY — CNN Non-stationarity Probe")
    print("=" * 60)
    print(f"  Test accuracy: {accuracy*100:.2f}%  (chance baseline: {chance*100:.1f}%)")
    print(f"  Improvement over chance: {(accuracy - chance)*100:.2f} pp")
    for c in range(N_CLASSES):
        if per_class_total[c] > 0:
            ca = per_class_correct[c] / per_class_total[c]
            print(f"    Decile {c}: {ca*100:.1f}%")
    print(f"  Time: {elapsed:.0f}s")

    if significant:
        conclusion = "SIGNIFICANT: CNN classifies temporal position above chance — sequence is NON-STATIONARY"
    else:
        conclusion = "No significant temporal classification — sequence appears stationary (consistent with equidistribution)"

    print(f"\n  CONCLUSION: {conclusion}")

    log_content = f"""# Experiment Log

- Date: 2026-03-27
- Title: CNN Non-stationarity Probe
- Goal: Detect non-stationarity by classifying which temporal decile a 512-bit window came from (attacks Problem 3)
- Setup: 10M bits split into 10 deciles, {TRAIN_WINDOWS:,} train windows / {TEST_WINDOWS:,} test windows, 1D CNN
- Method: 4-layer 1D CNN classifies decile label. Chance=10%. Significant: >15% accuracy.
- Result:
  - Test accuracy: {accuracy*100:.2f}%  (chance: {chance*100:.1f}%)
  - Improvement over chance: {(accuracy-chance)*100:.2f} pp
  - Per-decile accuracy: {[round(per_class_correct[c]/per_class_total[c]*100, 1) if per_class_total[c] > 0 else 0 for c in range(N_CLASSES)]}
  - Time: {elapsed:.0f}s
- Interpretation: {conclusion}
- Next Step: If non-stationary, investigate which deciles are separable and whether drift is monotonic
"""
    with open(LOG_FILE, "w") as f:
        f.write(log_content)
    print(f"Experiment log saved to {LOG_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()
