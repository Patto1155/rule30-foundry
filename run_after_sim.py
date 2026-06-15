#!/usr/bin/env python
"""Wait for the 46M simulation to finish, then run period search + M + N in sequence.

Usage:
    python run_after_sim.py

Polls for data/center_col_46M.bin every 60s. Once found (and size correct),
runs the three experiments in order and prints a summary.
"""

import time
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
TOTAL_BITS = 46_000_000
EXPECTED_BYTES = (TOTAL_BITS + 7) // 8
SIM_OUTPUT = REPO_ROOT / "data" / "center_col_46M.bin"
SCRIPTS = [
    (REPO_ROOT / "experiments" / "period_search_extended.py", "Extended Period Search"),
    (REPO_ROOT / "experiments" / "causal_sensitivity.py", "Causal Sensitivity Mapping"),
    (REPO_ROOT / "experiments" / "motif_mining.py", "Motif Mining & Grammar Compression"),
]

def wait_for_sim():
    print("Waiting for simulation output …")
    while True:
        if SIM_OUTPUT.exists():
            size = SIM_OUTPUT.stat().st_size
            if size >= EXPECTED_BYTES * 0.99:
                print(f"  Found: {SIM_OUTPUT}  ({size:,} bytes)")
                return
            else:
                print(f"  File exists but incomplete: {size:,}/{EXPECTED_BYTES:,} bytes")
        else:
            print(f"  Not found yet. Waiting 60s …")
        time.sleep(60)


def run_script(script_path: Path, name: str):
    print(f"\n{'='*65}")
    print(f"Running: {name}")
    print(f"{'='*65}")
    t0 = time.perf_counter()
    result = subprocess.run([sys.executable, str(script_path)], cwd=REPO_ROOT, check=False)
    elapsed = time.perf_counter() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\n  {name}: {status}  ({elapsed:.0f}s)")
    return result.returncode == 0


def main():
    wait_for_sim()
    print("\nSimulation complete. Starting experiments …\n")

    results = []
    for script_path, name in SCRIPTS:
        ok = run_script(script_path, name)
        results.append((name, ok))

    print(f"\n{'='*65}")
    print("All experiments complete:")
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'}  {name}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
