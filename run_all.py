#!/usr/bin/env python
"""Orchestrator — run experiments in priority order with telemetry guidance.

Usage:
  python run_all.py          # full runs (M ~5min, N ~3min, O ~2min, FFT ~30s)
  python run_all.py --test   # quick smoke-test of all scripts (~30s total)

Execution order (fastest -> biggest GPU):
  1. fft_autocorr   — seconds, closes the period-42795 question
  2. compress_probe — ~3 min CPU, attacks Prize Problem 2
  3. causal_sensitivity — ~5 min GPU, first dynamical geometry result
  4. column_mi      — ~2 min GPU, first 2D MI/TE analysis
  5. sim_200m       — SKIPPED here (overnight run, start separately)

For the overnight sim (Experiment 5):
  # In a separate terminal, start telemetry FIRST:
  nvidia-smi dmon -s pucvmet -d 5 -f data\\gpu_telemetry.csv
  # Then start the simulation:
  C:\\Python313\\python.exe experiments\\sim_200m.py
  # Watch progress:
  tail -f data\\sim_progress.log     (or: type data\\sim_progress.log on Windows)
"""
import sys, subprocess, time, datetime
from pathlib import Path

PYTHON  = r"C:\Python313\python.exe"
REPO    = Path(r"D:\APATPROJECTS\rule30-research")
EXPS    = REPO / "experiments"
TEST    = "--test" in sys.argv

EXPERIMENTS = [
    ("fft_autocorr",        "FFT Autocorrelation — closes period-42795 question", "data/fft_autocorr.progress.log"),
    ("compress_probe",      "Compression Probe   — attacks Prize Problem 2", "data/N_progress.log"),
    ("causal_sensitivity",  "Causal Sensitivity  — dynamical geometry (GPU)", "data/M_progress.log"),
    ("column_mi",           "Column MI + TE      — first 2D analysis (GPU)", "data/O_progress.log"),
]


def banner(title, char="=", width=65):
    print(char * width)
    print(f"  {title}")
    print(char * width, flush=True)


def run_experiment(name: str, label: str, progress_log: str) -> int:
    script = EXPS / f"{name}.py"
    if not script.exists():
        print(f"  [SKIP] {script} not found", flush=True)
        return -1

    banner(label, char="-")
    print(f"  Script:   {script}")
    print(f"  Progress: {progress_log}")
    print(f"  Start:    {datetime.datetime.now().strftime('%H:%M:%S')}")
    print(flush=True)

    cmd = [PYTHON, str(script)]
    if TEST:
        cmd.append("--test")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(REPO))
    elapsed = time.perf_counter() - t0

    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\n  [{status}] {label} — {elapsed:.1f}s", flush=True)
    return result.returncode


def main():
    banner(f"Rule 30 Experiment Runner  (TEST={TEST})")
    print(f"  Time: {datetime.datetime.now().isoformat()}")
    print(f"  Repo: {REPO}")
    print()

    print("  Telemetry tip — in a SEPARATE terminal run:")
    print(r"    nvidia-smi dmon -s pucvmet -d 5 -f data\gpu_telemetry.csv")
    print()
    print("  For the overnight 100M simulation (Experiment 5), run separately:")
    print(r"    C:\Python313\python.exe experiments\sim_200m.py")
    print(f"    (estimated runtime: {'~10s test' if TEST else '~9h for 100M steps'})")
    print()

    results = {}
    total_t0 = time.perf_counter()

    for name, label, progress_log in EXPERIMENTS:
        rc = run_experiment(name, label, progress_log)
        results[name] = rc
        print()

    total_elapsed = time.perf_counter() - total_t0
    banner("Summary")
    for name, label, _ in EXPERIMENTS:
        rc  = results.get(name, -1)
        sym = "PASS" if rc == 0 else ("?" if rc == -1 else "FAIL")
        print(f"  {sym}  {label}")
    print()
    print(f"  Total time: {total_elapsed:.1f}s")
    print()

    all_ok = all(rc == 0 for rc in results.values())
    if all_ok:
        print("  All experiments completed successfully.")
        print("  Results in data/ — plots in docs/plots/")
        print()
        print("  Next step: run the overnight 100M simulation:")
        print(r"    nvidia-smi dmon -s pucvmet -d 5 -f data\gpu_telemetry.csv")
        print(r"    C:\Python313\python.exe experiments\sim_200m.py")
    else:
        print("  Some experiments failed — check output above.")


if __name__ == "__main__":
    main()
