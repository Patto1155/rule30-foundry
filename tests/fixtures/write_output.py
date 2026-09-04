"""Writes a file outside runs/, the way a real experiment does.

Fixture for the regression that a run's declared outputs are preserved in
the run directory instead of being left untracked.
"""
import json
from pathlib import Path

out = Path(__file__).resolve().parents[2] / "data" / "wedge" / "workhorse_test_output.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")
print(json.dumps({"horizon": 1, "metrics": {}, "conclusions": []}))
