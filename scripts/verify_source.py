"""Verify released scientific Python against the experimental source hash map."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
identity = json.loads((root / "docs/experimental_identity.json").read_text())
failed = []
for name, expected in identity["exact_source_sha256"].items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failed.append(name)
if failed:
    raise SystemExit("scientific source mismatch: " + ", ".join(failed))
print(
    "Verified", len(identity["exact_source_sha256"]), "exact experimental Python files"
)
