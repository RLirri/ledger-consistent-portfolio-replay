"""Write the experiment configuration registry (no execution)."""
import json, pathlib, sys
from ledger_consistent_etf_trading.v2 import SPEC_VERSION, registry

out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[1] / "configs/v2/registry.json"
recs = registry.as_records()
out.write_text(json.dumps({"spec_version": SPEC_VERSION, "count": len(recs), "scenarios": recs}, indent=1) + "\n")
by = {}
for r in recs:
    by[r["experiment"]] = by.get(r["experiment"], 0) + 1
print(out, by, "total", len(recs))
