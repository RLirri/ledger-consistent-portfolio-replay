"""Bounded legacy characterization: rerun the untouched legacy engine on the pinned snapshot with the
baseline weekly configuration and compare with the saved artifacts. Writes a report; never writes
into the snapshot. Usage: v2_legacy_reproduction.py <snapshot> <out_dir> <author_decisions.json>"""
import hashlib, json, pathlib, resource, sys, time
import numpy as np, pandas as pd
import ledger_consistent_etf_trading as pkg
from ledger_consistent_etf_trading.execution.capacity import CapacityConfig
from ledger_consistent_etf_trading.execution.cost_model import CostConfig
from ledger_consistent_etf_trading.replay.replay_engine import ReplayConfig, run_replay

snap, out, auth = map(pathlib.Path, sys.argv[1:4])
# The Sprint 3 authorization (the name "legacy_reproduction" in `authorized_runs`) was used once and is spent.
# A further legacy run needs a fresh, scoped entry in the execution authorization: bound to this checkout's
# commit and to one output directory, which must not exist yet (so it can be used exactly once).
import subprocess
_block = json.loads(auth.read_text()).get("execution_authorization", {}).get("legacy_reproduction")
_head = subprocess.check_output(["git", "-C", str(pathlib.Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True).strip()
if not isinstance(_block, dict) or _block.get("authorized") is not True:
    raise SystemExit("legacy reproduction has no fresh scoped authorization (execution_authorization.legacy_reproduction); refusing")
if _block.get("code_commit") != _head or pathlib.Path(_block.get("output_dir", "")).resolve() != out.resolve():
    raise SystemExit("legacy reproduction authorization is bound to a different commit or output directory; refusing")
if out.exists():
    raise SystemExit("legacy reproduction output directory already exists; the authorization has been used")
out.mkdir(parents=True, exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
inputs = {k: snap / k for k in ("data/processed/panel_1d_aligned.parquet", "artifacts/weights_equal_weight_weekly.csv",
                                 "artifacts/equity_equal_weight_weekly.csv", "artifacts/fills_equal_weight_weekly.csv",
                                 "artifacts/positions_equal_weight_weekly.csv")}
panel = pd.read_parquet(inputs["data/processed/panel_1d_aligned.parquet"])
weights = pd.read_csv(inputs["artifacts/weights_equal_weight_weekly.csv"])
t0 = time.process_time()
fills, equity, pos = run_replay(panel, weights, CapacityConfig(max_participation=0.05, max_trade_notional=250_000, carry_unfilled=True),
                                CostConfig(fees_bps=0.0, half_spread_bps=0.5, slippage_a_bps=15.0, slippage_b=0.7, slippage_c=1.0, vol_ref=0.01),
                                ReplayConfig(initial_cash=1_000_000.0, price_convention="next_open", use_close_for_mtm=True, strict_cash=True, min_trade_notional=1_000.0))
cpu = time.process_time() - t0
se = pd.read_csv(inputs["artifacts/equity_equal_weight_weekly.csv"], parse_dates=["date"]).set_index("date")
sf = pd.read_csv(inputs["artifacts/fills_equal_weight_weekly.csv"], parse_dates=["date"])
sp = pd.read_csv(inputs["artifacts/positions_equal_weight_weekly.csv"], parse_dates=["date"]).set_index("date")
rel = (equity["equity"].to_numpy() - se["equity"].to_numpy()) / se["equity"].to_numpy()
rep = {"legacy_module": pkg.__file__, "replay_cpu_seconds": cpu, "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576,
       "sessions": int(len(equity)), "sessions_saved": int(len(se)), "dates_equal": bool(equity.index.equals(se.index)),
       "nav_max_abs_relative_difference": float(np.abs(rel).max()), "nav_end_rerun": float(equity["equity"].iloc[-1]), "nav_end_saved": float(se["equity"].iloc[-1]),
       "cash_max_abs_difference": float(np.abs(equity["cash"].to_numpy() - se["cash"].to_numpy()).max()),
       "fills_rerun": int(len(fills)), "fills_saved": int(len(sf)),
       "fill_shares_identical": bool(len(fills) == len(sf) and (fills["shares_delta"].to_numpy() == sf["shares_delta"].to_numpy()).all() and (fills["ticker"].to_numpy() == sf["ticker"].to_numpy()).all()),
       "total_cost_rerun": float(fills["total_cost_$"].sum()), "total_cost_saved": float(sf["total_cost_$"].sum()),
       "positions_identical": bool((pos.reindex(columns=sp.columns).fillna(0).to_numpy() == sp.to_numpy()).all()),
       "input_sha256": {k: sha(v) for k, v in inputs.items()}}
# Sprint 3C (F11): persist both full rerun tapes and compare every field, not only shares and tickers.
fills.to_csv(out / "fills_rerun.csv", index=False)
pos.to_csv(out / "positions_rerun.csv")
equity.to_csv(out / "equity_rerun.csv")
TOL = 1e-9
field_diff = {}
if len(fills) == len(sf):
    rf = fills.reset_index(drop=True); rf["date"] = pd.to_datetime(rf["date"], utc=True); ss = sf.reset_index(drop=True); ss["date"] = pd.to_datetime(ss["date"], utc=True)
    for c in ss.columns:
        if c not in rf.columns:
            field_diff[c] = "missing in rerun"
        elif pd.api.types.is_numeric_dtype(ss[c]) and not pd.api.types.is_bool_dtype(ss[c]):
            field_diff[c] = float(np.nanmax(np.abs(rf[c].to_numpy(float) - ss[c].to_numpy(float)) / np.maximum(1.0, np.abs(ss[c].to_numpy(float)))))
        else:
            field_diff[c] = 0.0 if (rf[c].to_numpy() == ss[c].to_numpy()).all() else "values differ"
else:
    field_diff["row_count"] = f"{len(fills)} rerun vs {len(sf)} saved"
rep["fill_fields_max_relative_difference"] = field_diff
rep["all_fill_fields_within_tolerance"] = bool(field_diff and all(isinstance(v, float) and v <= TOL for v in field_diff.values()))
rep["comparison_tolerance_relative"] = TOL
rep["peak_rss_unit"] = "MiB (bytes / 2**20), process high-water mark"
rep["reproduced_to_1e-6_relative"] = bool(rep["dates_equal"] and rep["nav_max_abs_relative_difference"] < 1e-6 and rep["fill_shares_identical"])
rep["reproduced_all_fields"] = bool(rep["reproduced_to_1e-6_relative"] and rep["all_fill_fields_within_tolerance"] and rep["positions_identical"])
(out / "legacy_reproduction.json").write_text(json.dumps(rep, indent=1) + "\n")
print(json.dumps(rep, indent=1))
sys.exit(0 if rep["reproduced_all_fields"] else 1)
