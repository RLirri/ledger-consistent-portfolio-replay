"""Single-scenario entry point. Since Sprint 3C it is a thin wrapper over the batch runner, so a single
run gets the same authorization, budget, validation and atomic-completion safeguards.

  v2_run.py --scenario pilot_252 --snapshot <dir> --out <dir> --authorization <author_decisions.json> --design-fingerprint <json>

The Sprint 3 pilot authorization (a scenario name in `authorized_runs`) was used once and does not
satisfy the batch runner; a new `execution_authorization` block is required.
"""
import pathlib, runpy, sys
sys.argv[0] = str(pathlib.Path(__file__).with_name("v2_batch.py"))
runpy.run_path(sys.argv[0], run_name="__main__")
