# Ledger-consistent portfolio replay

Research software for **Counterfactual Evaluation of Execution Costs and Liquidity Constraints in Systematic Portfolio Rebalancing**.

The study asks when incorporating execution costs into portfolio-state evolution changes measured outcomes relative to a matched post-hoc cost adjustment. It compares weekly equal weight and daily momentum across initial capital levels, with controlled cost/liquidity, timing and execution-policy comparisons. Individual execution mechanisms are established methods; the contribution is matched computational evaluation and conditional empirical interpretation.

## Version and scope

Version 2.0.0 packages the scientific source used for the manuscript. The private experimental commit is 8e8096fcac68adc7b2f3f7784035369c5895c4f3. File-level hashes in [experimental_identity.json](docs/experimental_identity.json) establish correspondence despite different public release commit history. V1 remains at tag v1.0.0 and is not the implementation of the current study.

The design has 62 economic scenarios, 66 predefined contrasts and 12 primary comparisons. Historical prices, trade tapes, aggregate results and rendered empirical figures are excluded because redistribution rights have not been established. No manuscript acceptance, publication or indexing is claimed.

## Organization

- src/ledger_consistent_etf_trading/v2/: replay, validation, cohorts, costs, metrics, resampling, registry and budgeted execution.
- configs/v2/: scenario records and contrast specification.
- tests/: synthetic accounting, temporal, numerical and batch contracts, plus explicit snapshot-dependent checks.
- scripts/reproduction.py: operator-owned plans and the unchanged validated runner.
- scripts/analyze_saved_runs.py and statistical_recipes.py: saved-path analysis and fixed estimators.
- scripts/render_publication.py: saved-evidence figure/table rendering.
- docs/: model, inputs, provenance and limitations.

Non-V2 modules and numbered scripts are retained V1 reference code. They are not the V2 workflow. Older acquisition scripts require legacy helper/provider modules absent from the original public snapshot; use the documented compatible-input interface instead.

## Installation

Use Python 3.11, Git and a POSIX system. The batch runner uses Unix resource controls.

    python -m venv .venv
    . .venv/bin/activate
    python -m pip install -r requirements.txt
    python -m pip install --no-build-isolation -e .

Dependencies pin the tested direct scientific and figure packages, not a complete binary lock. Original experiments used Python 3.11.11 on macOS ARM64. Another platform or snapshot has a new identity.

## Reproduction and tests

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) separates synthetic verification, rendering from lawfully held saved outputs and historical replay.

    python -B -m unittest discover -s tests -t . -v
    python -B scripts/reproduction.py registry
    python -B scripts/verify_source.py

Tests use synthetic markets. Snapshot/private-record checks explicitly skip when inputs are absent. No command above downloads data or executes the historical grid.

Inputs must be lawfully obtained. See [data schema](docs/DATA.md). A fresh download cannot be claimed to reproduce the preserved snapshot exactly. Historical execution requires an explicit new operator plan, budget and data-rights acknowledgement, without inheriting the author's earlier permission.

## Results and figures

Plotting code and shared style are included, without an empirical result bundle. Regeneration requires locally held canonical analysis and prepared saved diagnostics. The wrapper writes to a new directory without replay. Plot values are exported for verification; fonts are not distributed.

## Interpretation limits

Costs and ADV budgets are scenario assumptions, not calibrated institutional capacity. Price returns omit ordinary distributions; fractional adjusted units are not historical shares. Daily-bar batch execution, a retrospective universe, two strategies and one market path limit generalization. Uncapped modeled notional can exceed lagged ADV and normalized impact saturates. Completion uses each replay's own demand. Nominal pointwise bootstrap intervals have unvalidated stationarity/coverage. Lifecycle shortfall endpoints differ across policies.

## Citation and license

Use [CITATION.cff](CITATION.cff) for software citation. The related manuscript is unpublished; no paper/software DOI is assigned here. The existing [MIT license](LICENSE) is preserved and grants no rights to third-party data, papers or fonts. See [provenance](docs/PROVENANCE.md).
