# Reproducibility

Run from a full Git checkout after installation. Keep private inputs/plans/outputs in ignored local/ directories. No market data or result bundle is included.

## 1. Synthetic contracts

    python -B -m unittest discover -s tests -t . -v
    python -B scripts/verify_source.py
    python -B scripts/reproduction.py registry

The registry has 62 economic scenarios plus a pilot; enumeration performs no replay. The V1 characterization check requires history including commit 852d46f; use a full clone. Snapshot/private-record checks skip when unavailable. Tests contain deliberate child-process crash, CPU, memory and timeout fixtures: use an isolated local environment. Public packaging tests separately check source correspondence and authorization refusal.

## 2. Saved-output rendering

If you lawfully hold the compatible evidence bundle:

    python -B scripts/render_publication.py --evidence-root local/evidence --diagnostics local/publication-diagnostics --out local/rendered-F2 --figure F2 --tables

Expected: vector PDF/SVG, directly rendered 600/1200-DPI PNG, plot-value JSON, source manifests and five LaTeX table sources. Existing output directories are refused. Use --figure all for the complete collection, with higher memory/disk requirements. Optional --font path/to/licensed-font.ttf registers a locally held font. No font files are included; substitution prevents exact visual byte identity.

Evidence root requires experiments/v2_grid/analysis/{scorecard,contrasts,primary_contrasts,factorials}.json and FIGURE_CONTRACTS.json. Diagnostics requires data/ and manifests/evidence_verification.json. Panels also consume prepared dynamic, fill, duration and mechanism-scorecard files; exact names are in the generator. This is rendering from prepared diagnostics, not automatic preparation from arbitrary new data.

For a complete compatible set of 62 validated saved runs and six post-hoc paths:

    python -B scripts/analyze_saved_runs.py --runs-root local/saved-grid --out local/new-analysis

Required children: runs/ with complete manifests, tapes and completion markers; derived/ with six matched post-hoc CSV/metadata pairs. It verifies hashes, common identities/calendar and all registered cells, then recomputes the fixed statistics, including bootstrap. Output must be new. CPU is capped at 240 seconds with a polling memory watchdog, not a kernel-enforced memory ceiling. This historical statistical computation was not repeated in release preparation; synthetic recipe checks are separate.

## 3. Historical replay

Acquire compatible data under the provider's terms. See [DATA.md](docs/DATA.md). Matching original hashes are required for exact numerical reproduction. A different lawful snapshot is a new result, not a replacement silently attributed to the paper.

Create local/budget.json with finite positive cpu_seconds_total, integer cpu_seconds_per_scenario, parent_cpu_seconds_per_scenario, wall_seconds_per_scenario and rss_mb. An illustrative one-scenario budget is120totalCPU,30childCPU,20parentCPU,90wallseconds and1500monitoredMiB. These are operator choices, not inherited author approval or a guaranteed grid budget.

    python -B scripts/reproduction.py plan --snapshot local/snapshot --out local/replay --plan local/operator-plan --budget local/budget.json --scenario E2_A_10m_cost0_cap0 --owner "local operator" --acknowledge-data-rights

Planning validates the snapshot and binds clean Git commit, model/config hashes, data identity/input hashes, environment, selected scenarios, output root and one attempt each. It runs no scenario. Explicit execution is separate:

    python -B scripts/reproduction.py execute --plan local/operator-plan --execute-local-plan

The plan is a new local operator record. Do not copy private historical author records, change validation to pass a gate, reset existing ledgers or reuse consumed attempt allowances. The unchanged runner performs reservations, accounting, fail-closed validation/readback and immutable publication. Release preparation does not execute the historical command.

## Coverage limits

The package verifies synthetic scientific contracts and can render locally held saved outputs. Exact public historical reproduction remains conditional on unavailable redistribution-cleared inputs and was not rerun here. Public run identities differ from private experiment identities because code/design/environment bindings include release packaging; file hashes establish scientific-source correspondence.

V1 numbered acquisition scripts remain reference only and have incomplete original helper/dependency coverage. The V2 input loader and schema are the supported interface. Historical acquisition and preparation of full plotting diagnostics remain explicit prerequisites.
