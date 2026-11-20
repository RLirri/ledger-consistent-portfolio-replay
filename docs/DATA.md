# Data requirements and rights

No historical market data or empirical result bundle is included. Original provider/version and redistribution rights are insufficiently established to license these inputs here. MIT code licensing does not license returned data.

OpenBB is a connector, not a data-rights grant. Its [provider documentation](https://docs.openbb.co/odp/python/extensions/providers) states that source services set usage terms. New lawful sources may support another study; they are not exact replacements for the preserved snapshot. Do not infer adjustment semantics from column names or silently use changed provider defaults.

## Snapshot

ROOT/data/processed/TICKER_1d_clean.parquet must exist for each fixed universe asset. Timestamp index; numeric open, close and volume columns. Additional OHLCV columns may be present. The loader converts to UTC, sorts and aligns the common calendar. Keep complete warm-up history. Production requires finite positive open/close and finite nonnegative volume across all observations read.

The preserved study evaluates31 March 2010 through15 June 2026 with zero calendar loss. A new snapshot must independently verify calendar identity, adjustment and volume conventions, corporate actions and price-return meaning. Hashes record actual inputs, not licensing or accuracy.

## Saved results

Validated directories contain daily NAV/cash/trades/costs, order/parent/demand tapes, positions, targets, metrics, validation, manifest and final completion marker. The runner validates before and after readback. Post-hoc paths bind reference-run identity and file hashes. Analysis preserves scenario/run/comparator identities and statistical records.

Aggregation alone does not establish redistribution permission. No raw data, processed inputs, saved tapes, aggregate outcome bundle or empirical chart is added to this release. Private approvals, correspondence and unpublished manuscripts are excluded.
