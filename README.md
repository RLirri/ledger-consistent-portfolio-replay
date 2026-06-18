# Ledger-Consistent Portfolio Replay Framework


### Overview

This project provides a reproducible research framework for evaluating execution costs, implementation shortfall, tracking error, and capacity constraints in systematic portfolio rebalancing. The framework employs a ledger-consistent replay engine that explicitly models integer-share accounting, cash feasibility, transaction costs, participation-based trading limits, and carry-forward of unfilled orders.

The empirical study uses a diversified universe of U.S. exchange-traded funds (ETFs) and evaluates portfolio implementation under realistic execution assumptions across a long historical sample.

### Reproducibility

All figures, tables, and results reported in the paper are generated from a fixed configuration and automated experiment pipeline.

Key experiment settings are defined in:

```text
src/ledger_consistent_etf_trading/config.py
```

After configuring the experiment parameters, the complete workflow can be executed through:

```bash
python scripts/00_run_full_pipeline.py
```

The pipeline performs:

* Data acquisition
* Feature generation
* Panel construction
* Portfolio replay simulation
* Execution diagnostics
* Sensitivity analysis
* Capacity-stress evaluation
* Publication-figure generation

### Installation

Create the environment:

```bash
conda create -n etf-ledger python=3.12
conda activate etf-ledger
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Alternatively:

```bash
conda env create -f environment.yml
conda activate etf-ledger
```

### Repository Structure


```text
scripts/       End-to-end experiment pipeline
src/           Core framework implementation
data/          Downloaded and processed datasets
artifacts/     Generated outputs, diagnostics, scorecards, and figures
reports/       Time series figures 
```



### License

Released under the MIT License.
