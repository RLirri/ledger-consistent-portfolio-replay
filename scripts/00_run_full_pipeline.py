from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "01_fetch_data.py",
    "02_build_features.py",
    "03_build_aligned_panel.py",
    "04b_generate_baseline_weights_weekly.py",
    "05b_run_replay_equal_weight_weekly.py",
    "06b_build_scorecard_equal_weight_weekly.py",
    "07_execution_diagnostics.py",
    "08_sensitivity_grid_weekly.py",
    "09_capacity_stress_weekly.py",
    "10_make_capacity_figures.py",
    "11_make_regime_table.py",
    "12_make_timeseries_figures.py",
    "13_make_cost_decomposition_figures.py",
    "14_make_sensitivity_figures.py",
    "15_make_implementation_shortfall_figures.py",
    "16_make_is_vs_intensity_figures.py",
    "17_make_equity_curve_pub.py",
    "18_make_capacity_frontier_figure.py",
]


def main():
    scripts_dir = Path(__file__).resolve().parent

    for script in SCRIPTS:
        script_path = scripts_dir / script

        if not script_path.exists():
            raise FileNotFoundError(f"Missing script: {script_path}")

        print("\n" + "=" * 80)
        print(f"Running: {script}")
        print("=" * 80)

        subprocess.run([sys.executable, str(script_path)], check=True)

    print("\nFull pipeline completed successfully.")


if __name__ == "__main__":
    main()