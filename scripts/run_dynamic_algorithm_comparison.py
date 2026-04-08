"""Run the dynamic multi-step optimizer comparison across time-varying CSI."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isac_power_allocation.config import build_default_experiment_config
from isac_power_allocation.experiments.runner import run_dynamic_algorithm_comparison
from isac_power_allocation.plotting import (
    plot_dynamic_algorithm_comparison,
    plot_dynamic_objective_traces,
)


def main() -> None:
    config = build_default_experiment_config(scenario_name="UMi_NLOS")
    summary = run_dynamic_algorithm_comparison(config)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    summary_path = output_dir / "dynamic_algorithm_comparison_summary.json"
    ranking_plot_path = output_dir / "dynamic_algorithm_comparison.png"
    trace_plot_path = output_dir / "dynamic_objective_traces.png"

    summary.save_json(summary_path)
    plot_dynamic_algorithm_comparison(summary, ranking_plot_path)
    plot_dynamic_objective_traces(summary, trace_plot_path)

    print(f"Scenario: {summary.scenario_name}")
    print(f"Time steps: {summary.num_time_steps}")
    print(f"Alpha: {summary.alpha}")
    print("")
    print("Algorithm | Mean Obj | Std Obj | Mean Rate | Mean SNR (dB) | Mean Runtime (ms)")
    print("-" * 84)
    for item in summary.aggregates:
        print(
            f"{item.solver_name:9s} | "
            f"{item.mean_objective:8.3f} | "
            f"{item.std_objective:7.3f} | "
            f"{item.mean_rate_bps_hz:9.3f} | "
            f"{item.mean_sensing_snr_db:13.3f} | "
            f"{item.mean_runtime_ms:16.3f}"
        )

    print("")
    print(f"Saved: {summary_path}")
    print(f"Saved: {ranking_plot_path}")
    print(f"Saved: {trace_plot_path}")


if __name__ == "__main__":
    main()
