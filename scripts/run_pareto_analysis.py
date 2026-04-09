"""Run the default detection-probability and waveform-aware Pareto study."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isac_power_allocation.config import build_default_experiment_config
from isac_power_allocation.experiments.runner import run_pareto_study
from isac_power_allocation.plotting import plot_channel_snapshot, plot_pareto_front


def main() -> None:
    config = build_default_experiment_config(scenario_name="UMi_NLOS")
    summary, snapshot = run_pareto_study(config)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    pareto_path = output_dir / "pareto_front.png"
    channel_path = output_dir / "channel_snapshot.png"
    summary_path = output_dir / "pareto_summary.json"

    plot_pareto_front(
        weighted_results=summary.weighted_results,
        nsga2_results=summary.nsga2_results,
        baseline_results=summary.baseline_results,
        output_path=pareto_path,
    )
    plot_channel_snapshot(snapshot, channel_path)
    summary.save_json(summary_path)

    print(f"Scenario: {summary.scenario_name}")
    print(f"Snapshot index: {summary.snapshot_index} / {summary.channel_sequence_length - 1}")
    print(f"GWO Pareto points: {len(summary.weighted_results)}")
    print(f"NSGA-II Pareto points: {len(summary.nsga2_results)}")
    print(f"Saved: {pareto_path}")
    print(f"Saved: {channel_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
