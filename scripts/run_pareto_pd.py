"""Run Pareto analysis in probabilistic-detection mode."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isac_power_allocation.config import build_default_experiment_config
from isac_power_allocation.experiments.runner import run_pareto_study
from isac_power_allocation.plotting import plot_channel_snapshot, plot_pareto_front


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ISAC Pareto analysis in probabilistic-detection mode.")
    parser.add_argument("--scenario", default="UMi_NLOS", help="Scenario name (default: UMi_NLOS).")
    parser.add_argument(
        "--pfa",
        type=float,
        default=1e-3,
        help="False alarm probability for detection metric (default: 1e-3).",
    )
    parser.add_argument(
        "--integration-gain",
        type=float,
        default=0.02,
        help="Detection integration gain (default: 0.02).",
    )
    parser.add_argument(
        "--waveform",
        action="store_true",
        help="Enable joint power + waveform co-optimization.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = build_default_experiment_config(scenario_name=args.scenario)
    config = replace(
        config,
        objective=replace(
            config.objective,
            sensing_metric="detection_probability",
            detection_false_alarm_probability=float(args.pfa),
            detection_integration_gain=float(args.integration_gain),
            waveform_co_optimization=bool(args.waveform),
        ),
    )

    summary, snapshot = run_pareto_study(config)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    suffix = "_pd_wf" if args.waveform else "_pd"

    pareto_path = output_dir / f"pareto_front{suffix}.png"
    channel_path = output_dir / f"channel_snapshot{suffix}.png"
    summary_path = output_dir / f"pareto_summary{suffix}.json"

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
    print(f"Pfa: {args.pfa}")
    print(f"Integration gain: {args.integration_gain}")
    print(f"Waveform co-optimization: {args.waveform}")
    print(f"GWO Pareto points: {len(summary.weighted_results)}")
    print(f"NSGA-II Pareto points: {len(summary.nsga2_results)}")
    print(f"Saved: {pareto_path}")
    print(f"Saved: {channel_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
