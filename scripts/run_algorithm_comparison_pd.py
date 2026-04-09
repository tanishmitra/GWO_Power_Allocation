"""Run algorithm comparison in probabilistic-detection mode."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isac_power_allocation.config import build_default_experiment_config
from isac_power_allocation.experiments.runner import run_algorithm_comparison
from isac_power_allocation.plotting import plot_algorithm_comparison, plot_channel_snapshot


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ISAC algorithm comparison in probabilistic-detection mode.")
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
        dest="waveform",
        action="store_true",
        help="Enable joint power + waveform co-optimization (default: on).",
    )
    parser.add_argument(
        "--no-waveform",
        dest="waveform",
        action="store_false",
        help="Disable joint power + waveform co-optimization.",
    )
    parser.set_defaults(waveform=True)
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

    summary, snapshot = run_algorithm_comparison(config)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    suffix = "_pd_wf" if args.waveform else "_pd"

    summary_path = output_dir / f"algorithm_comparison{suffix}_summary.json"
    plot_path = output_dir / f"algorithm_comparison{suffix}.png"
    channel_path = output_dir / f"comparison_channel_snapshot{suffix}.png"

    summary.save_json(summary_path)
    plot_algorithm_comparison(summary.results, plot_path)
    plot_channel_snapshot(snapshot, channel_path)

    print(f"Scenario: {summary.scenario_name}")
    print(f"Snapshot index: {summary.snapshot_index} / {summary.channel_sequence_length - 1}")
    print(f"Alpha: {summary.alpha}")
    print(f"Pfa: {args.pfa}")
    print(f"Integration gain: {args.integration_gain}")
    print(f"Waveform co-optimization: {args.waveform}")
    print("")
    print("Algorithm | Rate (bps/Hz) | Pd | SNR (dB) | Objective | Runtime (ms)")
    print("-" * 86)
    for result in summary.results:
        runtime_ms = float(result.metadata.get("runtime_ms", 0.0))
        print(
            f"{result.solver_name:9s} | "
            f"{result.metrics.communication_rate_bps_hz:13.3f} | "
            f"{result.metrics.sensing_detection_probability:4.3f} | "
            f"{result.metrics.sensing_snr_db:8.3f} | "
            f"{result.metrics.weighted_objective:9.3f} | "
            f"{runtime_ms:11.3f}"
        )

    print("")
    print(f"Saved: {summary_path}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {channel_path}")


if __name__ == "__main__":
    main()
