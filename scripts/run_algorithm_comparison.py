"""Run the default detection-probability and waveform-aware optimizer comparison."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isac_power_allocation.config import build_default_experiment_config
from isac_power_allocation.experiments.runner import run_algorithm_comparison
from isac_power_allocation.plotting import plot_algorithm_comparison, plot_channel_snapshot


def main() -> None:
    config = build_default_experiment_config(scenario_name="UMi_NLOS")
    summary, snapshot = run_algorithm_comparison(config)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    summary_path = output_dir / "algorithm_comparison_summary.json"
    plot_path = output_dir / "algorithm_comparison.png"
    channel_path = output_dir / "comparison_channel_snapshot.png"

    summary.save_json(summary_path)
    plot_algorithm_comparison(summary.results, plot_path)
    plot_channel_snapshot(snapshot, channel_path)

    print(f"Scenario: {summary.scenario_name}")
    print(f"Snapshot index: {summary.snapshot_index} / {summary.channel_sequence_length - 1}")
    print(f"Alpha: {summary.alpha}")
    print("")
    print("Algorithm | Rate (bps/Hz) | Pd | SNR (dB) | Objective | Runtime (ms)")
    print("-" * 86)
    for result in summary.results:
        runtime_ms = float(result.metadata.get('runtime_ms', 0.0))
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
