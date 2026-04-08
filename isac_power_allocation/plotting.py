"""Plotting helpers for Pareto and channel diagnostics."""

from __future__ import annotations

import os
from pathlib import Path

_MPL_CONFIG_DIR = Path(__file__).resolve().parents[1] / ".mplconfig"
_MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np

from .channels.models import ChannelState
from .objectives import OptimizationResult


def _sensing_plot_values(results: list[OptimizationResult]) -> tuple[list[float], str]:
    if not results:
        return [], "Sensing utility"
    metric_name = results[0].metrics.sensing_metric_name
    if metric_name == "detection_probability":
        return [result.metrics.sensing_detection_probability for result in results], "Detection probability"
    return [result.metrics.sensing_snr_db for result in results], "Sensing SNR (dB)"


def plot_pareto_front(
    weighted_results: list[OptimizationResult],
    nsga2_results: list[OptimizationResult],
    baseline_results: list[OptimizationResult],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    reference_results = weighted_results or nsga2_results or baseline_results
    _, y_label = _sensing_plot_values(reference_results)

    if weighted_results:
        sensing_values, _ = _sensing_plot_values(weighted_results)
        plt.plot(
            [result.metrics.communication_rate_bps_hz for result in weighted_results],
            sensing_values,
            marker="o",
            linewidth=2,
            label="GWO alpha sweep",
        )

    if nsga2_results:
        sensing_values, _ = _sensing_plot_values(nsga2_results)
        plt.scatter(
            [result.metrics.communication_rate_bps_hz for result in nsga2_results],
            sensing_values,
            s=30,
            alpha=0.75,
            label="NSGA-II front",
        )

    for result in baseline_results:
        sensing_values, _ = _sensing_plot_values([result])
        plt.scatter(
            result.metrics.communication_rate_bps_hz,
            sensing_values[0],
            marker="x",
            s=80,
            label=result.solver_name,
        )

    plt.xlabel("Communication rate (bps/Hz)")
    plt.ylabel(y_label)
    plt.title("ISAC Pareto Frontier")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_channel_snapshot(channel_state: ChannelState, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subcarriers = np.arange(channel_state.communication_gain.size)
    plt.figure(figsize=(8, 5))
    plt.plot(subcarriers, 10.0 * np.log10(np.maximum(channel_state.communication_gain, 1e-18)), label="Comm gain")
    plt.plot(subcarriers, 10.0 * np.log10(np.maximum(channel_state.sensing_gain, 1e-18)), label="Sense gain")
    plt.xlabel("Subcarrier index")
    plt.ylabel("Channel gain (dB)")
    plt.title(f"Channel Snapshot At t = {channel_state.time_s * 1e3:.2f} ms")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_algorithm_comparison(results: list[OptimizationResult], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = [result.solver_name for result in results]
    objectives = [result.metrics.weighted_objective for result in results]
    runtimes = [float(result.metadata.get("runtime_ms", 0.0)) for result in results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(names, objectives, color="#3a6ea5")
    axes[0].set_title("Weighted Objective")
    axes[0].set_ylabel("Objective value")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(names, runtimes, color="#d67229")
    axes[1].set_title("Runtime")
    axes[1].set_ylabel("Runtime (ms)")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_dynamic_algorithm_comparison(summary, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = [item.solver_name for item in summary.aggregates]
    mean_objectives = [item.mean_objective for item in summary.aggregates]
    std_objectives = [item.std_objective for item in summary.aggregates]
    mean_runtimes = [item.mean_runtime_ms for item in summary.aggregates]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(names, mean_objectives, yerr=std_objectives, capsize=4, color="#3a6ea5")
    axes[0].set_title("Dynamic Mean Objective")
    axes[0].set_ylabel("Mean weighted objective")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(names, mean_runtimes, color="#d67229")
    axes[1].set_title("Mean Runtime Per Step")
    axes[1].set_ylabel("Runtime (ms)")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_dynamic_objective_traces(summary, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    time_index = np.arange(summary.num_time_steps)
    for item in summary.aggregates:
        plt.plot(time_index, item.per_step_objective, linewidth=1.8, label=item.solver_name)

    plt.xlabel("Time step")
    plt.ylabel("Weighted objective")
    plt.title("Dynamic Objective Across CSI Updates")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
