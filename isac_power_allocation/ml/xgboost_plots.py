"""Plot XGBoost surrogate diagnostics."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

_MPL_CONFIG_DIR = Path(__file__).resolve().parents[2] / ".mplconfig"
_MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np


def plot_xgboost_diagnostics(
    model_path: str | Path,
    prediction_path: str | Path,
    feature_metadata_path: str | Path,
    output_dir: str | Path,
    top_n_features: int = 20,
) -> dict[str, Path]:
    """Create predicted-vs-actual and feature-importance figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predicted_vs_actual_path = output_dir / "xgboost_predicted_vs_actual.png"
    feature_importance_path = output_dir / "xgboost_feature_importance.png"

    plot_predicted_vs_actual(prediction_path, predicted_vs_actual_path)
    plot_feature_importance(
        model_path=model_path,
        feature_metadata_path=feature_metadata_path,
        output_path=feature_importance_path,
        top_n_features=top_n_features,
    )

    return {
        "predicted_vs_actual": predicted_vs_actual_path,
        "feature_importance": feature_importance_path,
    }


def plot_predicted_vs_actual(
    prediction_path: str | Path,
    output_path: str | Path,
) -> None:
    prediction_path = Path(prediction_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    actual, predicted = _read_prediction_values(prediction_path)
    lower = float(min(np.min(actual), np.min(predicted)))
    upper = float(max(np.max(actual), np.max(predicted)))
    padding = 0.05 * max(upper - lower, 1.0)
    axis_limits = (lower - padding, upper + padding)

    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    r2 = _r2_score(actual, predicted)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(actual, predicted, s=46, color="#3a6ea5", alpha=0.82, edgecolor="white", linewidth=0.6)
    ax.plot(axis_limits, axis_limits, color="#d67229", linewidth=2.0, label="Ideal prediction")
    ax.set_xlim(axis_limits)
    ax.set_ylim(axis_limits)
    ax.set_xlabel("Actual weighted objective")
    ax.set_ylabel("Predicted weighted objective")
    ax.set_title("XGBoost Predicted vs Actual")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.text(
        0.98,
        0.04,
        f"MAE = {mae:.3f}\nRMSE = {rmse:.3f}\nR2 = {r2:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_feature_importance(
    model_path: str | Path,
    feature_metadata_path: str | Path,
    output_path: str | Path,
    top_n_features: int = 20,
) -> None:
    try:
        import xgboost as xgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xgboost is required for feature-importance plotting. Install dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from exc

    model_path = Path(model_path)
    feature_metadata_path = Path(feature_metadata_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    booster = xgb.Booster()
    booster.load_model(model_path)
    scores = booster.get_score(importance_type="gain")

    metadata = json.loads(feature_metadata_path.read_text(encoding="utf-8"))
    feature_names = metadata.get("feature_names", [])
    importances = _ordered_importances(scores, feature_names)
    if not importances:
        raise ValueError(f"No feature-importance scores found in model: {model_path}")

    top_features = importances[: max(1, top_n_features)]
    labels = [_pretty_feature_name(name) for name, _ in reversed(top_features)]
    values = [score for _, score in reversed(top_features)]

    height = max(5.5, 0.32 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(8.5, height))
    ax.barh(labels, values, color="#3a6ea5")
    ax.set_xlabel("Average gain")
    ax.set_title("XGBoost Feature Importance")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _read_prediction_values(prediction_path: Path) -> tuple[np.ndarray, np.ndarray]:
    actual: list[float] = []
    predicted: list[float] = []
    with prediction_path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            actual.append(float(row["actual"]))
            predicted.append(float(row["predicted"]))
    if not actual:
        raise ValueError(f"No prediction rows found in {prediction_path}")
    return np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)


def _r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum(np.square(predicted - actual)))
    total = float(np.sum(np.square(actual - np.mean(actual))))
    return float(1.0 - residual / total) if total > 0.0 else 0.0


def _ordered_importances(
    scores: dict[str, float],
    feature_names: list[str],
) -> list[tuple[str, float]]:
    if any(name in scores for name in feature_names):
        items = [(name, float(scores.get(name, 0.0))) for name in feature_names]
    else:
        items = [
            (feature_names[int(name[1:])], float(score))
            for name, score in scores.items()
            if name.startswith("f") and name[1:].isdigit() and int(name[1:]) < len(feature_names)
        ]
    return sorted((item for item in items if item[1] > 0.0), key=lambda item: item[1], reverse=True)


def _pretty_feature_name(name: str) -> str:
    replacements = {
        "_": " ",
        "communication": "comm",
        "sensing": "sense",
        "subcarrier": "subcarrier",
    }
    pretty = name
    for old, new in replacements.items():
        pretty = pretty.replace(old, new)
    return pretty
