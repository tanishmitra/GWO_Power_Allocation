"""Plot XGBoost surrogate diagnostics from saved training artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isac_power_allocation.ml.xgboost_plots import plot_xgboost_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create predicted-vs-actual and feature-importance plots for the XGBoost surrogate."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("outputs") / "xgboost" / "xgboost_weighted_objective.json",
        help="Saved XGBoost model JSON.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs") / "xgboost" / "xgboost_weighted_objective_features.json",
        help="Feature metadata JSON saved during training.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs") / "xgboost" / "xgboost_weighted_objective_predictions.csv",
        help="Prediction CSV saved during training.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "xgboost",
        help="Directory to write plots.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top features to show in the importance plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = plot_xgboost_diagnostics(
        model_path=args.model,
        prediction_path=args.predictions,
        feature_metadata_path=args.features,
        output_dir=args.output_dir,
        top_n_features=args.top_n,
    )
    print(f"Saved predicted-vs-actual plot: {paths['predicted_vs_actual']}")
    print(f"Saved feature-importance plot: {paths['feature_importance']}")


if __name__ == "__main__":
    main()
