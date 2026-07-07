"""Train an XGBoost surrogate model from the exported GWO dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isac_power_allocation.ml.xgboost_training import (
    DEFAULT_TARGET,
    save_training_summary,
    train_xgboost_surrogate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost surrogate model on exported GWO results."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("outputs") / "gwo_xgboost_dataset.csv",
        help="CSV dataset produced by scripts/export_gwo_dataset.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "xgboost",
        help="Directory for model, metrics, feature metadata, and predictions.",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="Regression target column to learn.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.25,
        help="Fraction of rows held out for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split and model training.",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_xgboost_surrogate(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        target_column=args.target,
        test_fraction=args.test_fraction,
        random_seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )
    metrics_path = save_training_summary(
        summary,
        args.output_dir / f"xgboost_{args.target}_metrics.json",
    )

    print(f"Target: {summary.target_column}")
    print(f"Rows: {summary.row_count} | Features: {summary.feature_count}")
    print(f"Train/Test: {summary.train_count}/{summary.test_count}")
    print(f"MAE: {summary.mae:.6f}")
    print(f"RMSE: {summary.rmse:.6f}")
    print(f"R2: {summary.r2:.6f}")
    print(f"Saved model: {summary.model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved predictions: {summary.prediction_path}")


if __name__ == "__main__":
    main()
