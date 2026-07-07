"""Train XGBoost surrogate models from exported GWO CSV datasets."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


DEFAULT_TARGET = "weighted_objective"
DEFAULT_EXCLUDED_PREFIXES = (
    "optimal_power_",
    "waveform_",
)
DEFAULT_EXCLUDED_COLUMNS = {
    "solver_name",
    "runtime_ms",
    "communication_rate_bps_hz",
    "sensing_snr_linear",
    "sensing_snr_db",
    "sensing_detection_probability",
    "sensing_utility",
    "sensing_metric_name",
    "weighted_objective",
    "history_start_objective",
    "history_end_objective",
    "history_best_objective",
    "history_improvement",
}
CATEGORICAL_COLUMNS = ("scenario_name", "sensing_metric")


@dataclass(frozen=True)
class XGBoostTrainingSummary:
    dataset_path: str
    model_path: str
    feature_metadata_path: str
    prediction_path: str
    target_column: str
    row_count: int
    feature_count: int
    train_count: int
    test_count: int
    random_seed: int
    test_fraction: float
    mae: float
    rmse: float
    r2: float
    target_mean: float
    target_std: float


def train_xgboost_surrogate(
    dataset_path: str | Path,
    output_dir: str | Path,
    target_column: str = DEFAULT_TARGET,
    test_fraction: float = 0.25,
    random_seed: int = 42,
    n_estimators: int = 300,
    max_depth: int = 3,
    learning_rate: float = 0.05,
) -> XGBoostTrainingSummary:
    """Train an XGBoost regressor and save model plus evaluation artifacts."""
    try:
        import xgboost as xgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xgboost is required for Step 2. Install project dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from exc

    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(dataset_path)
    if not rows:
        raise ValueError(f"No rows found in dataset: {dataset_path}")
    if target_column not in rows[0]:
        raise ValueError(f"Target column '{target_column}' is not present in {dataset_path}.")

    feature_names, x_matrix, y_vector = _build_matrix(rows, target_column)
    train_indices, test_indices = _split_indices(
        row_count=len(rows),
        test_fraction=test_fraction,
        random_seed=random_seed,
    )

    model_params = {
        "objective": "reg:squarederror",
        "max_depth": max_depth,
        "eta": learning_rate,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "seed": random_seed,
        "nthread": 1,
    }
    train_matrix = xgb.DMatrix(
        x_matrix[train_indices],
        label=y_vector[train_indices],
        feature_names=feature_names,
    )
    test_matrix = xgb.DMatrix(x_matrix[test_indices], feature_names=feature_names)
    model = xgb.train(
        params=model_params,
        dtrain=train_matrix,
        num_boost_round=n_estimators,
    )

    predictions = model.predict(test_matrix)
    metrics = _regression_metrics(y_true=y_vector[test_indices], y_pred=predictions)

    safe_target = _safe_name(target_column)
    model_path = output_dir / f"xgboost_{safe_target}.json"
    feature_metadata_path = output_dir / f"xgboost_{safe_target}_features.json"
    prediction_path = output_dir / f"xgboost_{safe_target}_predictions.csv"

    model.save_model(model_path)
    _save_feature_metadata(
        feature_metadata_path,
        target_column=target_column,
        feature_names=feature_names,
        model_params={**model_params, "num_boost_round": n_estimators},
    )
    _save_predictions(
        prediction_path=prediction_path,
        rows=rows,
        test_indices=test_indices,
        y_true=y_vector[test_indices],
        y_pred=predictions,
    )

    return XGBoostTrainingSummary(
        dataset_path=str(dataset_path),
        model_path=str(model_path),
        feature_metadata_path=str(feature_metadata_path),
        prediction_path=str(prediction_path),
        target_column=target_column,
        row_count=len(rows),
        feature_count=len(feature_names),
        train_count=len(train_indices),
        test_count=len(test_indices),
        random_seed=random_seed,
        test_fraction=test_fraction,
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        r2=metrics["r2"],
        target_mean=float(np.mean(y_vector)),
        target_std=float(np.std(y_vector)),
    )


def save_training_summary(summary: XGBoostTrainingSummary, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    return output_path


def _read_rows(dataset_path: Path) -> list[dict[str, str]]:
    with dataset_path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _build_matrix(
    rows: list[dict[str, str]],
    target_column: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    feature_columns = _select_numeric_feature_columns(rows[0], target_column)
    categorical_features = _categorical_feature_names(rows)
    feature_names = feature_columns + categorical_features

    x_matrix = np.zeros((len(rows), len(feature_names)), dtype=float)
    y_vector = np.zeros(len(rows), dtype=float)

    categorical_values = _categorical_values(rows)
    for row_index, row in enumerate(rows):
        y_vector[row_index] = _parse_float(row[target_column], target_column)
        for col_index, column_name in enumerate(feature_columns):
            x_matrix[row_index, col_index] = _parse_float(row[column_name], column_name)

        offset = len(feature_columns)
        for column_name in CATEGORICAL_COLUMNS:
            row_value = row.get(column_name, "")
            for category in categorical_values[column_name]:
                x_matrix[row_index, offset] = 1.0 if row_value == category else 0.0
                offset += 1

    return feature_names, x_matrix, y_vector


def _select_numeric_feature_columns(sample_row: dict[str, str], target_column: str) -> list[str]:
    excluded_columns = set(DEFAULT_EXCLUDED_COLUMNS)
    excluded_columns.add(target_column)

    feature_columns: list[str] = []
    for column_name, raw_value in sample_row.items():
        if column_name in excluded_columns or column_name in CATEGORICAL_COLUMNS:
            continue
        if any(column_name.startswith(prefix) for prefix in DEFAULT_EXCLUDED_PREFIXES):
            continue
        if _is_float(raw_value):
            feature_columns.append(column_name)
    return feature_columns


def _categorical_feature_names(rows: list[dict[str, str]]) -> list[str]:
    values = _categorical_values(rows)
    names: list[str] = []
    for column_name in CATEGORICAL_COLUMNS:
        names.extend(f"{column_name}={value}" for value in values[column_name])
    return names


def _categorical_values(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    return {
        column_name: sorted({row.get(column_name, "") for row in rows})
        for column_name in CATEGORICAL_COLUMNS
    }


def _split_indices(
    row_count: int,
    test_fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if row_count < 4:
        raise ValueError("At least 4 rows are required for a train/test split.")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")

    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(row_count)
    test_count = max(1, int(round(row_count * test_fraction)))
    test_count = min(test_count, row_count - 1)
    test_indices = np.sort(indices[:test_count])
    train_indices = np.sort(indices[test_count:])
    return train_indices, test_indices


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    total_variance = float(np.sum(np.square(y_true - np.mean(y_true))))
    residual_variance = float(np.sum(np.square(errors)))
    r2 = 1.0 - residual_variance / total_variance if total_variance > 0.0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": float(r2)}


def _save_feature_metadata(
    output_path: Path,
    target_column: str,
    feature_names: list[str],
    model_params: dict,
) -> None:
    payload = {
        "target_column": target_column,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "model_params": model_params,
        "excluded_columns": sorted(DEFAULT_EXCLUDED_COLUMNS),
        "excluded_prefixes": list(DEFAULT_EXCLUDED_PREFIXES),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_predictions(
    prediction_path: Path,
    rows: list[dict[str, str]],
    test_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    with prediction_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "row_index",
                "scenario_name",
                "time_index",
                "actual",
                "predicted",
                "error",
            ],
        )
        writer.writeheader()
        for local_index, row_index in enumerate(test_indices):
            row = rows[int(row_index)]
            actual = float(y_true[local_index])
            predicted = float(y_pred[local_index])
            writer.writerow(
                {
                    "row_index": int(row_index),
                    "scenario_name": row.get("scenario_name", ""),
                    "time_index": row.get("time_index", ""),
                    "actual": actual,
                    "predicted": predicted,
                    "error": predicted - actual,
                }
            )


def _is_float(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return value != ""


def _parse_float(value: str, column_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column '{column_name}' contains a non-numeric value: {value!r}") from exc


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower()
