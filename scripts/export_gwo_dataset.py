"""Export GWO optimization results as a CSV dataset for XGBoost analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isac_power_allocation.channels.scenarios import STANDARD_SCENARIOS
from isac_power_allocation.config import build_default_experiment_config
from isac_power_allocation.experiments.gwo_dataset import export_gwo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GWO over channel snapshots and export a CSV dataset."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "gwo_xgboost_dataset.csv",
        help="CSV path to write.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=sorted(STANDARD_SCENARIOS),
        choices=sorted(STANDARD_SCENARIOS),
        help="Scenario names to include. Defaults to all scenarios.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_default_experiment_config()
    output_path = export_gwo_dataset(
        output_path=args.output,
        config=config,
        scenario_names=args.scenarios,
    )
    print(f"Saved GWO dataset: {output_path}")
    print(f"Scenarios: {', '.join(args.scenarios)}")


if __name__ == "__main__":
    main()
