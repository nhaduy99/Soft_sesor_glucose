import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rhamnose_ml.config import load_config
from rhamnose_ml.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline Rhamnose models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/defaults.json"),
        help="Path to JSON config file.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
