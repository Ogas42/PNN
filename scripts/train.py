from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pnn_vit.config import load_config
from pnn_vit.trainer import train_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PNN-ViT.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_experiment(config, Path(args.output_dir))


if __name__ == "__main__":
    main()
