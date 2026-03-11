from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accelerate import Accelerator
from torch import nn

from pnn_vit.config import load_config
from pnn_vit.data.datasets import build_loaders
from pnn_vit.models.vit import build_model
from pnn_vit.trainer import evaluate
from pnn_vit.utils.io import dump_json, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PNN-ViT.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _, val_loader, _ = build_loaders(config.dataset)
    model = build_model(config.model)
    load_checkpoint(args.checkpoint, model)
    accelerator = Accelerator()
    criterion = nn.CrossEntropyLoss(label_smoothing=config.optim.label_smoothing)
    model, val_loader = accelerator.prepare(model, val_loader)
    metrics = evaluate(accelerator, model, val_loader, criterion, capture_router=True)
    output = Path(args.output) if args.output else Path(args.checkpoint).with_name("eval_metrics.json")
    dump_json(output, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
