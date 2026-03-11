# PNN-ViT

Research prototype for `PNN for Adaptive Token Routing in Small Vision Transformers`.

## Layout

- `src/pnn_vit/`: model, routing, data, metrics, and training code
- `configs/`: experiment and dataset YAML configs
- `scripts/`: CLI entrypoints for train/eval/benchmark/visualization
- `tests/`: unit and smoke tests

## Quick Start

```bash
pip install -e .
python scripts/train.py --config configs/experiments/pnn_cifar100.yaml --output-dir outputs/pnn_cifar100
python scripts/eval.py --config configs/experiments/pnn_cifar100.yaml --checkpoint outputs/pnn_cifar100/best.pt
python scripts/benchmark.py --config configs/experiments/pnn_cifar100.yaml --checkpoint outputs/pnn_cifar100/best.pt
python scripts/visualize.py --config configs/experiments/pnn_cifar100.yaml --checkpoint outputs/pnn_cifar100/best.pt
```

## Dataset Notes

- `CIFAR-100` downloads automatically by default.
- `Tiny-ImageNet` expects the original `tiny-imagenet-200` folder or a class-folder version.

## Methods

- `base`: DeiT-Tiny without pruning
- `random`: random token pruning
- `attention`: prune by `[CLS] -> token` attention score
- `mlp`: learned token importance MLP
- `dynamicvit_lite`: lightweight DynamicViT-style score predictor
- `pnn`: Physarum pre-router with conductance evolution
