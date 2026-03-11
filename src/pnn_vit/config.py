from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    name: str
    root: str = "./data"
    image_size: int = 224
    num_classes: int = 100
    batch_size: int = 32
    eval_batch_size: int = 64
    epochs: int = 100
    grad_accumulation: int = 1
    num_workers: int = 4
    download: bool = False
    probe_samples: int = 64


@dataclass
class OptimConfig:
    lr: float = 5e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    label_smoothing: float = 0.0
    amp: bool = True
    grad_clip: float | None = 1.0
    seed: int = 42


@dataclass
class PNNConfig:
    groups: int = 16
    steps: int = 4
    dt: float = 0.2
    gamma: float = 0.2
    mu: float = 1.0
    eps: float = 1e-4
    keep_ratio: float = 0.5
    symmetric: bool = True
    sparse_lambda: float = 0.01
    stable_lambda: float = 0.05
    use_dynamics: bool = True
    use_pressure: bool = True


@dataclass
class ModelConfig:
    model_name: str = "deit_tiny_patch16_224"
    method: str = "pnn"
    pretrained: bool = True
    num_classes: int = 100
    insert_layers: list[int] = field(default_factory=lambda: [6])
    keep_ratios: list[float] = field(default_factory=lambda: [0.5])
    pnn: PNNConfig = field(default_factory=PNNConfig)
    score_hidden_dim: int = 96
    drop_path_rate: float = 0.0


@dataclass
class LoggingConfig:
    log_interval: int = 20
    checkpoint_interval: int = 10


@dataclass
class ExperimentConfig:
    name: str
    dataset: DatasetConfig
    model: ModelConfig
    optim: OptimConfig = field(default_factory=OptimConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _merge_dicts(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if "base" in raw:
        base_path = (path.parent / raw.pop("base")).resolve()
        raw = _merge_dicts(_load_yaml(base_path), raw)
    return raw


def _build_dataclass(cls: type[Any], payload: dict[str, Any]) -> Any:
    kwargs = {}
    for field_name, field_info in cls.__dataclass_fields__.items():
        if field_name not in payload:
            continue
        value = payload[field_name]
        field_type = field_info.type
        if hasattr(field_type, "__dataclass_fields__") and isinstance(value, dict):
            kwargs[field_name] = _build_dataclass(field_type, value)
        else:
            kwargs[field_name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    payload = _load_yaml(Path(path).resolve())
    dataset = _build_dataclass(DatasetConfig, payload.get("dataset", {}))
    model_payload = payload.get("model", {})
    if "pnn" in model_payload:
        model_payload = dict(model_payload)
        model_payload["pnn"] = _build_dataclass(PNNConfig, model_payload["pnn"])
    model = _build_dataclass(ModelConfig, model_payload)
    optim = _build_dataclass(OptimConfig, payload.get("optim", {}))
    logging = _build_dataclass(LoggingConfig, payload.get("logging", {}))
    return ExperimentConfig(
        name=payload["name"],
        dataset=dataset,
        model=model,
        optim=optim,
        logging=logging,
    )
