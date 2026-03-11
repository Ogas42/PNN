import torch

from pnn_vit.config import ModelConfig, PNNConfig
from pnn_vit.metrics import estimate_vit_flops
from pnn_vit.models.vit import build_model


def test_model_forward_backward() -> None:
    config = ModelConfig(
        model_name="deit_tiny_patch16_224",
        method="pnn",
        pretrained=False,
        num_classes=10,
        insert_layers=[6],
        keep_ratios=[0.5],
        pnn=PNNConfig(groups=16, keep_ratio=0.5),
    )
    model = build_model(config)
    model.train()
    images = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 1])
    outputs = model(images)
    loss = torch.nn.functional.cross_entropy(outputs["logits"], labels)
    loss = loss + 0.01 * outputs["aux_losses"]["sparse"] + 0.05 * outputs["aux_losses"]["stable"]
    loss.backward()
    assert outputs["logits"].shape == (2, 10)
    assert len(outputs["records"]) == 1


def test_small_training_reduces_loss() -> None:
    config = ModelConfig(
        model_name="deit_tiny_patch16_224",
        method="pnn",
        pretrained=False,
        num_classes=2,
        insert_layers=[6],
        keep_ratios=[0.5],
        pnn=PNNConfig(groups=16, keep_ratio=0.5),
    )
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    images = torch.randn(4, 3, 224, 224)
    labels = torch.zeros(4, dtype=torch.long)
    losses = []
    model.train()
    for _ in range(10):
        optimizer.zero_grad()
        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs["logits"], labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] <= losses[0]


def test_flops_estimation() -> None:
    flops = estimate_vit_flops(
        image_size=224,
        patch_size=16,
        embed_dim=192,
        mlp_ratio=4.0,
        num_heads=3,
        token_counts=[197] * 12,
        num_classes=100,
    )
    assert flops > 0
