"""Unit tests for Low-Rank Adaptation (LoRA)."""

import pytest
import torch
import torch.nn as nn
from src.adapters.lora import LoRALinear


def test_lora_initialization():
    base = nn.Linear(64, 128)
    lora = LoRALinear(base, r=8, lora_alpha=16.0)

    assert not lora.base_layer.weight.requires_grad
    assert lora.lora_A.shape == (8, 64)
    assert lora.lora_B.shape == (128, 8)
    assert lora.scaling == 2.0  # 16.0 / 8

    # B must be initialized to zero
    assert torch.all(lora.lora_B == 0.0)


def test_lora_initial_forward_matches_base():
    base = nn.Linear(32, 64)
    lora = LoRALinear(base, r=4)
    x = torch.randn(2, 5, 32)

    # Since B=0 at init, lora(x) must exactly equal base(x)
    with torch.no_grad():
        out_base = base(x)
        out_lora = lora(x)
    assert torch.allclose(out_base, out_lora, atol=1e-5)


def test_lora_gradient_isolation():
    base = nn.Linear(32, 64)
    lora = LoRALinear(base, r=4)
    x = torch.randn(2, 32)
    
    out = lora(x).sum()
    out.backward()

    # Base weight must NOT receive gradients
    assert lora.base_layer.weight.grad is None
    # LoRA factor matrices must receive gradients
    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None


def test_lora_merge_and_unmerge_parity():
    base = nn.Linear(16, 32)
    lora = LoRALinear(base, r=4, lora_alpha=8.0)
    lora.eval()
    # Manually populate B with non-zero values
    with torch.no_grad():
        lora.lora_B.copy_(torch.randn_like(lora.lora_B))

    x = torch.randn(2, 16)
    out_unmerged = lora(x)

    lora.merge()
    assert lora.merged
    out_merged = lora(x)
    assert torch.allclose(out_unmerged, out_merged, atol=1e-5)

    lora.unmerge()
    assert not lora.merged
    out_restored = lora(x)
    assert torch.allclose(out_unmerged, out_restored, atol=1e-5)
