"""Unit tests for (IA)³ activation scaling adapters."""

import pytest
import torch
import torch.nn as nn
from src.adapters.ia3 import IA3Linear


def test_ia3_initialization():
    base = nn.Linear(32, 64)
    ia3 = IA3Linear(base, is_feedforward=False)

    assert not ia3.base_layer.weight.requires_grad
    assert ia3.ia3_scale.shape == (64,)
    # Scaling vectors must start at 1.0 (identity transformation)
    assert torch.all(ia3.ia3_scale == 1.0)


def test_ia3_initial_forward_matches_base():
    base = nn.Linear(32, 64)
    ia3 = IA3Linear(base)
    x = torch.randn(2, 32)

    with torch.no_grad():
        out_base = base(x)
        out_ia3 = ia3(x)

    assert torch.allclose(out_base, out_ia3, atol=1e-5)


def test_ia3_gradient_flow():
    base = nn.Linear(32, 64)
    ia3 = IA3Linear(base)
    x = torch.randn(2, 32)

    out = ia3(x).sum()
    out.backward()

    assert ia3.base_layer.weight.grad is None
    assert ia3.ia3_scale.grad is not None


def test_ia3_merge_and_unmerge():
    base = nn.Linear(16, 32, bias=True)
    ia3 = IA3Linear(base, is_feedforward=False)
    ia3.eval()
    with torch.no_grad():
        ia3.ia3_scale.copy_(torch.randn_like(ia3.ia3_scale).abs() + 0.5)

    x = torch.randn(2, 16)
    out_unmerged = ia3(x)

    ia3.merge()
    assert ia3.merged
    out_merged = ia3(x)
    assert torch.allclose(out_unmerged, out_merged, atol=1e-5)

    ia3.unmerge()
    assert not ia3.merged
    out_restored = ia3(x)
    assert torch.allclose(out_unmerged, out_restored, atol=1e-5)


def test_ia3_feedforward_merge_and_unmerge():
    base = nn.Linear(16, 32, bias=True)
    ia3 = IA3Linear(base, is_feedforward=True)
    ia3.eval()
    with torch.no_grad():
        ia3.ia3_scale.copy_(torch.randn_like(ia3.ia3_scale).abs() + 0.5)

    x = torch.randn(2, 16)
    out_unmerged = ia3(x)

    ia3.merge()
    assert ia3.merged
    out_merged = ia3(x)
    assert torch.allclose(out_unmerged, out_merged, atol=1e-5)

    ia3.unmerge()
    assert not ia3.merged
    out_restored = ia3(x)
    assert torch.allclose(out_unmerged, out_restored, atol=1e-5)
