"""Unit tests for QLoRA NormalFloat (NF4) quantization and dequantization."""

import pytest
import torch
import torch.nn as nn
from src.adapters.qlora_nf4 import NF4Quantizer, QLoRALinear, NF4_CODEBOOK


def test_nf4_codebook_properties():
    assert len(NF4_CODEBOOK) == 16
    assert NF4_CODEBOOK[0] == -1.0
    assert NF4_CODEBOOK[-1] == 1.0
    assert 0.0 in NF4_CODEBOOK
    # Ensure monotonic ordering
    assert torch.all(NF4_CODEBOOK[1:] >= NF4_CODEBOOK[:-1])


def test_nf4_blockwise_quantize_and_dequantize():
    # Normal distribution simulates real neural network weights
    torch.manual_seed(42)
    w = torch.randn(128, 64)

    packed, scales, shape, dq_meta = NF4Quantizer.quantize_blockwise(w, block_size=64, double_quant=True)
    
    # 128 * 64 = 8192 elements. Packed into 4-bit (2 per byte) = 4096 bytes!
    assert packed.dtype == torch.uint8
    assert packed.numel() == 4096

    restored = NF4Quantizer.dequantize_blockwise(packed, scales, shape, block_size=64, dq_meta=dq_meta)
    assert restored.shape == w.shape

    # Quantization error (MSE) should be very low for NF4 on standard normal weights
    mse = torch.mean((w - restored) ** 2).item()
    assert mse < 0.05


def test_qlora_linear_forward_and_backward():
    base = nn.Linear(64, 128)
    qlora = QLoRALinear(base, r=8, lora_alpha=16.0)

    x = torch.randn(2, 64)
    out = qlora(x)
    assert out.shape == (2, 128)

    loss = out.sum()
    loss.backward()

    # Trainable LoRA parameters receive gradients
    assert qlora.lora_A.grad is not None
    assert qlora.lora_B.grad is not None
