"""QLoRA 4-bit NormalFloat (NF4) Quantization and Low-Rank Adaptation."""

import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Theoretical 16-point NormalFloat4 (NF4) codebook quantiles for N(0, 1)
# Normalized such that the maximum absolute value is exactly 1.0
NF4_CODEBOOK = torch.tensor([
    -1.00000000, -0.69619280, -0.52507305, -0.39491749,
    -0.28444138, -0.18477343, -0.09105003,  0.00000000,
     0.07958030,  0.16093020,  0.24611230,  0.33791524,
     0.44070982,  0.56261700,  0.72295684,  1.00000000,
], dtype=torch.float32)


class NF4Quantizer:
    """Block-wise 4-bit NormalFloat quantizer with optional Double Quantization (DQ)."""
    @staticmethod
    def quantize_blockwise(
        tensor: torch.Tensor,
        block_size: int = 64,
        double_quant: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Quantize a float tensor into 4-bit NF4 indices with per-block scaling.
        
        Args:
            tensor: Input weight tensor (out_features, in_features).
            block_size: Number of elements per quantization block.
            double_quant: Whether to double-quantize block scales into 8-bit integers.
            
        Returns:
            packed_indices: Packed uint8 tensor containing two 4-bit values per byte.
            scales: Per-block scale constants.
            orig_shape: Original tensor shape for dequantization.
            dq_meta: Optional second-level scale and bias for double quantization.
        """
        orig_shape = tensor.shape
        flat = tensor.detach().float().flatten()
        n = flat.numel()
        
        # Pad to multiple of block_size if necessary
        pad_len = (block_size - (n % block_size)) % block_size
        if pad_len > 0:
            flat = F.pad(flat, (0, pad_len), "constant", 0.0)
            
        num_blocks = flat.numel() // block_size
        blocks = flat.view(num_blocks, block_size)
        
        # Absolute maximum scale per block
        scales = blocks.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
        norm_blocks = blocks / scales  # Values in [-1, 1]
        
        # Map normalized values to nearest NF4 codebook entry
        codebook = NF4_CODEBOOK.to(tensor.device)
        # Compute squared differences: (num_blocks, block_size, 16)
        diffs = (norm_blocks.unsqueeze(-1) - codebook.view(1, 1, 16)) ** 2
        indices = diffs.argmin(dim=-1).to(torch.uint8)  # Values in [0, 15]
        
        # Pack pairs of 4-bit integers into single uint8 bytes: (i0 << 4) | i1
        indices_flat = indices.flatten()
        half_n = indices_flat.numel() // 2
        packed = (indices_flat[0::2] << 4) | (indices_flat[1::2] & 0x0F)
        
        scales_flat = scales.squeeze(-1)
        dq_meta = None
        
        if double_quant and len(scales_flat) >= 16:
            # Quantize scales into 8-bit integers: s_i -> uint8
            dq_scale = scales_flat.max().clamp(min=1e-8) / 255.0
            scales_u8 = (scales_flat / dq_scale).round().clamp(0, 255).to(torch.uint8)
            dq_meta = (dq_scale, pad_len)
            scales_to_store = scales_u8
        else:
            scales_to_store = scales_flat
            
        return packed, scales_to_store, orig_shape, dq_meta

    @staticmethod
    def dequantize_blockwise(
        packed_indices: torch.Tensor,
        scales: torch.Tensor,
        orig_shape: Tuple[int, ...],
        block_size: int = 64,
        dq_meta: Optional[Tuple[torch.Tensor, int]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Dequantize 4-bit packed NF4 representation back to floating point."""
        device = packed_indices.device
        codebook = NF4_CODEBOOK.to(device)
        
        # Unpack uint8 into two 4-bit integers
        high = (packed_indices >> 4).to(torch.long)
        low = (packed_indices & 0x0F).to(torch.long)
        unpacked_indices = torch.empty(len(packed_indices) * 2, dtype=torch.long, device=device)
        unpacked_indices[0::2] = high
        unpacked_indices[1::2] = low
        
        # Map indices to NF4 values
        norm_values = codebook[unpacked_indices]
        
        num_blocks = len(norm_values) // block_size
        blocks = norm_values.view(num_blocks, block_size)
        
        # Restore scales if double quantized
        if dq_meta is not None:
            dq_scale, _ = dq_meta
            real_scales = scales.float() * dq_scale
        else:
            real_scales = scales.float()
            
        dequant_blocks = blocks * real_scales.view(-1, 1)
        flat_dequant = dequant_blocks.flatten()
        
        total_orig = math.prod(orig_shape)
        restored = flat_dequant[:total_orig].view(orig_shape).to(dtype)
        return restored


class QLoRALinear(nn.Module):
    """Quantized LoRA Linear layer with 4-bit NF4 base weights and trainable low-rank adapters."""
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        block_size: int = 64,
        double_quant: bool = True,
    ):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.block_size = block_size
        self.double_quant = double_quant

        # Quantize base layer weights to 4-bit NF4
        packed, scales, orig_shape, dq_meta = NF4Quantizer.quantize_blockwise(
            base_layer.weight, block_size=block_size, double_quant=double_quant
        )
        self.register_buffer("packed_weight", packed)
        self.register_buffer("scales", scales)
        self.orig_shape = orig_shape
        self.dq_meta = dq_meta

        if base_layer.bias is not None:
            self.register_buffer("bias", base_layer.bias.detach().clone())
        else:
            self.bias = None

        # Trainable low-rank adapter matrices (FP32/FP16)
        if r > 0:
            self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
            self.reset_parameters()
        else:
            self.lora_A = None
            self.lora_B = None

    def reset_parameters(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize 4-bit NF4 weight on-the-fly and execute forward pass with LoRA path."""
        # Dequantize base weight
        w_dequant = NF4Quantizer.dequantize_blockwise(
            self.packed_weight,
            self.scales,
            self.orig_shape,
            block_size=self.block_size,
            dq_meta=self.dq_meta,
            dtype=x.dtype,
        )

        # Base projection
        out = F.linear(x, w_dequant, self.bias)

        # Add trainable low-rank path
        if self.r > 0:
            x_dropped = self.lora_dropout(x)
            lora_out = (x_dropped @ self.lora_A.T) @ self.lora_B.T
            out += lora_out * self.scaling

        return out
