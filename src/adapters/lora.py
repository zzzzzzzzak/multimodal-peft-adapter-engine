"""Low-Rank Adaptation (LoRA) Layer Implementation."""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) linear layer wrapping a frozen base linear projection.
    
    Decomposes the weight update into low-rank matrices:
        h = W_0 * x + (alpha / r) * B * A * dropout(x)
    where:
        W_0 in R^{out_features x in_features} (frozen)
        A in R^{r x in_features} (Gaussian initialized)
        B in R^{out_features x r} (Zero initialized)
    """
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.merged = False
        self.merge_weights = merge_weights

        # Freeze base linear layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        if r > 0:
            # Low-rank factor matrices
            self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
            self.reset_parameters()
        else:
            self.lora_A = None
            self.lora_B = None

    def reset_parameters(self):
        """Initialize LoRA matrices: A with Kaiming uniform, B with zeros."""
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def merge(self):
        """Merge low-rank adapter weights into the base weight matrix to eliminate inference latency."""
        if self.r > 0 and not self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_w
            self.merged = True

    def unmerge(self):
        """Unmerge adapter weights from base layer to restore original weights."""
        if self.r > 0 and self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= delta_w
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining frozen base projection and low-rank adapter path."""
        # Base projection
        result = self.base_layer(x)

        if self.r > 0 and not self.merged:
            # Low-rank branch: (x * dropout) @ A.T @ B.T * scaling
            x_dropped = self.lora_dropout(x)
            lora_out = (x_dropped @ self.lora_A.T) @ self.lora_B.T
            result += lora_out * self.scaling

        return result
