"""(IA)³: Infused Adapter by Inhibiting and Amplifying Inner Activations."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class IA3Linear(nn.Module):
    """(IA)³ adapter layer that rescales inner activations via learned vectors.
    
    Instead of low-rank matrices B*A, (IA)³ injects a learned scaling vector l
    initialized to 1.0, reducing trainable parameters to < 0.01% of the base model:
        h_out = (l * (W_0 * x))  [for Key/Value projections]
        or
        h_out = W_0 * (l * x)    [for Feed-Forward down-projections]
    """
    def __init__(
        self,
        base_layer: nn.Linear,
        is_feedforward: bool = False,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.is_feedforward = is_feedforward
        self.merged = False

        # Freeze base layer weights
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Learned scaling vector initialized to 1.0
        scale_dim = self.in_features if is_feedforward else self.out_features
        self.ia3_scale = nn.Parameter(torch.ones(scale_dim))

    def merge(self):
        """Fold scaling vector into base weight matrix."""
        if not self.merged:
            if self.is_feedforward:
                # Scale columns: W_0 * diag(l)
                self.base_layer.weight.data *= self.ia3_scale.view(1, -1)
            else:
                # Scale rows: diag(l) * W_0
                self.base_layer.weight.data *= self.ia3_scale.view(-1, 1)
                if self.base_layer.bias is not None:
                    self.base_layer.bias.data *= self.ia3_scale
            self.merged = True

    def unmerge(self):
        """Unfold scaling vector from base weight matrix."""
        if self.merged:
            eps = 1e-8
            if self.is_feedforward:
                self.base_layer.weight.data /= (self.ia3_scale.view(1, -1) + eps)
            else:
                self.base_layer.weight.data /= (self.ia3_scale.view(-1, 1) + eps)
                if self.base_layer.bias is not None:
                    self.base_layer.bias.data /= (self.ia3_scale + eps)
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with activation rescaling."""
        if self.merged:
            return self.base_layer(x)

        if self.is_feedforward:
            # Rescale input activations before projection
            scaled_x = x * self.ia3_scale
            return self.base_layer(scaled_x)
        else:
            # Rescale output activations after projection
            out = self.base_layer(x)
            return out * self.ia3_scale
