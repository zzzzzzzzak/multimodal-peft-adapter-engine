"""Multimodal Vision-Language Transformer Backbone."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig


class MultiHeadAttentionBlock(nn.Module):
    """Multi-Head Attention block with separate linear projections for Q, K, V, and Output."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Explicit linear projections suitable for PEFT adapter injection
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, q_len, _ = query.shape
        _, kv_len, _ = key.shape

        q = self.q_proj(query).view(b, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(b, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(b, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = (attn @ v).transpose(1, 2).contiguous().view(b, q_len, self.embed_dim)

        return self.out_proj(context)


class FeedForwardBlock(nn.Module):
    """Two-layer MLP feed-forward block with explicit up and down projections."""
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.up_proj = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(self.act(self.up_proj(x))))


class MultimodalTransformerLayer(nn.Module):
    """Transformer layer with self-attention, cross-modal attention, and feed-forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionBlock(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.cross_attn = MultiHeadAttentionBlock(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = FeedForwardBlock(embed_dim, ffn_dim, dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(
        self, text_tokens: torch.Tensor, visual_tokens: torch.Tensor
    ) -> torch.Tensor:
        # 1. Text Self-Attention
        res = text_tokens
        x = self.norm1(text_tokens)
        text_tokens = res + self.self_attn(x, x, x)

        # 2. Text-to-Vision Cross-Attention (Query: Text, Key/Value: Vision)
        res = text_tokens
        x = self.norm2(text_tokens)
        v = self.norm2(visual_tokens)
        text_tokens = res + self.cross_attn(query=x, key=v, value=v)

        # 3. Feed-Forward Network
        res = text_tokens
        x = self.norm3(text_tokens)
        text_tokens = res + self.ffn(x)

        return text_tokens


class MultimodalTransformerBackbone(nn.Module):
    """Vision-Language Model backbone mapping vision & text tokens to multimodal embeddings."""
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        dim = self.config.hidden_dim

        # Input projections
        self.text_embed = nn.Embedding(2048, dim)
        self.vision_proj = nn.Linear(self.config.vision_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, dim) * 0.02)

        # Stack of multimodal transformer layers
        self.layers = nn.ModuleList([
            MultimodalTransformerLayer(
                embed_dim=dim,
                num_heads=self.config.num_heads,
                ffn_dim=dim * 2,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2)  # Binary alignment prediction

    def forward(
        self, text_ids: torch.Tensor, visual_feats: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            text_ids: (B, L) token indices.
            visual_feats: (B, K, vision_dim) visual keyframe embeddings.
            
        Returns:
            logits: (B, 2) multimodal classification logits.
        """
        b, seq_len = text_ids.shape
        t_tokens = self.text_embed(text_ids) + self.pos_embed[:, :seq_len, :]
        v_tokens = self.vision_proj(visual_feats)

        for layer in self.layers:
            t_tokens = layer(t_tokens, v_tokens)

        # Global average pooling across text tokens
        pooled = self.final_norm(t_tokens).mean(dim=1)
        logits = self.head(pooled)
        return logits
