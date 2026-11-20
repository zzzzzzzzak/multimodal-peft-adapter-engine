"""Typed configuration dataclasses for PEFT adapters and multimodal models."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class LoRAConfig:
    """Configuration for Low-Rank Adaptation (LoRA)."""
    r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"  # 'none', 'all', 'lora_only'
    merge_weights: bool = False


@dataclass
class QLoRAConfig:
    """Configuration for Quantized Low-Rank Adaptation (QLoRA) with NF4."""
    r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bits: int = 4
    quant_type: str = "nf4"  # 'nf4' or 'fp4'
    double_quant: bool = True
    block_size: int = 64


@dataclass
class IA3Config:
    """Configuration for Infused Adapter by Inhibiting and Amplifying Inner Activations (IA)³."""
    target_modules: List[str] = field(default_factory=lambda: ["k_proj", "v_proj", "down_proj"])
    feedforward_modules: List[str] = field(default_factory=lambda: ["down_proj"])


@dataclass
class ModelConfig:
    """Configuration for Multimodal Vision-Language Transformer backbone."""
    vision_dim: int = 256
    text_dim: int = 256
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.10


@dataclass
class BenchmarkConfig:
    """Configuration for comparative PEFT profiling."""
    batch_size: int = 16
    seq_len: int = 32
    num_keyframes: int = 4
    num_steps: int = 25
    learning_rate: float = 5e-4
    device: str = "auto"
