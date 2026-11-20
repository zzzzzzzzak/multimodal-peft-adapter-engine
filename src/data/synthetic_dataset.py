"""Synthetic vision-language data generator for reproducible PEFT benchmarks."""

from typing import Tuple
import torch
from torch.utils.data import Dataset


class SyntheticMultimodalDataset(Dataset):
    """Generates synthetic paired text token IDs and visual keyframe vectors."""
    def __init__(
        self,
        num_samples: int = 120,
        seq_len: int = 32,
        num_keyframes: int = 4,
        vision_dim: int = 256,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        torch.manual_seed(seed)
        
        # Synthetic text IDs in [1, 2047]
        self.text_ids = torch.randint(1, 2040, (num_samples, seq_len), dtype=torch.long)
        # Synthetic visual features
        self.visual_feats = torch.randn(num_samples, num_keyframes, vision_dim, dtype=torch.float32)
        # Binary target: 0 = unaligned, 1 = aligned
        self.labels = torch.randint(0, 2, (num_samples,), dtype=torch.long)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.text_ids[idx], self.visual_feats[idx], self.labels[idx]
