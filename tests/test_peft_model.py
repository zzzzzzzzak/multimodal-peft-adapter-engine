"""Unit tests for PEFTModel wrapper and parameter efficiency."""

import pytest
import torch
from src.config import ModelConfig, LoRAConfig, QLoRAConfig, IA3Config
from src.models.base_vlm import MultimodalTransformerBackbone
from src.models.peft_model import PEFTModel


def test_peft_model_lora_wrapping():
    base = MultimodalTransformerBackbone(ModelConfig())
    peft = PEFTModel(base, LoRAConfig(r=8, target_modules=["q_proj", "v_proj"]))

    trainable, total, ratio = peft.get_trainable_parameters()
    # Trainable parameters should be a tiny fraction of the total base model (< 5%)
    assert ratio < 5.0
    assert trainable > 0
    assert trainable < total

    # Check forward pass
    text_ids = torch.randint(1, 1000, (2, 16))
    visual_feats = torch.randn(2, 4, 256)
    out = peft(text_ids, visual_feats)
    assert out.shape == (2, 2)


def test_peft_model_ia3_wrapping():
    base = MultimodalTransformerBackbone(ModelConfig())
    peft = PEFTModel(base, IA3Config(target_modules=["k_proj", "v_proj"]))

    trainable, total, ratio = peft.get_trainable_parameters()
    # (IA)3 uses only vector scaling, ratio should be extremely low (< 0.5%)
    assert ratio < 0.50
    assert trainable > 0


def test_peft_adapter_state_dict_isolation():
    base = MultimodalTransformerBackbone(ModelConfig())
    peft = PEFTModel(base, LoRAConfig(r=4))

    adapter_state = peft.get_adapter_state_dict()
    assert len(adapter_state) > 0
    # Every parameter in adapter state must be a LoRA weight
    for name in adapter_state.keys():
        assert "lora_" in name or "ia3_" in name
