"""Universal Parameter-Efficient Fine-Tuning (PEFT) Model Wrapper."""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.config import LoRAConfig, QLoRAConfig, IA3Config
from src.adapters.lora import LoRALinear
from src.adapters.qlora_nf4 import QLoRALinear
from src.adapters.ia3 import IA3Linear


class PEFTModel(nn.Module):
    """Universal PEFT wrapper injecting LoRA, QLoRA NF4, or (IA)³ adapters into PyTorch models."""
    def __init__(
        self,
        base_model: nn.Module,
        adapter_config: Union[LoRAConfig, QLoRAConfig, IA3Config],
    ):
        super().__init__()
        self.base_model = base_model
        self.adapter_config = adapter_config
        self.adapter_type = type(adapter_config).__name__

        # Freeze all base parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Inject adapters into designated target modules
        self._inject_adapters()

    def _inject_adapters(self):
        """Recursively scan modules and replace target linear layers with adapter wrappers."""
        target_names = getattr(self.adapter_config, "target_modules", ["q_proj", "v_proj"])
        
        for name, module in list(self.base_model.named_modules()):
            for sub_name, child in list(module.named_children()):
                if any(sub_name == target or target in sub_name for target in target_names):
                    if isinstance(child, nn.Linear):
                        adapter_layer = self._create_adapter_layer(child, sub_name)
                        setattr(module, sub_name, adapter_layer)

    def _create_adapter_layer(self, linear_layer: nn.Linear, module_name: str) -> nn.Module:
        """Create appropriate adapter module based on config."""
        if isinstance(self.adapter_config, LoRAConfig):
            return LoRALinear(
                base_layer=linear_layer,
                r=self.adapter_config.r,
                lora_alpha=self.adapter_config.lora_alpha,
                lora_dropout=self.adapter_config.lora_dropout,
                merge_weights=self.adapter_config.merge_weights,
            )
        elif isinstance(self.adapter_config, QLoRAConfig):
            return QLoRALinear(
                base_layer=linear_layer,
                r=self.adapter_config.r,
                lora_alpha=self.adapter_config.lora_alpha,
                lora_dropout=self.adapter_config.lora_dropout,
                block_size=self.adapter_config.block_size,
                double_quant=self.adapter_config.double_quant,
            )
        elif isinstance(self.adapter_config, IA3Config):
            is_ff = any(ff in module_name for ff in self.adapter_config.feedforward_modules)
            return IA3Linear(base_layer=linear_layer, is_feedforward=is_ff)
        else:
            raise ValueError(f"Unsupported adapter config type: {type(self.adapter_config)}")

    def get_trainable_parameters(self) -> Tuple[int, int, float]:
        """Compute trainable parameters, total parameters, and trainable ratio percentage.
        
        Returns:
            trainable_params: Count of parameters requiring gradients.
            all_params: Total parameter count.
            trainable_ratio_pct: Percentage of parameters that are trainable.
        """
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.parameters())
        ratio = (trainable_params / all_params) * 100.0 if all_params > 0 else 0.0
        return trainable_params, all_params, ratio

    def print_trainable_parameters(self) -> str:
        """Format a summary string of parameter efficiency."""
        trainable, total, ratio = self.get_trainable_parameters()
        msg = (
            f"PEFT [{self.adapter_type}] Summary:\n"
            f"  Trainable Parameters: {trainable:,}\n"
            f"  Total Base Parameters: {total:,}\n"
            f"  Trainable Ratio:      {ratio:.3f}%\n"
            f"  Parameter Reduction:  {100.0 - ratio:.2f}% frozen"
        )
        print(msg)
        return msg

    def get_adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        """Extract only adapter parameters (excluding frozen base weights)."""
        adapter_state = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                adapter_state[name] = param.detach().cpu().clone()
        return adapter_state

    def merge_and_unload(self) -> nn.Module:
        """Merge adapter weights directly into base layers for zero-overhead inference."""
        for module in self.modules():
            if hasattr(module, "merge"):
                module.merge()
        return self.base_model

    def forward(self, *args, **kwargs) -> Any:
        return self.base_model(*args, **kwargs)
