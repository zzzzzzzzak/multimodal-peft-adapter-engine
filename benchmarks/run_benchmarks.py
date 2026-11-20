"""Comprehensive benchmark suite comparing Full Fine-Tuning vs LoRA vs QLoRA vs (IA)³."""

from dataclasses import asdict
import json
import os
import sys
from typing import Dict, List
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ModelConfig, LoRAConfig, QLoRAConfig, IA3Config
from src.models.base_vlm import MultimodalTransformerBackbone
from src.models.peft_model import PEFTModel
from src.profiler.memory_profiler import MemoryProfiler, ProfilingResult
from src.data.synthetic_dataset import SyntheticMultimodalDataset


def run_comparative_peft_benchmarks(
    device: str = "auto",
    output_json: str = "benchmarks/peft_benchmark_results.json",
) -> List[ProfilingResult]:
    """Execute head-to-head parameter, memory, and latency profiling across PEFT methods."""
    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else "cpu")
    print("=" * 70)
    print(f" PEFT ADAPTER ENGINE COMPARATIVE BENCHMARK (Device: {dev})")
    print("=" * 70)

    # Prepare standard batch
    dataset = SyntheticMultimodalDataset(num_samples=32, seq_len=32, num_keyframes=4, vision_dim=256)
    text_batch = dataset.text_ids[:16]
    visual_batch = dataset.visual_feats[:16]
    labels_batch = dataset.labels[:16]

    profiler = MemoryProfiler(device=dev)
    criterion = nn.CrossEntropyLoss()
    results = []

    # 1. Full Fine-Tuning (Baseline)
    print("\n[1/5] Profiling Full Fine-Tuning (All Base Parameters Trainable)...")
    base_model = MultimodalTransformerBackbone(ModelConfig())
    for p in base_model.parameters():
        p.requires_grad = True
    opt = optim.AdamW(base_model.parameters(), lr=1e-4)
    r_fft = profiler.profile_training_step(
        model=base_model,
        batch_inputs=(text_batch, visual_batch),
        targets=labels_batch,
        optimizer=opt,
        criterion=criterion,
        method_name="Full Fine-Tuning (Baseline)",
    )
    results.append(r_fft)

    # 2. LoRA (r=8)
    print("\n[2/5] Profiling LoRA (r=8, alpha=16)...")
    m_lora8 = MultimodalTransformerBackbone(ModelConfig())
    peft_lora8 = PEFTModel(m_lora8, LoRAConfig(r=8, lora_alpha=16.0))
    opt = optim.AdamW([p for p in peft_lora8.parameters() if p.requires_grad], lr=5e-4)
    r_lora8 = profiler.profile_training_step(
        model=peft_lora8,
        batch_inputs=(text_batch, visual_batch),
        targets=labels_batch,
        optimizer=opt,
        criterion=criterion,
        method_name="LoRA (r=8)",
    )
    results.append(r_lora8)

    # 3. LoRA (r=16)
    print("\n[3/5] Profiling LoRA (r=16, alpha=32)...")
    m_lora16 = MultimodalTransformerBackbone(ModelConfig())
    peft_lora16 = PEFTModel(m_lora16, LoRAConfig(r=16, lora_alpha=32.0))
    opt = optim.AdamW([p for p in peft_lora16.parameters() if p.requires_grad], lr=5e-4)
    r_lora16 = profiler.profile_training_step(
        model=peft_lora16,
        batch_inputs=(text_batch, visual_batch),
        targets=labels_batch,
        optimizer=opt,
        criterion=criterion,
        method_name="LoRA (r=16)",
    )
    results.append(r_lora16)

    # 4. QLoRA (NF4 4-bit with Double Quantization)
    print("\n[4/5] Profiling QLoRA (NF4 4-bit + Double Quantization)...")
    m_qlora = MultimodalTransformerBackbone(ModelConfig())
    peft_qlora = PEFTModel(m_qlora, QLoRAConfig(r=8, lora_alpha=16.0, double_quant=True))
    opt = optim.AdamW([p for p in peft_qlora.parameters() if p.requires_grad], lr=5e-4)
    r_qlora = profiler.profile_training_step(
        model=peft_qlora,
        batch_inputs=(text_batch, visual_batch),
        targets=labels_batch,
        optimizer=opt,
        criterion=criterion,
        method_name="QLoRA (NF4 4-bit)",
    )
    results.append(r_qlora)

    # 5. (IA)³ Adapter
    print("\n[5/5] Profiling (IA)³ (Inhibited/Amplified Inner Activation Scaling)...")
    m_ia3 = MultimodalTransformerBackbone(ModelConfig())
    peft_ia3 = PEFTModel(m_ia3, IA3Config())
    opt = optim.AdamW([p for p in peft_ia3.parameters() if p.requires_grad], lr=1e-3)
    r_ia3 = profiler.profile_training_step(
        model=peft_ia3,
        batch_inputs=(text_batch, visual_batch),
        targets=labels_batch,
        optimizer=opt,
        criterion=criterion,
        method_name="(IA)³ Adapter",
    )
    results.append(r_ia3)

    # Save to JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # Format Markdown summary table
    print("\n" + "=" * 80)
    print(" PEFT EFFICIENCY & RESOURCE BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Method':<28} | {'Trainable Params':<16} | {'Trainable %':<11} | {'Step Time':<10} | {'Loss':<6}")
    print("-" * 80)
    for r in results:
        print(f"{r.method_name:<28} | {r.trainable_params:>16,d} | {r.trainable_pct:>10.3f}% | {r.step_time_ms:>7.2f} ms | {r.convergence_loss:>6.4f}")
    print("=" * 80)
    print(f"Saved benchmark report to: {output_json}\n")

    return results


if __name__ == "__main__":
    run_comparative_peft_benchmarks()
