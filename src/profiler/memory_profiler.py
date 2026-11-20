"""Memory and parameter profiler for benchmarking PEFT efficiency."""

from dataclasses import dataclass
import gc
import time
from typing import Any, Callable, Dict, Optional, Tuple
import torch
import torch.nn as nn


@dataclass
class ProfilingResult:
    """Quantitative performance and resource footprint result."""
    method_name: str
    trainable_params: int
    total_params: int
    trainable_pct: float
    peak_memory_mb: float
    forward_time_ms: float
    backward_time_ms: float
    step_time_ms: float
    convergence_loss: float


class MemoryProfiler:
    """Profiles memory consumption, wall-clock latency, and parameter footprint."""
    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def _reset_memory(self):
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

    def _get_peak_memory_mb(self) -> float:
        if self.device.type == "cuda":
            return float(torch.cuda.max_memory_allocated(self.device) / (1024 * 1024))
        else:
            # CPU resident memory tracking heuristic
            import os
            try:
                import psutil
                process = psutil.Process(os.getpid())
                return float(process.memory_info().rss / (1024 * 1024))
            except Exception:
                return 0.0

    def profile_training_step(
        self,
        model: nn.Module,
        batch_inputs: Tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        method_name: str = "PEFT",
        num_warmup: int = 3,
        num_steps: int = 10,
    ) -> ProfilingResult:
        """Benchmark forward/backward step execution, memory allocation, and loss."""
        model.to(self.device)
        model.train()
        text_ids, visual_feats = [t.to(self.device) for t in batch_inputs]
        labels = targets.to(self.device)

        # Warmup
        for _ in range(num_warmup):
            optimizer.zero_grad()
            out = model(text_ids, visual_feats)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        self._reset_memory()

        fwd_times = []
        bwd_times = []
        last_loss = 0.0

        for _ in range(num_steps):
            optimizer.zero_grad()
            
            # Forward pass timing
            t0 = time.perf_counter()
            out = model(text_ids, visual_feats)
            loss = criterion(out, labels)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            fwd_times.append((t1 - t0) * 1000.0)

            # Backward pass timing
            loss.backward()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            bwd_times.append((t2 - t1) * 1000.0)

            optimizer.step()
            last_loss = float(loss.item())

        peak_mem = self._get_peak_memory_mb()
        fwd_avg = float(sum(fwd_times) / len(fwd_times))
        bwd_avg = float(sum(bwd_times) / len(bwd_times))

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        pct = (trainable_params / total_params) * 100.0 if total_params > 0 else 0.0

        return ProfilingResult(
            method_name=method_name,
            trainable_params=trainable_params,
            total_params=total_params,
            trainable_pct=round(pct, 3),
            peak_memory_mb=round(peak_mem, 2),
            forward_time_ms=round(fwd_avg, 2),
            backward_time_ms=round(bwd_avg, 2),
            step_time_ms=round(fwd_avg + bwd_avg, 2),
            convergence_loss=round(last_loss, 4),
        )
