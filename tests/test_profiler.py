"""Unit tests for MemoryProfiler."""

import torch
import torch.nn as nn
import torch.optim as optim
from src.profiler.memory_profiler import MemoryProfiler, ProfilingResult


def test_memory_profiler_execution():
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )
    profiler = MemoryProfiler(device=torch.device("cpu"))
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    dummy_inputs = (torch.randn(8, 32), torch.randn(8, 4, 16))
    # Fake model wrapper for test
    class SimpleWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x, v):
            return self.m(x)

    wrapped = SimpleWrapper(model)
    res = profiler.profile_training_step(
        model=wrapped,
        batch_inputs=dummy_inputs,
        targets=torch.randint(0, 2, (8,)),
        optimizer=optimizer,
        criterion=criterion,
        num_warmup=1,
        num_steps=2,
    )

    assert isinstance(res, ProfilingResult)
    assert res.trainable_params > 0
    assert res.step_time_ms >= 0.0
