from __future__ import annotations

import math
import time
from typing import Tuple

import numpy as np
import torch
from torch import nn
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from benchmarks.ir.builders import angle_embedding_ir, hardware_efficient_ansatz
from benchmarks.memory_utils import get_peak_rss_mb
from benchmarks.workloads import WorkloadResult


def prepare_dataset(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    features, labels = make_moons(n_samples=200, noise=0.1, random_state=seed)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    rng = np.random.default_rng(seed)
    indices = np.arange(features.shape[0])
    rng.shuffle(indices)
    return features[indices], labels[indices]


def run_classifier(backend, n_qubits: int, layers: int, seed: int) -> WorkloadResult:
    features, labels = prepare_dataset(seed)
    split = int(0.7 * len(features))
    train_x, test_x = features[:split], features[split:]
    train_y, test_y = labels[:split], labels[split:]

    feature_gates, next_index = angle_embedding_ir(num_features=min(2, n_qubits))
    ansatz_gates, total_index = hardware_efficient_ansatz(n_qubits, layers, start_index=next_index)
    gates = feature_gates + ansatz_gates
    num_weights = total_index - min(2, n_qubits)

    torch.manual_seed(seed)
    if num_weights <= 0:
        weights = nn.Parameter(torch.zeros(0, dtype=torch.float64))
    else:
        weights = nn.Parameter(0.1 * torch.randn(num_weights, dtype=torch.float64))
    optimizer = torch.optim.Adam([weights], lr=0.1)

    batch_size = min(32, len(train_x))
    rng = np.random.default_rng(seed)
    simulation_time = 0.0
    eps = 1e-9

    for step in range(10):
        idx = rng.choice(len(train_x), size=batch_size, replace=False)
        optimizer.zero_grad()
        loss = 0.0
        for i in idx:
            x = torch.tensor(train_x[i][: min(2, n_qubits)], dtype=torch.float64)
            theta = torch.cat([x, weights]) if weights.numel() > 0 else x
            start = time.perf_counter()
            state = backend.simulate_state(n_qubits, gates, theta, seed)
            simulation_time += time.perf_counter() - start
            prob_zero = torch.abs(state[0]) ** 2
            prob_one = 1 - prob_zero
            target = torch.tensor(float(train_y[i]), dtype=torch.float64)
            loss = loss + (-(target * torch.log(prob_one + eps) + (1 - target) * torch.log(prob_zero + eps)))
        loss = loss / len(idx)
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            return WorkloadResult(
                execution_time_s=simulation_time,
                peak_memory_mb=get_peak_rss_mb(),
                accuracy_name="test_accuracy_pct",
                accuracy_value=math.nan,
                success=False,
                error="no_autograd",
            )
        loss.backward()
        optimizer.step()

    correct = 0
    with torch.no_grad():
        weight_values = weights.detach()
        for x, y in zip(test_x, test_y):
            feature_tensor = torch.tensor(x[: min(2, n_qubits)], dtype=torch.float64)
            theta = torch.cat([feature_tensor, weight_values]) if weight_values.numel() > 0 else feature_tensor
            start = time.perf_counter()
            state = backend.simulate_state(n_qubits, gates, theta, seed)
            simulation_time += time.perf_counter() - start
            prob_zero = torch.abs(state[0]) ** 2
            pred = 1 if prob_zero > 0.5 else 0
            if pred == int(y):
                correct += 1

    accuracy = 100.0 * correct / len(test_x)
    peak_mem = get_peak_rss_mb()

    return WorkloadResult(
        execution_time_s=simulation_time,
        peak_memory_mb=peak_mem,
        accuracy_name="test_accuracy_pct",
        accuracy_value=accuracy,
        success=True,
        error=None,
    )
