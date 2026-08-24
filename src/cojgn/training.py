from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from .core import OJGN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-4
    geometry_weight: float = 1e-3
    grad_clip: float = 1.0
    device: str = "cpu"


def _loader(x: Tensor, y: Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def fit_classifier(
    model: nn.Module,
    x_train: Tensor,
    y_train: Tensor,
    config: TrainConfig | None = None,
) -> Dict[str, list[float]]:
    config = config or TrainConfig()
    device = torch.device(config.device)
    model.to(device)
    x_train, y_train = x_train.to(device), y_train.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    history: Dict[str, list[float]] = {"loss": [], "task": [], "geometry": []}

    for _ in range(config.epochs):
        model.train()
        loss_sum = task_sum = geo_sum = 0.0
        n = 0
        for xb, yb in _loader(x_train, y_train, config.batch_size, True):
            opt.zero_grad(set_to_none=True)
            if isinstance(model, OJGN):
                logits, geometry = model(xb, return_geometry=True)
                task = criterion(logits, yb)
                geo, _ = model.geometry_regularization(geometry)
                loss = task + config.geometry_weight * geo
            else:
                logits = model(xb)
                task = criterion(logits, yb)
                geo = task.new_zeros(())
                loss = task
            loss.backward()
            if config.grad_clip and config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            opt.step()
            bs = xb.shape[0]
            n += bs
            loss_sum += float(loss.detach()) * bs
            task_sum += float(task.detach()) * bs
            geo_sum += float(geo.detach()) * bs
        history["loss"].append(loss_sum / max(1, n))
        history["task"].append(task_sum / max(1, n))
        history["geometry"].append(geo_sum / max(1, n))
    return history


@torch.no_grad()
def evaluate_classifier(model: nn.Module, x: Tensor, y: Tensor, device: str = "cpu") -> Dict[str, float]:
    model.eval().to(device)
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)
    acc = (logits.argmax(-1) == y).float().mean()
    return {"loss": float(loss), "accuracy": float(acc)}
