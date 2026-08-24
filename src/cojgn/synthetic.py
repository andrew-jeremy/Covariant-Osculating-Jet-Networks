from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass
class CubicJetTarget:
    linear: Tensor
    quad_dirs: Tensor
    quad_values: Tensor
    cubic_dirs: Tensor
    cubic_values: Tensor
    bias: Tensor

    def __call__(self, x: Tensor) -> Tensor:
        y = x @ self.linear.T + self.bias
        q = x @ self.quad_dirs
        y = y + 0.5 * (q.square() * self.quad_values).sum(-1, keepdim=True)
        c = x @ self.cubic_dirs
        y = y + (c.pow(3) * self.cubic_values).sum(-1, keepdim=True) / 6.0
        return y


def _unit_cols(x: Tensor) -> Tensor:
    return x / x.square().sum(0, keepdim=True).sqrt().clamp_min(1e-8)


def make_cubic_jet_regression(
    n_train: int = 512,
    n_test: int = 256,
    dim: int = 8,
    quad_rank: int = 2,
    cubic_rank: int = 2,
    noise: float = 0.01,
    seed: int = 0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, CubicJetTarget]:
    g = torch.Generator().manual_seed(seed)
    linear = torch.randn(1, dim, generator=g) * 0.35
    quad_dirs = _unit_cols(torch.randn(dim, quad_rank, generator=g))
    quad_values = torch.randn(quad_rank, generator=g) * 0.6
    cubic_dirs = _unit_cols(torch.randn(dim, cubic_rank, generator=g))
    cubic_values = torch.randn(cubic_rank, generator=g) * 0.7
    bias = torch.randn(1, generator=g) * 0.1
    target = CubicJetTarget(linear, quad_dirs, quad_values, cubic_dirs, cubic_values, bias)

    x_train = torch.randn(n_train, dim, generator=g) * 0.8
    x_test = torch.randn(n_test, dim, generator=g) * 0.8
    y_train = target(x_train)
    y_test = target(x_test)
    if noise > 0:
        y_train = y_train + noise * torch.randn(y_train.shape, generator=g)
        y_test = y_test + noise * torch.randn(y_test.shape, generator=g)
    return x_train, y_train, x_test, y_test, target
