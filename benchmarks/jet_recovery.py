from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from cojgn import OsculatingJetLayer, make_cubic_jet_regression, parameter_count, set_seed


def fit(order: int, seed: int, epochs: int):
    set_seed(seed)
    xtr, ytr, xte, yte, _ = make_cubic_jet_regression(seed=seed)
    model = OsculatingJetLayer(
        xtr.shape[1], 1, num_anchors=1,
        curvature_rank=2, cubic_rank=2, metric_rank=2,
        order=order, dropout=0.0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=8e-3)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        mse = nn.functional.mse_loss(model(xte), yte).item()
    return {"order": order, "seed": seed, "test_mse": mse, "parameters": parameter_count(model)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    rows = [fit(order, seed, args.epochs) for order in (1,2,3) for seed in range(args.seeds)]
    print(json.dumps(rows, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n")

if __name__ == "__main__":
    main()
