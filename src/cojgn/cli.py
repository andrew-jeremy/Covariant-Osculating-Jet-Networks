from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .core import OJGN, parameter_count
from .training import TrainConfig, evaluate_classifier, fit_classifier, set_seed


def load_dataset(name: str, seed: int):
    loaders = {"digits": load_digits, "wine": load_wine, "breast_cancer": load_breast_cancer}
    data = loaders[name]()
    x = StandardScaler().fit_transform(data.data).astype(np.float32)
    y = data.target.astype(np.int64)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    return map(torch.from_numpy, (xtr, xte, ytr, yte)), len(np.unique(y))


def main() -> None:
    p = argparse.ArgumentParser(description="Train a Covariant Osculating Jet Geometric Network")
    p.add_argument("--dataset", choices=["digits", "wine", "breast_cancer"], default="digits")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--anchors", type=int, default=4)
    p.add_argument("--curvature-rank", type=int, default=4)
    p.add_argument("--cubic-rank", type=int, default=2)
    p.add_argument("--metric-rank", type=int, default=4)
    p.add_argument("--order", type=int, choices=[1,2,3], default=3)
    p.add_argument("--glue-order", type=int, choices=[0,1,2], default=2)
    p.add_argument("--geometry-weight", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    set_seed(args.seed)
    tensors, n_classes = load_dataset(args.dataset, args.seed)
    xtr, xte, ytr, yte = tensors
    model = OJGN(
        xtr.shape[1], args.hidden, n_classes,
        num_layers=args.layers, num_anchors=args.anchors,
        curvature_rank=args.curvature_rank, cubic_rank=args.cubic_rank,
        metric_rank=args.metric_rank, order=args.order, glue_order=args.glue_order,
    )
    cfg = TrainConfig(epochs=args.epochs, geometry_weight=args.geometry_weight)
    history = fit_classifier(model, xtr, ytr, cfg)
    metrics = evaluate_classifier(model, xte, yte)
    result = {
        "dataset": args.dataset,
        "seed": args.seed,
        "parameters": parameter_count(model),
        "metrics": metrics,
        "final_training": {k: v[-1] for k, v in history.items()},
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
