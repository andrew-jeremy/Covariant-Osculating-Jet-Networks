# COJGN: Covariant Osculating Jet Geometric Networks

Standalone PyTorch implementation of the architecture described in **Covariant Osculating Jet Networks**.

## Architecture

Each learned contact `c` uses local coordinates `delta = x - mu_c` and evaluates an order-3 jet

```text
q_c(x) = w0_c + J_c delta
       + 1/2 [sum_r lambda_cr (u_cr^T delta)^2 + alpha_c ||delta||^2]
       + 1/6 sum_s a_cs (t_cs^T delta)^3.
```

The layer routes each hidden state with a learned positive metric

```text
G_c = beta_c I + M_c M_c^T

gamma_c(x) = softmax_c(-(x-mu_c)^T G_c (x-mu_c)).
```

The output is `sum_c gamma_c q_c(x)`. The residual network stacks these layers behind pre-LayerNorm and SiLU.

### Implemented geometric mechanisms

- **Covariant metric routing:** low-rank Mahalanobis/Riemannian contact regions.
- **Signed principal curvature:** `U diag(lambda) U^T + alpha I`, allowing positive and negative off-axis curvature.
- **Order-3 jets:** symmetric low-rank CP cubic tensor `sum_s a_s (t_s^T delta)^3`.
- **Data-overlap gluing:** chart pairs are compared where `gamma_c gamma_d` is large.
- **C0/C1/C2 consistency:** matches values, Jacobians, and Hessians; C2 uses analytic Hessian-vector products with Hutchinson probes rather than dense Hessians.
- **Single-pass regularization:** task and geometry losses use the same hidden-state trajectory.
- **Stability controls:** positive softplus routing precision, metric/curvature/cubic penalties, routing balance, anchor separation, gradient clipping, residual pre-normalization.

The current implementation is exactly covariant under orthogonal changes of latent coordinates when learned vector/factor parameters are transformed accordingly. It does **not** claim invariance under arbitrary nonlinear coordinate changes.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[all]'
```

CPU PyTorch is sufficient for the included examples. CUDA/MPS can be used by passing a suitable `TrainConfig(device=...)`.

## Minimal use

```python
import torch
from cojgn import OJGN

model = OJGN(
    in_features=64,
    hidden_features=128,
    out_features=10,
    num_layers=3,
    num_anchors=6,
    curvature_rank=8,
    cubic_rank=4,
    metric_rank=8,
    order=3,
    glue_order=2,
)

x = torch.randn(32, 64)
logits, geometry = model(x, return_geometry=True)
reg, terms = model.geometry_regularization(geometry)
loss = torch.nn.functional.cross_entropy(logits, torch.randint(10, (32,))) + 1e-3 * reg
loss.backward()
```

## CLI benchmark

```bash
cojgn-train --dataset digits --epochs 40 --hidden 32 --layers 2 --anchors 4 \
  --curvature-rank 4 --cubic-rank 2 --metric-rank 4 --order 3 --glue-order 2
```

Available datasets: `digits`, `wine`, `breast_cancer`.

## Controlled third-order recovery benchmark

```bash
python benchmarks/jet_recovery.py --epochs 300 --seeds 3 --output results/jet_recovery.json
```

This target has known low-rank quadratic and cubic structure and is therefore a more direct test of the architectural hypothesis than ordinary tabular classification.

## Tests

```bash
pytest -q
```

Tests cover partition-of-unity routing, negative signed curvature, C2/HVP backpropagation, single-pass geometry regularization, orthogonal covariance, and a tiny training run.

## Repository layout

```text
src/cojgn/core.py        core OJGL/OJGN implementation
src/cojgn/training.py    training/evaluation utilities
src/cojgn/synthetic.py   controlled cubic-jet target generator
src/cojgn/cli.py         command-line benchmark entry point
examples/quickstart.py   minimal nonlinear classifier
benchmarks/              controlled experiments
tests/                   unit and smoke tests
configs/                  example settings
results/                  generated smoke/benchmark outputs
```

## Practical tuning

Start with `geometry_weight=1e-4` to `1e-3`. Strong C2 gluing can improve differential coherence while reducing task accuracy, so it should be tuned as a Pareto tradeoff rather than assumed to be an accuracy regularizer. For large hidden dimensions, keep `curvature_rank`, `cubic_rank`, and `metric_rank` much smaller than `hidden_features`.

## Complexity

For batch size `B`, anchors `K`, hidden/input dimension `D`, output dimension `O`, metric rank `Rg`, curvature rank `R2`, and cubic rank `R3`, the dominant factorized evaluation cost is approximately

```text
O(B K D Rg + B K O D (R2 + R3 + 1))
```

rather than materializing dense per-output quadratic and cubic tensors with `D^2` and `D^3` storage/evaluation.
