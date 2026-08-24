# COJGN How-To

## 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[all]'
```

If all dependencies are already installed and the machine is offline, use:

```bash
pip install -e . --no-deps --no-build-isolation
```

## 2. Run the tests

```bash
pytest -q
```

## 3. Run the quickstart

```bash
python examples/quickstart.py
```

## 4. Train on a standard dataset

```bash
cojgn-train --dataset digits \
  --epochs 40 \
  --hidden 32 \
  --layers 2 \
  --anchors 4 \
  --curvature-rank 4 \
  --cubic-rank 2 \
  --metric-rank 4 \
  --order 3 \
  --glue-order 2 \
  --geometry-weight 0.001 \
  --output results/digits.json
```

## 5. Use the geometry loss correctly

Always request geometry from the same task forward pass:

```python
logits, geometry = model(x, return_geometry=True)
task_loss = criterion(logits, y)
geometry_loss, terms = model.geometry_regularization(geometry)
loss = task_loss + 1e-3 * geometry_loss
loss.backward()
```

This avoids re-running dropout and routing when forming the auxiliary loss.

## 6. Select jet order

- `order=1`: learned contact values + Jacobians only.
- `order=2`: adds signed low-rank principal curvature.
- `order=3`: adds low-rank cubic curvature variation.

Parameters belonging to unused higher orders are frozen, so `parameter_count()` reports only trainable capacity.

## 7. Select gluing order

- `glue_order=0`: local function values agree in overlap.
- `glue_order=1`: values + Jacobians agree.
- `glue_order=2`: values + Jacobians + Hessians agree.

C2 is estimated with Hessian-vector products; dense Hessians are not constructed.

## 8. Tune the model

A practical starting range is:

```text
num_anchors       4-8
curvature_rank    2-8
cubic_rank        1-4
metric_rank       2-8
geometry_weight   1e-4 to 1e-3
```

Strong gluing can improve contact coherence while hurting task loss. Treat the geometry coefficient as a multi-objective tradeoff.

## 9. Inspect learned geometry

```python
h = model.in_proj(x)
xn = model.blocks[0].norm(h)
d = model.blocks[0].jet.diagnostics(xn)
print(d.routing_entropy)
print(d.mean_responsibility)
print(d.signed_curvature_rms)
print(d.cubic_coefficient_rms)
```

## 10. Run the controlled cubic benchmark

```bash
python benchmarks/jet_recovery.py \
  --epochs 300 \
  --seeds 3 \
  --output results/jet_recovery.json
```

This benchmark is designed to test whether the order-3 parameterization recovers known low-rank cubic structure.
