import torch
from cojgn import OJGN, TrainConfig, fit_classifier, evaluate_classifier, set_seed

set_seed(7)
# Tiny nonlinear 3-class example.
x = torch.randn(600, 10)
y = torch.stack([
    x[:, 0] + 0.5 * x[:, 1] ** 2,
    -x[:, 0] + 0.3 * x[:, 2] ** 3,
    x[:, 3] - 0.4 * x[:, 4] * x[:, 5],
], dim=1).argmax(dim=1)

model = OJGN(
    in_features=10,
    hidden_features=24,
    out_features=3,
    num_layers=2,
    num_anchors=4,
    curvature_rank=3,
    cubic_rank=2,
    metric_rank=3,
    order=3,
    glue_order=2,
    dropout=0.0,
)

history = fit_classifier(
    model,
    x[:480], y[:480],
    TrainConfig(epochs=20, batch_size=64, lr=3e-3, geometry_weight=5e-4),
)
print(evaluate_classifier(model, x[480:], y[480:]))

# Inspect learned geometry on the held-out states.
with torch.no_grad():
    h = model.in_proj(x[480:])
    d = model.blocks[0].jet.diagnostics(model.blocks[0].norm(h))
print("routing entropy:", float(d.routing_entropy))
print("mean responsibility:", d.mean_responsibility.tolist())
