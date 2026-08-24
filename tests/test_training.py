import torch
from cojgn import OJGN, TrainConfig, fit_classifier, evaluate_classifier, set_seed


def test_tiny_training_smoke():
    set_seed(11)
    x = torch.randn(120, 6)
    y = ((x[:,0] + x[:,1]**2 - 0.2*x[:,2]**3) > 0).long()
    model = OJGN(6, 12, 2, num_layers=1, num_anchors=3, curvature_rank=2, cubic_rank=2, metric_rank=2, dropout=0.0)
    hist = fit_classifier(model, x[:96], y[:96], TrainConfig(epochs=3, batch_size=32, lr=3e-3, geometry_weight=1e-4))
    m = evaluate_classifier(model, x[96:], y[96:])
    assert len(hist['loss']) == 3
    assert 0.0 <= m['accuracy'] <= 1.0
