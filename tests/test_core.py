import torch
from cojgn import OJGN, OsculatingJetLayer


def test_routing_is_partition_of_unity():
    torch.manual_seed(0)
    layer = OsculatingJetLayer(7, 5, num_anchors=4, curvature_rank=3, cubic_rank=2, metric_rank=3)
    x = torch.randn(11, 7)
    gamma = layer.anchor_weights(x)
    assert gamma.shape == (11, 4)
    assert torch.isfinite(gamma).all()
    assert torch.allclose(gamma.sum(-1), torch.ones(11), atol=1e-6)


def test_signed_curvature_can_be_negative():
    torch.manual_seed(1)
    layer = OsculatingJetLayer(3, 1, num_anchors=1, curvature_rank=1, cubic_rank=1, metric_rank=1, order=2, bias=False)
    with torch.no_grad():
        layer.anchors.zero_(); layer.w0.zero_(); layer.w1.zero_(); layer.curvature_iso.zero_()
        layer.curvature_dirs.zero_(); layer.curvature_dirs[0,0,0,0] = 1.0
        layer.curvature_values.fill_(-2.0)
    x = torch.tensor([[1.0, 0.0, 0.0]])
    y = layer(x)
    assert y.item() < 0.0


def test_c2_overlap_backward():
    torch.manual_seed(2)
    layer = OsculatingJetLayer(8, 6, num_anchors=3, curvature_rank=2, cubic_rank=2, metric_rank=2, order=3)
    x = torch.randn(12, 8)
    loss = layer.overlap_consistency_loss(x, glue_order=2, num_hutchinson=2)
    assert torch.isfinite(loss)
    loss.backward()
    assert layer.w1.grad is not None
    assert layer.curvature_values.grad is not None
    assert layer.cubic_values.grad is not None


def test_single_pass_model_geometry():
    torch.manual_seed(3)
    model = OJGN(10, 16, 4, num_layers=2, num_anchors=4, curvature_rank=3, cubic_rank=2, metric_rank=3)
    x = torch.randn(20, 10)
    logits, geometry = model(x, return_geometry=True)
    reg, terms = model.geometry_regularization(geometry)
    (logits.square().mean() + 1e-3 * reg).backward()
    assert logits.shape == (20,4)
    assert len(geometry) == 2
    assert torch.isfinite(reg)
    assert torch.isfinite(terms["overlap_consistency"])
