import torch
from cojgn import OsculatingJetLayer


def _orthogonal(d: int):
    q, _ = torch.linalg.qr(torch.randn(d, d))
    return q


def test_local_jet_orthogonal_covariance_single_anchor():
    torch.manual_seed(5)
    d = 6
    layer = OsculatingJetLayer(d, 3, num_anchors=1, curvature_rank=2, cubic_rank=2, metric_rank=2, order=3, dropout=0.0)
    layer.eval()
    x = torch.randn(9, d)
    Q = _orthogonal(d)
    ref = layer(x)

    clone = OsculatingJetLayer(d, 3, num_anchors=1, curvature_rank=2, cubic_rank=2, metric_rank=2, order=3, dropout=0.0)
    clone.load_state_dict(layer.state_dict())
    with torch.no_grad():
        # Row-vector coordinate convention: x' = x Q, vectors/factors transform by Q^T on stored feature axis.
        clone.anchors.copy_(layer.anchors @ Q)
        clone.w1.copy_(torch.einsum('kod,de->koe', layer.w1, Q))
        clone.metric_factor.copy_(torch.einsum('kdr,de->ker', layer.metric_factor, Q))
        clone.curvature_dirs.copy_(torch.einsum('kodr,de->koer', layer.curvature_dirs, Q))
        clone.cubic_dirs.copy_(torch.einsum('kodr,de->koer', layer.cubic_dirs, Q))
    out = clone(x @ Q)
    assert torch.allclose(ref, out, atol=2e-5, rtol=2e-5)
