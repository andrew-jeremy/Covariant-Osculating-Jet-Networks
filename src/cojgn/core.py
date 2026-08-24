"""Covariant Osculating Jet Geometric Network (COJGN).

A production-oriented order-3 local jet network with:
* low-rank Riemannian/Mahalanobis routing metrics G_c = beta_c I + M_c M_c^T;
* signed principal-curvature factors H_{c,o} = U diag(lambda) U^T + alpha I;
* symmetric CP third-order jets T[delta^3] = sum_s a_s (t_s^T delta)^3;
* data-overlap C0/C1/C2 consistency, with a Hutchinson HVP estimator for C2;
* single-pass geometry contexts so regularization uses the same hidden trajectory as task loss.

The parameterization is exactly covariant under orthogonal reparameterizations of the
latent coordinates when anchors/factors are transformed accordingly.  It is not
claimed to be invariant under arbitrary nonlinear coordinate changes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _inverse_softplus(x: float) -> float:
    return math.log(math.expm1(x))


def _unit_columns(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Normalize the penultimate (feature) dimension column-wise."""
    return x / x.square().sum(dim=-2, keepdim=True).add(eps).sqrt()


@dataclass
class LayerGeometry:
    x: Tensor
    gamma: Tensor


@dataclass
class OJGLDiagnostics:
    routing_entropy: Tensor
    mean_responsibility: Tensor
    min_anchor_distance: Tensor
    metric_factor_rms: Tensor
    signed_curvature_rms: Tensor
    cubic_coefficient_rms: Tensor
    effective_overlap: Tensor


class OsculatingJetLayer(nn.Module):
    """Metric-routed signed-curvature order-3 Osculating Jet Layer.

    For anchor c and output o, with delta = x - mu_c,

      q_{c,o}(x) = w0 + J delta
                 + 1/2 [sum_r lambda_r (u_r^T delta)^2 + alpha ||delta||^2]
                 + 1/6 sum_s a_s (t_s^T delta)^3.

    Routing uses a learned positive metric

      d_c^2 = beta_c ||delta||^2 + ||M_c^T delta||^2,
      gamma_c = softmax(-d_c^2).

    The low-rank metric and curvature parameterizations avoid D x D tensors while
    retaining signed off-axis curvature and anisotropic contact regions.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_anchors: int = 4,
        curvature_rank: int = 4,
        cubic_rank: int = 2,
        metric_rank: int = 4,
        order: int = 3,
        use_metric_routing: bool = True,
        bias: bool = True,
        dropout: float = 0.0,
        precision_init: float = 0.7,
        precision_floor: float = 1e-4,
        max_metric_distance_sq: Optional[float] = 1e3,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0 or num_anchors <= 0:
            raise ValueError("feature dimensions and num_anchors must be positive")
        if order not in (1, 2, 3):
            raise ValueError("order must be 1, 2, or 3")
        if min(curvature_rank, cubic_rank, metric_rank) <= 0:
            raise ValueError("ranks must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_anchors = int(num_anchors)
        self.curvature_rank = min(int(curvature_rank), self.in_features)
        self.cubic_rank = min(int(cubic_rank), self.in_features)
        self.metric_rank = min(int(metric_rank), self.in_features)
        self.order = int(order)
        self.use_metric_routing = bool(use_metric_routing)
        self.precision_floor = float(precision_floor)
        self.max_metric_distance_sq = max_metric_distance_sq

        K, D, O = self.num_anchors, self.in_features, self.out_features
        R2, R3, Rg = self.curvature_rank, self.cubic_rank, self.metric_rank

        self.anchors = nn.Parameter(torch.empty(K, D))
        self.log_precision = nn.Parameter(torch.full((K,), _inverse_softplus(precision_init)))
        self.metric_factor = nn.Parameter(torch.empty(K, D, Rg))

        self.w0 = nn.Parameter(torch.empty(K, O))
        self.w1 = nn.Parameter(torch.empty(K, O, D))

        # Signed principal curvature: U diag(lambda) U^T + alpha I.
        self.curvature_dirs = nn.Parameter(torch.empty(K, O, D, R2))
        self.curvature_values = nn.Parameter(torch.empty(K, O, R2))
        self.curvature_iso = nn.Parameter(torch.empty(K, O))

        # Symmetric CP representation of the cubic tensor.
        self.cubic_dirs = nn.Parameter(torch.empty(K, O, D, R3))
        self.cubic_values = nn.Parameter(torch.empty(K, O, R3))

        if bias:
            self.bias = nn.Parameter(torch.empty(O))
        else:
            self.register_parameter("bias", None)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

        # Keep lower-order ablations parameter-honest: parameters for terms that
        # are not evaluated are frozen and excluded from trainable counts.
        if self.order < 2:
            self.curvature_dirs.requires_grad_(False)
            self.curvature_values.requires_grad_(False)
            self.curvature_iso.requires_grad_(False)
        if self.order < 3:
            self.cubic_dirs.requires_grad_(False)
            self.cubic_values.requires_grad_(False)
        if not self.use_metric_routing:
            self.metric_factor.requires_grad_(False)

    @property
    def precision(self) -> Tensor:
        return F.softplus(self.log_precision) + self.precision_floor

    @property
    def U2(self) -> Tensor:
        return _unit_columns(self.curvature_dirs)

    @property
    def U3(self) -> Tensor:
        return _unit_columns(self.cubic_dirs)

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.anchors)
        nn.init.normal_(self.metric_factor, std=0.04)
        nn.init.kaiming_uniform_(self.w0, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.normal_(self.curvature_dirs, std=0.1)
        nn.init.normal_(self.curvature_values, std=0.01)
        nn.init.zeros_(self.curvature_iso)
        nn.init.normal_(self.cubic_dirs, std=0.1)
        nn.init.normal_(self.cubic_values, std=0.005)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def local_coordinates(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[-1] != self.in_features:
            raise ValueError(f"expected (B,{self.in_features}), got {tuple(x.shape)}")
        return x[:, None, :] - self.anchors[None, :, :]

    def metric_distances(self, x: Tensor) -> Tensor:
        delta = self.local_coordinates(x)
        base = self.precision[None, :] * delta.square().sum(-1)
        if self.use_metric_routing:
            anis = torch.einsum("bkd,kdr->bkr", delta, self.metric_factor).square().sum(-1)
            dist = base + anis
        else:
            dist = base
        if self.max_metric_distance_sq is not None:
            dist = dist.clamp_max(float(self.max_metric_distance_sq))
        return dist

    def anchor_weights(self, x: Tensor) -> Tensor:
        logits = -self.metric_distances(x)
        logits = logits - logits.amax(dim=-1, keepdim=True)
        return F.softmax(logits, dim=-1)

    def local_quantities(
        self,
        x: Tensor,
        probe: Optional[Tensor] = None,
        need_grad: bool = False,
        need_hvp: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Evaluate local jets and optionally analytic gradients/HVPs.

        `probe` is (B,D) and is shared across charts/outputs.  For a standard
        Gaussian or Rademacher probe z, E|| (H_a-H_b)z ||^2 equals the Frobenius
        norm squared of the Hessian difference, giving a scalable C2 estimator.
        """
        delta = self.local_coordinates(x)  # B,K,D
        q = self.w0[None] + torch.einsum("bkd,kod->bko", delta, self.w1)
        grad = self.w1[None].expand(x.shape[0], -1, -1, -1).clone() if need_grad else None
        hvp = None

        if self.order >= 2:
            U2 = self.U2
            z2 = torch.einsum("bkd,kodr->bkor", delta, U2)
            q2 = 0.5 * (self.curvature_values[None] * z2.square()).sum(-1)
            q2 = q2 + 0.5 * self.curvature_iso[None] * delta.square().sum(-1)[:, :, None]
            q = q + q2
            if need_grad:
                grad2 = torch.einsum(
                    "kodr,bkor->bkod", U2, self.curvature_values[None] * z2
                )
                grad2 = grad2 + self.curvature_iso[None, :, :, None] * delta[:, :, None, :]
                grad = grad + grad2

        if self.order >= 3:
            U3 = self.U3
            z3 = torch.einsum("bkd,kods->bkos", delta, U3)
            q3 = (self.cubic_values[None] * z3.pow(3)).sum(-1) / 6.0
            q = q + q3
            if need_grad:
                grad3 = 0.5 * torch.einsum(
                    "kods,bkos->bkod", U3, self.cubic_values[None] * z3.square()
                )
                grad = grad + grad3

        if need_hvp:
            if probe is None:
                raise ValueError("probe is required when need_hvp=True")
            if probe.shape != x.shape:
                raise ValueError("probe must have the same shape as x")
            B = x.shape[0]
            p = probe
            hvp = x.new_zeros(B, self.num_anchors, self.out_features, self.in_features)
            if self.order >= 2:
                U2 = self.U2
                up = torch.einsum("bd,kodr->bkor", p, U2)
                hvp = hvp + torch.einsum(
                    "kodr,bkor->bkod", U2, self.curvature_values[None] * up
                )
                hvp = hvp + self.curvature_iso[None, :, :, None] * p[:, None, None, :]
            if self.order >= 3:
                U3 = self.U3
                z3 = torch.einsum("bkd,kods->bkos", delta, U3)
                tp = torch.einsum("bd,kods->bkos", p, U3)
                coeff = self.cubic_values[None] * z3 * tp
                hvp = hvp + torch.einsum("kods,bkos->bkod", U3, coeff)
        return q, grad, hvp

    def local_jets(self, x: Tensor) -> Tensor:
        return self.local_quantities(x)[0]

    def forward(self, x: Tensor, return_routing: bool = False):
        gamma = self.anchor_weights(x)
        local = self.local_jets(x)
        y = torch.einsum("bk,bko->bo", gamma, local)
        if self.bias is not None:
            y = y + self.bias
        y = self.dropout(y)
        return (y, gamma) if return_routing else y

    def overlap_consistency_loss(
        self,
        x: Tensor,
        gamma: Optional[Tensor] = None,
        glue_order: int = 2,
        jacobian_weight: float = 0.25,
        hessian_weight: float = 0.10,
        min_pair_overlap: float = 1e-5,
        max_pairs: Optional[int] = None,
        num_hutchinson: int = 1,
    ) -> Tensor:
        """Data-overlap C0/C1/C2 contact consistency.

        Pair weights are proportional to gamma_c(x) gamma_d(x) on the *actual
        hidden states*.  We detach these weights so the router cannot reduce the
        gluing penalty simply by destroying overlap.  C2 uses analytic HVPs with
        Hutchinson probes instead of materializing dense Hessians.
        """
        if self.num_anchors < 2 or glue_order < 0:
            return x.new_zeros(())
        if glue_order not in (0, 1, 2):
            raise ValueError("glue_order must be 0,1,2")
        gamma = self.anchor_weights(x) if gamma is None else gamma
        pairs = torch.triu_indices(self.num_anchors, self.num_anchors, 1, device=x.device)
        a, b = pairs[0], pairs[1]
        overlap = (gamma[:, a] * gamma[:, b]).detach()  # B,P
        pair_score = overlap.mean(0)
        valid = pair_score > min_pair_overlap
        if not valid.any():
            return x.new_zeros(())
        a, b, overlap, pair_score = a[valid], b[valid], overlap[:, valid], pair_score[valid]
        if max_pairs is not None and a.numel() > max_pairs:
            keep = pair_score.topk(max_pairs).indices
            a, b, overlap = a[keep], b[keep], overlap[:, keep]
        w = overlap / overlap.sum(dim=0, keepdim=True).clamp_min(1e-12)

        need_grad = glue_order >= 1
        q, grad, _ = self.local_quantities(x, need_grad=need_grad)
        value_err = (q[:, a] - q[:, b]).square().mean(-1)  # B,P
        total = (w * value_err).sum(0).mean()
        if glue_order >= 1 and grad is not None:
            jac_err = (grad[:, a] - grad[:, b]).square().mean((-1, -2))
            total = total + jacobian_weight * (w * jac_err).sum(0).mean()
        if glue_order >= 2:
            hacc = x.new_zeros(())
            for _ in range(max(1, num_hutchinson)):
                # Rademacher probes make E[zz^T]=I and are cheap/reproducible.
                probe = torch.empty_like(x).bernoulli_(0.5).mul_(2).sub_(1)
                _, _, hvp = self.local_quantities(x, probe=probe, need_hvp=True)
                herr = (hvp[:, a] - hvp[:, b]).square().mean((-1, -2))
                hacc = hacc + (w * herr).sum(0).mean()
            total = total + hessian_weight * hacc / max(1, num_hutchinson)
        return total

    def regularization_terms(
        self,
        x: Optional[Tensor] = None,
        gamma: Optional[Tensor] = None,
        glue_order: int = 2,
        separation_margin: float = 0.75,
    ) -> Dict[str, Tensor]:
        curvature = self.curvature_values.square().mean() + self.curvature_iso.square().mean()
        cubic = self.cubic_values.square().mean() if self.order >= 3 else self.anchors.new_zeros(())
        metric = self.metric_factor.square().mean() + 0.02 * self.log_precision.square().mean()
        if self.num_anchors > 1:
            anchor_separation = F.relu(separation_margin - torch.pdist(self.anchors)).square().mean()
        else:
            anchor_separation = self.anchors.new_zeros(())
        if x is not None:
            gamma = self.anchor_weights(x) if gamma is None else gamma
            mean_resp = gamma.mean(0)
            target = torch.full_like(mean_resp, 1.0 / self.num_anchors)
            routing_balance = (mean_resp - target).square().mean()
            overlap = self.overlap_consistency_loss(x, gamma=gamma, glue_order=glue_order)
        else:
            routing_balance = self.anchors.new_zeros(())
            overlap = self.anchors.new_zeros(())
        return {
            "curvature": curvature,
            "cubic": cubic,
            "metric": metric,
            "anchor_separation": anchor_separation,
            "routing_balance": routing_balance,
            "overlap_consistency": overlap,
        }

    @torch.no_grad()
    def diagnostics(self, x: Tensor) -> OJGLDiagnostics:
        gamma = self.anchor_weights(x)
        entropy = -(gamma.clamp_min(1e-12) * gamma.clamp_min(1e-12).log()).sum(-1).mean()
        min_dist = torch.pdist(self.anchors).min() if self.num_anchors > 1 else x.new_tensor(float("inf"))
        pairs = torch.triu_indices(self.num_anchors, self.num_anchors, 1, device=x.device)
        if pairs.shape[1]:
            overlap = (gamma[:, pairs[0]] * gamma[:, pairs[1]]).sum(-1).mean()
        else:
            overlap = x.new_zeros(())
        return OJGLDiagnostics(
            routing_entropy=entropy,
            mean_responsibility=gamma.mean(0),
            min_anchor_distance=min_dist,
            metric_factor_rms=self.metric_factor.square().mean().sqrt(),
            signed_curvature_rms=self.curvature_values.square().mean().sqrt(),
            cubic_coefficient_rms=self.cubic_values.square().mean().sqrt(),
            effective_overlap=overlap,
        )


class OsculatingJetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_anchors: int = 4,
        curvature_rank: int = 4,
        cubic_rank: int = 2,
        metric_rank: int = 4,
        order: int = 3,
        use_metric_routing: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.jet = OsculatingJetLayer(
            dim, dim, num_anchors=num_anchors,
            curvature_rank=curvature_rank, cubic_rank=cubic_rank,
            metric_rank=metric_rank, order=order,
            use_metric_routing=use_metric_routing, dropout=dropout,
        )
        self.act = nn.SiLU()

    def forward(self, x: Tensor, return_geometry: bool = False):
        xn = self.norm(x)
        y, gamma = self.jet(xn, return_routing=True)
        h = x + self.act(y)
        return (h, LayerGeometry(x=xn, gamma=gamma)) if return_geometry else h


class OJGN(nn.Module):
    """Covariant Osculating Jet Geometric Network (COJGN/OJGN-3)."""
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 2,
        num_anchors: int = 4,
        curvature_rank: int = 4,
        cubic_rank: int = 2,
        metric_rank: int = 4,
        order: int = 3,
        use_metric_routing: bool = True,
        glue_order: int = 2,
        dropout: float = 0.05,
        # Legacy keyword retained for old experiment scripts.
        rank: Optional[int] = None,
    ) -> None:
        super().__init__()
        if rank is not None:
            curvature_rank = rank
        self.order = order
        self.glue_order = glue_order
        self.in_proj = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList([
            OsculatingJetBlock(
                hidden_features, num_anchors=num_anchors,
                curvature_rank=curvature_rank, cubic_rank=cubic_rank,
                metric_rank=metric_rank, order=order,
                use_metric_routing=use_metric_routing, dropout=dropout,
            ) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_features)
        self.head = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor, return_geometry: bool = False):
        h = self.in_proj(x)
        geometry: List[LayerGeometry] = []
        for block in self.blocks:
            if return_geometry:
                h, ctx = block(h, return_geometry=True)
                geometry.append(ctx)
            else:
                h = block(h)
        logits = self.head(self.final_norm(h))
        return (logits, geometry) if return_geometry else logits

    def geometry_regularization(
        self,
        geometry: Sequence[LayerGeometry],
        curvature_weight: float = 1.0,
        cubic_weight: float = 0.5,
        metric_weight: float = 0.05,
        separation_weight: float = 0.2,
        balance_weight: float = 0.2,
        overlap_weight: float = 0.1,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if len(geometry) != len(self.blocks):
            raise ValueError("geometry context must come from the same OJGN forward pass")
        totals: Dict[str, Tensor] = {}
        reg = self.head.weight.new_zeros(())
        weights = {
            "curvature": curvature_weight,
            "cubic": cubic_weight,
            "metric": metric_weight,
            "anchor_separation": separation_weight,
            "routing_balance": balance_weight,
            "overlap_consistency": overlap_weight,
        }
        for block, ctx in zip(self.blocks, geometry):
            terms = block.jet.regularization_terms(ctx.x, gamma=ctx.gamma, glue_order=self.glue_order)
            for name, value in terms.items():
                totals[name] = totals.get(name, value.new_zeros(())) + value
                reg = reg + weights[name] * value
        totals["total"] = reg
        return reg, totals

    def regularization_loss(self, x: Optional[Tensor] = None, **kwargs) -> Tensor:
        """Compatibility wrapper.

        New training code should call `forward(..., return_geometry=True)` followed
        by `geometry_regularization(...)` to avoid a second stochastic forward.
        """
        if x is None:
            reg = self.head.weight.new_zeros(())
            for block in self.blocks:
                terms = block.jet.regularization_terms(None, glue_order=self.glue_order)
                reg = reg + terms["curvature"] + 0.5 * terms["cubic"] + 0.05 * terms["metric"]
            return reg
        _, geometry = self.forward(x, return_geometry=True)
        # Map legacy names if provided.
        mapped = {}
        if "contact_weight" in kwargs:
            mapped["curvature_weight"] = kwargs.pop("contact_weight")
        if "separation_weight" in kwargs:
            mapped["separation_weight"] = kwargs.pop("separation_weight")
        if "balance_weight" in kwargs:
            mapped["balance_weight"] = kwargs.pop("balance_weight")
        if "overlap_weight" in kwargs:
            mapped["overlap_weight"] = kwargs.pop("overlap_weight")
        mapped.update(kwargs)
        return self.geometry_regularization(geometry, **mapped)[0]

    def contact_regularization_loss(self) -> Tensor:
        return self.regularization_loss(None)


class MLPBaseline(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int,
                 num_layers: int = 2, dropout: float = 0.05) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_features), nn.Linear(hidden_features, hidden_features),
                          nn.SiLU(), nn.Dropout(dropout)) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_features)
        self.head = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        h = self.in_proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.head(self.final_norm(h))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Explicit modern names while retaining the historical OJGL/OJGN API.
CovariantOsculatingJetLayer = OsculatingJetLayer
COJGN = OJGN
