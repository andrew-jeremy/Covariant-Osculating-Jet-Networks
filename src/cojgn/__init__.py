"""Covariant Osculating Jet Geometric Networks (COJGN)."""
from .core import (
    COJGN,
    OJGN,
    CovariantOsculatingJetLayer,
    OsculatingJetLayer,
    OsculatingJetBlock,
    MLPBaseline,
    LayerGeometry,
    OJGLDiagnostics,
    parameter_count,
)
from .training import TrainConfig, fit_classifier, evaluate_classifier, set_seed
from .synthetic import CubicJetTarget, make_cubic_jet_regression

__all__ = [
    "COJGN",
    "OJGN",
    "CovariantOsculatingJetLayer",
    "OsculatingJetLayer",
    "OsculatingJetBlock",
    "MLPBaseline",
    "LayerGeometry",
    "OJGLDiagnostics",
    "parameter_count",
    "TrainConfig",
    "fit_classifier",
    "evaluate_classifier",
    "set_seed",
    "CubicJetTarget",
    "make_cubic_jet_regression",
]

__version__ = "0.1.0"
