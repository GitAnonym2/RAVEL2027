"""RAVEL paper reference implementation."""

from .config import (
    ModelConfig,
    OptimizationConfig,
    RavelConfig,
    SpectralConfig,
    SplitConfig,
    paper_config,
)
from .io import (
    load_physics_json,
    load_pickle,
    save_config,
    save_physics_json,
    save_pickle,
    save_shared_fit,
)
from .metrics import (
    EvaluationCovarianceCache,
    complex_gaussian_trial_nll,
    predict_positions,
    score_positions,
    score_session_model,
)
from .model import atom_response, fit_session_model, refit_recruitment_only
from .optimize import (
    default_atoms,
    evaluate_dictionary,
    fit_context_scratch_shared_route,
    fit_independent_session,
    fit_selected_shared_route,
    fit_task_transfer,
    lock_route_specification,
    match_atoms,
    score_task_transfer,
)
from .preprocess import build_route_session_from_epochs, windowed_multitaper
from .presets import load_initial_atom_presets
from .types import AtomSpec, ContextWindowed, RouteSession, SessionFit, SharedRouteFit

__version__ = "1.0.0"

__all__ = [
    "AtomSpec",
    "ContextWindowed",
    "EvaluationCovarianceCache",
    "ModelConfig",
    "OptimizationConfig",
    "RavelConfig",
    "RouteSession",
    "SessionFit",
    "SharedRouteFit",
    "SpectralConfig",
    "SplitConfig",
    "atom_response",
    "build_route_session_from_epochs",
    "complex_gaussian_trial_nll",
    "default_atoms",
    "evaluate_dictionary",
    "fit_context_scratch_shared_route",
    "fit_independent_session",
    "fit_selected_shared_route",
    "fit_session_model",
    "fit_task_transfer",
    "load_physics_json",
    "load_initial_atom_presets",
    "load_pickle",
    "lock_route_specification",
    "match_atoms",
    "paper_config",
    "predict_positions",
    "refit_recruitment_only",
    "save_config",
    "save_physics_json",
    "save_pickle",
    "save_shared_fit",
    "score_positions",
    "score_session_model",
    "score_task_transfer",
    "windowed_multitaper",
]
