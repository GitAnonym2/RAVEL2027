"""Common predictive metrics for this version and external baselines."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import RavelConfig, paper_config
from .model import coordinate_prediction, transfer_from_coefficients, window_kernels
from .types import RouteSession, SessionFit


class EvaluationCovarianceCache:
    """Cache the fixed training-target covariance used by predictive NLL."""

    def __init__(self) -> None:
        self._cache: dict[tuple[Any, ...], np.ndarray] = {}

    def clear(self) -> None:
        self._cache.clear()

    def get(
        self,
        data: RouteSession,
        context: str,
        config: RavelConfig | None = None,
    ) -> np.ndarray:
        key = (
            id(data),
            data.route_name,
            str(data.session_id),
            context,
            data.representation_id,
        )
        if key not in self._cache:
            self._cache[key] = fit_evaluation_covariance(data, context, config)
        return self._cache[key]


DEFAULT_COVARIANCE_CACHE = EvaluationCovarianceCache()


def fit_evaluation_covariance(
    data: RouteSession,
    context: str,
    config: RavelConfig | None = None,
) -> np.ndarray:
    """Fit frequency-wise channel covariance on training targets only."""

    cfg = config or paper_config()
    context_data = data.contexts[context]
    if data.evaluation_positions is not None and context in data.evaluation_positions:
        positions = np.asarray(data.evaluation_positions[context], dtype=int)
    else:
        positions = data.positions(context, "train")
    observed = np.asarray(context_data.target_raw[positions])
    n_freq, n_channels = observed.shape[3], observed.shape[4]
    covariance = np.empty((n_freq, n_channels, n_channels), complex)
    for frequency in range(n_freq):
        values = observed[:, :, :, frequency].reshape(-1, n_channels)
        cov = values.conj().T @ values / max(len(values), 1)
        diagonal = np.diag(np.diag(cov))
        cov = (
            (1.0 - cfg.model.residual_shrinkage) * cov
            + cfg.model.residual_shrinkage * diagonal
        )
        scale = max(float(np.real(np.trace(cov))) / n_channels, 1e-10)
        covariance[frequency] = (
            0.5 * (cov + cov.conj().T)
            + 1e-5 * scale * np.eye(n_channels)
        )
    return covariance


def complex_gaussian_trial_nll(
    observed: np.ndarray,
    predicted: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Trial-level conditional complex-Gaussian reference score."""

    error = np.asarray(observed - predicted)
    trial_score = np.zeros(len(error), float)
    for frequency in range(error.shape[3]):
        cov = covariance[frequency]
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise RuntimeError("evaluation covariance is not positive definite")
        inverse = np.linalg.inv(cov)
        values = error[:, :, :, frequency]
        quadratic = np.real(
            np.einsum(
                "twkc,cd,twkd->twk",
                values.conj(),
                inverse,
                values,
                optimize=True,
            )
        )
        trial_score += logdet + quadratic.mean(axis=(1, 2))
    return trial_score / error.shape[3]


def predict_positions(
    data: RouteSession,
    model: SessionFit,
    context: str,
    positions: np.ndarray,
    *,
    recruitment_override: np.ndarray | None = None,
    config: RavelConfig | None = None,
) -> np.ndarray:
    cfg = config or paper_config()
    positions = np.asarray(positions, dtype=int)
    source = data.contexts[context].source_coord[positions]
    if recruitment_override is None:
        transfer = model.transfer
    else:
        kernels = window_kernels(
            model.atoms, data.freqs_hz, recruitment_override, cfg
        )
        transfer = transfer_from_coefficients(kernels, model.coefficients)
    prediction_coord = coordinate_prediction(source, transfer)
    return np.einsum(
        "twkfd,cd->twkfc",
        prediction_coord,
        data.target_axes,
        optimize=True,
    )


def score_positions(
    data: RouteSession,
    model: SessionFit,
    context: str,
    positions: np.ndarray,
    *,
    recruitment_override: np.ndarray | None = None,
    covariance_cache: EvaluationCovarianceCache | None = None,
    config: RavelConfig | None = None,
) -> dict[str, np.ndarray]:
    cfg = config or paper_config()
    cache = covariance_cache or DEFAULT_COVARIANCE_CACHE
    positions = np.asarray(positions, dtype=int)
    observed = data.contexts[context].target_raw[positions]
    predicted = predict_positions(
        data,
        model,
        context,
        positions,
        recruitment_override=recruitment_override,
        config=cfg,
    )
    covariance = cache.get(data, context, cfg)
    nll = complex_gaussian_trial_nll(observed, predicted, covariance)
    error_power = np.mean(
        np.abs(observed - predicted) ** 2, axis=(1, 2, 3, 4)
    )
    target_power = np.mean(np.abs(observed) ** 2, axis=(1, 2, 3, 4)) + 1e-12
    nmse = error_power / target_power
    return {
        "trial_nll": nll,
        "trial_nmse": nmse,
        "trial_r2": 1.0 - nmse,
        "predicted": predicted,
    }


def score_session_model(
    data: RouteSession,
    model: SessionFit,
    context: str,
    split_name: str,
    *,
    covariance_cache: EvaluationCovarianceCache | None = None,
    config: RavelConfig | None = None,
) -> dict[str, np.ndarray]:
    return score_positions(
        data,
        model,
        context,
        data.positions(context, split_name),
        covariance_cache=covariance_cache,
        config=config,
    )


def score_zero_prediction(
    data: RouteSession,
    context: str,
    split_name: str,
    *,
    covariance_cache: EvaluationCovarianceCache | None = None,
    config: RavelConfig | None = None,
) -> dict[str, np.ndarray]:
    cfg = config or paper_config()
    cache = covariance_cache or DEFAULT_COVARIANCE_CACHE
    positions = data.positions(context, split_name)
    observed = data.contexts[context].target_raw[positions]
    covariance = cache.get(data, context, cfg)
    nll = complex_gaussian_trial_nll(observed, np.zeros_like(observed), covariance)
    return {
        "trial_nll": nll,
        "trial_nmse": np.ones(len(observed), float),
        "trial_r2": np.zeros(len(observed), float),
    }


def shuffled_recruitment(model: SessionFit, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(model.recruitment, float).copy()
    for motif in range(shuffled.shape[1]):
        shuffled[:, motif] = shuffled[rng.permutation(len(shuffled)), motif]
    return shuffled
