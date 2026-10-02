"""Dataset-agnostic preprocessing and spectral representation."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal.windows import dpss

from .config import RavelConfig, paper_config
from .random import stable_seed
from .types import ContextWindowed, RegionTransform, RouteSession


def detrend_and_lowpass(
    epochs: np.ndarray,
    config: RavelConfig | None = None,
) -> np.ndarray:
    """Constant-detrend and zero-phase low-pass raw trial epochs.

    Parameters
    ----------
    epochs
        Real array with shape ``[trial, time, channel]`` sampled at
        ``config.spectral.fs_hz``.
    """

    cfg = config or paper_config()
    values = signal.detrend(np.asarray(epochs, float), axis=1, type="constant")
    cutoff = min(cfg.spectral.lowpass_hz, 0.40 * cfg.spectral.fs_hz)
    sos = signal.butter(
        cfg.spectral.lowpass_order,
        cutoff,
        btype="lowpass",
        fs=cfg.spectral.fs_hz,
        output="sos",
    )
    return signal.sosfiltfilt(sos, values, axis=1).astype(np.float32)


def zero_phase_notch(
    epochs: np.ndarray,
    config: RavelConfig | None = None,
) -> np.ndarray:
    cfg = config or paper_config()
    if cfg.spectral.notch_hz is None:
        return np.asarray(epochs, np.float32)
    b, a = signal.iirnotch(
        cfg.spectral.notch_hz,
        cfg.spectral.notch_q,
        fs=cfg.spectral.fs_hz,
    )
    values = np.asarray(epochs, float)
    n_time = values.shape[1]
    padlen = min(3 * (max(len(a), len(b)) - 1), max(n_time - 2, 0))
    return signal.filtfilt(b, a, values, axis=1, padlen=padlen).astype(np.float32)


def line_noise_ratio(
    epochs: np.ndarray,
    config: RavelConfig | None = None,
) -> float:
    """Power near 60 Hz divided by power in neighboring guard bands."""

    cfg = config or paper_config()
    x = np.asarray(epochs, float)
    frequency = np.fft.rfftfreq(x.shape[1], 1.0 / cfg.spectral.fs_hz)
    power = np.mean(np.abs(np.fft.rfft(x, axis=1)) ** 2, axis=(0, 2))
    center = cfg.spectral.notch_hz or 60.0
    line = (frequency >= center - 1.0) & (frequency <= center + 1.0)
    neighbor = (
        ((frequency >= center - 6.0) & (frequency <= center - 3.0))
        | ((frequency >= center + 3.0) & (frequency <= center + 6.0))
    )
    return float(np.mean(power[line]) / max(float(np.mean(power[neighbor])), 1e-12))


def fit_region_transform(train_epochs: np.ndarray) -> RegionTransform:
    """Fit robust channel scaling and the trial-average event template."""

    train_epochs = np.asarray(train_epochs, float)
    flat = train_epochs.reshape(-1, train_epochs.shape[-1])
    median = np.nanmedian(flat, axis=0)
    mad = np.nanmedian(np.abs(flat - median), axis=0)
    scale = 1.4826 * mad
    fallback = np.nanstd(flat, axis=0)
    scale = np.where(scale > 1e-10, scale, fallback + 1e-10)
    scaled = (train_epochs - median[None, None]) / scale[None, None]
    return RegionTransform(
        median=median,
        scale=scale,
        event_template=np.nanmean(scaled, axis=0),
    )


def apply_region_transform(
    epochs: np.ndarray,
    transform: RegionTransform,
) -> np.ndarray:
    scaled = (
        np.asarray(epochs, float) - transform.median[None, None]
    ) / transform.scale[None, None]
    scaled -= transform.event_template[None]
    return np.asarray(scaled, np.float32)


def deterministic_context_split(
    metadata: pd.DataFrame,
    session_id: int | str,
    route_name: str,
    context: str,
    config: RavelConfig | None = None,
) -> dict[str, np.ndarray]:
    """Four-way trial split, stratified by active outcome when available."""

    cfg = config or paper_config()
    rng = np.random.default_rng(
        stable_seed(cfg.base_seed, session_id, route_name, context, "four_way_split")
    )
    if context == "active" and "outcome" in metadata:
        labels = metadata["outcome"].fillna("other").astype(str).to_numpy()
    else:
        labels = np.repeat(context, len(metadata))
    keys = ("train", "val_select", "val_gate", "test")
    result: dict[str, list[int] | np.ndarray] = {key: [] for key in keys}
    fractions = np.cumsum(
        [
            cfg.split.train_fraction,
            cfg.split.val_select_fraction,
            cfg.split.val_gate_fraction,
            cfg.split.test_fraction,
        ]
    )
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        cuts = np.round(fractions[:-1] * len(idx)).astype(int)
        pieces = np.split(idx, cuts)
        for key, piece in zip(keys, pieces):
            result[key].extend(piece.tolist())  # type: ignore[union-attr]
    result = {
        key: np.sort(np.asarray(value, dtype=int)) for key, value in result.items()
    }
    validation_keys = ("val_select", "val_gate", "test")
    if min(len(result[key]) for key in validation_keys) < cfg.split.minimum_validation_trials:
        order = rng.permutation(len(metadata))
        cuts = np.round(fractions[:-1] * len(order)).astype(int)
        pieces = np.split(order, cuts)
        result = {
            key: np.sort(np.asarray(piece, dtype=int))
            for key, piece in zip(keys, pieces)
        }
    return result


def lagged_cross_covariance(
    source: np.ndarray,
    target: np.ndarray,
    lag_samples: int,
) -> np.ndarray:
    if lag_samples > 0:
        aa, bb = source[:, :-lag_samples], target[:, lag_samples:]
    elif lag_samples < 0:
        aa, bb = source[:, -lag_samples:], target[:, :lag_samples]
    else:
        aa, bb = source, target
    aa = aa.reshape(-1, aa.shape[-1]).astype(float)
    bb = bb.reshape(-1, bb.shape[-1]).astype(float)
    aa -= aa.mean(axis=0, keepdims=True)
    bb -= bb.mean(axis=0, keepdims=True)
    return aa.T @ bb / max(len(aa), 1)


def top_real_axes(matrix: np.ndarray, rank: int) -> np.ndarray:
    matrix = 0.5 * (np.asarray(matrix, float) + np.asarray(matrix, float).T)
    scale = max(float(np.trace(matrix)) / max(len(matrix), 1), 1e-12)
    values, vectors = np.linalg.eigh(matrix + 1e-8 * scale * np.eye(len(matrix)))
    axes = np.real(vectors[:, np.argsort(values)[::-1][: int(rank)]])
    for column in range(axes.shape[1]):
        anchor = int(np.argmax(np.abs(axes[:, column])))
        if axes[anchor, column] < 0:
            axes[:, column] *= -1
    return axes


def fit_pair_axes(
    source_epochs: np.ndarray,
    target_epochs: np.ndarray,
    rank: int,
    config: RavelConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit source and target communication coordinates on training trials only."""

    cfg = config or paper_config()
    if rank > min(source_epochs.shape[-1], target_epochs.shape[-1]):
        raise ValueError("rank exceeds the channel count in one region")
    score_source = np.zeros((source_epochs.shape[-1], source_epochs.shape[-1]), float)
    score_target = np.zeros((target_epochs.shape[-1], target_epochs.shape[-1]), float)
    lag_samples = np.unique(
        np.round(
            np.asarray(cfg.spectral.axis_lags_ms) * cfg.spectral.fs_hz / 1000.0
        ).astype(int)
    )
    for lag in lag_samples:
        cross = lagged_cross_covariance(source_epochs, target_epochs, int(lag))
        score_source += cross @ cross.T
        score_target += cross.T @ cross
    return top_real_axes(score_source, rank), top_real_axes(score_target, rank)


def frequency_keep_mask(
    frequency_hz: np.ndarray,
    config: RavelConfig | None = None,
) -> np.ndarray:
    cfg = config or paper_config()
    frequency_hz = np.asarray(frequency_hz, float)
    keep = (
        (frequency_hz >= cfg.spectral.fmin_hz)
        & (frequency_hz <= cfg.spectral.fmax_hz)
    )
    keep &= ~(
        (frequency_hz >= cfg.spectral.line_guard_low_hz)
        & (frequency_hz <= cfg.spectral.line_guard_high_hz)
    )
    return keep


def windowed_multitaper(
    epochs: np.ndarray,
    config: RavelConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one-sided complex DPSS coefficients.

    Output shape is ``[trial, window, taper, frequency, channel]``.
    """

    cfg = config or paper_config()
    values = np.asarray(epochs, float)
    n_window = int(round(cfg.spectral.window_seconds * cfg.spectral.fs_hz))
    n_step = int(round(cfg.spectral.window_step_seconds * cfg.spectral.fs_hz))
    if values.shape[1] < n_window:
        raise ValueError("epoch is shorter than the spectral window")
    starts = np.arange(0, values.shape[1] - n_window + 1, n_step, dtype=int)
    centers = (
        starts / cfg.spectral.fs_hz
        - cfg.spectral.epoch_pre_s
        + 0.5 * cfg.spectral.window_seconds
    )
    tapers = dpss(
        n_window,
        NW=cfg.spectral.multitaper_nw,
        Kmax=cfg.spectral.multitaper_k,
        sym=False,
    )
    frequency = np.fft.rfftfreq(n_window, d=1.0 / cfg.spectral.fs_hz)
    keep = np.flatnonzero(frequency_keep_mask(frequency, cfg))
    windows = []
    for start in starts:
        segment = values[:, start : start + n_window]
        tapered = []
        for taper in tapers:
            norm = math.sqrt(cfg.spectral.fs_hz * float(np.sum(taper**2)))
            z = np.fft.rfft(segment * taper[None, :, None], axis=1) / norm
            if n_window % 2 == 0:
                z[:, 1:-1] *= math.sqrt(2.0)
            else:
                z[:, 1:] *= math.sqrt(2.0)
            tapered.append(z[:, keep])
        windows.append(np.stack(tapered, axis=1))
    return (
        frequency[keep],
        centers,
        np.stack(windows, axis=1).astype(np.complex64),
    )


def _validate_epoch_mapping(
    source_epochs: Mapping[str, np.ndarray],
    target_epochs: Mapping[str, np.ndarray],
    metadata: Mapping[str, pd.DataFrame],
) -> None:
    if set(source_epochs) != set(target_epochs) or set(source_epochs) != set(metadata):
        raise ValueError("source, target, and metadata contexts must match")
    for context in source_epochs:
        source = np.asarray(source_epochs[context])
        target = np.asarray(target_epochs[context])
        if source.ndim != 3 or target.ndim != 3:
            raise ValueError("epochs must have shape [trial, time, channel]")
        if source.shape[:2] != target.shape[:2]:
            raise ValueError(f"unaligned source/target arrays in context {context}")
        if len(metadata[context]) != source.shape[0]:
            raise ValueError(f"metadata length mismatch in context {context}")


def build_route_session_from_epochs(
    *,
    source_epochs: Mapping[str, np.ndarray],
    target_epochs: Mapping[str, np.ndarray],
    metadata: Mapping[str, pd.DataFrame],
    route_name: str,
    source_region: str,
    target_region: str,
    session_id: int | str,
    subject_id: str,
    d: int,
    experience: str = "unknown",
    reference_context: str = "active",
    splits: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    input_is_preconditioned: bool = False,
    config: RavelConfig | None = None,
) -> RouteSession:
    """Create a complete session from aligned time-domain epochs.

    Dataset-specific code is responsible only for event selection, resampling to
    500 Hz, channel selection, and source/target trial alignment.  With
    ``input_is_preconditioned=False`` (the default), this function performs the
    detrend, low-pass, notch, robust scaling, template removal,
    communication-axis fit, and multitaper transform.  Set it to ``True`` only
    for Phase 0D-style caches that already contain detrended/low-passed epochs.
    """

    cfg = config or paper_config()
    cfg.validate()
    _validate_epoch_mapping(source_epochs, target_epochs, metadata)
    if reference_context not in source_epochs:
        raise KeyError(f"missing reference context {reference_context!r}")

    source_conditioned: dict[str, np.ndarray] = {}
    target_conditioned: dict[str, np.ndarray] = {}
    for context in source_epochs:
        source = np.asarray(source_epochs[context], np.float32)
        target = np.asarray(target_epochs[context], np.float32)
        if not input_is_preconditioned:
            source = detrend_and_lowpass(source, cfg)
            target = detrend_and_lowpass(target, cfg)
        source_conditioned[context] = zero_phase_notch(source, cfg)
        target_conditioned[context] = zero_phase_notch(target, cfg)

    if splits is None:
        split_map = {
            context: deterministic_context_split(
                metadata[context], session_id, route_name, context, cfg
            )
            for context in source_epochs
        }
    else:
        split_map = {
            context: {
                key: np.asarray(value, dtype=int) for key, value in context_splits.items()
            }
            for context, context_splits in splits.items()
        }

    reference_train = split_map[reference_context]["train"]
    source_transform = fit_region_transform(
        source_conditioned[reference_context][reference_train]
    )
    target_transform = fit_region_transform(
        target_conditioned[reference_context][reference_train]
    )
    source_residual = {
        context: apply_region_transform(values, source_transform)
        for context, values in source_conditioned.items()
    }
    target_residual = {
        context: apply_region_transform(values, target_transform)
        for context, values in target_conditioned.items()
    }
    source_axes, target_axes = fit_pair_axes(
        source_residual[reference_context][reference_train],
        target_residual[reference_context][reference_train],
        d,
        cfg,
    )

    contexts: dict[str, ContextWindowed] = {}
    common_frequency: np.ndarray | None = None
    common_centers: np.ndarray | None = None
    for context in source_epochs:
        fx, centers_x, source_fourier = windowed_multitaper(
            source_residual[context], cfg
        )
        fy, centers_y, target_fourier = windowed_multitaper(
            target_residual[context], cfg
        )
        if not (np.allclose(fx, fy) and np.allclose(centers_x, centers_y)):
            raise RuntimeError("source and target Fourier grids differ")
        common_frequency, common_centers = fx, centers_x
        contexts[context] = ContextWindowed(
            source_coord=np.einsum(
                "twkfc,cd->twkfd", source_fourier, source_axes, optimize=True
            ).astype(np.complex64),
            target_coord=np.einsum(
                "twkfc,cd->twkfd", target_fourier, target_axes, optimize=True
            ).astype(np.complex64),
            # "raw" means channel-space rather than coordinate-space; the data
            # have still received the frozen preprocessing transform.
            target_raw=np.asarray(target_fourier, np.complex64),
            metadata=metadata[context].reset_index(drop=True).copy(),
            splits={
                key: np.asarray(value, dtype=int)
                for key, value in split_map[context].items()
            },
        )

    assert common_frequency is not None and common_centers is not None
    return RouteSession(
        route_name=route_name,
        source_region=source_region,
        target_region=target_region,
        session_id=session_id,
        subject_id=str(subject_id),
        experience=str(experience),
        d=int(d),
        freqs_hz=np.asarray(common_frequency, float),
        window_centers_s=np.asarray(common_centers, float),
        source_axes=np.asarray(source_axes, float),
        target_axes=np.asarray(target_axes, float),
        source_scale=np.asarray(source_transform.scale, float),
        target_scale=np.asarray(target_transform.scale, float),
        contexts=contexts,
        representation_id=f"{reference_context}_train_axes",
    )

