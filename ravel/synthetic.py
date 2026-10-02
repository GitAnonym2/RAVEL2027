"""Deterministic synthetic fixtures for examples and regression tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RavelConfig, paper_config
from .model import coordinate_prediction, transfer_from_coefficients, window_kernels
from .random import stable_seed
from .types import AtomSpec, ContextWindowed, RouteSession


def make_dynamic_route_session(
    *,
    session_id: int = 1,
    subject_id: str | None = None,
    noise_scale: float = 0.04,
    seed_suffix: object | None = None,
    config: RavelConfig | None = None,
) -> tuple[RouteSession, list[AtomSpec], np.ndarray]:
    """Reproduce the this version's notebook's dynamic synthetic smoke fixture."""

    cfg = config or paper_config()
    seed_parts: tuple[object, ...] = ("phase0h", "synthetic")
    if seed_suffix is not None:
        seed_parts += (seed_suffix,)
    rng = np.random.default_rng(stable_seed(cfg.base_seed, *seed_parts))
    trials, windows, tapers, frequencies, d, channels = 18, 7, 2, 12, 2, 3
    freqs = np.linspace(6.0, 78.0, frequencies)
    centers = np.linspace(-0.7, 1.3, windows)
    source = (
        rng.normal(size=(trials, windows, tapers, frequencies, d))
        + 1j * rng.normal(size=(trials, windows, tapers, frequencies, d))
    ).astype(np.complex64)
    axes, _ = np.linalg.qr(rng.normal(size=(channels, d)))
    atoms = [AtomSpec("pole1", 6.0, 10.0), AtomSpec("pole1", 24.0, 18.0)]
    recruitment = np.column_stack(
        [np.linspace(1.7, 0.3, windows), np.linspace(0.25, 1.75, windows)]
    )
    coefficients = np.zeros((1 + cfg.model.nuisance_degree + 2, d, d), float)
    coefficients[0] = 0.05 * np.eye(d)
    coefficients[-2] = np.array([[0.8, 0.1], [0.0, 0.3]])
    coefficients[-1] = np.array([[0.1, 0.0], [0.2, 0.7]])
    kernels = window_kernels(atoms, freqs, recruitment, cfg)
    target_coord = coordinate_prediction(
        source, transfer_from_coefficients(kernels, coefficients)
    )
    target_coord += noise_scale * (
        rng.normal(size=target_coord.shape)
        + 1j * rng.normal(size=target_coord.shape)
    )
    target_raw = np.einsum(
        "twkfd,cd->twkfc", target_coord, axes, optimize=True
    )
    split = {
        "train": np.arange(10),
        "val_select": np.arange(10, 13),
        "val_gate": np.arange(13, 15),
        "test": np.arange(15, 18),
    }
    metadata = pd.DataFrame(
        {"outcome": ["hit", "miss"] * 9, "response_latency_s": np.nan}
    )
    context = ContextWindowed(
        source_coord=source,
        target_coord=target_coord.astype(np.complex64),
        target_raw=target_raw.astype(np.complex64),
        metadata=metadata,
        splits=split,
    )
    data = RouteSession(
        route_name="synthetic_X_to_Y",
        source_region="X",
        target_region="Y",
        session_id=session_id,
        subject_id=subject_id or f"synthetic_{session_id}",
        experience="synthetic",
        d=d,
        freqs_hz=freqs,
        window_centers_s=centers,
        source_axes=np.eye(d),
        target_axes=axes,
        source_scale=np.ones(d),
        target_scale=np.ones(channels),
        contexts={"active": context, "passive": context},
    )
    return data, atoms, recruitment


def make_dynamic_route_cohort(
    n_sessions: int = 3,
    *,
    config: RavelConfig | None = None,
) -> tuple[list[RouteSession], list[AtomSpec]]:
    sessions = []
    truth: list[AtomSpec] | None = None
    for session_id in range(1, n_sessions + 1):
        session, atoms, _ = make_dynamic_route_session(
            session_id=session_id, seed_suffix=session_id, config=config
        )
        sessions.append(session)
        truth = atoms
    assert truth is not None
    return sessions, truth
