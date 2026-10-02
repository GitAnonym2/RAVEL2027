"""Conditional complex spectral-regression model used in this version."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import lsq_linear

from .config import RavelConfig, paper_config
from .types import AtomSpec, RouteSession, SessionFit


def atom_key(atom: AtomSpec) -> tuple[object, ...]:
    return (
        atom.kind,
        round(float(atom.delay_ms), 8),
        None
        if not np.isfinite(atom.pole_hz)
        else round(float(atom.pole_hz), 8),
    )


def atom_response(
    atom: AtomSpec,
    freqs_hz: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Evaluate a pure-delay or delayed one-pole spectral atom."""

    frequency = np.asarray(freqs_hz, float)
    phase = np.exp(-1j * 2.0 * np.pi * frequency * atom.delay_ms / 1000.0)
    if atom.kind == "delay":
        response = phase
    elif atom.kind == "pole1":
        response = phase / (1.0 + 1j * frequency / float(atom.pole_hz))
    else:
        raise ValueError(f"unsupported atom kind {atom.kind!r}")
    if normalize:
        response /= max(float(np.sqrt(np.mean(np.abs(response) ** 2))), 1e-12)
    return response


def atom_shape_similarity(
    first: AtomSpec,
    second: AtomSpec,
    freqs_hz: np.ndarray,
) -> float:
    a = atom_response(first, freqs_hz)
    b = atom_response(second, freqs_hz)
    return float(
        abs(np.vdot(a, b)) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
    )


def zero_phase_nuisance_basis(
    freqs_hz: np.ndarray,
    degree: int,
) -> np.ndarray:
    frequency = np.asarray(freqs_hz, float)
    x = np.log(np.maximum(frequency, 1e-3))
    x = (x - x.mean()) / max(x.std(), 1e-12)
    previous = [np.ones(len(frequency), float)]
    columns: list[np.ndarray] = []
    for power in range(1, int(degree) + 1):
        value = x**power
        for basis in previous:
            value -= basis * np.dot(basis, value) / max(np.dot(basis, basis), 1e-12)
        value /= max(float(np.sqrt(np.mean(value**2))), 1e-12)
        previous.append(value)
        columns.append(value.astype(complex))
    return (
        np.column_stack(columns)
        if columns
        else np.empty((len(frequency), 0), complex)
    )


def base_kernels(
    atoms: list[AtomSpec],
    freqs_hz: np.ndarray,
    config: RavelConfig | None = None,
) -> np.ndarray:
    cfg = config or paper_config()
    columns = [np.ones(len(freqs_hz), complex)]
    nuisance = zero_phase_nuisance_basis(
        freqs_hz, degree=cfg.model.nuisance_degree
    )
    columns.extend([nuisance[:, index] for index in range(nuisance.shape[1])])
    columns.extend([atom_response(atom, freqs_hz) for atom in atoms])
    return np.column_stack(columns)


def window_kernels(
    atoms: list[AtomSpec],
    freqs_hz: np.ndarray,
    recruitment: np.ndarray,
    config: RavelConfig | None = None,
) -> np.ndarray:
    cfg = config or paper_config()
    base = base_kernels(atoms, freqs_hz, cfg)
    n_windows = recruitment.shape[0]
    result = np.repeat(base[None, :, :], n_windows, axis=0)
    atom_start = 1 + cfg.model.nuisance_degree
    for motif in range(len(atoms)):
        result[:, :, atom_start + motif] *= recruitment[:, motif, None]
    return result


def truncate_rank(matrix: np.ndarray, rank: int) -> np.ndarray:
    rank = min(int(rank), min(matrix.shape))
    if rank >= min(matrix.shape):
        return np.asarray(matrix, float)
    u, singular, vt = np.linalg.svd(np.asarray(matrix, float), full_matrices=False)
    return (u[:, :rank] * singular[:rank]) @ vt[:rank]


def transfer_from_coefficients(
    kernels: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    return np.einsum("wfb,bji->wfji", kernels, coefficients, optimize=True)


def coordinate_prediction(
    source_coord: np.ndarray,
    transfer: np.ndarray,
) -> np.ndarray:
    return np.einsum("wfji,twkfi->twkfj", transfer, source_coord, optimize=True)


def solve_one_windowed_basis(
    source: np.ndarray,
    residual: np.ndarray,
    kernel: np.ndarray,
    ridge: float,
) -> np.ndarray:
    z = source * kernel[None, :, None, :, None]
    z2 = z.reshape(-1, z.shape[-1])
    r2 = residual.reshape(-1, residual.shape[-1])
    gram = np.real(z2.conj().T @ z2)
    scale = max(float(np.trace(gram)) / max(len(gram), 1), 1e-12)
    lhs = gram + ridge * scale * np.eye(gram.shape[0])
    cross = np.real(z2.conj().T @ r2)
    return np.linalg.solve(lhs, cross).T


def fit_coefficients(
    source: np.ndarray,
    target: np.ndarray,
    kernels: np.ndarray,
    config: RavelConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Block-coordinate real ridge fits with post-update rank projection."""

    cfg = config or paper_config()
    d_source, d_target = source.shape[-1], target.shape[-1]
    n_basis = kernels.shape[-1]
    ranks = [d_source] + [
        min(cfg.model.nuisance_residue_rank, d_source)
    ] * cfg.model.nuisance_degree
    ranks += [min(cfg.model.motif_residue_rank, d_source)] * (
        n_basis - 1 - cfg.model.nuisance_degree
    )
    coefficients = np.zeros((n_basis, d_target, d_source), float)
    prediction = np.zeros_like(target)
    objective = float(np.mean(np.abs(target) ** 2))
    for _ in range(cfg.model.coefficient_sweeps):
        improved = False
        for basis in range(n_basis):
            current = np.einsum(
                "wf,ji,twkfi->twkfj",
                kernels[:, :, basis],
                coefficients[basis],
                source,
                optimize=True,
            )
            residual = target - (prediction - current)
            update = solve_one_windowed_basis(
                source,
                residual,
                kernels[:, :, basis],
                ridge=cfg.model.ridge,
            )
            update = truncate_rank(update, ranks[basis])
            candidate = np.einsum(
                "wf,ji,twkfi->twkfj",
                kernels[:, :, basis],
                update,
                source,
                optimize=True,
            )
            candidate_prediction = prediction - current + candidate
            candidate_objective = float(
                np.mean(np.abs(target - candidate_prediction) ** 2)
            )
            if candidate_objective <= objective * (1.0 + 1e-10):
                coefficients[basis] = update
                prediction = candidate_prediction
                improved |= candidate_objective < objective * (1.0 - 1e-9)
                objective = candidate_objective
        if not improved:
            break
    return coefficients, prediction, objective


def solve_recruitment_profile(
    base_piece: np.ndarray,
    residual_without: np.ndarray,
    config: RavelConfig | None = None,
) -> np.ndarray:
    cfg = config or paper_config()
    n_windows = base_piece.shape[1]
    qnorm = np.sum(np.abs(base_piece) ** 2, axis=(0, 2, 3, 4)).real
    cross = np.real(
        np.sum(base_piece.conj() * residual_without, axis=(0, 2, 3, 4))
    )
    positive = qnorm[qnorm > 1e-12]
    scale = float(np.median(positive)) if len(positive) else 1.0
    data_weight = np.sqrt(np.maximum(qnorm, 1e-12))
    matrices = [np.diag(data_weight)]
    targets = [cross / data_weight]
    ridge_weight = math.sqrt(cfg.model.recruitment_ridge_relative * scale)
    matrices.append(ridge_weight * np.eye(n_windows))
    targets.append(ridge_weight * np.ones(n_windows))
    if n_windows > 1:
        difference = np.zeros((n_windows - 1, n_windows), float)
        for index in range(n_windows - 1):
            difference[index, index] = -1.0
            difference[index, index + 1] = 1.0
        smooth_weight = math.sqrt(cfg.model.recruitment_smooth_relative * scale)
        matrices.append(smooth_weight * difference)
        targets.append(np.zeros(n_windows - 1))
    return lsq_linear(
        np.vstack(matrices),
        np.concatenate(targets),
        bounds=(0.0, cfg.model.recruitment_max),
        lsmr_tol="auto",
    ).x


def update_recruitment(
    source: np.ndarray,
    target: np.ndarray,
    atoms: list[AtomSpec],
    coefficients: np.ndarray,
    recruitment: np.ndarray,
    freqs_hz: np.ndarray,
    *,
    normalize: bool = True,
    config: RavelConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = config or paper_config()
    recruitment = np.asarray(recruitment, float).copy()
    coefficients = np.asarray(coefficients, float).copy()
    kernels = window_kernels(atoms, freqs_hz, recruitment, cfg)
    transfer = transfer_from_coefficients(kernels, coefficients)
    prediction = coordinate_prediction(source, transfer)
    atom_start = 1 + cfg.model.nuisance_degree
    for motif, atom in enumerate(atoms):
        basis = atom_start + motif
        response = atom_response(atom, freqs_hz)
        base_piece = np.einsum(
            "f,ji,twkfi->twkfj",
            response,
            coefficients[basis],
            source,
            optimize=True,
        )
        current_piece = (
            base_piece * recruitment[:, motif][None, :, None, None, None]
        )
        residual_without = target - (prediction - current_piece)
        solved = solve_recruitment_profile(base_piece, residual_without, cfg)
        if normalize:
            mean_level = max(float(np.mean(solved)), 1e-6)
            recruitment[:, motif] = solved / mean_level
            coefficients[basis] *= mean_level
        else:
            recruitment[:, motif] = solved
        kernels = window_kernels(atoms, freqs_hz, recruitment, cfg)
        transfer = transfer_from_coefficients(kernels, coefficients)
        prediction = coordinate_prediction(source, transfer)
    return recruitment, coefficients


def fit_session_model(
    data: RouteSession,
    atoms: list[AtomSpec],
    context: str,
    positions: np.ndarray | None = None,
    *,
    dynamic: bool = True,
    config: RavelConfig | None = None,
) -> SessionFit:
    """Fit all session-specific residues and recruitment for fixed atoms."""

    cfg = config or paper_config()
    context_data = data.contexts[context]
    fit_positions = (
        data.positions(context, "train")
        if positions is None
        else np.asarray(positions, dtype=int)
    )
    source = context_data.source_coord[fit_positions]
    target = context_data.target_coord[fit_positions]
    recruitment = np.ones((source.shape[1], len(atoms)), float)
    if dynamic and atoms:
        for _ in range(cfg.model.recruitment_alternations):
            kernels = window_kernels(atoms, data.freqs_hz, recruitment, cfg)
            coefficients, _, _ = fit_coefficients(source, target, kernels, cfg)
            recruitment, coefficients = update_recruitment(
                source,
                target,
                atoms,
                coefficients,
                recruitment,
                data.freqs_hz,
                normalize=True,
                config=cfg,
            )
    kernels = window_kernels(atoms, data.freqs_hz, recruitment, cfg)
    coefficients, prediction, mse = fit_coefficients(source, target, kernels, cfg)
    transfer = transfer_from_coefficients(kernels, coefficients)
    design = kernels.reshape(-1, kernels.shape[-1])
    real_design = np.vstack([design.real, design.imag])
    return SessionFit(
        route_name=data.route_name,
        session_id=data.session_id,
        context=context,
        atoms=list(atoms),
        coefficients=coefficients,
        recruitment=recruitment,
        transfer=transfer,
        train_mse=float(mse),
        kernel_condition=float(np.linalg.cond(real_design)),
        fit_positions=fit_positions,
    )


def refit_recruitment_only(
    data: RouteSession,
    base_model: SessionFit,
    context: str,
    positions: np.ndarray | None = None,
    config: RavelConfig | None = None,
) -> SessionFit:
    """Freeze atoms and coefficient residues; fit only target-context recruitment."""

    cfg = config or paper_config()
    context_data = data.contexts[context]
    fit_positions = (
        data.positions(context, "train")
        if positions is None
        else np.asarray(positions, dtype=int)
    )
    source = context_data.source_coord[fit_positions]
    target = context_data.target_coord[fit_positions]
    coefficients = np.asarray(base_model.coefficients, float).copy()
    recruitment = np.ones((source.shape[1], len(base_model.atoms)), float)
    for _ in range(cfg.model.recruitment_alternations + 1):
        recruitment, coefficients_copy = update_recruitment(
            source,
            target,
            base_model.atoms,
            coefficients.copy(),
            recruitment,
            data.freqs_hz,
            normalize=False,
            config=cfg,
        )
        coefficients = coefficients_copy
    kernels = window_kernels(
        base_model.atoms, data.freqs_hz, recruitment, cfg
    )
    transfer = transfer_from_coefficients(kernels, coefficients)
    prediction = coordinate_prediction(source, transfer)
    return SessionFit(
        route_name=data.route_name,
        session_id=data.session_id,
        context=context,
        atoms=list(base_model.atoms),
        coefficients=coefficients,
        recruitment=recruitment,
        transfer=transfer,
        train_mse=float(np.mean(np.abs(target - prediction) ** 2)),
        kernel_condition=base_model.kernel_condition,
        fit_positions=fit_positions,
    )


def transfer_similarity(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first).reshape(-1)
    b = np.asarray(second).reshape(-1)
    return float(abs(np.vdot(a, b)) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))

