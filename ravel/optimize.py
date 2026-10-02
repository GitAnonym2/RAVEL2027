"""Dictionary selection, restart stability, pooling, and task transfer."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Mapping
import hashlib
import json
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .config import RavelConfig, paper_config
from .metrics import (
    EvaluationCovarianceCache,
    score_positions,
    score_session_model,
    score_zero_prediction,
    shuffled_recruitment,
)
from .model import (
    atom_key,
    atom_shape_similarity,
    fit_session_model,
    refit_recruitment_only,
)
from .random import stable_seed
from .types import AtomSpec, RouteSession, SharedRouteFit, TaskTransferFit


def atoms_to_json(atoms: list[AtomSpec]) -> str:
    return json.dumps([asdict(atom) for atom in atoms])


def atoms_from_json(value: str) -> list[AtomSpec]:
    return [AtomSpec(**item) for item in json.loads(value)]


def default_atoms(motif_count: int) -> list[AtomSpec]:
    """Deterministic generic starts from this version's notebook."""

    delay_grids = {
        1: [14.0],
        2: [7.0, 23.0],
        3: [3.0, 13.0, 29.0],
        4: [2.5, 9.0, 21.0, 37.0],
    }
    pole_grids = {
        1: [10.0],
        2: [12.0, 8.0],
        3: [24.0, 11.0, 7.0],
        4: [35.0, 15.0, 9.0, 6.0],
    }
    if int(motif_count) not in delay_grids:
        raise ValueError("the frozen preset supports motif counts 1 through 4")
    return [
        AtomSpec("pole1", float(delay), float(pole))
        for delay, pole in zip(
            delay_grids[int(motif_count)], pole_grids[int(motif_count)]
        )
    ]


def valid_atom_set(
    atoms: list[AtomSpec],
    config: RavelConfig | None = None,
) -> bool:
    cfg = config or paper_config()
    delays = np.sort([float(atom.delay_ms) for atom in atoms])
    bounds = cfg.optimization
    in_bounds = all(
        bounds.delay_min_ms <= atom.delay_ms <= bounds.delay_max_ms
        and bounds.pole_min_hz <= atom.pole_hz <= bounds.pole_max_hz
        for atom in atoms
    )
    separated = (
        len(delays) < 2
        or np.min(np.diff(delays)) >= bounds.minimum_motif_separation_ms
    )
    return bool(in_bounds and separated)


def jitter_atoms(
    atoms: list[AtomSpec],
    seed: int,
    config: RavelConfig | None = None,
) -> list[AtomSpec]:
    cfg = config or paper_config()
    bounds = cfg.optimization
    rng = np.random.default_rng(seed)
    result = [
        AtomSpec(
            "pole1",
            float(
                np.clip(
                    atom.delay_ms
                    + rng.normal(0.0, bounds.restart_delay_jitter_ms),
                    bounds.delay_min_ms,
                    bounds.delay_max_ms,
                )
            ),
            float(
                np.clip(
                    atom.pole_hz
                    * np.exp(rng.normal(0.0, bounds.restart_log_pole_jitter)),
                    bounds.pole_min_hz,
                    bounds.pole_max_hz,
                )
            ),
        )
        for atom in atoms
    ]
    result = sorted(result, key=lambda atom: atom.delay_ms)
    return result if valid_atom_set(result, cfg) else list(atoms)


def _position_signature(positions: np.ndarray) -> str:
    values = np.asarray(positions, np.int64)
    return hashlib.sha1(values.tobytes()).hexdigest()[:12]


class NullModelCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[Any, ...], Any] = {}

    def clear(self) -> None:
        self._cache.clear()

    def get(
        self,
        data: RouteSession,
        context: str,
        positions: np.ndarray,
        config: RavelConfig | None = None,
    ):
        key = (
            id(data),
            data.route_name,
            str(data.session_id),
            context,
            data.representation_id,
            _position_signature(positions),
        )
        if key not in self._cache:
            self._cache[key] = fit_session_model(
                data,
                [],
                context,
                positions=np.asarray(positions, int),
                dynamic=False,
                config=config,
            )
        return self._cache[key]


DEFAULT_NULL_CACHE = NullModelCache()


def training_positions_map(
    data_list: list[RouteSession],
    context: str,
    override: Mapping[Any, np.ndarray] | None = None,
) -> dict[Any, np.ndarray]:
    if override is not None:
        return {
            key: np.asarray(value, dtype=int) for key, value in override.items()
        }
    return {
        data.session_id: data.positions(context, "train") for data in data_list
    }


def fit_dictionary_models(
    data_list: list[RouteSession],
    atoms: list[AtomSpec],
    context: str,
    positions_by_session: Mapping[Any, np.ndarray] | None = None,
    config: RavelConfig | None = None,
) -> dict[Any, Any]:
    cfg = config or paper_config()
    positions = training_positions_map(data_list, context, positions_by_session)
    return {
        data.session_id: fit_session_model(
            data,
            atoms,
            context,
            positions=positions[data.session_id],
            dynamic=True,
            config=cfg,
        )
        for data in data_list
    }


def evaluate_dictionary(
    data_list: list[RouteSession],
    atoms: list[AtomSpec],
    context: str,
    evaluation_split: str,
    *,
    positions_by_session: Mapping[Any, np.ndarray] | None = None,
    return_models: bool = False,
    null_cache: NullModelCache | None = None,
    covariance_cache: EvaluationCovarianceCache | None = None,
    config: RavelConfig | None = None,
):
    """Fit subject parameters and evaluate a shared physics dictionary.

    The minimized dictionary objective is exactly the robust session
    objective: median(delta NLL) + 0.25 * mean(delta NLL), where delta is full
    minus nuisance and therefore more-negative values are better.
    """

    if not data_list:
        raise ValueError("data_list is empty")
    cfg = config or paper_config()
    nulls = null_cache or DEFAULT_NULL_CACHE
    positions = training_positions_map(data_list, context, positions_by_session)
    models = fit_dictionary_models(data_list, atoms, context, positions, cfg)
    rows = []
    for data in data_list:
        session_id = data.session_id
        evaluation_positions = data.positions(context, evaluation_split)
        full = score_positions(
            data,
            models[session_id],
            context,
            evaluation_positions,
            covariance_cache=covariance_cache,
            config=cfg,
        )
        null_model = nulls.get(
            data, context, positions[session_id], config=cfg
        )
        nuisance = score_positions(
            data,
            null_model,
            context,
            evaluation_positions,
            covariance_cache=covariance_cache,
            config=cfg,
        )
        zero = score_zero_prediction(
            data,
            context,
            evaluation_split,
            covariance_cache=covariance_cache,
            config=cfg,
        )
        rows.append(
            {
                "route": data.route_name,
                "session_id": session_id,
                "subject_id": data.subject_id,
                "mouse_id": data.subject_id,
                "context": context,
                "split": evaluation_split,
                "full_nll": float(np.mean(full["trial_nll"])),
                "nuisance_nll": float(np.mean(nuisance["trial_nll"])),
                "zero_nll": float(np.mean(zero["trial_nll"])),
                "delta_nll_vs_nuisance": float(
                    np.mean(full["trial_nll"] - nuisance["trial_nll"])
                ),
                "full_nmse": float(np.mean(full["trial_nmse"])),
                "full_r2": float(np.mean(full["trial_r2"])),
            }
        )
    table = pd.DataFrame(rows)
    values = table["delta_nll_vs_nuisance"].to_numpy(float)
    objective = float(np.median(values) + 0.25 * np.mean(values))
    if return_models:
        return objective, table, models
    return objective, table


def local_refine_dictionary(
    data_list: list[RouteSession],
    initial_atoms: list[AtomSpec],
    context: str,
    *,
    evaluation_split: str = "val_select",
    positions_by_session: Mapping[Any, np.ndarray] | None = None,
    delay_steps_ms: list[float] | tuple[float, ...] | None = None,
    refine_poles: bool = False,
    config: RavelConfig | None = None,
) -> tuple[list[AtomSpec], float, pd.DataFrame, pd.DataFrame]:
    """Deterministic coordinate-pattern refinement of shared atom physics."""

    cfg = config or paper_config()
    steps = list(
        cfg.optimization.pooled_delay_steps_ms
        if delay_steps_ms is None
        else delay_steps_ms
    )
    cache: dict[tuple[tuple[object, ...], ...], tuple[float, pd.DataFrame]] = {}

    def evaluate(atoms: list[AtomSpec]) -> tuple[float, pd.DataFrame]:
        key = tuple(atom_key(atom) for atom in atoms)
        if key not in cache:
            cache[key] = evaluate_dictionary(
                data_list,
                atoms,
                context,
                evaluation_split,
                positions_by_session=positions_by_session,
                config=cfg,
            )
        return cache[key]

    atoms = list(initial_atoms)
    objective, per_session = evaluate(atoms)
    trace_rows = [
        {
            "step": "initial",
            "objective": objective,
            "atoms": atoms_to_json(atoms),
        }
    ]
    bounds = cfg.optimization
    for delay_step in steps:
        for motif in range(len(atoms)):
            best = (objective, atoms, per_session)
            proposals: list[list[AtomSpec]] = []
            for direction in (-1.0, 1.0):
                proposal = list(atoms)
                proposal[motif] = replace(
                    proposal[motif],
                    delay_ms=float(
                        np.clip(
                            proposal[motif].delay_ms + direction * delay_step,
                            bounds.delay_min_ms,
                            bounds.delay_max_ms,
                        )
                    ),
                )
                proposals.append(sorted(proposal, key=lambda atom: atom.delay_ms))
            if refine_poles:
                log_step = 0.20 if delay_step >= 2.0 else 0.10
                for direction in (-1.0, 1.0):
                    proposal = list(atoms)
                    proposal[motif] = replace(
                        proposal[motif],
                        pole_hz=float(
                            np.clip(
                                proposal[motif].pole_hz
                                * np.exp(direction * log_step),
                                bounds.pole_min_hz,
                                bounds.pole_max_hz,
                            )
                        ),
                    )
                    proposals.append(
                        sorted(proposal, key=lambda atom: atom.delay_ms)
                    )
            for proposal in proposals:
                if not valid_atom_set(proposal, cfg):
                    continue
                candidate_objective, candidate_session = evaluate(proposal)
                if candidate_objective < best[0] - 1e-8:
                    best = (
                        candidate_objective,
                        proposal,
                        candidate_session,
                    )
            objective, atoms, per_session = best
            trace_rows.append(
                {
                    "step": f"motif{motif}_step{delay_step:g}",
                    "objective": objective,
                    "atoms": atoms_to_json(atoms),
                }
            )
    return atoms, objective, per_session, pd.DataFrame(trace_rows)


def match_atoms(
    reference: list[AtomSpec],
    candidate: list[AtomSpec],
    freqs_hz: np.ndarray,
) -> list[dict[str, float | int]]:
    if len(reference) != len(candidate):
        raise ValueError("atom lists must have equal length")
    similarity = np.array(
        [
            [atom_shape_similarity(first, second, freqs_hz) for second in candidate]
            for first in reference
        ]
    )
    rows, columns = linear_sum_assignment(1.0 - similarity)
    return [
        {
            "reference_index": int(i),
            "candidate_index": int(j),
            "shape_similarity": float(similarity[i, j]),
            "delay_difference_ms": float(
                candidate[j].delay_ms - reference[i].delay_ms
            ),
            "candidate_delay_ms": float(candidate[j].delay_ms),
            "candidate_pole_hz": float(candidate[j].pole_hz),
        }
        for i, j in zip(rows, columns)
    ]


def _checkpoint_path(
    checkpoint_dir: str | Path | None,
    stage_name: str,
    route_name: str,
    context: str,
    motif_count: int,
    restart: int,
) -> Path | None:
    if checkpoint_dir is None:
        return None
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    safe_route = route_name.replace("/", "_").replace(" ", "_")
    return directory / (
        f"ravel_{stage_name}_{safe_route}_{context}_M{motif_count}_"
        f"restart{restart}.pkl"
    )


def run_dictionary_restarts(
    route_name: str,
    stage_name: str,
    data_list: list[RouteSession],
    initial_atoms: list[AtomSpec],
    context: str,
    *,
    n_restarts: int,
    delay_steps_ms: list[float] | tuple[float, ...],
    refine_poles: bool = False,
    positions_by_session: Mapping[Any, np.ndarray] | None = None,
    checkpoint_dir: str | Path | None = None,
    resume: bool = True,
    config: RavelConfig | None = None,
) -> list[dict[str, Any]]:
    cfg = config or paper_config()
    runs: list[dict[str, Any]] = []
    for restart in range(int(n_restarts)):
        path = _checkpoint_path(
            checkpoint_dir,
            stage_name,
            route_name,
            context,
            len(initial_atoms),
            restart,
        )
        if resume and path is not None and path.exists():
            with path.open("rb") as handle:
                runs.append(pickle.load(handle))
            continue
        start = jitter_atoms(
            initial_atoms,
            stable_seed(
                cfg.base_seed,
                stage_name,
                route_name,
                context,
                len(initial_atoms),
                restart,
            ),
            cfg,
        )
        atoms, objective, per_session, trace = local_refine_dictionary(
            data_list,
            start,
            context,
            evaluation_split="val_select",
            positions_by_session=positions_by_session,
            delay_steps_ms=delay_steps_ms,
            refine_poles=refine_poles,
            config=cfg,
        )
        result = {
            "route": route_name,
            "stage": stage_name,
            "context": context,
            "restart": int(restart),
            "initial_atoms": start,
            "atoms": atoms,
            "val_select_objective": float(objective),
            "val_select_per_session": per_session,
            "trace": trace,
        }
        if path is not None:
            with path.open("wb") as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        runs.append(result)
    return runs


def restart_stability(
    runs: list[dict[str, Any]],
    freqs_hz: np.ndarray,
    config: RavelConfig | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    cfg = config or paper_config()
    best = min(runs, key=lambda item: item["val_select_objective"])
    rows = []
    for run in runs:
        for item in match_atoms(best["atoms"], run["atoms"], freqs_hz):
            rows.append({"restart": run["restart"], **item})
    detail = pd.DataFrame(rows)
    summaries = []
    if len(detail):
        for motif, group in detail.groupby("reference_index"):
            support = np.mean(
                (
                    group["shape_similarity"]
                    >= cfg.optimization.minimum_atom_shape_similarity
                )
                & (
                    group["delay_difference_ms"].abs()
                    <= 2.0 * cfg.optimization.maximum_atom_delay_mad_ms
                )
            )
            delays = group["candidate_delay_ms"].to_numpy(float)
            summaries.append(
                {
                    "motif_index": int(motif),
                    "support_fraction": float(support),
                    "median_shape_similarity": float(
                        group["shape_similarity"].median()
                    ),
                    "delay_mad_ms": float(
                        np.median(np.abs(delays - np.median(delays)))
                    ),
                }
            )
    return best, detail, pd.DataFrame(summaries)


def competitive_run_medoid(
    runs: list[dict[str, Any]],
    freqs_hz: np.ndarray,
) -> dict[str, Any]:
    objectives = np.asarray(
        [run["val_select_objective"] for run in runs], float
    )
    best_index = int(np.argmin(objectives))
    per_session = runs[best_index]["val_select_per_session"][
        "delta_nll_vs_nuisance"
    ].to_numpy(float)
    se = (
        float(np.std(per_session, ddof=1) / np.sqrt(max(len(per_session), 1)))
        if len(per_session) > 1
        else 0.0
    )
    competitive = [
        run
        for run in runs
        if run["val_select_objective"] <= objectives[best_index] + se
    ]
    if len(competitive) == 1:
        return competitive[0]
    distances = np.zeros((len(competitive), len(competitive)), float)
    for i, first in enumerate(competitive):
        for j, second in enumerate(competitive):
            if i >= j:
                continue
            matching = match_atoms(first["atoms"], second["atoms"], freqs_hz)
            distance = np.mean(
                [
                    1.0
                    - float(item["shape_similarity"])
                    + 0.02 * abs(float(item["delay_difference_ms"]))
                    for item in matching
                ]
            )
            distances[i, j] = distances[j, i] = distance
    return competitive[int(np.argmin(distances.mean(axis=1)))]


def choose_restart_strategy(
    data_list: list[RouteSession],
    runs: list[dict[str, Any]],
    context: str,
    *,
    positions_by_session: Mapping[Any, np.ndarray] | None = None,
    config: RavelConfig | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = config or paper_config()
    best, detail, summary = restart_stability(runs, data_list[0].freqs_hz, cfg)
    consensus = competitive_run_medoid(runs, data_list[0].freqs_hz)
    candidates = {"best_restart": best, "consensus_medoid": consensus}
    rows = []
    for strategy, run in candidates.items():
        objective, per_session = evaluate_dictionary(
            data_list,
            run["atoms"],
            context,
            "val_gate",
            positions_by_session=positions_by_session,
            config=cfg,
        )
        values = per_session["delta_nll_vs_nuisance"].to_numpy(float)
        rows.append(
            {
                "strategy": strategy,
                "val_gate_objective": float(objective),
                "val_gate_mean": float(np.mean(values)),
                "val_gate_se": (
                    float(np.std(values, ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1
                    else 0.0
                ),
                "atoms": atoms_to_json(run["atoms"]),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["val_gate_mean", "strategy"]
    ).reset_index(drop=True)
    best_row = table.iloc[0]
    threshold = float(best_row["val_gate_mean"] + best_row["val_gate_se"])
    consensus_rows = table[
        (table["strategy"] == "consensus_medoid")
        & (table["val_gate_mean"] <= threshold)
    ]
    selected = consensus_rows.iloc[0] if len(consensus_rows) else best_row
    return selected, table, detail, summary


def lock_route_specification(
    route_name: str,
    development_data: list[RouteSession],
    motif_candidates: list[int] | tuple[int, ...],
    *,
    context: str = "active",
    initial_atoms_by_m: Mapping[int, list[AtomSpec]] | None = None,
    checkpoint_dir: str | Path | None = None,
    config: RavelConfig | None = None,
) -> dict[str, Any]:
    """Lock motif count and initial shared physics using development data only."""

    if len(development_data) < 1:
        raise ValueError("development_data is empty")
    cfg = config or paper_config()
    all_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    selections = []
    for motif_count in motif_candidates:
        initial = (
            list(initial_atoms_by_m[int(motif_count)])
            if initial_atoms_by_m is not None and int(motif_count) in initial_atoms_by_m
            else default_atoms(int(motif_count))
        )
        runs = run_dictionary_restarts(
            route_name,
            "method_lock",
            development_data,
            initial,
            context,
            n_restarts=cfg.optimization.lock_restarts,
            delay_steps_ms=cfg.optimization.lock_delay_steps_ms,
            refine_poles=True,
            checkpoint_dir=checkpoint_dir,
            config=cfg,
        )
        selected, strategy_table, details, stability = choose_restart_strategy(
            development_data, runs, context, config=cfg
        )
        strategy_table = strategy_table.copy()
        strategy_table["route"] = route_name
        strategy_table["M"] = int(motif_count)
        all_rows.extend(strategy_table.to_dict("records"))
        details = details.copy()
        details["route"] = route_name
        details["M"] = int(motif_count)
        detail_rows.extend(details.to_dict("records"))
        stability = stability.copy()
        stability["route"] = route_name
        stability["M"] = int(motif_count)
        summary_rows.extend(stability.to_dict("records"))
        min_support = (
            float(stability["support_fraction"].min()) if len(stability) else 0.0
        )
        max_mad = (
            float(stability["delay_mad_ms"].max()) if len(stability) else np.inf
        )
        selections.append(
            {
                "route": route_name,
                "M": int(motif_count),
                "strategy": str(selected["strategy"]),
                "val_gate_mean": float(selected["val_gate_mean"]),
                "val_gate_se": float(selected["val_gate_se"]),
                "min_restart_support": min_support,
                "max_delay_mad_ms": max_mad,
                "atoms": str(selected["atoms"]),
            }
        )
    selection_table = pd.DataFrame(selections).sort_values("M")
    best = selection_table.loc[selection_table["val_gate_mean"].idxmin()]
    threshold = float(best["val_gate_mean"] + best["val_gate_se"])
    eligible = selection_table[
        (selection_table["val_gate_mean"] <= threshold)
        & (
            selection_table["min_restart_support"]
            >= cfg.optimization.minimum_restart_support
        )
    ]
    if not len(eligible):
        eligible = selection_table[
            selection_table["val_gate_mean"] <= threshold
        ]
    chosen = (
        eligible.sort_values(["M", "val_gate_mean"]).iloc[0]
        if len(eligible)
        else best
    )
    return {
        "route": route_name,
        "M": int(chosen["M"]),
        "strategy": str(chosen["strategy"]),
        "atoms": atoms_from_json(chosen["atoms"]),
        "selection_table": selection_table,
        "strategy_table": pd.DataFrame(all_rows),
        "restart_details": pd.DataFrame(detail_rows),
        "restart_summary": pd.DataFrame(summary_rows),
    }


def fit_selected_shared_route(
    route_name: str,
    data_list: list[RouteSession],
    locked_atoms: list[AtomSpec],
    *,
    context: str = "active",
    stage_name: str = "pooled_active",
    checkpoint_dir: str | Path | None = None,
    config: RavelConfig | None = None,
) -> SharedRouteFit:
    """Pool route physics while keeping every session parameter separate."""

    cfg = config or paper_config()
    runs = run_dictionary_restarts(
        route_name,
        stage_name,
        data_list,
        locked_atoms,
        context,
        n_restarts=cfg.optimization.pooled_restarts,
        delay_steps_ms=cfg.optimization.pooled_delay_steps_ms,
        refine_poles=False,
        checkpoint_dir=checkpoint_dir,
        config=cfg,
    )
    selected, strategy_table, details, summary = choose_restart_strategy(
        data_list, runs, context, config=cfg
    )
    atoms = atoms_from_json(selected["atoms"])
    models = fit_dictionary_models(data_list, atoms, context, config=cfg)
    return SharedRouteFit(
        route_name=route_name,
        context=context,
        atoms=atoms,
        models=models,
        selected_strategy=str(selected["strategy"]),
        strategy_table=strategy_table,
        restart_details=details,
        restart_summary=summary,
    )


def _broad_random_atoms(
    motif_count: int,
    seed: int,
    config: RavelConfig,
) -> list[AtomSpec]:
    bounds = config.optimization
    rng = np.random.default_rng(seed)
    for _ in range(100):
        delays = np.sort(
            rng.uniform(bounds.delay_min_ms, bounds.delay_max_ms, motif_count)
        )
        if (
            motif_count < 2
            or np.min(np.diff(delays)) >= bounds.minimum_motif_separation_ms
        ):
            poles = np.exp(
                rng.uniform(
                    np.log(bounds.pole_min_hz),
                    np.log(bounds.pole_max_hz),
                    motif_count,
                )
            )
            return [
                AtomSpec("pole1", float(delay), float(pole))
                for delay, pole in zip(delays, poles)
            ]
    return default_atoms(motif_count)


def fit_independent_session(
    data: RouteSession,
    motif_count: int,
    *,
    context: str = "active",
    checkpoint_dir: str | Path | None = None,
    config: RavelConfig | None = None,
) -> dict[str, Any]:
    """Independent, unseeded session fit used in the 25-mouse comparison."""

    cfg = config or paper_config()
    final_path: Path | None = None
    if checkpoint_dir is not None:
        directory = Path(checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        safe_route = data.route_name.replace("/", "_").replace(" ", "_")
        final_path = directory / (
            f"ravel_independent_{safe_route}_{data.session_id}_M{motif_count}.pkl"
        )
        if final_path.exists():
            with final_path.open("rb") as handle:
                return pickle.load(handle)
    runs = []
    for restart in range(cfg.optimization.independent_restarts):
        restart_path = (
            final_path.with_name(final_path.stem + f"_restart{restart}.pkl")
            if final_path is not None
            else None
        )
        if restart_path is not None and restart_path.exists():
            with restart_path.open("rb") as handle:
                runs.append(pickle.load(handle))
            continue
        initial = (
            default_atoms(motif_count)
            if restart == 0
            else _broad_random_atoms(
                motif_count,
                stable_seed(
                    cfg.base_seed,
                    "independent_broad",
                    data.route_name,
                    data.session_id,
                    restart,
                ),
                cfg,
            )
        )
        # Independent fits use the generic grid once and genuinely broad starts
        # thereafter.  Unlike pooled restarts, this version did not add a second
        # local jitter to these starts.
        start = initial
        atoms, objective, per_session, trace = local_refine_dictionary(
            [data],
            start,
            context,
            evaluation_split="val_select",
            delay_steps_ms=cfg.optimization.independent_delay_steps_ms,
            refine_poles=True,
            config=cfg,
        )
        run = {
                "route": data.route_name,
                "stage": "independent_unseeded_v2",
                "context": context,
                "restart": restart,
                "initial_atoms": start,
                "atoms": atoms,
                "val_select_objective": objective,
                "val_select_per_session": per_session,
                "trace": trace,
            }
        if restart_path is not None:
            with restart_path.open("wb") as handle:
                pickle.dump(run, handle, protocol=pickle.HIGHEST_PROTOCOL)
        runs.append(run)
    selected, strategy_table, details, summary = choose_restart_strategy(
        [data], runs, context, config=cfg
    )
    atoms = atoms_from_json(selected["atoms"])
    model = fit_session_model(data, atoms, context, dynamic=True, config=cfg)
    result = {
        "session_id": data.session_id,
        "route": data.route_name,
        "atoms": atoms,
        "model": model,
        "strategy": str(selected["strategy"]),
        "strategy_table": strategy_table,
        "restart_details": details,
        "restart_summary": summary,
    }
    if final_path is not None:
        with final_path.open("wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def fit_context_scratch_shared_route(
    route_name: str,
    data_list: list[RouteSession],
    motif_count: int,
    context: str,
    *,
    checkpoint_dir: str | Path | None = None,
    config: RavelConfig | None = None,
) -> SharedRouteFit:
    """Fit target-context atoms without source-context atom initialization."""

    cfg = config or paper_config()
    runs = []
    for restart in range(cfg.optimization.pooled_restarts):
        initial = (
            default_atoms(motif_count)
            if restart == 0
            else _broad_random_atoms(
                motif_count,
                stable_seed(
                    cfg.base_seed,
                    "task_scratch_broad",
                    route_name,
                    context,
                    restart,
                ),
                cfg,
            )
        )
        # Do not add a second jitter here.  The notebook used each generic/broad
        # start as the initial set and the restart helper then applied one jitter.
        one = run_dictionary_restarts(
            route_name,
            f"scratch_unseeded_v2_{context}_start{restart}",
            data_list,
            initial,
            context,
            n_restarts=1,
            delay_steps_ms=cfg.optimization.lock_delay_steps_ms,
            refine_poles=True,
            checkpoint_dir=checkpoint_dir,
            config=cfg,
        )
        runs.extend(one)
    selected, strategy_table, details, summary = choose_restart_strategy(
        data_list, runs, context, config=cfg
    )
    atoms = atoms_from_json(selected["atoms"])
    return SharedRouteFit(
        route_name=route_name,
        context=context,
        atoms=atoms,
        models=fit_dictionary_models(data_list, atoms, context, config=cfg),
        selected_strategy=str(selected["strategy"]),
        strategy_table=strategy_table,
        restart_details=details,
        restart_summary=summary,
    )


def fit_task_transfer(
    data_list: list[RouteSession],
    source_fit: SharedRouteFit,
    *,
    target_context: str,
    fit_scratch: bool = True,
    checkpoint_dir: str | Path | None = None,
    config: RavelConfig | None = None,
) -> TaskTransferFit:
    """active-to-passive transfer panel.

    ``frozen_physics_refit_subject`` is the primary test: atoms are frozen while
    target-context residues and recruitment are fitted.  The stricter branch
    freezes both atoms and source-context residues, fitting recruitment only.
    """

    cfg = config or paper_config()
    refit_subject = {}
    strict = {}
    static = {}
    for data in data_list:
        session_id = data.session_id
        refit_subject[session_id] = fit_session_model(
            data,
            source_fit.atoms,
            target_context,
            dynamic=True,
            config=cfg,
        )
        strict[session_id] = refit_recruitment_only(
            data,
            source_fit.models[session_id],
            target_context,
            config=cfg,
        )
        static[session_id] = fit_session_model(
            data,
            source_fit.atoms,
            target_context,
            dynamic=False,
            config=cfg,
        )
    scratch = (
        fit_context_scratch_shared_route(
            source_fit.route_name,
            data_list,
            len(source_fit.atoms),
            target_context,
            checkpoint_dir=checkpoint_dir,
            config=cfg,
        )
        if fit_scratch
        else None
    )
    return TaskTransferFit(
        frozen_physics_refit_subject=refit_subject,
        strict_recruitment_only=strict,
        frozen_physics_static=static,
        context_scratch=scratch,
    )


def _mean_score_record(
    data: RouteSession,
    model,
    context: str,
    split_name: str,
    label: str,
    nuisance_model,
    *,
    covariance_cache: EvaluationCovarianceCache | None = None,
    config: RavelConfig | None = None,
    **extra: Any,
) -> dict[str, Any]:
    positions = data.positions(context, split_name)
    score = score_positions(
        data,
        model,
        context,
        positions,
        covariance_cache=covariance_cache,
        config=config,
    )
    nuisance = score_positions(
        data,
        nuisance_model,
        context,
        positions,
        covariance_cache=covariance_cache,
        config=config,
    )
    row = {
        "route": data.route_name,
        "session_id": data.session_id,
        "subject_id": data.subject_id,
        "mouse_id": data.subject_id,
        "experience": data.experience,
        "context": context,
        "split": split_name,
        "model": label,
        "n_trials": len(positions),
        "nll": float(np.mean(score["trial_nll"])),
        "nuisance_nll": float(np.mean(nuisance["trial_nll"])),
        "gain_over_nuisance": float(
            np.mean(nuisance["trial_nll"] - score["trial_nll"])
        ),
        "nmse": float(np.mean(score["trial_nmse"])),
        "r2": float(np.mean(score["trial_r2"])),
    }
    row.update(extra)
    return row


def score_task_transfer(
    data_list: list[RouteSession],
    transfer_fit: TaskTransferFit,
    *,
    target_context: str,
    split_name: str = "test",
    config: RavelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score all transfer controls and compute the notebook noninferiority test."""

    cfg = config or paper_config()
    covariance_cache = EvaluationCovarianceCache()
    null_cache = NullModelCache()
    rows: list[dict[str, Any]] = []
    for data in data_list:
        session_id = data.session_id
        nuisance_model = null_cache.get(
            data,
            target_context,
            data.positions(target_context, "train"),
            cfg,
        )
        rows.extend(
            [
                _mean_score_record(
                    data,
                    transfer_fit.strict_recruitment_only[session_id],
                    target_context,
                    split_name,
                    "source_physics_and_residues__recruitment_only",
                    nuisance_model,
                    covariance_cache=covariance_cache,
                    config=cfg,
                    physics_source="source_context",
                    fitted_in_target="recruitment",
                ),
                _mean_score_record(
                    data,
                    transfer_fit.frozen_physics_refit_subject[session_id],
                    target_context,
                    split_name,
                    "source_physics__refit_subject_parameters",
                    nuisance_model,
                    covariance_cache=covariance_cache,
                    config=cfg,
                    physics_source="source_context",
                    fitted_in_target="residues_and_recruitment",
                ),
                _mean_score_record(
                    data,
                    transfer_fit.frozen_physics_static[session_id],
                    target_context,
                    split_name,
                    "source_physics__static_recruitment",
                    nuisance_model,
                    covariance_cache=covariance_cache,
                    config=cfg,
                    physics_source="source_context",
                    fitted_in_target="residues_only",
                ),
            ]
        )
        dynamic_model = transfer_fit.frozen_physics_refit_subject[session_id]
        shuffled = shuffled_recruitment(
            dynamic_model,
            stable_seed(
                cfg.base_seed,
                "task_shuffle",
                data.route_name,
                data.session_id,
            ),
        )
        positions = data.positions(target_context, split_name)
        shuffled_score = score_positions(
            data,
            dynamic_model,
            target_context,
            positions,
            recruitment_override=shuffled,
            covariance_cache=covariance_cache,
            config=cfg,
        )
        nuisance_score = score_positions(
            data,
            nuisance_model,
            target_context,
            positions,
            covariance_cache=covariance_cache,
            config=cfg,
        )
        rows.append(
            {
                "route": data.route_name,
                "session_id": data.session_id,
                "subject_id": data.subject_id,
                "mouse_id": data.subject_id,
                "experience": data.experience,
                "context": target_context,
                "split": split_name,
                "model": "source_physics__time_shuffled_recruitment",
                "n_trials": len(positions),
                "nll": float(np.mean(shuffled_score["trial_nll"])),
                "nuisance_nll": float(np.mean(nuisance_score["trial_nll"])),
                "gain_over_nuisance": float(
                    np.mean(
                        nuisance_score["trial_nll"]
                        - shuffled_score["trial_nll"]
                    )
                ),
                "nmse": float(np.mean(shuffled_score["trial_nmse"])),
                "r2": float(np.mean(shuffled_score["trial_r2"])),
                "physics_source": "source_context",
                "fitted_in_target": "shuffled_recruitment",
            }
        )
        if transfer_fit.context_scratch is not None:
            rows.append(
                _mean_score_record(
                    data,
                    transfer_fit.context_scratch.models[session_id],
                    target_context,
                    split_name,
                    "target_physics__from_scratch",
                    nuisance_model,
                    covariance_cache=covariance_cache,
                    config=cfg,
                    physics_source="target_context",
                    fitted_in_target="all",
                )
            )
    scores = pd.DataFrame(rows)
    noninferiority_rows = []
    for (route, session_id), group in scores.groupby(["route", "session_id"]):
        lookup = group.set_index("model")
        transfer_label = "source_physics__refit_subject_parameters"
        scratch_label = "target_physics__from_scratch"
        if transfer_label not in lookup.index or scratch_label not in lookup.index:
            continue
        transfer = lookup.loc[transfer_label]
        scratch = lookup.loc[scratch_label]
        margin = cfg.optimization.noninferiority_fraction * max(
            float(scratch["gain_over_nuisance"]), 0.0
        )
        gap = float(transfer["nll"] - scratch["nll"])
        noninferiority_rows.append(
            {
                "route": route,
                "session_id": session_id,
                "subject_id": transfer["subject_id"],
                "mouse_id": transfer["subject_id"],
                "transfer_minus_scratch_nll": gap,
                "allowed_margin_nll": float(margin),
                "noninferior": bool(gap <= margin),
                "retained_gain_fraction": float(
                    transfer["gain_over_nuisance"]
                    / max(float(scratch["gain_over_nuisance"]), 1e-12)
                ),
            }
        )
    return scores, pd.DataFrame(noninferiority_rows)
