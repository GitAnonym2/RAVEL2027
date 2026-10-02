"""Paper-facing summaries built from the core model and optimizer."""

from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RavelConfig, paper_config
from .metrics import EvaluationCovarianceCache, score_session_model
from .model import transfer_similarity
from .optimize import NullModelCache, fit_independent_session, match_atoms
from .types import AtomSpec, RouteSession, SharedRouteFit


def score_shared_route(
    data_list: list[RouteSession],
    shared: SharedRouteFit,
    *,
    context: str,
    split_name: str = "test",
    model_label: str = "pooled_shared_physics",
    config: RavelConfig | None = None,
) -> pd.DataFrame:
    cfg = config or paper_config()
    covariance_cache = EvaluationCovarianceCache()
    null_cache = NullModelCache()
    rows = []
    for data in data_list:
        model = shared.models[data.session_id]
        score = score_session_model(
            data,
            model,
            context,
            split_name,
            covariance_cache=covariance_cache,
            config=cfg,
        )
        null_model = null_cache.get(
            data, context, data.positions(context, "train"), cfg
        )
        nuisance = score_session_model(
            data,
            null_model,
            context,
            split_name,
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
                "context": context,
                "split": split_name,
                "model": model_label,
                "nll": float(np.mean(score["trial_nll"])),
                "nuisance_nll": float(np.mean(nuisance["trial_nll"])),
                "gain_over_nuisance": float(
                    np.mean(nuisance["trial_nll"] - score["trial_nll"])
                ),
                "nmse": float(np.mean(score["trial_nmse"])),
                "r2": float(np.mean(score["trial_r2"])),
                "n_trials": len(data.positions(context, split_name)),
            }
        )
    return pd.DataFrame(rows)


def run_independent_vs_pooled(
    data_list: list[RouteSession],
    pooled: SharedRouteFit,
    development_reference_atoms: list[AtomSpec],
    *,
    context: str = "active",
    split_name: str = "test",
    checkpoint_dir: str | Path | None = None,
    config: RavelConfig | None = None,
) -> dict[str, Any]:
    """Reproduce this version's independent-versus-pooled summaries."""

    cfg = config or paper_config()
    covariance_cache = EvaluationCovarianceCache()
    null_cache = NullModelCache()
    results = {}
    score_rows = []
    atom_rows = []
    restart_rows = []
    for data in data_list:
        independent = fit_independent_session(
            data,
            len(pooled.atoms),
            context=context,
            checkpoint_dir=checkpoint_dir,
            config=cfg,
        )
        results[data.session_id] = independent
        test = score_session_model(
            data,
            independent["model"],
            context,
            split_name,
            covariance_cache=covariance_cache,
            config=cfg,
        )
        null_model = null_cache.get(
            data, context, data.positions(context, "train"), cfg
        )
        nuisance = score_session_model(
            data,
            null_model,
            context,
            split_name,
            covariance_cache=covariance_cache,
            config=cfg,
        )
        pooled_test = score_session_model(
            data,
            pooled.models[data.session_id],
            context,
            split_name,
            covariance_cache=covariance_cache,
            config=cfg,
        )
        score_rows.append(
            {
                "route": data.route_name,
                "session_id": data.session_id,
                "subject_id": data.subject_id,
                "mouse_id": data.subject_id,
                "experience": data.experience,
                "independent_nll": float(np.mean(test["trial_nll"])),
                "pooled_nll": float(np.mean(pooled_test["trial_nll"])),
                "nuisance_nll": float(np.mean(nuisance["trial_nll"])),
                "independent_nmse": float(np.mean(test["trial_nmse"])),
                "pooled_nmse": float(np.mean(pooled_test["trial_nmse"])),
                "independent_r2": float(np.mean(test["trial_r2"])),
                "pooled_r2": float(np.mean(pooled_test["trial_r2"])),
                "pooled_gain_over_independent": float(
                    np.mean(test["trial_nll"] - pooled_test["trial_nll"])
                ),
                "operator_similarity_to_pooled": transfer_similarity(
                    independent["model"].transfer,
                    pooled.models[data.session_id].transfer,
                ),
            }
        )
        # Label against the development-only reference; report pooled delays
        # only after that independent label assignment.
        matching = match_atoms(
            development_reference_atoms,
            independent["atoms"],
            data.freqs_hz,
        )
        pooled_matching = {
            int(item["reference_index"]): int(item["candidate_index"])
            for item in match_atoms(
                development_reference_atoms, pooled.atoms, data.freqs_hz
            )
        }
        for item in matching:
            reference_index = int(item["reference_index"])
            reference = development_reference_atoms[reference_index]
            candidate = independent["atoms"][int(item["candidate_index"])]
            pooled_atom = pooled.atoms[pooled_matching[reference_index]]
            atom_rows.append(
                {
                    "route": data.route_name,
                    "session_id": data.session_id,
                    "subject_id": data.subject_id,
                    "mouse_id": data.subject_id,
                    "experience": data.experience,
                    "motif_index": reference_index,
                    "development_reference_delay_ms": reference.delay_ms,
                    "pooled_delay_ms": pooled_atom.delay_ms,
                    "independent_delay_ms": candidate.delay_ms,
                    "delay_difference_ms": candidate.delay_ms
                    - pooled_atom.delay_ms,
                    "pooled_pole_hz": pooled_atom.pole_hz,
                    "independent_pole_hz": candidate.pole_hz,
                    "shape_similarity": item["shape_similarity"],
                }
            )
        summary = independent["restart_summary"]
        if len(summary):
            table = summary.copy()
            table["route"] = data.route_name
            table["session_id"] = data.session_id
            table["subject_id"] = data.subject_id
            table["mouse_id"] = data.subject_id
            restart_rows.extend(table.to_dict("records"))

    atom_table = pd.DataFrame(atom_rows)
    concentration_rows = []
    if len(atom_table):
        for (route, motif), group in atom_table.groupby(["route", "motif_index"]):
            pooled_delay = float(group["pooled_delay_ms"].iloc[0])
            values = group["independent_delay_ms"].to_numpy(float)
            center = np.median(values)
            concentration_rows.append(
                {
                    "route": route,
                    "motif_index": int(motif),
                    "n_subjects": len(group),
                    "pooled_delay_ms": pooled_delay,
                    "independent_median_delay_ms": float(center),
                    "independent_delay_mad_ms": float(
                        np.median(np.abs(values - center))
                    ),
                    "independent_delay_iqr_ms": float(
                        np.subtract(*np.percentile(values, [75, 25]))
                    ),
                    "fraction_within_3ms": float(
                        np.mean(np.abs(values - pooled_delay) <= 3.0)
                    ),
                    "fraction_within_5ms": float(
                        np.mean(np.abs(values - pooled_delay) <= 5.0)
                    ),
                    "fraction_within_3ms_of_independent_median": float(
                        np.mean(np.abs(values - center) <= 3.0)
                    ),
                    "fraction_within_5ms_of_independent_median": float(
                        np.mean(np.abs(values - center) <= 5.0)
                    ),
                    "median_shape_similarity": float(
                        group["shape_similarity"].median()
                    ),
                }
            )
    return {
        "results": results,
        "scores": pd.DataFrame(score_rows),
        "matched_atoms": atom_table,
        "restart_summary": pd.DataFrame(restart_rows),
        "concentration": pd.DataFrame(concentration_rows),
    }
