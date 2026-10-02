"""Template for applying RAVEL to an aligned-epoch dataset.

Replace only ``load_aligned_epochs``. Keep preprocessing, fitting, dictionary
selection, and metrics in the reusable RAVEL package.
"""

from pathlib import Path
import pandas as pd

from ravel import (
    build_route_session_from_epochs,
    fit_selected_shared_route,
    fit_task_transfer,
    lock_route_specification,
    paper_config,
    save_config,
    save_physics_json,
    score_task_transfer,
)
from ravel.workflows import run_independent_vs_pooled, score_shared_route


def load_aligned_epochs(session_id):
    """Return source_epochs, target_epochs, metadata, subject_id, experience.

    Each epoch mapping must contain matching ``active`` and ``passive`` arrays
    with shape [trial, 1500, channel], already resampled to 500 Hz and aligned
    by event.  Metadata values are pandas DataFrames with one row per trial.
    """

    raise NotImplementedError("implement only this dataset adapter")


def main(session_ids, development_ids) -> None:
    config = paper_config()
    output = Path("ravel_paper_run")
    sessions = []
    for session_id in session_ids:
        source, target, metadata, subject_id, experience = load_aligned_epochs(
            session_id
        )
        sessions.append(
            build_route_session_from_epochs(
                source_epochs=source,
                target_epochs=target,
                metadata=metadata,
                route_name="SOURCE_to_TARGET",
                source_region="SOURCE",
                target_region="TARGET",
                session_id=session_id,
                subject_id=subject_id,
                experience=experience,
                d=3,
                reference_context="active",
                config=config,
            )
        )
    development = [
        item for item in sessions if item.session_id in set(development_ids)
    ]
    locked = lock_route_specification(
        "SOURCE_to_TARGET",
        development,
        motif_candidates=[2, 3, 4],
        checkpoint_dir=output / "checkpoints",
        config=config,
    )
    pooled = fit_selected_shared_route(
        "SOURCE_to_TARGET",
        sessions,
        locked["atoms"],
        checkpoint_dir=output / "checkpoints",
        config=config,
    )
    output.mkdir(parents=True, exist_ok=True)
    score_shared_route(sessions, pooled, context="active", config=config).to_csv(
        output / "pooled_test_scores.csv", index=False
    )
    independent = run_independent_vs_pooled(
        sessions,
        pooled,
        locked["atoms"],
        context="active",
        checkpoint_dir=output / "checkpoints",
        config=config,
    )
    for name in ("scores", "matched_atoms", "restart_summary", "concentration"):
        independent[name].to_csv(output / f"independent_{name}.csv", index=False)

    transfer = fit_task_transfer(
        sessions,
        pooled,
        target_context="passive",
        checkpoint_dir=output / "checkpoints",
        config=config,
    )
    transfer_scores, noninferiority = score_task_transfer(
        sessions, transfer, target_context="passive", config=config
    )
    transfer_scores.to_csv(output / "active_to_passive_scores.csv", index=False)
    noninferiority.to_csv(
        output / "active_to_passive_noninferiority.csv", index=False
    )
    save_physics_json(
        pooled.route_name, pooled.atoms, output / "shared_physics.json"
    )
    save_config(config, output / "config.json")


if __name__ == "__main__":
    raise SystemExit(
        "Import main() after implementing load_aligned_epochs; this template "
        "does not guess dataset-specific event definitions."
    )
