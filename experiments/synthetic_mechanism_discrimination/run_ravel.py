from __future__ import annotations

"""Run the RAVEL side of the synthetic mechanism-discrimination benchmark.

The truth contains two delayed components and RAVEL is prospectively fixed to
M=2. The matched dynamic-delay comparator also has two components. ADM is run
separately with its native validation-based K selection.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAVEL_ROOT = HERE.parents[1]  # repository root
if str(RAVEL_ROOT) not in sys.path:
    sys.path.insert(0, str(RAVEL_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ravel import (  # noqa: E402
    fit_independent_session,
    fit_session_model,
    paper_config,
    save_config,
    save_physics_json,
    save_pickle,
    score_session_model,
)

from common import (  # noqa: E402
    FIXED_COMMUNICATION_ORDER,
    PROTOCOL,
    SyntheticBenchmarkConfig,
    VALID_WORLDS,
    WORLD_DYNAMIC,
    WORLD_STABLE,
    apparent_truth_delay_ms,
    build_ravel_session,
    dynamic_truth_vs_fixed_atoms,
    fit_matched_dynamic_delay_baseline,
    load_s1_dataset,
    match_dynamic_trajectories,
    matched_stable_truth_to_fit,
    pearson_or_nan,
    ravel_apparent_delay_ms,
    rmse,
    save_s1_dataset,
    simulate_s1_dataset,
    stable_recruitment_correlations,
    summarize_score,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", nargs="+", default=list(VALID_WORLDS), choices=VALID_WORLDS)
    p.add_argument("--seeds", nargs="+", type=int, default=[271101, 271102, 271103])
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--out", type=Path, default=HERE / "results")
    p.add_argument("--regenerate", action="store_true")
    p.add_argument(
        "--skip-matched-dynamic",
        action="store_true",
        help="Skip the experiment-only two-component static-gain/dynamic-delay comparator.",
    )
    return p.parse_args()


def _score(model, session) -> dict[str, float]:
    return summarize_score(
        score_session_model(session, model, "active", "test", config=paper_config())
    )


def _atoms_json(atoms) -> list[dict[str, float | str]]:
    return [
        {"kind": a.kind, "delay_ms": float(a.delay_ms), "pole_hz": float(a.pole_hz)}
        for a in atoms
    ]


def _dataset_path(root: Path, world: str, seed: int) -> Path:
    return root / world / f"seed_{seed}" / "data" / "dataset.npz"


def get_or_make_dataset(
    root: Path,
    world: str,
    seed: int,
    cfg: SyntheticBenchmarkConfig,
    regenerate: bool,
) -> tuple[dict[str, Any], Path]:
    path = _dataset_path(root, world, seed)
    if regenerate or not path.exists():
        ds = simulate_s1_dataset(world, seed=seed, cfg=cfg)
        save_s1_dataset(ds, path)
        return ds, path
    ds = load_s1_dataset(path)
    if int(ds["config"]["n_trials"]) != int(cfg.n_trials):
        raise RuntimeError(
            f"Existing {path} has n_trials={ds['config']['n_trials']} but requested {cfg.n_trials}. "
            "Delete results/ or rerun with --regenerate."
        )
    return ds, path


def _portable_dataset_ref(path: Path) -> str:
    """Store a repository-relative path when possible; never save a home path."""
    path = Path(path)
    try:
        return path.resolve().relative_to(HERE.resolve()).as_posix()
    except ValueError:
        return path.name


def run_one(world: str, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    cfg_synth = replace(SyntheticBenchmarkConfig(), n_trials=int(args.n_trials))
    run_dir = args.out / world / f"seed_{seed}"
    checkpoints = run_dir / "checkpoints"
    if args.regenerate and checkpoints.exists():
        shutil.rmtree(checkpoints)
    run_dir.mkdir(parents=True, exist_ok=True)

    ds, dataset_path = get_or_make_dataset(
        args.out, world, seed, cfg_synth, args.regenerate
    )
    session = build_ravel_session(
        ds,
        session_id=f"{world}_{seed}",
        subject_id=f"synthetic_{seed}",
    )

    print("\n" + "=" * 92)
    print(f"SYNTHETIC MECHANISM BENCHMARK | RAVEL | world={world} | seed={seed}")
    print(f"dataset          : {dataset_path}")
    print(f"trials           : {cfg_synth.n_trials}")
    print(f"fixed RAVEL M    : {FIXED_COMMUNICATION_ORDER}")
    print("model-order selection: fixed M=2 to match the known generative atom count")

    # Reference independent-session optimizer: three starts, full delay and pole
    # refinement, val_select/val_gate restart choice, TEST untouched.
    fit = fit_independent_session(
        session,
        FIXED_COMMUNICATION_ORDER,
        context="active",
        checkpoint_dir=checkpoints / "ravel_M2",
        config=paper_config(),
    )
    full = fit["model"]
    atoms = fit["atoms"]

    static = fit_session_model(
        session, atoms, "active", dynamic=False, config=paper_config()
    )
    nuisance = fit_session_model(
        session, [], "active", dynamic=False, config=paper_config()
    )

    rows = []
    for label, model in (
        ("ravel_dynamic_recruitment_M2", full),
        ("ravel_static_recruitment_M2", static),
        ("ravel_nuisance_only", nuisance),
    ):
        rows.append({"world": world, "seed": seed, "model": label, **_score(model, session)})

    matched_dynamic = None
    if not args.skip_matched_dynamic:
        matched_dynamic = fit_matched_dynamic_delay_baseline(session)
        rows.append(
            {
                "world": world,
                "seed": seed,
                "model": "matched_dynamic_delay_K2",
                **matched_dynamic["test"],
            }
        )
        matched_dynamic["candidate_table"].to_csv(
            run_dir / "matched_dynamic_candidates.csv", index=False
        )

    score_table = pd.DataFrame(rows)
    score_table.to_csv(run_dir / "test_scores.csv", index=False)

    mechanism: dict[str, Any] = {}
    truth_apparent = apparent_truth_delay_ms(ds, session.freqs_hz)
    ravel_apparent = ravel_apparent_delay_ms(
        full,
        session.freqs_hz,
        nuisance_degree=paper_config().model.nuisance_degree,
    )
    mechanism["truth_aggregate_apparent_delay_ms_w"] = truth_apparent.tolist()
    mechanism["ravel_aggregate_apparent_delay_ms_w"] = ravel_apparent.tolist()
    mechanism["ravel_aggregate_apparent_delay_rmse_ms"] = rmse(
        truth_apparent, ravel_apparent
    )
    mechanism["ravel_aggregate_apparent_delay_corr"] = pearson_or_nan(
        truth_apparent, ravel_apparent
    )

    if world == WORLD_STABLE:
        matching = matched_stable_truth_to_fit(ds, full, session.freqs_hz)
        recruitment = stable_recruitment_correlations(ds, full, matching)
        matching.to_csv(run_dir / "ravel_stable_atom_matching.csv", index=False)
        recruitment.to_csv(run_dir / "ravel_stable_recruitment_recovery.csv", index=False)
        mechanism["ravel_stable_atom_matching"] = matching.to_dict("records")
        mechanism["ravel_recruitment_correlations"] = [
            {
                "truth_index": int(x["truth_index"]),
                "fit_index": int(x["fit_index"]),
                "recruitment_corr": float(x["recruitment_corr"]),
            }
            for x in recruitment.to_dict("records")
        ]
        if matched_dynamic is not None:
            W = matched_dynamic["tau_ms_wm"].shape[0]
            truth_fixed = np.tile(
                np.asarray(ds["truth"]["stable_delays_ms"], float)[None, :],
                (W, 1),
            )
            dyn_match = match_dynamic_trajectories(
                matched_dynamic["tau_ms_wm"], truth_fixed
            )
            dyn_match.to_csv(run_dir / "matched_dynamic_stable_delay_matching.csv", index=False)
            mechanism["matched_dynamic_stable_delay_matching"] = dyn_match.to_dict("records")
            mechanism["matched_dynamic_delay_temporal_std_ms"] = np.std(
                matched_dynamic["tau_ms_wm"], axis=0
            ).tolist()
    else:
        fixed = dynamic_truth_vs_fixed_atoms(ds, full)
        fixed.to_csv(run_dir / "ravel_dynamic_truth_vs_fixed_atoms.csv", index=False)
        mechanism["ravel_dynamic_truth_vs_fixed_atoms"] = fixed.to_dict("records")
        mechanism["true_dynamic_delay_ms_wm"] = np.asarray(
            ds["truth"]["dynamic_delay_ms_wm"], float
        ).tolist()
        if matched_dynamic is not None:
            dyn_match = match_dynamic_trajectories(
                matched_dynamic["tau_ms_wm"], ds["truth"]["dynamic_delay_ms_wm"]
            )
            dyn_match.to_csv(run_dir / "matched_dynamic_trajectory_matching.csv", index=False)
            mechanism["matched_dynamic_trajectory_matching"] = dyn_match.to_dict("records")
            mechanism["matched_dynamic_mean_trajectory_rmse_ms"] = float(
                dyn_match["trajectory_rmse_ms"].mean()
            )
            mechanism["matched_dynamic_mean_trajectory_corr"] = float(
                dyn_match["trajectory_corr"].mean()
            )

    fit["strategy_table"].to_csv(run_dir / "ravel_strategy_table.csv", index=False)
    fit["restart_summary"].to_csv(run_dir / "ravel_restart_summary.csv", index=False)
    fit["restart_details"].to_csv(run_dir / "ravel_restart_details.csv", index=False)
    save_physics_json(
        session.route_name,
        atoms,
        run_dir / "ravel_physics.json",
        metadata={
            "protocol": PROTOCOL,
            "world": world,
            "seed": int(seed),
            "fixed_M": FIXED_COMMUNICATION_ORDER,
            "selected_strategy": fit["strategy"],
        },
    )
    save_pickle(fit, run_dir / "ravel_fit.pkl")
    save_config(paper_config(), run_dir / "ravel_paper_config.json")

    summary = {
        "protocol": PROTOCOL,
        "experiment": "synthetic_mechanism_discrimination",
        "world": world,
        "seed": int(seed),
        "dataset": _portable_dataset_ref(dataset_path),
        "n_trials": int(cfg_synth.n_trials),
        "split_sizes": {k: int(len(v)) for k, v in ds["splits"].items()},
        "ravel_version": paper_config().version,
        "ravel_config_preset": paper_config().preset_name,
        "communication_order_fixed": True,
        "fixed_M": FIXED_COMMUNICATION_ORDER,
        "model_order_selection_performed": False,
        "ravel_atoms": _atoms_json(atoms),
        "selected_strategy": fit["strategy"],
        "scores": score_table.to_dict("records"),
        "mechanism": mechanism,
        "matched_dynamic_baseline": None
        if matched_dynamic is None
        else {
            "fixed_order": 2,
            "static_gains_dynamic_delays": True,
            "pole_hz": float(matched_dynamic["pole_hz"]),
            "smoothness": float(matched_dynamic["smoothness"]),
            "static_gain_m": np.asarray(matched_dynamic["static_gain_m"], float).tolist(),
            "tau_ms_wm": np.asarray(matched_dynamic["tau_ms_wm"], float).tolist(),
            "val_select_nll": float(matched_dynamic["val_select_nll"]),
            "val_gate": matched_dynamic["val_gate"],
        },
        "truth": {
            "stable_delays_ms": np.asarray(ds["truth"]["stable_delays_ms"], float).tolist(),
            "stable_poles_hz": np.asarray(ds["truth"]["stable_poles_hz"], float).tolist(),
            "dynamic_delay_ms_wm": np.asarray(ds["truth"]["dynamic_delay_ms_wm"], float).tolist(),
            "dynamic_poles_hz": np.asarray(ds["truth"]["dynamic_poles_hz"], float).tolist(),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nHeld-out TEST scores")
    print(score_table.to_string(index=False))
    print("RAVEL atoms:", [a.label() for a in atoms])
    if world == WORLD_STABLE:
        print("Stable atom recovery:")
        print(pd.DataFrame(mechanism["ravel_stable_atom_matching"]).to_string(index=False))
        print("Recruitment correlations:", mechanism["ravel_recruitment_correlations"])
    elif matched_dynamic is not None:
        print(
            "Matched dynamic mean trajectory RMSE:",
            f"{mechanism['matched_dynamic_mean_trajectory_rmse_ms']:.3f} ms",
        )
    return summary


def main() -> None:
    args = parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for world in args.worlds:
        for seed in args.seeds:
            summaries.append(run_one(world, int(seed), args))

    rows = [rec for s in summaries for rec in s["scores"]]
    pd.DataFrame(rows).to_csv(args.out / "figure2_ravel_all_test_scores.csv", index=False)
    print("\nSaved:", args.out / "figure2_ravel_all_test_scores.csv")


if __name__ == "__main__":
    main()
