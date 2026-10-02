from __future__ import annotations

"""Aggregate synthetic benchmark outputs using the paper's final ADM selection."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORLDS = ("stable_recruitment", "dynamic_delay")
SEEDS = (271101, 271102, 271103)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=HERE / "results")
    p.add_argument("--out", type=Path, default=HERE / "results" / "summary")
    return p.parse_args()


def load_all(results_root: Path):
    scores = []
    runs = []
    for world in WORLDS:
        for seed in SEEDS:
            run_dir = results_root / world / f"seed_{seed}"
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                continue
            s = json.loads(summary_path.read_text())
            runs.append((run_dir, s))
            scores.extend(s["scores"])
            adm_path = run_dir / "adm_K_sweep" / "adm_K_sweep_summary.json"
            if adm_path.exists():
                adm = json.loads(adm_path.read_text())
                row = dict(adm["common_evaluator"])
                row["model"] = "adm_native_selectedK"
                row["selected_K"] = int(adm["selected_K"])
                scores.append(row)
    return runs, pd.DataFrame(scores)


def mechanism_rows(runs):
    rows = []
    for run_dir, s in runs:
        m = s["mechanism"]
        row = {
            "world": s["world"], "seed": int(s["seed"]),
            "ravel_mean_abs_delay_error_ms": np.nan,
            "ravel_mean_atom_shape_similarity": np.nan,
            "ravel_mean_recruitment_corr": np.nan,
            "ravel_mean_dynamic_truth_to_fixed_rmse_ms": np.nan,
            "matched_dynamic_mean_trajectory_rmse_ms": m.get("matched_dynamic_mean_trajectory_rmse_ms", np.nan),
            "matched_dynamic_mean_trajectory_corr": m.get("matched_dynamic_mean_trajectory_corr", np.nan),
            "adm_selected_K": np.nan,
            "adm_mean_dynamic_trajectory_rmse_ms": np.nan,
            "adm_mean_dynamic_trajectory_corr": np.nan,
        }
        if s["world"] == "stable_recruitment":
            atoms = pd.DataFrame(m.get("ravel_stable_atom_matching", []))
            rec = pd.DataFrame(m.get("ravel_recruitment_correlations", []))
            if len(atoms):
                row["ravel_mean_abs_delay_error_ms"] = float(atoms["delay_error_ms"].abs().mean())
                row["ravel_mean_atom_shape_similarity"] = float(atoms["shape_similarity"].mean())
            if len(rec):
                row["ravel_mean_recruitment_corr"] = float(rec["recruitment_corr"].mean())
        else:
            fixed = pd.DataFrame(m.get("ravel_dynamic_truth_vs_fixed_atoms", []))
            if len(fixed):
                row["ravel_mean_dynamic_truth_to_fixed_rmse_ms"] = float(fixed["trajectory_rmse_to_fixed_ms"].mean())
        adm_path = run_dir / "adm_K_sweep" / "adm_K_sweep_summary.json"
        if adm_path.exists():
            adm = json.loads(adm_path.read_text())
            row["adm_selected_K"] = int(adm["selected_K"])
            mech = adm.get("selected_mechanism", {})
            if s["world"] == "dynamic_delay":
                row["adm_mean_dynamic_trajectory_rmse_ms"] = mech.get("mean_trajectory_rmse_ms", np.nan)
                row["adm_mean_dynamic_trajectory_corr"] = mech.get("mean_trajectory_corr", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs, scores = load_all(args.results_root)
    if scores.empty:
        raise FileNotFoundError(f"No benchmark outputs under {args.results_root}")
    scores.to_csv(args.out / "paper_scores.csv", index=False)
    mechanism_rows(runs).to_csv(args.out / "mechanism_metrics.csv", index=False)
    stats = scores.groupby(["world", "model"])["nmse"].agg(["mean", "std", "count"])
    stats.to_csv(args.out / "nmse_mean_sd.csv")
    print(stats.to_string())


if __name__ == "__main__":
    main()
