from __future__ import annotations

"""ADM wrapper for the synthetic mechanism-discrimination benchmark.

Fits ADM with candidate across-region latent dimensions K (default 1, 2, 3)
on exactly the same TRAIN/validation splits used by the synthetic benchmark.
K is selected using ADM's native validation marginal likelihood; the sealed
TEST split is opened only after model selection. The selected model is then
converted to complete-target source-only predictions and scored by the common
spectral evaluator used for the paper comparison.

Defaults reproduce the submitted benchmark configuration
(lag_across=2, lag_within=2, initial kernel scale=0.1, and 200 deflation
iterations per round).
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
RAVEL_ROOT = HERE.parents[1]
if str(RAVEL_ROOT) not in sys.path:
    sys.path.insert(0, str(RAVEL_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    PROTOCOL,
    VALID_WORLDS,
    WORLD_DYNAMIC,
    match_dynamic_trajectories,
)

ADM_PROTOCOL = "FIGURE2_ADM_NATIVE_K_SELECTION_v2"


def _default_adm_repo() -> Path:
    candidates = [
        RAVEL_ROOT.parent / "Adaptive-Delay-Model",
        RAVEL_ROOT / "Adaptive-Delay-Model",
    ]
    for path in candidates:
        if (path / "examples").exists():
            return path
    return candidates[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=HERE / "results")
    p.add_argument("--adm-repo", type=Path, default=_default_adm_repo())
    p.add_argument("--worlds", nargs="+", default=list(VALID_WORLDS), choices=VALID_WORLDS)
    p.add_argument("--seeds", nargs="+", type=int, default=[271101])
    p.add_argument(
        "--k-grid", nargs="+", type=int, default=[1, 2, 3],
        help="ADM across-region latent counts to compare. Selection uses validation NLL only.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-within", type=int, default=1)
    p.add_argument("--lag-across", type=int, default=2)
    p.add_argument("--lag-within", type=int, default=2)
    p.add_argument("--init-sigma-across", type=float, default=0.1)
    p.add_argument("--init-sigma-within", type=float, default=0.1)
    p.add_argument("--deflation-iters", type=int, default=200)
    p.add_argument("--num-iters", type=int, default=300)
    p.add_argument("--lr-deflation", type=float, default=1e-2)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--lr-min", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed-offset", type=int, default=20270000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--target-mask-variance", type=float, default=1e10)
    p.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force", action="store_true", help="Rerun even if adm_summary.json exists")
    return p.parse_args()


def load_dataset(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    protocol = str(z["protocol"])
    if protocol != PROTOCOL:
        raise RuntimeError(f"Dataset protocol {protocol!r} != expected {PROTOCOL!r}")
    config = json.loads(str(z["config_json"]))
    return {
        "protocol": protocol,
        "world": str(z["world"]),
        "seed": int(z["seed"]),
        "config": config,
        "source": z["source"],
        "target": z["target"],
        "y": z["y"],
        "y_dims": tuple(int(x) for x in z["y_dims"]),
        "split_train": z["split_train"].astype(int),
        "split_val_select": z["split_val_select"].astype(int),
        "split_val_gate": z["split_val_gate"].astype(int),
        "split_test": z["split_test"].astype(int),
        "truth_stable_delays_ms": z["truth_stable_delays_ms"].astype(float),
        "truth_dynamic_delay_ms_wm": z["truth_dynamic_delay_ms_wm"].astype(float),
        "truth_window_centers_s": z["truth_window_centers_s"].astype(float),
        "fs": float(config["fs_hz"]),
    }


def native_marginal_nll_per_scalar(model, data) -> float:
    engine = model.inference
    with torch.no_grad():
        if hasattr(engine, "_marginal_ll"):
            ll = engine._marginal_ll(model, data)
        else:
            ll = engine._loss_value(model, data)
    return -float(ll.detach().cpu()) / max(int(data.y.numel()), 1)


def ravel_window_average(arr_tk: np.ndarray, *, fs: float) -> np.ndarray:
    arr = np.asarray(arr_tk, float)
    if arr.ndim == 1:
        arr = arr[:, None]
    n_window = int(round(0.40 * fs))
    n_step = int(round(0.25 * fs))
    starts = np.arange(0, arr.shape[0] - n_window + 1, n_step, dtype=int)
    return np.stack([arr[s:s + n_window].mean(axis=0) for s in starts], axis=0)


def stable_delay_metrics(delay_ms_wk: np.ndarray, truth_delays_ms: np.ndarray) -> dict[str, Any]:
    fitted = np.asarray(delay_ms_wk, float)
    med = np.median(fitted, axis=0)
    truth = np.asarray(truth_delays_ms, float)
    cost = np.abs(truth[:, None] - med[None, :])
    rows, cols = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(rows, cols):
        trajectory = fitted[:, j]
        matches.append(
            {
                "truth_index": int(i),
                "latent_index": int(j),
                "truth_delay_ms": float(truth[i]),
                "median_delay_ms": float(np.median(trajectory)),
                "median_delay_error_ms": float(np.median(trajectory) - truth[i]),
                "temporal_std_ms": float(np.std(trajectory)),
                "rmse_to_fixed_truth_ms": float(
                    np.sqrt(np.mean((trajectory - truth[i]) ** 2))
                ),
            }
        )
    return {
        "matches": matches,
        "mean_abs_median_delay_error_ms": float(
            np.mean([abs(m["median_delay_error_ms"]) for m in matches])
        ),
        "mean_fixed_truth_rmse_ms": float(
            np.mean([m["rmse_to_fixed_truth_ms"] for m in matches])
        ),
    }


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _make_adm_data(ds, indices, device, dtype, MultiRegionData):
    idx = np.asarray(indices, dtype=int)
    yt = torch.tensor(np.asarray(ds["y"])[idx], dtype=dtype, device=device)
    return MultiRegionData(y=yt, y_dims=ds["y_dims"], bin_width=1.0)


def build_adm_model(*, ds, k: int, args, adm_modules):
    (
        _demo,
        ADM,
        KalmanEMEngine,
        LatentSpec,
        MOSEKernel,
        _MultiRegionData,
        _MultiRegionLinearObservation,
    ) = adm_modules
    device = _resolve_device(args.device)
    dtype = torch.float64
    T = int(np.asarray(ds["y"]).shape[1])
    y_dims = ds["y_dims"]
    spec = LatentSpec(
        n_across=int(k),
        n_within=(int(args.n_within),) * 2,
    )
    engine = KalmanEMEngine(
        lr=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
    )
    model = ADM(
        latent_spec=spec,
        y_dims=y_dims,
        T=T,
        kernel_factory_across=lambda: MOSEKernel(
            num_regions=2, init_sigma=float(args.init_sigma_across)
        ),
        kernel_factory_within=lambda: MOSEKernel(
            num_regions=1, init_sigma=float(args.init_sigma_within)
        ),
        lag_across=int(args.lag_across),
        lag_within=int(args.lag_within),
        delay_smoothing_sigma_across=float(args.init_sigma_across),
        engine=engine,
        device=device,
        dtype=dtype,
    ).to(device)
    return model


def fit_adm_once(*, ds, k: int, fit_seed: int, args, adm_modules):
    (
        demo,
        ADM,
        KalmanEMEngine,
        LatentSpec,
        MOSEKernel,
        MultiRegionData,
        _MultiRegionLinearObservation,
    ) = adm_modules

    device = _resolve_device(args.device)
    dtype = torch.float64
    T = int(np.asarray(ds["y"]).shape[1])
    y_dims = ds["y_dims"]
    if len(y_dims) != 2:
        raise ValueError("Synthetic ADM wrapper expects exactly two regions")

    train_data = _make_adm_data(ds, ds["split_train"], device, dtype, MultiRegionData)
    val_select_data = _make_adm_data(ds, ds["split_val_select"], device, dtype, MultiRegionData)
    val_gate_data = _make_adm_data(ds, ds["split_val_gate"], device, dtype, MultiRegionData)

    torch.manual_seed(int(fit_seed))
    np.random.seed(int(fit_seed) % (2**32 - 1))

    model = build_adm_model(ds=ds, k=int(k), args=args, adm_modules=adm_modules)

    print(f"  phase 1: pCCA initialization | candidate K={int(k)}")
    t0 = time.perf_counter()
    model.initialize_from_data(train_data, mode="pcca")
    print(f"  phase 2: rank-1 deflation ({args.deflation_iters} iterations/round)")
    defl = demo.rank1_deflation_init(
        model,
        train_data,
        n_iters_per_round=args.deflation_iters,
        lr=args.lr_deflation,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        verbose=True,
    )
    print(f"  phase 3: ADM fit ({args.num_iters} iterations)")
    fit_result = model.fit(train_data, max_iter=args.num_iters, tol=1e-8)
    wall_s = time.perf_counter() - t0

    delay_bins_tk = demo.extract_delay(model, T)[:, 0, :]
    delay_ms_tk = delay_bins_tk * (1000.0 / ds["fs"])
    delay_ms_wk = ravel_window_average(delay_ms_tk, fs=ds["fs"])

    diagnostics = {
        "train_native_marginal_nll_per_scalar": native_marginal_nll_per_scalar(model, train_data),
        "val_select_native_marginal_nll_per_scalar": native_marginal_nll_per_scalar(model, val_select_data),
        "val_gate_native_marginal_nll_per_scalar": native_marginal_nll_per_scalar(model, val_gate_data),
        "wall_s": float(wall_s),
        "fit_iters": int(fit_result.n_iter),
        "deflation_rounds": int(defl["n_rounds"]),
    }
    arrays = {
        "delay_ms_tk": np.asarray(delay_ms_tk, np.float64),
        "delay_ms_wk": np.asarray(delay_ms_wk, np.float64),
        "score_trace": np.asarray(fit_result.score_trace, np.float64),
    }
    return model, diagnostics, arrays


def make_source_only_test_data(data, y_dims, MultiRegionData):
    y_masked = data.y.clone()
    source_dim = int(y_dims[0])
    y_masked[:, :, source_dim:] = 0.0
    return MultiRegionData(y=y_masked, y_dims=y_dims, bin_width=data.bin_width)


def adm_complete_target_prediction(
    *, model, test_data, y_dims, target_mask_variance, demo, MultiRegionData,
    MultiRegionLinearObservation,
) -> np.ndarray:
    obs = model.observation
    if not isinstance(obs, MultiRegionLinearObservation):
        raise TypeError(f"Expected MultiRegionLinearObservation, got {type(obs).__name__}")
    source_dim = int(y_dims[0])
    target_cols = torch.arange(
        source_dim,
        sum(int(x) for x in y_dims),
        device=obs.diag_R_param.device,
        dtype=torch.long,
    )
    masked_data = make_source_only_test_data(test_data, y_dims, MultiRegionData)
    original_R = obs.diag_R_param.data.clone()
    try:
        with torch.no_grad():
            obs.diag_R_param.data[target_cols] = float(target_mask_variance)
        y_recon = demo.extract_y_recon(model, masked_data)
    finally:
        with torch.no_grad():
            obs.diag_R_param.data.copy_(original_R)
    return np.asarray(y_recon[:, :, source_dim:], dtype=np.float64)


def build_ravel_reference_and_target_transform(ds):
    from ravel import build_route_session_from_epochs, paper_config
    from ravel.preprocess import detrend_and_lowpass, fit_region_transform, zero_phase_notch

    cfg = paper_config()
    metadata = pd.DataFrame(
        {
            "outcome": ["synthetic"] * len(ds["source"]),
            "trial_index": np.arange(len(ds["source"]), dtype=int),
        }
    )
    splits = {
        "active": {
            "train": ds["split_train"],
            "val_select": ds["split_val_select"],
            "val_gate": ds["split_val_gate"],
            "test": ds["split_test"],
        }
    }
    session = build_route_session_from_epochs(
        source_epochs={"active": ds["source"]},
        target_epochs={"active": ds["target"]},
        metadata={"active": metadata},
        route_name="synthetic_A_to_B",
        source_region="A",
        target_region="B",
        session_id=f"adm_common_eval_{ds['world']}_{ds['seed']}",
        subject_id=f"adm_common_eval_{ds['seed']}",
        d=1,
        experience=ds["world"],
        reference_context="active",
        splits=splits,
        input_is_preconditioned=False,
        config=cfg,
    )

    target_conditioned = detrend_and_lowpass(ds["target"], cfg)
    target_conditioned = zero_phase_notch(target_conditioned, cfg)
    target_transform = fit_region_transform(
        target_conditioned[np.asarray(ds["split_train"], dtype=int)]
    )
    return session, target_transform, cfg


def adm_prediction_to_ravel_fourier(
    prediction_target_time: np.ndarray,
    *, target_transform, cfg, reference_freqs_hz, reference_window_centers_s,
) -> np.ndarray:
    from ravel.preprocess import (
        apply_region_transform,
        detrend_and_lowpass,
        windowed_multitaper,
        zero_phase_notch,
    )

    values = detrend_and_lowpass(np.asarray(prediction_target_time, np.float32), cfg)
    values = zero_phase_notch(values, cfg)
    values = apply_region_transform(values, target_transform)
    freqs, centers, fourier = windowed_multitaper(values, cfg)
    if not np.allclose(freqs, reference_freqs_hz):
        raise RuntimeError("ADM prediction frequency grid differs from RAVEL grid")
    if not np.allclose(centers, reference_window_centers_s):
        raise RuntimeError("ADM prediction window grid differs from RAVEL grid")
    return np.asarray(fourier, np.complex64)


def score_external_ravel_prediction(*, session, predicted_fourier, test_positions, cfg):
    from ravel.metrics import complex_gaussian_trial_nll, fit_evaluation_covariance

    test_positions = np.asarray(test_positions, dtype=int)
    observed = np.asarray(session.contexts["active"].target_raw[test_positions])
    predicted = np.asarray(predicted_fourier)
    if predicted.shape != observed.shape:
        raise ValueError(f"prediction shape {predicted.shape} != observed {observed.shape}")
    covariance = fit_evaluation_covariance(session, "active", cfg)
    trial_nll = complex_gaussian_trial_nll(observed, predicted, covariance)
    error_power = np.mean(np.abs(observed - predicted) ** 2, axis=(1, 2, 3, 4))
    target_power = np.mean(np.abs(observed) ** 2, axis=(1, 2, 3, 4)) + 1e-12
    trial_nmse = error_power / target_power
    trial_r2 = 1.0 - trial_nmse
    return (
        {
            "nll": float(np.mean(trial_nll)),
            "nmse": float(np.mean(trial_nmse)),
            "r2": float(np.mean(trial_r2)),
            "n_test_trials": int(len(trial_nll)),
        },
        {
            "trial_nll": np.asarray(trial_nll, float),
            "trial_nmse": np.asarray(trial_nmse, float),
            "trial_r2": np.asarray(trial_r2, float),
            "observed_fourier": observed,
            "predicted_fourier": predicted,
        },
    )


def write_trial_scores(path: Path, positions: np.ndarray, arrays: dict[str, np.ndarray]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["test_order", "global_trial_index", "nll", "nmse", "r2"]
        )
        writer.writeheader()
        for i, pos in enumerate(np.asarray(positions, dtype=int)):
            writer.writerow(
                {
                    "test_order": int(i),
                    "global_trial_index": int(pos),
                    "nll": float(arrays["trial_nll"][i]),
                    "nmse": float(arrays["trial_nmse"][i]),
                    "r2": float(arrays["trial_r2"][i]),
                }
            )



def _candidate_mechanism(world: str, ds: dict[str, Any], delay_ms_wk: np.ndarray) -> dict[str, Any]:
    """Mechanism diagnostic only; never used to select K."""
    if world == WORLD_DYNAMIC:
        matching = match_dynamic_trajectories(delay_ms_wk, ds["truth_dynamic_delay_ms_wm"])
        return {
            "n_truth_matched": int(len(matching)),
            "trajectory_matching": matching.to_dict("records"),
            "mean_trajectory_rmse_ms": float(matching["trajectory_rmse_ms"].mean()),
            "mean_trajectory_corr": float(matching["trajectory_corr"].mean()),
        }
    out = stable_delay_metrics(delay_ms_wk, ds["truth_stable_delays_ms"])
    out["n_truth_matched"] = int(len(out["matches"]))
    return out


def _candidate_row(*, world: str, seed: int, k: int, diagnostics: dict[str, Any], mechanism: dict[str, Any]) -> dict[str, Any]:
    val_mean = 0.5 * (
        float(diagnostics["val_select_native_marginal_nll_per_scalar"])
        + float(diagnostics["val_gate_native_marginal_nll_per_scalar"])
    )
    row = {
        "world": world,
        "seed": int(seed),
        "K": int(k),
        "train_native_nll_per_scalar": float(diagnostics["train_native_marginal_nll_per_scalar"]),
        "val_select_native_nll_per_scalar": float(diagnostics["val_select_native_marginal_nll_per_scalar"]),
        "val_gate_native_nll_per_scalar": float(diagnostics["val_gate_native_marginal_nll_per_scalar"]),
        "val_mean_native_nll_per_scalar": float(val_mean),
        "wall_s": float(diagnostics["wall_s"]),
        "fit_iters": int(diagnostics["fit_iters"]),
        "n_truth_matched": int(mechanism.get("n_truth_matched", 0)),
    }
    if world == WORLD_DYNAMIC:
        row["mechanism_rmse_ms"] = float(mechanism["mean_trajectory_rmse_ms"])
        row["mechanism_corr"] = float(mechanism["mean_trajectory_corr"])
    else:
        row["mechanism_rmse_ms"] = float(mechanism["mean_fixed_truth_rmse_ms"])
        row["mechanism_corr"] = np.nan
    return row


def _portable_dataset_ref(path: Path) -> str:
    """Store a repository-relative path when possible; never save a home path."""
    path = Path(path)
    try:
        return path.resolve().relative_to(HERE.resolve()).as_posix()
    except ValueError:
        return path.name


def run_one_sweep(*, world: str, seed: int, dataset_path: Path, out_dir: Path, args, adm_modules):
    (
        demo,
        _ADM,
        _KalmanEMEngine,
        _LatentSpec,
        _MOSEKernel,
        MultiRegionData,
        MultiRegionLinearObservation,
    ) = adm_modules

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(dataset_path)
    device = _resolve_device(args.device)
    k_grid = sorted({int(k) for k in args.k_grid})
    if not k_grid or any(k < 1 for k in k_grid):
        raise ValueError(f"--k-grid must contain positive integers, got {args.k_grid}")

    print("\n" + "=" * 92)
    print(f"SYNTHETIC BENCHMARK | ADM NATIVE K SELECTION | world={world} | seed={seed}")
    print(f"dataset       : {dataset_path}")
    print(f"K candidates  : {k_grid}")
    print(f"train trials  : {len(ds['split_train'])}")
    print("selection     : lowest mean native NLL across val_select + val_gate")
    print("TEST policy   : TEST is opened only for the validation-selected K")

    candidate_rows: list[dict[str, Any]] = []
    candidate_state_paths: dict[int, Path] = {}
    candidate_arrays: dict[int, dict[str, np.ndarray]] = {}
    candidate_mechanisms: dict[int, dict[str, Any]] = {}
    candidate_fit_seeds: dict[int, int] = {}

    for k in k_grid:
        print("\n" + "-" * 92)
        print(f"ADM candidate K={k}")
        fit_seed = int(args.seed_offset + seed * 23 + int(k) * 1009)
        candidate_fit_seeds[k] = fit_seed
        model, diagnostics, arrays = fit_adm_once(
            ds=ds, k=int(k), fit_seed=fit_seed, args=args, adm_modules=adm_modules
        )
        mechanism = _candidate_mechanism(world, ds, arrays["delay_ms_wk"])
        row = _candidate_row(
            world=world, seed=seed, k=k, diagnostics=diagnostics, mechanism=mechanism
        )
        candidate_rows.append(row)
        candidate_arrays[k] = arrays
        candidate_mechanisms[k] = mechanism

        k_dir = out_dir / f"K_{k}"
        k_dir.mkdir(parents=True, exist_ok=True)
        (k_dir / "candidate_summary.json").write_text(
            json.dumps(
                {
                    "protocol": ADM_PROTOCOL,
                    "world": world,
                    "seed": int(seed),
                    "K": int(k),
                    "fit_seed": int(fit_seed),
                    "fit_diagnostics": diagnostics,
                    "selection_value_native_val_nll": row["val_mean_native_nll_per_scalar"],
                    "mechanism_diagnostic_NOT_used_for_selection": mechanism,
                    "test_opened": False,
                },
                indent=2,
            )
        )
        np.savez_compressed(
            k_dir / "candidate_arrays.npz",
            delay_ms_tk=arrays["delay_ms_tk"],
            delay_ms_wk=arrays["delay_ms_wk"],
            score_trace=arrays["score_trace"],
        )

        # Save fitted parameters on CPU so GPU memory does not accumulate.
        state_path = k_dir / "candidate_state_dict.pt"
        torch.save(
            {name: value.detach().cpu() for name, value in model.state_dict().items()},
            state_path,
        )
        candidate_state_paths[k] = state_path
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    candidates_df = pd.DataFrame(candidate_rows).sort_values("K").reset_index(drop=True)
    candidates_df.to_csv(out_dir / "adm_K_sweep_candidates.csv", index=False)

    # Lower native marginal NLL is better. No TEST quantity participates here.
    best_idx = candidates_df["val_mean_native_nll_per_scalar"].astype(float).idxmin()
    selected_k = int(candidates_df.loc[best_idx, "K"])
    print("\nValidation-only K selection")
    print(candidates_df.to_string(index=False))
    print(f"\nSELECTED K = {selected_k}")

    # Only now open TEST for the selected K.
    selected_model = build_adm_model(
        ds=ds, k=selected_k, args=args, adm_modules=adm_modules
    )
    selected_state = torch.load(candidate_state_paths[selected_k], map_location=device)
    selected_model.load_state_dict(selected_state)
    arrays = candidate_arrays[selected_k]
    mechanism = candidate_mechanisms[selected_k]
    dtype = torch.float64
    test_data = _make_adm_data(ds, ds["split_test"], device, dtype, MultiRegionData)
    native_test_nll = native_marginal_nll_per_scalar(selected_model, test_data)

    print("  phase TEST-1: strict source-only complete-target inference")
    prediction_target_time = adm_complete_target_prediction(
        model=selected_model,
        test_data=test_data,
        y_dims=ds["y_dims"],
        target_mask_variance=float(args.target_mask_variance),
        demo=demo,
        MultiRegionData=MultiRegionData,
        MultiRegionLinearObservation=MultiRegionLinearObservation,
    )

    print("  phase TEST-2: frozen RAVEL transform + common evaluator")
    ravel_session, target_transform, ravel_cfg = build_ravel_reference_and_target_transform(ds)
    prediction_fourier = adm_prediction_to_ravel_fourier(
        prediction_target_time,
        target_transform=target_transform,
        cfg=ravel_cfg,
        reference_freqs_hz=ravel_session.freqs_hz,
        reference_window_centers_s=ravel_session.window_centers_s,
    )
    common_summary, common_arrays = score_external_ravel_prediction(
        session=ravel_session,
        predicted_fourier=prediction_fourier,
        test_positions=ds["split_test"],
        cfg=ravel_cfg,
    )
    write_trial_scores(
        out_dir / "selected_K_common_test_trial_scores.csv",
        ds["split_test"],
        common_arrays,
    )

    summary = {
        "protocol": ADM_PROTOCOL,
        "dataset_protocol": PROTOCOL,
        "model": "ADM_official",
        "world": world,
        "seed": int(seed),
        "dataset": _portable_dataset_ref(dataset_path),
        "device": str(device),
        "k_grid": k_grid,
        "model_order_selection_performed": len(k_grid) > 1,
        "selection_rule": "minimum mean native marginal NLL across val_select and val_gate",
        "selection_uses_test": False,
        "selected_K": int(selected_k),
        "selected_fit_seed": int(candidate_fit_seeds[selected_k]),
        "candidate_table": candidates_df.to_dict("records"),
        "selected_mechanism": mechanism,
        "native_test_marginal_nll_per_scalar": float(native_test_nll),
        "common_evaluator": {
            "world": world,
            "seed": int(seed),
            "model": f"adm_K{selected_k}",
            **common_summary,
        },
        "source_only_inference": {
            "target_values_zeroed": True,
            "target_observation_variance": float(args.target_mask_variance),
            "scored_channels": "all target-region channels",
        },
        "split_sizes": {
            "train": int(len(ds["split_train"])),
            "val_select": int(len(ds["split_val_select"])),
            "val_gate": int(len(ds["split_val_gate"])),
            "test": int(len(ds["split_test"])),
        },
    }
    summary_path = out_dir / "adm_K_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        out_dir / "selected_K_arrays.npz",
        delay_ms_tk=arrays["delay_ms_tk"],
        delay_ms_wk=arrays["delay_ms_wk"],
        score_trace=arrays["score_trace"],
        truth_stable_delays_ms=ds["truth_stable_delays_ms"],
        truth_dynamic_delay_ms_wm=ds["truth_dynamic_delay_ms_wm"],
    )

    if args.save_predictions:
        np.savez_compressed(
            out_dir / "selected_K_source_only_test_predictions.npz",
            global_test_positions=np.asarray(ds["split_test"], dtype=int),
            target_prediction_time=np.asarray(prediction_target_time, np.float32),
            target_prediction_ravel_fourier=np.asarray(prediction_fourier, np.complex64),
            target_observed_ravel_fourier=np.asarray(common_arrays["observed_fourier"], np.complex64),
        )

    print("\nSELECTED-K common TEST score")
    print(json.dumps(summary["common_evaluator"], indent=2))
    print("SELECTED-K mechanism diagnostic")
    print(json.dumps(mechanism, indent=2))
    print("saved:", summary_path)

    # Release selected model.
    del selected_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary, candidates_df


def main() -> None:
    args = parse_args()
    examples_dir = args.adm_repo.resolve() / "examples"
    if not (examples_dir / "demo_common.py").exists():
        raise FileNotFoundError(
            f"Could not find {examples_dir / 'demo_common.py'}. "
            "Pass --adm-repo pointing to Adaptive-Delay-Model."
        )
    sys.path.insert(0, str(examples_dir))

    import demo_common as demo  # type: ignore
    from mbrila import ADM, KalmanEMEngine, LatentSpec, MOSEKernel
    from mbrila.core.data import MultiRegionData
    from mbrila.observations.multi_region import MultiRegionLinearObservation

    adm_modules = (
        demo,
        ADM,
        KalmanEMEngine,
        LatentSpec,
        MOSEKernel,
        MultiRegionData,
        MultiRegionLinearObservation,
    )

    print("ADM repo:", args.adm_repo.resolve())
    print("benchmark results root:", args.results_root.resolve())
    print("protocol:", ADM_PROTOCOL)
    print("K grid:", sorted({int(k) for k in args.k_grid}))
    print("Selection is validation-only; TEST is opened only for selected K.")

    all_candidate_frames = []
    selected_rows = []
    for world in args.worlds:
        for seed in args.seeds:
            dataset_path = args.results_root / world / f"seed_{seed}" / "data" / "dataset.npz"
            if not dataset_path.exists():
                raise FileNotFoundError(
                    f"Missing {dataset_path}. Run run_ravel.py first; ADM never regenerates benchmark data."
                )
            out_dir = args.results_root / world / f"seed_{seed}" / "adm_K_sweep"
            summary, candidates_df = run_one_sweep(
                world=world,
                seed=int(seed),
                dataset_path=dataset_path,
                out_dir=out_dir,
                args=args,
                adm_modules=adm_modules,
            )
            all_candidate_frames.append(candidates_df)
            selected_rows.append(
                {
                    "world": world,
                    "seed": int(seed),
                    "selected_K": int(summary["selected_K"]),
                    "test_nll": float(summary["common_evaluator"]["nll"]),
                    "test_nmse": float(summary["common_evaluator"]["nmse"]),
                    "test_r2": float(summary["common_evaluator"]["r2"]),
                }
            )

    if all_candidate_frames:
        all_candidates = pd.concat(all_candidate_frames, ignore_index=True)
        all_candidates.to_csv(args.results_root / "figure2_adm_K_candidates.csv", index=False)
        selected_df = pd.DataFrame(selected_rows)
        selected_df.to_csv(args.results_root / "figure2_adm_K_selected.csv", index=False)
        print("\n" + "=" * 92)
        print("ALL K-SWEEP CANDIDATES")
        print(all_candidates.to_string(index=False))
        print("\nVALIDATION-SELECTED K AND SEALED TEST")
        print(selected_df.to_string(index=False))
        print("\nsaved:", args.results_root / "figure2_adm_K_candidates.csv")
        print("saved:", args.results_root / "figure2_adm_K_selected.csv")


if __name__ == "__main__":
    main()
