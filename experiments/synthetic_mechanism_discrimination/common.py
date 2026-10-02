from __future__ import annotations

"""Common utilities for the Figure 2 synthetic mechanism benchmark.

The generative truth has two delayed components. RAVEL is fit with M=2, the
matched dynamic-delay comparator uses two components, and ADM retains its own
native K selection over {1,2,3}. Synthetic raw epochs pass through the same
RAVEL preprocessing and held-out evaluator used by the method implementation.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


WORLD_STABLE = "stable_recruitment"
WORLD_DYNAMIC = "dynamic_delay"
VALID_WORLDS = (WORLD_STABLE, WORLD_DYNAMIC)
PROTOCOL = "FIGURE2_MECHANISM_DISCRIMINATION_v1"
FIXED_COMMUNICATION_ORDER = 2


@dataclass(frozen=True)
class SyntheticBenchmarkConfig:
    # Keep the submitted RAVEL temporal/spectral geometry intact.
    fs_hz: float = 500.0
    epoch_pre_s: float = 1.0
    epoch_post_s: float = 2.0

    # Submitted benchmark size: 40 -> 24 train / 4 val_select / 4 val_gate / 8 test.
    n_trials: int = 40
    n_channels_per_region: int = 6

    # Broad enough excitation to identify one-pole responses in 6--80 Hz.
    source_spectral_alpha: float = 0.35
    source_low_corner_hz: float = 2.0
    source_high_cut_hz: float = 105.0

    # This mechanism-discrimination benchmark intentionally uses a relatively clean communication signal.
    local_noise_spectral_alpha: float = 0.70
    local_noise_low_corner_hz: float = 2.0
    local_noise_high_cut_hz: float = 90.0
    source_sensor_noise_sd: float = 0.005
    target_within_noise_sd: float = 0.10
    target_sensor_noise_sd: float = 0.05
    channel_loading_log_sd: float = 0.10

    # World A: two fixed propagation atoms, dynamic recruitment.
    stable_delays_ms: tuple[float, float] = (9.5, 21.5)
    stable_poles_hz: tuple[float, float] = (45.0, 45.0)
    stable_gains: tuple[float, float] = (0.70, 0.70)

    # World B: two communication components again, but now amplitudes are
    # static and their transport delays genuinely move over event time.
    dynamic_delay_start_ms: tuple[float, float] = (7.5, 19.0)
    dynamic_delay_end_ms: tuple[float, float] = (14.0, 27.0)
    dynamic_delay_sigmoid_width: tuple[float, float] = (0.16, 0.14)
    dynamic_poles_hz: tuple[float, float] = (45.0, 45.0)
    dynamic_gains: tuple[float, float] = (0.75, 0.65)

    pad_seconds: float = 0.75

    @property
    def n_samples(self) -> int:
        return int(round((self.epoch_pre_s + self.epoch_post_s) * self.fs_hz))

    @property
    def y_dims(self) -> tuple[int, int]:
        c = int(self.n_channels_per_region)
        return (c, c)


def ravel_window_centers_s(cfg: SyntheticBenchmarkConfig) -> np.ndarray:
    n_window = int(round(0.40 * cfg.fs_hz))
    n_step = int(round(0.25 * cfg.fs_hz))
    starts = np.arange(0, cfg.n_samples - n_window + 1, n_step, dtype=int)
    return starts / cfg.fs_hz - cfg.epoch_pre_s + 0.5 * 0.40


def stable_recruitment_profiles(window_centers_s: np.ndarray) -> np.ndarray:
    """Two positive, overlapping, non-complementary event-time profiles."""
    centers = np.asarray(window_centers_s, float)
    u = (centers - centers.min()) / max(float(np.ptp(centers)), 1e-12)
    z = 1.0 / (1.0 + np.exp(-(u - 0.50) / 0.13))
    r0 = 0.38 + 1.30 * (1.0 - z)
    r1 = 0.38 + 1.30 * z
    bump = 0.10 * np.exp(-0.5 * ((u - 0.50) / 0.16) ** 2)
    r0 = r0 + bump
    r1 = r1 + 0.75 * bump
    R = np.column_stack([r0, r1])
    R /= np.maximum(R.mean(axis=0, keepdims=True), 1e-12)
    return R


def dynamic_delay_profiles_ms(window_centers_s: np.ndarray, cfg: SyntheticBenchmarkConfig) -> np.ndarray:
    """Return [window, 2] smooth, ordered transport-delay trajectories."""
    centers = np.asarray(window_centers_s, float)
    u = (centers - centers.min()) / max(float(np.ptp(centers)), 1e-12)
    out = []
    for start, end, width in zip(
        cfg.dynamic_delay_start_ms,
        cfg.dynamic_delay_end_ms,
        cfg.dynamic_delay_sigmoid_width,
    ):
        z = 1.0 / (1.0 + np.exp(-(u - 0.50) / float(width)))
        z = (z - z[0]) / max(float(z[-1] - z[0]), 1e-12)
        out.append(float(start) + (float(end) - float(start)) * z)
    return np.column_stack(out)


def _make_four_way_split(n_trials: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(int(n_trials))
    n_train = int(round(0.60 * n_trials))
    n_vs = int(round(0.10 * n_trials))
    n_vg = int(round(0.10 * n_trials))
    n_test = n_trials - n_train - n_vs - n_vg
    if min(n_vs, n_vg, n_test) < 3:
        raise ValueError("Synthetic benchmark requires >=30 trials so val/test each contain >=3 trials")
    return {
        "train": np.sort(order[:n_train]),
        "val_select": np.sort(order[n_train:n_train + n_vs]),
        "val_gate": np.sort(order[n_train + n_vs:n_train + n_vs + n_vg]),
        "test": np.sort(order[n_train + n_vs + n_vg:]),
    }


def _broadband_colored_process(
    rng: np.random.Generator,
    n: int,
    fs_hz: float,
    *,
    alpha: float,
    low_corner_hz: float,
    high_cut_hz: float,
) -> np.ndarray:
    white = rng.standard_normal(int(n))
    spectrum = np.fft.rfft(white)
    f = np.fft.rfftfreq(int(n), d=1.0 / float(fs_hz))
    f_eff = np.maximum(f, float(low_corner_hz))
    amplitude = (f_eff / float(low_corner_hz)) ** (-0.5 * float(alpha))
    amplitude *= 1.0 / np.sqrt(1.0 + (f / max(float(high_cut_hz), 1e-6)) ** 8)
    amplitude[0] = 0.0
    z = np.fft.irfft(spectrum * amplitude, n=int(n)).real
    z -= z.mean()
    z /= max(float(z.std()), 1e-8)
    return z


def _fractional_delay(x: np.ndarray, delay_samples: float | np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    n = len(x)
    delay = np.asarray(delay_samples, float)
    if delay.ndim == 0:
        delay = np.full(n, float(delay), dtype=float)
    if delay.shape != (n,):
        raise ValueError("delay_samples must be scalar or length len(x)")
    grid = np.arange(n, dtype=float)
    pos = np.clip(grid - delay, 0.0, n - 1.0)
    return np.interp(pos, grid, x)


def _one_pole_filter(x: np.ndarray, pole_hz: float, fs_hz: float) -> np.ndarray:
    x = np.asarray(x, float)
    f = np.fft.rfftfreq(len(x), d=1.0 / float(fs_hz))
    h = 1.0 / (1.0 + 1j * f / float(pole_hz))
    return np.fft.irfft(np.fft.rfft(x) * h, n=len(x)).real


def _fixed_delay_one_pole_filter(
    x: np.ndarray, delay_ms: float, pole_hz: float, fs_hz: float
) -> np.ndarray:
    x = np.asarray(x, float)
    f = np.fft.rfftfreq(len(x), d=1.0 / float(fs_hz))
    h = np.exp(-1j * 2.0 * np.pi * f * float(delay_ms) / 1000.0)
    h /= 1.0 + 1j * f / float(pole_hz)
    return np.fft.irfft(np.fft.rfft(x) * h, n=len(x)).real


def _random_loading(rng: np.random.Generator, channels: int, log_sd: float) -> np.ndarray:
    v = rng.normal(size=int(channels))
    v /= max(float(np.sqrt(np.mean(v**2))), 1e-12)
    scales = np.exp(rng.normal(0.0, float(log_sd), size=int(channels)))
    return v * scales


def _interp_window_profile_to_samples(
    window_values: np.ndarray,
    window_centers_s: np.ndarray,
    cfg: SyntheticBenchmarkConfig,
) -> np.ndarray:
    t = np.arange(cfg.n_samples, dtype=float) / cfg.fs_hz - cfg.epoch_pre_s
    return np.interp(
        t,
        np.asarray(window_centers_s, float),
        np.asarray(window_values, float),
        left=float(window_values[0]),
        right=float(window_values[-1]),
    )


def simulate_s1_dataset(
    world: str,
    *,
    seed: int,
    cfg: SyntheticBenchmarkConfig | None = None,
) -> dict[str, Any]:
    if world not in VALID_WORLDS:
        raise ValueError(f"world must be one of {VALID_WORLDS}, got {world!r}")
    cfg = cfg or SyntheticBenchmarkConfig()
    rng = np.random.default_rng(int(seed))

    C = int(cfg.n_channels_per_region)
    T = int(cfg.n_samples)
    fs = float(cfg.fs_hz)
    pad = int(round(float(cfg.pad_seconds) * fs))
    Text = T + 2 * pad

    centers = ravel_window_centers_s(cfg)
    stable_r = stable_recruitment_profiles(centers)
    dynamic_tau_wm = dynamic_delay_profiles_ms(centers, cfg)

    source_loading = _random_loading(rng, C, cfg.channel_loading_log_sd)
    target_loading = _random_loading(rng, C, cfg.channel_loading_log_sd)

    source = np.zeros((cfg.n_trials, T, C), dtype=np.float32)
    target = np.zeros((cfg.n_trials, T, C), dtype=np.float32)
    latent_source = np.zeros((cfg.n_trials, T), dtype=np.float32)
    latent_target = np.zeros((cfg.n_trials, T), dtype=np.float32)

    if world == WORLD_STABLE:
        sample_r = np.column_stack(
            [_interp_window_profile_to_samples(stable_r[:, m], centers, cfg) for m in range(2)]
        )
    else:
        sample_tau_ms = np.column_stack(
            [
                _interp_window_profile_to_samples(dynamic_tau_wm[:, m], centers, cfg)
                for m in range(2)
            ]
        )
        sample_tau_ext_ms = np.vstack(
            [
                np.full((pad, 2), sample_tau_ms[0], dtype=float),
                sample_tau_ms,
                np.full((pad, 2), sample_tau_ms[-1], dtype=float),
            ]
        )

    for tr in range(cfg.n_trials):
        z_ext = _broadband_colored_process(
            rng,
            Text,
            fs,
            alpha=cfg.source_spectral_alpha,
            low_corner_hz=cfg.source_low_corner_hz,
            high_cut_hz=cfg.source_high_cut_hz,
        )
        z = z_ext[pad:pad + T]

        if world == WORLD_STABLE:
            comps = []
            for m in range(2):
                filt = _fixed_delay_one_pole_filter(
                    z_ext,
                    float(cfg.stable_delays_ms[m]),
                    float(cfg.stable_poles_hz[m]),
                    fs,
                )[pad:pad + T]
                comps.append(
                    float(cfg.stable_gains[m]) * sample_r[:, m] * filt
                )
            target_latent = comps[0] + comps[1]
        else:
            comps = []
            for m in range(2):
                delayed = _fractional_delay(
                    z_ext,
                    sample_tau_ext_ms[:, m] * fs / 1000.0,
                )
                filt = _one_pole_filter(delayed, float(cfg.dynamic_poles_hz[m]), fs)
                comps.append(float(cfg.dynamic_gains[m]) * filt[pad:pad + T])
            target_latent = comps[0] + comps[1]

        eta_target = float(cfg.target_within_noise_sd) * _broadband_colored_process(
            rng,
            T,
            fs,
            alpha=cfg.local_noise_spectral_alpha,
            low_corner_hz=cfg.local_noise_low_corner_hz,
            high_cut_hz=cfg.local_noise_high_cut_hz,
        )

        latent_source[tr] = z.astype(np.float32)
        latent_target[tr] = target_latent.astype(np.float32)
        source[tr] = (
            z[:, None] * source_loading[None, :]
            + float(cfg.source_sensor_noise_sd) * rng.standard_normal((T, C))
        ).astype(np.float32)
        target[tr] = (
            (target_latent + eta_target)[:, None] * target_loading[None, :]
            + float(cfg.target_sensor_noise_sd) * rng.standard_normal((T, C))
        ).astype(np.float32)

    splits = _make_four_way_split(cfg.n_trials, seed + 99173)
    y = np.concatenate([source, target], axis=-1)

    truth = {
        "world": world,
        "window_centers_s": centers,
        "stable_recruitment": stable_r,
        "stable_delays_ms": np.asarray(cfg.stable_delays_ms, float),
        "stable_poles_hz": np.asarray(cfg.stable_poles_hz, float),
        "stable_gains": np.asarray(cfg.stable_gains, float),
        "dynamic_delay_ms_wm": dynamic_tau_wm,
        "dynamic_poles_hz": np.asarray(cfg.dynamic_poles_hz, float),
        "dynamic_gains": np.asarray(cfg.dynamic_gains, float),
    }

    return {
        "protocol": PROTOCOL,
        "world": world,
        "seed": int(seed),
        "config": asdict(cfg),
        "source": source,
        "target": target,
        "y": y,
        "y_dims": np.asarray(cfg.y_dims, int),
        "source_loading": source_loading,
        "target_loading": target_loading,
        "latent_source": latent_source,
        "latent_target": latent_target,
        "splits": splits,
        "truth": truth,
    }


def save_s1_dataset(dataset: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    truth = dataset["truth"]
    np.savez_compressed(
        path,
        protocol=np.asarray(dataset["protocol"]),
        world=np.asarray(dataset["world"]),
        seed=np.asarray(dataset["seed"]),
        config_json=np.asarray(json.dumps(dataset["config"], sort_keys=True)),
        source=dataset["source"],
        target=dataset["target"],
        y=dataset["y"],
        y_dims=dataset["y_dims"],
        source_loading=dataset["source_loading"],
        target_loading=dataset["target_loading"],
        latent_source=dataset["latent_source"],
        latent_target=dataset["latent_target"],
        split_train=dataset["splits"]["train"],
        split_val_select=dataset["splits"]["val_select"],
        split_val_gate=dataset["splits"]["val_gate"],
        split_test=dataset["splits"]["test"],
        truth_window_centers_s=truth["window_centers_s"],
        truth_stable_recruitment=truth["stable_recruitment"],
        truth_stable_delays_ms=truth["stable_delays_ms"],
        truth_stable_poles_hz=truth["stable_poles_hz"],
        truth_stable_gains=truth["stable_gains"],
        truth_dynamic_delay_ms_wm=truth["dynamic_delay_ms_wm"],
        truth_dynamic_poles_hz=truth["dynamic_poles_hz"],
        truth_dynamic_gains=truth["dynamic_gains"],
    )
    return path


def load_s1_dataset(path: str | Path) -> dict[str, Any]:
    z = np.load(Path(path), allow_pickle=False)
    protocol = str(z["protocol"])
    if protocol != PROTOCOL:
        raise RuntimeError(
            f"Dataset protocol {protocol!r} != expected {PROTOCOL!r}. "
            "Delete old benchmark outputs or use --regenerate."
        )
    return {
        "protocol": protocol,
        "world": str(z["world"]),
        "seed": int(z["seed"]),
        "config": json.loads(str(z["config_json"])),
        "source": z["source"],
        "target": z["target"],
        "y": z["y"],
        "y_dims": z["y_dims"].astype(int),
        "source_loading": z["source_loading"],
        "target_loading": z["target_loading"],
        "latent_source": z["latent_source"],
        "latent_target": z["latent_target"],
        "splits": {
            "train": z["split_train"].astype(int),
            "val_select": z["split_val_select"].astype(int),
            "val_gate": z["split_val_gate"].astype(int),
            "test": z["split_test"].astype(int),
        },
        "truth": {
            "window_centers_s": z["truth_window_centers_s"],
            "stable_recruitment": z["truth_stable_recruitment"],
            "stable_delays_ms": z["truth_stable_delays_ms"],
            "stable_poles_hz": z["truth_stable_poles_hz"],
            "stable_gains": z["truth_stable_gains"],
            "dynamic_delay_ms_wm": z["truth_dynamic_delay_ms_wm"],
            "dynamic_poles_hz": z["truth_dynamic_poles_hz"],
            "dynamic_gains": z["truth_dynamic_gains"],
        },
    }


def build_ravel_session(dataset: dict[str, Any], *, session_id: str, subject_id: str):
    from ravel import build_route_session_from_epochs, paper_config

    metadata = pd.DataFrame(
        {
            "outcome": ["synthetic"] * len(dataset["source"]),
            "trial_index": np.arange(len(dataset["source"]), dtype=int),
        }
    )
    return build_route_session_from_epochs(
        source_epochs={"active": dataset["source"]},
        target_epochs={"active": dataset["target"]},
        metadata={"active": metadata},
        route_name="synthetic_A_to_B",
        source_region="A",
        target_region="B",
        session_id=session_id,
        subject_id=subject_id,
        d=1,
        experience=dataset["world"],
        reference_context="active",
        splits={"active": dataset["splits"]},
        input_is_preconditioned=False,
        config=paper_config(),
    )


def _phase_slope_delay_ms(response_wf: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
    x = np.asarray(response_wf)
    if x.ndim == 1:
        x = x[None, :]
    f = np.asarray(freqs_hz, float)
    keep = (f >= 6.0) & (f <= 40.0)
    A = np.column_stack([f[keep], np.ones(np.sum(keep))])
    out = np.full(x.shape[0], np.nan, dtype=float)
    for w in range(x.shape[0]):
        phase = np.unwrap(np.angle(x[w, keep]))
        if not np.all(np.isfinite(phase)):
            continue
        slope, _ = np.linalg.lstsq(A, phase, rcond=None)[0]
        out[w] = -1000.0 * float(slope) / (2.0 * np.pi)
    return out


def truth_transfer_on_ravel_grid(dataset: dict[str, Any], freqs_hz: np.ndarray) -> np.ndarray:
    f = np.asarray(freqs_hz, float)
    truth = dataset["truth"]
    W = len(truth["window_centers_s"])
    total = np.zeros((W, len(f)), dtype=np.complex128)
    if dataset["world"] == WORLD_STABLE:
        R = np.asarray(truth["stable_recruitment"], float)
        for m in range(2):
            tau = float(truth["stable_delays_ms"][m])
            pole = float(truth["stable_poles_hz"][m])
            gain = float(truth["stable_gains"][m])
            h = np.exp(-1j * 2.0 * np.pi * f * tau / 1000.0) / (1.0 + 1j * f / pole)
            h /= max(float(np.sqrt(np.mean(np.abs(h) ** 2))), 1e-12)
            total += gain * R[:, m, None] * h[None, :]
        return total

    tau_wm = np.asarray(truth["dynamic_delay_ms_wm"], float)
    for m in range(2):
        pole = float(truth["dynamic_poles_hz"][m])
        gain = float(truth["dynamic_gains"][m])
        for w in range(W):
            tau = float(tau_wm[w, m])
            h = np.exp(-1j * 2.0 * np.pi * f * tau / 1000.0) / (1.0 + 1j * f / pole)
            h /= max(float(np.sqrt(np.mean(np.abs(h) ** 2))), 1e-12)
            total[w] += gain * h
    return total


def apparent_truth_delay_ms(dataset: dict[str, Any], freqs_hz: np.ndarray) -> np.ndarray:
    return _phase_slope_delay_ms(truth_transfer_on_ravel_grid(dataset, freqs_hz), freqs_hz)


def ravel_motif_transfer_scalar(model, freqs_hz: np.ndarray, nuisance_degree: int = 2) -> np.ndarray:
    from ravel.model import atom_response

    W = model.recruitment.shape[0]
    H = np.zeros((W, len(freqs_hz)), dtype=np.complex128)
    atom_start = 1 + int(nuisance_degree)
    for m, atom in enumerate(model.atoms):
        h = atom_response(atom, freqs_hz)
        a = float(model.coefficients[atom_start + m, 0, 0])
        H += model.recruitment[:, m, None] * a * h[None, :]
    return H


def ravel_apparent_delay_ms(model, freqs_hz: np.ndarray, nuisance_degree: int = 2) -> np.ndarray:
    return _phase_slope_delay_ms(
        ravel_motif_transfer_scalar(model, freqs_hz, nuisance_degree), freqs_hz
    )


def matched_stable_truth_to_fit(dataset: dict[str, Any], model, freqs_hz: np.ndarray) -> pd.DataFrame:
    from scipy.optimize import linear_sum_assignment
    from ravel import AtomSpec
    from ravel.model import atom_shape_similarity

    truth_atoms = [
        AtomSpec("pole1", float(t), float(p))
        for t, p in zip(
            dataset["truth"]["stable_delays_ms"], dataset["truth"]["stable_poles_hz"]
        )
    ]
    similarity = np.array(
        [[atom_shape_similarity(t, c, freqs_hz) for c in model.atoms] for t in truth_atoms],
        dtype=float,
    )
    rows, cols = linear_sum_assignment(1.0 - similarity)
    out = []
    for i, j in zip(rows, cols):
        out.append(
            {
                "truth_index": int(i),
                "fit_index": int(j),
                "truth_delay_ms": float(truth_atoms[i].delay_ms),
                "fit_delay_ms": float(model.atoms[j].delay_ms),
                "delay_error_ms": float(model.atoms[j].delay_ms - truth_atoms[i].delay_ms),
                "truth_pole_hz": float(truth_atoms[i].pole_hz),
                "fit_pole_hz": float(model.atoms[j].pole_hz),
                "pole_error_hz": float(model.atoms[j].pole_hz - truth_atoms[i].pole_hz),
                "shape_similarity": float(similarity[i, j]),
            }
        )
    return pd.DataFrame(out)


def stable_recruitment_correlations(
    dataset: dict[str, Any], model, matching: pd.DataFrame
) -> pd.DataFrame:
    truth_r = np.asarray(dataset["truth"]["stable_recruitment"], float)
    rows = []
    for rec in matching.to_dict("records"):
        i = int(rec["truth_index"])
        j = int(rec["fit_index"])
        a = truth_r[:, i]
        b = np.asarray(model.recruitment[:, j], float)
        corr = pearson_or_nan(a, b)
        rows.append(
            {
                "truth_index": i,
                "fit_index": j,
                "recruitment_corr": corr,
                "truth_recruitment": a.tolist(),
                "fit_recruitment": b.tolist(),
            }
        )
    return pd.DataFrame(rows)


def dynamic_truth_vs_fixed_atoms(dataset: dict[str, Any], model) -> pd.DataFrame:
    """How well can each truth trajectory be represented by one fixed RAVEL atom delay?"""
    from scipy.optimize import linear_sum_assignment

    truth = np.asarray(dataset["truth"]["dynamic_delay_ms_wm"], float)
    fitted = np.asarray([float(a.delay_ms) for a in model.atoms], float)
    cost = np.empty((truth.shape[1], len(fitted)), float)
    for m in range(truth.shape[1]):
        for j, tau in enumerate(fitted):
            cost[m, j] = rmse(truth[:, m], np.full(truth.shape[0], tau))
    rows, cols = linear_sum_assignment(cost)
    return pd.DataFrame(
        [
            {
                "truth_index": int(i),
                "fit_index": int(j),
                "truth_start_ms": float(truth[0, i]),
                "truth_end_ms": float(truth[-1, i]),
                "truth_mean_ms": float(np.mean(truth[:, i])),
                "fit_fixed_delay_ms": float(fitted[j]),
                "trajectory_rmse_to_fixed_ms": float(cost[i, j]),
            }
            for i, j in zip(rows, cols)
        ]
    )


def summarize_score(score: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "nll": float(np.mean(score["trial_nll"])),
        "nmse": float(np.mean(score["trial_nmse"])),
        "r2": float(np.mean(score["trial_r2"])),
        "n_test_trials": int(len(score["trial_nll"])),
    }


def pearson_or_nan(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size != b.size or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


# -----------------------------------------------------------------------------
# Matched alternative: two communication components with STATIC gains but
# genuinely DYNAMIC delay trajectories.  This is an experiment-only baseline
# on RAVEL's own source/target coordinates and common evaluator; it imports no
# MERGED code.
# -----------------------------------------------------------------------------


def _dp_delay_path(cost_wd: np.ndarray, tau_grid_ms: np.ndarray, lam: float) -> np.ndarray:
    cost = np.asarray(cost_wd, float)
    tau = np.asarray(tau_grid_ms, float)
    W, D = cost.shape
    scale = max(float(np.nanmedian(cost)), 1e-12)
    c = cost / scale
    dp = np.full((W, D), np.inf, dtype=float)
    back = np.zeros((W, D), dtype=int)
    dp[0] = c[0]
    for w in range(1, W):
        penalty = float(lam) * ((tau[:, None] - tau[None, :]) / 10.0) ** 2
        total = dp[w - 1][:, None] + penalty
        back[w] = np.argmin(total, axis=0)
        dp[w] = c[w] + np.min(total, axis=0)
    path = np.zeros(W, dtype=int)
    path[-1] = int(np.argmin(dp[-1]))
    for w in range(W - 2, -1, -1):
        path[w] = back[w + 1, path[w + 1]]
    return path


def _zero_phase_kernels(freqs_hz: np.ndarray, nuisance_degree: int) -> np.ndarray:
    from ravel.model import zero_phase_nuisance_basis

    nuisance = zero_phase_nuisance_basis(freqs_hz, degree=int(nuisance_degree))
    return np.column_stack([np.ones(len(freqs_hz), complex), nuisance])


def _dynamic_response_matrix(
    tau_ms_w: np.ndarray,
    pole_hz: float,
    freqs_hz: np.ndarray,
) -> np.ndarray:
    from ravel import AtomSpec
    from ravel.model import atom_response

    return np.stack(
        [
            atom_response(AtomSpec("pole1", float(tau), float(pole_hz)), freqs_hz)
            for tau in np.asarray(tau_ms_w, float)
        ],
        axis=0,
    )


def _matched_dynamic_kernels(
    tau_wm: np.ndarray,
    pole_hz: float,
    freqs_hz: np.ndarray,
    nuisance_degree: int,
) -> np.ndarray:
    W = tau_wm.shape[0]
    zero = _zero_phase_kernels(freqs_hz, nuisance_degree)
    zero = np.repeat(zero[None, :, :], W, axis=0)
    dyn = np.stack(
        [_dynamic_response_matrix(tau_wm[:, m], pole_hz, freqs_hz) for m in range(2)],
        axis=-1,
    )
    return np.concatenate([zero, dyn], axis=-1)


def _delay_cost_for_static_gain(
    source: np.ndarray,
    residual: np.ndarray,
    responses_df: np.ndarray,
    gain: float,
    other_tau_w: np.ndarray,
    tau_grid_ms: np.ndarray,
    min_separation_ms: float,
) -> np.ndarray:
    W = source.shape[1]
    D = responses_df.shape[0]
    cost = np.empty((W, D), dtype=float)
    for w in range(W):
        xw = source[:, w, :, :, 0]
        yw = residual[:, w, :, :, 0]
        for d_idx, h in enumerate(responses_df):
            z = xw * h[None, None, :]
            value = float(np.mean(np.abs(yw - float(gain) * z) ** 2))
            if abs(float(tau_grid_ms[d_idx]) - float(other_tau_w[w])) < float(min_separation_ms):
                value += 1e3 * max(value, 1e-6)
            cost[w, d_idx] = value
    return cost


def _fit_two_dynamic_candidate(
    session,
    *,
    context: str,
    pole_hz: float,
    delay_smoothness: float,
    tau_grid_ms: np.ndarray,
    initial_tau_ms: tuple[float, float],
    alternations: int,
    minimum_separation_ms: float,
):
    from ravel import AtomSpec, paper_config
    from ravel.model import coordinate_prediction, fit_coefficients, transfer_from_coefficients
    from ravel.types import SessionFit

    cfg = paper_config()
    train_pos = session.positions(context, "train")
    source = session.contexts[context].source_coord[train_pos]
    target = session.contexts[context].target_coord[train_pos]
    W = source.shape[1]
    tau_wm = np.tile(np.asarray(initial_tau_ms, float)[None, :], (W, 1))

    responses_grid = np.stack(
        [
            _dynamic_response_matrix(
                np.full(W, float(tau)), float(pole_hz), session.freqs_hz
            )[0]
            for tau in tau_grid_ms
        ],
        axis=0,
    )

    trace = []
    for iteration in range(int(alternations)):
        kernels = _matched_dynamic_kernels(
            tau_wm, pole_hz, session.freqs_hz, cfg.model.nuisance_degree
        )
        coefficients, prediction, mse = fit_coefficients(source, target, kernels, cfg)
        n_zero = 1 + int(cfg.model.nuisance_degree)
        gains = np.asarray(coefficients[n_zero:n_zero + 2, 0, 0], float)

        old = tau_wm.copy()
        transfer_all = transfer_from_coefficients(kernels, coefficients)
        full_pred = coordinate_prediction(source, transfer_all)

        for m in range(2):
            response_current = _dynamic_response_matrix(
                tau_wm[:, m], pole_hz, session.freqs_hz
            )
            this_piece = (
                source
                * response_current[None, :, None, :, None]
                * float(gains[m])
            )
            residual_for_m = target - (full_pred - this_piece)
            cost = _delay_cost_for_static_gain(
                source,
                residual_for_m,
                responses_grid,
                float(gains[m]),
                tau_wm[:, 1 - m],
                tau_grid_ms,
                minimum_separation_ms,
            )
            path = _dp_delay_path(cost, tau_grid_ms, float(delay_smoothness))
            tau_wm[:, m] = tau_grid_ms[path]

        # Components are exchangeable because this matched baseline uses a
        # common pole. Keep a deterministic low-delay/high-delay identity.
        swap = tau_wm[:, 0] > tau_wm[:, 1]
        if np.any(swap):
            a = tau_wm[swap, 0].copy()
            tau_wm[swap, 0] = tau_wm[swap, 1]
            tau_wm[swap, 1] = a

        trace.append(
            {
                "iteration": int(iteration),
                "train_mse": float(mse),
                "mean_tau0_ms": float(np.mean(tau_wm[:, 0])),
                "mean_tau1_ms": float(np.mean(tau_wm[:, 1])),
            }
        )
        if np.array_equal(old, tau_wm) and iteration >= 1:
            break

    kernels = _matched_dynamic_kernels(
        tau_wm, pole_hz, session.freqs_hz, cfg.model.nuisance_degree
    )
    coefficients, _, mse = fit_coefficients(source, target, kernels, cfg)
    transfer = transfer_from_coefficients(kernels, coefficients)
    n_zero = 1 + int(cfg.model.nuisance_degree)
    gains = np.asarray(coefficients[n_zero:n_zero + 2, 0, 0], float)

    model = SessionFit(
        route_name=session.route_name,
        session_id=session.session_id,
        context=context,
        atoms=[
            AtomSpec("pole1", float(np.median(tau_wm[:, 0])), float(pole_hz)),
            AtomSpec("pole1", float(np.median(tau_wm[:, 1])), float(pole_hz)),
        ],
        coefficients=coefficients,
        recruitment=np.ones((W, 2), dtype=float),
        transfer=transfer,
        train_mse=float(mse),
        kernel_condition=np.nan,
        fit_positions=train_pos.copy(),
    )
    return {
        "model": model,
        "tau_ms_wm": tau_wm.copy(),
        "static_gain_m": gains.copy(),
        "train_mse": float(mse),
        "trace": trace,
    }


def fit_matched_dynamic_delay_baseline(
    session,
    *,
    context: str = "active",
    pole_grid_hz: Iterable[float] = (30.0, 45.0, 60.0),
    smoothness_grid: Iterable[float] = (0.05, 0.20, 0.80),
    tau_grid_ms: np.ndarray | None = None,
    initial_tau_ms: tuple[float, float] = (10.0, 23.0),
    alternations: int = 4,
    minimum_separation_ms: float = 2.0,
):
    """Matched two-component alternative with static gains + dynamic delays.

    Model family:
        H[w,f] = H_zero_phase[f]
                 + a1 h(f; tau1[w], p)
                 + a2 h(f; tau2[w], p)

    ``a1,a2`` are static across event time; only the two delay paths vary. This
    deliberately contrasts RAVEL's fixed delays + dynamic recruitment while
    keeping the known communication order equal to two.
    """
    from ravel import paper_config
    from ravel.metrics import score_session_model

    cfg = paper_config()
    if session.d != 1:
        raise ValueError("Matched dynamic-delay baseline assumes d=1")
    if tau_grid_ms is None:
        tau_grid_ms = np.arange(
            cfg.optimization.delay_min_ms,
            cfg.optimization.delay_max_ms + 1e-9,
            1.0,
        )
    tau_grid_ms = np.asarray(tau_grid_ms, float)

    candidates = []
    for pole in pole_grid_hz:
        for lam in smoothness_grid:
            fit = _fit_two_dynamic_candidate(
                session,
                context=context,
                pole_hz=float(pole),
                delay_smoothness=float(lam),
                tau_grid_ms=tau_grid_ms,
                initial_tau_ms=initial_tau_ms,
                alternations=int(alternations),
                minimum_separation_ms=float(minimum_separation_ms),
            )
            val = score_session_model(
                session, fit["model"], context, "val_select", config=cfg
            )
            candidates.append(
                {
                    **fit,
                    "pole_hz": float(pole),
                    "smoothness": float(lam),
                    "val_select_nll": float(np.mean(val["trial_nll"])),
                    "val_select_nmse": float(np.mean(val["trial_nmse"])),
                }
            )

    best = min(candidates, key=lambda item: item["val_select_nll"])
    gate = score_session_model(session, best["model"], context, "val_gate", config=cfg)
    test = score_session_model(session, best["model"], context, "test", config=cfg)
    return {
        "model": best["model"],
        "pole_hz": float(best["pole_hz"]),
        "smoothness": float(best["smoothness"]),
        "tau_ms_wm": np.asarray(best["tau_ms_wm"], float),
        "static_gain_m": np.asarray(best["static_gain_m"], float),
        "train_mse": float(best["train_mse"]),
        "val_select_nll": float(best["val_select_nll"]),
        "val_select_nmse": float(best["val_select_nmse"]),
        "val_gate": summarize_score(gate),
        "test": summarize_score(test),
        "candidate_table": pd.DataFrame(
            [
                {
                    "pole_hz": float(c["pole_hz"]),
                    "smoothness": float(c["smoothness"]),
                    "train_mse": float(c["train_mse"]),
                    "val_select_nll": float(c["val_select_nll"]),
                    "val_select_nmse": float(c["val_select_nmse"]),
                    "mean_tau0_ms": float(np.mean(c["tau_ms_wm"][:, 0])),
                    "mean_tau1_ms": float(np.mean(c["tau_ms_wm"][:, 1])),
                }
                for c in candidates
            ]
        ),
    }


def match_dynamic_trajectories(
    fitted_wk: np.ndarray, truth_wm: np.ndarray
) -> pd.DataFrame:
    """Hungarian match fitted dynamic-delay trajectories to the two truths."""
    from scipy.optimize import linear_sum_assignment

    fitted = np.asarray(fitted_wk, float)
    truth = np.asarray(truth_wm, float)
    if fitted.ndim == 1:
        fitted = fitted[:, None]
    cost = np.empty((truth.shape[1], fitted.shape[1]), float)
    for i in range(truth.shape[1]):
        for j in range(fitted.shape[1]):
            cost[i, j] = rmse(truth[:, i], fitted[:, j])
    rows, cols = linear_sum_assignment(cost)
    return pd.DataFrame(
        [
            {
                "truth_index": int(i),
                "fit_index": int(j),
                "trajectory_rmse_ms": float(cost[i, j]),
                "trajectory_corr": pearson_or_nan(truth[:, i], fitted[:, j]),
                "truth_start_ms": float(truth[0, i]),
                "truth_end_ms": float(truth[-1, i]),
                "fit_start_ms": float(fitted[0, j]),
                "fit_end_ms": float(fitted[-1, j]),
            }
            for i, j in zip(rows, cols)
        ]
    )
