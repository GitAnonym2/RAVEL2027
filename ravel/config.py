"""Configuration objects for the frozen RAVEL implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SpectralConfig:
    """Paper spectral preprocessing preset.

    Dataset adapters may select
    events and channels, but paper comparisons should not change these values.
    """

    fs_hz: float = 500.0
    epoch_pre_s: float = 1.0
    epoch_post_s: float = 2.0
    lowpass_hz: float = 100.0
    lowpass_order: int = 4
    notch_hz: float | None = 60.0
    notch_q: float = 30.0
    fmin_hz: float = 6.0
    fmax_hz: float = 80.0
    line_guard_low_hz: float = 58.0
    line_guard_high_hz: float = 62.0
    window_seconds: float = 0.40
    window_step_seconds: float = 0.25
    multitaper_nw: float = 2.5
    multitaper_k: int = 3
    axis_lags_ms: tuple[float, ...] = (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0)

    @property
    def epoch_seconds(self) -> float:
        return self.epoch_pre_s + self.epoch_post_s


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.60
    val_select_fraction: float = 0.10
    val_gate_fraction: float = 0.10
    test_fraction: float = 0.20
    minimum_validation_trials: int = 3

    def validate(self) -> None:
        total = (
            self.train_fraction
            + self.val_select_fraction
            + self.val_gate_fraction
            + self.test_fraction
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split fractions must sum to 1, got {total}")


@dataclass(frozen=True)
class ModelConfig:
    nuisance_degree: int = 2
    nuisance_residue_rank: int = 1
    motif_residue_rank: int = 1
    ridge: float = 2e-3
    residual_shrinkage: float = 0.25
    coefficient_sweeps: int = 4
    recruitment_alternations: int = 3
    recruitment_max: float = 3.0
    recruitment_ridge_relative: float = 0.04
    recruitment_smooth_relative: float = 0.20


@dataclass(frozen=True)
class OptimizationConfig:
    delay_min_ms: float = 2.0
    delay_max_ms: float = 45.0
    pole_min_hz: float = 5.0
    pole_max_hz: float = 75.0
    minimum_motif_separation_ms: float = 1.5
    restart_delay_jitter_ms: float = 3.0
    restart_log_pole_jitter: float = 0.15
    lock_restarts: int = 4
    pooled_restarts: int = 3
    independent_restarts: int = 3
    lock_delay_steps_ms: tuple[float, ...] = (4.0, 2.0, 1.0)
    pooled_delay_steps_ms: tuple[float, ...] = (2.0, 1.0)
    independent_delay_steps_ms: tuple[float, ...] = (6.0, 3.0, 1.0)
    minimum_restart_support: float = 0.75
    minimum_atom_shape_similarity: float = 0.90
    maximum_atom_delay_mad_ms: float = 3.0
    noninferiority_fraction: float = 0.10


@dataclass(frozen=True)
class RavelConfig:
    """Complete immutable configuration for a RAVEL analysis."""

    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    base_seed: int = 20260921
    version: str = "1.0.0"
    preset_name: str = "paper_reference"

    def validate(self) -> None:
        self.split.validate()
        if self.spectral.fmin_hz >= self.spectral.fmax_hz:
            raise ValueError("fmin_hz must be below fmax_hz")
        if self.optimization.delay_min_ms >= self.optimization.delay_max_ms:
            raise ValueError("delay bounds are invalid")
        if self.optimization.pole_min_hz >= self.optimization.pole_max_hz:
            raise ValueError("pole bounds are invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def paper_config() -> RavelConfig:
    """Return the frozen paper preset."""

    cfg = RavelConfig()
    cfg.validate()
    return cfg

