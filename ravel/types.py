"""Typed containers shared by preprocessing, fitting, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RegionTransform:
    median: np.ndarray
    scale: np.ndarray
    event_template: np.ndarray


@dataclass
class ContextWindowed:
    source_coord: np.ndarray
    target_coord: np.ndarray
    target_raw: np.ndarray
    metadata: pd.DataFrame
    splits: dict[str, np.ndarray]


@dataclass
class RouteSession:
    route_name: str
    source_region: str
    target_region: str
    session_id: int | str
    subject_id: str
    experience: str
    d: int
    freqs_hz: np.ndarray
    window_centers_s: np.ndarray
    source_axes: np.ndarray
    target_axes: np.ndarray
    source_scale: np.ndarray
    target_scale: np.ndarray
    contexts: dict[str, ContextWindowed]
    representation_id: str = "full_active_train"
    evaluation_positions: dict[str, np.ndarray] | None = None

    @property
    def mouse_id(self) -> str:
        """Compatibility alias used by this version's notebook."""

        return self.subject_id

    def positions(self, context: str, split_name: str) -> np.ndarray:
        return np.asarray(self.contexts[context].splits[split_name], dtype=int)


@dataclass(frozen=True)
class AtomSpec:
    kind: str
    delay_ms: float
    pole_hz: float = np.nan

    def label(self) -> str:
        if self.kind == "delay":
            return f"delay tau={self.delay_ms:.2f}ms"
        return f"pole1 tau={self.delay_ms:.2f}ms fc={self.pole_hz:.1f}Hz"


@dataclass
class SessionFit:
    route_name: str
    session_id: int | str
    context: str
    atoms: list[AtomSpec]
    coefficients: np.ndarray
    recruitment: np.ndarray
    transfer: np.ndarray
    train_mse: float
    kernel_condition: float
    fit_positions: np.ndarray


@dataclass
class SharedRouteFit:
    route_name: str
    context: str
    atoms: list[AtomSpec]
    models: dict[Any, SessionFit]
    selected_strategy: str
    strategy_table: pd.DataFrame
    restart_details: pd.DataFrame
    restart_summary: pd.DataFrame


@dataclass
class TaskTransferFit:
    frozen_physics_refit_subject: dict[Any, SessionFit]
    strict_recruitment_only: dict[Any, SessionFit]
    frozen_physics_static: dict[Any, SessionFit]
    context_scratch: SharedRouteFit | None

