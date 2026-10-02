"""Small, explicit persistence helpers for long-running RAVEL analyses."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import json
import pickle

import numpy as np

from .config import RavelConfig
from .types import AtomSpec, SharedRouteFit


def save_pickle(value: Any, path: str | Path) -> Path:
    """Checkpoint a Python artifact using the highest pickle protocol."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)
    return destination


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def save_config(config: RavelConfig, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return destination


def save_physics_json(
    route_name: str,
    atoms: list[AtomSpec],
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write the portable, subject-agnostic portion of a fitted route."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "route_name": route_name,
        "motif_count": len(atoms),
        "atoms": [asdict(atom) for atom in atoms],
        "metadata": metadata or {},
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def load_physics_json(path: str | Path) -> tuple[str, list[AtomSpec], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    atoms = [AtomSpec(**item) for item in payload["atoms"]]
    return str(payload["route_name"]), atoms, dict(payload.get("metadata", {}))


def save_shared_fit(shared: SharedRouteFit, path: str | Path) -> Path:
    """Checkpoint the complete shared fit, including session-specific models."""

    return save_pickle(shared, path)

