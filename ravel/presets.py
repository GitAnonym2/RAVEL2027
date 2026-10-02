"""Load explicit development-only atom presets from JSON."""

from __future__ import annotations

from pathlib import Path
import json

from .types import AtomSpec


def load_initial_atom_presets(
    path: str | Path,
    route_name: str,
) -> dict[int, list[AtomSpec]]:
    """Return ``M -> atoms`` for one route from a handoff preset file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    route = payload.get(route_name, {})
    return {
        int(motif_count): [AtomSpec(**item) for item in atoms]
        for motif_count, atoms in route.items()
    }

