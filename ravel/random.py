"""Deterministic seed helpers."""

from __future__ import annotations

import hashlib


def stable_seed(base_seed: int, *parts: object) -> int:
    """Reproduce the notebook's platform-independent 32-bit seed derivation."""

    payload = "|".join(map(str, (int(base_seed), *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")

