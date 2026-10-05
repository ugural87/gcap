"""Deterministic seeding (Section 8's reproducibility discipline, extended
to the harness): every stochastic draw the harness itself controls (which
today means only the pseudo-random ``provider_seed`` recorded on the
RunRecord — the fake adapter's own draws are separately seeded per-prompt
in agents/llm.py's ``_seeded_bool``, and a future real adapter would accept
a ``seed`` field the same way) is derived purely from
``(manifest_hash, case_id, replicate, draw_id)``. Never from wall-clock,
never from a mutable global RNG — re-running the identical
(manifest, case, replicate[, draw]) tuple always yields the identical seed
(invariant 7: every run reproducible from its manifest).
"""

from __future__ import annotations

import hashlib
import random

# Masked to 63 bits so the derived seed fits a signed 64-bit database
# column on every backend (sqlite/Postgres), regardless of how the full
# 256-bit digest's leading byte happens to fall.
_SEED_MASK = 0x7FFFFFFFFFFFFFFF


def derive_seed(manifest_hash: str, case_id: str, replicate: int, draw_id: str | None = None) -> int:
    payload = f"{manifest_hash}:{case_id}:{replicate}:{draw_id or ''}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & _SEED_MASK


def rng_for(manifest_hash: str, case_id: str, replicate: int, draw_id: str | None = None) -> random.Random:
    return random.Random(derive_seed(manifest_hash, case_id, replicate, draw_id))
