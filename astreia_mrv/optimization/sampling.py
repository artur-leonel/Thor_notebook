from __future__ import annotations

from astreia_mrv.parameters import LIFTING_DEFAULT_RX, latin_hypercube


def sample_lifting_body_rx(n: int, seed: int | None = 7) -> list[dict[str, float]]:
    keys = list(LIFTING_DEFAULT_RX.keys())
    return latin_hypercube(n, keys, seed=seed)
