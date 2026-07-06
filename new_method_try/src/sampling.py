"""Seeded Top-K / softmax sampling for stochastic random-walk beam search.

Used to make beam survival and anchor selection probabilistic (Chapter 2's
"Context-Aware Random Walk + Top-K sampling") so each seeded run produces a different,
valid ordering while still favoring low-cost choices. All randomness flows through a
caller-supplied ``numpy`` Generator so runs are reproducible per seed.
"""

import numpy as np

# Costs below this spread are treated as effectively tied (uniform among the top-k).
_MIN_COST_SPREAD = 1e-12


def softmax_topk_choice(
    scores: np.ndarray,
    count: int,
    rng: np.random.Generator,
    top_k: int,
    temperature: float,
) -> list[int]:
    """Choose ``count`` distinct indices, lower score preferred, without replacement.

    Candidates are restricted to the ``top_k`` lowest scores, then sampled with
    probability proportional to ``exp(-(score - min)/temperature)``. With
    ``temperature -> 0`` this approaches deterministic top-k selection; with large
    ``temperature`` it approaches uniform sampling among the top-k.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if count <= 0 or n == 0:
        return []
    count = min(count, n)
    if temperature <= 0:
        # Deterministic: the lowest-score indices, ties broken by index.
        return sorted(range(n), key=lambda i: (scores[i], i))[:count]

    order = np.argsort(scores, kind="stable")
    pool = list(order[: max(top_k, count)])
    chosen: list[int] = []
    remaining = pool.copy()
    while remaining and len(chosen) < count:
        local_scores = scores[remaining]
        shifted = local_scores - local_scores.min()
        spread = float(shifted.max())
        if spread < _MIN_COST_SPREAD:
            weights = np.ones(len(remaining), dtype=float)
        else:
            weights = np.exp(-shifted / temperature)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            weights = np.ones(len(remaining), dtype=float)
            total = weights.sum()
        probabilities = weights / total
        pick_position = int(rng.choice(len(remaining), p=probabilities))
        chosen.append(int(remaining.pop(pick_position)))
    return chosen
