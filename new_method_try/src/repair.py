"""Method-agnostic micro-level repair (windowed local reordering).

This is the post-hoc transition-smoothing pass the methodology describes but which the
original pipeline only ever prototyped inside main.py. It operates on a *finished*
playlist from any constructor (greedy, BIBS, MM beam, ...) and locally reorders small
windows around the worst transitions, accepting a reordering only when it:

1. reduces total transition cost, and
2. does not increase arc RMSE by more than ``arc_guard`` (so it never trades the arc
   away for smoothness), and
3. keeps the playlist exact-once with fixed endpoints (positions 0 and L-1 untouched).

Because every accepted move is guarded and exact-once is preserved by construction,
repair can only help (or no-op) on the metrics it targets.
"""

from dataclasses import dataclass
from itertools import permutations

import numpy as np


@dataclass(frozen=True)
class RepairConfig:
    """Configuration for the windowed local-reordering repair pass."""

    window_radius: int = 2
    max_windows: int = 20
    max_window_size: int = 6
    arc_guard: float = 0.02


def _total_transition_cost(playlist: list[int], c_trans: np.ndarray) -> float:
    indices = np.asarray(playlist, dtype=int)
    return float(c_trans[indices[:-1], indices[1:]].sum())


def _arc_rmse(playlist: list[int], c_arc: np.ndarray) -> float:
    positions = np.arange(len(playlist))
    costs = c_arc[np.asarray(playlist, dtype=int), positions]
    return float(np.sqrt(np.mean(np.square(costs))))


def repair_playlist(
    playlist: list[int],
    c_arc: np.ndarray,
    c_trans: np.ndarray,
    config: RepairConfig = RepairConfig(),
) -> tuple[list[int], list[dict]]:
    """Return a guarded, locally-repaired copy of the playlist plus a trace.

    The original playlist is never mutated; endpoints stay fixed; the result is always a
    valid exact-once permutation.
    """
    length = len(playlist)
    repaired = list(playlist)
    trace: list[dict] = []
    if length <= 3:
        return repaired, trace

    baseline_arc = _arc_rmse(repaired, c_arc)
    transition_costs = np.asarray(
        [
            float(c_trans[repaired[i], repaired[i + 1]])
            for i in range(length - 1)
        ]
    )
    worst_positions = np.argsort(-transition_costs, kind="stable")[
        : config.max_windows
    ].tolist()

    for transition_position in worst_positions:
        start = max(1, int(transition_position) - config.window_radius)
        end = min(length - 2, int(transition_position) + 1 + config.window_radius)
        window_size = end - start + 1
        if window_size < 2 or window_size > config.max_window_size:
            continue

        window_tracks = repaired[start : end + 1]
        before_total = _total_transition_cost(repaired, c_trans)
        best_playlist = repaired
        best_total = before_total
        for ordering in permutations(window_tracks):
            if ordering == tuple(window_tracks):
                continue
            candidate = list(repaired)
            candidate[start : end + 1] = list(ordering)
            candidate_total = _total_transition_cost(candidate, c_trans)
            if candidate_total < best_total:
                best_total = candidate_total
                best_playlist = candidate

        if best_playlist is repaired:
            continue
        candidate_arc = _arc_rmse(best_playlist, c_arc)
        accepted = candidate_arc <= baseline_arc + config.arc_guard
        if accepted:
            repaired = best_playlist
            baseline_arc = max(baseline_arc, candidate_arc)
        trace.append(
            {
                "window_start": start,
                "window_end": end,
                "window_size": window_size,
                "transition_total_before": before_total,
                "transition_total_after": best_total if accepted else before_total,
                "arc_rmse_after": candidate_arc,
                "accepted": accepted,
            }
        )

    if len(set(repaired)) != length:
        raise ValueError("Repair produced a non-exact-once playlist.")
    return repaired, trace
