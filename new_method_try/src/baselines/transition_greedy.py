"""Greedy baseline that minimizes only the transition cost (ignores the arc).

This isolates the transition "ceiling" for a greedy method. It must be compared to
other methods ONLY on transition-dependent metrics (transition cost, coherence,
worst-case transitions), never on arc_rmse, because it does not optimize the arc.
"""

import numpy as np

from src.transition_graph import TransitionGraphData


class TransitionGreedyBaseline:
    """Left-to-right greedy on C_trans only, with fixed endpoints."""

    def __init__(self, use_transition_graph: bool = True) -> None:
        self.use_transition_graph = use_transition_graph

    def generate(
        self,
        c_trans: np.ndarray,
        start_index: int,
        end_index: int,
        transition_graph: TransitionGraphData | dict[int, list[int]] | None = None,
    ) -> list[int]:
        """Return a playlist greedily minimizing each next transition cost."""
        number_of_tracks = c_trans.shape[0]
        if start_index == end_index:
            raise ValueError("start_index and end_index must differ.")
        available = set(range(number_of_tracks)) - {start_index, end_index}
        playlist = [start_index]

        while available:
            previous = playlist[-1]
            candidates: set[int] = available
            if self.use_transition_graph and transition_graph is not None:
                outgoing = (
                    transition_graph.outgoing_neighbors
                    if isinstance(transition_graph, TransitionGraphData)
                    else transition_graph
                )
                graph_candidates = available.intersection(outgoing.get(previous, []))
                if graph_candidates:
                    candidates = graph_candidates
            nxt = min(
                candidates,
                key=lambda track: (float(c_trans[previous, track]), track),
            )
            playlist.append(nxt)
            available.remove(nxt)

        playlist.append(end_index)
        return playlist
