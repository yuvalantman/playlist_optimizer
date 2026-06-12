"""Greedy baseline for constructing a complete playlist."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GreedyBaselineConfig:
    """Weights and graph behavior for greedy playlist construction."""

    arc_weight: float = 1.0
    transition_weight: float = 1.0
    use_transition_graph: bool = True


class GreedyPlaylistBaseline:
    """Construct a playlist by repeatedly choosing the lowest greedy cost."""

    def __init__(self, config: GreedyBaselineConfig) -> None:
        self.config = config

    @staticmethod
    def _validate_inputs(
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        start_index: int | None,
        end_index: int | None,
    ) -> int:
        if not isinstance(c_arc, np.ndarray) or c_arc.ndim != 2:
            raise ValueError("c_arc must be a 2D numpy array.")
        if not isinstance(c_trans, np.ndarray) or c_trans.ndim != 2:
            raise ValueError("c_trans must be a square 2D numpy array.")
        if c_trans.shape[0] == 0 or c_trans.shape[0] != c_trans.shape[1]:
            raise ValueError("c_trans must be a non-empty square matrix.")
        if c_arc.shape[0] != c_trans.shape[0]:
            raise ValueError("c_arc and c_trans must have the same number of tracks.")
        if c_arc.shape[1] != c_arc.shape[0]:
            raise ValueError(
                "c_arc must have one playlist position for every track."
            )
        if not np.issubdtype(c_arc.dtype, np.number) or not np.isfinite(c_arc).all():
            raise ValueError("c_arc must contain only finite numeric values.")
        if not np.issubdtype(c_trans.dtype, np.number) or not np.isfinite(
            c_trans
        ).all():
            raise ValueError("c_trans must contain only finite numeric values.")

        number_of_tracks = c_trans.shape[0]
        for name, index in (("start_index", start_index), ("end_index", end_index)):
            if index is not None and (
                not isinstance(index, (int, np.integer))
                or not 0 <= index < number_of_tracks
            ):
                raise ValueError(f"{name} must be a valid track index.")
        if start_index is not None and start_index == end_index:
            raise ValueError("start_index and end_index cannot be the same.")

        return number_of_tracks

    def _choose_track(
        self,
        available_tracks: set[int],
        position: int,
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        previous_track: int | None,
        transition_graph: dict[int, list[int]] | None,
    ) -> int:
        candidates = available_tracks
        if (
            previous_track is not None
            and self.config.use_transition_graph
            and transition_graph is not None
        ):
            graph_candidates = available_tracks.intersection(
                transition_graph.get(previous_track, [])
            )
            if graph_candidates:
                candidates = graph_candidates

        def greedy_score(track_index: int) -> tuple[float, int]:
            score = self.config.arc_weight * c_arc[track_index, position]
            if previous_track is not None:
                score += (
                    self.config.transition_weight
                    * c_trans[previous_track, track_index]
                )
            return float(score), track_index

        return min(candidates, key=greedy_score)

    def generate(
        self,
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        start_index: int | None = None,
        end_index: int | None = None,
        transition_graph: dict[int, list[int]] | None = None,
    ) -> list[int]:
        """Return a greedy playlist containing every track exactly once."""
        number_of_tracks = self._validate_inputs(
            c_arc,
            c_trans,
            start_index,
            end_index,
        )
        available_tracks = set(range(number_of_tracks))
        if end_index is not None:
            available_tracks.remove(end_index)

        playlist: list[int] = []
        if start_index is not None:
            playlist.append(start_index)
            available_tracks.remove(start_index)

        positions_before_end = number_of_tracks - int(end_index is not None)
        while len(playlist) < positions_before_end:
            previous_track = playlist[-1] if playlist else None
            next_track = self._choose_track(
                available_tracks,
                len(playlist),
                c_arc,
                c_trans,
                previous_track,
                transition_graph,
            )
            playlist.append(next_track)
            available_tracks.remove(next_track)

        if end_index is not None:
            playlist.append(end_index)

        return playlist
