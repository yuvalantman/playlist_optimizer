"""Build sparse Top-M transition graphs from pairwise transition costs."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransitionGraphConfig:
    """Configuration for sparse transition graph construction."""

    top_m_neighbors: int = 30
    exclude_self_transitions: bool = True


class TransitionGraphBuilder:
    """Build a lowest-cost neighbor graph for each track."""

    def __init__(self, config: TransitionGraphConfig) -> None:
        self.config = config
        if self.config.top_m_neighbors <= 0:
            raise ValueError("top_m_neighbors must be positive.")

    def _validate_c_trans(self, c_trans: np.ndarray) -> None:
        if not isinstance(c_trans, np.ndarray) or c_trans.ndim != 2:
            raise ValueError("c_trans must be a 2D numpy array.")
        if c_trans.shape[0] == 0 or c_trans.shape[0] != c_trans.shape[1]:
            raise ValueError("c_trans must be a non-empty square matrix.")
        if not np.issubdtype(c_trans.dtype, np.number):
            raise ValueError("c_trans must contain numeric values.")
        if not np.isfinite(c_trans).all():
            raise ValueError("c_trans must contain only finite values.")

        number_of_tracks = c_trans.shape[0]
        possible_neighbors = number_of_tracks - int(
            self.config.exclude_self_transitions
        )
        if self.config.top_m_neighbors > possible_neighbors:
            raise ValueError(
                "top_m_neighbors cannot exceed the number of valid possible "
                f"neighbors ({possible_neighbors})."
            )

    def _sorted_neighbor_indices(self, c_trans: np.ndarray, track_index: int) -> np.ndarray:
        ordered_indices = np.argsort(c_trans[track_index], kind="stable")
        if self.config.exclude_self_transitions:
            ordered_indices = ordered_indices[ordered_indices != track_index]
        return ordered_indices[: self.config.top_m_neighbors]

    def build_top_m_graph(self, c_trans: np.ndarray) -> dict[int, list[int]]:
        """Return each track's Top-M lowest-cost transition neighbors."""
        self._validate_c_trans(c_trans)
        return {
            track_index: self._sorted_neighbor_indices(
                c_trans,
                track_index,
            ).tolist()
            for track_index in range(c_trans.shape[0])
        }

    def build_top_m_graph_with_costs(
        self,
        c_trans: np.ndarray,
    ) -> dict[int, list[tuple[int, float]]]:
        """Return sorted Top-M neighbors and their transition costs."""
        self._validate_c_trans(c_trans)
        graph: dict[int, list[tuple[int, float]]] = {}

        for track_index in range(c_trans.shape[0]):
            neighbor_indices = self._sorted_neighbor_indices(c_trans, track_index)
            graph[track_index] = [
                (int(neighbor_index), float(c_trans[track_index, neighbor_index]))
                for neighbor_index in neighbor_indices
            ]

        return graph
