"""Evaluation metrics for complete playlist sequences."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for playlist evaluation metrics."""

    tail_fraction: float = 0.20


class PlaylistEvaluator:
    """Validate and evaluate complete playlist sequences."""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        if not 0 < self.config.tail_fraction <= 1:
            raise ValueError("tail_fraction must be greater than 0 and at most 1.")

    @staticmethod
    def validate_playlist(playlist: list[int], number_of_tracks: int) -> None:
        """Validate that a playlist contains each valid track exactly once."""
        if number_of_tracks <= 0:
            raise ValueError("number_of_tracks must be positive.")
        if len(playlist) != number_of_tracks:
            raise ValueError("Playlist length must equal number_of_tracks.")
        if any(
            not isinstance(track_index, (int, np.integer))
            or not 0 <= track_index < number_of_tracks
            for track_index in playlist
        ):
            raise ValueError("Playlist contains an invalid track index.")
        if len(set(playlist)) != number_of_tracks:
            raise ValueError("Every track must appear exactly once in the playlist.")

    @staticmethod
    def _validate_arc_matrix(c_arc: np.ndarray) -> int:
        if not isinstance(c_arc, np.ndarray) or c_arc.ndim != 2:
            raise ValueError("c_arc must be a 2D numpy array.")
        if c_arc.shape[0] == 0 or c_arc.shape[0] != c_arc.shape[1]:
            raise ValueError("c_arc must have one playlist position per track.")
        if not np.issubdtype(c_arc.dtype, np.number) or not np.isfinite(c_arc).all():
            raise ValueError("c_arc must contain only finite numeric values.")
        return c_arc.shape[0]

    @staticmethod
    def _validate_transition_matrix(c_trans: np.ndarray) -> int:
        if not isinstance(c_trans, np.ndarray) or c_trans.ndim != 2:
            raise ValueError("c_trans must be a 2D numpy array.")
        if c_trans.shape[0] == 0 or c_trans.shape[0] != c_trans.shape[1]:
            raise ValueError("c_trans must be a non-empty square matrix.")
        if not np.issubdtype(c_trans.dtype, np.number) or not np.isfinite(
            c_trans
        ).all():
            raise ValueError("c_trans must contain only finite numeric values.")
        return c_trans.shape[0]

    def compute_arc_rmse(self, playlist: list[int], c_arc: np.ndarray) -> float:
        """Return RMSE of assigned track-position arc costs."""
        number_of_tracks = self._validate_arc_matrix(c_arc)
        self.validate_playlist(playlist, number_of_tracks)
        positions = np.arange(number_of_tracks)
        costs = c_arc[np.asarray(playlist, dtype=int), positions]
        return float(np.sqrt(np.mean(np.square(costs))))

    def compute_total_transition_cost(
        self,
        playlist: list[int],
        c_trans: np.ndarray,
    ) -> float:
        """Return the sum of costs between all consecutive playlist tracks."""
        number_of_tracks = self._validate_transition_matrix(c_trans)
        self.validate_playlist(playlist, number_of_tracks)
        if number_of_tracks == 1:
            return 0.0

        playlist_indices = np.asarray(playlist, dtype=int)
        return float(c_trans[playlist_indices[:-1], playlist_indices[1:]].sum())

    def compute_tail_transition_cost(
        self,
        playlist: list[int],
        c_trans: np.ndarray,
    ) -> float:
        """Return transition cost over the final configured fraction."""
        number_of_tracks = self._validate_transition_matrix(c_trans)
        self.validate_playlist(playlist, number_of_tracks)
        number_of_transitions = number_of_tracks - 1
        if number_of_transitions == 0:
            return 0.0

        tail_size = math.ceil(self.config.tail_fraction * number_of_transitions)
        playlist_indices = np.asarray(playlist, dtype=int)
        transition_costs = c_trans[
            playlist_indices[:-1],
            playlist_indices[1:],
        ]
        return float(transition_costs[-tail_size:].sum())

    def get_transition_costs_by_position(
        self,
        playlist: list[int],
        c_trans: np.ndarray,
    ) -> list[float]:
        """Return one transition cost for every consecutive playlist pair."""
        number_of_tracks = self._validate_transition_matrix(c_trans)
        self.validate_playlist(playlist, number_of_tracks)
        if number_of_tracks == 1:
            return []

        playlist_indices = np.asarray(playlist, dtype=int)
        return c_trans[
            playlist_indices[:-1],
            playlist_indices[1:],
        ].astype(float).tolist()

    def get_worst_transitions(
        self,
        playlist: list[int],
        c_trans: np.ndarray,
        top_k: int = 10,
    ) -> list[dict]:
        """Return the highest-cost consecutive transitions."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        transition_costs = self.get_transition_costs_by_position(
            playlist,
            c_trans,
        )
        transitions = [
            {
                "position": position,
                "from_track_index": int(playlist[position]),
                "to_track_index": int(playlist[position + 1]),
                "transition_cost": cost,
            }
            for position, cost in enumerate(transition_costs)
        ]
        transitions.sort(
            key=lambda transition: (
                -transition["transition_cost"],
                transition["position"],
            )
        )
        return transitions[:top_k]

    def evaluate(
        self,
        playlist: list[int],
        c_arc: np.ndarray,
        c_trans: np.ndarray,
    ) -> dict[str, float]:
        """Return arc, full-transition, and tail-transition metrics."""
        number_of_tracks = self._validate_arc_matrix(c_arc)
        transition_tracks = self._validate_transition_matrix(c_trans)
        if number_of_tracks != transition_tracks:
            raise ValueError("c_arc and c_trans must have the same number of tracks.")
        self.validate_playlist(playlist, number_of_tracks)

        arc_rmse = self.compute_arc_rmse(playlist, c_arc)
        total_transition_cost = self.compute_total_transition_cost(
            playlist,
            c_trans,
        )
        tail_transition_cost = self.compute_tail_transition_cost(
            playlist,
            c_trans,
        )
        number_of_transitions = number_of_tracks - 1
        tail_size = (
            math.ceil(self.config.tail_fraction * number_of_transitions)
            if number_of_transitions
            else 0
        )

        return {
            "arc_rmse": arc_rmse,
            "total_transition_cost": total_transition_cost,
            "average_transition_cost": (
                total_transition_cost / number_of_transitions
                if number_of_transitions
                else 0.0
            ),
            "tail_transition_cost": tail_transition_cost,
            "average_tail_transition_cost": (
                tail_transition_cost / tail_size if tail_size else 0.0
            ),
        }
