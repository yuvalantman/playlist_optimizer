"""Detect difficult tracks and positions from an arc cost matrix."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BottleneckConfig:
    """Configuration for bottleneck scoring and candidate-set construction."""

    song_k_best: int = 5
    location_k_best: int = 10
    candidate_set_size: int = 20
    bottleneck_percentile: float = 0.90


class BottleneckDetector:
    """Identify hard-to-place tracks and hard-to-fill playlist positions."""

    def __init__(self, config: BottleneckConfig) -> None:
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.song_k_best <= 0:
            raise ValueError("song_k_best must be positive.")
        if self.config.location_k_best <= 0:
            raise ValueError("location_k_best must be positive.")
        if self.config.candidate_set_size <= 0:
            raise ValueError("candidate_set_size must be positive.")
        if not 0 <= self.config.bottleneck_percentile <= 1:
            raise ValueError("bottleneck_percentile must be between 0 and 1.")

    @staticmethod
    def _validate_c_arc(c_arc: np.ndarray) -> None:
        if not isinstance(c_arc, np.ndarray) or c_arc.ndim != 2:
            raise ValueError("c_arc must be a 2D numpy array.")
        if c_arc.shape[0] == 0 or c_arc.shape[1] == 0:
            raise ValueError("c_arc dimensions must be non-empty.")
        if not np.issubdtype(c_arc.dtype, np.number):
            raise ValueError("c_arc must contain numeric values.")
        if not np.isfinite(c_arc).all():
            raise ValueError("c_arc must contain only finite values.")

    def compute_song_bottleneck_scores(self, c_arc: np.ndarray) -> np.ndarray:
        """Average each track's k lowest costs across playlist positions."""
        self._validate_c_arc(c_arc)
        number_of_positions = c_arc.shape[1]
        if self.config.song_k_best > number_of_positions:
            raise ValueError(
                "song_k_best cannot exceed the number of playlist positions."
            )

        best_costs = np.partition(
            c_arc,
            kth=self.config.song_k_best - 1,
            axis=1,
        )[:, : self.config.song_k_best]
        return best_costs.mean(axis=1)

    def compute_location_bottleneck_scores(self, c_arc: np.ndarray) -> np.ndarray:
        """Average each position's k lowest costs across all tracks."""
        self._validate_c_arc(c_arc)
        number_of_tracks = c_arc.shape[0]
        if self.config.location_k_best > number_of_tracks:
            raise ValueError("location_k_best cannot exceed the number of tracks.")

        best_costs = np.partition(
            c_arc,
            kth=self.config.location_k_best - 1,
            axis=0,
        )[: self.config.location_k_best, :]
        return best_costs.mean(axis=0)

    def get_bottleneck_track_indices(
        self,
        song_scores: np.ndarray,
    ) -> np.ndarray:
        """Return track indices at or above the configured score percentile."""
        if not isinstance(song_scores, np.ndarray) or song_scores.ndim != 1:
            raise ValueError("song_scores must be a 1D numpy array.")
        if len(song_scores) == 0:
            raise ValueError("song_scores must not be empty.")
        if not np.issubdtype(song_scores.dtype, np.number):
            raise ValueError("song_scores must contain numeric values.")
        if not np.isfinite(song_scores).all():
            raise ValueError("song_scores must contain only finite values.")

        threshold = np.quantile(song_scores, self.config.bottleneck_percentile)
        return np.flatnonzero(song_scores >= threshold)

    def build_candidate_sets(self, c_arc: np.ndarray) -> dict[int, list[int]]:
        """Return the lowest-cost candidate track indices for each position."""
        self._validate_c_arc(c_arc)
        number_of_tracks, number_of_positions = c_arc.shape
        if self.config.candidate_set_size > number_of_tracks:
            raise ValueError(
                "candidate_set_size cannot exceed the number of tracks."
            )

        return {
            position: np.argsort(c_arc[:, position], kind="stable")[
                : self.config.candidate_set_size
            ].tolist()
            for position in range(number_of_positions)
        }

    def detect(self, c_arc: np.ndarray) -> dict[str, object]:
        """Run all bottleneck analyses and return their results."""
        song_scores = self.compute_song_bottleneck_scores(c_arc)
        location_scores = self.compute_location_bottleneck_scores(c_arc)

        return {
            "song_bottleneck_scores": song_scores,
            "location_bottleneck_scores": location_scores,
            "bottleneck_track_indices": self.get_bottleneck_track_indices(
                song_scores
            ),
            "candidate_sets": self.build_candidate_sets(c_arc),
        }
