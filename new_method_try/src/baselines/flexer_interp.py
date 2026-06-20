"""Flexer et al. (2008) Divergence-Ratio interpolation baseline.

Original method (Playlist Generation Using Start and End Songs): rank candidate tracks
by their acoustic distance to the start song vs the end song,
    R(i) = D(i, start) / D(i, end),
define evenly-spaced target ratios between R(start) and R(end), and assign each position
the track whose ratio best matches its target.

Adaptation for this project:
- We have feature vectors, not MFCC Gaussians, so D is a normalized-feature Euclidean
  distance instead of KL divergence.
- The raw ratio R = D(i,start)/D(i,end) is numerically unstable (it diverges as a track
  approaches the end song). We use the equivalent, stabilized relative position
      pos(i) = D(i, start) / (D(i, start) + D(i, end) + eps)  in [0, 1],
  which is 0 near the start song and 1 near the end song, and target R_hat(t) = t/(L-1).
- Interior tracks are assigned to interior positions by the Hungarian algorithm to
  minimize total |R_hat(pos) - pos(track)|, guaranteeing an exact-once ordering.

This is a CONSTRUCTION baseline. The same pos(i)/R_hat(t) machinery can later be reused
inside the search as an extra cost dimension (see plan, Flexer role 2) and as an
auxiliary "ratio-adherence" evaluation metric (role 3).
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

_FEATURE_COLUMNS = (
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
)


def feature_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return a symmetric normalized-feature Euclidean distance matrix."""
    features = [column for column in _FEATURE_COLUMNS if column in df.columns]
    if not features:
        raise ValueError("No usable feature columns for Flexer distance.")
    values = df[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Feature columns must contain only finite values.")
    minimums = values.min(axis=0)
    ranges = values.max(axis=0) - minimums
    ranges[ranges == 0] = 1.0
    normalized = (values - minimums) / ranges
    differences = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    return np.linalg.norm(differences, axis=2)


def relative_positions(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
    distance_matrix: np.ndarray | None = None,
    eps: float = 1e-9,
) -> np.ndarray:
    """Return pos(i) = D(i,start)/(D(i,start)+D(i,end)+eps) in [0, 1] per track."""
    distances = (
        distance_matrix if distance_matrix is not None else feature_distance_matrix(df)
    )
    to_start = distances[:, start_index]
    to_end = distances[:, end_index]
    return to_start / (to_start + to_end + eps)


class FlexerInterpolationBaseline:
    """Assign tracks along the start->end acoustic gradient (Flexer 2008)."""

    def generate(
        self,
        df: pd.DataFrame,
        start_index: int,
        end_index: int,
        distance_matrix: np.ndarray | None = None,
    ) -> list[int]:
        """Return the exact-once interpolation playlist with fixed endpoints."""
        number_of_tracks = len(df)
        if start_index == end_index:
            raise ValueError("start_index and end_index must differ.")
        positions_value = relative_positions(
            df, start_index, end_index, distance_matrix
        )
        interior_tracks = [
            track
            for track in range(number_of_tracks)
            if track not in (start_index, end_index)
        ]
        interior_positions = list(range(1, number_of_tracks - 1))
        targets = np.asarray(
            [position / (number_of_tracks - 1) for position in interior_positions]
        )
        cost = np.abs(
            positions_value[interior_tracks, np.newaxis] - targets[np.newaxis, :]
        )
        row_ind, col_ind = linear_sum_assignment(cost)

        playlist: list[int | None] = [None] * number_of_tracks
        playlist[0] = start_index
        playlist[-1] = end_index
        for row, col in zip(row_ind, col_ind):
            playlist[interior_positions[col]] = interior_tracks[row]
        if any(track is None for track in playlist):
            raise ValueError("Flexer interpolation failed to fill every position.")
        return [int(track) for track in playlist]
