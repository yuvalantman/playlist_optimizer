"""Optimal arc-assignment baseline: the arc_rmse lower bound.

Assigns interior tracks to interior positions to minimize total C_arc using the
Hungarian algorithm (scipy.optimize.linear_sum_assignment), with the fixed start and
end held at positions 0 and L-1. Because this is the cost-minimizing one-to-one
assignment on C_arc, it is the best achievable arc adherence for the pool and serves as
a lower bound; its transition quality is reported only as the trade-off it pays.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class ArcAssignmentBaseline:
    """Minimum-total-arc-cost assignment with fixed endpoints."""

    def generate(
        self,
        c_arc: np.ndarray,
        start_index: int,
        end_index: int,
    ) -> list[int]:
        """Return the exact-once playlist minimizing total arc cost."""
        number_of_tracks = c_arc.shape[0]
        if start_index == end_index:
            raise ValueError("start_index and end_index must differ.")

        interior_tracks = [
            track
            for track in range(number_of_tracks)
            if track not in (start_index, end_index)
        ]
        interior_positions = list(range(1, number_of_tracks - 1))
        cost = c_arc[np.ix_(interior_tracks, interior_positions)]
        row_ind, col_ind = linear_sum_assignment(cost)

        playlist: list[int | None] = [None] * number_of_tracks
        playlist[0] = start_index
        playlist[-1] = end_index
        for row, col in zip(row_ind, col_ind):
            playlist[interior_positions[col]] = interior_tracks[row]
        if any(track is None for track in playlist):
            raise ValueError("Arc assignment failed to fill every position.")
        return [int(track) for track in playlist]
