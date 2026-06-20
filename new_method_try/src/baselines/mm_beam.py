"""Bidirectional meet-in-the-middle beam baseline (non-recursive, commits everything).

Isolates the value of the recursive/anchor structure in BIBS: this method runs a single
forward beam from the start over the first-half positions and a single backward beam from
the end over the second-half positions, then joins them at the midpoint. Unlike BIBS, it
*commits* the full beam paths (every track placed by the beams), with no recursion.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _Beam:
    path: tuple[int, ...]
    used: frozenset[int]
    score: float


class MMBeamBaseline:
    """Forward + backward beams meeting at the midpoint, committing all tracks."""

    def __init__(
        self,
        beam_width: int = 8,
        arc_weight: float = 1.0,
        transition_weight: float = 1.0,
    ) -> None:
        if beam_width <= 0:
            raise ValueError("beam_width must be positive.")
        self.beam_width = beam_width
        self.arc_weight = arc_weight
        self.transition_weight = transition_weight

    def _forward(
        self,
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        start_index: int,
        end_index: int,
        forward_positions: list[int],
    ) -> list[_Beam]:
        number_of_tracks = c_arc.shape[0]
        all_tracks = frozenset(range(number_of_tracks))
        beams = [_Beam((start_index,), frozenset({start_index}), 0.0)]
        for position in forward_positions:
            expanded: list[_Beam] = []
            for beam in beams:
                last = beam.path[-1]
                for candidate in all_tracks - beam.used - {end_index}:
                    step = (
                        self.arc_weight * float(c_arc[candidate, position])
                        + self.transition_weight * float(c_trans[last, candidate])
                    )
                    expanded.append(
                        _Beam(
                            beam.path + (candidate,),
                            beam.used | {candidate},
                            beam.score + step,
                        )
                    )
            expanded.sort(key=lambda beam: (beam.score, beam.path))
            beams = expanded[: self.beam_width]
        return beams

    def _backward(
        self,
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        end_index: int,
        backward_positions: list[int],
        available: frozenset[int],
    ) -> list[_Beam]:
        # path is in placement order (descending positions); path[-1] is the
        # innermost placed track (nearest the midpoint). Its successor is end_index.
        beams = [_Beam((end_index,), frozenset(), 0.0)]
        for position in backward_positions:
            expanded: list[_Beam] = []
            for beam in beams:
                successor = beam.path[-1]
                for candidate in available - beam.used:
                    step = (
                        self.arc_weight * float(c_arc[candidate, position])
                        + self.transition_weight * float(c_trans[candidate, successor])
                    )
                    expanded.append(
                        _Beam(
                            beam.path + (candidate,),
                            beam.used | {candidate},
                            beam.score + step,
                        )
                    )
            expanded.sort(key=lambda beam: (beam.score, beam.path))
            beams = expanded[: self.beam_width]
        return beams

    def generate(
        self,
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        start_index: int,
        end_index: int,
    ) -> list[int]:
        """Return the best joined forward+backward playlist (commits all tracks)."""
        number_of_tracks = c_arc.shape[0]
        if start_index == end_index:
            raise ValueError("start_index and end_index must differ.")
        midpoint = (number_of_tracks - 1) // 2
        forward_positions = list(range(1, midpoint + 1))
        backward_positions = list(range(number_of_tracks - 2, midpoint, -1))

        forward_beams = self._forward(
            c_arc, c_trans, start_index, end_index, forward_positions
        )

        best_playlist: list[int] | None = None
        best_total = float("inf")
        for forward in forward_beams:
            remaining = (
                frozenset(range(number_of_tracks)) - forward.used - {end_index}
            )
            backward_beams = self._backward(
                c_arc, c_trans, end_index, backward_positions, remaining
            )
            forward_last = forward.path[-1]
            for backward in backward_beams:
                if len(backward.used) != len(backward_positions):
                    continue
                innermost = backward.path[-1] if backward.path[1:] else end_index
                join = self.transition_weight * float(
                    c_trans[forward_last, innermost]
                ) if backward_positions else 0.0
                total = forward.score + backward.score + join
                if total < best_total:
                    best_total = total
                    # backward.path = (end_index, p[N-2], p[N-3], ..., p[mid+1])
                    second_half = list(backward.path[1:])[::-1]
                    best_playlist = [*forward.path, *second_half, end_index]

        if best_playlist is None:
            raise ValueError("MM beam failed to produce a complete playlist.")
        if len(set(best_playlist)) != number_of_tracks:
            raise ValueError("MM beam produced a non-exact-once playlist.")
        return best_playlist
