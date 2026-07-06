"""One-direction (forward) beam-search baseline minimizing arc + transition cost.

Isolates the value of bidirectionality: it is a lookahead generalization of the greedy
baseline (beam_width=1 ~ greedy) that builds strictly left-to-right.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _Beam:
    path: tuple[int, ...]
    used: frozenset[int]
    score: float


class ForwardBeamBaseline:
    """Forward beam search over interior positions with fixed endpoints."""

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

    def generate(
        self,
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        start_index: int,
        end_index: int,
    ) -> list[int]:
        """Return the best beam playlist (fixed start/end, arc + transition)."""
        number_of_tracks = c_arc.shape[0]
        if start_index == end_index:
            raise ValueError("start_index and end_index must differ.")
        all_tracks = frozenset(range(number_of_tracks))
        beams = [_Beam(path=(start_index,), used=frozenset({start_index}), score=0.0)]

        for position in range(1, number_of_tracks - 1):
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
                            path=beam.path + (candidate,),
                            used=beam.used | {candidate},
                            score=beam.score + step,
                        )
                    )
            expanded.sort(key=lambda beam: (beam.score, beam.path))
            beams = expanded[: self.beam_width]

        last_position = number_of_tracks - 1
        best = min(
            beams,
            key=lambda beam: (
                beam.score
                + self.transition_weight * float(c_trans[beam.path[-1], end_index])
                + self.arc_weight * float(c_arc[end_index, last_position]),
                beam.path,
            ),
        )
        return [*best.path, end_index]
