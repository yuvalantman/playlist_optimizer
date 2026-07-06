"""Bidirectional meet-in-the-middle beam baseline (non-recursive, commits everything).

Isolates the value of the recursive/anchor structure in BIBS: this method runs a single
forward beam from the start over the first-half positions and a single backward beam from
the end over the second-half positions, then joins them at the midpoint. Unlike BIBS, it
*commits* the full beam paths (every track placed by the beams), with no recursion.

Supports stochastic (random-walk) pruning via ``MMBeamConfig.stochastic=True``, which
uses the same ``softmax_topk_choice`` sampler as stochastic BIBS so results are directly
comparable across methods.

With ``use_orchestrator=True``, candidate generation at each step uses the same
``CandidateOrchestrator`` as BIBS: candidates are pre-filtered by arc cost (EV distance to
trajectory), transition graph neighbors, and bottleneck scores, instead of iterating all
remaining tracks. This is the "our method" variant of the stochastic MM beam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.sampling import softmax_topk_choice

if TYPE_CHECKING:
    from src.candidate_orchestrator import CandidateOrchestrator
    from src.transition_graph import TransitionGraphData


@dataclass(frozen=True)
class MMBeamConfig:
    """Configuration for the MM beam baseline."""

    beam_width: int = 8
    arc_weight: float = 1.0
    transition_weight: float = 1.0
    stochastic: bool = False
    top_k: int = 5
    temperature: float = 0.15
    random_seed: int = 42
    # When True, use CandidateOrchestrator for candidate filtering (like BIBS).
    # Requires graph_data, bottleneck_results, and candidate_orchestrator in generate().
    use_orchestrator: bool = False


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
        config: MMBeamConfig | None = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = MMBeamConfig(
                beam_width=beam_width,
                arc_weight=arc_weight,
                transition_weight=transition_weight,
            )
        if self.config.beam_width <= 0:
            raise ValueError("beam_width must be positive.")
        # Convenience aliases for backward compatibility.
        self.beam_width = self.config.beam_width
        self.arc_weight = self.config.arc_weight
        self.transition_weight = self.config.transition_weight
        self._rng = np.random.default_rng(self.config.random_seed)

    def _prune(self, expanded: list[_Beam]) -> list[_Beam]:
        """Prune ``expanded`` to ``beam_width`` beams (deterministic or stochastic)."""
        if len(expanded) <= self.config.beam_width:
            return expanded
        if self.config.stochastic:
            scores = np.asarray([b.score for b in expanded], dtype=float)
            chosen = softmax_topk_choice(
                scores,
                self.config.beam_width,
                self._rng,
                self.config.top_k,
                self.config.temperature,
            )
            return [expanded[i] for i in chosen]
        expanded.sort(key=lambda b: (b.score, b.path))
        return expanded[: self.config.beam_width]

    def _get_candidates(
        self,
        last: int,
        position: int,
        available_set: frozenset[int],
        direction: str,
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        graph_data: "TransitionGraphData | None",
        bottleneck_results: dict | None,
        orchestrator: "CandidateOrchestrator | None",
        # right_boundary only used by backward beam to form "anchor" call shape
        right_boundary: int | None = None,
    ) -> list[int]:
        if orchestrator is not None and graph_data is not None and bottleneck_results is not None:
            available_mutable = set(available_set)
            if direction == "forward":
                return orchestrator.build_forward_candidates(
                    last, position, available_mutable, graph_data, bottleneck_results, c_arc
                )
            else:
                rb = right_boundary if right_boundary is not None else last
                return orchestrator.build_backward_candidates(
                    last, position, available_mutable, graph_data, bottleneck_results, c_arc
                )
        return list(available_set)

    def _forward(
        self,
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        start_index: int,
        end_index: int,
        forward_positions: list[int],
        graph_data: "TransitionGraphData | None" = None,
        bottleneck_results: dict | None = None,
        orchestrator: "CandidateOrchestrator | None" = None,
    ) -> list[_Beam]:
        number_of_tracks = c_arc.shape[0]
        all_tracks = frozenset(range(number_of_tracks))
        beams = [_Beam((start_index,), frozenset({start_index}), 0.0)]
        for position in forward_positions:
            expanded: list[_Beam] = []
            for beam in beams:
                last = beam.path[-1]
                pool = frozenset(all_tracks - beam.used - {end_index})
                candidates = self._get_candidates(
                    last, position, pool, "forward",
                    c_arc, c_trans, graph_data, bottleneck_results, orchestrator,
                )
                for candidate in candidates:
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
            beams = self._prune(expanded)
        return beams

    def _backward(
        self,
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        end_index: int,
        backward_positions: list[int],
        available: frozenset[int],
        graph_data: "TransitionGraphData | None" = None,
        bottleneck_results: dict | None = None,
        orchestrator: "CandidateOrchestrator | None" = None,
    ) -> list[_Beam]:
        # path is in placement order (descending positions); path[-1] is the
        # innermost placed track (nearest the midpoint). Its successor is end_index.
        beams = [_Beam((end_index,), frozenset(), 0.0)]
        for position in backward_positions:
            expanded: list[_Beam] = []
            for beam in beams:
                successor = beam.path[-1]
                pool = available - beam.used
                candidates = self._get_candidates(
                    successor, position, frozenset(pool), "backward",
                    c_arc, c_trans, graph_data, bottleneck_results, orchestrator,
                )
                for candidate in candidates:
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
            beams = self._prune(expanded)
        return beams

    def generate(
        self,
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        start_index: int,
        end_index: int,
        graph_data: "TransitionGraphData | None" = None,
        bottleneck_results: dict | None = None,
        candidate_orchestrator: "CandidateOrchestrator | None" = None,
    ) -> list[int]:
        """Return the best joined forward+backward playlist (commits all tracks)."""
        number_of_tracks = c_arc.shape[0]
        if start_index == end_index:
            raise ValueError("start_index and end_index must differ.")
        orch = candidate_orchestrator if self.config.use_orchestrator else None
        self._rng = np.random.default_rng(self.config.random_seed)
        midpoint = (number_of_tracks - 1) // 2
        forward_positions = list(range(1, midpoint + 1))
        backward_positions = list(range(number_of_tracks - 2, midpoint, -1))

        forward_beams = self._forward(
            c_arc, c_trans, start_index, end_index, forward_positions,
            graph_data=graph_data, bottleneck_results=bottleneck_results, orchestrator=orch,
        )

        best_playlist: list[int] | None = None
        best_total = float("inf")
        for forward in forward_beams:
            remaining = (
                frozenset(range(number_of_tracks)) - forward.used - {end_index}
            )
            backward_beams = self._backward(
                c_arc, c_trans, end_index, backward_positions, remaining,
                graph_data=graph_data, bottleneck_results=bottleneck_results, orchestrator=orch,
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
