"""Bottleneck-Guided Bidirectional Beam Search playlist construction."""

from dataclasses import dataclass
from itertools import permutations

import numpy as np
import pandas as pd

from src.bottleneck_detector import validate_bottleneck_results
from src.candidate_orchestrator import (
    CandidateOrchestrator,
    CandidateOrchestratorConfig,
)
from src.transition_graph import (
    TransitionGraphBuilder,
    TransitionGraphConfig,
    TransitionGraphData,
)


@dataclass(frozen=True)
class BIBSConfig:
    """Configuration for Bottleneck-Guided Bidirectional Beam Search."""

    # Number of partial paths retained during beam expansion.
    beam_width: int = 4
    # Number of positions processed per chunk during beam expansion progress.
    beam_length: int = 3
    # Recursion threshold where intervals switch to local base-case matching.
    base_case_size: int = 4
    max_recursion_depth: int = 30
    arc_weight: float = 1.0
    transition_weight: float = 1.0
    bottleneck_weight: float = 0.6
    anchor_arc_weight: float = 1.0
    anchor_transition_weight: float = 1.2
    anchor_bottleneck_weight: float = 0.6
    anchor_balance_weight: float = 0.3
    base_arc_weight: float = 0.3
    base_transition_weight: float = 1.5
    base_case_max_candidates: int = 12


@dataclass(frozen=True)
class _BeamItem:
    path: tuple[int, ...]
    used: frozenset[int]
    last_track: int
    score: float
    arc_component: float = 0.0
    transition_component: float = 0.0
    bottleneck_component: float = 0.0
    balance_component: float = 0.0


class BIBS:
    """Construct an exact-once playlist with recursive bidirectional search."""

    SCORE_COMPONENT_NAMES = (
        "arc_component",
        "transition_component",
        "bottleneck_component",
        "balance_component",
    )

    DIAGNOSTIC_NAMES = (
        "intervals_solved",
        "anchors_selected",
        "base_cases_solved",
        "forward_beam_expansions",
        "backward_beam_expansions",
        "candidate_orchestrator_calls",
        "selected_bottleneck_anchors",
        "base_case_forced_assignment_count",
        "anchors_with_arc_cost_above_0_25",
        "anchors_with_arc_cost_above_0_35",
    )

    def __init__(self, config: BIBSConfig) -> None:
        self.config = config
        self.anchor_history: list[dict] = []
        self.decision_trace: list[dict] = []
        self.recursion_trace: list[dict] = []
        self.beam_trace: list[dict] = []
        self.anchor_selection_trace: list[dict] = []
        self.base_case_trace: list[dict] = []
        self._diagnostics: dict[str, int | float | bool] = {}
        self._anchor_score_total = 0.0
        self._ev_scores = np.asarray([], dtype=float)
        self._camelot_codes = np.asarray([], dtype=str)
        self._camelot_numbers = np.asarray([], dtype=str)
        self._validate_config()
        self._reset_state()

    def _validate_config(self) -> None:
        integer_fields = (
            "beam_width",
            "beam_length",
            "base_case_size",
            "max_recursion_depth",
            "base_case_max_candidates",
        )
        for name in integer_fields:
            if getattr(self.config, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        weight_fields = (
            "arc_weight",
            "transition_weight",
            "bottleneck_weight",
            "anchor_arc_weight",
            "anchor_transition_weight",
            "anchor_bottleneck_weight",
            "anchor_balance_weight",
            "base_arc_weight",
            "base_transition_weight",
        )
        for name in weight_fields:
            if getattr(self.config, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.config.base_case_max_candidates < self.config.base_case_size:
            raise ValueError(
                "base_case_max_candidates cannot be smaller than base_case_size."
            )

    def _reset_state(self) -> None:
        self.anchor_history = []
        self.decision_trace = []
        self.recursion_trace = []
        self.beam_trace = []
        self.anchor_selection_trace = []
        self.base_case_trace = []
        self._diagnostics = {name: 0 for name in self.DIAGNOSTIC_NAMES}
        self._diagnostics["graph_used"] = False
        self._diagnostics["bottleneck_results_used"] = False
        self._diagnostics["base_case_candidate_count_total"] = 0.0
        self._diagnostics["worst_base_case_transition_component"] = 0.0
        for component in self.SCORE_COMPONENT_NAMES:
            self._diagnostics[f"total_{component}"] = 0.0
        self._anchor_score_total = 0.0

    def _record_score_components(self, components: dict[str, float]) -> None:
        """Accumulate the weighted components of one selected major decision."""
        for component in self.SCORE_COMPONENT_NAMES:
            self._diagnostics[f"total_{component}"] += float(
                components.get(component, 0.0)
            )

    @staticmethod
    def _sum_components(
        *component_sets: dict[str, float],
    ) -> dict[str, float]:
        """Return the element-wise sum of score component dictionaries."""
        return {
            component: float(
                sum(values.get(component, 0.0) for values in component_sets)
            )
            for component in BIBS.SCORE_COMPONENT_NAMES
        }

    @staticmethod
    def _normalize_arc_cost(value: float) -> float:
        """Scale arc deviation into a bounded decision-cost range."""
        return float(np.clip(float(value), 0.0, 1.0))

    @staticmethod
    def _normalize_transition_cost(value: float) -> float:
        """Scale and clip transition cost so a single edge cannot dominate."""
        return float(np.clip(float(value), 0.0, 2.5))

    @staticmethod
    def _normalize_bottleneck_score(value: float) -> float:
        """Scale bottleneck difficulty into a useful bounded bonus range."""
        return float(np.clip(float(value), 0.0, 1.0))

    def _bottleneck_bonus(
        self,
        score: float,
        weight: float,
    ) -> float:
        """Return the methodology bottleneck priority bonus."""
        return float(-weight * self._normalize_bottleneck_score(score))

    @staticmethod
    def _validate_inputs(
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        bottleneck_results: dict,
        graph_data: TransitionGraphData,
        candidate_orchestrator: CandidateOrchestrator,
        target_arc: np.ndarray,
        track_pool: pd.DataFrame,
        start_index: int,
        end_index: int,
    ) -> int:
        if (
            not isinstance(c_arc, np.ndarray)
            or c_arc.ndim != 2
            or c_arc.shape[0] == 0
            or c_arc.shape[0] != c_arc.shape[1]
            or not np.issubdtype(c_arc.dtype, np.number)
            or not np.isfinite(c_arc).all()
        ):
            raise ValueError("c_arc must be a finite non-empty square matrix.")
        number_of_tracks = c_arc.shape[0]
        if (
            not isinstance(c_trans, np.ndarray)
            or c_trans.shape != c_arc.shape
            or not np.issubdtype(c_trans.dtype, np.number)
            or not np.isfinite(c_trans).all()
        ):
            raise ValueError("c_trans must be a finite square matrix matching c_arc.")
        target = np.asarray(target_arc, dtype=float)
        if target.shape != (number_of_tracks,) or not np.isfinite(target).all():
            raise ValueError("target_arc must contain one finite value per track.")
        if not isinstance(track_pool, pd.DataFrame) or len(track_pool) != number_of_tracks:
            raise ValueError("track_pool must contain one row per track.")
        required_columns = {"EV_score", "camelot"}
        if not required_columns.issubset(track_pool.columns):
            raise ValueError("track_pool must contain EV_score and camelot columns.")
        if not isinstance(graph_data, TransitionGraphData):
            raise ValueError("graph_data must be TransitionGraphData.")
        expected_nodes = set(range(number_of_tracks))
        if (
            set(graph_data.outgoing_neighbors) != expected_nodes
            or set(graph_data.incoming_neighbors) != expected_nodes
        ):
            raise ValueError("graph_data must contain every track.")
        if not isinstance(candidate_orchestrator, CandidateOrchestrator):
            raise ValueError("candidate_orchestrator must be CandidateOrchestrator.")
        validate_bottleneck_results(
            bottleneck_results,
            number_of_tracks,
            number_of_tracks,
        )
        for name, index in (("start_index", start_index), ("end_index", end_index)):
            if (
                not isinstance(index, (int, np.integer))
                or not 0 <= int(index) < number_of_tracks
            ):
                raise ValueError(f"{name} must be a valid track index.")
        if start_index == end_index:
            raise ValueError("start_index and end_index must be different.")
        return number_of_tracks

    def _expand_beam(
        self,
        boundary_track: int,
        positions: list[int],
        direction: str,
        available_tracks: set[int],
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        bottleneck_results: dict,
        graph_data: TransitionGraphData,
        candidate_orchestrator: CandidateOrchestrator,
        target_arc: np.ndarray,
        track_pool: pd.DataFrame,
    ) -> list[_BeamItem]:
        del target_arc, track_pool
        beams = [
            _BeamItem(
                path=(),
                used=frozenset(),
                last_track=boundary_track,
                score=0.0,
            )
        ]
        song_scores = np.asarray(bottleneck_results["song_bottleneck_scores"])

        def chunked_positions():
            """Yield positions in beam_length-sized progress chunks."""
            for chunk_index, chunk_start in enumerate(
                range(0, len(positions), self.config.beam_length)
            ):
                chunk = positions[chunk_start : chunk_start + self.config.beam_length]
                for position_index_in_chunk, chunk_position in enumerate(chunk):
                    yield chunk_index, position_index_in_chunk, chunk_position

        # beam_length controls chunked expansion progress; beam_width prunes
        # surviving partial paths after each position.
        for chunk_index, position_index_in_chunk, position in chunked_positions():
            beams_before_expansion = len(beams)
            expanded: list[_BeamItem] = []
            for beam in beams:
                beam_available = available_tracks.difference(beam.used)
                self._diagnostics["candidate_orchestrator_calls"] += 1
                if direction == "forward":
                    candidates = candidate_orchestrator.build_forward_candidates(
                        beam.last_track,
                        position,
                        beam_available,
                        graph_data,
                        bottleneck_results,
                        c_arc,
                    )
                else:
                    candidates = candidate_orchestrator.build_backward_candidates(
                        beam.last_track,
                        position,
                        beam_available,
                        graph_data,
                        bottleneck_results,
                        c_arc,
                    )
                for candidate in candidates:
                    if direction == "forward":
                        transition = c_trans[beam.last_track, candidate]
                    else:
                        transition = c_trans[candidate, beam.last_track]
                    normalized_arc = self._normalize_arc_cost(
                        c_arc[candidate, position]
                    )
                    normalized_transition = self._normalize_transition_cost(
                        transition
                    )
                    arc_component = (
                        self.config.arc_weight * normalized_arc
                    )
                    transition_component = (
                        self.config.transition_weight * normalized_transition
                    )
                    bottleneck_component = self._bottleneck_bonus(
                        song_scores[candidate],
                        self.config.bottleneck_weight,
                    )
                    local_score = (
                        arc_component
                        + transition_component
                        + bottleneck_component
                    )
                    expanded.append(
                        _BeamItem(
                            path=beam.path + (candidate,),
                            used=beam.used.union({candidate}),
                            last_track=candidate,
                            score=float(beam.score + local_score),
                            arc_component=float(
                                beam.arc_component + arc_component
                            ),
                            transition_component=float(
                                beam.transition_component + transition_component
                            ),
                            bottleneck_component=float(
                                beam.bottleneck_component + bottleneck_component
                            ),
                            balance_component=float(beam.balance_component),
                        )
                    )
            if direction == "forward":
                self._diagnostics["forward_beam_expansions"] += len(expanded)
            else:
                self._diagnostics["backward_beam_expansions"] += len(expanded)
            if not expanded:
                self.beam_trace.append(
                    {
                        "direction": direction,
                        "position": position,
                        "chunk_index": chunk_index,
                        "position_index_in_chunk": position_index_in_chunk,
                        "total_positions_in_call": len(positions),
                        "beam_length": self.config.beam_length,
                        "beam_width": self.config.beam_width,
                        "beams_before_expansion": beams_before_expansion,
                        "expanded_candidate_count": len(expanded),
                        "beams_after_pruning": len(beams),
                        "returned_early_due_to_no_expansion": True,
                    }
                )
                return beams
            expanded.sort(key=lambda beam: (beam.score, beam.path))
            beams = expanded[: self.config.beam_width]
            self.beam_trace.append(
                {
                    "direction": direction,
                    "position": position,
                    "chunk_index": chunk_index,
                    "position_index_in_chunk": position_index_in_chunk,
                    "total_positions_in_call": len(positions),
                    "beam_length": self.config.beam_length,
                    "beam_width": self.config.beam_width,
                    "beams_before_expansion": beams_before_expansion,
                    "expanded_candidate_count": len(expanded),
                    "beams_after_pruning": len(beams),
                    "returned_early_due_to_no_expansion": False,
                }
            )
        return beams

    def _select_anchor(
        self,
        interval_left_pos: int,
        interval_right_pos: int,
        left_track: int,
        right_track: int,
        midpoint: int,
        forward_beams: list[_BeamItem],
        backward_beams: list[_BeamItem],
        available_tracks: set[int],
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        bottleneck_results: dict,
        graph_data: TransitionGraphData,
        candidate_orchestrator: CandidateOrchestrator,
        target_arc: np.ndarray,
        track_pool: pd.DataFrame,
    ) -> tuple[int, int, int, float, dict]:
        del target_arc, track_pool
        self._diagnostics["candidate_orchestrator_calls"] += 1
        anchor_candidates = candidate_orchestrator.build_anchor_candidates(
            left_track,
            right_track,
            midpoint,
            available_tracks,
            graph_data,
            bottleneck_results,
            c_arc,
            c_trans,
        )
        if not anchor_candidates:
            raise ValueError("CandidateOrchestrator returned no anchor candidates.")
        candidate_sources = candidate_orchestrator.get_last_anchor_candidate_sources()
        song_scores = np.asarray(bottleneck_results["song_bottleneck_scores"])
        best: tuple[float, int, int, int, dict[str, float]] | None = None
        best_scores_by_candidate: dict[int, float] = {}
        best_components_by_candidate: dict[int, dict[str, float]] = {}
        for forward in forward_beams:
            for backward in backward_beams:
                if not forward.used.isdisjoint(backward.used):
                    continue
                for anchor in anchor_candidates:
                    if anchor in forward.used or anchor in backward.used:
                        continue
                    left_neighbor = forward.last_track
                    right_neighbor = backward.last_track
                    left_transition_cost = float(c_trans[left_neighbor, anchor])
                    right_transition_cost = float(c_trans[anchor, right_neighbor])
                    arc_component = (
                        forward.arc_component
                        + backward.arc_component
                        + self.config.anchor_arc_weight
                        * float(c_arc[anchor, midpoint])
                    )
                    left_transition_component = (
                        self.config.anchor_transition_weight
                        * left_transition_cost
                    )
                    right_transition_component = (
                        self.config.anchor_transition_weight
                        * right_transition_cost
                    )
                    transition_component = (
                        forward.transition_component
                        + backward.transition_component
                        + left_transition_component
                        + right_transition_component
                    )
                    bottleneck_component = (
                        forward.bottleneck_component
                        + backward.bottleneck_component
                        + self._bottleneck_bonus(
                            song_scores[anchor],
                            self.config.anchor_bottleneck_weight,
                        )
                    )
                    balance_gap = abs(left_transition_cost - right_transition_cost)
                    balance_component = (
                        forward.balance_component
                        + backward.balance_component
                        + self.config.anchor_balance_weight * balance_gap
                    )
                    score = (
                        arc_component
                        + transition_component
                        + bottleneck_component
                        + balance_component
                    )
                    components = {
                        "arc_component": float(arc_component),
                        "transition_component": float(transition_component),
                        "bottleneck_component": float(bottleneck_component),
                        "balance_component": float(balance_component),
                        "balance_gap": float(balance_gap),
                        "left_transition_cost": left_transition_cost,
                        "right_transition_cost": right_transition_cost,
                        "raw_arc_cost": float(c_arc[anchor, midpoint]),
                        "raw_bottleneck_score": float(song_scores[anchor]),
                        "left_transition_component": float(
                            left_transition_component
                        ),
                        "right_transition_component": float(
                            right_transition_component
                        ),
                    }
                    candidate_score = float(score)
                    if candidate_score < best_scores_by_candidate.get(
                        anchor,
                        float("inf"),
                    ):
                        best_scores_by_candidate[anchor] = candidate_score
                        best_components_by_candidate[anchor] = components
                    key = (
                        candidate_score,
                        anchor,
                        left_neighbor,
                        right_neighbor,
                    )
                    if best is None or key < best[:4]:
                        best = (*key, components)
        if best is None:
            anchor = anchor_candidates[0]
            arc_component = float(
                self.config.anchor_arc_weight
                * float(c_arc[anchor, midpoint])
            )
            left_transition_cost = float(c_trans[left_track, anchor])
            right_transition_cost = float(c_trans[anchor, right_track])
            left_transition_component = float(
                self.config.anchor_transition_weight
                * left_transition_cost
            )
            right_transition_component = float(
                self.config.anchor_transition_weight
                * right_transition_cost
            )
            bottleneck_component = self._bottleneck_bonus(
                song_scores[anchor],
                self.config.anchor_bottleneck_weight,
            )
            balance_gap = abs(left_transition_cost - right_transition_cost)
            balance_component = float(
                self.config.anchor_balance_weight * balance_gap
            )
            components = {
                component: 0.0 for component in self.SCORE_COMPONENT_NAMES
            }
            components.update(
                {
                    "arc_component": arc_component,
                    "transition_component": (
                        left_transition_component + right_transition_component
                    ),
                    "bottleneck_component": bottleneck_component,
                    "balance_component": balance_component,
                    "balance_gap": float(balance_gap),
                    "left_transition_cost": left_transition_cost,
                    "right_transition_cost": right_transition_cost,
                    "raw_arc_cost": float(c_arc[anchor, midpoint]),
                    "raw_bottleneck_score": float(song_scores[anchor]),
                    "left_transition_component": left_transition_component,
                    "right_transition_component": right_transition_component,
                }
            )
            score = float(
                arc_component
                + left_transition_component
                + right_transition_component
                + bottleneck_component
                + balance_component
            )
            best_scores_by_candidate[anchor] = score
            best_components_by_candidate[anchor] = components
            left_neighbor = left_track
            right_neighbor = right_track
        else:
            score, anchor, left_neighbor, right_neighbor, components = best

        ranked_candidates = sorted(
            best_scores_by_candidate,
            key=lambda candidate: (
                best_scores_by_candidate[candidate],
                candidate,
            ),
        )
        sources = sorted(candidate_sources.get(anchor, set()))
        top_anchor_candidates = []
        for candidate in ranked_candidates[:5]:
            candidate_components = best_components_by_candidate[candidate]
            top_anchor_candidates.append(
                {
                    "track": int(candidate),
                    "score": float(best_scores_by_candidate[candidate]),
                    "arc_component": float(
                        candidate_components["arc_component"]
                    ),
                    "transition_component": float(
                        candidate_components["transition_component"]
                    ),
                    "bottleneck_component": float(
                        candidate_components["bottleneck_component"]
                    ),
                    "balance_component": float(
                        candidate_components["balance_component"]
                    ),
                    "raw_bottleneck_score": float(
                        candidate_components["raw_bottleneck_score"]
                    ),
                }
            )
        selection_details = {
            **components,
            "candidate_count": len(anchor_candidates),
            "selected_candidate_rank": (
                ranked_candidates.index(anchor) + 1
                if anchor in ranked_candidates
                else None
            ),
            "best_candidate_score": (
                float(best_scores_by_candidate[ranked_candidates[0]])
                if ranked_candidates
                else float(score)
            ),
            "selected_candidate_score": float(score),
            "score_gap_from_best": (
                float(score - best_scores_by_candidate[ranked_candidates[0]])
                if ranked_candidates
                else 0.0
            ),
            "selection_reason": "best_compatible_candidate",
            "selected_candidate_sources": ", ".join(sources),
            "whether_selected_candidate_from_graph": bool(
                {"outgoing_graph", "incoming_graph", "intersection"}.intersection(
                    sources
                )
            ),
            "whether_selected_candidate_from_candidate_set": (
                "candidate_set" in sources
            ),
            "whether_selected_candidate_from_bottleneck": "bottleneck" in sources,
            "whether_selected_candidate_from_bridge": "bridge" in sources,
            "best_candidate_track": (
                int(ranked_candidates[0]) if ranked_candidates else None
            ),
            "top_anchor_candidates": top_anchor_candidates,
        }
        self.anchor_selection_trace.append(
            {
                "interval_left_pos": interval_left_pos,
                "interval_right_pos": interval_right_pos,
                "midpoint": midpoint,
                "selected_anchor_track": anchor,
                "selected_left_neighbor": left_neighbor,
                "selected_right_neighbor": right_neighbor,
                "selected_anchor_score": float(score),
                "selected_anchor_rank": selection_details[
                    "selected_candidate_rank"
                ],
                "candidate_count": len(anchor_candidates),
                "arc_component": float(selection_details["arc_component"]),
                "transition_component": float(
                    selection_details["transition_component"]
                ),
                "bottleneck_component": float(
                    selection_details["bottleneck_component"]
                ),
                "balance_component": float(
                    selection_details["balance_component"]
                ),
                "left_transition_cost": float(
                    selection_details["left_transition_cost"]
                ),
                "right_transition_cost": float(
                    selection_details["right_transition_cost"]
                ),
                "balance_gap": float(selection_details["balance_gap"]),
                "raw_arc_cost": float(selection_details["raw_arc_cost"]),
                "raw_bottleneck_score": float(
                    selection_details["raw_bottleneck_score"]
                ),
                "selected_candidate_sources": selection_details[
                    "selected_candidate_sources"
                ],
                "whether_selected_candidate_from_graph": selection_details[
                    "whether_selected_candidate_from_graph"
                ],
                "whether_selected_candidate_from_candidate_set": selection_details[
                    "whether_selected_candidate_from_candidate_set"
                ],
                "whether_selected_candidate_from_bottleneck": selection_details[
                    "whether_selected_candidate_from_bottleneck"
                ],
                "whether_selected_candidate_from_bridge": selection_details[
                    "whether_selected_candidate_from_bridge"
                ],
                "best_candidate_track": selection_details["best_candidate_track"],
                "best_candidate_score": selection_details[
                    "best_candidate_score"
                ],
                "score_gap_from_best": selection_details[
                    "score_gap_from_best"
                ],
                "top_anchor_candidates": top_anchor_candidates,
            }
        )
        return anchor, left_neighbor, right_neighbor, float(score), selection_details

    def _record_anchor(
        self,
        interval_left_pos: int,
        interval_right_pos: int,
        interval_left_track: int,
        interval_right_track: int,
        position: int,
        anchor: int,
        left_neighbor: int,
        right_neighbor: int,
        score: float,
        selection_details: dict,
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        bottleneck_results: dict,
        track_pool: pd.DataFrame,
    ) -> None:
        song_scores = np.asarray(bottleneck_results["song_bottleneck_scores"])
        alignment = float(
            c_arc[anchor, position]
            + c_trans[left_neighbor, anchor]
            + c_trans[anchor, right_neighbor]
        )
        anchor_record = {
            "position": position,
            "anchor_track": anchor,
            "left_neighbor": left_neighbor,
            "right_neighbor": right_neighbor,
            "arc_cost": float(c_arc[anchor, position]),
            "left_transition_cost": float(c_trans[left_neighbor, anchor]),
            "right_transition_cost": float(c_trans[anchor, right_neighbor]),
            "bs_song": float(song_scores[anchor]),
            "anchor_alignment_cost": alignment,
            "anchor_score": float(score),
        }
        self.anchor_history.append(anchor_record)
        if anchor_record["arc_cost"] > 0.25:
            self._diagnostics["anchors_with_arc_cost_above_0_25"] += 1
        if anchor_record["arc_cost"] > 0.35:
            self._diagnostics["anchors_with_arc_cost_above_0_35"] += 1
        trace_record = {
            "decision_type": "anchor",
            "interval_left_pos": interval_left_pos,
            "interval_right_pos": interval_right_pos,
            "midpoint": position,
            "left_track": interval_left_track,
            "right_track": interval_right_track,
            "selected_left_neighbor": left_neighbor,
            "selected_right_neighbor": right_neighbor,
            "selected_anchor": anchor,
            "selected_anchor_name": (
                str(track_pool.loc[anchor, "track_name"])
                if "track_name" in track_pool.columns
                else ""
            ),
            "selected_anchor_EV_score": float(self._ev_scores[anchor]),
            "selected_anchor_camelot": str(self._camelot_codes[anchor]),
            "anchor_score_total": float(score),
            **selection_details,
        }
        self.decision_trace.append(trace_record)
        self._record_score_components(selection_details)

    def _base_case_score(
        self,
        sequence: tuple[int, ...],
        positions: list[int],
        left_track: int,
        right_track: int,
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        bottleneck_results: dict,
        target_arc: np.ndarray,
        track_pool: pd.DataFrame,
    ) -> tuple[float, dict[str, float]]:
        del target_arc, track_pool
        song_scores = np.asarray(bottleneck_results["song_bottleneck_scores"])
        arc_cost = sum(
            self._normalize_arc_cost(c_arc[track, position])
            for track, position in zip(sequence, positions)
        )
        path = (left_track,) + sequence + (right_track,)
        raw_transition_costs = [
            float(c_trans[source, target])
            for source, target in zip(path, path[1:])
        ]
        transition_cost = sum(
            self._normalize_transition_cost(cost)
            for cost in raw_transition_costs
        )
        components = {
            "arc_component": float(self.config.base_arc_weight * arc_cost),
            "transition_component": float(
                self.config.base_transition_weight * transition_cost
            ),
            "bottleneck_component": float(
                sum(
                    self._bottleneck_bonus(
                        song_scores[track],
                        self.config.bottleneck_weight,
                    )
                    for track, position in zip(sequence, positions)
                )
            ),
            "balance_component": 0.0,
        }
        score = (
            components["arc_component"]
            + components["transition_component"]
            + components["bottleneck_component"]
        )
        return float(score), components

    def _fill_base_case(
        self,
        playlist: list[int | None],
        left_pos: int,
        right_pos: int,
        available_tracks: set[int],
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        bottleneck_results: dict,
        graph_data: TransitionGraphData,
        candidate_orchestrator: CandidateOrchestrator,
        target_arc: np.ndarray,
        track_pool: pd.DataFrame,
    ) -> None:
        positions = list(range(left_pos + 1, right_pos))
        if not positions:
            return
        left_track = int(playlist[left_pos])
        right_track = int(playlist[right_pos])
        self._diagnostics["candidate_orchestrator_calls"] += 1
        candidates = candidate_orchestrator.build_base_case_candidates(
            left_track,
            right_track,
            positions,
            available_tracks,
            graph_data,
            bottleneck_results,
            c_arc,
        )
        initial_candidate_count = len(candidates)
        forced_assignment = initial_candidate_count <= len(positions)
        if forced_assignment:
            self._diagnostics["base_case_forced_assignment_count"] += 1

        candidate_sources: list[int] = list(candidates)
        candidate_sources.extend(graph_data.outgoing_neighbors[left_track])
        candidate_sources.extend(graph_data.incoming_neighbors[right_track])
        for position in positions:
            arc_order = np.argsort(c_arc[:, position], kind="stable")
            candidate_sources.extend(
                int(track)
                for track in arc_order
                if int(track) in available_tracks
            )
            candidate_sources.extend(
                int(track)
                for track in bottleneck_results["candidate_sets"][position]
            )
        interval_target_min = float(np.min(target_arc[positions]))
        interval_target_max = float(np.max(target_arc[positions]))
        candidate_sources.extend(
            int(track)
            for track in bottleneck_results["bottleneck_track_indices"]
            if (
                track in available_tracks
                and interval_target_min - 0.20
                <= self._ev_scores[int(track)]
                <= interval_target_max + 0.20
            )
        )

        merged_candidates: list[int] = []
        seen: set[int] = set()
        for track in candidate_sources:
            track = int(track)
            if track in available_tracks and track not in seen:
                seen.add(track)
                merged_candidates.append(track)

        def candidate_quality(track: int) -> tuple[float, int]:
            arc_fit = min(
                self._normalize_arc_cost(c_arc[track, position])
                for position in positions
            )
            boundary_fit = min(
                self._normalize_transition_cost(c_trans[left_track, track]),
                self._normalize_transition_cost(c_trans[track, right_track]),
            )
            return float(arc_fit + 0.25 * boundary_fit), track

        desired_count = min(
            len(available_tracks),
            self.config.base_case_max_candidates,
        )
        candidates = sorted(merged_candidates, key=candidate_quality)[:desired_count]
        if len(candidates) < len(positions):
            candidates.extend(
                sorted(
                    available_tracks.difference(candidates),
                    key=candidate_quality,
                )[: len(positions) - len(candidates)]
            )
        best_sequence: tuple[int, ...] | None = None
        best_components: dict[str, float] | None = None
        best_score = float("inf")
        permutation_count = 0
        for sequence in permutations(candidates, len(positions)):
            permutation_count += 1
            score, components = self._base_case_score(
                sequence,
                positions,
                left_track,
                right_track,
                c_arc,
                c_trans,
                bottleneck_results,
                target_arc,
                track_pool,
            )
            if (score, sequence) < (best_score, best_sequence or sequence):
                best_score = score
                best_sequence = sequence
                best_components = components
        if best_sequence is None or best_components is None:
            raise ValueError("Unable to solve BIBS base case.")
        for position, track in zip(positions, best_sequence):
            playlist[position] = track
            available_tracks.remove(track)
        selected_path = (left_track,) + best_sequence + (right_track,)
        selected_transition_costs = [
            float(c_trans[source, target])
            for source, target in zip(selected_path, selected_path[1:])
        ]
        average_arc_cost_selected = float(
            np.mean(
                [
                    float(c_arc[track, position])
                    for track, position in zip(best_sequence, positions)
                ]
            )
        )
        self.base_case_trace.append(
            {
                "left_pos": left_pos,
                "right_pos": right_pos,
                "interval_size": right_pos - left_pos + 1,
                "empty_count": len(positions),
                "empty_positions": list(positions),
                "left_boundary_track": left_track,
                "right_boundary_track": right_track,
                "candidate_count": len(candidates),
                "initial_candidate_count": initial_candidate_count,
                "forced_assignment_warning": forced_assignment,
                "permutation_count_tried": permutation_count,
                "selected_tracks": list(best_sequence),
                "local_score_total": float(best_score),
                "arc_component": float(best_components["arc_component"]),
                "transition_component": float(
                    best_components["transition_component"]
                ),
                "bottleneck_component": float(
                    best_components["bottleneck_component"]
                ),
                "balance_component": float(
                    best_components.get("balance_component", 0.0)
                ),
                "average_arc_cost_selected": average_arc_cost_selected,
                "total_transition_cost_selected": float(
                    sum(selected_transition_costs)
                ),
                "first_transition_cost": float(selected_transition_costs[0]),
                "last_transition_cost": float(selected_transition_costs[-1]),
            }
        )
        self._diagnostics["base_cases_solved"] += 1
        self._diagnostics["base_case_candidate_count_total"] += len(candidates)
        self._diagnostics["worst_base_case_transition_component"] = max(
            float(self._diagnostics["worst_base_case_transition_component"]),
            float(best_components["transition_component"]),
        )
        self.decision_trace.append(
            {
                "decision_type": "base_case",
                "interval_left_pos": left_pos,
                "interval_right_pos": right_pos,
                "empty_positions": positions,
                "selected_tracks": list(best_sequence),
                "local_score_total": float(best_score),
                **best_components,
                "candidate_count": len(candidates),
                "initial_candidate_count": initial_candidate_count,
                "forced_assignment_warning": forced_assignment,
                "permutation_count_tried": permutation_count,
            }
        )
        self._record_score_components(best_components)

    def _solve_interval(
        self,
        playlist: list[int | None],
        left_pos: int,
        right_pos: int,
        available_tracks: set[int],
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        bottleneck_results: dict,
        graph_data: TransitionGraphData,
        candidate_orchestrator: CandidateOrchestrator,
        target_arc: np.ndarray,
        track_pool: pd.DataFrame,
        depth: int,
    ) -> None:
        empty_count = right_pos - left_pos - 1
        if empty_count <= 0:
            return
        self._diagnostics["intervals_solved"] += 1
        if empty_count <= self.config.base_case_size or depth >= self.config.max_recursion_depth:
            self.recursion_trace.append(
                {
                    "depth": depth,
                    "left_pos": left_pos,
                    "right_pos": right_pos,
                    "interval_size": right_pos - left_pos + 1,
                    "empty_count": empty_count,
                    "midpoint": None,
                    "is_base_case": True,
                    "forward_positions_start": None,
                    "forward_positions_end": None,
                    "forward_positions_count": 0,
                    "backward_positions_start": None,
                    "backward_positions_end": None,
                    "backward_positions_count": 0,
                    "selected_anchor_position": None,
                    "selected_anchor_track": None,
                    "left_child_interval": None,
                    "right_child_interval": None,
                }
            )
            self._fill_base_case(
                playlist,
                left_pos,
                right_pos,
                available_tracks,
                c_arc,
                c_trans,
                bottleneck_results,
                graph_data,
                candidate_orchestrator,
                target_arc,
                track_pool,
            )
            return

        midpoint = (left_pos + right_pos) // 2
        left_track = int(playlist[left_pos])
        right_track = int(playlist[right_pos])
        forward_positions = list(range(left_pos + 1, midpoint))
        backward_positions = list(range(right_pos - 1, midpoint, -1))
        forward_beams = self._expand_beam(
            left_track,
            forward_positions,
            "forward",
            available_tracks,
            c_arc,
            c_trans,
            bottleneck_results,
            graph_data,
            candidate_orchestrator,
            target_arc,
            track_pool,
        )
        backward_beams = self._expand_beam(
            right_track,
            backward_positions,
            "backward",
            available_tracks,
            c_arc,
            c_trans,
            bottleneck_results,
            graph_data,
            candidate_orchestrator,
            target_arc,
            track_pool,
        )
        (
            anchor,
            left_neighbor,
            right_neighbor,
            score,
            selection_details,
        ) = self._select_anchor(
            left_pos,
            right_pos,
            left_track,
            right_track,
            midpoint,
            forward_beams,
            backward_beams,
            available_tracks,
            c_arc,
            c_trans,
            bottleneck_results,
            graph_data,
            candidate_orchestrator,
            target_arc,
            track_pool,
        )
        playlist[midpoint] = anchor
        available_tracks.remove(anchor)
        self._diagnostics["anchors_selected"] += 1
        if anchor in set(
            int(track) for track in bottleneck_results["bottleneck_track_indices"]
        ):
            self._diagnostics["selected_bottleneck_anchors"] += 1
        self._anchor_score_total += score
        self._record_anchor(
            left_pos,
            right_pos,
            left_track,
            right_track,
            midpoint,
            anchor,
            left_neighbor,
            right_neighbor,
            score,
            selection_details,
            c_arc,
            c_trans,
            bottleneck_results,
            track_pool,
        )
        self.recursion_trace.append(
            {
                "depth": depth,
                "left_pos": left_pos,
                "right_pos": right_pos,
                "interval_size": right_pos - left_pos + 1,
                "empty_count": empty_count,
                "midpoint": midpoint,
                "is_base_case": False,
                "forward_positions_start": (
                    forward_positions[0] if forward_positions else None
                ),
                "forward_positions_end": (
                    forward_positions[-1] if forward_positions else None
                ),
                "forward_positions_count": len(forward_positions),
                "backward_positions_start": (
                    backward_positions[0] if backward_positions else None
                ),
                "backward_positions_end": (
                    backward_positions[-1] if backward_positions else None
                ),
                "backward_positions_count": len(backward_positions),
                "selected_anchor_position": midpoint,
                "selected_anchor_track": anchor,
                "left_child_interval": [left_pos, midpoint],
                "right_child_interval": [midpoint, right_pos],
            }
        )
        self._solve_interval(
            playlist,
            left_pos,
            midpoint,
            available_tracks,
            c_arc,
            c_trans,
            bottleneck_results,
            graph_data,
            candidate_orchestrator,
            target_arc,
            track_pool,
            depth + 1,
        )
        self._solve_interval(
            playlist,
            midpoint,
            right_pos,
            available_tracks,
            c_arc,
            c_trans,
            bottleneck_results,
            graph_data,
            candidate_orchestrator,
            target_arc,
            track_pool,
            depth + 1,
        )

    def generate(
        self,
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        bottleneck_results: dict,
        graph_data: TransitionGraphData | None = None,
        candidate_orchestrator: CandidateOrchestrator | None = None,
        target_arc: np.ndarray | None = None,
        track_pool: pd.DataFrame | None = None,
        start_index: int | None = None,
        end_index: int | None = None,
        transition_graph: dict[int, list[int]] | None = None,
    ) -> list[int]:
        """Return a complete exact-once BIBS playlist."""
        del transition_graph
        if graph_data is None:
            graph_data = TransitionGraphBuilder(
                TransitionGraphConfig()
            ).build_transition_graph_data(c_trans)
        if candidate_orchestrator is None:
            candidate_orchestrator = CandidateOrchestrator(
                CandidateOrchestratorConfig()
            )
        if track_pool is None or start_index is None or end_index is None:
            raise ValueError("track_pool, start_index, and end_index are required.")
        if target_arc is None:
            target_arc = np.linspace(
                float(track_pool.iloc[start_index]["EV_score"]),
                float(track_pool.iloc[end_index]["EV_score"]),
                len(track_pool),
            )
        number_of_tracks = self._validate_inputs(
            c_arc,
            c_trans,
            bottleneck_results,
            graph_data,
            candidate_orchestrator,
            target_arc,
            track_pool,
            start_index,
            end_index,
        )
        self._reset_state()
        self._ev_scores = pd.to_numeric(
            track_pool["EV_score"],
            errors="raise",
        ).to_numpy(dtype=float)
        self._camelot_codes = track_pool["camelot"].astype(str).to_numpy()
        if "camelot_number" in track_pool.columns:
            self._camelot_numbers = (
                track_pool["camelot_number"].astype(str).to_numpy()
            )
        else:
            self._camelot_numbers = np.asarray(
                [
                    "".join(character for character in code if character.isdigit())
                    for code in self._camelot_codes
                ]
            )
        self._diagnostics["graph_used"] = True
        self._diagnostics["bottleneck_results_used"] = True
        playlist: list[int | None] = [None] * number_of_tracks
        playlist[0] = int(start_index)
        playlist[-1] = int(end_index)
        available_tracks = set(range(number_of_tracks)).difference(
            {int(start_index), int(end_index)}
        )
        self._solve_interval(
            playlist,
            0,
            number_of_tracks - 1,
            available_tracks,
            c_arc,
            c_trans,
            bottleneck_results,
            graph_data,
            candidate_orchestrator,
            np.asarray(target_arc, dtype=float),
            track_pool,
            depth=0,
        )
        if available_tracks or any(track is None for track in playlist):
            raise ValueError("BIBS failed to place every track exactly once.")
        result = [int(track) for track in playlist]
        if (
            len(result) != number_of_tracks
            or len(set(result)) != number_of_tracks
            or result[0] != start_index
            or result[-1] != end_index
        ):
            raise ValueError("BIBS final playlist failed exact-once validation.")
        return result

    def get_diagnostics(self) -> dict:
        """Return BIBS internal search diagnostics."""
        diagnostics = dict(self._diagnostics)
        anchors_selected = int(diagnostics["anchors_selected"])
        decision_count = len(self.decision_trace)
        recursion_base_case_count = sum(
            1 for record in self.recursion_trace if record["is_base_case"]
        )
        recursion_non_base_case_count = (
            len(self.recursion_trace) - recursion_base_case_count
        )
        diagnostics["recursion_trace"] = [dict(record) for record in self.recursion_trace]
        diagnostics["recursion_trace_count"] = len(self.recursion_trace)
        diagnostics["recursion_base_case_count"] = recursion_base_case_count
        diagnostics["recursion_non_base_case_count"] = recursion_non_base_case_count
        diagnostics["max_observed_recursion_depth"] = (
            max((record["depth"] for record in self.recursion_trace), default=0)
        )
        diagnostics["beam_trace"] = [dict(record) for record in self.beam_trace]
        diagnostics["beam_trace_count"] = len(self.beam_trace)
        diagnostics["forward_beam_trace_count"] = sum(
            1 for record in self.beam_trace if record["direction"] == "forward"
        )
        diagnostics["backward_beam_trace_count"] = sum(
            1 for record in self.beam_trace if record["direction"] == "backward"
        )
        diagnostics["max_beams_after_pruning"] = max(
            (record["beams_after_pruning"] for record in self.beam_trace),
            default=0,
        )
        diagnostics["max_expanded_candidate_count"] = max(
            (record["expanded_candidate_count"] for record in self.beam_trace),
            default=0,
        )
        diagnostics["max_positions_in_beam_call"] = max(
            (record["total_positions_in_call"] for record in self.beam_trace),
            default=0,
        )
        anchor_trace_count = len(self.anchor_selection_trace)
        diagnostics["anchor_selection_trace"] = [
            dict(record) for record in self.anchor_selection_trace
        ]
        diagnostics["anchor_selection_trace_count"] = anchor_trace_count

        def average_anchor_trace_value(name: str) -> float:
            return (
                float(
                    sum(
                        float(record[name])
                        for record in self.anchor_selection_trace
                    )
                )
                / anchor_trace_count
                if anchor_trace_count
                else 0.0
            )

        diagnostics["average_selected_anchor_score"] = (
            average_anchor_trace_value("selected_anchor_score")
        )
        diagnostics["average_anchor_arc_component"] = average_anchor_trace_value(
            "arc_component"
        )
        diagnostics["average_anchor_transition_component"] = (
            average_anchor_trace_value("transition_component")
        )
        diagnostics["average_anchor_bottleneck_component"] = (
            average_anchor_trace_value("bottleneck_component")
        )
        diagnostics["average_anchor_balance_component"] = (
            average_anchor_trace_value("balance_component")
        )
        diagnostics["max_anchor_score_gap_from_best"] = max(
            (
                float(record["score_gap_from_best"])
                for record in self.anchor_selection_trace
            ),
            default=0.0,
        )
        diagnostics["average_anchor_candidate_count"] = average_anchor_trace_value(
            "candidate_count"
        )
        diagnostics["selected_anchor_from_graph_count"] = sum(
            1
            for record in self.anchor_selection_trace
            if record["whether_selected_candidate_from_graph"]
        )
        diagnostics["selected_anchor_from_candidate_set_count"] = sum(
            1
            for record in self.anchor_selection_trace
            if record["whether_selected_candidate_from_candidate_set"]
        )
        diagnostics["selected_anchor_from_bottleneck_count"] = sum(
            1
            for record in self.anchor_selection_trace
            if record["whether_selected_candidate_from_bottleneck"]
        )
        diagnostics["selected_anchor_from_bridge_count"] = sum(
            1
            for record in self.anchor_selection_trace
            if record["whether_selected_candidate_from_bridge"]
        )
        diagnostics["average_anchor_score"] = (
            self._anchor_score_total / anchors_selected if anchors_selected else 0.0
        )
        diagnostics["major_decisions_recorded"] = decision_count
        base_cases = int(diagnostics["base_cases_solved"])
        base_case_trace_count = len(self.base_case_trace)
        diagnostics["base_case_trace"] = [
            dict(record) for record in self.base_case_trace
        ]
        diagnostics["base_case_trace_count"] = base_case_trace_count
        diagnostics["average_base_case_candidate_count"] = (
            float(diagnostics["base_case_candidate_count_total"]) / base_cases
            if base_cases
            else 0.0
        )
        diagnostics["average_base_case_permutation_count"] = (
            float(
                sum(
                    record["permutation_count_tried"]
                    for record in self.base_case_trace
                )
            )
            / base_case_trace_count
            if base_case_trace_count
            else 0.0
        )
        diagnostics["average_base_case_local_score"] = (
            float(
                sum(record["local_score_total"] for record in self.base_case_trace)
            )
            / base_case_trace_count
            if base_case_trace_count
            else 0.0
        )
        diagnostics["average_base_case_transition_component"] = (
            float(
                sum(
                    record["transition_component"]
                    for record in self.base_case_trace
                )
            )
            / base_case_trace_count
            if base_case_trace_count
            else 0.0
        )
        diagnostics["max_base_case_transition_component"] = max(
            (
                float(record["transition_component"])
                for record in self.base_case_trace
            ),
            default=0.0,
        )
        diagnostics["max_base_case_candidate_count"] = max(
            (record["candidate_count"] for record in self.base_case_trace),
            default=0,
        )
        diagnostics["max_base_case_permutation_count"] = max(
            (
                record["permutation_count_tried"]
                for record in self.base_case_trace
            ),
            default=0,
        )
        for component in self.SCORE_COMPONENT_NAMES:
            diagnostics[f"average_{component}_per_decision"] = (
                float(diagnostics[f"total_{component}"]) / decision_count
                if decision_count
                else 0.0
            )
        diagnostics["average_normalized_arc_component"] = diagnostics[
            "average_arc_component_per_decision"
        ]
        diagnostics["average_normalized_transition_component"] = diagnostics[
            "average_transition_component_per_decision"
        ]
        return diagnostics
