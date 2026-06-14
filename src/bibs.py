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
from src.cost_functions import energy_momentum_penalty
from src.transition_graph import (
    TransitionGraphBuilder,
    TransitionGraphConfig,
    TransitionGraphData,
)


@dataclass(frozen=True)
class BIBSConfig:
    """Configuration for Bottleneck-Guided Bidirectional Beam Search."""

    beam_width: int = 8
    base_case_size: int = 4
    max_recursion_depth: int = 30
    arc_weight: float = 1.0
    transition_weight: float = 1.0
    bottleneck_weight: float = 0.6
    location_bottleneck_weight: float = 0.5
    energy_momentum_weight: float = 1.0
    harmonic_diversity_weight: float = 0.4
    anchor_arc_weight: float = 1.0
    anchor_transition_weight: float = 1.2
    anchor_bottleneck_weight: float = 0.6
    anchor_location_bottleneck_weight: float = 0.5
    anchor_energy_momentum_weight: float = 1.0
    base_arc_weight: float = 0.4
    base_transition_weight: float = 1.4
    base_energy_momentum_weight: float = 1.0
    harmonic_window: int = 8
    max_same_camelot_in_window: int = 3
    max_same_camelot_number_in_window: int = 4
    harmonic_stagnation_penalty: float = 0.3
    high_energy_threshold: float = 0.65
    allowed_ev_drop: float = 0.08
    ev_drop_weight: float = 2.0
    max_preferred_camelot_jump: int = 2
    large_camelot_jump_penalty: float = 0.8
    extreme_camelot_jump_penalty: float = 1.5
    enable_harmonic_motion_penalty: bool = True
    high_arc_threshold: float = 0.65
    max_allowed_arc_deviation_high_energy: float = 0.15
    high_energy_arc_penalty_weight: float = 3.0
    high_energy_drop_penalty_weight: float = 3.0
    late_position_fraction: float = 0.70
    late_low_ev_penalty_weight: float = 3.0
    feasibility_penalty_weight: float = 1.0
    min_candidates_per_position: int = 4
    base_case_extra_candidates_per_position: int = 4
    base_case_max_candidates: int = 12
    bad_transition_threshold: float = 1.5
    bad_transition_penalty_weight: float = 1.2
    base_high_energy_penalty_weight: float = 3.0


@dataclass(frozen=True)
class _BeamItem:
    path: tuple[int, ...]
    used: frozenset[int]
    last_track: int
    score: float
    arc_component: float = 0.0
    transition_component: float = 0.0
    bottleneck_component: float = 0.0
    location_bottleneck_component: float = 0.0
    energy_momentum_component: float = 0.0
    harmonic_diversity_component: float = 0.0
    harmonic_motion_component: float = 0.0
    feasibility_component: float = 0.0


class BIBS:
    """Construct an exact-once playlist with recursive bidirectional search."""

    SCORE_COMPONENT_NAMES = (
        "arc_component",
        "transition_component",
        "bottleneck_component",
        "location_bottleneck_component",
        "energy_momentum_component",
        "harmonic_diversity_component",
        "harmonic_motion_component",
        "feasibility_component",
    )

    DIAGNOSTIC_NAMES = (
        "intervals_solved",
        "anchors_selected",
        "base_cases_solved",
        "forward_beam_expansions",
        "backward_beam_expansions",
        "candidate_orchestrator_calls",
        "selected_bottleneck_anchors",
        "energy_momentum_penalties_applied",
        "harmonic_penalties_applied",
        "harmonic_motion_penalties_applied",
        "base_case_forced_assignment_count",
        "high_energy_arc_penalties_applied",
        "late_low_ev_penalties_applied",
        "feasibility_penalties_applied",
        "anchors_with_arc_cost_above_0_25",
        "anchors_with_arc_cost_above_0_35",
        "final_repair_swaps",
    )

    def __init__(self, config: BIBSConfig) -> None:
        self.config = config
        self.anchor_history: list[dict] = []
        self.decision_trace: list[dict] = []
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
            "base_case_size",
            "max_recursion_depth",
            "harmonic_window",
            "max_same_camelot_in_window",
            "max_same_camelot_number_in_window",
            "max_preferred_camelot_jump",
            "min_candidates_per_position",
            "base_case_extra_candidates_per_position",
            "base_case_max_candidates",
        )
        for name in integer_fields:
            if getattr(self.config, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        weight_fields = (
            "arc_weight",
            "transition_weight",
            "bottleneck_weight",
            "location_bottleneck_weight",
            "energy_momentum_weight",
            "harmonic_diversity_weight",
            "anchor_arc_weight",
            "anchor_transition_weight",
            "anchor_bottleneck_weight",
            "anchor_location_bottleneck_weight",
            "anchor_energy_momentum_weight",
            "base_arc_weight",
            "base_transition_weight",
            "base_energy_momentum_weight",
            "harmonic_stagnation_penalty",
            "ev_drop_weight",
            "large_camelot_jump_penalty",
            "extreme_camelot_jump_penalty",
            "high_energy_arc_penalty_weight",
            "high_energy_drop_penalty_weight",
            "late_low_ev_penalty_weight",
            "feasibility_penalty_weight",
            "bad_transition_penalty_weight",
            "base_high_energy_penalty_weight",
        )
        for name in weight_fields:
            if getattr(self.config, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.config.allowed_ev_drop < 0:
            raise ValueError("allowed_ev_drop must be non-negative.")
        for name in (
            "high_arc_threshold",
            "max_allowed_arc_deviation_high_energy",
            "late_position_fraction",
        ):
            value = getattr(self.config, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.config.bad_transition_threshold < 0:
            raise ValueError("bad_transition_threshold must be non-negative.")
        if self.config.base_case_max_candidates < self.config.base_case_size:
            raise ValueError(
                "base_case_max_candidates cannot be smaller than base_case_size."
            )

    def _reset_state(self) -> None:
        self.anchor_history = []
        self.decision_trace = []
        self._diagnostics = {name: 0 for name in self.DIAGNOSTIC_NAMES}
        self._diagnostics["graph_used"] = False
        self._diagnostics["bottleneck_results_used"] = False
        self._diagnostics["base_case_candidate_count_total"] = 0.0
        self._diagnostics["worst_base_case_transition_component"] = 0.0
        self._diagnostics["feasibility_penalty_total"] = 0.0
        self._diagnostics["late_energy_penalty_total"] = 0.0
        self._diagnostics["late_energy_penalty_count"] = 0
        self._diagnostics["final_repair_transition_improvement"] = 0.0
        self._diagnostics["final_repair_harmonic_increase"] = 0.0
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
    def _normalize_energy_momentum(value: float) -> float:
        """Scale energy-momentum penalties into a visible bounded range."""
        return float(np.clip(float(value), 0.0, 1.0))

    @staticmethod
    def _normalize_bottleneck_score(value: float) -> float:
        """Scale bottleneck difficulty into a useful bounded bonus range."""
        return float(np.clip(float(value), 0.0, 1.0))

    def _bottleneck_bonus(
        self,
        score: float,
        arc_cost: float,
        weight: float,
    ) -> float:
        """Return a fit-gated negative bonus for a difficult track."""
        if arc_cost <= 0.20:
            fit_multiplier = 1.0
        elif arc_cost <= 0.35:
            fit_multiplier = 0.5
        else:
            fit_multiplier = 0.0
        return float(
            -weight
            * fit_multiplier
            * self._normalize_bottleneck_score(score)
        )

    def _late_energy_penalty(
        self,
        candidate: int,
        previous_track: int,
        position: int,
        target_arc: np.ndarray,
        track_pool: pd.DataFrame,
        c_arc: np.ndarray,
    ) -> float:
        """Penalize low-EV or poor-arc placements in late/high-energy regions."""
        del track_pool
        target_value = float(target_arc[position])
        is_late = position >= int(len(target_arc) * self.config.late_position_fraction)
        if not is_late and target_value < self.config.high_arc_threshold:
            return 0.0

        arc_excess = max(
            0.0,
            float(c_arc[candidate, position])
            - self.config.max_allowed_arc_deviation_high_energy,
        )
        low_ev_excess = max(
            0.0,
            target_value
            - float(self._ev_scores[candidate])
            - self.config.max_allowed_arc_deviation_high_energy,
        )
        drop_excess = max(
            0.0,
            float(self._ev_scores[previous_track])
            - float(self._ev_scores[candidate])
            - self.config.allowed_ev_drop,
        )
        if arc_excess > 0:
            self._diagnostics["high_energy_arc_penalties_applied"] += 1
        if low_ev_excess > 0:
            self._diagnostics["late_low_ev_penalties_applied"] += 1
        penalty = (
            self.config.high_energy_arc_penalty_weight * arc_excess
            + self.config.late_low_ev_penalty_weight * low_ev_excess
            + self.config.high_energy_drop_penalty_weight * drop_excess
        )
        if penalty > 0:
            self._diagnostics["late_energy_penalty_total"] += float(penalty)
            self._diagnostics["late_energy_penalty_count"] += 1
        return float(penalty)

    def _estimate_subinterval_feasibility(
        self,
        left_pos: int,
        right_pos: int,
        available_tracks: set[int],
        c_arc: np.ndarray,
        bottleneck_results: dict,
        target_arc: np.ndarray,
        track_pool: pd.DataFrame,
    ) -> tuple[float, dict[str, float]]:
        """Estimate whether remaining tracks can reasonably fill an interval."""
        del track_pool
        positions = list(range(left_pos + 1, right_pos))
        if not positions:
            return 0.0
        available = np.asarray(sorted(available_tracks), dtype=int)
        if len(available) < len(positions):
            return float(len(positions) - len(available) + 1)

        candidate_sets = bottleneck_results["candidate_sets"]
        penalty = 0.0
        for position in positions:
            candidate_count = sum(
                candidate in available_tracks
                for candidate in candidate_sets[position]
            )
            penalty += max(
                0,
                self.config.min_candidates_per_position - candidate_count,
            ) / self.config.min_candidates_per_position
            closest_arc = float(np.min(c_arc[available, position]))
            penalty += max(
                0.0,
                closest_arc - self.config.max_allowed_arc_deviation_high_energy,
            )
            if (
                position >= int(len(target_arc) * self.config.late_position_fraction)
                or target_arc[position] >= self.config.high_arc_threshold
            ):
                high_enough = np.count_nonzero(
                    self._ev_scores[available]
                    >= target_arc[position]
                    - self.config.max_allowed_arc_deviation_high_energy
                )
                penalty += max(0, 1 - high_enough)
        return float(penalty / len(positions))

    @staticmethod
    def _local_transition_total(
        playlist: list[int],
        positions: set[int],
        c_trans: np.ndarray,
    ) -> float:
        edge_positions = {
            edge
            for position in positions
            for edge in (position - 1, position)
            if 0 <= edge < len(playlist) - 1
        }
        return float(
            sum(
                c_trans[playlist[edge], playlist[edge + 1]]
                for edge in edge_positions
            )
        )

    def _large_drop_severity(
        self,
        playlist: list[int],
        positions: set[int],
    ) -> float:
        edge_positions = {
            edge
            for position in positions
            for edge in (position - 1, position)
            if 0 <= edge < len(playlist) - 1
        }
        return float(
            sum(
                max(
                    0.0,
                    self._ev_scores[playlist[edge]]
                    - self._ev_scores[playlist[edge + 1]]
                    - 0.12,
                )
                for edge in edge_positions
            )
        )

    def _harmonic_motion_severity(
        self,
        playlist: list[int],
        positions: set[int],
    ) -> float:
        edge_positions = {
            edge
            for position in positions
            for edge in (position - 1, position)
            if 0 <= edge < len(playlist) - 1
        }
        severity = 0.0
        for edge in edge_positions:
            source = int(float(self._camelot_numbers[playlist[edge]]))
            target = int(float(self._camelot_numbers[playlist[edge + 1]]))
            direct = abs(source - target)
            distance = min(direct, 12 - direct)
            severity += max(0, distance - self.config.max_preferred_camelot_jump)
        return float(severity)

    def _final_repair(
        self,
        playlist: list[int],
        c_arc: np.ndarray,
        c_trans: np.ndarray,
    ) -> list[int]:
        """Deterministically improve final adjacency without worsening arc or EV drops."""
        repaired = list(playlist)
        total_improvement = 0.0
        harmonic_increase_used = 0.0
        for _ in range(2):
            improved = False
            for first in range(1, len(repaired) - 1):
                best_swap: tuple[float, int, float] | None = None
                for second in range(first + 1, len(repaired) - 1):
                    positions = {first, second}
                    old_transition = self._local_transition_total(
                        repaired,
                        positions,
                        c_trans,
                    )
                    old_drop = self._large_drop_severity(repaired, positions)
                    old_harmonic = self._harmonic_motion_severity(
                        repaired,
                        positions,
                    )
                    first_track = repaired[first]
                    second_track = repaired[second]
                    old_arc = (
                        c_arc[first_track, first] ** 2
                        + c_arc[second_track, second] ** 2
                    )
                    new_arc = (
                        c_arc[second_track, first] ** 2
                        + c_arc[first_track, second] ** 2
                    )
                    if new_arc > old_arc + 1e-12:
                        continue
                    repaired[first], repaired[second] = second_track, first_track
                    new_transition = self._local_transition_total(
                        repaired,
                        positions,
                        c_trans,
                    )
                    new_drop = self._large_drop_severity(repaired, positions)
                    new_harmonic = self._harmonic_motion_severity(
                        repaired,
                        positions,
                    )
                    repaired[first], repaired[second] = first_track, second_track
                    improvement = old_transition - new_transition
                    harmonic_increase = max(0.0, new_harmonic - old_harmonic)
                    if (
                        improvement <= 1e-12
                        or new_drop > old_drop + 1e-12
                    ):
                        continue
                    repair_score = improvement - 0.15 * harmonic_increase
                    key = (float(repair_score), float(improvement), -second)
                    if best_swap is None:
                        best_key = None
                    else:
                        best_key = (
                            best_swap[0] - 0.15 * best_swap[2],
                            best_swap[0],
                            -best_swap[1],
                        )
                    if best_key is None or key > best_key:
                        best_swap = (
                            float(improvement),
                            second,
                            float(harmonic_increase),
                        )
                if best_swap is None:
                    continue
                improvement, second, harmonic_increase = best_swap
                repaired[first], repaired[second] = repaired[second], repaired[first]
                total_improvement += improvement
                harmonic_increase_used += harmonic_increase
                self._diagnostics["final_repair_swaps"] += 1
                improved = True
            if not improved:
                break
        self._diagnostics["final_repair_transition_improvement"] = total_improvement
        self._diagnostics["final_repair_harmonic_increase"] = harmonic_increase_used
        return repaired

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

    def _energy_penalty(
        self,
        previous_track: int,
        candidate: int,
        target_value: float,
        track_pool: pd.DataFrame,
    ) -> float:
        penalty = energy_momentum_penalty(
            float(self._ev_scores[previous_track]),
            float(self._ev_scores[candidate]),
            float(target_value),
            high_energy_threshold=self.config.high_energy_threshold,
            allowed_drop=self.config.allowed_ev_drop,
            drop_weight=self.config.ev_drop_weight,
        )
        if penalty > 0:
            self._diagnostics["energy_momentum_penalties_applied"] += 1
        return penalty

    def _harmonic_diversity_penalty(
        self,
        candidate: int,
        recent_tracks: list[int] | tuple[int, ...],
        track_pool: pd.DataFrame,
    ) -> float:
        context = list(recent_tracks)[-(self.config.harmonic_window - 1) :]
        candidate_code = self._camelot_codes[candidate]
        candidate_number = self._camelot_numbers[candidate]
        numbers = [self._camelot_numbers[track] for track in context]
        codes = [self._camelot_codes[track] for track in context]
        code_count = codes.count(candidate_code) + 1
        number_count = numbers.count(candidate_number) + 1
        excess = max(0, code_count - self.config.max_same_camelot_in_window)
        excess += max(
            0,
            number_count - self.config.max_same_camelot_number_in_window,
        )
        immediate_run = 0
        for code in reversed(codes):
            if code != candidate_code:
                break
            immediate_run += 1
        excess += max(0, immediate_run - self.config.max_same_camelot_in_window + 1)
        penalty = self.config.harmonic_stagnation_penalty * excess
        if penalty > 0:
            self._diagnostics["harmonic_penalties_applied"] += 1
        return float(penalty)

    def _harmonic_motion_penalty(
        self,
        source: int,
        target: int,
    ) -> float:
        """Return a soft penalty for large circular Camelot-number jumps."""
        if not self.config.enable_harmonic_motion_penalty:
            return 0.0
        source_number = int(float(self._camelot_numbers[source]))
        target_number = int(float(self._camelot_numbers[target]))
        direct = abs(source_number - target_number)
        distance = min(direct, 12 - direct)
        excess = distance - self.config.max_preferred_camelot_jump
        if excess <= 0:
            return 0.0
        penalty = self.config.large_camelot_jump_penalty * excess
        if distance >= 5:
            penalty += self.config.extreme_camelot_jump_penalty * (distance - 4)
        self._diagnostics["harmonic_motion_penalties_applied"] += 1
        return float(penalty)

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
        beams = [
            _BeamItem(
                path=(),
                used=frozenset(),
                last_track=boundary_track,
                score=0.0,
            )
        ]
        song_scores = np.asarray(bottleneck_results["song_bottleneck_scores"])
        location_scores = np.asarray(
            bottleneck_results["location_bottleneck_scores"]
        )
        for position in positions:
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
                        motion = self._harmonic_motion_penalty(
                            beam.last_track,
                            candidate,
                        )
                        energy = self._energy_penalty(
                            beam.last_track,
                            candidate,
                            target_arc[position],
                            track_pool,
                        )
                    else:
                        transition = c_trans[candidate, beam.last_track]
                        motion = self._harmonic_motion_penalty(
                            candidate,
                            beam.last_track,
                        )
                        energy = self._energy_penalty(
                            candidate,
                            beam.last_track,
                            target_arc[min(position + 1, len(target_arc) - 1)],
                            track_pool,
                        )
                    harmonic = self._harmonic_diversity_penalty(
                        candidate,
                        beam.path,
                        track_pool,
                    )
                    late_energy = self._late_energy_penalty(
                        candidate,
                        beam.last_track,
                        position,
                        target_arc,
                        track_pool,
                        c_arc,
                    )
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
                        c_arc[candidate, position],
                        self.config.bottleneck_weight,
                    )
                    location_bottleneck_component = (
                        self.config.location_bottleneck_weight
                        * self._normalize_bottleneck_score(location_scores[position])
                        * normalized_arc
                    )
                    energy_momentum_component = (
                        self.config.energy_momentum_weight
                        * self._normalize_energy_momentum(energy)
                        + late_energy
                    )
                    harmonic_diversity_component = (
                        self.config.harmonic_diversity_weight * harmonic
                    )
                    harmonic_motion_component = motion
                    feasibility_component = 0.0
                    local_score = (
                        arc_component
                        + transition_component
                        + bottleneck_component
                        + location_bottleneck_component
                        + energy_momentum_component
                        + harmonic_diversity_component
                        + harmonic_motion_component
                        + feasibility_component
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
                            location_bottleneck_component=float(
                                beam.location_bottleneck_component
                                + location_bottleneck_component
                            ),
                            energy_momentum_component=float(
                                beam.energy_momentum_component
                                + energy_momentum_component
                            ),
                            harmonic_diversity_component=float(
                                beam.harmonic_diversity_component
                                + harmonic_diversity_component
                            ),
                            harmonic_motion_component=float(
                                beam.harmonic_motion_component
                                + harmonic_motion_component
                            ),
                            feasibility_component=float(
                                beam.feasibility_component
                                + feasibility_component
                            ),
                        )
                    )
            if direction == "forward":
                self._diagnostics["forward_beam_expansions"] += len(expanded)
            else:
                self._diagnostics["backward_beam_expansions"] += len(expanded)
            if not expanded:
                return beams
            expanded.sort(key=lambda beam: (beam.score, beam.path))
            beams = expanded[: self.config.beam_width]
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
        location_scores = np.asarray(
            bottleneck_results["location_bottleneck_scores"]
        )
        feasibility_by_anchor: dict[int, float] = {}
        for anchor in anchor_candidates:
            remaining = available_tracks.difference({anchor})
            feasibility_by_anchor[anchor] = (
                self._estimate_subinterval_feasibility(
                    interval_left_pos,
                    midpoint,
                    remaining,
                    c_arc,
                    bottleneck_results,
                    target_arc,
                    track_pool,
                )
                + self._estimate_subinterval_feasibility(
                    midpoint,
                    interval_right_pos,
                    remaining,
                    c_arc,
                    bottleneck_results,
                    target_arc,
                    track_pool,
                )
            )
        best: tuple[float, int, int, int, dict[str, float]] | None = None
        best_scores_by_candidate: dict[int, float] = {}
        for forward in forward_beams:
            for backward in backward_beams:
                if not forward.used.isdisjoint(backward.used):
                    continue
                for anchor in anchor_candidates:
                    if anchor in forward.used or anchor in backward.used:
                        continue
                    left_neighbor = forward.last_track
                    right_neighbor = backward.last_track
                    energy = self._energy_penalty(
                        left_neighbor,
                        anchor,
                        target_arc[midpoint],
                        track_pool,
                    )
                    harmonic = self._harmonic_diversity_penalty(
                        anchor,
                        forward.path,
                        track_pool,
                    )
                    motion = self._harmonic_motion_penalty(
                        left_neighbor,
                        anchor,
                    ) + self._harmonic_motion_penalty(
                        anchor,
                        right_neighbor,
                    )
                    late_energy = self._late_energy_penalty(
                        anchor,
                        left_neighbor,
                        midpoint,
                        target_arc,
                        track_pool,
                        c_arc,
                    )
                    normalized_arc = self._normalize_arc_cost(
                        c_arc[anchor, midpoint]
                    )
                    arc_component = (
                        forward.arc_component
                        + backward.arc_component
                        + self.config.anchor_arc_weight * normalized_arc
                    )
                    left_transition_component = (
                        self.config.anchor_transition_weight
                        * self._normalize_transition_cost(
                            c_trans[left_neighbor, anchor]
                        )
                    )
                    right_transition_component = (
                        self.config.anchor_transition_weight
                        * self._normalize_transition_cost(
                            c_trans[anchor, right_neighbor]
                        )
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
                            c_arc[anchor, midpoint],
                            self.config.anchor_bottleneck_weight,
                        )
                    )
                    location_bottleneck_component = (
                        forward.location_bottleneck_component
                        + backward.location_bottleneck_component
                        + self.config.anchor_location_bottleneck_weight
                        * self._normalize_bottleneck_score(
                            location_scores[midpoint]
                        )
                        * normalized_arc
                    )
                    energy_momentum_component = (
                        forward.energy_momentum_component
                        + backward.energy_momentum_component
                        + self.config.anchor_energy_momentum_weight
                        * self._normalize_energy_momentum(energy)
                        + late_energy
                    )
                    harmonic_diversity_component = (
                        forward.harmonic_diversity_component
                        + backward.harmonic_diversity_component
                        + self.config.harmonic_diversity_weight * harmonic
                    )
                    harmonic_motion_component = (
                        forward.harmonic_motion_component
                        + backward.harmonic_motion_component
                        + motion
                    )
                    feasibility_component = (
                        forward.feasibility_component
                        + backward.feasibility_component
                        + self.config.feasibility_penalty_weight
                        * feasibility_by_anchor[anchor]
                    )
                    score = (
                        arc_component
                        + transition_component
                        + bottleneck_component
                        + location_bottleneck_component
                        + energy_momentum_component
                        + harmonic_diversity_component
                        + harmonic_motion_component
                        + feasibility_component
                    )
                    components = {
                        "arc_component": float(arc_component),
                        "transition_component": float(transition_component),
                        "bottleneck_component": float(bottleneck_component),
                        "location_bottleneck_component": float(
                            location_bottleneck_component
                        ),
                        "energy_momentum_component": float(
                            energy_momentum_component
                        ),
                        "harmonic_diversity_component": float(
                            harmonic_diversity_component
                        ),
                        "harmonic_motion_component": float(
                            harmonic_motion_component
                        ),
                        "feasibility_component": float(feasibility_component),
                        "feasibility_penalty": float(
                            feasibility_by_anchor[anchor]
                        ),
                        "late_energy_penalty": float(late_energy),
                        "left_transition_component": float(
                            left_transition_component
                        ),
                        "right_transition_component": float(
                            right_transition_component
                        ),
                    }
                    candidate_score = float(score)
                    best_scores_by_candidate[anchor] = min(
                        candidate_score,
                        best_scores_by_candidate.get(anchor, float("inf")),
                    )
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
                * self._normalize_arc_cost(c_arc[anchor, midpoint])
            )
            left_transition_component = float(
                self.config.anchor_transition_weight
                * self._normalize_transition_cost(c_trans[left_track, anchor])
            )
            right_transition_component = float(
                self.config.anchor_transition_weight
                * self._normalize_transition_cost(c_trans[anchor, right_track])
            )
            harmonic_motion_component = float(
                self._harmonic_motion_penalty(left_track, anchor)
                + self._harmonic_motion_penalty(anchor, right_track)
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
                    "harmonic_motion_component": harmonic_motion_component,
                    "left_transition_component": left_transition_component,
                    "right_transition_component": right_transition_component,
                    "feasibility_component": float(
                        self.config.feasibility_penalty_weight
                        * feasibility_by_anchor[anchor]
                    ),
                    "feasibility_penalty": float(feasibility_by_anchor[anchor]),
                    "late_energy_penalty": 0.0,
                }
            )
            score = float(
                arc_component
                + left_transition_component
                + right_transition_component
                + harmonic_motion_component
                + components["feasibility_component"]
            )
            best_scores_by_candidate[anchor] = score
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
        selected_feasibility = float(feasibility_by_anchor[anchor])
        if selected_feasibility > 0:
            self._diagnostics["feasibility_penalties_applied"] += 1
            self._diagnostics["feasibility_penalty_total"] += selected_feasibility
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
        }
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
        location_scores = np.asarray(
            bottleneck_results["location_bottleneck_scores"]
        )
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
            "bs_loc": float(location_scores[position]),
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
    ) -> float:
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
        bad_transition_penalty = self.config.bad_transition_penalty_weight * sum(
            max(0.0, cost - self.config.bad_transition_threshold)
            for cost in raw_transition_costs
        )
        energy_cost = 0.0
        late_energy_cost = 0.0
        harmonic_cost = 0.0
        harmonic_motion_cost = sum(
            self._harmonic_motion_penalty(source, target)
            for source, target in zip(path, path[1:])
        )
        recent: list[int] = [left_track]
        for offset, track in enumerate(sequence):
            previous = recent[-1]
            energy_cost += self._energy_penalty(
                previous,
                track,
                target_arc[positions[offset]],
                track_pool,
            )
            late_energy_cost += self._late_energy_penalty(
                track,
                previous,
                positions[offset],
                target_arc,
                track_pool,
                c_arc,
            )
            harmonic_cost += self._harmonic_diversity_penalty(
                track,
                recent,
                track_pool,
            )
            recent.append(track)
        energy_cost += self._energy_penalty(
            recent[-1],
            right_track,
            target_arc[min(positions[-1] + 1, len(target_arc) - 1)],
            track_pool,
        )
        components = {
            "arc_component": float(self.config.base_arc_weight * arc_cost),
            "transition_component": float(
                self.config.base_transition_weight * transition_cost
                + bad_transition_penalty
            ),
            "bottleneck_component": float(
                sum(
                    self._bottleneck_bonus(
                        song_scores[track],
                        c_arc[track, position],
                        self.config.bottleneck_weight,
                    )
                    for track, position in zip(sequence, positions)
                )
            ),
            "location_bottleneck_component": 0.0,
            "energy_momentum_component": float(
                self.config.base_energy_momentum_weight
                * self._normalize_energy_momentum(energy_cost)
                + (
                    self.config.base_high_energy_penalty_weight
                    / 3.0
                )
                * late_energy_cost
            ),
            "harmonic_diversity_component": float(
                self.config.harmonic_diversity_weight * harmonic_cost
            ),
            "harmonic_motion_component": float(harmonic_motion_cost),
            "feasibility_component": 0.0,
            "bad_transition_penalty": float(bad_transition_penalty),
            "late_energy_penalty": float(late_energy_cost),
        }
        score = (
            components["arc_component"]
            + components["transition_component"]
            + components["bottleneck_component"]
            + components["energy_momentum_component"]
            + components["harmonic_diversity_component"]
            + components["harmonic_motion_component"]
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
        per_position_limit = self.config.base_case_extra_candidates_per_position
        for position in positions:
            arc_order = np.argsort(c_arc[:, position], kind="stable")
            candidate_sources.extend(
                int(track)
                for track in arc_order
                if int(track) in available_tracks
            )
            ev_order = sorted(
                available_tracks,
                key=lambda track: (
                    abs(float(self._ev_scores[track]) - target_arc[position]),
                    track,
                ),
            )
            candidate_sources.extend(ev_order[:per_position_limit])
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
            ev_fit = min(
                abs(float(self._ev_scores[track]) - target_arc[position])
                for position in positions
            )
            boundary_fit = min(
                self._normalize_transition_cost(c_trans[left_track, track]),
                self._normalize_transition_cost(c_trans[track, right_track]),
            )
            return float(arc_fit + ev_fit + 0.25 * boundary_fit), track

        desired_count = min(
            len(available_tracks),
            self.config.base_case_max_candidates,
            len(positions)
            + self.config.base_case_extra_candidates_per_position * len(positions),
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
                "harmonic_component": float(
                    best_components["harmonic_diversity_component"]
                    + best_components["harmonic_motion_component"]
                ),
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
        horizon = self.config.base_case_size
        forward_positions = list(
            range(left_pos + 1, min(midpoint, left_pos + 1 + horizon))
        )
        backward_positions = list(
            range(right_pos - 1, max(midpoint, right_pos - 1 - horizon), -1)
        )
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
        result = self._final_repair(result, c_arc, c_trans)
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
        diagnostics["average_anchor_score"] = (
            self._anchor_score_total / anchors_selected if anchors_selected else 0.0
        )
        diagnostics["major_decisions_recorded"] = decision_count
        base_cases = int(diagnostics["base_cases_solved"])
        diagnostics["average_base_case_candidate_count"] = (
            float(diagnostics["base_case_candidate_count_total"]) / base_cases
            if base_cases
            else 0.0
        )
        diagnostics["average_feasibility_penalty"] = (
            float(diagnostics["feasibility_penalty_total"]) / anchors_selected
            if anchors_selected
            else 0.0
        )
        late_energy_count = int(diagnostics["late_energy_penalty_count"])
        diagnostics["average_late_energy_penalty"] = (
            float(diagnostics["late_energy_penalty_total"]) / late_energy_count
            if late_energy_count
            else 0.0
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
