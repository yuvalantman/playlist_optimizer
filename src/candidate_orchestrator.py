"""Combine transition-graph and bottleneck outputs into candidate pools."""

from dataclasses import dataclass

import numpy as np

from src.bottleneck_detector import validate_bottleneck_results
from src.transition_graph import TransitionGraphData


@dataclass(frozen=True)
class CandidateOrchestratorConfig:
    """Configuration for deterministic candidate-pool orchestration."""

    max_candidates: int = 60
    graph_priority: float = 1.0
    candidate_set_priority: float = 0.8
    bottleneck_priority: float = 0.6
    fallback_priority: float = 0.2
    use_bottlenecks: bool = True
    use_location_bottlenecks: bool = True
    graph_left_weight: float = 1.0
    graph_right_weight: float = 1.0
    bridge_balance_weight: float = 0.5
    balance_weight: float = 0.5
    arc_preview_weight: float = 0.5
    bottleneck_preview_weight: float = 0.2


class CandidateOrchestrator:
    """Build candidate pools without selecting the final playlist track."""

    COUNTER_NAMES = (
        "outgoing_graph_candidates_added",
        "incoming_graph_candidates_added",
        "intersection_anchor_candidates_added",
        "candidate_set_candidates_added",
        "bottleneck_candidates_added",
        "fallback_candidates_added",
        "empty_candidate_pool_count",
        "anchor_intersection_size_total",
        "anchor_union_size_total",
        "anchor_candidates_from_outgoing_only",
        "anchor_candidates_from_incoming_only",
        "anchor_candidates_from_both_sides",
        "anchor_candidates_from_candidate_set",
        "anchor_candidates_from_bottleneck",
        "anchor_candidates_from_fallback",
        "bridge_candidates_added",
        "empty_anchor_candidate_pool_count",
    )

    def __init__(self, config: CandidateOrchestratorConfig) -> None:
        self.config = config
        if self.config.max_candidates <= 0:
            raise ValueError("max_candidates must be positive.")
        for name in (
            "graph_priority",
            "candidate_set_priority",
            "bottleneck_priority",
            "fallback_priority",
            "graph_left_weight",
            "graph_right_weight",
            "bridge_balance_weight",
            "balance_weight",
            "arc_preview_weight",
            "bottleneck_preview_weight",
        ):
            if getattr(self.config, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        self._diagnostics = {name: 0 for name in self.COUNTER_NAMES}
        self._anchor_build_count = 0
        self._anchor_candidate_count_total = 0
        self._last_candidate_sources: dict[int, set[str]] = {}
        self._last_anchor_candidate_sources: dict[int, set[str]] = {}
        self._anchor_top10_left_cost_total = 0.0
        self._anchor_top10_right_cost_total = 0.0
        self._anchor_top10_balance_gap_total = 0.0
        self._anchor_top10_count = 0

    @staticmethod
    def _validate_inputs(
        available_tracks: set[int],
        graph_data: TransitionGraphData,
        bottleneck_results: dict,
        c_arc: np.ndarray,
    ) -> tuple[int, int]:
        if not isinstance(c_arc, np.ndarray) or c_arc.ndim != 2:
            raise ValueError("c_arc must be a 2D numpy array.")
        if not np.issubdtype(c_arc.dtype, np.number) or not np.isfinite(c_arc).all():
            raise ValueError("c_arc must contain finite numeric values.")
        number_of_tracks, playlist_length = c_arc.shape
        if set(graph_data.outgoing_neighbors) != set(range(number_of_tracks)):
            raise ValueError("graph_data must contain every outgoing graph node.")
        if set(graph_data.incoming_neighbors) != set(range(number_of_tracks)):
            raise ValueError("graph_data must contain every incoming graph node.")
        if any(
            not isinstance(track, (int, np.integer))
            or not 0 <= int(track) < number_of_tracks
            for track in available_tracks
        ):
            raise ValueError("available_tracks contains invalid track indices.")
        validate_bottleneck_results(
            bottleneck_results,
            number_of_tracks,
            playlist_length,
        )
        return number_of_tracks, playlist_length

    def _build_candidates(
        self,
        sources: list[tuple[str, list[int], float]],
        available_tracks: set[int],
    ) -> list[int]:
        candidates: list[int] = []
        seen: set[int] = set()
        candidate_sources: dict[int, set[str]] = {}
        source_labels = {
            "outgoing_graph_candidates_added": "outgoing_graph",
            "incoming_graph_candidates_added": "incoming_graph",
            "intersection_anchor_candidates_added": "intersection",
            "candidate_set_candidates_added": "candidate_set",
            "bottleneck_candidates_added": "bottleneck",
            "fallback_candidates_added": "fallback",
        }
        ordered_sources = sorted(
            enumerate(sources),
            key=lambda item: (-item[1][2], item[0]),
        )
        for _, (counter_name, source, _) in ordered_sources:
            source_label = source_labels[counter_name]
            for raw_track in source:
                track = int(raw_track)
                if track in available_tracks:
                    candidate_sources.setdefault(track, set()).add(source_label)
        for _, (counter_name, source, _) in ordered_sources:
            for raw_track in source:
                track = int(raw_track)
                if track not in available_tracks or track in seen:
                    continue
                candidates.append(track)
                seen.add(track)
                self._diagnostics[counter_name] += 1
                if len(candidates) >= self.config.max_candidates:
                    self._last_candidate_sources = {
                        track: candidate_sources[track] for track in candidates
                    }
                    return candidates
        if not candidates:
            self._diagnostics["empty_candidate_pool_count"] += 1
        self._last_candidate_sources = {
            track: candidate_sources[track] for track in candidates
        }
        return candidates

    @staticmethod
    def _validate_c_trans(c_trans: np.ndarray, number_of_tracks: int) -> None:
        if (
            not isinstance(c_trans, np.ndarray)
            or c_trans.shape != (number_of_tracks, number_of_tracks)
            or not np.issubdtype(c_trans.dtype, np.number)
            or not np.isfinite(c_trans).all()
        ):
            raise ValueError("c_trans must be a finite square matrix matching c_arc.")

    @staticmethod
    def _bottleneck_order(bottleneck_results: dict) -> list[int]:
        scores = np.asarray(bottleneck_results["song_bottleneck_scores"])
        bottleneck_tracks = set(
            int(track) for track in bottleneck_results["bottleneck_track_indices"]
        )
        return [
            int(track)
            for track in np.argsort(-scores, kind="stable")
            if int(track) in bottleneck_tracks
        ]

    @staticmethod
    def _fallback_for_position(c_arc: np.ndarray, position: int) -> list[int]:
        return np.argsort(c_arc[:, position], kind="stable").tolist()

    def build_forward_candidates(
        self,
        last_track: int,
        position: int,
        available_tracks: set[int],
        graph_data: TransitionGraphData,
        bottleneck_results: dict,
        c_arc: np.ndarray,
    ) -> list[int]:
        """Build candidates suitable after the current forward boundary."""
        number_of_tracks, playlist_length = self._validate_inputs(
            available_tracks, graph_data, bottleneck_results, c_arc
        )
        if not 0 <= last_track < number_of_tracks or not 0 <= position < playlist_length:
            raise ValueError("last_track and position must be valid.")
        sources = [
            (
                "outgoing_graph_candidates_added",
                graph_data.outgoing_neighbors[last_track],
                self.config.graph_priority,
            ),
            (
                "candidate_set_candidates_added",
                bottleneck_results["candidate_sets"][position],
                self.config.candidate_set_priority,
            ),
        ]
        if self.config.use_bottlenecks:
            sources.append(
                (
                    "bottleneck_candidates_added",
                    self._bottleneck_order(bottleneck_results),
                    self.config.bottleneck_priority,
                )
            )
        sources.append(
            (
                "fallback_candidates_added",
                self._fallback_for_position(c_arc, position),
                self.config.fallback_priority,
            )
        )
        return self._build_candidates(sources, available_tracks)

    def build_backward_candidates(
        self,
        next_track: int,
        position: int,
        available_tracks: set[int],
        graph_data: TransitionGraphData,
        bottleneck_results: dict,
        c_arc: np.ndarray,
    ) -> list[int]:
        """Build candidates suitable before the current backward boundary."""
        number_of_tracks, playlist_length = self._validate_inputs(
            available_tracks, graph_data, bottleneck_results, c_arc
        )
        if not 0 <= next_track < number_of_tracks or not 0 <= position < playlist_length:
            raise ValueError("next_track and position must be valid.")
        sources = [
            (
                "incoming_graph_candidates_added",
                graph_data.incoming_neighbors[next_track],
                self.config.graph_priority,
            ),
            (
                "candidate_set_candidates_added",
                bottleneck_results["candidate_sets"][position],
                self.config.candidate_set_priority,
            ),
        ]
        if self.config.use_bottlenecks:
            sources.append(
                (
                    "bottleneck_candidates_added",
                    self._bottleneck_order(bottleneck_results),
                    self.config.bottleneck_priority,
                )
            )
        sources.append(
            (
                "fallback_candidates_added",
                self._fallback_for_position(c_arc, position),
                self.config.fallback_priority,
            )
        )
        return self._build_candidates(sources, available_tracks)

    def build_anchor_candidates(
        self,
        left_track: int,
        right_track: int,
        midpoint: int,
        available_tracks: set[int],
        graph_data: TransitionGraphData,
        bottleneck_results: dict,
        c_arc: np.ndarray,
        c_trans: np.ndarray,
    ) -> list[int]:
        """Build midpoint candidates that connect both interval boundaries."""
        number_of_tracks, playlist_length = self._validate_inputs(
            available_tracks, graph_data, bottleneck_results, c_arc
        )
        if (
            not 0 <= left_track < number_of_tracks
            or not 0 <= right_track < number_of_tracks
            or not 0 <= midpoint < playlist_length
        ):
            raise ValueError("Boundary tracks and midpoint must be valid.")
        self._validate_c_trans(c_trans, number_of_tracks)
        outgoing = set(graph_data.outgoing_neighbors[left_track]).intersection(
            available_tracks
        )
        incoming = set(graph_data.incoming_neighbors[right_track]).intersection(
            available_tracks
        )
        intersection = outgoing.intersection(incoming)
        union = outgoing.union(incoming)
        candidate_set = set(
            int(track)
            for track in bottleneck_results["candidate_sets"][midpoint]
        ).intersection(available_tracks)
        bottlenecks = (
            set(self._bottleneck_order(bottleneck_results)).intersection(
                available_tracks
            )
            if self.config.use_bottlenecks
            else set()
        )
        song_scores = np.asarray(bottleneck_results["song_bottleneck_scores"])

        def bridge_score(candidate: int) -> tuple[float, int]:
            left_cost = c_trans[left_track, candidate]
            right_cost = c_trans[candidate, right_track]
            score = (
                left_cost
                + right_cost
                + self.config.bridge_balance_weight * abs(left_cost - right_cost)
                + self.config.arc_preview_weight * c_arc[candidate, midpoint]
                - self.config.bottleneck_preview_weight * song_scores[candidate]
            )
            return float(score), candidate

        bridge_candidates = set(
            sorted(available_tracks, key=bridge_score)[: self.config.max_candidates]
        )
        fallback = set(available_tracks).difference(
            outgoing,
            incoming,
            candidate_set,
            bottlenecks,
            bridge_candidates,
        )

        candidate_sources: dict[int, set[str]] = {}

        def add_source(tracks: set[int], label: str) -> None:
            for track in tracks:
                candidate_sources.setdefault(track, set()).add(label)

        add_source(outgoing, "outgoing_graph")
        add_source(incoming, "incoming_graph")
        add_source(intersection, "intersection")
        add_source(bridge_candidates, "bridge")
        add_source(candidate_set, "candidate_set")
        add_source(bottlenecks, "bottleneck")
        add_source(fallback, "fallback")

        def anchor_pre_score(candidate: int) -> tuple[float, int]:
            sources = candidate_sources[candidate]
            left_cost = c_trans[left_track, candidate]
            right_cost = c_trans[candidate, right_track]
            score = (
                self.config.graph_left_weight * left_cost
                + self.config.graph_right_weight * right_cost
                + self.config.balance_weight * abs(left_cost - right_cost)
                + self.config.arc_preview_weight * c_arc[candidate, midpoint]
                - self.config.bottleneck_preview_weight * song_scores[candidate]
            )
            if "intersection" in sources:
                score -= 0.30
            if "bridge" in sources:
                score -= 0.20
            if "candidate_set" in sources and (
                "outgoing_graph" in sources or "incoming_graph" in sources
            ):
                score -= 0.10
            return float(score), candidate

        ranked = sorted(candidate_sources, key=anchor_pre_score)
        candidates = ranked[: self.config.max_candidates]
        selected_sources = {
            candidate: candidate_sources[candidate] for candidate in candidates
        }
        self._last_candidate_sources = selected_sources
        self._last_anchor_candidate_sources = selected_sources
        self._anchor_build_count += 1
        self._anchor_candidate_count_total += len(candidates)
        self._diagnostics["anchor_intersection_size_total"] += len(intersection)
        self._diagnostics["anchor_union_size_total"] += len(union)
        self._diagnostics["intersection_anchor_candidates_added"] += sum(
            "intersection" in sources for sources in selected_sources.values()
        )
        self._diagnostics["outgoing_graph_candidates_added"] += sum(
            "outgoing_graph" in sources for sources in selected_sources.values()
        )
        self._diagnostics["incoming_graph_candidates_added"] += sum(
            "incoming_graph" in sources for sources in selected_sources.values()
        )
        self._diagnostics["candidate_set_candidates_added"] += sum(
            "candidate_set" in sources for sources in selected_sources.values()
        )
        self._diagnostics["bottleneck_candidates_added"] += sum(
            "bottleneck" in sources for sources in selected_sources.values()
        )
        self._diagnostics["fallback_candidates_added"] += sum(
            "fallback" in sources for sources in selected_sources.values()
        )
        self._diagnostics["bridge_candidates_added"] += sum(
            "bridge" in sources for sources in selected_sources.values()
        )
        self._diagnostics["anchor_candidates_from_outgoing_only"] += sum(
            "outgoing_graph" in sources and "incoming_graph" not in sources
            for sources in selected_sources.values()
        )
        self._diagnostics["anchor_candidates_from_incoming_only"] += sum(
            "incoming_graph" in sources and "outgoing_graph" not in sources
            for sources in selected_sources.values()
        )
        self._diagnostics["anchor_candidates_from_both_sides"] += sum(
            "outgoing_graph" in sources and "incoming_graph" in sources
            for sources in selected_sources.values()
        )
        self._diagnostics["anchor_candidates_from_candidate_set"] += sum(
            "candidate_set" in sources for sources in selected_sources.values()
        )
        self._diagnostics["anchor_candidates_from_bottleneck"] += sum(
            "bottleneck" in sources for sources in selected_sources.values()
        )
        self._diagnostics["anchor_candidates_from_fallback"] += sum(
            "fallback" in sources for sources in selected_sources.values()
        )
        top_candidates = candidates[:10]
        self._anchor_top10_left_cost_total += sum(
            float(c_trans[left_track, candidate]) for candidate in top_candidates
        )
        self._anchor_top10_right_cost_total += sum(
            float(c_trans[candidate, right_track]) for candidate in top_candidates
        )
        self._anchor_top10_balance_gap_total += sum(
            abs(
                float(c_trans[left_track, candidate])
                - float(c_trans[candidate, right_track])
            )
            for candidate in top_candidates
        )
        self._anchor_top10_count += len(top_candidates)
        if not candidates:
            self._diagnostics["empty_candidate_pool_count"] += 1
            self._diagnostics["empty_anchor_candidate_pool_count"] += 1
        return candidates

    def build_base_case_candidates(
        self,
        left_track: int,
        right_track: int,
        positions: list[int],
        available_tracks: set[int],
        graph_data: TransitionGraphData,
        bottleneck_results: dict,
        c_arc: np.ndarray,
    ) -> list[int]:
        """Build candidates for a small interval between fixed boundaries."""
        number_of_tracks, playlist_length = self._validate_inputs(
            available_tracks, graph_data, bottleneck_results, c_arc
        )
        if not 0 <= left_track < number_of_tracks or not 0 <= right_track < number_of_tracks:
            raise ValueError("Boundary tracks must be valid.")
        if not positions or any(not 0 <= position < playlist_length for position in positions):
            raise ValueError("positions must contain valid playlist positions.")
        ordered_positions = list(positions)
        if self.config.use_location_bottlenecks:
            location_scores = np.asarray(
                bottleneck_results["location_bottleneck_scores"]
            )
            ordered_positions.sort(key=lambda position: (-location_scores[position], position))
        candidate_set_tracks = [
            track
            for position in ordered_positions
            for track in bottleneck_results["candidate_sets"][position]
        ]
        fallback_scores = c_arc[:, positions].mean(axis=1)
        sources = [
            (
                "outgoing_graph_candidates_added",
                graph_data.outgoing_neighbors[left_track],
                self.config.graph_priority,
            ),
            (
                "incoming_graph_candidates_added",
                graph_data.incoming_neighbors[right_track],
                self.config.graph_priority,
            ),
            (
                "candidate_set_candidates_added",
                candidate_set_tracks,
                self.config.candidate_set_priority,
            ),
        ]
        if self.config.use_bottlenecks:
            sources.append(
                (
                    "bottleneck_candidates_added",
                    self._bottleneck_order(bottleneck_results),
                    self.config.bottleneck_priority,
                )
            )
        sources.append(
            (
                "fallback_candidates_added",
                np.argsort(fallback_scores, kind="stable").tolist(),
                self.config.fallback_priority,
            )
        )
        return self._build_candidates(sources, available_tracks)

    def get_diagnostics(self) -> dict:
        """Return cumulative candidate-source counters."""
        diagnostics = dict(self._diagnostics)
        diagnostics["average_anchor_candidate_count"] = (
            self._anchor_candidate_count_total / self._anchor_build_count
            if self._anchor_build_count
            else 0.0
        )
        diagnostics["average_anchor_left_cost_top10"] = (
            self._anchor_top10_left_cost_total / self._anchor_top10_count
            if self._anchor_top10_count
            else 0.0
        )
        diagnostics["average_anchor_right_cost_top10"] = (
            self._anchor_top10_right_cost_total / self._anchor_top10_count
            if self._anchor_top10_count
            else 0.0
        )
        diagnostics["average_anchor_balance_gap_top10"] = (
            self._anchor_top10_balance_gap_total / self._anchor_top10_count
            if self._anchor_top10_count
            else 0.0
        )
        diagnostics["fallback_anchor_ratio"] = (
            self._diagnostics["anchor_candidates_from_fallback"]
            / self._anchor_candidate_count_total
            if self._anchor_candidate_count_total
            else 0.0
        )
        return diagnostics

    def get_last_candidate_sources(self) -> dict[int, set[str]]:
        """Return source labels for the most recently built candidate pool."""
        return {
            track: set(sources)
            for track, sources in self._last_candidate_sources.items()
        }

    def get_last_anchor_candidate_sources(self) -> dict[int, set[str]]:
        """Return source labels for the most recent anchor candidate pool."""
        return {
            track: set(sources)
            for track, sources in self._last_anchor_candidate_sources.items()
        }
