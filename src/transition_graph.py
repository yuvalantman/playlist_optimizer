"""Build sparse directed transition graphs from pairwise transition costs."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransitionGraphConfig:
    """Configuration for threshold-based transition graph construction."""

    threshold_percentile: float = 0.15
    absolute_threshold: float | None = None
    min_out_degree: int = 5
    min_in_degree: int = 5
    max_out_degree: int | None = 40
    max_in_degree: int | None = 40
    exclude_self_transitions: bool = True
    add_fallback_edges: bool = True
    top_m_neighbors: int = 30
    max_tonality_distance: float | None = None
    max_rhythm_distance: float | None = None


@dataclass(frozen=True)
class TransitionGraphData:
    """All directed graph views and diagnostics used by playlist search."""

    outgoing_neighbors: dict[int, list[int]]
    incoming_neighbors: dict[int, list[int]]
    outgoing_neighbors_with_costs: dict[int, list[tuple[int, float]]]
    incoming_neighbors_with_costs: dict[int, list[tuple[int, float]]]
    edge_metadata: dict[tuple[int, int], dict]
    threshold_used: float
    diagnostics: dict


class TransitionGraphBuilder:
    """Build threshold-based and legacy Top-M transition graphs."""

    def __init__(self, config: TransitionGraphConfig) -> None:
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        if not 0 <= self.config.threshold_percentile <= 1:
            raise ValueError("threshold_percentile must be between 0 and 1.")
        if (
            self.config.absolute_threshold is not None
            and not np.isfinite(self.config.absolute_threshold)
        ):
            raise ValueError("absolute_threshold must be finite when provided.")
        for name in ("max_tonality_distance", "max_rhythm_distance"):
            value = getattr(self.config, name)
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative.")
        for name, value in (
            ("min_out_degree", self.config.min_out_degree),
            ("min_in_degree", self.config.min_in_degree),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.config.top_m_neighbors <= 0:
            raise ValueError("top_m_neighbors must be positive.")
        for name, value in (
            ("max_out_degree", self.config.max_out_degree),
            ("max_in_degree", self.config.max_in_degree),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided.")
        if (
            self.config.max_out_degree is not None
            and self.config.min_out_degree > self.config.max_out_degree
        ):
            raise ValueError("min_out_degree cannot exceed max_out_degree.")
        if (
            self.config.max_in_degree is not None
            and self.config.min_in_degree > self.config.max_in_degree
        ):
            raise ValueError("min_in_degree cannot exceed max_in_degree.")

    def _validate_c_trans(self, c_trans: np.ndarray) -> int:
        if not isinstance(c_trans, np.ndarray) or c_trans.ndim != 2:
            raise ValueError("c_trans must be a 2D numpy array.")
        if c_trans.shape[0] == 0 or c_trans.shape[0] != c_trans.shape[1]:
            raise ValueError("c_trans must be a non-empty square matrix.")
        if not np.issubdtype(c_trans.dtype, np.number):
            raise ValueError("c_trans must contain numeric values.")
        if not np.isfinite(c_trans).all():
            raise ValueError("c_trans must contain only finite values.")

        return c_trans.shape[0]

    def _valid_edges(self, number_of_tracks: int) -> list[tuple[int, int]]:
        return [
            (source, target)
            for source in range(number_of_tracks)
            for target in range(number_of_tracks)
            if not self.config.exclude_self_transitions or source != target
        ]

    @staticmethod
    def _sorted_edges(
        edges: list[tuple[int, int]],
        c_trans: np.ndarray,
    ) -> list[tuple[int, int]]:
        return sorted(
            edges,
            key=lambda edge: (float(c_trans[edge]), edge[0], edge[1]),
        )

    def _compute_threshold(
        self,
        c_trans: np.ndarray,
        valid_edges: list[tuple[int, int]],
    ) -> float:
        if self.config.absolute_threshold is not None:
            return float(self.config.absolute_threshold)
        if not valid_edges:
            return 0.0
        costs = np.asarray([c_trans[edge] for edge in valid_edges], dtype=float)
        return float(np.quantile(costs, self.config.threshold_percentile))

    def _within_caps(
        self,
        source: int,
        target: int,
        outgoing: dict[int, set[int]],
        incoming: dict[int, set[int]],
    ) -> bool:
        out_cap = self.config.max_out_degree
        in_cap = self.config.max_in_degree
        return (
            (out_cap is None or len(outgoing[source]) < out_cap)
            and (in_cap is None or len(incoming[target]) < in_cap)
        )

    def _add_edge(
        self,
        source: int,
        target: int,
        edge_type: str,
        c_trans: np.ndarray,
        outgoing: dict[int, set[int]],
        incoming: dict[int, set[int]],
        metadata: dict[tuple[int, int], dict],
    ) -> bool:
        if target in outgoing[source] or not self._within_caps(
            source,
            target,
            outgoing,
            incoming,
        ):
            return False
        outgoing[source].add(target)
        incoming[target].add(source)
        metadata[(source, target)] = {
            "cost": float(c_trans[source, target]),
            "edge_type": edge_type,
        }
        return True

    def build_transition_graph_data(
        self,
        c_trans: np.ndarray,
        tonality_distances: np.ndarray | None = None,
        rhythm_distances: np.ndarray | None = None,
    ) -> TransitionGraphData:
        """Build a threshold graph with incoming views and fallback edges."""
        number_of_tracks = self._validate_c_trans(c_trans)
        for name, matrix in (
            ("tonality_distances", tonality_distances),
            ("rhythm_distances", rhythm_distances),
        ):
            if matrix is not None and (
                not isinstance(matrix, np.ndarray)
                or matrix.shape != c_trans.shape
                or not np.issubdtype(matrix.dtype, np.number)
                or not np.isfinite(matrix).all()
            ):
                raise ValueError(f"{name} must be a finite matrix matching c_trans.")
        valid_edges = self._valid_edges(number_of_tracks)
        threshold = self._compute_threshold(c_trans, valid_edges)
        threshold_edges = {
            edge
            for edge in valid_edges
            if float(c_trans[edge]) <= threshold
            and (
                self.config.max_tonality_distance is None
                or tonality_distances is None
                or float(tonality_distances[edge])
                <= self.config.max_tonality_distance
            )
            and (
                self.config.max_rhythm_distance is None
                or rhythm_distances is None
                or float(rhythm_distances[edge])
                <= self.config.max_rhythm_distance
            )
        }
        outgoing = {track: set() for track in range(number_of_tracks)}
        incoming = {track: set() for track in range(number_of_tracks)}
        metadata: dict[tuple[int, int], dict] = {}

        for source, target in self._sorted_edges(list(threshold_edges), c_trans):
            self._add_edge(
                source,
                target,
                "threshold",
                c_trans,
                outgoing,
                incoming,
                metadata,
            )

        if self.config.add_fallback_edges:
            sorted_edges = self._sorted_edges(valid_edges, c_trans)
            for source in range(number_of_tracks):
                for edge_source, target in sorted_edges:
                    if len(outgoing[source]) >= self.config.min_out_degree:
                        break
                    if edge_source != source:
                        continue
                    self._add_edge(
                        source,
                        target,
                        "fallback",
                        c_trans,
                        outgoing,
                        incoming,
                        metadata,
                    )
            for target in range(number_of_tracks):
                for source, edge_target in sorted_edges:
                    if len(incoming[target]) >= self.config.min_in_degree:
                        break
                    if edge_target != target:
                        continue
                    self._add_edge(
                        source,
                        target,
                        "fallback",
                        c_trans,
                        outgoing,
                        incoming,
                        metadata,
                    )

        outgoing_with_costs = {
            source: sorted(
                (
                    (target, float(c_trans[source, target]))
                    for target in outgoing[source]
                ),
                key=lambda item: (item[1], item[0]),
            )
            for source in range(number_of_tracks)
        }
        incoming_with_costs = {
            target: sorted(
                (
                    (source, float(c_trans[source, target]))
                    for source in incoming[target]
                ),
                key=lambda item: (item[1], item[0]),
            )
            for target in range(number_of_tracks)
        }
        outgoing_neighbors = {
            source: [target for target, _ in neighbors]
            for source, neighbors in outgoing_with_costs.items()
        }
        incoming_neighbors = {
            target: [source for source, _ in neighbors]
            for target, neighbors in incoming_with_costs.items()
        }
        out_degrees = np.asarray(
            [len(outgoing_neighbors[node]) for node in range(number_of_tracks)]
        )
        in_degrees = np.asarray(
            [len(incoming_neighbors[node]) for node in range(number_of_tracks)]
        )
        threshold_edge_count = sum(
            edge["edge_type"] == "threshold" for edge in metadata.values()
        )
        fallback_edge_count = len(metadata) - threshold_edge_count
        possible_edge_count = number_of_tracks * (
            number_of_tracks - int(self.config.exclude_self_transitions)
        )
        diagnostics = {
            "number_of_nodes": number_of_tracks,
            "number_of_edges": len(metadata),
            "threshold_used": threshold,
            "threshold_edge_count": threshold_edge_count,
            "fallback_edge_count": fallback_edge_count,
            "average_out_degree": float(out_degrees.mean()),
            "average_in_degree": float(in_degrees.mean()),
            "min_out_degree": int(out_degrees.min()),
            "max_out_degree": int(out_degrees.max()),
            "min_in_degree": int(in_degrees.min()),
            "max_in_degree": int(in_degrees.max()),
            "isolated_out_nodes": np.flatnonzero(out_degrees == 0).tolist(),
            "isolated_in_nodes": np.flatnonzero(in_degrees == 0).tolist(),
            "graph_density": (
                float(len(metadata) / possible_edge_count)
                if possible_edge_count
                else 0.0
            ),
        }
        return TransitionGraphData(
            outgoing_neighbors=outgoing_neighbors,
            incoming_neighbors=incoming_neighbors,
            outgoing_neighbors_with_costs=outgoing_with_costs,
            incoming_neighbors_with_costs=incoming_with_costs,
            edge_metadata=metadata,
            threshold_used=threshold,
            diagnostics=diagnostics,
        )

    def _sorted_neighbor_indices(
        self,
        c_trans: np.ndarray,
        track_index: int,
    ) -> np.ndarray:
        ordered_indices = np.argsort(c_trans[track_index], kind="stable")
        if self.config.exclude_self_transitions:
            ordered_indices = ordered_indices[ordered_indices != track_index]
        return ordered_indices[: self.config.top_m_neighbors]

    def build_top_m_graph(self, c_trans: np.ndarray) -> dict[int, list[int]]:
        """Return the legacy Top-M lowest-cost outgoing neighbor graph."""
        number_of_tracks = self._validate_c_trans(c_trans)
        possible_neighbors = number_of_tracks - int(
            self.config.exclude_self_transitions
        )
        if self.config.top_m_neighbors > possible_neighbors:
            raise ValueError("top_m_neighbors exceeds possible neighbors.")
        return {
            track: self._sorted_neighbor_indices(c_trans, track).tolist()
            for track in range(number_of_tracks)
        }

    def build_top_m_graph_with_costs(
        self,
        c_trans: np.ndarray,
    ) -> dict[int, list[tuple[int, float]]]:
        """Return legacy Top-M outgoing neighbors and their costs."""
        graph = self.build_top_m_graph(c_trans)
        return {
            source: [
                (target, float(c_trans[source, target]))
                for target in targets
            ]
            for source, targets in graph.items()
        }
