"""Bottleneck-Guided Bidirectional Beam Search playlist construction."""

from dataclasses import dataclass
from itertools import permutations
import math

import numpy as np


@dataclass(frozen=True)
class BIBSConfig:
    """Configuration for Bottleneck-Guided Bidirectional Beam Search."""

    beam_width: int = 8
    base_case_size: int = 4
    candidate_pool_size: int = 40
    arc_weight: float = 1.0
    transition_weight: float = 1.2
    bottleneck_weight: float = 0.25
    meet_transition_weight: float = 1.2
    boundary_transition_weight: float = 1.0
    graph_neighbor_bonus: float = 0.25
    non_neighbor_penalty: float = 0.75
    use_transition_graph: bool = True
    max_recursion_depth: int = 30
    enable_local_repair: bool = True
    local_repair_passes: int = 2
    local_repair_window: int = 4
    local_repair_arc_tolerance: float = 0.03


class BIBS:
    """Construct a complete playlist with bottleneck-guided recursion."""

    def __init__(self, config: BIBSConfig) -> None:
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.beam_width <= 0:
            raise ValueError("beam_width must be positive.")
        if self.config.base_case_size <= 0:
            raise ValueError("base_case_size must be positive.")
        if self.config.candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive.")
        if self.config.max_recursion_depth <= 0:
            raise ValueError("max_recursion_depth must be positive.")
        if self.config.local_repair_passes <= 0:
            raise ValueError("local_repair_passes must be positive.")
        if self.config.local_repair_window <= 0:
            raise ValueError("local_repair_window must be positive.")
        if self.config.local_repair_arc_tolerance < 0:
            raise ValueError("local_repair_arc_tolerance must be non-negative.")
        if self.config.graph_neighbor_bonus < 0:
            raise ValueError("graph_neighbor_bonus must be non-negative.")
        if self.config.non_neighbor_penalty < 0:
            raise ValueError("non_neighbor_penalty must be non-negative.")
        if self.config.meet_transition_weight < 0:
            raise ValueError("meet_transition_weight must be non-negative.")

    @staticmethod
    def _validate_inputs(
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        bottleneck_results: dict,
        start_index: int | None,
        end_index: int | None,
    ) -> int:
        if not isinstance(c_arc, np.ndarray) or c_arc.ndim != 2:
            raise ValueError("c_arc must be a 2D numpy array.")
        if not isinstance(c_trans, np.ndarray) or c_trans.ndim != 2:
            raise ValueError("c_trans must be a square 2D numpy array.")
        if c_trans.shape[0] == 0 or c_trans.shape[0] != c_trans.shape[1]:
            raise ValueError("c_trans must be a non-empty square matrix.")
        if c_arc.shape != c_trans.shape:
            raise ValueError(
                "c_arc and c_trans must use the same tracks and positions."
            )
        if not np.issubdtype(c_arc.dtype, np.number) or not np.isfinite(c_arc).all():
            raise ValueError("c_arc must contain only finite numeric values.")
        if not np.issubdtype(c_trans.dtype, np.number) or not np.isfinite(
            c_trans
        ).all():
            raise ValueError("c_trans must contain only finite numeric values.")
        if not isinstance(bottleneck_results, dict):
            raise ValueError("bottleneck_results must be a dictionary.")

        required_keys = {
            "song_bottleneck_scores",
            "bottleneck_track_indices",
            "candidate_sets",
        }
        missing_keys = required_keys.difference(bottleneck_results)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"bottleneck_results is missing keys: {missing}")

        number_of_tracks = c_trans.shape[0]
        song_scores = np.asarray(bottleneck_results["song_bottleneck_scores"])
        if song_scores.ndim != 1 or len(song_scores) != number_of_tracks:
            raise ValueError(
                "song_bottleneck_scores must contain one score per track."
            )
        if not np.issubdtype(song_scores.dtype, np.number) or not np.isfinite(
            song_scores
        ).all():
            raise ValueError("song_bottleneck_scores must be finite and numeric.")
        if not isinstance(bottleneck_results["candidate_sets"], dict):
            raise ValueError("candidate_sets must be a dictionary.")

        for name, index in (("start_index", start_index), ("end_index", end_index)):
            if index is not None and (
                not isinstance(index, (int, np.integer))
                or not 0 <= index < number_of_tracks
            ):
                raise ValueError(f"{name} must be a valid track index.")
        if start_index is not None and start_index == end_index:
            raise ValueError("start_index and end_index cannot be the same.")

        return number_of_tracks

    @staticmethod
    def _ordered_available_by_arc(
        available_tracks: set[int],
        position: int,
        c_arc: np.ndarray,
    ) -> list[int]:
        return sorted(
            available_tracks,
            key=lambda track: (float(c_arc[track, position]), track),
        )

    def _build_candidate_pool(
        self,
        positions: list[int],
        available_tracks: set[int],
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        bottleneck_results: dict,
        transition_graph: dict[int, list[int]] | None,
        boundary_tracks: list[int],
        minimum_size: int,
    ) -> list[int]:
        candidate_sets = bottleneck_results["candidate_sets"]
        bottleneck_tracks = bottleneck_results["bottleneck_track_indices"]
        candidates: set[int] = set()

        left_track = boundary_tracks[0]
        right_track = boundary_tracks[-1]
        if self.config.use_transition_graph and transition_graph is not None:
            candidates.update(transition_graph.get(left_track, []))

        for position in positions:
            candidates.update(candidate_sets.get(position, []))
        candidates.update(int(track) for track in bottleneck_tracks)
        candidates.intersection_update(available_tracks)

        center_position = positions[len(positions) // 2]
        best_by_arc = self._ordered_available_by_arc(
            available_tracks,
            center_position,
            c_arc,
        )
        for track in best_by_arc:
            candidates.add(track)
            if len(candidates) >= max(minimum_size, self.config.candidate_pool_size):
                break

        left_neighbors = (
            set(transition_graph.get(left_track, []))
            if self.config.use_transition_graph and transition_graph is not None
            else set()
        )
        use_graph_relations = (
            self.config.use_transition_graph and transition_graph is not None
        )

        def candidate_score(track: int) -> tuple[float, int]:
            arc_cost = float(
                np.mean([c_arc[track, position] for position in positions])
            )
            boundary_cost = (
                c_trans[left_track, track] + c_trans[track, right_track]
            ) / 2
            score = (
                self.config.arc_weight * arc_cost
                + self.config.boundary_transition_weight * boundary_cost
            )
            if left_neighbors:
                score += (
                    -self.config.graph_neighbor_bonus
                    if track in left_neighbors
                    else self.config.non_neighbor_penalty
                )
            if use_graph_relations and right_track not in transition_graph.get(track, []):
                score += self.config.non_neighbor_penalty / 2
            return float(score), track

        limit = max(minimum_size, self.config.candidate_pool_size)
        return sorted(candidates, key=candidate_score)[:limit]

    def _beam_candidate_pool(
        self,
        position: int,
        last_track: int,
        available_tracks: set[int],
        used_tracks: set[int],
        c_arc: np.ndarray,
        bottleneck_scores: np.ndarray,
        candidate_sets: dict[int, list[int]],
        transition_graph: dict[int, list[int]] | None,
    ) -> list[int]:
        candidates: list[int] = []
        seen: set[int] = set()

        def add_tracks(tracks: list[int] | np.ndarray) -> None:
            for track_value in tracks:
                track = int(track_value)
                if (
                    track in available_tracks
                    and track not in used_tracks
                    and track not in seen
                ):
                    candidates.append(track)
                    seen.add(track)
                    if len(candidates) >= self.config.candidate_pool_size:
                        return

        add_tracks(candidate_sets.get(position, []))
        if (
            len(candidates) < self.config.candidate_pool_size
            and self.config.use_transition_graph
            and transition_graph is not None
        ):
            add_tracks(transition_graph.get(last_track, []))
        if len(candidates) < self.config.candidate_pool_size:
            bottleneck_order = np.argsort(-bottleneck_scores, kind="stable")
            add_tracks(bottleneck_order)
        if len(candidates) < self.config.candidate_pool_size:
            add_tracks(
                np.argsort(c_arc[:, position], kind="stable")
            )
        return candidates

    def _expand_beam(
        self,
        start_track: int,
        positions: list[int],
        available_tracks: set[int],
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        bottleneck_scores: np.ndarray,
        candidate_sets: dict[int, list[int]],
        transition_graph: dict[int, list[int]] | None,
        direction: str,
    ) -> list[dict]:
        """Expand a deterministic forward or backward beam."""
        if direction not in {"forward", "backward"}:
            raise ValueError("direction must be 'forward' or 'backward'.")

        beams: list[dict] = [
            {
                "path": [],
                "used": set(),
                "last_track": start_track,
                "score": 0.0,
            }
        ]
        for position in positions:
            expanded: list[dict] = []
            for beam in beams:
                candidate_pool = self._beam_candidate_pool(
                    position,
                    beam["last_track"],
                    available_tracks,
                    beam["used"],
                    c_arc,
                    bottleneck_scores,
                    candidate_sets,
                    transition_graph,
                )
                for candidate in candidate_pool:
                    transition_cost = (
                        c_trans[beam["last_track"], candidate]
                        if direction == "forward"
                        else c_trans[candidate, beam["last_track"]]
                    )
                    score = (
                        beam["score"]
                        + self.config.arc_weight * c_arc[candidate, position]
                        + self.config.transition_weight * transition_cost
                        - self.config.bottleneck_weight
                        * bottleneck_scores[candidate]
                    )
                    expanded.append(
                        {
                            "path": beam["path"] + [candidate],
                            "used": beam["used"] | {candidate},
                            "last_track": candidate,
                            "score": float(score),
                        }
                    )

            if not expanded:
                return []
            expanded.sort(
                key=lambda beam: (
                    beam["score"],
                    tuple(beam["path"]),
                )
            )
            beams = expanded[: self.config.beam_width]
        return beams

    def _anchor_candidates(
        self,
        midpoint: int,
        available_tracks: set[int],
        excluded_tracks: set[int],
        forward_last_track: int,
        c_arc: np.ndarray,
        bottleneck_results: dict,
        transition_graph: dict[int, list[int]] | None,
    ) -> list[int]:
        candidates: list[int] = []
        seen: set[int] = set()

        def add_tracks(tracks: list[int] | np.ndarray) -> None:
            for track_value in tracks:
                track = int(track_value)
                if (
                    track in available_tracks
                    and track not in excluded_tracks
                    and track not in seen
                ):
                    candidates.append(track)
                    seen.add(track)
                    if len(candidates) >= self.config.candidate_pool_size:
                        return

        add_tracks(bottleneck_results["candidate_sets"].get(midpoint, []))
        add_tracks(bottleneck_results["bottleneck_track_indices"])
        if self.config.use_transition_graph and transition_graph is not None:
            add_tracks(transition_graph.get(forward_last_track, []))
        add_tracks(np.argsort(c_arc[:, midpoint], kind="stable"))
        return candidates

    def _select_bidirectional_meet(
        self,
        midpoint: int,
        forward_beams: list[dict],
        backward_beams: list[dict],
        available_tracks: set[int],
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        bottleneck_results: dict,
        transition_graph: dict[int, list[int]] | None,
    ) -> tuple[dict, dict, int] | None:
        bottleneck_scores = np.asarray(
            bottleneck_results["song_bottleneck_scores"]
        )
        best: tuple[float, tuple[int, ...], tuple[int, ...], int, dict, dict] | None = None

        for forward_beam in forward_beams:
            for backward_beam in backward_beams:
                if forward_beam["used"].intersection(backward_beam["used"]):
                    continue
                excluded = forward_beam["used"] | backward_beam["used"]
                anchors = self._anchor_candidates(
                    midpoint,
                    available_tracks,
                    excluded,
                    forward_beam["last_track"],
                    c_arc,
                    bottleneck_results,
                    transition_graph,
                )
                for anchor in anchors:
                    score = (
                        forward_beam["score"]
                        + backward_beam["score"]
                        + self.config.arc_weight * c_arc[anchor, midpoint]
                        + self.config.meet_transition_weight
                        * c_trans[forward_beam["last_track"], anchor]
                        + self.config.meet_transition_weight
                        * c_trans[anchor, backward_beam["last_track"]]
                        - self.config.bottleneck_weight
                        * bottleneck_scores[anchor]
                    )
                    candidate = (
                        float(score),
                        tuple(forward_beam["path"]),
                        tuple(backward_beam["path"]),
                        anchor,
                        forward_beam,
                        backward_beam,
                    )
                    if best is None or candidate[:4] < best[:4]:
                        best = candidate

        if best is None:
            return None
        return best[4], best[5], best[3]

    def _select_anchor(
        self,
        position: int,
        left_track: int,
        right_track: int,
        available_tracks: set[int],
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        bottleneck_results: dict,
        transition_graph: dict[int, list[int]] | None,
    ) -> int:
        """Select an anchor fallback for degenerate beam intervals."""
        candidates = self._anchor_candidates(
            position,
            available_tracks,
            set(),
            left_track,
            c_arc,
            bottleneck_results,
            transition_graph,
        )
        song_scores = np.asarray(bottleneck_results["song_bottleneck_scores"])
        return min(
            candidates,
            key=lambda track: (
                self.config.arc_weight * c_arc[track, position]
                + self.config.meet_transition_weight
                * c_trans[left_track, track]
                + self.config.meet_transition_weight
                * c_trans[track, right_track]
                - self.config.bottleneck_weight * song_scores[track],
                track,
            ),
        )

    def _local_score(
        self,
        arrangement: tuple[int, ...],
        positions: list[int],
        left_track: int,
        right_track: int,
        c_arc: np.ndarray,
        c_trans: np.ndarray,
    ) -> float:
        score = 0.0
        previous_track = left_track
        for position, track in zip(positions, arrangement):
            score += self.config.arc_weight * c_arc[track, position]
            score += self.config.transition_weight * c_trans[previous_track, track]
            previous_track = track
        score += self.config.transition_weight * c_trans[previous_track, right_track]
        return float(score)

    def _beam_arrangement(
        self,
        candidate_pool: list[int],
        positions: list[int],
        left_track: int,
        right_track: int,
        c_arc: np.ndarray,
        c_trans: np.ndarray,
    ) -> tuple[int, ...]:
        beam: list[tuple[float, tuple[int, ...]]] = [(0.0, ())]

        for position in positions:
            expanded: list[tuple[float, tuple[int, ...]]] = []
            for score, arrangement in beam:
                previous_track = arrangement[-1] if arrangement else left_track
                used = set(arrangement)
                for track in candidate_pool:
                    if track in used:
                        continue
                    next_score = (
                        score
                        + self.config.arc_weight * c_arc[track, position]
                        + self.config.transition_weight
                        * c_trans[previous_track, track]
                    )
                    expanded.append((float(next_score), arrangement + (track,)))

            expanded.sort(key=lambda item: (item[0], item[1]))
            beam = expanded[: self.config.beam_width]

        return min(
            beam,
            key=lambda item: (
                item[0]
                + self.config.transition_weight
                * c_trans[item[1][-1], right_track],
                item[1],
            ),
        )[1]

    def _fill_base_case(
        self,
        playlist: list[int | None],
        left_pos: int,
        right_pos: int,
        available_tracks: set[int],
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        bottleneck_results: dict,
        transition_graph: dict[int, list[int]] | None,
    ) -> None:
        """Fill a small interval using exact permutations or local beam search."""
        positions = [
            position
            for position in range(left_pos + 1, right_pos)
            if playlist[position] is None
        ]
        if not positions:
            return

        left_track = playlist[left_pos]
        right_track = playlist[right_pos]
        if left_track is None or right_track is None:
            raise RuntimeError("Base-case boundaries must already be assigned.")

        candidate_pool = self._build_candidate_pool(
            positions,
            available_tracks,
            c_arc,
            c_trans,
            bottleneck_results,
            transition_graph,
            [left_track, right_track],
            minimum_size=len(positions),
        )
        permutation_count = math.perm(len(candidate_pool), len(positions))

        if permutation_count <= 10_000:
            arrangement = min(
                permutations(candidate_pool, len(positions)),
                key=lambda option: (
                    self._local_score(
                        option,
                        positions,
                        left_track,
                        right_track,
                        c_arc,
                        c_trans,
                    ),
                    option,
                ),
            )
        else:
            arrangement = self._beam_arrangement(
                candidate_pool,
                positions,
                left_track,
                right_track,
                c_arc,
                c_trans,
            )

        for position, track in zip(positions, arrangement):
            playlist[position] = track
            available_tracks.remove(track)

    @staticmethod
    def _swap_transition_cost(
        playlist: list[int],
        first_position: int,
        second_position: int,
        c_trans: np.ndarray,
    ) -> float:
        affected_transitions = {
            first_position - 1,
            first_position,
            second_position - 1,
            second_position,
        }
        valid_transitions = sorted(
            position
            for position in affected_transitions
            if 0 <= position < len(playlist) - 1
        )
        return float(
            sum(
                c_trans[playlist[position], playlist[position + 1]]
                for position in valid_transitions
            )
        )

    def _local_repair(
        self,
        playlist: list[int],
        c_arc: np.ndarray,
        c_trans: np.ndarray,
    ) -> list[int]:
        """Repair costly transitions with deterministic local swaps."""
        repaired = playlist.copy()
        if len(repaired) < 4:
            return repaired

        for _ in range(self.config.local_repair_passes):
            improved = False
            transition_positions = sorted(
                range(len(repaired) - 1),
                key=lambda position: (
                    -float(c_trans[repaired[position], repaired[position + 1]]),
                    position,
                ),
            )

            for transition_position in transition_positions:
                window_start = max(
                    1,
                    transition_position - self.config.local_repair_window,
                )
                window_end = min(
                    len(repaired) - 2,
                    transition_position + 1 + self.config.local_repair_window,
                )
                best_swap: tuple[float, int, int] | None = None

                for first_position in range(window_start, window_end):
                    for second_position in range(first_position + 1, window_end + 1):
                        old_transition_cost = self._swap_transition_cost(
                            repaired,
                            first_position,
                            second_position,
                            c_trans,
                        )
                        old_arc_cost = (
                            c_arc[repaired[first_position], first_position]
                            + c_arc[repaired[second_position], second_position]
                        ) / 2

                        repaired[first_position], repaired[second_position] = (
                            repaired[second_position],
                            repaired[first_position],
                        )
                        new_transition_cost = self._swap_transition_cost(
                            repaired,
                            first_position,
                            second_position,
                            c_trans,
                        )
                        new_arc_cost = (
                            c_arc[repaired[first_position], first_position]
                            + c_arc[repaired[second_position], second_position]
                        ) / 2
                        repaired[first_position], repaired[second_position] = (
                            repaired[second_position],
                            repaired[first_position],
                        )

                        improvement = old_transition_cost - new_transition_cost
                        if (
                            improvement > 1e-12
                            and new_arc_cost
                            <= old_arc_cost + self.config.local_repair_arc_tolerance
                        ):
                            candidate = (
                                float(improvement),
                                first_position,
                                second_position,
                            )
                            if best_swap is None or (
                                -candidate[0],
                                candidate[1],
                                candidate[2],
                            ) < (
                                -best_swap[0],
                                best_swap[1],
                                best_swap[2],
                            ):
                                best_swap = candidate

                if best_swap is not None:
                    _, first_position, second_position = best_swap
                    repaired[first_position], repaired[second_position] = (
                        repaired[second_position],
                        repaired[first_position],
                    )
                    improved = True

            if not improved:
                break

        return repaired

    def _solve_interval(
        self,
        playlist: list[int | None],
        left_pos: int,
        right_pos: int,
        available_tracks: set[int],
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        bottleneck_results: dict,
        transition_graph: dict[int, list[int]] | None,
        depth: int,
    ) -> None:
        """Recursively anchor and solve a playlist interval."""
        empty_positions = [
            position
            for position in range(left_pos + 1, right_pos)
            if playlist[position] is None
        ]
        if not empty_positions:
            return
        if (
            len(empty_positions) <= self.config.base_case_size
            or depth >= self.config.max_recursion_depth
        ):
            self._fill_base_case(
                playlist,
                left_pos,
                right_pos,
                available_tracks,
                c_arc,
                c_trans,
                bottleneck_results,
                transition_graph,
            )
            return

        midpoint = (left_pos + right_pos) // 2
        left_track = playlist[left_pos]
        right_track = playlist[right_pos]
        if left_track is None or right_track is None:
            raise RuntimeError("Recursive interval boundaries must be assigned.")

        forward_positions = list(range(left_pos + 1, midpoint))
        backward_positions = list(range(right_pos - 1, midpoint, -1))
        bottleneck_scores = np.asarray(
            bottleneck_results["song_bottleneck_scores"]
        )
        candidate_sets = bottleneck_results["candidate_sets"]

        forward_beams = self._expand_beam(
            left_track,
            forward_positions,
            available_tracks,
            c_arc,
            c_trans,
            bottleneck_scores,
            candidate_sets,
            transition_graph,
            direction="forward",
        )
        backward_beams = self._expand_beam(
            right_track,
            backward_positions,
            available_tracks,
            c_arc,
            c_trans,
            bottleneck_scores,
            candidate_sets,
            transition_graph,
            direction="backward",
        )
        meet = self._select_bidirectional_meet(
            midpoint,
            forward_beams,
            backward_beams,
            available_tracks,
            c_arc,
            c_trans,
            bottleneck_results,
            transition_graph,
        )

        if meet is None:
            compatible_backward_beams: list[dict] = []
            for forward_beam in forward_beams:
                compatible_backward_beams.extend(
                    self._expand_beam(
                        right_track,
                        backward_positions,
                        available_tracks.difference(forward_beam["used"]),
                        c_arc,
                        c_trans,
                        bottleneck_scores,
                        candidate_sets,
                        transition_graph,
                        direction="backward",
                    )
                )
            meet = self._select_bidirectional_meet(
                midpoint,
                forward_beams,
                compatible_backward_beams,
                available_tracks,
                c_arc,
                c_trans,
                bottleneck_results,
                transition_graph,
            )

        if meet is None:
            anchor = self._select_anchor(
                midpoint,
                left_track,
                right_track,
                available_tracks,
                c_arc,
                c_trans,
                bottleneck_results,
                transition_graph,
            )
            playlist[midpoint] = anchor
            available_tracks.remove(anchor)
        else:
            forward_beam, backward_beam, anchor = meet
            for position, track in zip(forward_positions, forward_beam["path"]):
                playlist[position] = track
            for position, track in zip(backward_positions, backward_beam["path"]):
                playlist[position] = track
            playlist[midpoint] = anchor

            used_tracks = (
                forward_beam["used"]
                | backward_beam["used"]
                | {anchor}
            )
            available_tracks.difference_update(used_tracks)

        self._solve_interval(
            playlist,
            left_pos,
            midpoint,
            available_tracks,
            c_arc,
            c_trans,
            bottleneck_results,
            transition_graph,
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
            transition_graph,
            depth + 1,
        )

    def generate(
        self,
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        bottleneck_results: dict,
        transition_graph: dict[int, list[int]] | None = None,
        start_index: int | None = None,
        end_index: int | None = None,
    ) -> list[int]:
        """Generate a deterministic exact-once BIBS playlist."""
        number_of_tracks = self._validate_inputs(
            c_arc,
            c_trans,
            bottleneck_results,
            start_index,
            end_index,
        )
        if number_of_tracks == 1:
            only_track = 0 if start_index is None else int(start_index)
            if end_index is not None and end_index != only_track:
                raise ValueError("A single-track playlist can only use track 0.")
            return [only_track]

        available_tracks = set(range(number_of_tracks))
        if start_index is None:
            start_index = self._ordered_available_by_arc(
                available_tracks,
                0,
                c_arc,
            )[0]
        available_tracks.remove(int(start_index))

        if end_index is None:
            end_index = self._ordered_available_by_arc(
                available_tracks,
                number_of_tracks - 1,
                c_arc,
            )[0]
        elif end_index not in available_tracks:
            raise ValueError("start_index and end_index cannot be the same.")
        available_tracks.remove(int(end_index))

        playlist: list[int | None] = [None] * number_of_tracks
        playlist[0] = int(start_index)
        playlist[-1] = int(end_index)
        self._solve_interval(
            playlist,
            0,
            number_of_tracks - 1,
            available_tracks,
            c_arc,
            c_trans,
            bottleneck_results,
            transition_graph,
            depth=0,
        )

        if any(track is None for track in playlist):
            raise RuntimeError("BIBS left one or more playlist positions unfilled.")
        completed_playlist = [int(track) for track in playlist if track is not None]
        if (
            len(completed_playlist) != number_of_tracks
            or len(set(completed_playlist)) != number_of_tracks
            or available_tracks
        ):
            raise RuntimeError("BIBS did not place every track exactly once.")

        if self.config.enable_local_repair:
            completed_playlist = self._local_repair(
                completed_playlist,
                c_arc,
                c_trans,
            )
        if len(set(completed_playlist)) != number_of_tracks:
            raise RuntimeError("Local repair violated exact-once playlist validity.")
        return completed_playlist
