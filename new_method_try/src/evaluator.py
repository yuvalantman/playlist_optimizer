"""Evaluation metrics for complete playlist sequences."""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for playlist evaluation metrics."""
    pass


class PlaylistEvaluator:
    """Validate and evaluate complete playlist sequences."""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    @staticmethod
    def validate_playlist(playlist: list[int], number_of_tracks: int) -> None:
        """Validate that a playlist contains each valid track exactly once."""
        if number_of_tracks <= 0:
            raise ValueError("number_of_tracks must be positive.")
        if len(playlist) != number_of_tracks:
            raise ValueError("Playlist length must equal number_of_tracks.")
        if any(
            not isinstance(track_index, (int, np.integer))
            or not 0 <= track_index < number_of_tracks
            for track_index in playlist
        ):
            raise ValueError("Playlist contains an invalid track index.")
        if len(set(playlist)) != number_of_tracks:
            raise ValueError("Every track must appear exactly once in the playlist.")

    @staticmethod
    def _validate_arc_matrix(c_arc: np.ndarray) -> int:
        if not isinstance(c_arc, np.ndarray) or c_arc.ndim != 2:
            raise ValueError("c_arc must be a 2D numpy array.")
        if c_arc.shape[0] == 0 or c_arc.shape[0] != c_arc.shape[1]:
            raise ValueError("c_arc must have one playlist position per track.")
        if not np.issubdtype(c_arc.dtype, np.number) or not np.isfinite(c_arc).all():
            raise ValueError("c_arc must contain only finite numeric values.")
        return c_arc.shape[0]

    @staticmethod
    def _validate_transition_matrix(c_trans: np.ndarray) -> int:
        if not isinstance(c_trans, np.ndarray) or c_trans.ndim != 2:
            raise ValueError("c_trans must be a 2D numpy array.")
        if c_trans.shape[0] == 0 or c_trans.shape[0] != c_trans.shape[1]:
            raise ValueError("c_trans must be a non-empty square matrix.")
        if not np.issubdtype(c_trans.dtype, np.number) or not np.isfinite(
            c_trans
        ).all():
            raise ValueError("c_trans must contain only finite numeric values.")
        return c_trans.shape[0]

    def compute_arc_rmse(self, playlist: list[int], c_arc: np.ndarray) -> float:
        """Return RMSE of assigned track-position arc costs."""
        number_of_tracks = self._validate_arc_matrix(c_arc)
        self.validate_playlist(playlist, number_of_tracks)
        positions = np.arange(number_of_tracks)
        costs = c_arc[np.asarray(playlist, dtype=int), positions]
        return float(np.sqrt(np.mean(np.square(costs))))

    def compute_total_transition_cost(
        self,
        playlist: list[int],
        c_trans: np.ndarray,
    ) -> float:
        """Return the sum of costs between all consecutive playlist tracks."""
        number_of_tracks = self._validate_transition_matrix(c_trans)
        self.validate_playlist(playlist, number_of_tracks)
        if number_of_tracks == 1:
            return 0.0

        playlist_indices = np.asarray(playlist, dtype=int)
        return float(c_trans[playlist_indices[:-1], playlist_indices[1:]].sum())

    def get_transition_costs_by_position(
        self,
        playlist: list[int],
        c_trans: np.ndarray,
    ) -> list[float]:
        """Return one transition cost for every consecutive playlist pair."""
        number_of_tracks = self._validate_transition_matrix(c_trans)
        self.validate_playlist(playlist, number_of_tracks)
        if number_of_tracks == 1:
            return []

        playlist_indices = np.asarray(playlist, dtype=int)
        return c_trans[
            playlist_indices[:-1],
            playlist_indices[1:],
        ].astype(float).tolist()

    def get_worst_transitions(
        self,
        playlist: list[int],
        c_trans: np.ndarray,
        top_k: int = 10,
    ) -> list[dict]:
        """Return the highest-cost consecutive transitions."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        transition_costs = self.get_transition_costs_by_position(
            playlist,
            c_trans,
        )
        transitions = [
            {
                "position": position,
                "from_track_index": int(playlist[position]),
                "to_track_index": int(playlist[position + 1]),
                "transition_cost": cost,
            }
            for position, cost in enumerate(transition_costs)
        ]
        transitions.sort(
            key=lambda transition: (
                -transition["transition_cost"],
                transition["position"],
            )
        )
        return transitions[:top_k]

    @staticmethod
    def attribute_decision_position(
        position: int,
        decision_trace: list[dict],
    ) -> str:
        """Attribute a transition position to its nearest major BIBS decision."""
        anchor_positions = {
            int(decision["midpoint"])
            for decision in decision_trace
            if decision.get("decision_type") == "anchor"
            and decision.get("midpoint") is not None
        }
        if position in anchor_positions or position + 1 in anchor_positions:
            return "near_anchor"
        for decision in decision_trace:
            if decision.get("decision_type") != "base_case":
                continue
            left = int(decision["interval_left_pos"])
            right = int(decision["interval_right_pos"])
            if left <= position < right:
                return "base_case"
        return "recursive_boundary"

    @staticmethod
    def _camelot_jump(source_code: str, target_code: str) -> int:
        source_number = int(
            "".join(character for character in str(source_code) if character.isdigit())
        )
        target_number = int(
            "".join(character for character in str(target_code) if character.isdigit())
        )
        direct = abs(source_number - target_number)
        return min(direct, 12 - direct)

    def get_worst_transition_diagnostics(
        self,
        playlist: list[int],
        c_trans: np.ndarray,
        track_pool: pd.DataFrame,
        decision_trace: list[dict],
        top_k: int = 15,
    ) -> list[dict]:
        """Return worst transitions with track metadata and decision attribution."""
        records: list[dict] = []
        for transition in self.get_worst_transitions(playlist, c_trans, top_k):
            position = int(transition["position"])
            source = int(transition["from_track_index"])
            target = int(transition["to_track_index"])
            source_row = track_pool.loc[source]
            target_row = track_pool.loc[target]
            source_ev = float(source_row["EV_score"])
            target_ev = float(target_row["EV_score"])
            source_camelot = str(source_row["camelot"])
            target_camelot = str(target_row["camelot"])
            records.append(
                {
                    "position": position,
                    "from_track_index": source,
                    "from_track_name": str(source_row["track_name"]),
                    "from_EV_score": source_ev,
                    "from_camelot": source_camelot,
                    "to_track_index": target,
                    "to_track_name": str(target_row["track_name"]),
                    "to_EV_score": target_ev,
                    "to_camelot": target_camelot,
                    "transition_cost": float(transition["transition_cost"]),
                    "EV_delta": target_ev - source_ev,
                    "camelot_jump": self._camelot_jump(
                        source_camelot,
                        target_camelot,
                    ),
                    "nearest_decision_type": self.attribute_decision_position(
                        position,
                        decision_trace,
                    ),
                }
            )
        return records

    def get_worst_energy_drops(
        self,
        playlist: list[int],
        track_pool: pd.DataFrame,
        target_arc: np.ndarray,
        c_trans: np.ndarray,
        decision_trace: list[dict],
        top_k: int = 15,
    ) -> list[dict]:
        """Return the largest consecutive EV drops with decision attribution."""
        self.validate_playlist(playlist, len(track_pool))
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        target = np.asarray(target_arc, dtype=float)
        if target.shape != (len(playlist),):
            raise ValueError("target_arc must contain one value per playlist position.")
        records: list[dict] = []
        for position, (source, target_track) in enumerate(
            zip(playlist, playlist[1:])
        ):
            source_row = track_pool.loc[source]
            target_row = track_pool.loc[target_track]
            source_ev = float(source_row["EV_score"])
            target_ev = float(target_row["EV_score"])
            records.append(
                {
                    "position": position,
                    "from_track": int(source),
                    "to_track": int(target_track),
                    "from_EV_score": source_ev,
                    "to_EV_score": target_ev,
                    "EV_drop": source_ev - target_ev,
                    "target_arc_value_at_position": float(target[position]),
                    "from_camelot": str(source_row["camelot"]),
                    "to_camelot": str(target_row["camelot"]),
                    "transition_cost": float(c_trans[source, target_track]),
                    "nearest_decision_type": self.attribute_decision_position(
                        position,
                        decision_trace,
                    ),
                }
            )
        records.sort(key=lambda record: (-record["EV_drop"], record["position"]))
        return records[:top_k]

    def get_anchor_diagnostics(
        self,
        anchor_history: list[dict],
        track_pool: pd.DataFrame,
    ) -> list[dict]:
        """Return anchors ordered from worst to best alignment cost."""
        records: list[dict] = []
        for anchor in anchor_history:
            track = int(anchor["anchor_track"])
            track_row = track_pool.loc[track]
            records.append(
                {
                    "position": int(anchor["position"]),
                    "anchor_track": track,
                    "track_name": str(track_row["track_name"]),
                    "EV_score": float(track_row["EV_score"]),
                    "camelot": str(track_row["camelot"]),
                    "arc_cost": float(anchor["arc_cost"]),
                    "left_transition_cost": float(
                        anchor["left_transition_cost"]
                    ),
                    "right_transition_cost": float(
                        anchor["right_transition_cost"]
                    ),
                    "anchor_alignment_cost": float(
                        anchor["anchor_alignment_cost"]
                    ),
                }
            )
        records.sort(
            key=lambda record: (
                -record["anchor_alignment_cost"],
                record["position"],
            )
        )
        return records

    @staticmethod
    def compute_anchor_quality_summary(
        anchor_history: list[dict],
    ) -> dict[str, object]:
        """Return aggregate anchor placement quality diagnostics."""
        if not anchor_history:
            return {
                "number_of_anchors": 0,
                "mean_anchor_arc_cost": 0.0,
                "mean_anchor_left_transition_cost": 0.0,
                "mean_anchor_right_transition_cost": 0.0,
                "mean_anchor_alignment_cost": 0.0,
                "worst_10_anchor_alignment_costs": [],
            }
        alignment_costs = sorted(
            (
                float(anchor["anchor_alignment_cost"])
                for anchor in anchor_history
            ),
            reverse=True,
        )
        return {
            "number_of_anchors": len(anchor_history),
            "mean_anchor_arc_cost": float(
                np.mean([anchor["arc_cost"] for anchor in anchor_history])
            ),
            "mean_anchor_left_transition_cost": float(
                np.mean(
                    [
                        anchor["left_transition_cost"]
                        for anchor in anchor_history
                    ]
                )
            ),
            "mean_anchor_right_transition_cost": float(
                np.mean(
                    [
                        anchor["right_transition_cost"]
                        for anchor in anchor_history
                    ]
                )
            ),
            "mean_anchor_alignment_cost": float(
                np.mean(alignment_costs)
            ),
            "worst_10_anchor_alignment_costs": alignment_costs[:10],
        }

    def compute_global_coherence(
        self,
        playlist: list[int],
        c_trans: np.ndarray,
    ) -> float:
        """Measure sequential smoothness relative to all pool transitions."""
        number_of_tracks = self._validate_transition_matrix(c_trans)
        self.validate_playlist(playlist, number_of_tracks)
        if number_of_tracks == 1:
            return 1.0

        sequential = np.square(
            np.asarray(self.get_transition_costs_by_position(playlist, c_trans))
        ).mean()
        off_diagonal = c_trans[~np.eye(number_of_tracks, dtype=bool)]
        population = np.square(off_diagonal).mean()
        if population == 0:
            return 1.0 if sequential == 0 else 0.0
        return float(1.0 - sequential / population)

    def get_camelot_sequence(
        self,
        playlist: list[int],
        track_pool: pd.DataFrame,
    ) -> list[str]:
        """Return Camelot codes in playlist order."""
        self.validate_playlist(playlist, len(track_pool))
        if "camelot" not in track_pool.columns:
            raise ValueError("track_pool must contain a camelot column.")
        return track_pool.loc[playlist, "camelot"].astype(str).tolist()

    @staticmethod
    def _run_lengths(values: list[str]) -> list[int]:
        if not values:
            return []
        lengths: list[int] = []
        current_length = 1
        for previous, current in zip(values, values[1:]):
            if current == previous:
                current_length += 1
            else:
                lengths.append(current_length)
                current_length = 1
        lengths.append(current_length)
        return lengths

    def compute_harmonic_stagnation_metrics(
        self,
        playlist: list[int],
        track_pool: pd.DataFrame,
    ) -> dict[str, float | int]:
        """Return diagnostics for repeated Camelot regions."""
        camelot_sequence = self.get_camelot_sequence(playlist, track_pool)
        if "camelot_number" in track_pool.columns:
            number_sequence = (
                track_pool.loc[playlist, "camelot_number"].astype(str).tolist()
            )
        else:
            number_sequence = [
                "".join(character for character in code if character.isdigit())
                for code in camelot_sequence
            ]

        exact_runs = self._run_lengths(camelot_sequence)
        number_runs = self._run_lengths(number_sequence)
        return {
            "longest_exact_camelot_run": max(exact_runs, default=0),
            "longest_camelot_number_run": max(number_runs, default=0),
            "average_exact_camelot_run_length": (
                float(np.mean(exact_runs)) if exact_runs else 0.0
            ),
            "number_of_runs_longer_than_3": sum(
                run_length > 3 for run_length in exact_runs
            ),
            "number_of_runs_longer_than_5": sum(
                run_length > 5 for run_length in exact_runs
            ),
            "unique_camelot_count": len(set(camelot_sequence)),
            "unique_camelot_count_first_50": len(set(camelot_sequence[:50])),
        }

    def compute_harmonic_motion_diagnostics(
        self,
        playlist: list[int],
        track_pool: pd.DataFrame,
        max_preferred_camelot_jump: int = 2,
    ) -> dict[str, object]:
        """Return circular Camelot-number motion diagnostics."""
        self.validate_playlist(playlist, len(track_pool))
        if max_preferred_camelot_jump < 0:
            raise ValueError("max_preferred_camelot_jump must be non-negative.")
        camelot_sequence = self.get_camelot_sequence(playlist, track_pool)
        if "camelot_number" in track_pool.columns:
            numbers = (
                track_pool.loc[playlist, "camelot_number"]
                .astype(int)
                .to_numpy()
            )
        else:
            numbers = np.asarray(
                [
                    int("".join(character for character in code if character.isdigit()))
                    for code in camelot_sequence
                ]
            )
        direct = np.abs(np.diff(numbers))
        distances = np.minimum(direct, 12 - direct).astype(int)
        large_positions = np.flatnonzero(
            distances > max_preferred_camelot_jump
        ).tolist()
        transitions = [
            {
                "position": position,
                "from_camelot": camelot_sequence[position],
                "to_camelot": camelot_sequence[position + 1],
                "camelot_distance": int(distances[position]),
            }
            for position in range(min(30, len(distances)))
        ]
        return {
            "max_camelot_jump": int(distances.max()) if len(distances) else 0,
            "number_of_large_camelot_jumps": len(large_positions),
            "average_camelot_jump": (
                float(distances.mean()) if len(distances) else 0.0
            ),
            "positions_of_large_camelot_jumps": large_positions,
            "first_30_camelot_transitions": transitions,
            "first_30_camelot_transition_distances": distances[:30].tolist(),
        }

    def compute_energy_flow_diagnostics(
        self,
        playlist: list[int],
        track_pool: pd.DataFrame,
        high_energy_threshold: float = 0.65,
        large_drop_threshold: float = 0.12,
    ) -> dict[str, float | int]:
        """Return diagnostics for EV changes along a playlist."""
        self.validate_playlist(playlist, len(track_pool))
        if "EV_score" not in track_pool.columns:
            raise ValueError("track_pool must contain an EV_score column.")
        ev_scores = track_pool.loc[playlist, "EV_score"].to_numpy(dtype=float)
        deltas = np.diff(ev_scores)
        drops = -deltas
        previous_high = ev_scores[:-1] >= high_energy_threshold
        large_drops = drops > large_drop_threshold
        return {
            "max_ev_drop": float(max(0.0, drops.max(initial=0.0))),
            "number_of_large_ev_drops": int(large_drops.sum()),
            "number_of_large_ev_drops_after_high_energy": int(
                np.logical_and(large_drops, previous_high).sum()
            ),
            "average_ev_delta": float(deltas.mean()) if len(deltas) else 0.0,
            "min_ev_delta": float(deltas.min()) if len(deltas) else 0.0,
            "max_ev_delta": float(deltas.max()) if len(deltas) else 0.0,
        }

    def compute_tempo_flow_diagnostics(
        self,
        playlist: list[int],
        track_pool: pd.DataFrame,
        target_tempo_arc: np.ndarray | None = None,
        large_tempo_jump_threshold: float = 35.0,
    ) -> dict[str, float | int]:
        """Return diagnostics for absolute BPM changes along a playlist."""
        self.validate_playlist(playlist, len(track_pool))
        if "tempo" not in track_pool.columns:
            raise ValueError("track_pool must contain a tempo column.")
        if large_tempo_jump_threshold <= 0:
            raise ValueError("large_tempo_jump_threshold must be positive.")

        tempos = pd.to_numeric(
            track_pool.loc[playlist, "tempo"],
            errors="coerce",
        ).to_numpy(dtype=float)
        if not np.isfinite(tempos).all() or np.any(tempos <= 0):
            raise ValueError("Playlist tempos must be finite and positive.")

        jumps = np.abs(np.diff(tempos))
        diagnostics: dict[str, float | int] = {
            "max_tempo_jump": float(jumps.max()) if len(jumps) else 0.0,
            "average_tempo_jump": float(jumps.mean()) if len(jumps) else 0.0,
            "number_of_large_tempo_jumps": int(
                (jumps > large_tempo_jump_threshold).sum()
            ),
            "tempo_start": float(tempos[0]),
            "tempo_end": float(tempos[-1]),
            "playlist_tempo_min": float(tempos.min()),
            "playlist_tempo_max": float(tempos.max()),
        }
        if target_tempo_arc is not None:
            target = np.asarray(target_tempo_arc, dtype=float)
            if target.shape != (len(playlist),) or not np.isfinite(target).all():
                raise ValueError(
                    "target_tempo_arc must contain one finite value per track."
                )
            diagnostics["tempo_arc_rmse"] = float(
                np.sqrt(np.mean(np.square(tempos - target)))
            )
        return diagnostics

    @staticmethod
    def compute_anchor_alignment_cost(anchor_history: list[dict] | None) -> float | None:
        """Return mean anchor alignment cost, or None when unavailable."""
        if not anchor_history:
            return None
        costs = [
            float(anchor["anchor_alignment_cost"])
            for anchor in anchor_history
            if "anchor_alignment_cost" in anchor
        ]
        return float(np.mean(costs)) if costs else None

    def evaluate(
        self,
        playlist: list[int],
        c_arc: np.ndarray,
        c_trans: np.ndarray,
    ) -> dict[str, float]:
        """Return the official playlist evaluation metrics."""
        number_of_tracks = self._validate_arc_matrix(c_arc)
        transition_tracks = self._validate_transition_matrix(c_trans)
        if number_of_tracks != transition_tracks:
            raise ValueError("c_arc and c_trans must have the same number of tracks.")
        self.validate_playlist(playlist, number_of_tracks)

        arc_rmse = self.compute_arc_rmse(playlist, c_arc)
        total_transition_cost = self.compute_total_transition_cost(
            playlist,
            c_trans,
        )
        number_of_transitions = number_of_tracks - 1

        return {
            "arc_rmse": arc_rmse,
            "total_transition_cost": total_transition_cost,
            "average_transition_cost": (
                total_transition_cost / number_of_transitions
                if number_of_transitions
                else 0.0
            ),
            "global_coherence": self.compute_global_coherence(
                playlist,
                c_trans,
            ),
        }


# ---------------------------------------------------------------------------
# Additive metrics for the new-method experiments. These augment (never replace)
# the original metrics above.
# ---------------------------------------------------------------------------


def dtw_distance(sequence_a: np.ndarray, sequence_b: np.ndarray) -> float:
    """Return the Dynamic Time Warping distance between two 1-D sequences.

    Used only as an auxiliary shape-adherence metric (never in optimization): it tells
    us whether a playlist follows the arc *shape* even if slightly misaligned, which
    arc_rmse (strict positional) cannot. O(L^2) DP, |value| differences.
    """
    a = np.asarray(sequence_a, dtype=float)
    b = np.asarray(sequence_b, dtype=float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        raise ValueError("DTW requires non-empty sequences.")
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            local = abs(a[i - 1] - b[j - 1])
            cost[i, j] = local + min(
                cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]
            )
    return float(cost[n, m])


def dtw_shape_distance(
    playlist: list[int],
    ev_scores: np.ndarray,
    target_arc: np.ndarray,
) -> float:
    """Return the DTW distance between the playlist EV trajectory and target arc."""
    ev = np.asarray(ev_scores, dtype=float)[np.asarray(playlist, dtype=int)]
    return dtw_distance(ev, np.asarray(target_arc, dtype=float))


def transition_percentiles(
    playlist: list[int],
    c_trans: np.ndarray,
    percentiles: tuple[float, ...] = (90.0, 95.0, 100.0),
) -> dict[str, float]:
    """Return worst-case transition percentiles along the playlist."""
    indices = np.asarray(playlist, dtype=int)
    costs = c_trans[indices[:-1], indices[1:]].astype(float)
    return {
        f"p{int(p)}_transition": float(np.percentile(costs, p))
        for p in percentiles
    }


def ratio_adherence_rmse(
    playlist: list[int],
    relative_positions: np.ndarray,
) -> float:
    """RMSE between each track's acoustic relative position and its target ratio.

    ``relative_positions`` is the Flexer pos(i) in [0, 1] per track (see
    baselines.flexer_interp.relative_positions). The target for playlist position t is
    ``t / (L - 1)``. Auxiliary "acoustic-interpolation adherence" metric.
    """
    length = len(playlist)
    targets = np.arange(length, dtype=float) / (length - 1)
    actual = np.asarray(relative_positions, dtype=float)[
        np.asarray(playlist, dtype=int)
    ]
    return float(np.sqrt(np.mean(np.square(actual - targets))))


def playlist_diversity(playlists: list[list[int]]) -> float:
    """Mean pairwise position-disagreement across a set of playlists in [0, 1].

    0 means all runs are identical orderings; values near 1 mean highly varied
    orderings. Quantifies the novelty/diversity benefit of stochastic methods.
    """
    count = len(playlists)
    if count < 2:
        return 0.0
    length = len(playlists[0])
    arrays = [np.asarray(p, dtype=int) for p in playlists]
    total = 0.0
    pairs = 0
    for i in range(count):
        for j in range(i + 1, count):
            total += float(np.mean(arrays[i] != arrays[j]))
            pairs += 1
    return total / pairs if pairs else 0.0
