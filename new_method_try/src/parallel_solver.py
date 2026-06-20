"""Level-synchronous (round-based) parallel interval solver.

Realizes the project owner's parallelism vision safely: after the first split both
halves are processed "at the same time", then all four sub-intervals, and so on. The
intervals at a given recursion depth form one *level*; they all propose their anchor
picks independently from the shared track pool, then a single barrier + deterministic
arbitration resolves any conflicts:

    if two intervals want the same track, the interval where that track yields the lower
    anchor step-score keeps it; the loser re-proposes from the remaining tracks
    (repeat until no conflicts).

This keeps the uniqueness invariant intact and is reproducible given a seed (unlike
free-running threads, where the result would depend on scheduling). It is implemented as
a logically-parallel structure that runs sequentially; because each level's work is
independent after arbitration, it can later be mapped onto real processes if needed
(Python's GIL makes threads pointless for this numpy-light workload, and N<=300 runs in
well under a second).

Anchor step-score (lower is better):
    arc_weight * C_arc[track, midpoint]
  + transition_weight * (C_trans[left, track] + C_trans[track, right])
  - bottleneck_weight * normalized_bottleneck(track)   # omitted in "eligibility" mode
"""

from dataclasses import dataclass

import numpy as np

from src.sampling import softmax_topk_choice


@dataclass(frozen=True)
class ParallelSolverConfig:
    """Configuration for the level-synchronous parallel solver."""

    arc_weight: float = 1.0
    transition_weight: float = 1.0
    bottleneck_weight: float = 0.6
    bottleneck_mode: str = "score"  # "score" or "eligibility"
    base_case_size: int = 4
    stochastic: bool = False
    top_k: int = 5
    temperature: float = 0.15
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.bottleneck_mode not in ("score", "eligibility"):
            raise ValueError("bottleneck_mode must be 'score' or 'eligibility'.")
        if self.base_case_size <= 0:
            raise ValueError("base_case_size must be positive.")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative.")


@dataclass(frozen=True)
class _Interval:
    left_pos: int
    right_pos: int
    left_track: int
    right_track: int


class ParallelLevelSolver:
    """Solve a playlist with level-synchronous parallel anchor placement."""

    def __init__(self, config: ParallelSolverConfig) -> None:
        self.config = config
        self.level_trace: list[dict] = []

    def _anchor_score(
        self,
        track: int,
        midpoint: int,
        left_track: int,
        right_track: int,
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        song_scores: np.ndarray,
    ) -> float:
        score = (
            self.config.arc_weight * float(c_arc[track, midpoint])
            + self.config.transition_weight
            * (float(c_trans[left_track, track]) + float(c_trans[track, right_track]))
        )
        if self.config.bottleneck_mode == "score":
            score -= self.config.bottleneck_weight * float(
                np.clip(song_scores[track], 0.0, 1.0)
            )
        return float(score)

    def _propose(
        self,
        interval: _Interval,
        forbidden: set[int],
        available: set[int],
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        song_scores: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[int, float] | None:
        """Return this interval's best (track, score) anchor proposal, if any."""
        midpoint = (interval.left_pos + interval.right_pos) // 2
        candidates = [track for track in available if track not in forbidden]
        if not candidates:
            return None
        scores = np.asarray(
            [
                self._anchor_score(
                    track,
                    midpoint,
                    interval.left_track,
                    interval.right_track,
                    c_arc,
                    c_trans,
                    song_scores,
                )
                for track in candidates
            ]
        )
        if self.config.stochastic:
            pick = softmax_topk_choice(
                scores, 1, rng, self.config.top_k, self.config.temperature
            )
            index = pick[0] if pick else int(np.argmin(scores))
        else:
            index = int(np.argmin(scores))
        return candidates[index], float(scores[index])

    def _resolve_level(
        self,
        intervals: list[_Interval],
        available: set[int],
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        song_scores: np.ndarray,
        rng: np.random.Generator,
    ) -> dict[int, int]:
        """Return {interval_index: anchor_track} after barrier + arbitration."""
        assigned: dict[int, int] = {}
        claimed: set[int] = set()
        pending = list(range(len(intervals)))
        rounds = 0
        while pending:
            rounds += 1
            proposals: dict[int, tuple[int, float]] = {}
            for index in pending:
                proposal = self._propose(
                    intervals[index], claimed, available, c_arc, c_trans,
                    song_scores, rng,
                )
                if proposal is not None:
                    proposals[index] = proposal
            if not proposals:
                break
            # Group by proposed track; lowest score wins, ties by interval index.
            winners_by_track: dict[int, int] = {}
            for index, (track, score) in proposals.items():
                current = winners_by_track.get(track)
                if current is None or (score, index) < (
                    proposals[current][1],
                    current,
                ):
                    winners_by_track[track] = index
            new_pending: list[int] = []
            for index in pending:
                if index not in proposals:
                    continue
                track, _ = proposals[index]
                if winners_by_track.get(track) == index:
                    assigned[index] = track
                    claimed.add(track)
                else:
                    new_pending.append(index)
            pending = new_pending
        self.level_trace.append(
            {
                "intervals": len(intervals),
                "assigned": len(assigned),
                "arbitration_rounds": rounds,
            }
        )
        return assigned

    def _fill_base_case(
        self,
        interval: _Interval,
        playlist: list[int | None],
        available: set[int],
        c_arc: np.ndarray,
        c_trans: np.ndarray,
    ) -> None:
        """Sequentially fill a small interval, minimizing local transitions."""
        positions = list(range(interval.left_pos + 1, interval.right_pos))
        previous = interval.left_track
        for offset, position in enumerate(positions):
            is_last = offset == len(positions) - 1
            successor = interval.right_track if is_last else None

            def cost(track: int) -> float:
                value = (
                    self.config.arc_weight * float(c_arc[track, position])
                    + self.config.transition_weight * float(c_trans[previous, track])
                )
                if successor is not None:
                    value += self.config.transition_weight * float(
                        c_trans[track, successor]
                    )
                return value

            best = min(available, key=lambda track: (cost(track), track))
            playlist[position] = best
            available.remove(best)
            previous = best

    def generate(
        self,
        c_arc: np.ndarray,
        c_trans: np.ndarray,
        bottleneck_results: dict,
        start_index: int,
        end_index: int,
    ) -> list[int]:
        """Return an exact-once playlist via level-synchronous parallel solving."""
        number_of_tracks = c_arc.shape[0]
        if start_index == end_index:
            raise ValueError("start_index and end_index must differ.")
        self.level_trace = []
        rng = np.random.default_rng(self.config.random_seed)
        song_scores = np.asarray(bottleneck_results["song_bottleneck_scores"])
        playlist: list[int | None] = [None] * number_of_tracks
        playlist[0] = start_index
        playlist[-1] = end_index
        available = set(range(number_of_tracks)) - {start_index, end_index}

        level = [_Interval(0, number_of_tracks - 1, start_index, end_index)]
        base_queue: list[_Interval] = []

        while level:
            anchor_intervals = [
                interval
                for interval in level
                if interval.right_pos - interval.left_pos - 1 > self.config.base_case_size
            ]
            base_queue.extend(
                interval
                for interval in level
                if interval.right_pos - interval.left_pos - 1
                <= self.config.base_case_size
                and interval.right_pos - interval.left_pos - 1 > 0
            )
            if not anchor_intervals:
                break
            assigned = self._resolve_level(
                anchor_intervals, available, c_arc, c_trans, song_scores, rng
            )
            next_level: list[_Interval] = []
            for index, interval in enumerate(anchor_intervals):
                if index not in assigned:
                    # No anchor available: demote to base case for sequential fill.
                    base_queue.append(interval)
                    continue
                anchor = assigned[index]
                midpoint = (interval.left_pos + interval.right_pos) // 2
                playlist[midpoint] = anchor
                available.discard(anchor)
                next_level.append(
                    _Interval(interval.left_pos, midpoint, interval.left_track, anchor)
                )
                next_level.append(
                    _Interval(midpoint, interval.right_pos, anchor, interval.right_track)
                )
            level = next_level

        # Sequentially resolve all base-case intervals (leaves), pool shrinks as we go.
        for interval in base_queue:
            self._fill_base_case(interval, playlist, available, c_arc, c_trans)

        if available or any(track is None for track in playlist):
            raise ValueError("Parallel solver failed to place every track exactly once.")
        result = [int(track) for track in playlist]
        if len(set(result)) != number_of_tracks:
            raise ValueError("Parallel solver produced a non-exact-once playlist.")
        return result
