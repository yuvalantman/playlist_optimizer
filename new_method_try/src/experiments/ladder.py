"""Ablation ladder: run every construction method on one pool and compare metrics.

Run from the new_method_try/ directory:
    python -m src.experiments.ladder

Reports the original four metrics (arc_rmse, total/avg transition cost, global coherence)
plus the new additive metrics: worst-case transition percentiles (p90/p95/max), DTW
shape distance vs the target arc, Flexer ratio-adherence RMSE, and (for stochastic
methods) run-to-run diversity. Transition-only greedy and random are not arc-optimized,
so their arc_rmse should be ignored in comparisons.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.arcs import TargetArcConfig, create_target_arc
import time

from src.baselines import (
    ArcAssignmentBaseline,
    FlexerInterpolationBaseline,
    ForwardBeamBaseline,
    MMBeamBaseline,
    RandomBaseline,
    TransitionGreedyBaseline,
)
from src.baselines.mm_beam import MMBeamConfig
from src.baselines.flexer_interp import relative_positions
from src.bibs import BIBS, BIBSConfig
from src.bottleneck_detector import BottleneckConfig, BottleneckDetector
from src.candidate_orchestrator import CandidateOrchestrator, CandidateOrchestratorConfig
from src.cost_functions import compute_transition_cost_matrix
from src.data_preparation import TrackPoolBuilder, TrackPoolConfig
from src.evaluator import (
    EvaluationConfig,
    PlaylistEvaluator,
    dtw_shape_distance,
    playlist_diversity,
    ratio_adherence_rmse,
    transition_percentiles,
)
from src.feature_engineering import (
    EnergyValenceConfig,
    StartEndSelectionConfig,
    compute_arc_cost_matrix,
    compute_ev_score,
    select_start_end_tracks,
)
from src.greedy_baseline import GreedyBaselineConfig, GreedyPlaylistBaseline
from src.repair import RepairConfig, repair_playlist
from src.transition_graph import TransitionGraphBuilder, TransitionGraphConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_CSV = PROJECT_ROOT / "data" / "spotify_tracks.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "results" / "ladder"
STOCHASTIC_SEEDS = (1, 2, 3, 4, 5)


def build_inputs(genre: str, pool_size: int, seed: int) -> dict:
    """Build the shared pipeline inputs for one (genre, pool_size, seed)."""
    track_pool = TrackPoolBuilder(
        TrackPoolConfig(
            csv_path=DATA_CSV, genre=genre, pool_size=pool_size, random_seed=seed
        )
    ).build()
    track_pool = compute_ev_score(track_pool, EnergyValenceConfig())
    c_trans = compute_transition_cost_matrix(track_pool)
    start_index, end_index = select_start_end_tracks(
        track_pool, c_trans, StartEndSelectionConfig(random_seed=seed)
    )
    target_arc = create_target_arc(
        track_pool, start_index, end_index, TargetArcConfig(shape="linear")
    )
    c_arc = compute_arc_cost_matrix(track_pool, target_arc)
    bottleneck_results = BottleneckDetector(BottleneckConfig()).detect(c_arc)
    graph_data = TransitionGraphBuilder(
        TransitionGraphConfig()
    ).build_transition_graph_data(c_trans)
    return {
        "track_pool": track_pool,
        "c_trans": c_trans,
        "c_arc": c_arc,
        "target_arc": target_arc,
        "start_index": start_index,
        "end_index": end_index,
        "bottleneck_results": bottleneck_results,
        "graph_data": graph_data,
    }


def _run_bibs(config: BIBSConfig, inp: dict) -> list[int]:
    return BIBS(config).generate(
        inp["c_arc"],
        inp["c_trans"],
        inp["bottleneck_results"],
        graph_data=inp["graph_data"],
        candidate_orchestrator=CandidateOrchestrator(CandidateOrchestratorConfig()),
        target_arc=inp["target_arc"],
        track_pool=inp["track_pool"],
        start_index=inp["start_index"],
        end_index=inp["end_index"],
    )


def run_ladder(
    genre: str = "pop",
    pool_size: int = 150,
    seed: int = 42,
    include_tempo_variants: bool = False,
) -> pd.DataFrame:
    """Run all methods on one pool and return a comparison table."""
    inp = build_inputs(genre, pool_size, seed)
    track_pool = inp["track_pool"]
    c_arc, c_trans = inp["c_arc"], inp["c_trans"]
    start_index, end_index = inp["start_index"], inp["end_index"]
    graph_data, bottleneck_results = inp["graph_data"], inp["bottleneck_results"]
    target_arc = inp["target_arc"]
    number_of_tracks = len(track_pool)
    evaluator = PlaylistEvaluator(EvaluationConfig())
    ev_scores = track_pool["EV_score"].to_numpy(dtype=float)
    rel_pos = relative_positions(track_pool, start_index, end_index)

    def evaluate_full(playlist: list[int]) -> dict[str, float]:
        evaluator.validate_playlist(playlist, number_of_tracks)
        metrics = evaluator.evaluate(playlist, c_arc, c_trans)
        metrics.update(transition_percentiles(playlist, c_trans))
        metrics["dtw_shape"] = dtw_shape_distance(playlist, ev_scores, target_arc)
        metrics["ratio_adherence_rmse"] = ratio_adherence_rmse(playlist, rel_pos)
        return metrics

    def repair(playlist: list[int]) -> list[int]:
        repaired, _ = repair_playlist(playlist, c_arc, c_trans, RepairConfig())
        return repaired

    rows: list[dict[str, object]] = []

    def add_row(
        name: str,
        playlist: list[int],
        arc_optimized: bool,
        diversity=np.nan,
        wall_time_s=np.nan,
    ):
        metrics = evaluate_full(playlist)
        rows.append(
            {
                "method": name,
                "arc_optimized": arc_optimized,
                "arc_rmse": metrics["arc_rmse"],
                "total_transition_cost": metrics["total_transition_cost"],
                "average_transition_cost": metrics["average_transition_cost"],
                "global_coherence": metrics["global_coherence"],
                "p95_transition": metrics["p95_transition"],
                "max_transition": metrics["p100_transition"],
                "dtw_shape": metrics["dtw_shape"],
                "ratio_adherence_rmse": metrics["ratio_adherence_rmse"],
                "diversity": diversity,
                "wall_time_s": wall_time_s,
            }
        )

    def timed_add(name: str, fn, arc_optimized: bool, diversity=np.nan):
        t0 = time.perf_counter()
        playlist = fn()
        elapsed = time.perf_counter() - t0
        add_row(name, playlist, arc_optimized, diversity=diversity, wall_time_s=elapsed)

    # --- baselines ---
    timed_add("random", lambda: RandomBaseline(seed=seed).generate(number_of_tracks, start_index, end_index), False)
    timed_add("transition_greedy", lambda: TransitionGreedyBaseline().generate(c_trans, start_index, end_index, graph_data), False)
    timed_add("arc_assignment", lambda: ArcAssignmentBaseline().generate(c_arc, start_index, end_index), True)
    timed_add("flexer_interp", lambda: FlexerInterpolationBaseline().generate(track_pool, start_index, end_index), True)
    timed_add("forward_beam", lambda: ForwardBeamBaseline().generate(c_arc, c_trans, start_index, end_index), True)
    timed_add("mm_beam", lambda: MMBeamBaseline().generate(c_arc, c_trans, start_index, end_index), True)
    greedy = GreedyPlaylistBaseline(GreedyBaselineConfig()).generate(
        c_arc, c_trans, start_index=start_index, end_index=end_index,
        transition_graph=graph_data,
    )
    add_row("greedy_arc_trans", greedy, True)
    add_row("greedy_arc_trans+repair", repair(greedy), True)

    # --- BIBS variants (deterministic) ---
    bibs_current = _run_bibs(BIBSConfig(), inp)
    add_row("bibs_current", bibs_current, True)
    add_row("bibs_current+repair", repair(bibs_current), True)
    timed_add("bibs_eligibility", lambda: _run_bibs(BIBSConfig(bottleneck_mode="eligibility"), inp), True)
    timed_add("bibs_commit", lambda: _run_bibs(BIBSConfig(commit_beams=True), inp), True)

    # --- BIBS stochastic (multi-seed: report mean metrics + diversity) ---
    def stochastic_block(name: str, base_config: dict, with_repair: bool) -> None:
        playlists = [
            _run_bibs(BIBSConfig(stochastic=True, random_seed=s, **base_config), inp)
            for s in STOCHASTIC_SEEDS
        ]
        if with_repair:
            playlists = [repair(p) for p in playlists]
        metric_dicts = [evaluate_full(p) for p in playlists]
        diversity = playlist_diversity(playlists)
        mean = {
            key: float(np.mean([m[key] for m in metric_dicts]))
            for key in metric_dicts[0]
        }
        rows.append(
            {
                "method": name,
                "arc_optimized": True,
                "arc_rmse": mean["arc_rmse"],
                "total_transition_cost": mean["total_transition_cost"],
                "average_transition_cost": mean["average_transition_cost"],
                "global_coherence": mean["global_coherence"],
                "p95_transition": mean["p95_transition"],
                "max_transition": mean["p100_transition"],
                "dtw_shape": mean["dtw_shape"],
                "ratio_adherence_rmse": mean["ratio_adherence_rmse"],
                "diversity": diversity,
                "wall_time_s": np.nan,
            }
        )

    stochastic_block("bibs_stochastic(mean5)", {}, with_repair=False)
    stochastic_block(
        "bibs_stoch_elig+repair(mean5)",
        {"bottleneck_mode": "eligibility"},
        with_repair=True,
    )

    if include_tempo_variants:
        # Tempo-aware BIBS: BPM-jump penalty (threshold 0.30)
        timed_add(
            "bibs_tempo_aware",
            lambda: _run_bibs(BIBSConfig(tempo_jump_threshold=0.30, tempo_jump_penalty=3.0), inp),
            True,
        )
        timed_add(
            "bibs_tempo_aware+repair",
            lambda: repair(_run_bibs(BIBSConfig(tempo_jump_threshold=0.30, tempo_jump_penalty=3.0), inp)),
            True,
        )

        # Alpha-boost: heavier rhythm weight in C_trans
        c_trans_alpha2 = compute_transition_cost_matrix(track_pool, alpha=2.0, beta=0.4, gamma=0.6)
        inp_alpha2 = {**inp, "c_trans": c_trans_alpha2}
        timed_add("bibs_alpha2", lambda: _run_bibs(BIBSConfig(), inp_alpha2), True)
        timed_add(
            "mm_beam_alpha2",
            lambda: MMBeamBaseline().generate(c_arc, c_trans_alpha2, start_index, end_index),
            True,
        )

        # Stochastic MM beam
        timed_add(
            "mm_beam_stochastic",
            lambda: MMBeamBaseline(config=MMBeamConfig(stochastic=True, random_seed=seed)).generate(
                c_arc, c_trans, start_index, end_index
            ),
            True,
        )

    return pd.DataFrame(rows)


def main() -> None:
    genre, pool_size, seed = "pop", 150, 42
    frame = run_ladder(genre, pool_size, seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"ladder_{genre}_{pool_size}_seed{seed}.csv"
    frame.to_csv(out_path, index=False)
    print(f"Ladder: genre={genre} pool_size={pool_size} seed={seed}")
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved: {out_path}")
    print(
        "\nNotes: ignore arc_rmse for random/transition_greedy (not arc-optimized); "
        "arc_assignment is the arc_rmse lower bound; diversity is NaN for deterministic "
        "methods and in [0,1] for stochastic (higher = more varied orderings)."
    )


if __name__ == "__main__":
    main()
