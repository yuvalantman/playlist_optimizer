"""Reusable experiment runner used by the Streamlit dashboard and the CLI experiments.

Generalizes the single-genre `ladder.build_inputs` to arbitrary track subsets
(all-genres random, similar-genre groups, or user-chosen genres), configurable arc
shapes, and a uniform "run every method + evaluate" interface.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.arcs import SHAPES, TargetArcConfig, create_target_arc
from src.baselines import (
    ArcAssignmentBaseline,
    FlexerInterpolationBaseline,
    ForwardBeamBaseline,
    MMBeamBaseline,
    RandomBaseline,
    TransitionGreedyBaseline,
)
from src.baselines.mm_beam import MMBeamConfig
from src.cost_functions import TransitionCostConfig
from src.baselines.flexer_interp import feature_distance_matrix, relative_positions
from src.bibs import BIBS, BIBSConfig
from src.bottleneck_detector import BottleneckConfig, BottleneckDetector
from src.candidate_orchestrator import CandidateOrchestrator, CandidateOrchestratorConfig
from src.cost_functions import compute_transition_cost_matrix
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
from src.parallel_solver import ParallelLevelSolver, ParallelSolverConfig
from src.repair import RepairConfig, repair_playlist
from src.transition_graph import TransitionGraphBuilder, TransitionGraphConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_CSV = PROJECT_ROOT / "data" / "spotify_tracks.csv"

# Audio features used to measure genre similarity (normalized before distance).
_GENRE_FEATURES = (
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
)


# --------------------------------------------------------------------------- #
# Catalog / genre helpers
# --------------------------------------------------------------------------- #
def load_catalog(csv_path: Path | str = DATA_CSV) -> pd.DataFrame:
    """Load the full track catalog."""
    return pd.read_csv(csv_path)


def genre_counts(catalog: pd.DataFrame) -> pd.Series:
    """Return number of tracks per genre, descending."""
    return catalog["track_genre"].value_counts()


def genre_centroids(catalog: pd.DataFrame) -> pd.DataFrame:
    """Return per-genre normalized-feature centroids (for similarity)."""
    features = [c for c in _GENRE_FEATURES if c in catalog.columns]
    values = catalog[features].apply(pd.to_numeric, errors="coerce")
    normalized = (values - values.min()) / (values.max() - values.min()).replace(0, 1.0)
    normalized["track_genre"] = catalog["track_genre"].values
    return normalized.groupby("track_genre").mean()


def similar_genres(
    catalog: pd.DataFrame,
    genre: str,
    count: int = 3,
    centroids: pd.DataFrame | None = None,
) -> list[str]:
    """Return ``genre`` plus its nearest genres by centroid distance."""
    centroids = centroids if centroids is not None else genre_centroids(catalog)
    if genre not in centroids.index:
        raise ValueError(f"Unknown genre {genre!r}.")
    target = centroids.loc[genre].to_numpy(dtype=float)
    distances = {
        other: float(np.linalg.norm(centroids.loc[other].to_numpy(dtype=float) - target))
        for other in centroids.index
        if other != genre
    }
    nearest = sorted(distances, key=distances.get)[: max(0, count - 1)]
    return [genre, *nearest]


# --------------------------------------------------------------------------- #
# Pool sampling
# --------------------------------------------------------------------------- #
def sample_pool(
    catalog: pd.DataFrame,
    genres: list[str] | None,
    length: int,
    seed: int,
) -> pd.DataFrame:
    """Filter to the chosen genres (or all), clean, and sample ``length`` tracks."""
    df = catalog.copy()
    if genres:
        wanted = {g.casefold() for g in genres}
        df = df[df["track_genre"].astype(str).str.casefold().isin(wanted)]
    df["tempo"] = pd.to_numeric(df["tempo"], errors="coerce")
    df = df[df["tempo"].notna() & np.isfinite(df["tempo"]) & (df["tempo"] > 0)]
    df = df.drop_duplicates(subset="track_id")
    name = df["track_name"].astype(str).str.strip().str.casefold()
    artist = df["artists"].astype(str).str.strip().str.casefold()
    df = df.loc[~pd.DataFrame({"n": name, "a": artist}).duplicated()]
    if len(df) < length:
        raise ValueError(
            f"Only {len(df)} tracks available for genres={genres}; need {length}."
        )
    return df.sample(n=length, random_state=seed, replace=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Pipeline preparation
# --------------------------------------------------------------------------- #
@dataclass
class PlaylistInputs:
    track_pool: pd.DataFrame
    c_trans: np.ndarray
    c_arc: np.ndarray
    target_arc: np.ndarray
    start_index: int
    end_index: int
    bottleneck_results: dict
    graph_data: object
    ev_scores: np.ndarray
    relative_positions: np.ndarray
    arc_shape: str
    genres: list[str] | None


def prepare_pipeline_timed(
    track_pool: pd.DataFrame,
    seed: int,
    arc_shape: str = "linear",
    genres: list[str] | None = None,
    endpoint_mode: str = "quantile",
) -> tuple["PlaylistInputs", float]:
    """Like ``prepare_pipeline`` but also returns wall-clock precompute time in seconds."""
    t0 = time.perf_counter()
    inp = prepare_pipeline(track_pool, seed, arc_shape, genres, endpoint_mode)
    return inp, time.perf_counter() - t0


def prepare_pipeline(
    track_pool: pd.DataFrame,
    seed: int,
    arc_shape: str = "linear",
    genres: list[str] | None = None,
    endpoint_mode: str = "quantile",
) -> PlaylistInputs:
    """Compute EV, costs, endpoints, arc, bottlenecks, and graph for a pool."""
    track_pool = compute_ev_score(track_pool, EnergyValenceConfig())
    c_trans = compute_transition_cost_matrix(track_pool)
    start_index, end_index = select_start_end_tracks(
        track_pool, c_trans, StartEndSelectionConfig(random_seed=seed, endpoint_mode=endpoint_mode)
    )
    target_arc = create_target_arc(
        track_pool, start_index, end_index, TargetArcConfig(shape=arc_shape)
    )
    c_arc = compute_arc_cost_matrix(track_pool, target_arc)
    bottleneck_results = BottleneckDetector(BottleneckConfig()).detect(c_arc)
    graph_data = TransitionGraphBuilder(
        TransitionGraphConfig()
    ).build_transition_graph_data(c_trans)
    return PlaylistInputs(
        track_pool=track_pool,
        c_trans=c_trans,
        c_arc=c_arc,
        target_arc=target_arc,
        start_index=start_index,
        end_index=end_index,
        bottleneck_results=bottleneck_results,
        graph_data=graph_data,
        ev_scores=track_pool["EV_score"].to_numpy(dtype=float),
        relative_positions=relative_positions(track_pool, start_index, end_index),
        arc_shape=arc_shape,
        genres=genres,
    )


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #
# category: "baseline" or "ours"; arc_optimized=False means ignore arc_rmse.
METHOD_INFO: dict[str, dict] = {
    "random": {"category": "baseline", "arc_optimized": False,
               "desc": "Uniformly random interior order (floor)."},
    "transition_greedy": {"category": "baseline", "arc_optimized": False,
               "desc": "Greedy on transition cost only (ignores the arc)."},
    "arc_assignment": {"category": "baseline", "arc_optimized": True,
               "desc": "Hungarian assignment minimizing total arc cost (arc lower bound)."},
    "flexer_interp": {"category": "baseline", "arc_optimized": True,
               "desc": "Flexer start/end acoustic interpolation."},
    "forward_beam": {"category": "baseline", "arc_optimized": True,
               "desc": "One-direction beam (arc + transition)."},
    "mm_beam": {"category": "baseline", "arc_optimized": True,
               "desc": "Bidirectional MM beam, commits everything (no recursion)."},
    "greedy_arc_trans": {"category": "baseline", "arc_optimized": True,
               "desc": "Greedy on arc + transition (main baseline)."},
    "bibs_current": {"category": "ours", "arc_optimized": True,
               "desc": "Recursive bottleneck-guided bidirectional beam search."},
    "bibs_repair": {"category": "ours", "arc_optimized": True,
               "desc": "BIBS + guarded micro-repair pass."},
    "bibs_stochastic": {"category": "ours", "arc_optimized": True,
               "desc": "BIBS with random-walk / Top-K sampled beams (seeded)."},
    "bibs_stoch_repair": {"category": "ours", "arc_optimized": True,
               "desc": "Stochastic BIBS + repair."},
    "bibs_parallel": {"category": "ours", "arc_optimized": True,
               "desc": "Level-synchronous parallel anchor placement."},
    # --- stochastic MM beam variants (ours: random-walk beam + repair) ---
    "mm_beam_stochastic": {"category": "ours", "arc_optimized": True,
               "desc": "MM beam with random-walk / Top-K sampled beam pruning (stochastic)."},
    "mm_beam_stochastic_repair": {"category": "ours", "arc_optimized": True,
               "desc": "Stochastic MM beam + guarded micro-repair."},
    # --- orchestrated stochastic MM beam (ours: adds BIBS-style candidate filtering) ---
    "mm_beam_orch_stochastic": {"category": "ours", "arc_optimized": True,
               "desc": "Stochastic MM beam with orchestrated candidate filtering (arc+transition+bottleneck)."},
    "mm_beam_orch_stochastic_repair": {"category": "ours", "arc_optimized": True,
               "desc": "Orchestrated stochastic MM beam + guarded micro-repair."},
    # --- tempo-aware / alpha-boost variants ---
    "bibs_tempo_aware": {"category": "ours", "arc_optimized": True,
               "desc": "BIBS with BPM-jump penalty (threshold 0.30, excuse for good arc fit)."},
    "bibs_alpha2": {"category": "ours", "arc_optimized": True,
               "desc": "BIBS with rhythm weight α=2.0 in C_trans (tempo-heavy)."},
    "mm_beam_alpha2": {"category": "baseline", "arc_optimized": True,
               "desc": "MM beam with rhythm weight α=2.0 in C_trans (tempo-heavy)."},
    # --- trajectory-first endpoint selection ---
    "bibs_traj_first": {"category": "ours", "arc_optimized": True,
               "desc": "BIBS with trajectory-first endpoint selection."},
}


def _bibs(config: BIBSConfig, inp: PlaylistInputs) -> list[int]:
    return BIBS(config).generate(
        inp.c_arc, inp.c_trans, inp.bottleneck_results,
        graph_data=inp.graph_data,
        candidate_orchestrator=CandidateOrchestrator(CandidateOrchestratorConfig()),
        target_arc=inp.target_arc, track_pool=inp.track_pool,
        start_index=inp.start_index, end_index=inp.end_index,
    )


def run_all_methods(
    inp: PlaylistInputs,
    seed: int = 42,
    include_new_variants: bool = True,
) -> tuple[dict[str, list[int]], dict[str, float]]:
    """Run every method on one prepared pool.

    Returns:
        playlists: name -> ordered track-index list
        timings: name -> wall-clock seconds for that method's generate() call
    """
    N = len(inp.track_pool)
    s, e = inp.start_index, inp.end_index
    playlists: dict[str, list[int]] = {}
    timings: dict[str, float] = {}

    def _run(name: str, fn):
        t0 = time.perf_counter()
        result = fn()
        timings[name] = time.perf_counter() - t0
        playlists[name] = result

    _run("random", lambda: RandomBaseline(seed=seed).generate(N, s, e))
    _run("transition_greedy", lambda: TransitionGreedyBaseline().generate(
        inp.c_trans, s, e, inp.graph_data
    ))
    _run("arc_assignment", lambda: ArcAssignmentBaseline().generate(inp.c_arc, s, e))
    _run("flexer_interp", lambda: FlexerInterpolationBaseline().generate(inp.track_pool, s, e))
    _run("forward_beam", lambda: ForwardBeamBaseline().generate(inp.c_arc, inp.c_trans, s, e))
    _run("mm_beam", lambda: MMBeamBaseline().generate(inp.c_arc, inp.c_trans, s, e))
    _run("greedy_arc_trans", lambda: GreedyPlaylistBaseline(GreedyBaselineConfig()).generate(
        inp.c_arc, inp.c_trans, start_index=s, end_index=e, transition_graph=inp.graph_data
    ))

    bibs_current = _bibs(BIBSConfig(), inp)
    _run("bibs_current", lambda: bibs_current)
    # Re-time repair on top of already-computed bibs_current.
    _run("bibs_repair", lambda: repair_playlist(
        bibs_current, inp.c_arc, inp.c_trans, RepairConfig()
    )[0])
    stoch = _bibs(BIBSConfig(stochastic=True, random_seed=seed), inp)
    _run("bibs_stochastic", lambda: stoch)
    _run("bibs_stoch_repair", lambda: repair_playlist(
        stoch, inp.c_arc, inp.c_trans, RepairConfig()
    )[0])
    _run("bibs_parallel", lambda: ParallelLevelSolver(
        ParallelSolverConfig(random_seed=seed)
    ).generate(inp.c_arc, inp.c_trans, inp.bottleneck_results, s, e))

    if include_new_variants:
        # Stochastic MM beam (plain)
        mm_stoch_cfg = MMBeamConfig(stochastic=True, random_seed=seed)
        mm_stoch = MMBeamBaseline(config=mm_stoch_cfg).generate(inp.c_arc, inp.c_trans, s, e)
        _run("mm_beam_stochastic", lambda: mm_stoch)
        _run("mm_beam_stochastic_repair", lambda: repair_playlist(
            mm_stoch, inp.c_arc, inp.c_trans, RepairConfig()
        )[0])

        # Orchestrated stochastic MM beam (adds BIBS-style candidate filtering)
        from src.candidate_orchestrator import CandidateOrchestrator, CandidateOrchestratorConfig
        _orch = CandidateOrchestrator(CandidateOrchestratorConfig())
        mm_orch_cfg = MMBeamConfig(stochastic=True, random_seed=seed, use_orchestrator=True)
        mm_orch = MMBeamBaseline(config=mm_orch_cfg).generate(
            inp.c_arc, inp.c_trans, s, e,
            graph_data=inp.graph_data,
            bottleneck_results=inp.bottleneck_results,
            candidate_orchestrator=_orch,
        )
        _run("mm_beam_orch_stochastic", lambda: mm_orch)
        _run("mm_beam_orch_stochastic_repair", lambda: repair_playlist(
            mm_orch, inp.c_arc, inp.c_trans, RepairConfig()
        )[0])

        # Tempo-aware BIBS (BPM-jump penalty, threshold 0.30)
        _run("bibs_tempo_aware", lambda: _bibs(
            BIBSConfig(tempo_jump_threshold=0.30, tempo_jump_penalty=3.0), inp
        ))

        # Alpha-boost (α=2.0): build a second C_trans with heavier rhythm weight
        from src.cost_functions import compute_transition_cost_matrix as _ctm
        c_trans_alpha2 = _ctm(inp.track_pool, alpha=2.0, beta=0.4, gamma=0.6)
        inp_alpha2 = PlaylistInputs(
            track_pool=inp.track_pool,
            c_trans=c_trans_alpha2,
            c_arc=inp.c_arc,
            target_arc=inp.target_arc,
            start_index=inp.start_index,
            end_index=inp.end_index,
            bottleneck_results=inp.bottleneck_results,
            graph_data=inp.graph_data,
            ev_scores=inp.ev_scores,
            relative_positions=inp.relative_positions,
            arc_shape=inp.arc_shape,
            genres=inp.genres,
        )
        _run("bibs_alpha2", lambda: _bibs(BIBSConfig(), inp_alpha2))
        _run("mm_beam_alpha2", lambda: MMBeamBaseline().generate(
            inp.c_arc, c_trans_alpha2, s, e
        ))

        # Trajectory-first BIBS: re-run endpoint selection with trajectory_fit mode
        try:
            traj_pool = inp.track_pool.copy()
            from src.feature_engineering import StartEndSelectionConfig, select_start_end_tracks
            traj_cfg = StartEndSelectionConfig(endpoint_mode="trajectory_fit")
            s_tf, e_tf = select_start_end_tracks(traj_pool, inp.c_trans, traj_cfg)
            traj_arc = create_target_arc(traj_pool, s_tf, e_tf, TargetArcConfig(shape=inp.arc_shape))
            c_arc_tf = compute_arc_cost_matrix(traj_pool, traj_arc)
            inp_tf = PlaylistInputs(
                track_pool=traj_pool,
                c_trans=inp.c_trans,
                c_arc=c_arc_tf,
                target_arc=traj_arc,
                start_index=s_tf,
                end_index=e_tf,
                bottleneck_results=inp.bottleneck_results,
                graph_data=inp.graph_data,
                ev_scores=inp.ev_scores,
                relative_positions=inp.relative_positions,
                arc_shape=inp.arc_shape,
                genres=inp.genres,
            )
            _run("bibs_traj_first", lambda: _bibs(BIBSConfig(), inp_tf))
        except Exception:
            pass  # trajectory_fit can fail on narrow pools; silently skip

    return playlists, timings


def evaluate_methods(
    playlists: dict[str, list[int]],
    inp: PlaylistInputs,
    timings: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Return a metrics table (original + additive) for each method's playlist."""
    evaluator = PlaylistEvaluator(EvaluationConfig())
    rows: list[dict[str, object]] = []
    for name, playlist in playlists.items():
        metrics = evaluator.evaluate(playlist, inp.c_arc, inp.c_trans)
        pct = transition_percentiles(playlist, inp.c_trans)
        rows.append(
            {
                "method": name,
                "category": METHOD_INFO.get(name, {}).get("category", ""),
                "arc_optimized": METHOD_INFO.get(name, {}).get("arc_optimized", True),
                "arc_rmse": metrics["arc_rmse"],
                "total_transition_cost": metrics["total_transition_cost"],
                "average_transition_cost": metrics["average_transition_cost"],
                "global_coherence": metrics["global_coherence"],
                "p90_transition": pct["p90_transition"],
                "p95_transition": pct["p95_transition"],
                "max_transition": pct["p100_transition"],
                "dtw_shape": dtw_shape_distance(playlist, inp.ev_scores, inp.target_arc),
                "ratio_adherence_rmse": ratio_adherence_rmse(
                    playlist, inp.relative_positions
                ),
                "wall_time_s": round(timings[name], 4) if timings and name in timings else None,
            }
        )
    return pd.DataFrame(rows)


def bibs_variability(
    inp: PlaylistInputs,
    runs: int = 8,
    base_seed: int = 1,
    bottleneck_mode: str = "score",
) -> tuple[list[list[int]], pd.DataFrame, float]:
    """Run stochastic BIBS multiple times; return playlists, metrics, diversity."""
    evaluator = PlaylistEvaluator(EvaluationConfig())
    playlists = [
        _bibs(
            BIBSConfig(stochastic=True, random_seed=base_seed + i,
                       bottleneck_mode=bottleneck_mode),
            inp,
        )
        for i in range(runs)
    ]
    rows = []
    for i, playlist in enumerate(playlists):
        m = evaluator.evaluate(playlist, inp.c_arc, inp.c_trans)
        rows.append(
            {
                "run": i + 1,
                "seed": base_seed + i,
                "arc_rmse": m["arc_rmse"],
                "total_transition_cost": m["total_transition_cost"],
                "global_coherence": m["global_coherence"],
            }
        )
    return playlists, pd.DataFrame(rows), playlist_diversity(playlists)
