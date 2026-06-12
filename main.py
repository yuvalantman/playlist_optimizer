"""Run the first playlist-optimizer data and cost-matrix pipeline."""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.bibs import BIBS, BIBSConfig
from src.bottleneck_detector import BottleneckConfig, BottleneckDetector
from src.cost_functions import compute_transition_cost_matrix, transition_cost
from src.data_preparation import TrackPoolBuilder, TrackPoolConfig
from src.evaluator import EvaluationConfig, PlaylistEvaluator
from src.experiment_runner import ExperimentConfig, ExperimentRunner
from src.feature_engineering import (
    EnergyValenceConfig,
    compute_arc_cost_matrix,
    compute_ev_score,
    create_linear_target_arc,
)
from src.greedy_baseline import GreedyBaselineConfig, GreedyPlaylistBaseline
from src.output_writer import OutputWriter
from src.transition_graph import TransitionGraphBuilder, TransitionGraphConfig

RUN_EXPERIMENTS = True


def build_transition_details(
    transitions: list[dict],
    track_pool: pd.DataFrame,
) -> pd.DataFrame:
    """Add track metadata to transition diagnostic records."""
    records: list[dict] = []
    for transition in transitions:
        from_track = track_pool.loc[transition["from_track_index"]]
        to_track = track_pool.loc[transition["to_track_index"]]
        records.append(
            {
                "position": transition["position"],
                "from_track_index": transition["from_track_index"],
                "from_track_name": from_track["track_name"],
                "from_artists": from_track["artists"],
                "from_EV_score": from_track["EV_score"],
                "from_tempo": from_track["tempo"],
                "from_camelot": from_track["camelot"],
                "to_track_index": transition["to_track_index"],
                "to_track_name": to_track["track_name"],
                "to_artists": to_track["artists"],
                "to_EV_score": to_track["EV_score"],
                "to_tempo": to_track["tempo"],
                "to_camelot": to_track["camelot"],
                "transition_cost": transition["transition_cost"],
            }
        )
    return pd.DataFrame(records)


def build_transition_summary(
    label: str,
    transition_costs: list[float],
) -> dict[str, float | str]:
    """Build distribution diagnostics for transition costs."""
    costs = np.asarray(transition_costs, dtype=float)
    return {
        "Playlist": label,
        "min": float(np.min(costs)),
        "mean": float(np.mean(costs)),
        "median": float(np.median(costs)),
        "max": float(np.max(costs)),
        "90th_percentile": float(np.percentile(costs, 90)),
        "95th_percentile": float(np.percentile(costs, 95)),
    }


def main() -> None:
    """Build an example pool and print representative pipeline outputs."""
    project_root = Path(__file__).resolve().parent
    pool_config = TrackPoolConfig(
        csv_path=project_root / "data" / "spotify_tracks.csv",
        genre="pop",
        pool_size=300,
        random_seed=42,
        min_tempo=40.0,
        max_tempo=220.0,
    )

    track_pool = TrackPoolBuilder(pool_config).build()
    duplicate_track_ids = track_pool["track_id"].duplicated().sum()
    normalized_song_pairs = track_pool[["track_name", "artists"]].astype(str)
    normalized_song_pairs = normalized_song_pairs.apply(
        lambda column: column.str.strip().str.casefold()
    )
    duplicate_songs = normalized_song_pairs.duplicated().sum()

    track_pool = compute_ev_score(track_pool, EnergyValenceConfig())
    target_arc = create_linear_target_arc(
        length=len(track_pool),
        start_value=0.2,
        end_value=0.8,
    )
    c_arc = compute_arc_cost_matrix(track_pool, target_arc)
    c_trans = compute_transition_cost_matrix(track_pool, alpha=1.0, beta=1.0)
    transition_graph_builder = TransitionGraphBuilder(TransitionGraphConfig())
    transition_graph = transition_graph_builder.build_top_m_graph(c_trans)
    transition_graph_with_costs = (
        transition_graph_builder.build_top_m_graph_with_costs(c_trans)
    )
    bottleneck_results = BottleneckDetector(BottleneckConfig()).detect(c_arc)
    greedy_playlist = GreedyPlaylistBaseline(GreedyBaselineConfig()).generate(
        c_arc,
        c_trans,
        transition_graph=transition_graph,
    )
    evaluator = PlaylistEvaluator(EvaluationConfig())
    greedy_evaluation = evaluator.evaluate(
        greedy_playlist,
        c_arc,
        c_trans,
    )
    bibs_playlist = BIBS(BIBSConfig()).generate(
        c_arc,
        c_trans,
        bottleneck_results,
        transition_graph=transition_graph,
    )
    bibs_evaluation = evaluator.evaluate(
        bibs_playlist,
        c_arc,
        c_trans,
    )
    output_paths = OutputWriter(
        playlists_directory=project_root / "outputs" / "playlists",
        results_directory=project_root / "outputs" / "results",
    ).save_single_run_outputs(
        greedy_playlist,
        bibs_playlist,
        track_pool,
        greedy_evaluation,
        bibs_evaluation,
    )

    song_scores = bottleneck_results["song_bottleneck_scores"]
    location_scores = bottleneck_results["location_bottleneck_scores"]
    bottleneck_indices = bottleneck_results["bottleneck_track_indices"]
    candidate_sets = bottleneck_results["candidate_sets"]

    first = track_pool.iloc[0]
    second = track_pool.iloc[1]
    example_transition = transition_cost(
        first["tempo"],
        second["tempo"],
        int(first["camelot_number"]),
        int(second["camelot_number"]),
    )

    print(f"Track pool shape: {track_pool.shape}")
    print(f"Duplicated track_id values: {duplicate_track_ids}")
    print(f"Duplicated track_name + artists pairs: {duplicate_songs}")
    print(f"C_arc shape: {c_arc.shape}")
    print(f"C_trans shape: {c_trans.shape}")
    print(f"Target arc endpoints: {target_arc[0]:.3f}, {target_arc[-1]:.3f}")
    print("\nExample tracks:")
    print(
        track_pool[
            ["track_name", "artists", "tempo", "camelot", "EV_score"]
        ].head(3).to_string(index=False)
    )
    print(f"\nFirst track arc costs: {c_arc[0, :5]}")
    print(f"Transition cost from track 0 to track 1: {example_transition:.4f}")
    print(f"Matching matrix entry: {c_trans[0, 1]:.4f}")
    print(f"\nNumber of nodes in transition graph: {len(transition_graph)}")
    print(f"Number of neighbors for track 0: {len(transition_graph[0])}")

    track_zero_neighbors = transition_graph_with_costs[0][:5]
    neighbor_indices = [track_index for track_index, _ in track_zero_neighbors]
    neighbor_costs = [cost for _, cost in track_zero_neighbors]
    neighbor_details = track_pool.loc[
        neighbor_indices,
        ["track_name", "artists", "tempo", "camelot"],
    ].copy()
    neighbor_details.insert(0, "track_index", neighbor_indices)
    neighbor_details["transition_cost"] = neighbor_costs
    print("\nTop 5 transition neighbors of track 0:")
    print(neighbor_details.to_string(index=False))

    print(f"\nGreedy playlist length: {len(greedy_playlist)}")
    print(f"Number of unique tracks in greedy playlist: {len(set(greedy_playlist))}")
    print(f"First 10 greedy track indices: {greedy_playlist[:10]}")

    first_greedy_indices = greedy_playlist[:5]
    first_greedy_tracks = track_pool.loc[
        first_greedy_indices,
        ["track_name", "artists", "EV_score", "tempo", "camelot"],
    ].copy()
    first_greedy_tracks.insert(0, "track_index", first_greedy_indices)
    first_greedy_tracks.insert(0, "position", range(len(first_greedy_tracks)))
    print("\nFirst 5 tracks in greedy playlist:")
    print(first_greedy_tracks.to_string(index=False))
    print(f"\nGreedy arc RMSE: {greedy_evaluation['arc_rmse']:.6f}")
    print(
        "Greedy total transition cost: "
        f"{greedy_evaluation['total_transition_cost']:.6f}"
    )
    print(
        "Greedy average transition cost: "
        f"{greedy_evaluation['average_transition_cost']:.6f}"
    )
    print(
        "Greedy tail transition cost: "
        f"{greedy_evaluation['tail_transition_cost']:.6f}"
    )
    print(
        "Greedy average tail transition cost: "
        f"{greedy_evaluation['average_tail_transition_cost']:.6f}"
    )

    print(f"\nBIBS playlist length: {len(bibs_playlist)}")
    print(f"Number of unique tracks in BIBS playlist: {len(set(bibs_playlist))}")
    print(f"First 10 BIBS track indices: {bibs_playlist[:10]}")

    first_bibs_indices = bibs_playlist[:5]
    first_bibs_tracks = track_pool.loc[
        first_bibs_indices,
        ["track_name", "artists", "EV_score", "tempo", "camelot"],
    ].copy()
    first_bibs_tracks.insert(0, "track_index", first_bibs_indices)
    first_bibs_tracks.insert(0, "position", range(len(first_bibs_tracks)))
    print("\nFirst 5 tracks in BIBS playlist:")
    print(first_bibs_tracks.to_string(index=False))
    print(f"\nBIBS arc RMSE: {bibs_evaluation['arc_rmse']:.6f}")
    print(
        "BIBS total transition cost: "
        f"{bibs_evaluation['total_transition_cost']:.6f}"
    )
    print(
        "BIBS average transition cost: "
        f"{bibs_evaluation['average_transition_cost']:.6f}"
    )
    print(
        "BIBS tail transition cost: "
        f"{bibs_evaluation['tail_transition_cost']:.6f}"
    )
    print(
        "BIBS average tail transition cost: "
        f"{bibs_evaluation['average_tail_transition_cost']:.6f}"
    )

    comparison = pd.DataFrame(
        {
            "Metric": list(greedy_evaluation),
            "Greedy": list(greedy_evaluation.values()),
            "BIBS": [
                bibs_evaluation[metric]
                for metric in greedy_evaluation
            ],
        }
    )
    print("\nMetric comparison:")
    print(comparison.to_string(index=False))
    print("\nSingle-run outputs saved:")
    print(f"Greedy playlist: {output_paths['greedy_playlist']}")
    print(f"BIBS playlist: {output_paths['bibs_playlist']}")
    print(f"Metric comparison: {output_paths['metric_comparison']}")

    greedy_transition_costs = evaluator.get_transition_costs_by_position(
        greedy_playlist,
        c_trans,
    )
    bibs_transition_costs = evaluator.get_transition_costs_by_position(
        bibs_playlist,
        c_trans,
    )
    transition_summary = pd.DataFrame(
        [
            build_transition_summary("Greedy", greedy_transition_costs),
            build_transition_summary("BIBS", bibs_transition_costs),
        ]
    )
    print("\nTransition cost summary:")
    print(transition_summary.to_string(index=False))

    worst_greedy = evaluator.get_worst_transitions(
        greedy_playlist,
        c_trans,
        top_k=10,
    )
    print("\nTop 10 worst Greedy transitions:")
    print(build_transition_details(worst_greedy, track_pool).to_string(index=False))

    worst_bibs = evaluator.get_worst_transitions(
        bibs_playlist,
        c_trans,
        top_k=10,
    )
    print("\nTop 10 worst BIBS transitions:")
    print(build_transition_details(worst_bibs, track_pool).to_string(index=False))

    bibs_tail_size = math.ceil(
        evaluator.config.tail_fraction * len(bibs_transition_costs)
    )
    bibs_tail_start = len(bibs_transition_costs) - bibs_tail_size
    bibs_tail_transitions = [
        transition
        for transition in evaluator.get_worst_transitions(
            bibs_playlist,
            c_trans,
            top_k=len(bibs_transition_costs),
        )
        if transition["position"] >= bibs_tail_start
    ][:10]
    print("\nTop 10 worst BIBS transitions in final 20 percent:")
    print(
        build_transition_details(
            bibs_tail_transitions,
            track_pool,
        ).to_string(index=False)
    )

    print(f"\nSong bottleneck scores shape: {song_scores.shape}")
    print(f"Location bottleneck scores shape: {location_scores.shape}")
    print(f"Number of bottleneck tracks: {len(bottleneck_indices)}")
    print(f"First 10 bottleneck track indices: {bottleneck_indices[:10]}")
    print(f"Candidate set size for position 0: {len(candidate_sets[0])}")

    hardest_tracks = track_pool[
        ["track_id", "track_name", "artists", "EV_score"]
    ].copy()
    hardest_tracks["bottleneck_score"] = song_scores
    hardest_tracks = hardest_tracks.sort_values(
        by="bottleneck_score",
        ascending=False,
        kind="stable",
    ).head(5)
    print("\nTop 5 hardest tracks by bottleneck score:")
    print(hardest_tracks.to_string(index=False))


def run_experiments() -> None:
    """Run the example multi-genre, multi-seed experiment grid."""
    project_root = Path(__file__).resolve().parent
    config = ExperimentConfig(
        csv_path=str(project_root / "data" / "spotify_tracks.csv"),
        genres=["pop", "rock", "dance", "edm", "indie"],
        seeds=[1, 2, 3, 4, 5],
        pool_size=300,
        output_results_path=str(
            project_root / "outputs" / "results" / "experiment_results.csv"
        ),
        output_summary_path=str(
            project_root / "outputs" / "results" / "experiment_summary.csv"
        ),
    )
    raw_results, summary = ExperimentRunner(config).run_all()

    print("\nRaw experiment results:")
    print(raw_results.head().to_string(index=False))
    print("\nExperiment summary:")
    print(summary.to_string(index=False))
    print(f"\nRaw results saved to: {config.output_results_path}")
    print(f"Summary saved to: {config.output_summary_path}")


if __name__ == "__main__":
    if RUN_EXPERIMENTS:
        run_experiments()
    else:
        main()
