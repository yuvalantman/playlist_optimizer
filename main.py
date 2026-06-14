"""Run the playlist optimizer demo or experiment grid."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.bibs import BIBS, BIBSConfig
from src.bibs_config_sweep import run_bibs_config_sweep
from src.bottleneck_detector import BottleneckConfig, BottleneckDetector
from src.candidate_orchestrator import (
    CandidateOrchestrator,
    CandidateOrchestratorConfig,
)
from src.cost_functions import compute_transition_cost_matrix
from src.data_preparation import TrackPoolBuilder, TrackPoolConfig
from src.evaluator import EvaluationConfig, PlaylistEvaluator
from src.experiment_runner import ExperimentConfig, ExperimentRunner
from src.feature_engineering import (
    EnergyValenceConfig,
    StartEndSelectionConfig,
    compute_arc_cost_matrix,
    compute_ev_score,
    create_target_tempo_arc_from_tracks,
    create_target_arc_from_tracks,
    create_tempo_envelope_from_tracks,
    select_start_end_tracks,
)
from src.greedy_baseline import GreedyBaselineConfig, GreedyPlaylistBaseline
from src.output_writer import OutputWriter
from src.transition_graph import TransitionGraphBuilder, TransitionGraphConfig

RUN_EXPERIMENTS = False
RUN_BIBS_SWEEP = False
POOL_SIZE = 150
EXPERIMENT_POOL_SIZE = 150
GENRE = "pop"
RANDOM_SEED = 42
SEPARATOR_WIDTH = 90


def print_section(title: str) -> None:
    """Print a major terminal-output section."""
    print("\n" + "=" * SEPARATOR_WIDTH)
    print(title)
    print("=" * SEPARATOR_WIDTH)


def print_subsection(title: str) -> None:
    """Print a minor terminal-output section."""
    print("\n" + "-" * SEPARATOR_WIDTH)
    print(title)
    print("-" * SEPARATOR_WIDTH)


def print_table(frame: pd.DataFrame) -> None:
    """Print a DataFrame without its index and with compact numeric values."""
    print(
        frame.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )


def print_metric_block(metrics: dict[str, float]) -> None:
    """Print one metric per line with consistent precision."""
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")


def print_playlist_preview(
    playlist: list[int],
    track_pool: pd.DataFrame,
    label: str,
    number_of_tracks: int = 5,
) -> None:
    """Print playlist validity information and its first few tracks."""
    print(f"{label} playlist length: {len(playlist)}")
    print(f"Number of unique tracks: {len(set(playlist))}")
    print(f"First 10 indices: {playlist[:10]}")

    indices = playlist[:number_of_tracks]
    preview = track_pool.loc[
        indices,
        ["track_name", "artists", "EV_score", "tempo", "camelot"],
    ].copy()
    preview.insert(0, "track_index", indices)
    preview.insert(0, "position", range(len(preview)))
    print_subsection(f"First {number_of_tracks} Tracks")
    print_table(preview)


def build_metric_comparison(
    greedy_metrics: dict[str, float],
    bibs_metrics: dict[str, float],
    anchor_alignment_cost: float | None,
) -> pd.DataFrame:
    """Build a direction-aware official metric comparison table."""
    metric_directions = {
        "arc_rmse": "lower",
        "total_transition_cost": "lower",
        "average_transition_cost": "lower",
        "global_coherence": "higher",
    }
    rows: list[dict[str, object]] = []
    for metric, better_when in metric_directions.items():
        if metric not in greedy_metrics or metric not in bibs_metrics:
            continue
        greedy_value = greedy_metrics[metric]
        bibs_value = bibs_metrics[metric]
        improvement = (
            greedy_value - bibs_value
            if better_when == "lower"
            else bibs_value - greedy_value
        )
        rows.append(
            {
                "metric": metric,
                "greedy": greedy_value,
                "bibs": bibs_value,
                "improvement": improvement,
                "better_when": better_when,
            }
        )
    if anchor_alignment_cost is not None:
        rows.append(
            {
                "metric": "anchor_alignment_cost",
                "greedy": np.nan,
                "bibs": anchor_alignment_cost,
                "improvement": np.nan,
                "better_when": "lower",
            }
        )
    return pd.DataFrame(rows)


def print_endpoint(
    role: str,
    track_index: int,
    track_pool: pd.DataFrame,
) -> None:
    """Print one selected endpoint and its identifying metadata."""
    columns = [
        "track_id",
        "track_name",
        "artists",
        "EV_score",
        "tempo",
        "key",
        "mode",
        "camelot",
    ]
    endpoint = track_pool.loc[[track_index], columns].copy()
    endpoint.insert(0, "index", track_index)
    print_subsection(f"Selected {role} Track")
    print_table(endpoint)


def main() -> None:
    """Run one playlist sequencing demonstration."""
    project_root = Path(__file__).resolve().parent
    pool_config = TrackPoolConfig(
        csv_path=project_root / "data" / "spotify_tracks.csv",
        genre=GENRE,
        pool_size=POOL_SIZE,
        random_seed=RANDOM_SEED,
    )

    track_pool = TrackPoolBuilder(pool_config).build()
    duplicate_track_ids = int(track_pool["track_id"].duplicated().sum())
    normalized_song_pairs = track_pool[["track_name", "artists"]].astype(str)
    normalized_song_pairs = normalized_song_pairs.apply(
        lambda column: column.str.strip().str.casefold()
    )
    duplicate_songs = int(normalized_song_pairs.duplicated().sum())

    track_pool = compute_ev_score(track_pool, EnergyValenceConfig())
    c_trans = compute_transition_cost_matrix(track_pool)
    start_index, end_index = select_start_end_tracks(
        track_pool,
        c_trans,
        StartEndSelectionConfig(random_seed=pool_config.random_seed),
    )
    tempo_min, tempo_max = create_tempo_envelope_from_tracks(
        track_pool,
        start_index,
        end_index,
    )
    target_tempo_arc = create_target_tempo_arc_from_tracks(
        track_pool,
        start_index,
        end_index,
    )
    target_arc = create_target_arc_from_tracks(track_pool, start_index, end_index)
    c_arc = compute_arc_cost_matrix(track_pool, target_arc)

    ev_start = float(track_pool.loc[start_index, "EV_score"])
    ev_end = float(track_pool.loc[end_index, "EV_score"])
    if not np.isclose(target_arc[0], ev_start) or not np.isclose(
        target_arc[-1],
        ev_end,
    ):
        raise ValueError("Target arc endpoints must match selected track EV scores.")
    if c_trans.shape != (len(track_pool), len(track_pool)):
        raise ValueError("C_trans must be square with one row per track.")

    bottleneck_results = BottleneckDetector(BottleneckConfig()).detect(c_arc)
    transition_graph_builder = TransitionGraphBuilder(TransitionGraphConfig())
    graph_data = transition_graph_builder.build_transition_graph_data(c_trans)

    available_for_demo = set(range(len(track_pool))) - {start_index, end_index}
    orchestrator = CandidateOrchestrator(CandidateOrchestratorConfig())
    forward_candidates = orchestrator.build_forward_candidates(
        start_index,
        1,
        available_for_demo,
        graph_data,
        bottleneck_results,
        c_arc,
    )
    backward_candidates = orchestrator.build_backward_candidates(
        end_index,
        len(track_pool) - 2,
        available_for_demo,
        graph_data,
        bottleneck_results,
        c_arc,
    )
    anchor_candidates = orchestrator.build_anchor_candidates(
        start_index,
        end_index,
        len(track_pool) // 2,
        available_for_demo,
        graph_data,
        bottleneck_results,
        c_arc,
        c_trans,
    )
    anchor_candidate_sources = orchestrator.get_last_anchor_candidate_sources()
    orchestrator_diagnostics = orchestrator.get_diagnostics()
    evaluator = PlaylistEvaluator(EvaluationConfig())

    if RUN_BIBS_SWEEP:
        sweep_results = run_bibs_config_sweep(
            c_arc,
            c_trans,
            bottleneck_results,
            graph_data,
            orchestrator,
            target_arc,
            track_pool,
            start_index,
            end_index,
            evaluator,
        )
        sweep_path = (
            project_root / "outputs" / "results" / "bibs_config_sweep.csv"
        )
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        sweep_results.to_csv(sweep_path, index=False)

        print_section("BIBS CONFIG SWEEP: RUN CONFIGURATION")
        print(f"RUN_EXPERIMENTS: {RUN_EXPERIMENTS}")
        print(f"RUN_BIBS_SWEEP: {RUN_BIBS_SWEEP}")
        print(f"Genre: {pool_config.genre}")
        print(f"Pool size: {pool_config.pool_size}")
        print(f"Random seed: {pool_config.random_seed}")

        print_section("BIBS CONFIG SWEEP: RANKED RESULTS")
        print_table(sweep_results)

        print_section("BIBS CONFIG SWEEP: TOP 3 CONFIGS")
        print_table(sweep_results.head(3))

        print_section("BIBS CONFIG SWEEP: OUTPUT FILE")
        print(f"Sweep results CSV: {sweep_path.resolve()}")
        return

    greedy_playlist = GreedyPlaylistBaseline(GreedyBaselineConfig()).generate(
        c_arc,
        c_trans,
        start_index=start_index,
        end_index=end_index,
        transition_graph=graph_data,
    )
    bibs = BIBS(BIBSConfig())
    bibs_playlist = bibs.generate(
        c_arc,
        c_trans,
        bottleneck_results,
        graph_data=graph_data,
        candidate_orchestrator=orchestrator,
        target_arc=target_arc,
        start_index=start_index,
        end_index=end_index,
        track_pool=track_pool,
    )
    bibs_internal_diagnostics = bibs.get_diagnostics()

    greedy_metrics = evaluator.evaluate(greedy_playlist, c_arc, c_trans)
    bibs_metrics = evaluator.evaluate(bibs_playlist, c_arc, c_trans)
    anchor_alignment_cost = None
    if hasattr(evaluator, "compute_anchor_alignment_cost"):
        anchor_alignment_cost = evaluator.compute_anchor_alignment_cost(
            bibs.anchor_history
        )

    output_paths = OutputWriter(
        playlists_directory=project_root / "outputs" / "playlists",
        results_directory=project_root / "outputs" / "results",
    ).save_single_run_outputs(
        greedy_playlist,
        bibs_playlist,
        track_pool,
        greedy_metrics,
        bibs_metrics,
        run_name=f"single_run_{POOL_SIZE}",
        anchor_alignment_cost=anchor_alignment_cost,
    )

    song_scores = bottleneck_results["song_bottleneck_scores"]
    location_scores = bottleneck_results["location_bottleneck_scores"]
    bottleneck_indices = bottleneck_results["bottleneck_track_indices"]
    hardest_tracks = track_pool[
        ["track_id", "track_name", "artists", "EV_score"]
    ].copy()
    hardest_tracks["bottleneck_score"] = song_scores
    hardest_tracks = hardest_tracks.sort_values(
        by="bottleneck_score",
        ascending=False,
        kind="stable",
    ).head(5)

    outgoing_track_zero = graph_data.outgoing_neighbors_with_costs[0][:5]
    outgoing_indices = [track_index for track_index, _ in outgoing_track_zero]
    outgoing_details = track_pool.loc[
        outgoing_indices,
        ["track_name", "artists", "tempo", "camelot"],
    ].copy()
    outgoing_details.insert(0, "track_index", outgoing_indices)
    outgoing_details["transition_cost"] = [
        cost for _, cost in outgoing_track_zero
    ]
    incoming_track_zero = graph_data.incoming_neighbors_with_costs[0][:5]
    incoming_indices = [track_index for track_index, _ in incoming_track_zero]
    incoming_details = track_pool.loc[
        incoming_indices,
        ["track_name", "artists", "tempo", "camelot"],
    ].copy()
    incoming_details.insert(0, "track_index", incoming_indices)
    incoming_details["transition_cost"] = [
        cost for _, cost in incoming_track_zero
    ]
    anchor_preview_indices = anchor_candidates[:10]
    anchor_details = track_pool.loc[
        anchor_preview_indices,
        ["track_name", "artists", "EV_score", "tempo", "camelot"],
    ].copy()
    anchor_details.insert(0, "track_index", anchor_preview_indices)
    anchor_details["C_trans_left_to_candidate"] = [
        c_trans[start_index, candidate] for candidate in anchor_preview_indices
    ]
    anchor_details["C_trans_candidate_to_right"] = [
        c_trans[candidate, end_index] for candidate in anchor_preview_indices
    ]
    midpoint = len(track_pool) // 2
    anchor_details["C_arc_midpoint"] = [
        c_arc[candidate, midpoint] for candidate in anchor_preview_indices
    ]
    anchor_details["balance_gap"] = [
        abs(
            c_trans[start_index, candidate]
            - c_trans[candidate, end_index]
        )
        for candidate in anchor_preview_indices
    ]
    anchor_details["bridge_score"] = [
        (
            c_trans[start_index, candidate]
            + c_trans[candidate, end_index]
            + orchestrator.config.bridge_balance_weight
            * abs(
                c_trans[start_index, candidate]
                - c_trans[candidate, end_index]
            )
            + orchestrator.config.arc_preview_weight * c_arc[candidate, midpoint]
            - orchestrator.config.bottleneck_preview_weight * song_scores[candidate]
        )
        for candidate in anchor_preview_indices
    ]
    anchor_details["candidate_sources"] = [
        ", ".join(sorted(anchor_candidate_sources[candidate]))
        for candidate in anchor_preview_indices
    ]

    print_section("SECTION: RUN CONFIGURATION")
    print(f"RUN_EXPERIMENTS: {RUN_EXPERIMENTS}")
    print(f"RUN_BIBS_SWEEP: {RUN_BIBS_SWEEP}")
    print(f"POOL_SIZE: {POOL_SIZE}")
    print(f"GENRE: {GENRE}")
    print(f"RANDOM_SEED: {RANDOM_SEED}")

    print_section("SECTION: DATA PREPARATION")
    print(f"Track pool shape: {track_pool.shape}")
    print(f"Duplicated track_id count: {duplicate_track_ids}")
    print(f"Duplicated track_name + artists count: {duplicate_songs}")

    print_section("SECTION: START/END SELECTION")
    print_endpoint("Start", start_index, track_pool)
    print_endpoint("End", end_index, track_pool)
    print(f"\nEV_start: {ev_start:.6f}")
    print(f"EV_end: {ev_end:.6f}")
    print(f"EV_gap: {ev_end - ev_start:.6f}")
    print(f"Target arc endpoints: {target_arc[0]:.6f}, {target_arc[-1]:.6f}")

    print_section("SECTION: TEMPO ENVELOPE")
    print(f"Start tempo: {target_tempo_arc[0]:.6f}")
    print(f"End tempo: {target_tempo_arc[-1]:.6f}")
    print(f"Derived tempo_min: {tempo_min:.6f}")
    print(f"Derived tempo_max: {tempo_max:.6f}")
    print(
        "Target tempo arc endpoints: "
        f"{target_tempo_arc[0]:.6f}, {target_tempo_arc[-1]:.6f}"
    )

    print_section("SECTION: COST MATRICES")
    print(f"C_arc shape: {c_arc.shape}")
    print(f"C_trans shape: {c_trans.shape}")
    print(f"Example transition cost from track 0 to track 1: {c_trans[0, 1]:.6f}")
    print(f"Matching matrix entry C_trans[0, 1]: {c_trans[0, 1]:.6f}")

    print_section("SECTION: BOTTLENECK DETECTION")
    print(f"Song bottleneck scores shape: {song_scores.shape}")
    print(f"Location bottleneck scores shape: {location_scores.shape}")
    print(f"Number of bottleneck tracks: {len(bottleneck_indices)}")
    print(f"First 10 bottleneck track indices: {bottleneck_indices[:10]}")
    print_subsection("Top 5 Hardest Tracks by Bottleneck Score")
    print_table(hardest_tracks)

    print_section("SECTION: TRANSITION GRAPH")
    graph_diagnostics = graph_data.diagnostics
    print(f"Threshold used: {graph_diagnostics['threshold_used']:.6f}")
    print(f"Number of nodes: {graph_diagnostics['number_of_nodes']}")
    print(f"Number of edges: {graph_diagnostics['number_of_edges']}")
    print(f"Threshold edge count: {graph_diagnostics['threshold_edge_count']}")
    print(f"Fallback edge count: {graph_diagnostics['fallback_edge_count']}")
    print(f"Average out degree: {graph_diagnostics['average_out_degree']:.6f}")
    print(f"Average in degree: {graph_diagnostics['average_in_degree']:.6f}")
    print(
        "Min/max out degree: "
        f"{graph_diagnostics['min_out_degree']}/"
        f"{graph_diagnostics['max_out_degree']}"
    )
    print(
        "Min/max in degree: "
        f"{graph_diagnostics['min_in_degree']}/"
        f"{graph_diagnostics['max_in_degree']}"
    )
    print(f"Isolated out nodes: {graph_diagnostics['isolated_out_nodes']}")
    print(f"Isolated in nodes: {graph_diagnostics['isolated_in_nodes']}")
    print(f"Graph density: {graph_diagnostics['graph_density']:.6f}")
    print_subsection("Top 5 Outgoing Neighbors of Track 0")
    print_table(outgoing_details)
    print_subsection("Top 5 Incoming Neighbors of Track 0")
    print_table(incoming_details)

    print_section("SECTION: CANDIDATE ORCHESTRATION")
    print(f"Forward candidate count: {len(forward_candidates)}")
    print(f"Backward candidate count: {len(backward_candidates)}")
    print(f"Anchor candidate count: {len(anchor_candidates)}")
    print(f"First 10 forward candidates: {forward_candidates[:10]}")
    print(f"First 10 backward candidates: {backward_candidates[:10]}")
    print(f"First 10 anchor candidates: {anchor_candidates[:10]}")
    print_subsection("First 10 Anchor Candidate Details")
    print_table(anchor_details)
    print_subsection("Orchestrator Diagnostics")
    diagnostics_table = pd.DataFrame(
        [
            {"diagnostic": name, "value": value}
            for name, value in orchestrator_diagnostics.items()
        ]
    )
    print_table(diagnostics_table)

    print_section("SECTION: GREEDY BASELINE")
    print_playlist_preview(greedy_playlist, track_pool, "Greedy")
    print_subsection("Greedy Metrics")
    print_metric_block(greedy_metrics)

    print_section("SECTION: BIBS ALGORITHM")
    print_playlist_preview(bibs_playlist, track_pool, "BIBS")
    print_subsection("BIBS Metrics")
    print_metric_block(bibs_metrics)

    print_section("SECTION: BIBS INTERNAL DIAGNOSTICS")
    bibs_diagnostics_table = pd.DataFrame(
        [
            {"diagnostic": name, "value": value}
            for name, value in bibs_internal_diagnostics.items()
        ]
    )
    print_table(bibs_diagnostics_table)

    print_section("SECTION: OFFICIAL METRIC COMPARISON")
    print_table(
        build_metric_comparison(
            greedy_metrics,
            bibs_metrics,
            anchor_alignment_cost,
        )
    )

    print_section("SECTION: HARMONIC DIAGNOSTICS")
    harmonic_method = getattr(
        evaluator,
        "compute_harmonic_stagnation_metrics",
        None,
    )
    camelot_method = getattr(evaluator, "get_camelot_sequence", None)
    if harmonic_method is None or camelot_method is None:
        print("Harmonic diagnostics not available.")
    else:
        harmonic_table = pd.DataFrame(
            [
                {"playlist": "Greedy", **harmonic_method(greedy_playlist, track_pool)},
                {"playlist": "BIBS", **harmonic_method(bibs_playlist, track_pool)},
            ]
        )
        print_table(harmonic_table)
        print_subsection("First 30 Camelot Codes")
        print(f"Greedy: {camelot_method(greedy_playlist, track_pool)[:30]}")
        print(f"BIBS: {camelot_method(bibs_playlist, track_pool)[:30]}")

    print_section("SECTION: HARMONIC MOTION DIAGNOSTICS")
    harmonic_motion_method = getattr(
        evaluator,
        "compute_harmonic_motion_diagnostics",
        None,
    )
    if harmonic_motion_method is None:
        print("Harmonic motion diagnostics not available.")
    else:
        greedy_motion = harmonic_motion_method(greedy_playlist, track_pool)
        bibs_motion = harmonic_motion_method(bibs_playlist, track_pool)
        motion_summary = pd.DataFrame(
            [
                {
                    "playlist": "Greedy",
                    "max_camelot_jump": greedy_motion["max_camelot_jump"],
                    "number_of_large_camelot_jumps": greedy_motion[
                        "number_of_large_camelot_jumps"
                    ],
                    "average_camelot_jump": greedy_motion["average_camelot_jump"],
                },
                {
                    "playlist": "BIBS",
                    "max_camelot_jump": bibs_motion["max_camelot_jump"],
                    "number_of_large_camelot_jumps": bibs_motion[
                        "number_of_large_camelot_jumps"
                    ],
                    "average_camelot_jump": bibs_motion["average_camelot_jump"],
                },
            ]
        )
        print_table(motion_summary)
        print_subsection("First 30 Camelot Codes and Transition Distances")
        print(f"Greedy codes: {camelot_method(greedy_playlist, track_pool)[:30]}")
        print(
            "Greedy distances: "
            f"{greedy_motion['first_30_camelot_transition_distances']}"
        )
        print(f"BIBS codes: {camelot_method(bibs_playlist, track_pool)[:30]}")
        print(
            "BIBS distances: "
            f"{bibs_motion['first_30_camelot_transition_distances']}"
        )

    print_section("SECTION: ENERGY FLOW DIAGNOSTICS")
    energy_flow_method = getattr(evaluator, "compute_energy_flow_diagnostics", None)
    if energy_flow_method is None:
        print("Energy flow diagnostics not available.")
    else:
        energy_flow_table = pd.DataFrame(
            [
                {"playlist": "Greedy", **energy_flow_method(greedy_playlist, track_pool)},
                {"playlist": "BIBS", **energy_flow_method(bibs_playlist, track_pool)},
            ]
        )
        print_table(energy_flow_table)

    print_section("SECTION: TEMPO FLOW DIAGNOSTICS")
    tempo_flow_method = getattr(evaluator, "compute_tempo_flow_diagnostics", None)
    if tempo_flow_method is None:
        print("Tempo flow diagnostics not available.")
    else:
        tempo_flow_table = pd.DataFrame(
            [
                {
                    "playlist": "Greedy",
                    **tempo_flow_method(
                        greedy_playlist,
                        track_pool,
                        target_tempo_arc,
                    ),
                },
                {
                    "playlist": "BIBS",
                    **tempo_flow_method(
                        bibs_playlist,
                        track_pool,
                        target_tempo_arc,
                    ),
                },
            ]
        )
        print_table(tempo_flow_table)

    print_section("SECTION: OUTPUT FILES")
    print(f"Greedy playlist CSV: {output_paths['greedy_playlist']}")
    print(f"BIBS playlist CSV: {output_paths['bibs_playlist']}")
    print(f"Metric comparison CSV: {output_paths['metric_comparison']}")


def run_experiments() -> None:
    """Run the example multi-genre, multi-seed experiment grid."""
    project_root = Path(__file__).resolve().parent
    config = ExperimentConfig(
        csv_path=str(project_root / "data" / "spotify_tracks.csv"),
        genres=["pop", "rock", "dance", "edm", "indie"],
        seeds=[1, 2, 3, 4, 5],
        pool_size=EXPERIMENT_POOL_SIZE,
        output_results_path=str(
            project_root / "outputs" / "results" / "experiment_results.csv"
        ),
        output_summary_path=str(
            project_root / "outputs" / "results" / "experiment_summary.csv"
        ),
    )
    raw_results, summary = ExperimentRunner(config).run_all()

    print_section("EXPERIMENTS: RUN CONFIGURATION")
    print(f"RUN_EXPERIMENTS: {RUN_EXPERIMENTS}")
    print(f"Genres: {config.genres}")
    print(f"Seeds: {config.seeds}")
    print(f"EXPERIMENT_POOL_SIZE: {EXPERIMENT_POOL_SIZE}")

    print_section("EXPERIMENTS: RAW RESULTS HEAD")
    print_table(raw_results.head())

    print_section("EXPERIMENTS: SUMMARY")
    print_table(summary)

    print_section("EXPERIMENTS: OUTPUT FILES")
    print(f"Raw results CSV: {Path(config.output_results_path).resolve()}")
    print(f"Summary CSV: {Path(config.output_summary_path).resolve()}")


if __name__ == "__main__":
    if RUN_EXPERIMENTS:
        run_experiments()
    else:
        main()
