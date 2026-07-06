"""Run the playlist optimizer demo or experiment grid."""

from itertools import permutations
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
from src.cost_functions import (
    compute_transition_component_matrices,
    compute_transition_cost_matrix,
)
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
from src.arcs import TargetArcConfig, create_target_arc

RUN_EXPERIMENTS = False
RUN_BIBS_SWEEP = False
POOL_SIZE = 150
EXPERIMENT_POOL_SIZE = 150
GENRE = "pop"
RANDOM_SEED = 42
SEPARATOR_WIDTH = 90
# Target-arc shape: "linear" reproduces the original pipeline exactly. Other options:
# "double_peak", "log_rise", "inverted_parabola", "wave" (see src/arcs.py).
TARGET_ARC_SHAPE = "linear"


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
        csv_path=project_root.parent / "data" / "spotify_tracks.csv",
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
    transition_components = compute_transition_component_matrices(track_pool)
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
    target_arc = create_target_arc(
        track_pool,
        start_index,
        end_index,
        TargetArcConfig(shape=TARGET_ARC_SHAPE),
    )
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
    bibs_config = BIBSConfig()
    bibs = BIBS(bibs_config)
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
    bibs_diagnostics = bibs.get_diagnostics()
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

    def playlist_coverage_report(
        label: str,
        playlist: list[int | None],
    ) -> dict[str, object]:
        expected_length = len(track_pool)
        valid_values = [
            track
            for track in playlist
            if track is not None
            and isinstance(track, (int, np.integer))
            and 0 <= int(track) < expected_length
        ]
        valid_indices = [int(track) for track in valid_values]
        unique_tracks = set(valid_indices)
        duplicate_tracks = sorted(
            {
                track
                for track in valid_indices
                if valid_indices.count(track) > 1
            }
        )
        missing_tracks = sorted(set(range(expected_length)).difference(unique_tracks))
        out_of_range_values = [
            track
            for track in playlist
            if track is not None
            and (
                not isinstance(track, (int, np.integer))
                or not 0 <= int(track) < expected_length
            )
        ]
        none_positions = [
            position for position, track in enumerate(playlist) if track is None
        ]
        duplicate_count = len(valid_indices) - len(unique_tracks)
        none_count = len(none_positions)
        out_of_range_count = len(out_of_range_values)
        start_position_correct = bool(playlist and playlist[0] == start_index)
        end_position_correct = bool(playlist and playlist[-1] == end_index)
        exact_once_valid = (
            len(playlist) == expected_length
            and len(unique_tracks) == expected_length
            and duplicate_count == 0
            and len(missing_tracks) == 0
            and none_count == 0
            and out_of_range_count == 0
            and start_position_correct
            and end_position_correct
        )
        return {
            "playlist": label,
            "expected_length": expected_length,
            "actual_length": len(playlist),
            "unique_tracks": len(unique_tracks),
            "duplicate_count": duplicate_count,
            "missing_count": len(missing_tracks),
            "none_count": none_count,
            "out_of_range_count": out_of_range_count,
            "start_position_correct": start_position_correct,
            "end_position_correct": end_position_correct,
            "exact_once_valid": exact_once_valid,
            "missing_tracks": missing_tracks,
            "duplicate_tracks": duplicate_tracks,
            "out_of_range_values": out_of_range_values,
            "none_positions": none_positions,
        }

    coverage_reports = [
        playlist_coverage_report("Greedy", greedy_playlist),
        playlist_coverage_report("BIBS", bibs_playlist),
    ]

    def coverage_failure_details() -> list[dict[str, object]]:
        return [
            {
                "playlist": report["playlist"],
                "first_20_missing_track_indices": report["missing_tracks"][:20],
                "first_20_duplicate_track_indices": report["duplicate_tracks"][:20],
                "first_20_out_of_range_values": report["out_of_range_values"][:20],
                "none_positions": report["none_positions"],
            }
            for report in coverage_reports
            if not report["exact_once_valid"]
        ]

    def coverage_structure_violations() -> list[dict[str, object]]:
        violations: list[dict[str, object]] = []
        for report in coverage_reports:
            playlist_name = str(report["playlist"])
            if report["actual_length"] != report["expected_length"]:
                violations.append(
                    {"playlist": playlist_name, "violation": "length mismatch"}
                )
            if report["duplicate_count"] > 0:
                violations.append(
                    {"playlist": playlist_name, "violation": "duplicate tracks"}
                )
            if report["missing_count"] > 0:
                violations.append(
                    {"playlist": playlist_name, "violation": "missing tracks"}
                )
            if report["none_count"] > 0:
                violations.append(
                    {"playlist": playlist_name, "violation": "None values"}
                )
            if report["out_of_range_count"] > 0:
                violations.append(
                    {"playlist": playlist_name, "violation": "out-of-range values"}
                )
            if not report["start_position_correct"]:
                violations.append(
                    {"playlist": playlist_name, "violation": "start mismatch"}
                )
            if not report["end_position_correct"]:
                violations.append(
                    {"playlist": playlist_name, "violation": "end mismatch"}
                )
        return violations

    recursion_trace_columns = [
        "depth",
        "left_pos",
        "right_pos",
        "interval_size",
        "empty_count",
        "midpoint",
        "is_base_case",
        "forward_positions_start",
        "forward_positions_end",
        "forward_positions_count",
        "backward_positions_start",
        "backward_positions_end",
        "backward_positions_count",
        "selected_anchor_position",
        "selected_anchor_track",
        "left_child_interval",
        "right_child_interval",
    ]
    recursion_trace = bibs_diagnostics["recursion_trace"]
    beam_trace_columns = [
        "direction",
        "position",
        "chunk_index",
        "position_index_in_chunk",
        "total_positions_in_call",
        "beam_length",
        "beam_width",
        "beams_before_expansion",
        "expanded_candidate_count",
        "beams_after_pruning",
        "returned_early_due_to_no_expansion",
    ]
    beam_trace = bibs_diagnostics["beam_trace"]
    anchor_trace_columns = [
        "interval_left_pos",
        "interval_right_pos",
        "midpoint",
        "selected_anchor_track",
        "selected_left_neighbor",
        "selected_right_neighbor",
        "candidate_count",
        "selected_anchor_score",
        "selected_anchor_rank",
        "score_gap_from_best",
        "raw_arc_cost",
        "raw_bottleneck_score",
        "left_transition_cost",
        "right_transition_cost",
        "balance_gap",
        "arc_component",
        "transition_component",
        "bottleneck_component",
        "balance_component",
        "selected_candidate_sources",
        "whether_selected_candidate_from_graph",
        "whether_selected_candidate_from_candidate_set",
        "whether_selected_candidate_from_bottleneck",
        "whether_selected_candidate_from_bridge",
    ]
    anchor_selection_trace = bibs_diagnostics["anchor_selection_trace"]
    base_case_trace_columns = [
        "left_pos",
        "right_pos",
        "interval_size",
        "empty_count",
        "empty_positions",
        "candidate_count",
        "initial_candidate_count",
        "forced_assignment_warning",
        "permutation_count_tried",
        "selected_tracks",
        "local_score_total",
        "arc_component",
        "transition_component",
        "bottleneck_component",
        "average_arc_cost_selected",
        "total_transition_cost_selected",
        "first_transition_cost",
        "last_transition_cost",
    ]
    base_case_trace = bibs_diagnostics["base_case_trace"]

    def recursion_structure_violations() -> list[dict[str, object]]:
        violations: list[dict[str, object]] = []
        for row_index, row in enumerate(recursion_trace):
            if row["is_base_case"]:
                continue
            row_violations: list[str] = []
            if row["selected_anchor_position"] != row["midpoint"]:
                row_violations.append("selected_anchor_position != midpoint")
            if row["forward_positions_count"] > 0:
                if row["forward_positions_start"] != row["left_pos"] + 1:
                    row_violations.append("forward_positions_start mismatch")
                if row["forward_positions_end"] != row["midpoint"] - 1:
                    row_violations.append("forward_positions_end mismatch")
            if row["backward_positions_count"] > 0:
                if row["backward_positions_start"] != row["right_pos"] - 1:
                    row_violations.append("backward_positions_start mismatch")
                if row["backward_positions_end"] != row["midpoint"] + 1:
                    row_violations.append("backward_positions_end mismatch")
            if row["left_child_interval"] != [row["left_pos"], row["midpoint"]]:
                row_violations.append("left_child_interval mismatch")
            if row["right_child_interval"] != [row["midpoint"], row["right_pos"]]:
                row_violations.append("right_child_interval mismatch")
            if row_violations:
                violations.append(
                    {
                        "row_index": row_index,
                        "depth": row["depth"],
                        "left_pos": row["left_pos"],
                        "right_pos": row["right_pos"],
                        "violations": "; ".join(row_violations),
                    }
                )
        return violations

    def beam_structure_violations() -> list[dict[str, object]]:
        violations: list[dict[str, object]] = []
        for row_index, row in enumerate(beam_trace):
            row_violations: list[str] = []
            if row["beams_after_pruning"] > row["beam_width"]:
                row_violations.append("beams_after_pruning > beam_width")
            if row["position_index_in_chunk"] >= row["beam_length"]:
                row_violations.append("position_index_in_chunk >= beam_length")
            if row["beam_length"] != bibs_config.beam_length:
                row_violations.append("beam_length != BIBSConfig.beam_length")
            if row["beam_width"] != bibs_config.beam_width:
                row_violations.append("beam_width != BIBSConfig.beam_width")
            if row_violations:
                violations.append(
                    {
                        "row_index": row_index,
                        "direction": row["direction"],
                        "position": row["position"],
                        "violations": "; ".join(row_violations),
                    }
                )
        return violations

    def anchor_structure_violations() -> list[dict[str, object]]:
        violations: list[dict[str, object]] = []
        component_names = [
            "arc_component",
            "transition_component",
            "bottleneck_component",
            "balance_component",
        ]
        for row_index, row in enumerate(anchor_selection_trace):
            row_violations: list[str] = []
            if row["selected_anchor_track"] is None:
                row_violations.append("selected_anchor_track is null")
            if not np.isfinite(float(row["selected_anchor_score"])):
                row_violations.append("selected_anchor_score is not finite")
            if row["candidate_count"] <= 0:
                row_violations.append("candidate_count <= 0")
            if row["selected_anchor_rank"] is None or row["selected_anchor_rank"] < 1:
                row_violations.append("selected_anchor_rank < 1")
            if row["score_gap_from_best"] < 0:
                row_violations.append("score_gap_from_best < 0")
            for component_name in component_names:
                if not np.isfinite(float(row[component_name])):
                    row_violations.append(f"{component_name} is not finite")
            if row["transition_component"] < 0:
                row_violations.append("transition_component < 0")
            if row["balance_component"] < 0:
                row_violations.append("balance_component < 0")
            if row_violations:
                violations.append(
                    {
                        "row_index": row_index,
                        "midpoint": row["midpoint"],
                        "selected_anchor_track": row["selected_anchor_track"],
                        "violations": "; ".join(row_violations),
                    }
                )
        return violations

    max_reconstruction_error = float(
        np.max(np.abs(c_trans - transition_components["total_costs"]))
    )
    average_weighted_rhythm_cost = float(
        np.mean(transition_components["weighted_rhythm_costs"])
    )
    average_weighted_tonality_cost = float(
        np.mean(transition_components["weighted_tonality_costs"])
    )
    average_weighted_texture_cost = float(
        np.mean(transition_components["weighted_texture_costs"])
    )
    average_total_transition_cost = float(
        np.mean(transition_components["total_costs"])
    )
    if average_total_transition_cost:
        rhythm_share = average_weighted_rhythm_cost / average_total_transition_cost
        tonality_share = (
            average_weighted_tonality_cost / average_total_transition_cost
        )
        texture_share = average_weighted_texture_cost / average_total_transition_cost
    else:
        rhythm_share = 0.0
        tonality_share = 0.0
        texture_share = 0.0

    def c_trans_component_rows() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        pair_count = min(10, len(bibs_playlist) - 1)
        for position in range(pair_count):
            source_track = bibs_playlist[position]
            target_track = bibs_playlist[position + 1]
            rows.append(
                {
                    "position": position,
                    "source_track": source_track,
                    "target_track": target_track,
                    "source_camelot": track_pool.loc[source_track, "camelot"],
                    "target_camelot": track_pool.loc[target_track, "camelot"],
                    "source_tempo": float(track_pool.loc[source_track, "tempo"]),
                    "target_tempo": float(track_pool.loc[target_track, "tempo"]),
                    "weighted_rhythm_cost": float(
                        transition_components["weighted_rhythm_costs"][
                            source_track,
                            target_track,
                        ]
                    ),
                    "weighted_tonality_cost": float(
                        transition_components["weighted_tonality_costs"][
                            source_track,
                            target_track,
                        ]
                    ),
                    "weighted_texture_cost": float(
                        transition_components["weighted_texture_costs"][
                            source_track,
                            target_track,
                        ]
                    ),
                    "total_transition_cost": float(
                        c_trans[source_track, target_track]
                    ),
                    "reconstructed_total_cost": float(
                        transition_components["total_costs"][
                            source_track,
                            target_track,
                        ]
                    ),
                }
            )
        return rows

    def c_trans_structure_violations() -> list[dict[str, object]]:
        violations: list[dict[str, object]] = []
        if max_reconstruction_error >= 1e-9:
            violations.append(
                {
                    "check": "max_reconstruction_error",
                    "details": f"{max_reconstruction_error:.12f} >= 1e-9",
                }
            )
        for name, matrix in transition_components.items():
            if matrix.shape != c_trans.shape:
                violations.append(
                    {
                        "check": f"{name}_shape",
                        "details": f"{matrix.shape} != {c_trans.shape}",
                    }
                )
            if not np.isfinite(matrix).all():
                violations.append(
                    {
                        "check": f"{name}_finite",
                        "details": "matrix contains non-finite values",
                    }
                )
            if np.any(matrix < 0):
                violations.append(
                    {
                        "check": f"{name}_non_negative",
                        "details": "matrix contains negative values",
                    }
                )
        return violations

    has_camelot_columns = {"camelot_number", "camelot_mode"}.issubset(
        track_pool.columns
    )
    if has_camelot_columns:
        camelot_numbers = pd.to_numeric(
            track_pool["camelot_number"], errors="raise"
        ).to_numpy(dtype=int)
        camelot_modes = (
            track_pool["camelot_mode"].astype(str).str.strip().str.upper().to_numpy()
        )
        direct_camelot_distances = np.abs(
            camelot_numbers[:, np.newaxis] - camelot_numbers[np.newaxis, :]
        )
        camelot_sector_distances = np.minimum(
            direct_camelot_distances,
            12 - direct_camelot_distances,
        )
        camelot_mode_distances = (
            camelot_modes[:, np.newaxis] != camelot_modes[np.newaxis, :]
        ).astype(int)
    else:
        camelot_numbers = np.asarray([], dtype=int)
        camelot_modes = np.asarray([], dtype=str)
        camelot_sector_distances = np.asarray([[]], dtype=int)
        camelot_mode_distances = np.asarray([[]], dtype=int)

    def find_camelot_pair(kind: str) -> tuple[int, int] | None:
        if not has_camelot_columns:
            return None
        track_indices = list(range(len(track_pool)))
        preferred_pairs = {
            "same_exact": lambda source, target: (
                source != target
                and track_pool.loc[source, "camelot"] == track_pool.loc[target, "camelot"]
            ),
            "same_number_different_mode": lambda source, target: (
                source != target
                and camelot_numbers[source] == camelot_numbers[target]
                and camelot_modes[source] != camelot_modes[target]
            ),
            "neighboring_number_same_mode": lambda source, target: (
                source != target
                and camelot_sector_distances[source, target] == 1
                and camelot_modes[source] == camelot_modes[target]
                and (
                    (camelot_numbers[source], camelot_numbers[target]) in {(1, 12), (12, 1)}
                    or True
                )
            ),
            "larger_jump": lambda source, target: (
                source != target
                and camelot_sector_distances[source, target] >= 4
            ),
        }
        if kind == "neighboring_number_same_mode":
            for source in track_indices:
                for target in track_indices:
                    if (
                        source != target
                        and camelot_modes[source] == camelot_modes[target]
                        and (camelot_numbers[source], camelot_numbers[target])
                        in {(1, 12), (12, 1)}
                    ):
                        return source, target
        if kind == "larger_jump":
            for source in track_indices:
                for target in track_indices:
                    if (
                        source != target
                        and (track_pool.loc[source, "camelot"], track_pool.loc[target, "camelot"])
                        in {("12B", "8B"), ("8B", "12B")}
                    ):
                        return source, target
        predicate = preferred_pairs[kind]
        for source in track_indices:
            for target in track_indices:
                if predicate(source, target):
                    return source, target
        return None

    def camelot_example_rows() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        examples = [
            ("same_exact_camelot", find_camelot_pair("same_exact")),
            (
                "same_number_different_mode",
                find_camelot_pair("same_number_different_mode"),
            ),
            (
                "neighboring_number_same_mode",
                find_camelot_pair("neighboring_number_same_mode"),
            ),
            ("larger_jump", find_camelot_pair("larger_jump")),
        ]
        for example_type, pair in examples:
            if pair is None:
                continue
            source_track, target_track = pair
            rows.append(
                {
                    "example_type": example_type,
                    "source_track": source_track,
                    "target_track": target_track,
                    "source_camelot": track_pool.loc[source_track, "camelot"],
                    "target_camelot": track_pool.loc[target_track, "camelot"],
                    "source_camelot_number": int(camelot_numbers[source_track]),
                    "target_camelot_number": int(camelot_numbers[target_track]),
                    "source_camelot_mode": camelot_modes[source_track],
                    "target_camelot_mode": camelot_modes[target_track],
                    "camelot_sector_distance": int(
                        camelot_sector_distances[source_track, target_track]
                    ),
                    "camelot_mode_distance": int(
                        camelot_mode_distances[source_track, target_track]
                    ),
                    "raw_tonality_distance": float(
                        transition_components["tonality_costs"][
                            source_track,
                            target_track,
                        ]
                    ),
                    "weighted_tonality_cost": float(
                        transition_components["weighted_tonality_costs"][
                            source_track,
                            target_track,
                        ]
                    ),
                    "total_transition_cost": float(
                        c_trans[source_track, target_track]
                    ),
                }
            )
        return rows

    def camelot_structure_violations() -> list[dict[str, object]]:
        violations: list[dict[str, object]] = []
        if not has_camelot_columns:
            return [{"check": "camelot_columns", "details": "missing Camelot columns"}]
        circular_pair = None
        for source in range(len(track_pool)):
            for target in range(len(track_pool)):
                if (
                    source != target
                    and camelot_modes[source] == camelot_modes[target]
                    and (camelot_numbers[source], camelot_numbers[target])
                    in {(1, 12), (12, 1)}
                ):
                    circular_pair = (source, target)
                    break
            if circular_pair is not None:
                break
        if circular_pair is not None:
            source, target = circular_pair
            if camelot_sector_distances[source, target] != 1:
                violations.append(
                    {
                        "check": "12_1_circular_distance",
                        "details": f"pair {source}->{target} sector distance != 1",
                    }
                )
        same_exact_pair = find_camelot_pair("same_exact")
        if same_exact_pair is not None:
            source, target = same_exact_pair
            if transition_components["tonality_costs"][source, target] != 0:
                violations.append(
                    {
                        "check": "same_exact_camelot",
                        "details": f"pair {source}->{target} tonality != 0",
                    }
                )
        same_number_mode_pair = find_camelot_pair("same_number_different_mode")
        if same_number_mode_pair is not None:
            source, target = same_number_mode_pair
            if camelot_sector_distances[source, target] != 0:
                violations.append(
                    {
                        "check": "same_number_sector_distance",
                        "details": f"pair {source}->{target} sector distance != 0",
                    }
                )
            if camelot_mode_distances[source, target] != 1:
                violations.append(
                    {
                        "check": "same_number_mode_distance",
                        "details": f"pair {source}->{target} mode distance != 1",
                    }
                )
        if np.any((camelot_sector_distances < 0) | (camelot_sector_distances > 6)):
            violations.append(
                {
                    "check": "camelot_sector_distance_range",
                    "details": "sector distances must be between 0 and 6",
                }
            )
        if not np.isin(camelot_mode_distances, [0, 1]).all():
            violations.append(
                {
                    "check": "camelot_mode_distance_values",
                    "details": "mode distances must be 0 or 1",
                }
            )
        return violations

    def base_case_structure_violations() -> list[dict[str, object]]:
        violations: list[dict[str, object]] = []
        for row_index, row in enumerate(base_case_trace):
            row_violations: list[str] = []
            selected_tracks = list(row["selected_tracks"])
            if row["empty_count"] <= 0:
                row_violations.append("empty_count <= 0")
            if row["empty_count"] > bibs_config.base_case_size:
                row_violations.append("empty_count > base_case_size")
            if row["candidate_count"] < row["empty_count"]:
                row_violations.append("candidate_count < empty_count")
            if row["permutation_count_tried"] <= 0:
                row_violations.append("permutation_count_tried <= 0")
            if len(selected_tracks) != row["empty_count"]:
                row_violations.append("len(selected_tracks) != empty_count")
            if len(selected_tracks) != len(set(selected_tracks)):
                row_violations.append("selected_tracks has duplicates")
            if not np.isfinite(float(row["local_score_total"])):
                row_violations.append("local_score_total is not finite")
            if (
                not np.isfinite(float(row["transition_component"]))
                or row["transition_component"] < 0
            ):
                row_violations.append(
                    "transition_component is not finite/non-negative"
                )
            if (
                not np.isfinite(float(row["total_transition_cost_selected"]))
                or row["total_transition_cost_selected"] < 0
            ):
                row_violations.append(
                    "total_transition_cost_selected is not finite/non-negative"
                )
            if row_violations:
                violations.append(
                    {
                        "row_index": row_index,
                        "left_pos": row["left_pos"],
                        "right_pos": row["right_pos"],
                        "violations": "; ".join(row_violations),
                    }
                )
        return violations

    removed_bibs_config_fields = {
        "location_bottleneck_weight",
        "energy_momentum_weight",
        "harmonic_diversity_weight",
        "anchor_location_bottleneck_weight",
        "anchor_energy_momentum_weight",
        "base_energy_momentum_weight",
        "harmonic_window",
        "max_same_camelot_in_window",
        "max_same_camelot_number_in_window",
        "harmonic_stagnation_penalty",
        "high_energy_threshold",
        "allowed_ev_drop",
        "ev_drop_weight",
        "max_preferred_camelot_jump",
        "large_camelot_jump_penalty",
        "extreme_camelot_jump_penalty",
        "enable_harmonic_motion_penalty",
        "high_arc_threshold",
        "max_allowed_arc_deviation_high_energy",
        "high_energy_arc_penalty_weight",
        "high_energy_drop_penalty_weight",
        "late_position_fraction",
        "late_low_ev_penalty_weight",
        "feasibility_penalty_weight",
        "min_candidates_per_position",
        "base_case_extra_candidates_per_position",
        "bad_transition_threshold",
        "bad_transition_penalty_weight",
        "base_high_energy_penalty_weight",
    }
    current_bibs_config_fields = set(bibs_config.__dataclass_fields__)
    config_simplified = not removed_bibs_config_fields.intersection(
        current_bibs_config_fields
    )

    structure_violations = recursion_structure_violations()
    beam_violations = beam_structure_violations()
    anchor_violations = anchor_structure_violations()
    c_trans_violations = c_trans_structure_violations()
    camelot_violations = camelot_structure_violations()
    base_case_violations = base_case_structure_violations()
    coverage_violations = coverage_structure_violations()

    coverage_checks_passed = (
        len(coverage_violations) == 0
        and all(report["exact_once_valid"] for report in coverage_reports)
    )
    structural_checks_passed = (
        len(structure_violations) == 0
        and len(beam_violations) == 0
        and len(anchor_violations) == 0
        and len(c_trans_violations) == 0
        and len(camelot_violations) == 0
        and len(base_case_violations) == 0
    )
    methodology_diagnostics_ready = (
        bibs_diagnostics["recursion_trace_count"] > 0
        and bibs_diagnostics["beam_trace_count"] > 0
        and bibs_diagnostics["anchor_selection_trace_count"] > 0
        and bibs_diagnostics["base_case_trace_count"] > 0
        and len(coverage_reports) > 0
    )
    greedy_total_transition = greedy_metrics.get("total_transition_cost")
    bibs_total_transition = bibs_metrics.get("total_transition_cost")
    greedy_global_coherence = greedy_metrics.get("global_coherence")
    bibs_global_coherence = bibs_metrics.get("global_coherence")
    needs_metric_tuning = (
        (
            greedy_total_transition is not None
            and bibs_total_transition is not None
            and bibs_total_transition > greedy_total_transition
        )
        or (
            greedy_global_coherence is not None
            and bibs_global_coherence is not None
            and bibs_global_coherence < greedy_global_coherence
        )
    )
    implementation_structurally_valid = structural_checks_passed
    implementation_exact_once_valid = coverage_checks_passed
    implementation_methodology_aligned = (
        methodology_diagnostics_ready and structural_checks_passed
    )
    recommended_next_step = (
        "Start tuning BIBS weights and bottleneck/anchor balance after repeated tests."
        if needs_metric_tuning
        else "Run larger experiments and export final results."
    )

    def playlist_transition_costs(playlist: list[int]) -> np.ndarray:
        costs = [
            float(c_trans[source_track, target_track])
            for source_track, target_track in zip(playlist[:-1], playlist[1:])
        ]
        return np.asarray(costs, dtype=float)

    def transition_distribution_rows() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for playlist_name, playlist in [
            ("Greedy", greedy_playlist),
            ("BIBS", bibs_playlist),
        ]:
            transition_costs = playlist_transition_costs(playlist)
            rows.append(
                {
                    "playlist": playlist_name,
                    "transition_count": len(transition_costs),
                    "mean_transition": float(np.mean(transition_costs)),
                    "median_transition": float(np.median(transition_costs)),
                    "p75_transition": float(np.percentile(transition_costs, 75)),
                    "p90_transition": float(np.percentile(transition_costs, 90)),
                    "p95_transition": float(np.percentile(transition_costs, 95)),
                    "max_transition": float(np.max(transition_costs)),
                    "total_transition": float(np.sum(transition_costs)),
                }
            )
        return rows

    def worst_bibs_transition_rows(limit: int = 15) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for position, (source_track, target_track) in enumerate(
            zip(bibs_playlist[:-1], bibs_playlist[1:])
        ):
            rows.append(
                {
                    "position": position,
                    "source_track": source_track,
                    "target_track": target_track,
                    "source_track_name": track_pool.loc[source_track, "track_name"],
                    "target_track_name": track_pool.loc[target_track, "track_name"],
                    "source_artists": track_pool.loc[source_track, "artists"],
                    "target_artists": track_pool.loc[target_track, "artists"],
                    "source_camelot": track_pool.loc[source_track, "camelot"],
                    "target_camelot": track_pool.loc[target_track, "camelot"],
                    "source_tempo": float(track_pool.loc[source_track, "tempo"]),
                    "target_tempo": float(track_pool.loc[target_track, "tempo"]),
                    "rhythm_cost_weighted": float(
                        transition_components["weighted_rhythm_costs"][
                            source_track,
                            target_track,
                        ]
                    ),
                    "tonality_cost_weighted": float(
                        transition_components["weighted_tonality_costs"][
                            source_track,
                            target_track,
                        ]
                    ),
                    "texture_cost_weighted": float(
                        transition_components["weighted_texture_costs"][
                            source_track,
                            target_track,
                        ]
                    ),
                    "total_transition_cost": float(
                        c_trans[source_track, target_track]
                    ),
                }
            )
        return sorted(
            rows,
            key=lambda row: row["total_transition_cost"],
            reverse=True,
        )[:limit]

    anchor_arc_average = bibs_diagnostics["average_anchor_arc_component"]
    anchor_transition_average = bibs_diagnostics[
        "average_anchor_transition_component"
    ]
    transition_to_arc_ratio = (
        anchor_transition_average / anchor_arc_average
        if anchor_arc_average
        else np.nan
    )
    anchor_trace_count = bibs_diagnostics["anchor_selection_trace_count"]
    selected_bottleneck_count = bibs_diagnostics[
        "selected_anchor_from_bottleneck_count"
    ]
    bottleneck_anchor_share = (
        selected_bottleneck_count / anchor_trace_count
        if anchor_trace_count
        else np.nan
    )
    bottleneck_track_count = len(
        bottleneck_results.get("bottleneck_track_indices", [])
    )
    bottleneck_recommendation = (
        "Increase bottleneck influence or candidate exposure after reviewing metric stability."
        if bottleneck_anchor_share < 0.2
        else "Bottleneck influence appears visible."
    )
    greedy_coverage_report = next(
        report for report in coverage_reports if report["playlist"] == "Greedy"
    )
    greedy_uses_same_c_trans_assumption = True
    greedy_uses_same_c_arc_assumption = True
    greedy_compared_with_same_evaluator = True

    def sorted_full_neighbors(source_track: int) -> list[int]:
        return sorted(
            (track for track in range(len(track_pool)) if track != source_track),
            key=lambda target_track: (
                float(c_trans[source_track, target_track]),
                target_track,
            ),
        )

    def rank_track_by_cost(
        target_track: int,
        candidates: list[int],
        cost_function,
    ) -> int | None:
        ranked_candidates = sorted(
            candidates,
            key=lambda candidate: (float(cost_function(candidate)), candidate),
        )
        try:
            return ranked_candidates.index(target_track) + 1
        except ValueError:
            return None

    def greedy_local_decision_rows() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        all_tracks = set(range(len(track_pool)))
        for position, source_track in enumerate(greedy_playlist[:-1]):
            selected_target_track = greedy_playlist[position + 1]
            used_tracks = set(greedy_playlist[: position + 1])
            remaining_tracks = all_tracks.difference(used_tracks)
            if position + 1 == len(greedy_playlist) - 1:
                valid_candidates = [end_index] if end_index in remaining_tracks else []
            else:
                valid_candidates = sorted(remaining_tracks.difference({end_index}))
            selected_transition_cost = float(
                c_trans[source_track, selected_target_track]
            )
            selected_arc_cost = float(c_arc[selected_target_track, position + 1])
            selected_combined_cost = selected_transition_cost + selected_arc_cost
            if valid_candidates:
                transition_costs = [
                    float(c_trans[source_track, candidate])
                    for candidate in valid_candidates
                ]
                combined_costs = [
                    float(c_trans[source_track, candidate])
                    + float(c_arc[candidate, position + 1])
                    for candidate in valid_candidates
                ]
                best_possible_transition_cost = min(transition_costs)
                best_possible_combined_cost = min(combined_costs)
                selected_transition_rank = rank_track_by_cost(
                    selected_target_track,
                    valid_candidates,
                    lambda candidate: c_trans[source_track, candidate],
                )
                selected_combined_rank = rank_track_by_cost(
                    selected_target_track,
                    valid_candidates,
                    lambda candidate: (
                        c_trans[source_track, candidate]
                        + c_arc[candidate, position + 1]
                    ),
                )
            else:
                best_possible_transition_cost = np.nan
                best_possible_combined_cost = np.nan
                selected_transition_rank = None
                selected_combined_rank = None
            rows.append(
                {
                    "position": position,
                    "source_track": source_track,
                    "selected_target_track": selected_target_track,
                    "selected_transition_cost": selected_transition_cost,
                    "best_possible_transition_cost": best_possible_transition_cost,
                    "selected_transition_rank": selected_transition_rank,
                    "selected_transition_gap_from_best": (
                        selected_transition_cost - best_possible_transition_cost
                    ),
                    "selected_arc_cost": selected_arc_cost,
                    "selected_combined_cost": selected_combined_cost,
                    "best_possible_combined_cost": best_possible_combined_cost,
                    "selected_combined_rank": selected_combined_rank,
                    "selected_combined_gap_from_best": (
                        selected_combined_cost - best_possible_combined_cost
                    ),
                    "source_camelot": track_pool.loc[source_track, "camelot"],
                    "selected_target_camelot": track_pool.loc[
                        selected_target_track,
                        "camelot",
                    ],
                    "source_tempo": float(track_pool.loc[source_track, "tempo"]),
                    "selected_target_tempo": float(
                        track_pool.loc[selected_target_track, "tempo"]
                    ),
                }
            )
        return rows

    greedy_audit_rows = greedy_local_decision_rows()
    greedy_audit_frame = pd.DataFrame(greedy_audit_rows)
    greedy_audit_positions = len(greedy_audit_frame)
    greedy_transition_rank_1_count = int(
        (greedy_audit_frame["selected_transition_rank"] == 1).sum()
    )
    greedy_transition_rank_above_10_count = int(
        (greedy_audit_frame["selected_transition_rank"] > 10).sum()
    )

    def greedy_fairness_violations() -> list[dict[str, object]]:
        violations: list[dict[str, object]] = []
        if not greedy_coverage_report["exact_once_valid"]:
            violations.append({"check": "greedy_exact_once_valid"})
        if not greedy_coverage_report["start_position_correct"]:
            violations.append({"check": "fixed_start"})
        if not greedy_coverage_report["end_position_correct"]:
            violations.append({"check": "fixed_end"})
        if greedy_coverage_report["none_count"] > 0:
            violations.append({"check": "none_values"})
        if greedy_coverage_report["duplicate_count"] > 0:
            violations.append({"check": "duplicates"})
        if greedy_coverage_report["missing_count"] > 0:
            violations.append({"check": "missing_tracks"})
        if len(greedy_playlist) != len(bibs_playlist):
            violations.append({"check": "same_pool_size"})
        if not greedy_compared_with_same_evaluator:
            violations.append({"check": "same_evaluator_metrics"})
        if not greedy_uses_same_c_trans_assumption:
            violations.append({"check": "same_c_trans"})
        return violations

    outgoing_neighbors_with_costs = graph_data.outgoing_neighbors_with_costs
    incoming_neighbors_with_costs = graph_data.incoming_neighbors_with_costs
    graph_node_count = len(outgoing_neighbors_with_costs)
    expected_node_count = len(track_pool)
    outgoing_degrees = [
        len(outgoing_neighbors_with_costs.get(node, []))
        for node in range(expected_node_count)
    ]
    incoming_degrees = [
        len(incoming_neighbors_with_costs.get(node, []))
        for node in range(expected_node_count)
    ]
    graph_edge_costs = np.asarray(
        [
            float(cost)
            for neighbors in outgoing_neighbors_with_costs.values()
            for _, cost in neighbors
        ],
        dtype=float,
    )

    def graph_recall_metrics() -> dict[str, float]:
        recall_totals = {5: [], 10: [], 20: []}
        best_full_costs: list[float] = []
        best_graph_costs: list[float] = []
        graph_best_gaps: list[float] = []
        for source_track in range(expected_node_count):
            full_neighbors = sorted_full_neighbors(source_track)
            graph_neighbors = {
                target_track
                for target_track, _ in outgoing_neighbors_with_costs.get(
                    source_track,
                    [],
                )
            }
            for top_k in recall_totals:
                top_full_neighbors = set(full_neighbors[:top_k])
                recall_totals[top_k].append(
                    len(top_full_neighbors.intersection(graph_neighbors)) / top_k
                )
            best_full_target = full_neighbors[0]
            best_full_cost = float(c_trans[source_track, best_full_target])
            best_full_costs.append(best_full_cost)
            graph_costs = [
                float(cost)
                for _, cost in outgoing_neighbors_with_costs.get(source_track, [])
            ]
            if graph_costs:
                best_graph_cost = min(graph_costs)
                best_graph_costs.append(best_graph_cost)
                graph_best_gaps.append(best_graph_cost - best_full_cost)
        return {
            "graph_recall_at_5": float(np.mean(recall_totals[5])),
            "graph_recall_at_10": float(np.mean(recall_totals[10])),
            "graph_recall_at_20": float(np.mean(recall_totals[20])),
            "average_best_full_neighbor_cost": float(np.mean(best_full_costs)),
            "average_best_graph_neighbor_cost": float(np.mean(best_graph_costs)),
            "average_graph_best_gap_from_full_best": float(np.mean(graph_best_gaps)),
            "max_graph_best_gap_from_full_best": float(np.max(graph_best_gaps)),
        }

    graph_recall = graph_recall_metrics()

    def bibs_graph_transition_rows(
        playlist: list[int] | None = None,
    ) -> list[dict[str, object]]:
        playlist = bibs_playlist if playlist is None else playlist
        rows: list[dict[str, object]] = []
        for position, (source_track, selected_target_track) in enumerate(
            zip(playlist[:-1], playlist[1:])
        ):
            graph_neighbors = outgoing_neighbors_with_costs.get(source_track, [])
            graph_targets = [target for target, _ in graph_neighbors]
            full_neighbors = sorted_full_neighbors(source_track)
            selected_transition_cost = float(
                c_trans[source_track, selected_target_track]
            )
            best_graph_target_track = graph_targets[0] if graph_targets else None
            best_graph_neighbor_cost = (
                float(c_trans[source_track, best_graph_target_track])
                if best_graph_target_track is not None
                else np.nan
            )
            best_full_target_track = full_neighbors[0]
            best_full_neighbor_cost = float(
                c_trans[source_track, best_full_target_track]
            )
            selected_rank_in_graph = (
                graph_targets.index(selected_target_track) + 1
                if selected_target_track in graph_targets
                else np.nan
            )
            selected_rank_in_full = full_neighbors.index(selected_target_track) + 1
            rows.append(
                {
                    "position": position,
                    "source_track": source_track,
                    "selected_target_track": selected_target_track,
                    "selected_transition_cost": selected_transition_cost,
                    "selected_rank_in_graph": selected_rank_in_graph,
                    "selected_rank_in_full_matrix": selected_rank_in_full,
                    "best_graph_target_track": best_graph_target_track,
                    "best_graph_neighbor_cost": best_graph_neighbor_cost,
                    "best_full_target_track": best_full_target_track,
                    "best_full_neighbor_cost": best_full_neighbor_cost,
                    "gap_from_best_graph_neighbor": (
                        selected_transition_cost - best_graph_neighbor_cost
                    ),
                    "gap_from_best_full_neighbor": (
                        selected_transition_cost - best_full_neighbor_cost
                    ),
                    "source_camelot": track_pool.loc[source_track, "camelot"],
                    "selected_target_camelot": track_pool.loc[
                        selected_target_track,
                        "camelot",
                    ],
                    "best_graph_target_camelot": (
                        track_pool.loc[best_graph_target_track, "camelot"]
                        if best_graph_target_track is not None
                        else None
                    ),
                    "best_full_target_camelot": track_pool.loc[
                        best_full_target_track,
                        "camelot",
                    ],
                    "source_tempo": float(track_pool.loc[source_track, "tempo"]),
                    "selected_target_tempo": float(
                        track_pool.loc[selected_target_track, "tempo"]
                    ),
                    "best_graph_target_tempo": (
                        float(track_pool.loc[best_graph_target_track, "tempo"])
                        if best_graph_target_track is not None
                        else np.nan
                    ),
                    "best_full_target_tempo": float(
                        track_pool.loc[best_full_target_track, "tempo"]
                    ),
                }
            )
        return rows

    bibs_graph_rows = bibs_graph_transition_rows()
    bibs_graph_frame = pd.DataFrame(bibs_graph_rows)
    bibs_transitions_present_in_graph_count = int(
        bibs_graph_frame["selected_rank_in_graph"].notna().sum()
    )
    bibs_transitions_present_in_graph_share = (
        bibs_transitions_present_in_graph_count / len(bibs_graph_frame)
        if len(bibs_graph_frame)
        else np.nan
    )

    def graph_structure_violations() -> list[dict[str, object]]:
        violations: list[dict[str, object]] = []
        if graph_node_count != expected_node_count:
            violations.append({"check": "graph_node_count"})
        if any(degree == 0 for degree in outgoing_degrees):
            violations.append({"check": "zero_outgoing_degree"})
        if any(degree == 0 for degree in incoming_degrees):
            violations.append({"check": "zero_incoming_degree"})
        for source_track, neighbors in outgoing_neighbors_with_costs.items():
            if not 0 <= source_track < expected_node_count:
                violations.append({"check": "invalid_source_track"})
                break
            for target_track, cost in neighbors:
                if not 0 <= target_track < expected_node_count:
                    violations.append({"check": "invalid_target_track"})
                    break
                if not np.isfinite(float(cost)):
                    violations.append({"check": "non_finite_edge_cost"})
                    break
        if graph_recall["graph_recall_at_10"] < 0.5:
            violations.append({"check": "graph_recall_at_10_below_0.5"})
        if graph_recall["average_graph_best_gap_from_full_best"] > 0.5:
            violations.append({"check": "average_graph_best_gap_above_0.5"})
        return violations

    greedy_fairness_issue_rows = greedy_fairness_violations()
    graph_structure_issue_rows = graph_structure_violations()
    greedy_baseline_fair = len(greedy_fairness_issue_rows) == 0
    graph_audit_passed = len(graph_structure_issue_rows) == 0
    graph_candidate_exposure_sufficient = (
        graph_recall["graph_recall_at_10"] >= 0.5
        and graph_recall["average_graph_best_gap_from_full_best"] <= 0.5
    )
    if not greedy_baseline_fair:
        recommended_next_step = "Fix Greedy baseline before BIBS tuning."
    elif not graph_candidate_exposure_sufficient:
        recommended_next_step = (
            "Tune transition graph / candidate exposure before BIBS weights."
        )
    else:
        recommended_next_step = (
            "Proceed to BIBS weight and anchor/bottleneck tuning."
        )

    def transition_distribution_for_playlist(playlist: list[int]) -> dict[str, float]:
        transition_costs = playlist_transition_costs(playlist)
        return {
            "median_transition": float(np.median(transition_costs)),
            "p90_transition": float(np.percentile(transition_costs, 90)),
            "p95_transition": float(np.percentile(transition_costs, 95)),
            "max_transition": float(np.max(transition_costs)),
        }

    def experiment_structure_valid(
        diagnostics: dict[str, object],
        config: BIBSConfig,
    ) -> bool:
        return (
            diagnostics["recursion_trace_count"] > 0
            and diagnostics["beam_trace_count"] > 0
            and diagnostics["anchor_selection_trace_count"]
            == diagnostics["recursion_non_base_case_count"]
            and diagnostics["base_case_trace_count"] > 0
            and diagnostics["max_beams_after_pruning"] <= config.beam_width
        )

    tuning_configs = [
        ("baseline_current", BIBSConfig()),
        ("wider_beam_8", BIBSConfig(beam_width=8)),
        ("wider_beam_12", BIBSConfig(beam_width=12)),
        (
            "more_base_candidates",
            BIBSConfig(beam_width=8, base_case_max_candidates=16),
        ),
        (
            "anchor_more_arc",
            BIBSConfig(
                beam_width=8,
                anchor_arc_weight=1.6,
                anchor_transition_weight=1.0,
                anchor_balance_weight=0.6,
            ),
        ),
        (
            "anchor_more_balance",
            BIBSConfig(
                beam_width=8,
                anchor_arc_weight=1.3,
                anchor_transition_weight=1.0,
                anchor_balance_weight=1.0,
            ),
        ),
        (
            "bottleneck_boost",
            BIBSConfig(
                beam_width=8,
                bottleneck_weight=1.0,
                anchor_bottleneck_weight=1.2,
            ),
        ),
        (
            "combined_conservative",
            BIBSConfig(
                beam_width=8,
                base_case_max_candidates=16,
                anchor_arc_weight=1.4,
                anchor_transition_weight=1.0,
                anchor_balance_weight=0.8,
                bottleneck_weight=1.0,
                anchor_bottleneck_weight=1.2,
            ),
        ),
        (
            "transition_reduced_anchor",
            BIBSConfig(
                beam_width=8,
                anchor_arc_weight=1.5,
                anchor_transition_weight=0.8,
                anchor_balance_weight=1.0,
                bottleneck_weight=1.0,
                anchor_bottleneck_weight=1.2,
            ),
        ),
    ]

    tuning_rows: list[dict[str, object]] = []
    tuning_playlists: dict[str, list[int]] = {}
    for config_name, experiment_config in tuning_configs:
        experiment_bibs = BIBS(experiment_config)
        experiment_playlist = experiment_bibs.generate(
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
        tuning_playlists[config_name] = experiment_playlist
        experiment_metrics = evaluator.evaluate(experiment_playlist, c_arc, c_trans)
        experiment_diagnostics = experiment_bibs.get_diagnostics()
        experiment_coverage = playlist_coverage_report(
            config_name,
            experiment_playlist,
        )
        experiment_distribution = transition_distribution_for_playlist(
            experiment_playlist
        )
        experiment_graph_frame = pd.DataFrame(
            bibs_graph_transition_rows(experiment_playlist)
        )
        experiment_graph_share = (
            float(experiment_graph_frame["selected_rank_in_graph"].notna().mean())
            if len(experiment_graph_frame)
            else np.nan
        )
        experiment_anchor_count = experiment_diagnostics[
            "anchor_selection_trace_count"
        ]
        experiment_bottleneck_count = experiment_diagnostics[
            "selected_anchor_from_bottleneck_count"
        ]
        experiment_bottleneck_share = (
            experiment_bottleneck_count / experiment_anchor_count
            if experiment_anchor_count
            else np.nan
        )
        experiment_anchor_arc = experiment_diagnostics[
            "average_anchor_arc_component"
        ]
        experiment_anchor_transition = experiment_diagnostics[
            "average_anchor_transition_component"
        ]
        tuning_rows.append(
            {
                "config_name": config_name,
                "beam_width": experiment_config.beam_width,
                "base_case_max_candidates": (
                    experiment_config.base_case_max_candidates
                ),
                "anchor_arc_weight": experiment_config.anchor_arc_weight,
                "anchor_transition_weight": (
                    experiment_config.anchor_transition_weight
                ),
                "anchor_balance_weight": experiment_config.anchor_balance_weight,
                "bottleneck_weight": experiment_config.bottleneck_weight,
                "anchor_bottleneck_weight": (
                    experiment_config.anchor_bottleneck_weight
                ),
                "arc_rmse": experiment_metrics["arc_rmse"],
                "total_transition_cost": experiment_metrics[
                    "total_transition_cost"
                ],
                "average_transition_cost": experiment_metrics[
                    "average_transition_cost"
                ],
                "median_transition": experiment_distribution[
                    "median_transition"
                ],
                "p90_transition": experiment_distribution["p90_transition"],
                "p95_transition": experiment_distribution["p95_transition"],
                "max_transition": experiment_distribution["max_transition"],
                "global_coherence": experiment_metrics["global_coherence"],
                "exact_once_valid": experiment_coverage["exact_once_valid"],
                "structure_valid": experiment_structure_valid(
                    experiment_diagnostics,
                    experiment_config,
                ),
                "anchor_selection_trace_count": experiment_anchor_count,
                "selected_anchor_from_bottleneck_count": (
                    experiment_bottleneck_count
                ),
                "bottleneck_anchor_share": experiment_bottleneck_share,
                "average_anchor_arc_component": experiment_anchor_arc,
                "average_anchor_transition_component": (
                    experiment_anchor_transition
                ),
                "average_anchor_bottleneck_component": experiment_diagnostics[
                    "average_anchor_bottleneck_component"
                ],
                "average_anchor_balance_component": experiment_diagnostics[
                    "average_anchor_balance_component"
                ],
                "transition_to_arc_ratio": (
                    experiment_anchor_transition / experiment_anchor_arc
                    if experiment_anchor_arc
                    else np.nan
                ),
                "bibs_transitions_present_in_graph_share": (
                    experiment_graph_share
                ),
                "average_bibs_selected_rank_in_full_matrix": float(
                    experiment_graph_frame[
                        "selected_rank_in_full_matrix"
                    ].mean()
                ),
            }
        )

    tuning_frame = pd.DataFrame(tuning_rows)
    baseline_experiment = tuning_frame[
        tuning_frame["config_name"] == "baseline_current"
    ].iloc[0]
    valid_tuning_frame = tuning_frame[
        tuning_frame["exact_once_valid"] & tuning_frame["structure_valid"]
    ].copy()
    best_by_total_transition_cost = valid_tuning_frame.loc[
        valid_tuning_frame["total_transition_cost"].idxmin(),
        "config_name",
    ]
    best_by_average_transition_cost = valid_tuning_frame.loc[
        valid_tuning_frame["average_transition_cost"].idxmin(),
        "config_name",
    ]
    best_by_p95_transition = valid_tuning_frame.loc[
        valid_tuning_frame["p95_transition"].idxmin(),
        "config_name",
    ]
    best_by_global_coherence = valid_tuning_frame.loc[
        valid_tuning_frame["global_coherence"].idxmax(),
        "config_name",
    ]
    best_by_arc_rmse = valid_tuning_frame.loc[
        valid_tuning_frame["arc_rmse"].idxmin(),
        "config_name",
    ]
    balanced_scored_frame = valid_tuning_frame.assign(
        balanced_total_improved=(
            valid_tuning_frame["total_transition_cost"]
            < baseline_experiment["total_transition_cost"]
        ).astype(int),
        balanced_p95_improved=(
            valid_tuning_frame["p95_transition"]
            < baseline_experiment["p95_transition"]
        ).astype(int),
        balanced_arc_ok=(
            valid_tuning_frame["arc_rmse"]
            <= baseline_experiment["arc_rmse"] + 0.02
        ).astype(int),
        balanced_coherence_improved=(
            valid_tuning_frame["global_coherence"]
            > baseline_experiment["global_coherence"]
        ).astype(int),
    )
    balanced_scored_frame = balanced_scored_frame.assign(
        balanced_score=(
            4 * balanced_scored_frame["balanced_total_improved"]
            + 3 * balanced_scored_frame["balanced_p95_improved"]
            + 2 * balanced_scored_frame["balanced_arc_ok"]
            + balanced_scored_frame["balanced_coherence_improved"]
        )
    )
    best_balanced_row = balanced_scored_frame.sort_values(
        by=[
            "balanced_score",
            "balanced_total_improved",
            "balanced_p95_improved",
            "balanced_arc_ok",
            "balanced_coherence_improved",
            "total_transition_cost",
            "p95_transition",
        ],
        ascending=[False, False, False, False, False, True, True],
        kind="stable",
    ).iloc[0]
    best_balanced_candidate = best_balanced_row["config_name"]
    best_experiment_config = best_balanced_candidate
    improved_vs_baseline_bibs = bool(
        best_balanced_row["total_transition_cost"]
        < baseline_experiment["total_transition_cost"]
        or best_balanced_row["p95_transition"]
        < baseline_experiment["p95_transition"]
        or best_balanced_row["global_coherence"]
        > baseline_experiment["global_coherence"]
    )
    improved_vs_greedy = bool(
        best_balanced_row["total_transition_cost"] < greedy_total_transition
        and best_balanced_row["global_coherence"] > greedy_global_coherence
    )
    if improved_vs_greedy:
        recommended_next_step = (
            "Promote best config to default after repeated validation."
        )
    elif improved_vs_baseline_bibs:
        recommended_next_step = (
            "Adopt best config as candidate, then run targeted BIBS repair/tuning."
        )
    else:
        recommended_next_step = (
            "Inspect candidate orchestration or add methodology-aligned local repair."
        )

    wider_beam_rows = tuning_frame[
        tuning_frame["config_name"].isin(["wider_beam_8", "wider_beam_12"])
    ]
    wider_beam_improves = bool(
        (wider_beam_rows["total_transition_cost"]
        < baseline_experiment["total_transition_cost"]).any()
        or (wider_beam_rows["p95_transition"]
        < baseline_experiment["p95_transition"]).any()
        or (wider_beam_rows["global_coherence"]
        > baseline_experiment["global_coherence"]).any()
    )
    anchor_more_balance_row = tuning_frame[
        tuning_frame["config_name"] == "anchor_more_balance"
    ].iloc[0]
    anchor_balance_improves_tail = bool(
        anchor_more_balance_row["p90_transition"]
        < baseline_experiment["p90_transition"]
        or anchor_more_balance_row["p95_transition"]
        < baseline_experiment["p95_transition"]
    )
    bottleneck_boost_row = tuning_frame[
        tuning_frame["config_name"] == "bottleneck_boost"
    ].iloc[0]
    bottleneck_boost_visible = bool(
        bottleneck_boost_row["bottleneck_anchor_share"]
        > baseline_experiment["bottleneck_anchor_share"]
    )
    transition_reduced_row = tuning_frame[
        tuning_frame["config_name"] == "transition_reduced_anchor"
    ].iloc[0]
    transition_reduced_improves_both = bool(
        transition_reduced_row["arc_rmse"] < baseline_experiment["arc_rmse"]
        and transition_reduced_row["total_transition_cost"]
        < baseline_experiment["total_transition_cost"]
    )
    any_tuning_improves = bool(
        (tuning_frame["total_transition_cost"]
        < baseline_experiment["total_transition_cost"]).any()
        or (tuning_frame["p95_transition"]
        < baseline_experiment["p95_transition"]).any()
        or (tuning_frame["global_coherence"]
        > baseline_experiment["global_coherence"]).any()
    )

    def simulate_local_window_repair(
        playlist,
        c_arc,
        c_trans,
        window_radius=2,
        max_windows=20,
        max_window_size=6,
        arc_guard=0.02,
    ):
        repaired_playlist = list(playlist)
        original_arc_rmse = evaluator.compute_arc_rmse(repaired_playlist, c_arc)
        transition_costs = playlist_transition_costs(repaired_playlist)
        worst_positions = np.argsort(-transition_costs, kind="stable")[
            :max_windows
        ].tolist()
        repair_trace: list[dict[str, object]] = []

        def local_costs(candidate_playlist: list[int], start: int, end: int):
            local_tracks = candidate_playlist[start : end + 1]
            local_transition_total = float(
                c_trans[candidate_playlist[start - 1], local_tracks[0]]
            )
            for source_track, target_track in zip(local_tracks[:-1], local_tracks[1:]):
                local_transition_total += float(c_trans[source_track, target_track])
            local_transition_total += float(
                c_trans[local_tracks[-1], candidate_playlist[end + 1]]
            )
            local_arc_total = float(
                sum(
                    c_arc[track, position]
                    for position, track in enumerate(
                        local_tracks,
                        start=start,
                    )
                )
            )
            return local_transition_total, local_arc_total

        for transition_position in worst_positions:
            start = max(1, int(transition_position) - window_radius)
            end = min(
                len(repaired_playlist) - 2,
                int(transition_position) + 1 + window_radius,
            )
            window_size = end - start + 1
            if window_size > max_window_size:
                repair_trace.append(
                    {
                        "window_start": start,
                        "window_end": end,
                        "window_size": window_size,
                        "original_local_transition": np.nan,
                        "repaired_local_transition": np.nan,
                        "local_transition_delta": np.nan,
                        "original_local_arc": np.nan,
                        "repaired_local_arc": np.nan,
                        "local_arc_delta": np.nan,
                        "accepted": False,
                        "reason": "window_size_exceeds_limit",
                    }
                )
                continue

            original_total_transition = evaluator.compute_total_transition_cost(
                repaired_playlist,
                c_trans,
            )
            original_local_transition, original_local_arc = local_costs(
                repaired_playlist,
                start,
                end,
            )
            window_tracks = repaired_playlist[start : end + 1]
            best_playlist = list(repaired_playlist)
            best_local_transition = original_local_transition
            best_local_arc = original_local_arc
            best_score = original_local_transition + 0.3 * original_local_arc

            for ordering in permutations(window_tracks):
                candidate_playlist = list(repaired_playlist)
                candidate_playlist[start : end + 1] = list(ordering)
                candidate_transition, candidate_arc = local_costs(
                    candidate_playlist,
                    start,
                    end,
                )
                candidate_score = candidate_transition + 0.3 * candidate_arc
                if candidate_score < best_score:
                    best_score = candidate_score
                    best_playlist = candidate_playlist
                    best_local_transition = candidate_transition
                    best_local_arc = candidate_arc

            repaired_total_transition = evaluator.compute_total_transition_cost(
                best_playlist,
                c_trans,
            )
            repaired_arc_rmse = evaluator.compute_arc_rmse(best_playlist, c_arc)
            exact_once_valid = playlist_coverage_report(
                "repair_candidate",
                best_playlist,
            )["exact_once_valid"]
            transition_improves = repaired_total_transition < original_total_transition
            arc_guard_ok = repaired_arc_rmse <= original_arc_rmse + arc_guard
            accepted = bool(transition_improves and arc_guard_ok and exact_once_valid)
            if accepted:
                repaired_playlist = best_playlist
                reason = "accepted"
            elif not transition_improves:
                reason = "no_total_transition_improvement"
            elif not arc_guard_ok:
                reason = "arc_guard_failed"
            else:
                reason = "exact_once_failed"

            repair_trace.append(
                {
                    "window_start": start,
                    "window_end": end,
                    "window_size": window_size,
                    "original_local_transition": original_local_transition,
                    "repaired_local_transition": best_local_transition,
                    "local_transition_delta": (
                        best_local_transition - original_local_transition
                    ),
                    "original_local_arc": original_local_arc,
                    "repaired_local_arc": best_local_arc,
                    "local_arc_delta": best_local_arc - original_local_arc,
                    "accepted": accepted,
                    "reason": reason,
                }
            )

        return repaired_playlist, repair_trace

    def playlist_quality_row(
        variant_name: str,
        playlist: list[int],
        own_before_total: float | None = None,
    ) -> dict[str, object]:
        metrics = evaluator.evaluate(playlist, c_arc, c_trans)
        distribution = transition_distribution_for_playlist(playlist)
        own_improvement = (
            own_before_total - metrics["total_transition_cost"]
            if own_before_total is not None
            else np.nan
        )
        return {
            "playlist_variant": variant_name,
            "arc_rmse": metrics["arc_rmse"],
            "total_transition_cost": metrics["total_transition_cost"],
            "average_transition_cost": metrics["average_transition_cost"],
            "median_transition": distribution["median_transition"],
            "p90_transition": distribution["p90_transition"],
            "p95_transition": distribution["p95_transition"],
            "max_transition": distribution["max_transition"],
            "global_coherence": metrics["global_coherence"],
            "exact_once_valid": playlist_coverage_report(
                variant_name,
                playlist,
            )["exact_once_valid"],
            "improvement_vs_own_before": own_improvement,
            "improvement_vs_greedy": (
                greedy_total_transition - metrics["total_transition_cost"]
            ),
        }

    bottleneck_boost_playlist = tuning_playlists["bottleneck_boost"]
    repaired_bibs_playlist, baseline_repair_trace = simulate_local_window_repair(
        bibs_playlist,
        c_arc,
        c_trans,
        window_radius=2,
        max_windows=20,
        max_window_size=6,
        arc_guard=0.02,
    )
    repaired_bottleneck_playlist, bottleneck_repair_trace = (
        simulate_local_window_repair(
            bottleneck_boost_playlist,
            c_arc,
            c_trans,
            window_radius=2,
            max_windows=20,
            max_window_size=6,
            arc_guard=0.02,
        )
    )
    baseline_bibs_metrics = evaluator.evaluate(bibs_playlist, c_arc, c_trans)
    repaired_bibs_metrics = evaluator.evaluate(repaired_bibs_playlist, c_arc, c_trans)
    bottleneck_before_metrics = evaluator.evaluate(
        bottleneck_boost_playlist,
        c_arc,
        c_trans,
    )
    repaired_bottleneck_metrics = evaluator.evaluate(
        repaired_bottleneck_playlist,
        c_arc,
        c_trans,
    )
    local_repair_summary_rows = [
        playlist_quality_row("bibs_baseline_before", bibs_playlist),
        playlist_quality_row(
            "bibs_baseline_after_repair",
            repaired_bibs_playlist,
            baseline_bibs_metrics["total_transition_cost"],
        ),
        playlist_quality_row("bottleneck_boost_before", bottleneck_boost_playlist),
        playlist_quality_row(
            "bottleneck_boost_after_repair",
            repaired_bottleneck_playlist,
            bottleneck_before_metrics["total_transition_cost"],
        ),
        playlist_quality_row("greedy_baseline", greedy_playlist),
    ]

    def repair_trace_summary_row(
        variant_name: str,
        trace: list[dict[str, object]],
        before_metrics: dict[str, float],
        after_metrics: dict[str, float],
    ) -> dict[str, object]:
        trace_frame = pd.DataFrame(trace)
        accepted = trace_frame[trace_frame["accepted"] == True]
        total_transition_delta = (
            after_metrics["total_transition_cost"]
            - before_metrics["total_transition_cost"]
        )
        return {
            "variant_name": variant_name,
            "attempted_windows": len(trace_frame),
            "accepted_repairs": len(accepted),
            "rejected_repairs": len(trace_frame) - len(accepted),
            "total_transition_delta": total_transition_delta,
            "average_transition_delta_per_accepted_repair": (
                total_transition_delta / len(accepted)
                if len(accepted)
                else np.nan
            ),
            "max_single_window_improvement": (
                float(accepted["local_transition_delta"].min())
                if len(accepted)
                else np.nan
            ),
            "arc_rmse_before": before_metrics["arc_rmse"],
            "arc_rmse_after": after_metrics["arc_rmse"],
            "arc_rmse_delta": after_metrics["arc_rmse"] - before_metrics["arc_rmse"],
        }

    repair_trace_summary_rows = [
        repair_trace_summary_row(
            "bibs_baseline",
            baseline_repair_trace,
            baseline_bibs_metrics,
            repaired_bibs_metrics,
        ),
        repair_trace_summary_row(
            "bottleneck_boost",
            bottleneck_repair_trace,
            bottleneck_before_metrics,
            repaired_bottleneck_metrics,
        ),
    ]
    accepted_repair_rows: list[dict[str, object]] = []
    for variant_name, trace in [
        ("bibs_baseline", baseline_repair_trace),
        ("bottleneck_boost", bottleneck_repair_trace),
    ]:
        for row in trace:
            if row["accepted"]:
                accepted_repair_rows.append({"variant_name": variant_name, **row})
    accepted_repair_frame = pd.DataFrame(accepted_repair_rows)
    bottleneck_repair_improvement = (
        bottleneck_before_metrics["total_transition_cost"]
        - repaired_bottleneck_metrics["total_transition_cost"]
    )
    local_repair_helped = bool(bottleneck_repair_improvement > 0)
    local_repair_beats_or_nears_greedy = bool(
        repaired_bottleneck_metrics["total_transition_cost"]
        <= greedy_total_transition * 1.1
    )
    best_repaired_variant = (
        "bottleneck_boost_after_repair"
        if repaired_bottleneck_metrics["total_transition_cost"]
        <= repaired_bibs_metrics["total_transition_cost"]
        else "bibs_baseline_after_repair"
    )
    best_unrepaired_config = best_experiment_config
    if local_repair_helped:
        recommended_next_step = (
            "Implement methodology-aligned local repair inside BIBS after validation."
        )
    else:
        recommended_next_step = (
            "Return to anchor selection scoring and interval splitting diagnostics."
        )

    print_section("SECTION: TASK STATUS")
    print("[V] 1. Config")
    print("[V] 2. BIBS recursion")
    print("[V] 3. Beam expansion")
    print("[V] 4. Anchor selection")
    print("[V] 5. C_trans לפי המסמך")
    print("[V] 6. Camelot בתוך ההחלטות")
    print("[V] 7. Base case")
    print("[V] 8. Coverage checks")
    print("[V] 9. Diagnostics")
    print("[V] 10. הרצות בדיקה")
    print("[ ] 11. Post-validation tuning")

    print_section("SECTION: RUN CONFIGURATION")
    print(f"POOL_SIZE: {POOL_SIZE}")
    print(f"GENRE: {GENRE}")
    print(f"RANDOM_SEED: {RANDOM_SEED}")

    print_section("SECTION: STRUCTURAL HEALTH CHECK")
    print(
        "number_of_recursion_structure_violations: "
        f"{len(structure_violations)}"
    )
    print(f"number_of_beam_structure_violations: {len(beam_violations)}")
    print(f"number_of_anchor_structure_violations: {len(anchor_violations)}")
    print(f"number_of_c_trans_structure_violations: {len(c_trans_violations)}")
    print(f"number_of_camelot_structure_violations: {len(camelot_violations)}")
    print(f"number_of_base_case_structure_violations: {len(base_case_violations)}")
    print(f"number_of_coverage_violations: {len(coverage_violations)}")
    print(f"structural_checks_passed: {structural_checks_passed}")
    print(f"coverage_checks_passed: {coverage_checks_passed}")

    print_section("SECTION: GREEDY BASELINE AUDIT")
    print(f"greedy_playlist_length: {greedy_coverage_report['actual_length']}")
    print(f"greedy_unique_tracks: {greedy_coverage_report['unique_tracks']}")
    print(f"greedy_duplicate_count: {greedy_coverage_report['duplicate_count']}")
    print(f"greedy_missing_count: {greedy_coverage_report['missing_count']}")
    print(f"greedy_none_count: {greedy_coverage_report['none_count']}")
    print(f"greedy_out_of_range_count: {greedy_coverage_report['out_of_range_count']}")
    print(
        "greedy_start_position_correct: "
        f"{greedy_coverage_report['start_position_correct']}"
    )
    print(
        "greedy_end_position_correct: "
        f"{greedy_coverage_report['end_position_correct']}"
    )
    print(f"greedy_exact_once_valid: {greedy_coverage_report['exact_once_valid']}")
    print(f"greedy_uses_same_c_trans_assumption: {greedy_uses_same_c_trans_assumption}")
    print(f"greedy_uses_same_c_arc_assumption: {greedy_uses_same_c_arc_assumption}")
    print(f"greedy_compared_with_same_evaluator: {greedy_compared_with_same_evaluator}")

    print_section("SECTION: GREEDY LOCAL DECISION AUDIT")
    print(f"greedy_audit_positions: {greedy_audit_positions}")
    print(
        "average_selected_transition_rank: "
        f"{greedy_audit_frame['selected_transition_rank'].mean():.6f}"
    )
    print(
        "median_selected_transition_rank: "
        f"{greedy_audit_frame['selected_transition_rank'].median():.6f}"
    )
    print(
        "max_selected_transition_rank: "
        f"{greedy_audit_frame['selected_transition_rank'].max():.6f}"
    )
    print(
        "average_transition_gap_from_best: "
        f"{greedy_audit_frame['selected_transition_gap_from_best'].mean():.6f}"
    )
    print(
        "max_transition_gap_from_best: "
        f"{greedy_audit_frame['selected_transition_gap_from_best'].max():.6f}"
    )
    print(f"count_transition_rank_1_choices: {greedy_transition_rank_1_count}")
    print(
        "share_transition_rank_1_choices: "
        f"{greedy_transition_rank_1_count / greedy_audit_positions:.6f}"
    )
    print(f"count_transition_rank_above_10: {greedy_transition_rank_above_10_count}")
    print(
        "share_transition_rank_above_10: "
        f"{greedy_transition_rank_above_10_count / greedy_audit_positions:.6f}"
    )
    print(
        "average_selected_combined_rank: "
        f"{greedy_audit_frame['selected_combined_rank'].mean():.6f}"
    )
    print(
        "median_selected_combined_rank: "
        f"{greedy_audit_frame['selected_combined_rank'].median():.6f}"
    )
    print(
        "max_selected_combined_rank: "
        f"{greedy_audit_frame['selected_combined_rank'].max():.6f}"
    )
    print(
        "average_combined_gap_from_best: "
        f"{greedy_audit_frame['selected_combined_gap_from_best'].mean():.6f}"
    )
    print(
        "max_combined_gap_from_best: "
        f"{greedy_audit_frame['selected_combined_gap_from_best'].max():.6f}"
    )

    print_section("SECTION: WORST GREEDY LOCAL DECISIONS")
    worst_greedy_columns = [
        "position",
        "source_track",
        "selected_target_track",
        "selected_transition_cost",
        "best_possible_transition_cost",
        "selected_transition_rank",
        "selected_transition_gap_from_best",
        "selected_combined_cost",
        "best_possible_combined_cost",
        "selected_combined_rank",
        "selected_combined_gap_from_best",
        "source_camelot",
        "selected_target_camelot",
        "source_tempo",
        "selected_target_tempo",
    ]
    print_table(
        greedy_audit_frame.sort_values(
            by="selected_transition_gap_from_best",
            ascending=False,
            kind="stable",
        )[worst_greedy_columns].head(10)
    )

    print_section("SECTION: GREEDY FAIRNESS CHECK")
    print(
        "number_of_greedy_fairness_violations: "
        f"{len(greedy_fairness_issue_rows)}"
    )
    if greedy_fairness_issue_rows:
        print("first_5_greedy_fairness_violations:")
        print_table(pd.DataFrame(greedy_fairness_issue_rows).head(5))

    print_section("SECTION: METRIC COMPARISON")
    print_table(
        build_metric_comparison(
            greedy_metrics,
            bibs_metrics,
            anchor_alignment_cost,
        )
    )

    print_section("SECTION: TRANSITION DISTRIBUTION COMPARISON")
    print_table(pd.DataFrame(transition_distribution_rows()))

    print_section("SECTION: WORST BIBS TRANSITIONS FOR TUNING")
    print_table(pd.DataFrame(worst_bibs_transition_rows()))

    print_section("SECTION: ANCHOR TUNING SIGNALS")
    print(f"anchor_selection_trace_count: {anchor_trace_count}")
    print(
        "average_anchor_candidate_count: "
        f"{bibs_diagnostics['average_anchor_candidate_count']:.6f}"
    )
    print(f"average_anchor_arc_component: {anchor_arc_average:.6f}")
    print(f"average_anchor_transition_component: {anchor_transition_average:.6f}")
    print(
        "average_anchor_bottleneck_component: "
        f"{bibs_diagnostics['average_anchor_bottleneck_component']:.6f}"
    )
    print(
        "average_anchor_balance_component: "
        f"{bibs_diagnostics['average_anchor_balance_component']:.6f}"
    )
    print(
        "selected_anchor_from_graph_count: "
        f"{bibs_diagnostics['selected_anchor_from_graph_count']}"
    )
    print(
        "selected_anchor_from_candidate_set_count: "
        f"{bibs_diagnostics['selected_anchor_from_candidate_set_count']}"
    )
    print(f"selected_anchor_from_bottleneck_count: {selected_bottleneck_count}")
    print(
        "selected_anchor_from_bridge_count: "
        f"{bibs_diagnostics['selected_anchor_from_bridge_count']}"
    )
    print(f"transition_to_arc_ratio: {transition_to_arc_ratio:.6f}")

    print_section("SECTION: BOTTLENECK TUNING SIGNALS")
    print(f"bottleneck_track_count: {bottleneck_track_count}")
    print(f"anchor_selection_trace_count: {anchor_trace_count}")
    print(f"selected_anchor_from_bottleneck_count: {selected_bottleneck_count}")
    print(f"bottleneck_anchor_share: {bottleneck_anchor_share:.6f}")
    print(
        "average_anchor_bottleneck_component: "
        f"{bibs_diagnostics['average_anchor_bottleneck_component']:.6f}"
    )
    print(f"recommendation: {bottleneck_recommendation}")

    print_section("SECTION: TRANSITION GRAPH AUDIT")
    print(f"graph_node_count: {graph_node_count}")
    print(f"expected_node_count: {expected_node_count}")
    print(f"graph_total_outgoing_edges: {sum(outgoing_degrees)}")
    print(f"graph_average_out_degree: {float(np.mean(outgoing_degrees)):.6f}")
    print(f"graph_min_out_degree: {min(outgoing_degrees)}")
    print(f"graph_max_out_degree: {max(outgoing_degrees)}")
    print(f"graph_total_incoming_edges: {sum(incoming_degrees)}")
    print(f"graph_average_in_degree: {float(np.mean(incoming_degrees)):.6f}")
    print(f"graph_min_in_degree: {min(incoming_degrees)}")
    print(f"graph_max_in_degree: {max(incoming_degrees)}")
    print(f"isolated_outgoing_node_count: {sum(degree == 0 for degree in outgoing_degrees)}")
    print(f"isolated_incoming_node_count: {sum(degree == 0 for degree in incoming_degrees)}")
    print(f"graph_edge_cost_mean: {float(np.mean(graph_edge_costs)):.6f}")
    print(f"graph_edge_cost_median: {float(np.median(graph_edge_costs)):.6f}")
    print(f"graph_edge_cost_p90: {float(np.percentile(graph_edge_costs, 90)):.6f}")
    print(f"graph_edge_cost_p95: {float(np.percentile(graph_edge_costs, 95)):.6f}")
    print(f"graph_edge_cost_max: {float(np.max(graph_edge_costs)):.6f}")

    print_section("SECTION: GRAPH RECALL AGAINST FULL C_TRANS")
    print(f"graph_recall_at_5: {graph_recall['graph_recall_at_5']:.6f}")
    print(f"graph_recall_at_10: {graph_recall['graph_recall_at_10']:.6f}")
    print(f"graph_recall_at_20: {graph_recall['graph_recall_at_20']:.6f}")
    print(
        "average_best_full_neighbor_cost: "
        f"{graph_recall['average_best_full_neighbor_cost']:.6f}"
    )
    print(
        "average_best_graph_neighbor_cost: "
        f"{graph_recall['average_best_graph_neighbor_cost']:.6f}"
    )
    print(
        "average_graph_best_gap_from_full_best: "
        f"{graph_recall['average_graph_best_gap_from_full_best']:.6f}"
    )
    print(
        "max_graph_best_gap_from_full_best: "
        f"{graph_recall['max_graph_best_gap_from_full_best']:.6f}"
    )

    print_section("SECTION: GRAPH COVERAGE OF BIBS TRANSITIONS")
    print(f"bibs_transition_count: {len(bibs_graph_frame)}")
    print(
        "bibs_transitions_present_in_graph_count: "
        f"{bibs_transitions_present_in_graph_count}"
    )
    print(
        "bibs_transitions_present_in_graph_share: "
        f"{bibs_transitions_present_in_graph_share:.6f}"
    )
    print(
        "average_bibs_selected_rank_in_graph: "
        f"{bibs_graph_frame['selected_rank_in_graph'].mean():.6f}"
    )
    print(
        "average_bibs_selected_rank_in_full_matrix: "
        f"{bibs_graph_frame['selected_rank_in_full_matrix'].mean():.6f}"
    )
    print(
        "average_bibs_gap_from_best_graph_neighbor: "
        f"{bibs_graph_frame['gap_from_best_graph_neighbor'].mean():.6f}"
    )
    print(
        "average_bibs_gap_from_best_full_neighbor: "
        f"{bibs_graph_frame['gap_from_best_full_neighbor'].mean():.6f}"
    )
    print(
        "worst_bibs_gap_from_best_graph_neighbor: "
        f"{bibs_graph_frame['gap_from_best_graph_neighbor'].max():.6f}"
    )
    print(
        "worst_bibs_gap_from_best_full_neighbor: "
        f"{bibs_graph_frame['gap_from_best_full_neighbor'].max():.6f}"
    )

    print_section("SECTION: WORST GRAPH-LIMITED BIBS TRANSITIONS")
    worst_graph_columns = [
        "position",
        "source_track",
        "selected_target_track",
        "selected_transition_cost",
        "selected_rank_in_graph",
        "selected_rank_in_full_matrix",
        "best_graph_target_track",
        "best_graph_neighbor_cost",
        "best_full_target_track",
        "best_full_neighbor_cost",
        "gap_from_best_graph_neighbor",
        "gap_from_best_full_neighbor",
        "source_camelot",
        "selected_target_camelot",
        "best_graph_target_camelot",
        "best_full_target_camelot",
        "source_tempo",
        "selected_target_tempo",
        "best_graph_target_tempo",
        "best_full_target_tempo",
    ]
    worst_graph_frame = bibs_graph_frame.assign(
        graph_limit_sort_key=bibs_graph_frame[
            ["gap_from_best_graph_neighbor", "gap_from_best_full_neighbor"]
        ].max(axis=1)
    ).sort_values(
        by="graph_limit_sort_key",
        ascending=False,
        kind="stable",
    )
    print_table(worst_graph_frame[worst_graph_columns].head(10))

    print_section("SECTION: GRAPH STRUCTURE CHECK")
    print(
        "number_of_graph_structure_violations: "
        f"{len(graph_structure_issue_rows)}"
    )
    if graph_structure_issue_rows:
        print("first_5_graph_structure_violations:")
        print_table(pd.DataFrame(graph_structure_issue_rows).head(5))

    print_section("SECTION: GREEDY AND GRAPH AUDIT INTERPRETATION")
    print(
        "Greedy appears to be a valid fair baseline."
        if not greedy_fairness_issue_rows
        else "Fix Greedy baseline before tuning BIBS."
    )
    print(
        "Greedy is a strong local transition baseline."
        if greedy_transition_rank_1_count / greedy_audit_positions >= 0.8
        else "Greedy may not be purely transition-optimal; inspect scoring."
    )
    print(
        "Transition graph may be limiting BIBS candidate exposure."
        if graph_recall["graph_recall_at_10"] < 0.5
        else "Transition graph captures many good local neighbors."
    )
    print(
        "BIBS may be using non-graph bridge/base candidates often; inspect CandidateOrchestrator."
        if bibs_transitions_present_in_graph_share < 0.8
        else "BIBS selected transitions are mostly represented in the graph."
    )
    if (
        bibs_graph_frame["gap_from_best_graph_neighbor"].max() > 0.5
        or bibs_graph_frame["gap_from_best_full_neighbor"].max() > 0.5
    ):
        print(
            "Quality issue may come from anchor/interval decisions rather than missing transition edges."
        )

    print_section("SECTION: BIBS TUNING EXPERIMENTS")
    tuning_columns = [
        "config_name",
        "beam_width",
        "base_case_max_candidates",
        "anchor_arc_weight",
        "anchor_transition_weight",
        "anchor_balance_weight",
        "bottleneck_weight",
        "anchor_bottleneck_weight",
        "arc_rmse",
        "total_transition_cost",
        "average_transition_cost",
        "median_transition",
        "p90_transition",
        "p95_transition",
        "max_transition",
        "global_coherence",
        "exact_once_valid",
        "structure_valid",
        "bottleneck_anchor_share",
        "transition_to_arc_ratio",
        "bibs_transitions_present_in_graph_share",
        "average_bibs_selected_rank_in_full_matrix",
    ]
    print_table(tuning_frame[tuning_columns])

    print_section("SECTION: BEST TUNING CANDIDATES")
    print(f"best_by_total_transition_cost: {best_by_total_transition_cost}")
    print(f"best_by_average_transition_cost: {best_by_average_transition_cost}")
    print(f"best_by_p95_transition: {best_by_p95_transition}")
    print(f"best_by_global_coherence: {best_by_global_coherence}")
    print(f"best_by_arc_rmse: {best_by_arc_rmse}")
    print(f"best_balanced_candidate: {best_balanced_candidate}")

    print_section("SECTION: TUNING INTERPRETATION")
    if wider_beam_improves:
        print("Search width is limiting BIBS.")
    if anchor_balance_improves_tail:
        print("Anchor balance is helping reduce bad transitions.")
    if bottleneck_boost_visible:
        print("Bottleneck influence is becoming visible.")
    if transition_reduced_improves_both:
        print("Anchor transition dominance was too high.")
    if not any_tuning_improves:
        print("Issue may require candidate exposure changes or local repair, not only weights.")

    print_section("SECTION: LOCAL REPAIR POTENTIAL SUMMARY")
    repair_summary_columns = [
        "playlist_variant",
        "arc_rmse",
        "total_transition_cost",
        "average_transition_cost",
        "median_transition",
        "p90_transition",
        "p95_transition",
        "max_transition",
        "global_coherence",
        "exact_once_valid",
        "improvement_vs_own_before",
        "improvement_vs_greedy",
    ]
    print_table(pd.DataFrame(local_repair_summary_rows)[repair_summary_columns])

    print_section("SECTION: LOCAL REPAIR TRACE SUMMARY")
    print_table(pd.DataFrame(repair_trace_summary_rows))

    print_section("SECTION: TOP ACCEPTED LOCAL REPAIRS")
    accepted_repair_columns = [
        "variant_name",
        "window_start",
        "window_end",
        "window_size",
        "original_local_transition",
        "repaired_local_transition",
        "local_transition_delta",
        "original_local_arc",
        "repaired_local_arc",
        "local_arc_delta",
    ]
    if accepted_repair_frame.empty:
        print("No accepted repairs.")
    else:
        print_table(
            accepted_repair_frame.sort_values(
                by="local_transition_delta",
                ascending=True,
                kind="stable",
            )[accepted_repair_columns].head(10)
        )

    print_section("SECTION: LOCAL REPAIR INTERPRETATION")
    if bottleneck_repair_improvement > 10:
        print(
            "Local seam repair has strong potential; BIBS structure is usable but needs post-recursion repair."
        )
    if local_repair_beats_or_nears_greedy:
        print("BIBS is close enough for focused tuning.")
    if (
        repaired_bottleneck_metrics["total_transition_cost"]
        < bottleneck_before_metrics["total_transition_cost"]
        and repaired_bottleneck_metrics["arc_rmse"]
        > bottleneck_before_metrics["arc_rmse"] + 0.02
    ):
        print("Repair must include stronger arc guard.")
    if bottleneck_repair_improvement <= 2:
        print("Issue is likely earlier anchor/interval selection, not only local seams.")

    print_section("SECTION: FINAL DECISION BEFORE TUNING")
    print(f"best_unrepaired_config: {best_unrepaired_config}")
    print(f"best_repaired_variant: {best_repaired_variant}")
    print(f"local_repair_helped: {local_repair_helped}")
    print(f"local_repair_beats_or_nears_greedy: {local_repair_beats_or_nears_greedy}")
    print(f"recommended_next_step: {recommended_next_step}")

    print_section("SECTION: NEXT STEP")
    print("Next step: tune BIBS weights using the tuning dashboard.")
    print(
        "Expected next output: compare baseline metrics before/after changing "
        "BIBSConfig weights."
    )


def run_experiments() -> None:
    """Run the example multi-genre, multi-seed experiment grid."""
    project_root = Path(__file__).resolve().parent
    config = ExperimentConfig(
        csv_path=str(project_root.parent / "data" / "spotify_tracks.csv"),
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
