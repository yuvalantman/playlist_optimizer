"""Run controlled playlist sequencing experiments across genres and seeds."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.bibs import BIBS, BIBSConfig
from src.bottleneck_detector import BottleneckConfig, BottleneckDetector
from src.cost_functions import compute_transition_cost_matrix
from src.data_preparation import TrackPoolBuilder, TrackPoolConfig
from src.evaluator import EvaluationConfig, PlaylistEvaluator
from src.feature_engineering import (
    EnergyValenceConfig,
    compute_arc_cost_matrix,
    compute_ev_score,
    create_linear_target_arc,
)
from src.greedy_baseline import GreedyBaselineConfig, GreedyPlaylistBaseline
from src.transition_graph import TransitionGraphBuilder, TransitionGraphConfig


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a deterministic experiment grid."""

    csv_path: str
    genres: list[str]
    seeds: list[int]
    pool_size: int = 300
    target_arc_start: float = 0.2
    target_arc_end: float = 0.8
    output_results_path: str = "outputs/results/experiment_results.csv"
    output_summary_path: str = "outputs/results/experiment_summary.csv"


class ExperimentRunner:
    """Run and summarize Greedy versus BIBS experiments."""

    METRIC_NAMES = (
        "arc_rmse",
        "total_transition_cost",
        "average_transition_cost",
        "tail_transition_cost",
        "average_tail_transition_cost",
    )
    IMPROVEMENT_NAMES = (
        "arc_rmse_improvement",
        "average_transition_improvement",
        "tail_transition_improvement",
    )
    WIN_NAMES = (
        "bibs_wins_arc_rmse",
        "bibs_wins_average_transition",
        "bibs_wins_tail_transition",
        "bibs_wins_all_three",
    )
    RESULT_COLUMNS = (
        "genre",
        "seed",
        "pool_size",
        *(f"greedy_{metric}" for metric in METRIC_NAMES),
        *(f"bibs_{metric}" for metric in METRIC_NAMES),
        *IMPROVEMENT_NAMES,
        *WIN_NAMES,
    )

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        if self.config.pool_size <= 0:
            raise ValueError("pool_size must be positive.")
        if not self.config.genres:
            raise ValueError("genres must not be empty.")
        if not self.config.seeds:
            raise ValueError("seeds must not be empty.")

    def _run_single_experiment(self, genre: str, seed: int) -> dict:
        """Run one deterministic genre and seed combination."""
        track_pool = TrackPoolBuilder(
            TrackPoolConfig(
                csv_path=self.config.csv_path,
                genre=genre,
                pool_size=self.config.pool_size,
                random_seed=seed,
                min_tempo=40.0,
                max_tempo=220.0,
            )
        ).build()
        track_pool = compute_ev_score(track_pool, EnergyValenceConfig())
        target_arc = create_linear_target_arc(
            len(track_pool),
            self.config.target_arc_start,
            self.config.target_arc_end,
        )
        c_arc = compute_arc_cost_matrix(track_pool, target_arc)
        c_trans = compute_transition_cost_matrix(track_pool)
        bottleneck_results = BottleneckDetector(BottleneckConfig()).detect(c_arc)
        transition_graph = TransitionGraphBuilder(
            TransitionGraphConfig()
        ).build_top_m_graph(c_trans)

        greedy_playlist = GreedyPlaylistBaseline(
            GreedyBaselineConfig()
        ).generate(c_arc, c_trans, transition_graph=transition_graph)
        bibs_playlist = BIBS(BIBSConfig()).generate(
            c_arc,
            c_trans,
            bottleneck_results,
            transition_graph=transition_graph,
        )

        evaluator = PlaylistEvaluator(EvaluationConfig())
        greedy_metrics = evaluator.evaluate(greedy_playlist, c_arc, c_trans)
        bibs_metrics = evaluator.evaluate(bibs_playlist, c_arc, c_trans)

        result: dict[str, object] = {
            "genre": genre,
            "seed": seed,
            "pool_size": self.config.pool_size,
        }
        result.update(
            {
                f"greedy_{metric}": greedy_metrics[metric]
                for metric in self.METRIC_NAMES
            }
        )
        result.update(
            {
                f"bibs_{metric}": bibs_metrics[metric]
                for metric in self.METRIC_NAMES
            }
        )

        result["arc_rmse_improvement"] = (
            greedy_metrics["arc_rmse"] - bibs_metrics["arc_rmse"]
        )
        result["average_transition_improvement"] = (
            greedy_metrics["average_transition_cost"]
            - bibs_metrics["average_transition_cost"]
        )
        result["tail_transition_improvement"] = (
            greedy_metrics["average_tail_transition_cost"]
            - bibs_metrics["average_tail_transition_cost"]
        )
        result["bibs_wins_arc_rmse"] = (
            bibs_metrics["arc_rmse"] < greedy_metrics["arc_rmse"]
        )
        result["bibs_wins_average_transition"] = (
            bibs_metrics["average_transition_cost"]
            < greedy_metrics["average_transition_cost"]
        )
        result["bibs_wins_tail_transition"] = (
            bibs_metrics["average_tail_transition_cost"]
            < greedy_metrics["average_tail_transition_cost"]
        )
        result["bibs_wins_all_three"] = all(
            result[win_name]
            for win_name in (
                "bibs_wins_arc_rmse",
                "bibs_wins_average_transition",
                "bibs_wins_tail_transition",
            )
        )
        return result

    def _build_summary(self, raw_results: pd.DataFrame) -> pd.DataFrame:
        """Aggregate mean metrics, improvements, and win rates by genre."""
        if raw_results.empty:
            return pd.DataFrame(
                columns=[
                    "genre",
                    "number_of_runs",
                    "arc_rmse_win_rate",
                    "average_transition_win_rate",
                    "tail_transition_win_rate",
                    "all_three_win_rate",
                ]
            )

        mean_columns = [
            *(f"greedy_{metric}" for metric in self.METRIC_NAMES),
            *(f"bibs_{metric}" for metric in self.METRIC_NAMES),
            *self.IMPROVEMENT_NAMES,
        ]
        summary = (
            raw_results.groupby("genre", sort=True)[mean_columns]
            .mean()
            .add_prefix("mean_")
        )
        summary["number_of_runs"] = raw_results.groupby("genre").size()
        summary["arc_rmse_win_rate"] = raw_results.groupby("genre")[
            "bibs_wins_arc_rmse"
        ].mean()
        summary["average_transition_win_rate"] = raw_results.groupby("genre")[
            "bibs_wins_average_transition"
        ].mean()
        summary["tail_transition_win_rate"] = raw_results.groupby("genre")[
            "bibs_wins_tail_transition"
        ].mean()
        summary["all_three_win_rate"] = raw_results.groupby("genre")[
            "bibs_wins_all_three"
        ].mean()
        return summary.reset_index()

    def run_all(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run every genre-seed combination, save CSVs, and return results."""
        results_path = Path(self.config.output_results_path).resolve()
        summary_path = Path(self.config.output_summary_path).resolve()
        results_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        results: list[dict] = []
        raw_results = pd.DataFrame(columns=self.RESULT_COLUMNS)
        summary = self._build_summary(raw_results)
        raw_results.to_csv(results_path, index=False)
        summary.to_csv(summary_path, index=False)

        print(f"Experiment results will be saved to: {results_path}")
        print(f"Experiment summary will be saved to: {summary_path}")

        total_experiments = len(self.config.genres) * len(self.config.seeds)
        experiment_number = 0
        for genre in self.config.genres:
            for seed in self.config.seeds:
                experiment_number += 1
                print(
                    f"Running experiment {experiment_number}/{total_experiments}: "
                    f"genre={genre!r}, seed={seed}"
                )
                try:
                    results.append(self._run_single_experiment(genre, seed))
                except ValueError as error:
                    print(
                        f"Warning: skipping genre={genre!r}, seed={seed}: {error}"
                    )
                    continue

                raw_results = pd.DataFrame(results, columns=self.RESULT_COLUMNS)
                summary = self._build_summary(raw_results)
                raw_results.to_csv(results_path, index=False)
                summary.to_csv(summary_path, index=False)
                print(f"Saved {len(raw_results)} completed experiment(s).")

        return raw_results, summary
