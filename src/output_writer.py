"""Save generated playlists and metric comparisons to CSV files."""

from pathlib import Path

import pandas as pd


class OutputWriter:
    """Write single-run playlist outputs into the project output folders."""

    OFFICIAL_METRIC_DIRECTIONS = {
        "arc_rmse": "lower",
        "total_transition_cost": "lower",
        "average_transition_cost": "lower",
        "global_coherence": "higher",
    }

    PLAYLIST_COLUMNS = [
        "position",
        "track_index",
        "track_id",
        "track_name",
        "artists",
        "track_genre",
        "EV_score",
        "tempo",
        "camelot",
        "energy",
        "valence",
    ]
    OPTIONAL_PLAYLIST_COLUMNS = [
        "key",
        "mode",
        "time_signature",
        "loudness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "speechiness",
    ]

    def __init__(
        self,
        playlists_directory: str | Path = "outputs/playlists",
        results_directory: str | Path = "outputs/results",
    ) -> None:
        self.playlists_directory = Path(playlists_directory)
        self.results_directory = Path(results_directory)
        self.playlists_directory.mkdir(parents=True, exist_ok=True)
        self.results_directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_playlist(
        playlist: list[int],
        track_pool: pd.DataFrame,
    ) -> None:
        if len(playlist) != len(track_pool):
            raise ValueError("Playlist length must equal the track pool length.")
        if len(set(playlist)) != len(playlist):
            raise ValueError("Playlist must contain every track exactly once.")
        if any(
            not isinstance(track_index, int)
            or not 0 <= track_index < len(track_pool)
            for track_index in playlist
        ):
            raise ValueError("Playlist contains an invalid track index.")

    def save_playlist(
        self,
        playlist: list[int],
        track_pool: pd.DataFrame,
        output_path: str,
    ) -> None:
        """Save one ordered playlist with track metadata."""
        self._validate_playlist(playlist, track_pool)
        metadata_columns = self.PLAYLIST_COLUMNS[2:]
        missing_columns = set(metadata_columns).difference(track_pool.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Track pool is missing output columns: {missing}")

        output_columns = metadata_columns + [
            column
            for column in self.OPTIONAL_PLAYLIST_COLUMNS
            if column in track_pool.columns
        ]
        playlist_frame = track_pool.loc[playlist, output_columns].copy()
        playlist_frame.insert(0, "track_index", playlist)
        playlist_frame.insert(0, "position", range(len(playlist)))

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        playlist_frame.to_csv(path, index=False)

    def save_metric_comparison(
        self,
        greedy_metrics: dict,
        bibs_metrics: dict,
        output_path: str,
        anchor_alignment_cost: float | None = None,
    ) -> None:
        """Save direction-aware official metric comparisons."""
        rows: list[dict[str, object]] = []
        for metric, better_when in self.OFFICIAL_METRIC_DIRECTIONS.items():
            if metric not in greedy_metrics or metric not in bibs_metrics:
                continue
            greedy_value = float(greedy_metrics[metric])
            bibs_value = float(bibs_metrics[metric])
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
                    "greedy": float("nan"),
                    "bibs": float(anchor_alignment_cost),
                    "improvement": float("nan"),
                    "better_when": "lower",
                }
            )
        comparison = pd.DataFrame(rows)

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(path, index=False)

    def save_single_run_outputs(
        self,
        greedy_playlist: list[int],
        bibs_playlist: list[int],
        track_pool: pd.DataFrame,
        greedy_metrics: dict,
        bibs_metrics: dict,
        run_name: str = "single_run",
        anchor_alignment_cost: float | None = None,
    ) -> dict[str, Path]:
        """Save both playlists and their metric comparison."""
        greedy_path = self.playlists_directory / f"{run_name}_greedy_playlist.csv"
        bibs_path = self.playlists_directory / f"{run_name}_bibs_playlist.csv"
        comparison_path = (
            self.results_directory / f"{run_name}_metric_comparison.csv"
        )

        self.save_playlist(greedy_playlist, track_pool, str(greedy_path))
        self.save_playlist(bibs_playlist, track_pool, str(bibs_path))
        self.save_metric_comparison(
            greedy_metrics,
            bibs_metrics,
            str(comparison_path),
            anchor_alignment_cost=anchor_alignment_cost,
        )
        return {
            "greedy_playlist": greedy_path.resolve(),
            "bibs_playlist": bibs_path.resolve(),
            "metric_comparison": comparison_path.resolve(),
        }
