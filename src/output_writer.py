"""Save generated playlists and metric comparisons to CSV files."""

from pathlib import Path

import pandas as pd


class OutputWriter:
    """Write single-run playlist outputs into the project output folders."""

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

        playlist_frame = track_pool.loc[playlist, metadata_columns].copy()
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
    ) -> None:
        """Save matching Greedy and BIBS metrics with improvements."""
        if set(greedy_metrics) != set(bibs_metrics):
            raise ValueError("Greedy and BIBS metric dictionaries must match.")

        comparison = pd.DataFrame(
            {
                "metric": list(greedy_metrics),
                "greedy": list(greedy_metrics.values()),
                "bibs": [
                    bibs_metrics[metric]
                    for metric in greedy_metrics
                ],
            }
        )
        comparison["improvement"] = comparison["greedy"] - comparison["bibs"]

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
        )
        return {
            "greedy_playlist": greedy_path.resolve(),
            "bibs_playlist": bibs_path.resolve(),
            "metric_comparison": comparison_path.resolve(),
        }
