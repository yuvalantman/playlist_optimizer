"""Utilities for building a fixed track pool from Spotify track data."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrackPoolConfig:
    """Configuration for loading, filtering, and sampling a track pool."""

    csv_path: str | Path
    genre: str | None = None
    pool_size: int = 300
    random_seed: int = 42
    min_tempo: float | None = None
    max_tempo: float | None = None
    remove_duplicate_track_ids: bool = True
    remove_duplicate_songs: bool = True


class TrackPoolBuilder:
    """Build a reproducible pool from filtered Spotify track data."""

    def __init__(self, config: TrackPoolConfig) -> None:
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.pool_size <= 0:
            raise ValueError("pool_size must be greater than zero.")
        if self.config.min_tempo is not None and self.config.min_tempo <= 0:
            raise ValueError("min_tempo must be greater than zero.")
        if self.config.max_tempo is not None and self.config.max_tempo <= 0:
            raise ValueError("max_tempo must be greater than zero.")
        if (
            self.config.min_tempo is not None
            and self.config.max_tempo is not None
            and self.config.max_tempo < self.config.min_tempo
        ):
            raise ValueError("max_tempo must be greater than or equal to min_tempo.")

    def load(self) -> pd.DataFrame:
        """Load the configured CSV file."""
        csv_path = Path(self.config.csv_path)
        if not csv_path.is_file():
            raise FileNotFoundError(f"Track CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        required_columns = {
            "track_id",
            "track_genre",
            "tempo",
            "track_name",
            "artists",
        }
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV is missing required columns: {missing}")
        return df

    def build(self) -> pd.DataFrame:
        """Return a filtered, randomly sampled pool of exactly pool_size tracks."""
        df = self.load().copy()

        if self.config.genre is not None:
            genre = self.config.genre.casefold()
            df = df[df["track_genre"].astype(str).str.casefold() == genre]

        df["tempo"] = pd.to_numeric(df["tempo"], errors="coerce")
        valid_tempo = df["tempo"].notna() & np.isfinite(df["tempo"]) & (df["tempo"] > 0)
        if self.config.min_tempo is not None:
            valid_tempo &= df["tempo"] >= self.config.min_tempo
        if self.config.max_tempo is not None:
            valid_tempo &= df["tempo"] <= self.config.max_tempo
        df = df.loc[valid_tempo]

        if self.config.remove_duplicate_track_ids:
            df = df.drop_duplicates(subset="track_id")

        if self.config.remove_duplicate_songs:
            normalized_track_names = (
                df["track_name"].astype(str).str.strip().str.casefold()
            )
            normalized_artists = df["artists"].astype(str).str.strip().str.casefold()
            df = df.loc[
                ~pd.DataFrame(
                    {
                        "track_name": normalized_track_names,
                        "artists": normalized_artists,
                    },
                    index=df.index,
                ).duplicated(keep="first")
            ]

        if len(df) < self.config.pool_size:
            genre_text = f" for genre {self.config.genre!r}" if self.config.genre else ""
            raise ValueError(
                f"Cannot sample {self.config.pool_size} tracks{genre_text}; "
                f"only {len(df)} valid tracks are available."
            )

        return df.sample(
            n=self.config.pool_size,
            random_state=self.config.random_seed,
            replace=False,
        ).reset_index(drop=True)
