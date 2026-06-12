"""Feature engineering and target-arc cost construction."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EnergyValenceConfig:
    """Weights used to combine normalized audio features into an EV score."""

    energy_weight: float = 1.0
    valence_weight: float = 1.0
    danceability_weight: float = 1.0
    loudness_weight: float = 1.0
    tempo_weight: float = 1.0


def normalize_column(column: pd.Series) -> pd.Series:
    """Min-max normalize a numeric Series to [0, 1]."""
    numeric = pd.to_numeric(column, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"Column {column.name!r} contains missing or non-numeric values.")

    minimum = numeric.min()
    value_range = numeric.max() - minimum
    if value_range == 0:
        return pd.Series(0.0, index=column.index, name=column.name)
    return (numeric - minimum) / value_range


def normalize_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return a copy of df with the selected columns min-max normalized."""
    missing_columns = set(columns).difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"DataFrame is missing columns to normalize: {missing}")

    normalized = df.copy()
    for column in columns:
        normalized[column] = normalize_column(normalized[column])
    return normalized


def compute_ev_score(df: pd.DataFrame,config: EnergyValenceConfig) -> pd.DataFrame:
    """Return a copy of df with a weighted normalized EV_score column."""
    feature_weights = {
        "energy": config.energy_weight,
        "valence": config.valence_weight,
        "danceability": config.danceability_weight,
        "loudness": config.loudness_weight,
        "tempo": config.tempo_weight,
    }
    if any(weight < 0 for weight in feature_weights.values()):
        raise ValueError("EV feature weights must be non-negative.")

    total_weight = sum(feature_weights.values())
    if total_weight == 0:
        raise ValueError("At least one EV feature weight must be greater than zero.")

    normalized = normalize_columns(df, list(feature_weights))
    scored = df.copy()
    scored["EV_score"] = sum(
        normalized[feature] * weight
        for feature, weight in feature_weights.items()
    ) / total_weight
    return scored


def create_linear_target_arc(
    length: int,
    start_value: float,
    end_value: float,
) -> np.ndarray:
    """Create a linearly spaced target EV arc."""
    if length <= 0:
        raise ValueError("length must be greater than zero.")
    return np.linspace(start_value, end_value, num=length, dtype=float)


def compute_arc_cost_matrix(
    df: pd.DataFrame,
    target_arc: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return absolute EV deviation costs for every track-position pair."""
    if "EV_score" not in df.columns:
        raise ValueError("DataFrame must contain an EV_score column.")

    ev_scores = pd.to_numeric(df["EV_score"], errors="coerce").to_numpy(dtype=float)
    target = np.asarray(target_arc, dtype=float)
    if target.ndim != 1 or len(target) == 0:
        raise ValueError("target_arc must be a non-empty one-dimensional sequence.")
    if not np.isfinite(ev_scores).all() or not np.isfinite(target).all():
        raise ValueError("EV scores and target arc values must be finite.")

    return np.abs(ev_scores[:, np.newaxis] - target[np.newaxis, :])
