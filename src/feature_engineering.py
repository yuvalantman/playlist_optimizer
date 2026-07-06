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


@dataclass(frozen=True)
class StartEndSelectionConfig:
    """Configuration for deterministic start and end track selection."""

    min_ev_gap: float = 0.35
    preferred_ev_gap: float = 0.60
    start_quantile: float = 0.30
    end_quantile: float = 0.70
    transition_potential_k: int = 20
    transition_potential_weight: float = 0.25
    random_seed: int = 42


@dataclass(frozen=True)
class TempoEnvelopeConfig:
    """Margins used to derive a diagnostic tempo envelope from endpoints."""

    lower_margin: float = 0.25
    upper_margin: float = 0.25


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


def compute_ev_score(
    df: pd.DataFrame,
    config: EnergyValenceConfig,
) -> pd.DataFrame:
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


def select_start_end_tracks(
    df: pd.DataFrame,
    c_trans: np.ndarray | None = None,
    config: StartEndSelectionConfig = StartEndSelectionConfig(),
) -> tuple[int, int]:
    """Select deterministic low-EV start and high-EV end tracks."""
    if "EV_score" not in df.columns:
        raise ValueError("DataFrame must contain an EV_score column.")
    if not 0 <= config.start_quantile < config.end_quantile <= 1:
        raise ValueError("Start and end quantiles must be ordered within [0, 1].")
    if config.transition_potential_k <= 0:
        raise ValueError("transition_potential_k must be positive.")

    ev_scores = pd.to_numeric(df["EV_score"], errors="coerce").to_numpy(dtype=float)
    if len(ev_scores) < 2 or not np.isfinite(ev_scores).all():
        raise ValueError("At least two finite EV scores are required.")
    number_of_tracks = len(ev_scores)

    if c_trans is not None:
        if (
            not isinstance(c_trans, np.ndarray)
            or c_trans.shape != (number_of_tracks, number_of_tracks)
            or not np.isfinite(c_trans).all()
        ):
            raise ValueError("c_trans must be a finite square matrix matching df.")
        k = min(config.transition_potential_k, number_of_tracks - 1)
        costs_without_self = c_trans.copy()
        np.fill_diagonal(costs_without_self, np.inf)
        outgoing = np.partition(costs_without_self, kth=k - 1, axis=1)[:, :k].mean(
            axis=1
        )
        incoming = np.partition(costs_without_self, kth=k - 1, axis=0)[:k, :].mean(
            axis=0
        )
    else:
        outgoing = np.zeros(number_of_tracks)
        incoming = np.zeros(number_of_tracks)

    start_threshold = np.quantile(ev_scores, config.start_quantile)
    end_threshold = np.quantile(ev_scores, config.end_quantile)
    start_candidates = np.flatnonzero(ev_scores <= start_threshold)
    end_candidates = np.flatnonzero(ev_scores >= end_threshold)

    valid_pairs = [
        (int(start), int(end))
        for start in start_candidates
        for end in end_candidates
        if start != end and ev_scores[end] - ev_scores[start] >= config.min_ev_gap
    ]
    if not valid_pairs:
        raise ValueError(
            "Unable to select distinct start and end tracks meeting min_ev_gap."
        )

    def pair_score(pair: tuple[int, int]) -> tuple[float, int, int]:
        start, end = pair
        gap = ev_scores[end] - ev_scores[start]
        score = abs(gap - config.preferred_ev_gap)
        score += config.transition_potential_weight * (
            outgoing[start] + incoming[end]
        )
        return float(score), start, end

    return min(valid_pairs, key=pair_score)


def create_target_arc_from_tracks(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
) -> np.ndarray:
    """Create a linear target arc from selected endpoint EV scores."""
    if "EV_score" not in df.columns:
        raise ValueError("DataFrame must contain an EV_score column.")
    if (
        not 0 <= start_index < len(df)
        or not 0 <= end_index < len(df)
        or start_index == end_index
    ):
        raise ValueError("Start and end indices must be valid and distinct.")
    return create_linear_target_arc(
        len(df),
        float(df.iloc[start_index]["EV_score"]),
        float(df.iloc[end_index]["EV_score"]),
    )


def create_tempo_envelope_from_tracks(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
    config: TempoEnvelopeConfig = TempoEnvelopeConfig(),
) -> tuple[float, float]:
    """Derive a diagnostic tempo range from selected endpoint tempos."""
    if "tempo" not in df.columns:
        raise ValueError("DataFrame must contain a tempo column.")
    if (
        not 0 <= start_index < len(df)
        or not 0 <= end_index < len(df)
        or start_index == end_index
    ):
        raise ValueError("Start and end indices must be valid and distinct.")
    if not 0 <= config.lower_margin < 1 or config.upper_margin < 0:
        raise ValueError(
            "lower_margin must be in [0, 1) and upper_margin must be non-negative."
        )

    tempo_start = float(df.iloc[start_index]["tempo"])
    tempo_end = float(df.iloc[end_index]["tempo"])
    if not np.isfinite([tempo_start, tempo_end]).all() or min(
        tempo_start,
        tempo_end,
    ) <= 0:
        raise ValueError("Start and end tempos must be finite and positive.")

    base_min = min(tempo_start, tempo_end)
    base_max = max(tempo_start, tempo_end)
    return (
        base_min * (1.0 - config.lower_margin),
        base_max * (1.0 + config.upper_margin),
    )


def create_target_tempo_arc_from_tracks(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
) -> np.ndarray:
    """Create a linear diagnostic tempo arc between selected endpoints."""
    if "tempo" not in df.columns:
        raise ValueError("DataFrame must contain a tempo column.")
    if (
        not 0 <= start_index < len(df)
        or not 0 <= end_index < len(df)
        or start_index == end_index
    ):
        raise ValueError("Start and end indices must be valid and distinct.")

    tempo_start = float(df.iloc[start_index]["tempo"])
    tempo_end = float(df.iloc[end_index]["tempo"])
    if not np.isfinite([tempo_start, tempo_end]).all() or min(
        tempo_start,
        tempo_end,
    ) <= 0:
        raise ValueError("Start and end tempos must be finite and positive.")
    return np.linspace(tempo_start, tempo_end, num=len(df), dtype=float)


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
