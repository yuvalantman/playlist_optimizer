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
    """Configuration for deterministic start and end track selection.

    endpoint_mode: "quantile" (default) picks EV-extreme tracks then snaps the
        trajectory to them. "trajectory_fit" defines the trajectory from the pool's
        natural EV range first, then picks the tracks that best fit positions 0 and
        L-1 of that un-snapped trajectory.
    """

    min_ev_gap: float = 0.35
    preferred_ev_gap: float = 0.60
    start_quantile: float = 0.30
    end_quantile: float = 0.70
    transition_potential_k: int = 20
    transition_potential_weight: float = 0.25
    random_seed: int = 42
    endpoint_mode: str = "quantile"  # "quantile" | "trajectory_fit"


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


def _select_start_end_quantile(
    df: pd.DataFrame,
    c_trans: np.ndarray | None,
    config: "StartEndSelectionConfig",
) -> tuple[int, int]:
    """Original quantile-based start/end selection."""
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


def _select_trajectory_fit(
    df: pd.DataFrame,
    config: "StartEndSelectionConfig",
) -> tuple[int, int]:
    """Trajectory-first endpoint selection.

    Defines the target trajectory from the pool's Q05-Q95 EV range (no endpoint
    snapping), computes a preliminary arc cost matrix, then picks the tracks that
    best fit the first and last positions of that trajectory.  The caller still
    runs create_target_arc with endpoint-snapping afterward, so the final arc is
    self-consistent.
    """
    ev_scores = pd.to_numeric(df["EV_score"], errors="coerce").to_numpy(dtype=float)
    n = len(ev_scores)
    ev_low = float(np.quantile(ev_scores, 0.05))
    ev_high = float(np.quantile(ev_scores, 0.95))
    prelim_arc = np.linspace(ev_low, ev_high, n)
    arc_costs = np.abs(ev_scores[:, np.newaxis] - prelim_arc[np.newaxis, :])

    start_candidates = np.where(ev_scores <= np.median(ev_scores))[0]
    end_candidates = np.where(ev_scores > np.median(ev_scores))[0]

    start_scores = arc_costs[start_candidates, 0]
    end_scores = arc_costs[end_candidates, n - 1]
    start_order = start_candidates[np.argsort(start_scores)]
    end_order = end_candidates[np.argsort(end_scores)]

    for s in start_order:
        for e in end_order:
            if s != e and ev_scores[e] - ev_scores[s] >= config.min_ev_gap:
                return int(s), int(e)

    raise ValueError(
        "trajectory_fit: no valid start/end pair satisfies min_ev_gap. "
        "Try lowering min_ev_gap or switching to endpoint_mode='quantile'."
    )


def _select_ev_proximity(
    df: pd.DataFrame,
) -> tuple[int, int]:
    """Simplest endpoint selection: closest EV to the natural trajectory endpoints.

    Computes a preliminary linear arc from the pool's Q05-Q95 EV range, then picks:
      start = argmin |EV[track] - arc[0]|
      end   = argmin |EV[track] - arc[N-1]|  (with end ≠ start)
    No minimum EV gap enforced — just pure EV proximity. This is the fairest way to
    anchor the trajectory when you want the playlist to literally begin and end at the
    natural low/high energy of the pool.
    """
    ev_scores = pd.to_numeric(df["EV_score"], errors="coerce").to_numpy(dtype=float)
    n = len(ev_scores)
    ev_low = float(np.quantile(ev_scores, 0.05))
    ev_high = float(np.quantile(ev_scores, 0.95))
    start_ev = ev_low
    end_ev = ev_high

    start_dist = np.abs(ev_scores - start_ev)
    end_dist = np.abs(ev_scores - end_ev)

    start_order = np.argsort(start_dist, kind="stable")
    start_index = int(start_order[0])

    end_candidates = np.argsort(end_dist, kind="stable")
    end_index = int(next(i for i in end_candidates if i != start_index))
    return start_index, end_index


def select_start_end_tracks(
    df: pd.DataFrame,
    c_trans: np.ndarray | None = None,
    config: "StartEndSelectionConfig" = StartEndSelectionConfig(),
) -> tuple[int, int]:
    """Dispatch to the appropriate endpoint selection strategy."""
    if config.endpoint_mode == "trajectory_fit":
        return _select_trajectory_fit(df, config)
    if config.endpoint_mode == "ev_proximity":
        return _select_ev_proximity(df)
    return _select_start_end_quantile(df, c_trans, config)


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
