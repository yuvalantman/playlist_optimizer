"""Non-linear target Energy-Valence arcs with per-playlist normalization.

The original pipeline only supported a linear arc (``np.linspace(EV_start, EV_end)``).
This module generalizes the target arc to several musically meaningful shapes while
preserving the linear shape as a byte-identical default so the existing pipeline is
unaffected.

Design (locked with the project owner):
- A shape function ``s(t)`` is defined over normalized playlist progress ``t in [0, 1]``
  and returns values in ``[0, 1]``.
- The shape is mapped to the *playlist's own* EV range using robust quantiles
  (Q05-Q95 by default), because EV_score has no universal absolute scale across pools.
- The endpoints are then *snapped* toward the fixed start/end track EV scores using a
  short, locally-decaying correction, so ``arc[0] == EV(start)`` and
  ``arc[-1] == EV(end)`` exactly while the interior keeps the chosen shape.

The ``"linear"`` shape bypasses normalization/snapping and returns the exact same
values as the original ``create_target_arc_from_tracks`` (a pure endpoint linspace).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


SHAPES = ("linear", "double_peak", "log_rise", "inverted_parabola", "wave")


@dataclass(frozen=True)
class TargetArcConfig:
    """Configuration for target-arc shape and per-playlist normalization."""

    shape: str = "linear"
    ev_low_quantile: float = 0.05
    ev_high_quantile: float = 0.95
    # Fraction of playlist length over which each endpoint snap correction decays.
    snap_fraction: float = 0.15
    # Steepness of the logarithmic rise (higher = faster early rise).
    log_rise_k: float = 9.0
    # Standard deviation (in t) of the two bumps in the double-peak shape.
    double_peak_sigma: float = 0.13
    # Number of oscillations in the wave shape.
    wave_cycles: float = 3.0

    def __post_init__(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}, got {self.shape!r}.")
        if not 0.0 <= self.ev_low_quantile < self.ev_high_quantile <= 1.0:
            raise ValueError("EV quantiles must satisfy 0 <= low < high <= 1.")
        if not 0.0 <= self.snap_fraction < 0.5:
            raise ValueError("snap_fraction must be in [0, 0.5).")
        if self.log_rise_k <= 0:
            raise ValueError("log_rise_k must be positive.")
        if self.double_peak_sigma <= 0:
            raise ValueError("double_peak_sigma must be positive.")
        if self.wave_cycles <= 0:
            raise ValueError("wave_cycles must be positive.")


def _normalize_unit(values: np.ndarray) -> np.ndarray:
    """Scale an array to span [0, 1]; flat arrays map to all zeros."""
    minimum = float(values.min())
    value_range = float(values.max()) - minimum
    if value_range == 0.0:
        return np.zeros_like(values)
    return (values - minimum) / value_range


def _shape_curve(shape: str, t: np.ndarray, config: TargetArcConfig) -> np.ndarray:
    """Return the raw shape values in [0, 1] over progress ``t`` in [0, 1]."""
    if shape == "linear":
        curve = t
    elif shape == "log_rise":
        # Rises quickly, then levels off (concave).
        k = config.log_rise_k
        curve = np.log1p(k * t) / np.log1p(k)
    elif shape == "inverted_parabola":
        # Rises to a single mid-playlist peak, then falls.
        curve = 4.0 * t * (1.0 - t)
    elif shape == "double_peak":
        # Low start, peak, dip to a local minimum, second peak, lower toward the end.
        sigma = config.double_peak_sigma
        first = np.exp(-(((t - 0.30) / sigma) ** 2))
        second = np.exp(-(((t - 0.70) / sigma) ** 2))
        curve = first + second
    elif shape == "wave":
        # Starts high, oscillates with an upward-drifting floor so each local
        # minimum is higher than the previous one; peaks stay near the top.
        drift = 0.5 + 0.5 * t
        ripple = 0.5 * (1.0 + np.cos(2.0 * np.pi * config.wave_cycles * t))
        # Lower envelope rises with drift; upper envelope stays high.
        curve = drift + (1.0 - drift) * ripple
    else:  # pragma: no cover - guarded by TargetArcConfig validation
        raise ValueError(f"Unknown shape {shape!r}.")
    return _normalize_unit(np.asarray(curve, dtype=float))


def _snap_endpoints(
    arc: np.ndarray,
    start_value: float,
    end_value: float,
    snap_fraction: float,
) -> np.ndarray:
    """Blend the arc ends toward fixed endpoint values with decaying corrections."""
    length = len(arc)
    snapped = arc.astype(float).copy()
    start_correction = start_value - snapped[0]
    end_correction = end_value - snapped[-1]
    snap_length = max(1, int(round(snap_fraction * length)))
    indices = np.arange(length)
    start_weight = np.clip(1.0 - indices / snap_length, 0.0, 1.0)
    end_weight = np.clip(1.0 - (length - 1 - indices) / snap_length, 0.0, 1.0)
    snapped = snapped + start_correction * start_weight + end_correction * end_weight
    # Guarantee exact endpoints regardless of any snap-region overlap.
    snapped[0] = start_value
    snapped[-1] = end_value
    return snapped


def create_target_arc(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
    config: TargetArcConfig = TargetArcConfig(),
) -> np.ndarray:
    """Create a per-playlist normalized target EV arc of the configured shape.

    The ``"linear"`` shape returns exactly ``np.linspace(EV_start, EV_end, N)`` to
    remain byte-identical with the original pipeline.
    """
    if "EV_score" not in df.columns:
        raise ValueError("DataFrame must contain an EV_score column.")
    length = len(df)
    if (
        not 0 <= start_index < length
        or not 0 <= end_index < length
        or start_index == end_index
    ):
        raise ValueError("Start and end indices must be valid and distinct.")

    ev_scores = pd.to_numeric(df["EV_score"], errors="raise").to_numpy(dtype=float)
    start_value = float(ev_scores[start_index])
    end_value = float(ev_scores[end_index])

    if config.shape == "linear":
        return np.linspace(start_value, end_value, num=length, dtype=float)

    t = np.linspace(0.0, 1.0, num=length, dtype=float)
    raw = _shape_curve(config.shape, t, config)

    ev_low = float(np.quantile(ev_scores, config.ev_low_quantile))
    ev_high = float(np.quantile(ev_scores, config.ev_high_quantile))
    if ev_high <= ev_low:
        # Degenerate EV spread: fall back to an endpoint linspace.
        return np.linspace(start_value, end_value, num=length, dtype=float)

    arc = ev_low + raw * (ev_high - ev_low)
    return _snap_endpoints(arc, start_value, end_value, config.snap_fraction)
