"""Cost functions for transitions between playlist tracks."""

import numpy as np
import pandas as pd


def tempo_distance(tempo_u: float, tempo_v: float) -> float:
    """Return octave-aware absolute log-ratio distance between two tempos."""
    if tempo_u <= 0 or tempo_v <= 0:
        raise ValueError("Tempo values must be greater than zero.")
    return float(abs(np.log2(tempo_v / tempo_u)))


def camelot_distance(h_u: int, h_v: int) -> int:
    """Return circular distance between two Camelot wheel numbers."""
    if not 1 <= h_u <= 12 or not 1 <= h_v <= 12:
        raise ValueError("Camelot numbers must be integers from 1 through 12.")
    direct_distance = abs(h_u - h_v)
    return min(direct_distance, 12 - direct_distance)


def transition_cost(
    tempo_u: float,
    tempo_v: float,
    camelot_u: int,
    camelot_v: int,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Combine tempo and Camelot distances into one transition cost."""
    return (
        alpha * tempo_distance(tempo_u, tempo_v)
        + beta * camelot_distance(camelot_u, camelot_v)
    )


def compute_transition_cost_matrix(
    df: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """Return pairwise transition costs where entry [u, v] moves u to v."""
    required_columns = {"tempo", "camelot_number"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"DataFrame is missing transition columns: {missing}")

    tempos = pd.to_numeric(df["tempo"], errors="coerce").to_numpy(dtype=float)
    camelot = pd.to_numeric(
        df["camelot_number"], errors="coerce"
    ).to_numpy(dtype=float)

    if not np.isfinite(tempos).all() or np.any(tempos <= 0):
        raise ValueError("All tempo values must be finite and greater than zero.")
    if (
        not np.isfinite(camelot).all()
        or np.any(camelot < 1)
        or np.any(camelot > 12)
        or np.any(camelot != np.floor(camelot))
    ):
        raise ValueError("All Camelot numbers must be integers from 1 through 12.")

    tempo_costs = np.abs(np.log2(tempos[np.newaxis, :] / tempos[:, np.newaxis]))
    direct_camelot = np.abs(camelot[np.newaxis, :] - camelot[:, np.newaxis])
    camelot_costs = np.minimum(direct_camelot, 12 - direct_camelot)
    return alpha * tempo_costs + beta * camelot_costs
