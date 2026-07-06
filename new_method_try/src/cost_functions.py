"""Cost functions for transitions between playlist tracks."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TransitionCostConfig:
    """Weights and thresholds for the transition cost matrix.

    alpha: rhythm (tempo + meter) weight.
    beta: tonality weight.
    gamma: texture weight.
    meter_penalty: added to rhythm cost when time signatures differ.
    mode_height: z-height of major vs minor on the circle-of-fifths prism.
    tempo_jump_threshold: log2-ratio above which an extra penalty fires
        (0 = disabled; 0.30 ≈ 26% BPM change is a good DJ-motivated default).
    tempo_jump_penalty: extra cost multiplied by the excess above the threshold.
    """

    alpha: float = 1.0
    beta: float = 0.4
    gamma: float = 0.6
    meter_penalty: float = 0.25
    mode_height: float = 0.5
    tempo_jump_threshold: float = 0.0
    tempo_jump_penalty: float = 3.0


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


def circle_of_fifths_position(key: int) -> int:
    """Return a Spotify key's position on the circle of fifths."""
    if not isinstance(key, (int, np.integer)) or not 0 <= key <= 11:
        raise ValueError("key must be an integer from 0 through 11.")
    spotify_key_at_position = (0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5)
    return spotify_key_at_position.index(int(key))


def tonality_prism_coordinates(
    key: int,
    mode: int,
    mode_height: float = 0.5,
) -> tuple[float, float, float]:
    """Return circle-of-fifths prism coordinates for key and mode."""
    if mode not in (0, 1):
        raise ValueError("mode must be 0 (minor) or 1 (major).")
    position = circle_of_fifths_position(key)
    angle = 2 * np.pi * position / 12
    return (
        float(np.cos(angle)),
        float(np.sin(angle)),
        float(mode_height if mode else 0),
    )


def tonality_distance(
    key_u: int,
    mode_u: int,
    key_v: int,
    mode_v: int,
    mode_height: float = 0.5,
) -> float:
    """Return Euclidean distance on the circle-of-fifths prism."""
    coordinates_u = np.asarray(
        tonality_prism_coordinates(key_u, mode_u, mode_height)
    )
    coordinates_v = np.asarray(
        tonality_prism_coordinates(key_v, mode_v, mode_height)
    )
    return float(np.linalg.norm(coordinates_u - coordinates_v))


def camelot_tonality_distance(
    sector_u: int,
    mode_u: str,
    sector_v: int,
    mode_v: str,
) -> float:
    """Return circular Camelot sector distance plus A/B mode distance."""
    if not 1 <= int(sector_u) <= 12 or not 1 <= int(sector_v) <= 12:
        raise ValueError("Camelot sectors must be integers from 1 through 12.")
    mode_u = str(mode_u).strip().upper()
    mode_v = str(mode_v).strip().upper()
    if mode_u not in {"A", "B"} or mode_v not in {"A", "B"}:
        raise ValueError("Camelot modes must be A or B.")
    direct_distance = abs(int(sector_v) - int(sector_u))
    sector_distance = min(direct_distance, 12 - direct_distance)
    mode_distance = 0 if mode_u == mode_v else 1
    return float(sector_distance + mode_distance)


def _camelot_tonality_costs(df: pd.DataFrame) -> np.ndarray:
    """Return vectorized Camelot tonality distances for a track table."""
    sectors = pd.to_numeric(
        df["camelot_number"], errors="coerce"
    ).to_numpy(dtype=float)
    modes = df["camelot_mode"].astype(str).str.strip().str.upper().to_numpy()
    if (
        not np.isfinite(sectors).all()
        or np.any(sectors < 1)
        or np.any(sectors > 12)
        or np.any(sectors != np.floor(sectors))
    ):
        raise ValueError("Camelot numbers must be finite integers from 1 through 12.")
    if not np.isin(modes, ["A", "B"]).all():
        raise ValueError("Camelot modes must be A or B.")
    direct_sector = np.abs(sectors[:, np.newaxis] - sectors[np.newaxis, :])
    sector_costs = np.minimum(direct_sector, 12 - direct_sector)
    mode_costs = (modes[:, np.newaxis] != modes[np.newaxis, :]).astype(float)
    return sector_costs + mode_costs


def energy_momentum_penalty(
    previous_ev: float,
    candidate_ev: float,
    target_value: float,
    high_energy_threshold: float = 0.65,
    allowed_drop: float = 0.08,
    drop_weight: float = 2.0,
) -> float:
    """Penalize large EV drops after reaching a high-energy target region."""
    if (
        target_value < high_energy_threshold
        or candidate_ev >= previous_ev - allowed_drop
    ):
        return 0.0
    return float(drop_weight * ((previous_ev - candidate_ev) - allowed_drop))


def transition_cost(
    tempo_u: float,
    tempo_v: float,
    camelot_u: int | None = None,
    camelot_v: int | None = None,
    alpha: float = 1.0,
    beta: float = 0.4,
    gamma: float = 0.6,
    time_signature_u: float | None = None,
    time_signature_v: float | None = None,
    key_u: int | None = None,
    key_v: int | None = None,
    mode_u: int | None = None,
    mode_v: int | None = None,
    texture_u: np.ndarray | None = None,
    texture_v: np.ndarray | None = None,
    meter_penalty: float = 0.25,
    mode_height: float = 0.5,
) -> float:
    """Return one full-model transition cost.

    Texture vectors are expected to be normalized. Camelot numbers are used
    only when key and mode values are not supplied.
    """
    rhythm = tempo_distance(tempo_u, tempo_v)
    if (
        time_signature_u is not None
        and time_signature_v is not None
        and time_signature_u != time_signature_v
    ):
        rhythm += meter_penalty

    if None not in (key_u, key_v, mode_u, mode_v):
        tonality = tonality_distance(
            int(key_u), int(mode_u), int(key_v), int(mode_v), mode_height
        )
    elif camelot_u is not None and camelot_v is not None:
        tonality = camelot_distance(camelot_u, camelot_v)
    else:
        raise ValueError("Provide key/mode values or fallback Camelot numbers.")

    texture = 0.0
    if texture_u is not None and texture_v is not None:
        texture = float(
            np.linalg.norm(
                np.asarray(texture_u, dtype=float)
                - np.asarray(texture_v, dtype=float)
            )
        )
    return float(alpha * rhythm + beta * tonality + gamma * texture)


def compute_transition_cost_matrix(
    df: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 0.4,
    gamma: float = 0.6,
    meter_penalty: float = 0.25,
    mode_height: float = 0.5,
    tempo_jump_threshold: float = 0.0,
    tempo_jump_penalty: float = 3.0,
    config: "TransitionCostConfig | None" = None,
) -> np.ndarray:
    """Return full rhythm, tonality, and texture transition costs.

    Pass a ``TransitionCostConfig`` via ``config`` to set all parameters at once;
    individual keyword args are used when ``config`` is None (backward-compatible).
    When ``tempo_jump_threshold > 0`` (or set via config), a soft ramp penalty is
    added to the rhythm component for any pair whose log2-BPM ratio exceeds the
    threshold — discouraging jarring tempo jumps without creating a hard cliff.
    """
    if config is not None:
        alpha = config.alpha
        beta = config.beta
        gamma = config.gamma
        meter_penalty = config.meter_penalty
        mode_height = config.mode_height
        tempo_jump_threshold = config.tempo_jump_threshold
        tempo_jump_penalty = config.tempo_jump_penalty
    required_columns = {"tempo"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"DataFrame is missing transition columns: {missing}")

    tempos = pd.to_numeric(df["tempo"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(tempos).all() or np.any(tempos <= 0):
        raise ValueError("All tempo values must be finite and greater than zero.")

    tempo_costs = np.abs(np.log2(tempos[np.newaxis, :] / tempos[:, np.newaxis]))
    if "time_signature" in df.columns:
        meters = pd.to_numeric(
            df["time_signature"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(meters).all():
            raise ValueError("All time_signature values must be finite.")
        meter_costs = (meters[:, np.newaxis] != meters[np.newaxis, :]).astype(float)
    else:
        meter_costs = np.zeros_like(tempo_costs)
    rhythm_costs = tempo_costs + meter_penalty * meter_costs
    if tempo_jump_threshold > 0.0:
        excess = np.maximum(0.0, tempo_costs - tempo_jump_threshold)
        rhythm_costs = rhythm_costs + tempo_jump_penalty * excess

    if {"camelot_number", "camelot_mode"}.issubset(df.columns):
        tonality_costs = _camelot_tonality_costs(df)
    elif {"key", "mode"}.issubset(df.columns):
        keys = pd.to_numeric(df["key"], errors="coerce").to_numpy(dtype=float)
        modes = pd.to_numeric(df["mode"], errors="coerce").to_numpy(dtype=float)
        if (
            not np.isfinite(keys).all()
            or np.any(keys < 0)
            or np.any(keys > 11)
            or not np.isfinite(modes).all()
        ):
            raise ValueError("Key and mode values must be finite and valid.")
        if np.any(keys != np.floor(keys)) or np.any(modes != np.floor(modes)):
            raise ValueError("Key and mode values must be integers.")
        if np.any((modes < 0) | (modes > 1)):
            raise ValueError("Mode values must be 0 or 1.")
        coordinates = np.asarray(
            [
                tonality_prism_coordinates(int(key), int(mode), mode_height)
                for key, mode in zip(keys, modes)
            ]
        )
        tonal_differences = (
            coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
        )
        tonality_costs = np.linalg.norm(tonal_differences, axis=2)
    elif "camelot_number" in df.columns:
        camelot = pd.to_numeric(
            df["camelot_number"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(camelot).all():
            raise ValueError("Camelot numbers must be finite.")
        direct_camelot = np.abs(camelot[:, np.newaxis] - camelot[np.newaxis, :])
        tonality_costs = np.minimum(direct_camelot, 12 - direct_camelot)
    else:
        raise ValueError("DataFrame must contain key/mode or camelot_number.")

    texture_features = [
        feature
        for feature in (
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
        )
        if feature in df.columns
    ]
    if texture_features:
        texture = df[texture_features].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(texture).all():
            raise ValueError("Texture features must contain only finite values.")
        minimums = texture.min(axis=0)
        ranges = texture.max(axis=0) - minimums
        ranges[ranges == 0] = 1.0
        normalized_texture = (texture - minimums) / ranges
        differences = (
            normalized_texture[:, np.newaxis, :]
            - normalized_texture[np.newaxis, :, :]
        )
        texture_costs = np.linalg.norm(differences, axis=2)
    else:
        raise ValueError("At least one texture feature is required.")

    return (
        alpha * rhythm_costs
        + beta * tonality_costs
        + gamma * texture_costs
    )


def compute_transition_component_matrices(
    df: pd.DataFrame,
    alpha: float = 1.0,
    beta: float = 0.4,
    gamma: float = 0.6,
    meter_penalty: float = 0.25,
    mode_height: float = 0.5,
) -> dict[str, np.ndarray]:
    """Return transition cost component matrices matching C_trans exactly."""
    required_columns = {"tempo"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"DataFrame is missing transition columns: {missing}")

    tempos = pd.to_numeric(df["tempo"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(tempos).all() or np.any(tempos <= 0):
        raise ValueError("All tempo values must be finite and greater than zero.")

    tempo_costs = np.abs(np.log2(tempos[np.newaxis, :] / tempos[:, np.newaxis]))
    if "time_signature" in df.columns:
        meters = pd.to_numeric(
            df["time_signature"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(meters).all():
            raise ValueError("All time_signature values must be finite.")
        meter_costs = (meters[:, np.newaxis] != meters[np.newaxis, :]).astype(float)
    else:
        meter_costs = np.zeros_like(tempo_costs)
    rhythm_costs = tempo_costs + meter_penalty * meter_costs

    if {"camelot_number", "camelot_mode"}.issubset(df.columns):
        tonality_costs = _camelot_tonality_costs(df)
    elif {"key", "mode"}.issubset(df.columns):
        keys = pd.to_numeric(df["key"], errors="coerce").to_numpy(dtype=float)
        modes = pd.to_numeric(df["mode"], errors="coerce").to_numpy(dtype=float)
        if (
            not np.isfinite(keys).all()
            or np.any(keys < 0)
            or np.any(keys > 11)
            or not np.isfinite(modes).all()
        ):
            raise ValueError("Key and mode values must be finite and valid.")
        if np.any(keys != np.floor(keys)) or np.any(modes != np.floor(modes)):
            raise ValueError("Key and mode values must be integers.")
        if np.any((modes < 0) | (modes > 1)):
            raise ValueError("Mode values must be 0 or 1.")
        coordinates = np.asarray(
            [
                tonality_prism_coordinates(int(key), int(mode), mode_height)
                for key, mode in zip(keys, modes)
            ]
        )
        tonal_differences = (
            coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
        )
        tonality_costs = np.linalg.norm(tonal_differences, axis=2)
    elif "camelot_number" in df.columns:
        camelot = pd.to_numeric(
            df["camelot_number"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(camelot).all():
            raise ValueError("Camelot numbers must be finite.")
        direct_camelot = np.abs(camelot[:, np.newaxis] - camelot[np.newaxis, :])
        tonality_costs = np.minimum(direct_camelot, 12 - direct_camelot)
    else:
        raise ValueError("DataFrame must contain key/mode or camelot_number.")

    texture_features = [
        feature
        for feature in (
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
        )
        if feature in df.columns
    ]
    if texture_features:
        texture = df[texture_features].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(texture).all():
            raise ValueError("Texture features must contain only finite values.")
        minimums = texture.min(axis=0)
        ranges = texture.max(axis=0) - minimums
        ranges[ranges == 0] = 1.0
        normalized_texture = (texture - minimums) / ranges
        differences = (
            normalized_texture[:, np.newaxis, :]
            - normalized_texture[np.newaxis, :, :]
        )
        texture_costs = np.linalg.norm(differences, axis=2)
    else:
        raise ValueError("At least one texture feature is required.")

    weighted_rhythm_costs = alpha * rhythm_costs
    weighted_tonality_costs = beta * tonality_costs
    weighted_texture_costs = gamma * texture_costs
    total_costs = (
        weighted_rhythm_costs
        + weighted_tonality_costs
        + weighted_texture_costs
    )
    matrices = {
        "rhythm_costs": rhythm_costs,
        "tonality_costs": tonality_costs,
        "texture_costs": texture_costs,
        "weighted_rhythm_costs": weighted_rhythm_costs,
        "weighted_tonality_costs": weighted_tonality_costs,
        "weighted_texture_costs": weighted_texture_costs,
        "total_costs": total_costs,
    }
    expected_shape = (len(df), len(df))
    for name, matrix in matrices.items():
        if matrix.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}.")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} must contain only finite values.")
    if total_costs.shape != compute_transition_cost_matrix(
        df,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        meter_penalty=meter_penalty,
        mode_height=mode_height,
    ).shape:
        raise ValueError("total_costs must match C_trans shape.")
    return matrices
