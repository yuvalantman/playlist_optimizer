"""Search-weight sweep for the CURRENT BIBSConfig (replaces the broken legacy sweep).

The original src/bibs_config_sweep.py references config fields and diagnostics that were
deleted when BIBS was simplified, so it crashes. This sweep tunes only the *core* knobs
that still exist (arc/transition/bottleneck weights, beam width, and the new
stochastic/commit/eligibility modes) and ranks presets by a direction-aware composite.

Tuning these SEARCH weights against transition/coherence metrics is legitimate (they do
not redefine the metrics). We never tune the cost-definition weights (EV / C_arc / C_trans)
against the metrics, which would be circular.

Run from new_method_try/:
    python -m src.experiments.search_weight_sweep
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.bibs import BIBSConfig
from src.evaluator import (
    EvaluationConfig,
    PlaylistEvaluator,
    transition_percentiles,
)
from src.experiments.ladder import _run_bibs, build_inputs
from src.repair import RepairConfig, repair_playlist

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "results" / "sweep"
STOCHASTIC_SEEDS = (1, 2, 3, 4, 5)


def get_presets() -> dict[str, dict]:
    """Return named presets as (BIBSConfig kwargs, multi_seed, with_repair)."""
    base = BIBSConfig()
    return {
        "baseline_current": {"config": base, "multi_seed": False, "repair": False},
        "stronger_transition": {
            "config": replace(
                base,
                transition_weight=1.4,
                anchor_transition_weight=1.7,
                base_transition_weight=1.9,
            ),
            "multi_seed": False,
            "repair": False,
        },
        "stronger_arc": {
            "config": replace(
                base, arc_weight=1.5, anchor_arc_weight=1.5, base_arc_weight=0.6
            ),
            "multi_seed": False,
            "repair": False,
        },
        "wider_beam_8": {
            "config": replace(base, beam_width=8),
            "multi_seed": False,
            "repair": False,
        },
        "eligibility": {
            "config": replace(base, bottleneck_mode="eligibility"),
            "multi_seed": False,
            "repair": False,
        },
        "commit_beams": {
            "config": replace(base, commit_beams=True),
            "multi_seed": False,
            "repair": False,
        },
        "baseline+repair": {"config": base, "multi_seed": False, "repair": True},
        "stochastic": {
            "config": replace(base, stochastic=True),
            "multi_seed": True,
            "repair": False,
        },
        "stoch_elig+repair": {
            "config": replace(base, stochastic=True, bottleneck_mode="eligibility"),
            "multi_seed": True,
            "repair": True,
        },
    }


def _add_composite(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a direction-aware normalized composite (lower is better)."""
    lower_is_better = ("arc_rmse", "average_transition_cost", "p95_transition")
    higher_is_better = ("global_coherence",)
    components: list[pd.Series] = []
    for metric in lower_is_better + higher_is_better:
        values = pd.to_numeric(frame[metric], errors="coerce")
        spread = values.max() - values.min()
        if not np.isfinite(spread) or spread == 0:
            continue
        if metric in lower_is_better:
            components.append((values - values.min()) / spread)
        else:
            components.append((values.max() - values) / spread)
    ranked = frame.copy()
    ranked["composite_score"] = (
        pd.concat(components, axis=1).mean(axis=1) if components else 0.0
    )
    return ranked.sort_values("composite_score", kind="stable").reset_index(drop=True)


def run_sweep(genre: str = "pop", pool_size: int = 150, seed: int = 42) -> pd.DataFrame:
    """Run all presets and return a composite-ranked table."""
    inp = build_inputs(genre, pool_size, seed)
    c_arc, c_trans = inp["c_arc"], inp["c_trans"]
    evaluator = PlaylistEvaluator(EvaluationConfig())

    def evaluate(playlist: list[int]) -> dict[str, float]:
        metrics = evaluator.evaluate(playlist, c_arc, c_trans)
        metrics.update(transition_percentiles(playlist, c_trans))
        return metrics

    rows: list[dict[str, object]] = []
    for name, spec in get_presets().items():
        config = spec["config"]
        if spec["multi_seed"]:
            playlists = [
                _run_bibs(replace(config, random_seed=s), inp)
                for s in STOCHASTIC_SEEDS
            ]
        else:
            playlists = [_run_bibs(config, inp)]
        if spec["repair"]:
            playlists = [
                repair_playlist(p, c_arc, c_trans, RepairConfig())[0]
                for p in playlists
            ]
        metric_dicts = [evaluate(p) for p in playlists]
        mean = {
            key: float(np.mean([m[key] for m in metric_dicts]))
            for key in metric_dicts[0]
        }
        rows.append(
            {
                "preset": name,
                "arc_rmse": mean["arc_rmse"],
                "total_transition_cost": mean["total_transition_cost"],
                "average_transition_cost": mean["average_transition_cost"],
                "global_coherence": mean["global_coherence"],
                "p95_transition": mean["p95_transition"],
                "max_transition": mean["p100_transition"],
            }
        )
    return _add_composite(pd.DataFrame(rows))


def main() -> None:
    genre, pool_size, seed = "pop", 150, 42
    frame = run_sweep(genre, pool_size, seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"sweep_{genre}_{pool_size}_seed{seed}.csv"
    frame.to_csv(out_path, index=False)
    print(f"Search-weight sweep: genre={genre} pool_size={pool_size} seed={seed}")
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
