"""Deterministic BIBS configuration sweep for algorithmic calibration."""

from dataclasses import replace

import numpy as np
import pandas as pd

from src.bibs import BIBS, BIBSConfig
from src.candidate_orchestrator import CandidateOrchestrator
from src.evaluator import PlaylistEvaluator
from src.transition_graph import TransitionGraphData


def get_bibs_config_presets() -> dict[str, BIBSConfig]:
    """Return named BIBS configurations for controlled calibration."""
    baseline = BIBSConfig()
    return {
        "baseline_current": baseline,
        "stronger_arc": replace(
            baseline,
            arc_weight=1.5,
            anchor_arc_weight=1.5,
            base_arc_weight=0.6,
        ),
        "stronger_transition": replace(
            baseline,
            transition_weight=1.4,
            anchor_transition_weight=1.7,
            base_transition_weight=1.9,
        ),
        "softer_harmonic_motion": replace(
            baseline,
            large_camelot_jump_penalty=0.4,
            extreme_camelot_jump_penalty=0.8,
            harmonic_diversity_weight=0.25,
        ),
        "stronger_energy_momentum": replace(
            baseline,
            energy_momentum_weight=1.5,
            anchor_energy_momentum_weight=1.5,
            base_energy_momentum_weight=1.5,
            allowed_ev_drop=0.06,
        ),
        "balanced_dj_flow": replace(
            baseline,
            arc_weight=1.25,
            transition_weight=1.35,
            bottleneck_weight=0.25,
            energy_momentum_weight=1.25,
            harmonic_diversity_weight=0.3,
            anchor_arc_weight=1.25,
            anchor_transition_weight=1.5,
            anchor_bottleneck_weight=0.3,
            anchor_energy_momentum_weight=1.25,
            base_arc_weight=0.5,
            base_transition_weight=1.7,
            base_energy_momentum_weight=1.25,
            large_camelot_jump_penalty=0.5,
            extreme_camelot_jump_penalty=0.9,
        ),
        "bottleneck_light": replace(
            baseline,
            bottleneck_weight=0.15,
            anchor_bottleneck_weight=0.2,
        ),
        "bottleneck_strong": replace(
            baseline,
            bottleneck_weight=0.6,
            anchor_bottleneck_weight=0.7,
        ),
    }


def _add_composite_score(results: pd.DataFrame) -> pd.DataFrame:
    """Add a direction-aware normalized lower-is-better composite score."""
    lower_is_better = (
        "arc_rmse",
        "average_transition_cost",
        "anchor_alignment_cost",
        "number_of_large_camelot_jumps",
        "number_of_large_ev_drops_after_high_energy",
    )
    higher_is_better = ("global_coherence",)
    normalized_components: list[pd.Series] = []

    for metric in lower_is_better + higher_is_better:
        values = pd.to_numeric(results[metric], errors="coerce")
        valid = values.dropna()
        if valid.empty or np.isclose(valid.max(), valid.min()):
            continue
        if metric in lower_is_better:
            normalized = (values - valid.min()) / (valid.max() - valid.min())
        else:
            normalized = (valid.max() - values) / (valid.max() - valid.min())
        normalized_components.append(normalized)

    ranked = results.copy()
    if normalized_components:
        ranked["composite_score"] = pd.concat(
            normalized_components,
            axis=1,
        ).mean(axis=1, skipna=True)
    else:
        ranked["composite_score"] = 0.0
    return ranked.sort_values(
        ["composite_score", "config_name"],
        kind="stable",
    ).reset_index(drop=True)


def run_bibs_config_sweep(
    c_arc: np.ndarray,
    c_trans: np.ndarray,
    bottleneck_results: dict,
    graph_data: TransitionGraphData,
    candidate_orchestrator: CandidateOrchestrator,
    target_arc: np.ndarray,
    track_pool: pd.DataFrame,
    start_index: int,
    end_index: int,
    evaluator: PlaylistEvaluator,
) -> pd.DataFrame:
    """Run named BIBS presets and return a composite-ranked result table."""
    rows: list[dict[str, object]] = []
    for config_name, config in get_bibs_config_presets().items():
        orchestrator = CandidateOrchestrator(candidate_orchestrator.config)
        bibs = BIBS(config)
        playlist = bibs.generate(
            c_arc,
            c_trans,
            bottleneck_results,
            graph_data=graph_data,
            candidate_orchestrator=orchestrator,
            target_arc=target_arc,
            track_pool=track_pool,
            start_index=start_index,
            end_index=end_index,
        )
        metrics = evaluator.evaluate(playlist, c_arc, c_trans)
        harmonic = evaluator.compute_harmonic_stagnation_metrics(
            playlist,
            track_pool,
        )
        motion = evaluator.compute_harmonic_motion_diagnostics(
            playlist,
            track_pool,
        )
        energy = evaluator.compute_energy_flow_diagnostics(playlist, track_pool)
        diagnostics = bibs.get_diagnostics()
        anchor_alignment_method = getattr(
            evaluator,
            "compute_anchor_alignment_cost",
            None,
        )
        anchor_alignment_cost = (
            anchor_alignment_method(bibs.anchor_history)
            if anchor_alignment_method is not None
            else np.nan
        )
        rows.append(
            {
                "config_name": config_name,
                "arc_rmse": metrics["arc_rmse"],
                "total_transition_cost": metrics["total_transition_cost"],
                "average_transition_cost": metrics["average_transition_cost"],
                "global_coherence": metrics["global_coherence"],
                "anchor_alignment_cost": anchor_alignment_cost,
                "longest_exact_camelot_run": harmonic[
                    "longest_exact_camelot_run"
                ],
                "number_of_runs_longer_than_3": harmonic[
                    "number_of_runs_longer_than_3"
                ],
                "max_camelot_jump": motion["max_camelot_jump"],
                "number_of_large_camelot_jumps": motion[
                    "number_of_large_camelot_jumps"
                ],
                "average_camelot_jump": motion["average_camelot_jump"],
                "max_ev_drop": energy["max_ev_drop"],
                "number_of_large_ev_drops": energy["number_of_large_ev_drops"],
                "number_of_large_ev_drops_after_high_energy": energy[
                    "number_of_large_ev_drops_after_high_energy"
                ],
                "selected_bottleneck_anchors": diagnostics[
                    "selected_bottleneck_anchors"
                ],
                "energy_momentum_penalties_applied": diagnostics[
                    "energy_momentum_penalties_applied"
                ],
                "harmonic_penalties_applied": diagnostics[
                    "harmonic_penalties_applied"
                ],
                "harmonic_motion_penalties_applied": diagnostics[
                    "harmonic_motion_penalties_applied"
                ],
            }
        )
    return _add_composite_score(pd.DataFrame(rows))
