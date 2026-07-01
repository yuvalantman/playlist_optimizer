"""Wide-range evaluation suite for the playlist-ordering pipeline.

Runs 9 experiment cells (3 experiment types × 3 playlist lengths) with configurable
playlist counts, saves raw per-playlist-per-method rows as CSV, and generates aggregated
summary tables sliced by N size, trajectory shape, experiment type, and genre set.

Experiment types
----------------
single_genre    : each playlist draws from one randomly chosen genre.
similar_genres  : each playlist draws from 3 auto-similar genres; playlists are grouped
                  into "sets" of 3 (one per trajectory shape) to cover all shapes evenly.
all_genres      : each playlist draws from up to ``max_genres_all`` random genres.

Endpoint selection
------------------
All experiments use the ``ev_proximity`` mode: start/end tracks are chosen as the songs
whose EV score is closest to the natural low/high endpoints of the target trajectory
(pool Q05 and Q95 EV), with no minimum-gap constraint.

Timing
------
Precompute time (EV scores, C_trans, C_arc, bottleneck detection, graph) is recorded
separately from per-method time so total_time_s = precompute_time_s + method_time_s.

Run from new_method_try/:
    python -m src.experiments.evaluation_suite            # default: 40 playlists/cell
    python -m src.experiments.evaluation_suite --playlists 10  # quick test
    python -m src.experiments.evaluation_suite --resume   # skip already-computed cells
"""

import argparse
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.arcs import SHAPES
from src.evaluator import (
    EvaluationConfig,
    PlaylistEvaluator,
    dtw_shape_distance,
    playlist_diversity,
    ratio_adherence_rmse,
    transition_percentiles,
)
from src.experiments.runner import (
    METHOD_INFO,
    bibs_variability,
    evaluate_methods,
    genre_counts,
    genre_centroids,
    load_catalog,
    prepare_pipeline_timed,
    run_all_methods,
    sample_pool,
    similar_genres,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "new_method_try" / "outputs" / "results" / "evaluation_suite"

TRAJECTORY_SHAPES = list(SHAPES)  # linear, double_peak, log_rise, inverted_parabola, wave


@dataclass
class EvalConfig:
    n_sizes: tuple[int, ...] = (60, 90, 120)
    playlists_per_cell: int = 40
    max_genres_all: int = 10
    base_seed: int = 42
    endpoint_mode: str = "ev_proximity"
    include_new_variants: bool = True
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)

    @property
    def shapes_cycle(self) -> list[str]:
        return TRAJECTORY_SHAPES


# --------------------------------------------------------------------------- #
# Pool builders for each experiment type
# --------------------------------------------------------------------------- #

def _random_shape(rng: np.random.Generator) -> str:
    return str(rng.choice(TRAJECTORY_SHAPES))


def _build_single_genre_playlists(
    catalog: pd.DataFrame,
    counts: pd.Series,
    n: int,
    cfg: EvalConfig,
) -> list[dict]:
    """Return metadata dicts for single-genre playlists."""
    genres = counts[counts >= n].index.tolist()
    if not genres:
        raise ValueError(f"No genre has >= {n} tracks.")
    rng = np.random.default_rng(cfg.base_seed + n)
    metas = []
    for i in range(cfg.playlists_per_cell):
        seed = cfg.base_seed + n * 1000 + i
        genre = str(rng.choice(genres))
        shape = _random_shape(rng)
        metas.append({
            "playlist_idx": i,
            "seed": seed,
            "genre_type": "single_genre",
            "genres": [genre],
            "trajectory_shape": shape,
            "genre_set": genre,
            "n_size": n,
        })
    return metas


def _build_similar_genre_playlists(
    catalog: pd.DataFrame,
    counts: pd.Series,
    centroids: pd.DataFrame,
    n: int,
    cfg: EvalConfig,
) -> list[dict]:
    """Return metadata dicts for similar-genre playlists in sets of 3 (one shape each)."""
    eligible = counts[counts >= n].index.tolist()
    rng = np.random.default_rng(cfg.base_seed + n + 1)
    n_sets = cfg.playlists_per_cell // 3
    metas = []
    shape_cycle = TRAJECTORY_SHAPES.copy()
    for set_idx in range(n_sets):
        anchor = str(rng.choice(eligible))
        try:
            genre_group = similar_genres(catalog, anchor, count=3, centroids=centroids)
        except ValueError:
            genre_group = [anchor, anchor, anchor]
        set_name = "+".join(genre_group)
        # Assign one trajectory shape per playlist within the set
        for playlist_within_set in range(3):
            shape = shape_cycle[(set_idx * 3 + playlist_within_set) % len(shape_cycle)]
            seed = cfg.base_seed + n * 1000 + set_idx * 3 + playlist_within_set + 10000
            metas.append({
                "playlist_idx": set_idx * 3 + playlist_within_set,
                "seed": seed,
                "genre_type": "similar_genres",
                "genres": genre_group,
                "trajectory_shape": shape,
                "genre_set": set_name,
                "n_size": n,
            })
    return metas


def _build_all_genre_playlists(
    catalog: pd.DataFrame,
    counts: pd.Series,
    n: int,
    cfg: EvalConfig,
) -> list[dict]:
    """Return metadata dicts for all-genre (up to max_genres_all random) playlists."""
    all_genres_list = counts.index.tolist()
    rng = np.random.default_rng(cfg.base_seed + n + 2)
    metas = []
    for i in range(cfg.playlists_per_cell):
        seed = cfg.base_seed + n * 1000 + i + 20000
        n_genres = int(rng.integers(2, cfg.max_genres_all + 1))
        chosen = [str(g) for g in rng.choice(all_genres_list, size=min(n_genres, len(all_genres_list)), replace=False)]
        shape = _random_shape(rng)
        metas.append({
            "playlist_idx": i,
            "seed": seed,
            "genre_type": "all_genres",
            "genres": chosen,
            "trajectory_shape": shape,
            "genre_set": "(all genres)",
            "n_size": n,
        })
    return metas


# --------------------------------------------------------------------------- #
# Per-playlist evaluation
# --------------------------------------------------------------------------- #

def _evaluate_one_playlist(
    meta: dict,
    catalog: pd.DataFrame,
    cfg: EvalConfig,
) -> list[dict]:
    """Run the full pipeline for one playlist and return per-method result rows."""
    genres = meta["genres"] if meta["genre_type"] != "all_genres" else meta["genres"]
    try:
        pool = sample_pool(catalog, genres, meta["n_size"], meta["seed"])
    except ValueError as exc:
        return [{"error": str(exc), **meta}]

    try:
        inp, precompute_time = prepare_pipeline_timed(
            pool,
            meta["seed"],
            arc_shape=meta["trajectory_shape"],
            genres=genres,
            endpoint_mode=cfg.endpoint_mode,
        )
    except Exception as exc:
        return [{"error": f"precompute: {exc}", **meta}]

    try:
        playlists, timings = run_all_methods(
            inp,
            seed=meta["seed"],
            include_new_variants=cfg.include_new_variants,
        )
    except Exception as exc:
        return [{"error": f"run_all_methods: {exc}", **meta}]

    try:
        metrics_df = evaluate_methods(playlists, inp, timings=timings)
    except Exception as exc:
        return [{"error": f"evaluate: {exc}", **meta}]

    rows = []
    for _, row in metrics_df.iterrows():
        method = str(row["method"])
        method_time = float(timings.get(method, 0.0))
        rows.append({
            # experiment metadata
            "experiment_type": meta["genre_type"],
            "n_size": meta["n_size"],
            "playlist_idx": meta["playlist_idx"],
            "seed": meta["seed"],
            "genre_type": meta["genre_type"],
            "genres": json.dumps(meta["genres"]),
            "genre_set": meta["genre_set"],
            "trajectory_shape": meta["trajectory_shape"],
            # method
            "method": method,
            "category": row.get("category", ""),
            "arc_optimized": row.get("arc_optimized", True),
            # timing
            "precompute_time_s": round(precompute_time, 4),
            "method_time_s": round(method_time, 4),
            "total_time_s": round(precompute_time + method_time, 4),
            # metrics
            "arc_rmse": row.get("arc_rmse"),
            "total_transition_cost": row.get("total_transition_cost"),
            "average_transition_cost": row.get("average_transition_cost"),
            "global_coherence": row.get("global_coherence"),
            "p90_transition": row.get("p90_transition"),
            "p95_transition": row.get("p95_transition"),
            "max_transition": row.get("max_transition"),
            "dtw_shape": row.get("dtw_shape"),
            "ratio_adherence_rmse": row.get("ratio_adherence_rmse"),
        })
    return rows


# --------------------------------------------------------------------------- #
# Cell runner
# --------------------------------------------------------------------------- #

def run_cell(
    experiment_type: str,
    n: int,
    catalog: pd.DataFrame,
    counts: pd.Series,
    centroids: pd.DataFrame,
    cfg: EvalConfig,
    output_dir: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run one experiment cell (experiment_type × N), save incrementally, return DF."""
    cell_path = output_dir / f"{experiment_type}_N{n}.csv"

    if experiment_type == "single_genre":
        metas = _build_single_genre_playlists(catalog, counts, n, cfg)
    elif experiment_type == "similar_genres":
        metas = _build_similar_genre_playlists(catalog, counts, centroids, n, cfg)
    else:
        metas = _build_all_genre_playlists(catalog, counts, n, cfg)

    all_rows: list[dict] = []
    for i, meta in enumerate(metas):
        if verbose:
            print(
                f"  [{experiment_type} N={n}] playlist {i+1}/{len(metas)} "
                f"| genres={meta['genres'][:2]}... shape={meta['trajectory_shape']}"
            )
        rows = _evaluate_one_playlist(meta, catalog, cfg)
        all_rows.extend(rows)
        # Incremental save after each playlist to allow resume inspection
        pd.DataFrame(all_rows).to_csv(cell_path, index=False)

    df = pd.DataFrame(all_rows)
    if verbose:
        print(f"  ✓ {len(df)} rows saved to {cell_path.name}")
    return df


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

METRIC_COLS = [
    "arc_rmse", "total_transition_cost", "average_transition_cost", "global_coherence",
    "p90_transition", "p95_transition", "max_transition", "dtw_shape",
    "ratio_adherence_rmse", "method_time_s", "total_time_s",
]


def _agg(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    available_metrics = [c for c in METRIC_COLS if c in df.columns]
    return (
        df.groupby(group_cols, sort=False, dropna=False)[available_metrics]
        .mean()
        .round(5)
        .reset_index()
    )


def build_summaries(raw: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    """Build and save all aggregated tables. Returns dict of table name → DataFrame."""
    tables: dict[str, pd.DataFrame] = {}

    tables["overall"] = _agg(raw, ["method", "category"])
    tables["by_n_size"] = _agg(raw, ["method", "n_size"])
    tables["by_trajectory_shape"] = _agg(raw, ["method", "trajectory_shape"])
    tables["by_experiment_type"] = _agg(raw, ["method", "experiment_type"])
    tables["by_genre_set"] = _agg(
        raw[raw["experiment_type"] == "similar_genres"],
        ["method", "genre_set"],
    )
    tables["by_n_and_shape"] = _agg(raw, ["method", "n_size", "trajectory_shape"])
    tables["by_n_and_type"] = _agg(raw, ["method", "n_size", "experiment_type"])

    for name, table in tables.items():
        table.to_csv(output_dir / f"summary_{name}.csv", index=False)
        print(f"  → summary_{name}.csv ({len(table)} rows)")

    return tables


def rank_methods(overall: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Rank methods per metric (lower is better for cost metrics, higher for coherence)."""
    lower_better = [
        "arc_rmse", "total_transition_cost", "average_transition_cost",
        "p90_transition", "p95_transition", "max_transition",
        "dtw_shape", "ratio_adherence_rmse", "method_time_s",
    ]
    higher_better = ["global_coherence"]
    ranks: dict[str, pd.Series] = {"method": overall["method"]}
    for col in lower_better:
        if col in overall.columns:
            ranks[f"rank_{col}"] = overall[col].rank(ascending=True)
    for col in higher_better:
        if col in overall.columns:
            ranks[f"rank_{col}"] = overall[col].rank(ascending=False)
    rank_cols = [c for c in ranks if c.startswith("rank_")]
    rank_df = pd.DataFrame(ranks)
    rank_df["composite_rank"] = rank_df[rank_cols].mean(axis=1)
    rank_df = rank_df.sort_values("composite_rank").reset_index(drop=True)
    rank_df.to_csv(output_dir / "ranking.csv", index=False)
    print(f"  → ranking.csv")
    return rank_df


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def run_suite(cfg: EvalConfig, resume: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Run the full evaluation suite; return the combined raw results DataFrame."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading catalog...")
    catalog = load_catalog()
    counts = genre_counts(catalog)
    centroids = genre_centroids(catalog)

    experiment_types = ["single_genre", "similar_genres", "all_genres"]
    all_dfs: list[pd.DataFrame] = []

    for exp_type in experiment_types:
        for n in cfg.n_sizes:
            cell_path = cfg.output_dir / f"{exp_type}_N{n}.csv"
            if resume and cell_path.exists():
                print(f"[SKIP] {exp_type} N={n} (already exists, --resume)")
                all_dfs.append(pd.read_csv(cell_path))
                continue
            print(f"\n[RUN] {exp_type} N={n} ({cfg.playlists_per_cell} playlists)...")
            try:
                df = run_cell(exp_type, n, catalog, counts, centroids, cfg, cfg.output_dir, verbose)
                all_dfs.append(df)
            except Exception:
                print(f"  ERROR in {exp_type} N={n}:")
                traceback.print_exc()

    if not all_dfs:
        raise RuntimeError("No experiment cells completed.")

    raw = pd.concat(all_dfs, ignore_index=True)
    raw_path = cfg.output_dir / "raw_results.csv"
    raw.to_csv(raw_path, index=False)
    print(f"\nRaw results: {len(raw)} rows → {raw_path.name}")

    print("\nBuilding summary tables...")
    tables = build_summaries(raw, cfg.output_dir)

    print("\nRanking methods...")
    rank_methods(tables["overall"], cfg.output_dir)

    print("\nDone.")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Playlist ordering evaluation suite")
    parser.add_argument("--playlists", type=int, default=40,
                        help="Playlists per experiment cell (default 40)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip cells whose CSV already exists")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 3 playlists/cell, N=60 only, no new variants")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        cfg = EvalConfig(
            n_sizes=(60,),
            playlists_per_cell=3,
            include_new_variants=False,
            base_seed=args.seed,
        )
    else:
        cfg = EvalConfig(
            playlists_per_cell=args.playlists,
            base_seed=args.seed,
        )

    run_suite(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
