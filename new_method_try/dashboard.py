"""Streamlit dashboard for the new_method_try playlist-ordering pipeline.

Run from the new_method_try/ directory (or anywhere):
    streamlit run dashboard.py

Lets you build song pools (all genres, auto-similar genres, or chosen genres), generate
several playlists with random or chosen arc shapes, run every baseline + our BIBS
variants, and compare all metrics. Extra tabs show per-playlist orderings with an
EV-vs-arc plot, BIBS run-to-run variability, every formula/cost function, and how each
metric is computed. All computation uses the new_method_try pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.arcs import SHAPES  # noqa: E402
from src.experiments.explain import FORMULAS, METRIC_EXPLANATIONS  # noqa: E402
from src.experiments.runner import (  # noqa: E402
    bibs_variability,
    evaluate_methods,
    genre_centroids,
    genre_counts,
    load_catalog,
    prepare_pipeline,
    run_all_methods,
    sample_pool,
    similar_genres,
    METHOD_INFO,
)

st.set_page_config(page_title="Playlist Ordering Lab", layout="wide")

DISPLAY_COLUMNS = [
    "method", "category", "arc_optimized", "arc_rmse", "total_transition_cost",
    "average_transition_cost", "global_coherence", "p90_transition", "p95_transition",
    "max_transition", "dtw_shape", "ratio_adherence_rmse",
]


@st.cache_data(show_spinner=False)
def cached_catalog() -> pd.DataFrame:
    return load_catalog()


@st.cache_data(show_spinner=False)
def cached_counts(_catalog_sig: int) -> pd.Series:
    return genre_counts(cached_catalog())


@st.cache_data(show_spinner=False)
def cached_centroids(_catalog_sig: int) -> pd.DataFrame:
    return genre_centroids(cached_catalog())


def track_table(playlist: list[int], pool: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("track_name", "artists", "EV_score", "camelot", "tempo")
            if c in pool.columns]
    table = pool.loc[playlist, cols].copy()
    table.insert(0, "position", range(len(table)))
    return table.reset_index(drop=True)


def run_experiment(config: dict) -> dict:
    """Generate playlists per the config and run all methods on each."""
    catalog = cached_catalog()
    centroids = cached_centroids(len(catalog))
    all_genres = sorted(centroids.index.tolist())
    rng = np.random.default_rng(config["base_seed"])
    results = []
    for i in range(config["num_playlists"]):
        seed = config["base_seed"] + i
        # --- choose genres for this playlist ---
        if config["genre_mode"] == "all":
            genres = None
        elif config["genre_mode"] == "similar":
            anchor = str(rng.choice(all_genres))
            genres = similar_genres(catalog, anchor, config["group_size"], centroids)
        else:  # chosen
            genres = config["chosen_genres"]
        # --- choose arc shape ---
        shape_pool = config["arc_shapes"]
        arc_shape = str(rng.choice(shape_pool))
        # --- build + run ---
        pool = sample_pool(catalog, genres, config["length"], seed)
        inp = prepare_pipeline(pool, seed, arc_shape, genres)
        playlists = run_all_methods(inp, seed)
        metrics = evaluate_methods(playlists, inp)
        results.append(
            {
                "index": i,
                "seed": seed,
                "genres": genres if genres else ["(all genres)"],
                "arc_shape": arc_shape,
                "inp": inp,
                "playlists": playlists,
                "metrics": metrics,
                "start_name": str(inp.track_pool.loc[inp.start_index, "track_name"]),
                "end_name": str(inp.track_pool.loc[inp.end_index, "track_name"]),
            }
        )
    # aggregate mean metrics across playlists per method
    combined = pd.concat([r["metrics"] for r in results], ignore_index=True)
    numeric = [c for c in DISPLAY_COLUMNS if c not in ("method", "category", "arc_optimized")]
    summary = (
        combined.groupby("method", sort=False)[numeric].mean().reset_index()
    )
    info = pd.DataFrame(
        [{"method": m, "category": v["category"], "arc_optimized": v["arc_optimized"]}
         for m, v in METHOD_INFO.items()]
    )
    summary = info.merge(summary, on="method", how="right")
    return {"results": results, "summary": summary, "config": config}


# --------------------------------------------------------------------------- #
# Sidebar configuration
# --------------------------------------------------------------------------- #
catalog = cached_catalog()
counts = cached_counts(len(catalog))
all_genres = sorted(counts.index.tolist())

st.sidebar.title("Experiment setup")
genre_mode_label = st.sidebar.radio(
    "Playlist source",
    ["All genres (random mix)", "Similar genres (auto groups)", "Choose genres"],
)
genre_mode = {"All genres (random mix)": "all",
              "Similar genres (auto groups)": "similar",
              "Choose genres": "chosen"}[genre_mode_label]

group_size = 3
chosen_genres: list[str] = []
if genre_mode == "similar":
    group_size = st.sidebar.slider("Genres per playlist (similar group)", 2, 3, 3)
elif genre_mode == "chosen":
    chosen_genres = st.sidebar.multiselect(
        "Genres to draw from", all_genres, default=all_genres[:3]
    )

num_playlists = st.sidebar.slider("Number of playlists", 1, 8, 3)
length = st.sidebar.slider("Playlist length (tracks)", 20, 200, 60, step=10)

arc_mode = st.sidebar.radio("Arc shape", ["Random per playlist", "Choose shape(s)"])
if arc_mode == "Random per playlist":
    arc_shapes = list(SHAPES)
else:
    arc_shapes = st.sidebar.multiselect(
        "Allowed shapes (random per playlist if multiple)",
        list(SHAPES), default=["double_peak", "inverted_parabola", "log_rise"],
    )
base_seed = st.sidebar.number_input("Base random seed", value=42, step=1)

run_clicked = st.sidebar.button("Run experiments", type="primary")
st.sidebar.caption(
    f"Estimated work: {num_playlists} playlists x {len(METHOD_INFO)} methods. "
    "Larger length/playlists take longer (BIBS dominates)."
)

if run_clicked:
    if genre_mode == "chosen" and not chosen_genres:
        st.sidebar.error("Pick at least one genre.")
    elif not arc_shapes:
        st.sidebar.error("Pick at least one arc shape.")
    else:
        config = {
            "genre_mode": genre_mode,
            "group_size": group_size,
            "chosen_genres": chosen_genres,
            "num_playlists": int(num_playlists),
            "length": int(length),
            "arc_shapes": arc_shapes,
            "base_seed": int(base_seed),
        }
        try:
            with st.spinner("Generating playlists and running all methods..."):
                st.session_state["experiment"] = run_experiment(config)
            st.sidebar.success("Done.")
        except ValueError as error:
            st.sidebar.error(str(error))


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_results, tab_examples, tab_var, tab_formulas, tab_metrics, tab_genres = st.tabs(
    ["Run & Results", "Playlist Examples", "BIBS Variability",
     "Formulas", "Metric Definitions", "Genre Explorer"]
)

experiment = st.session_state.get("experiment")


with tab_results:
    st.header("Results")
    if experiment is None:
        st.info("Configure the experiment in the sidebar and click **Run experiments**.")
    else:
        cfg = experiment["config"]
        st.caption(
            f"{cfg['num_playlists']} playlists x length {cfg['length']} | "
            f"source: {cfg['genre_mode']} | arc shapes: {', '.join(cfg['arc_shapes'])} | "
            f"base seed {cfg['base_seed']}"
        )
        st.subheader("Mean metrics across playlists (per method)")
        summary = experiment["summary"][DISPLAY_COLUMNS]
        st.dataframe(
            summary.style.format(
                {c: "{:.4f}" for c in summary.columns
                 if summary[c].dtype.kind in "fc"}
            ),
            use_container_width=True,
        )
        st.caption(
            "Ignore arc_rmse for random / transition_greedy (not arc-optimized). "
            "arc_assignment is the arc_rmse lower bound. Lower is better for all except "
            "global_coherence (higher is better)."
        )
        with st.expander("Per-playlist metric tables"):
            for r in experiment["results"]:
                st.markdown(
                    f"**Playlist {r['index'] + 1}** - shape `{r['arc_shape']}` - "
                    f"genres: {', '.join(r['genres'])}"
                )
                st.dataframe(
                    r["metrics"][DISPLAY_COLUMNS].style.format(
                        {c: "{:.4f}" for c in DISPLAY_COLUMNS
                         if r['metrics'][c].dtype.kind in 'fc'}
                    ),
                    use_container_width=True,
                )


with tab_examples:
    st.header("Playlist examples & orderings")
    if experiment is None:
        st.info("Run an experiment first.")
    else:
        results = experiment["results"]
        labels = [
            f"Playlist {r['index'] + 1} ({r['arc_shape']}, {', '.join(r['genres'])})"
            for r in results
        ]
        choice = st.selectbox("Choose a playlist", range(len(results)),
                              format_func=lambda i: labels[i])
        r = results[choice]
        inp = r["inp"]
        st.write(
            f"**Start:** {r['start_name']}  -  **End:** {r['end_name']}  |  "
            f"**Arc shape:** `{r['arc_shape']}`  |  **Length:** {len(inp.track_pool)}"
        )

        method_names = list(r["playlists"].keys())
        plot_method = st.selectbox(
            "Plot EV trajectory for method", method_names,
            index=method_names.index("bibs_repair") if "bibs_repair" in method_names else 0,
        )
        playlist = r["playlists"][plot_method]
        chart_df = pd.DataFrame(
            {
                "target_arc": inp.target_arc,
                "playlist_EV": inp.ev_scores[np.asarray(playlist, dtype=int)],
            }
        )
        st.line_chart(chart_df, height=320)
        st.caption(
            "Blue = target arc; orange = EV_score of the chosen method's ordering. "
            "Closer = better arc adherence."
        )

        st.subheader("Orderings per method")
        st.caption(f"{METHOD_INFO.get(plot_method, {}).get('desc', '')}")
        for name, pl in r["playlists"].items():
            info = METHOD_INFO.get(name, {})
            with st.expander(f"{name}  -  {info.get('desc', '')}"):
                st.dataframe(track_table(pl, inp.track_pool), use_container_width=True,
                             height=300)


with tab_var:
    st.header("BIBS run-to-run variability (stochastic)")
    if experiment is None:
        st.info("Run an experiment first to get a pool to test on.")
    else:
        results = experiment["results"]
        labels = [f"Playlist {r['index'] + 1} ({r['arc_shape']})" for r in results]
        choice = st.selectbox("Pool to test", range(len(results)),
                              format_func=lambda i: labels[i], key="var_pool")
        runs = st.slider("Number of stochastic BIBS runs", 2, 16, 8)
        mode = st.radio("Bottleneck mode", ["score", "eligibility"], horizontal=True)
        if st.button("Run BIBS multiple times"):
            inp = results[choice]["inp"]
            with st.spinner("Running stochastic BIBS..."):
                playlists, metrics_df, diversity = bibs_variability(
                    inp, runs=runs, base_seed=1, bottleneck_mode=mode
                )
            st.metric("Diversity (mean pairwise position-disagreement)",
                      f"{diversity:.3f}",
                      help="0 = identical every run; near 1 = highly varied orderings.")
            st.dataframe(metrics_df.style.format(
                {c: "{:.4f}" for c in metrics_df.columns if metrics_df[c].dtype.kind in 'fc'}
            ), use_container_width=True)
            st.subheader("First 25 positions of the first 3 runs (to see differences)")
            preview = pd.DataFrame(
                {f"run {i + 1}": playlists[i][:25] for i in range(min(3, len(playlists)))}
            )
            st.dataframe(preview, use_container_width=True)
            st.caption("Track indices into the pool. Differing columns => different orderings.")


with tab_formulas:
    st.header("Formulas & cost functions")
    st.caption("Weights shown are the current defaults used by the pipeline.")
    for title, body in FORMULAS:
        with st.expander(title, expanded=False):
            st.markdown(body)


with tab_metrics:
    st.header("Metric definitions")
    st.caption("Every metric reported in the Results tab, and how it is computed.")
    for metric, body in METRIC_EXPLANATIONS.items():
        with st.expander(metric, expanded=False):
            st.markdown(body)


with tab_genres:
    st.header("Genre explorer")
    st.write(f"Catalog: **{len(catalog)}** tracks across **{len(all_genres)}** genres.")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Tracks per genre")
        st.bar_chart(counts.head(40))
        st.dataframe(counts.rename("count").reset_index().rename(
            columns={"index": "genre"}), use_container_width=True, height=300)
    with col2:
        st.subheader("Similar genres")
        g = st.selectbox("Genre", all_genres,
                         index=all_genres.index("pop") if "pop" in all_genres else 0)
        k = st.slider("How many", 2, 6, 3)
        sims = similar_genres(catalog, g, k, cached_centroids(len(catalog)))
        st.write("Closest by audio-feature centroid:")
        st.write(sims)
