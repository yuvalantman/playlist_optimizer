# Playlist and Transition Optimization for Parties

A computational system that orders a fixed pool of songs into a playlist that (1) follows a
desired **energy trajectory** and (2) keeps **transitions between consecutive tracks smooth**.
The problem is treated as an exact-coverage sequencing problem: every track in the pool is used
exactly once, between a chosen start and end song.

The current, recommended implementation lives in **`new_method_try/`** and ships with an
interactive **Streamlit dashboard**. The original prototype (`main.py`, `src/`) is kept for
reference.

---

## Repository layout

```
playlist_optimizer/
├── README.md                     ← you are here
├── requirements.txt              ← deps for the original prototype
├── data/
│   └── spotify_tracks.csv        ← shared track catalog (read-only, used by everything)
│
├── new_method_try/               ← ★ CURRENT method + dashboard (use this)
│   ├── dashboard.py              ← Streamlit app (interactive experiments)
│   ├── requirements.txt          ← deps for the dashboard + pipeline
│   ├── src/                      ← pipeline: arcs, cost functions, BIBS, baselines, repair…
│   │   ├── experiments/          ← evaluation_suite.py, ladder.py, sweep, runner, explain
│   │   └── baselines/            ← random, greedy, Hungarian, Flexer, forward/MM beam
│   ├── outputs/results/          ← saved results (incl. the full evaluation suite)
│   └── FINAL_RESULTS_REPORT.md   ← detailed results write-up
│
├── docs/                         ← ★ ALL documents live here
│   ├── final_report/             ← the final report (DESIGNED .docx + 2 PDFs)
│   ├── report_versions/          ← earlier report drafts (archive)
│   ├── proposal_and_background/  ← initial proposal, Chapter 2 drafts, links
│   ├── notes/                    ← working notes (methods, formulas, citations)
│   └── references/               ← the cited academic papers (PDFs)
│
├── main.py                       ← original prototype entry point (legacy)
├── src/                          ← original prototype pipeline (legacy)
└── outputs/                      ← original prototype outputs (legacy)
```

> **Note on the two pipelines.** `new_method_try/` is a self-contained, improved copy of the
> pipeline with all the new methods and the dashboard. The top-level `main.py` / `src/` /
> `outputs/` are the earlier prototype, left untouched so older results stay reproducible.
> For anything new, use `new_method_try/`.

---

## Running the dashboard

The dashboard lets you build song pools, generate playlists with different energy-arc shapes,
run every baseline and every one of our methods, and compare all metrics side by side — with
per-playlist orderings, run-to-run variability, and the underlying formulas.

```bash
cd new_method_try
pip install -r requirements.txt      # numpy, pandas, scipy, streamlit, altair, plotly
streamlit run dashboard.py
```

It opens in your browser (usually http://localhost:8501). The track catalog is loaded
automatically from `../data/spotify_tracks.csv`.

**Using it:** in the left sidebar choose the genre mode (all / auto-similar / pick genres),
the number of playlists, the playlist length, the arc shape(s), and a random seed, then click
**Run experiments**. The tabs then show:

| Tab | Shows |
|-----|-------|
| **Results** | Mean metrics per method across your playlists (the main comparison table) |
| **Examples** | A specific playlist ordering with its EV-vs-target-arc plot |
| **Variability** | How much the stochastic methods change from run to run |
| **Evaluation** | Per-playlist metric tables |
| **Formulas / Metrics** | Every cost function and metric, defined |

> On Windows, if console diagnostics print garbled text, set `PYTHONIOENCODING=utf-8`.

---

## Reproducing the full evaluation (the "big test")

The headline results come from a wide evaluation suite: **9 cells** (3 experiment types ×
3 playlist lengths, N = 60/90/120), **40 playlists per cell = 360 playlists**, each ordered by
**20 methods**, across 5 arc shapes and 3 genre-diversity settings.

```bash
cd new_method_try
python -m src.experiments.evaluation_suite            # full run (40 playlists/cell)
python -m src.experiments.evaluation_suite --quick    # fast smoke test (few playlists)
python -m src.experiments.evaluation_suite --resume   # skip cells already computed
```

Results are written to `new_method_try/outputs/results/evaluation_suite/`:

- `ranking.csv` — the composite ranking (each method's average rank across all metrics)
- `summary_overall.csv` — mean of every metric per method, averaged over all 360 playlists
- `summary_by_*.csv` — the same, sliced by N size, arc shape, experiment type, genre set
- `<experiment_type>_N<size>.csv` and `raw_results.csv` — per-playlist rows

Two smaller experiments are also available:

```bash
python -m src.experiments.ladder               # ablation ladder on one pool
python -m src.experiments.search_weight_sweep  # BIBSConfig weight sweep
```

---

## How to read the results

Each ordering is scored on several metrics. Lower is better unless noted.

| Metric | Meaning | Better |
|--------|---------|--------|
| **arc_rmse** | How closely track energy follows the target trajectory | lower |
| **average_transition_cost** | Mean cost between consecutive tracks (smoothness) | lower |
| **total_transition_cost** | Sum of all consecutive-pair costs | lower |
| **p90 / p95 / max_transition** | Worst-case single "jarring" jumps | lower |
| **global_coherence** | How much smoother the sequence is than a random shuffle (0 = random, 1 = near-ideal) | **higher** |
| **dtw_shape** | Shape match to the arc, tolerant of small timing shifts | lower |
| **ratio_adherence_rmse** | Acoustic interpolation quality (Flexer-style) | lower |
| **method_time_s** | Algorithm run time | lower |
| **composite_rank** | Average of the method's rank on every metric above — the single headline score | **lower** |

The central difficulty is a **trade-off**: methods that nail the energy arc (e.g. the Hungarian
`arc_assignment`) tend to have terrible transitions, and methods with the smoothest transitions
(`transition_greedy`) drift off the arc. A good playlist has to balance both — which is exactly
what the top methods do.

---

## Final results (full 360-playlist suite)

### Composite ranking (lower = better)

| Rank | Method | Type | Composite |
|-----:|--------|------|----------:|
| 1 | **mm_beam_orch_stochastic** | ours | **4.35** |
| 2 | **mm_beam_orch_stochastic_repair** | ours | **4.90** |
| 3 | mm_beam_stochastic | ours | 5.10 |
| 4 | mm_beam_stochastic_repair | ours | 5.30 |
| 5 | mm_beam | baseline | 6.55 |
| 6 | greedy_arc_trans | baseline | 7.35 |
| 7 | transition_greedy | baseline | 7.60 |
| 8 | mm_beam_alpha2 | baseline | 9.30 |
| 9 | forward_beam | baseline | 10.40 |
| 10 | bibs_stoch_repair | ours | 11.30 |
| 11 | bibs_repair | ours | 11.40 |
| 12 | arc_assignment | baseline | 11.60 |
| 13 | bibs_parallel | ours | 12.10 |
| 14 | bibs_stochastic | ours | 12.30 |
| 15 | flexer_interp | baseline | 12.80 |
| 16 | bibs_current | ours | 13.55 |
| 17 | bibs_tempo_aware | ours | 14.60 |
| 18 | bibs_traj_first | ours | 15.10 |
| 19 | bibs_alpha2 | ours | 17.00 |
| 20 | random | baseline | 17.40 |

### Key metrics (averaged over all 360 playlists)

| Method | arc_rmse | avg_trans | global_coh | p95_trans | max_trans |
|--------|---------:|----------:|-----------:|----------:|----------:|
| **mm_beam_orch_stochastic** | 0.176 | 0.920 | 0.818 | 1.827 | 2.804 |
| **mm_beam_orch_stochastic_repair** | 0.176 | 0.913 | 0.823 | **1.789** | **2.625** |
| mm_beam (baseline) | 0.176 | 0.920 | 0.818 | 1.828 | 2.805 |
| greedy_arc_trans (baseline) | 0.184 | 0.909 | 0.814 | 1.853 | 3.293 |
| transition_greedy (baseline) | 0.199 | **0.894** | 0.820 | 1.813 | 3.267 |
| arc_assignment (baseline) | **0.094** | 2.187 | 0.086 | 3.522 | 4.090 |
| bibs_current (ours) | 0.184 | 1.192 | 0.657 | 2.890 | 4.104 |
| bibs_repair (ours) | 0.184 | 1.098 | 0.736 | 2.284 | 3.167 |
| random (baseline) | 0.194 | 2.292 | 0.006 | 3.640 | 4.222 |

### What the results say

- **The best methods are ours, and they are not recursive BIBS.** The top four are all
  meet-in-the-middle (MM) beam variants with **stochastic sampling** and **candidate
  orchestration**. `mm_beam_orch_stochastic` has the best composite score; its
  `_repair` sibling has the best worst-case transitions (lowest p95 and max) in the whole study,
  which makes it the best pick for a real product.
- **Recursive BIBS underperforms its own simpler pieces.** `bibs_current` ranks 16th — its
  midpoint anchoring creates high-cost seams. But BIBS is still the *foundation*: its bottleneck
  detection and candidate orchestration are exactly what lift the winning MM-beam methods.
- **The repair pass is the highest-leverage single step** — it cuts worst-case transitions
  (p95/max) at essentially no cost to arc adherence, on top of any method.
- **Stochastic mode adds diversity for free** — different playlists per run at the same quality,
  which a real DJ tool needs.

Full discussion and figures are in **`docs/final_report/`** and
**`new_method_try/FINAL_RESULTS_REPORT.md`**.

---

## Data

`data/spotify_tracks.csv` is a catalog of tracks with pre-computed acoustic features (tempo,
energy, valence, danceability, loudness, key/mode, time signature, and more). Both pipelines
read it directly; it is never modified.
