# New-Method Session — What We Built, Why, and How to Use It

This document summarizes the work done in this session on the playlist-ordering project:
the motivation, every change, the key findings, and exactly how to run and interpret the
new Streamlit dashboard. **The original working pipeline was not modified** — all new work
lives in the isolated `new_method_try/` folder.

---

## 1. Why this session happened

The project compares a **Greedy** baseline against **BIBS** (a recursive,
bottleneck-guided, bidirectional beam search) for ordering a fixed pool of songs to follow
an Energy–Valence (EV) "arc" while keeping song-to-song transitions smooth.

Two problems motivated the work:

1. **BIBS was losing.** On the committed results, BIBS beat Greedy on arc adherence but
   *lost* on transition smoothness and global coherence in essentially every run. The
   project's own success criterion (beat Greedy on all metrics) was unmet.
2. **The pipeline was narrow.** Only a single genre (`pop`), a single pool, one seed, and a
   **linear** target arc were ever tested — far less than the methodology/Chapter 2 describe
   (multiple genres, non-linear "double peak"/"ramp" arcs, stochastic Top-K "random walk").

Goal: improve the method and broaden the experiments **without breaking the working
pipeline**, and make it all easy to run and compare.

---

## 2. The approach: an isolated copy

Everything new is in **`new_method_try/`**, a faithful copy of the pipeline (its own `src/`,
`main.py`, `outputs/`). It reads the shared dataset `../data/spotify_tracks.csv` read-only.

- The copy with default settings **reproduces the original single-run numbers exactly**
  (verified byte-for-byte on the metric CSV and both playlists).
- New behavior is first-class but configurable; the linear-arc / deterministic defaults
  match the original, so nothing silently changed.

Why a copy instead of flags in the original: the owner wanted the working pipeline kept
pristine for reference while we experiment freely.

---

## 3. What was added (and why)

| Area | Files | What & why |
|---|---|---|
| **Non-linear target arcs** | `src/arcs.py` | The arc was linear-only. Added `double_peak`, `log_rise`, `inverted_parabola`, `wave` (plus `linear` default = identical). Each shape is normalized to the playlist's own EV range (robust Q05–Q95 quantiles) and **endpoint-snapped** so it starts/ends exactly at the chosen start/end tracks. Why: real DJ sets use shaped energy arcs; non-linear arcs are where global planning should beat greedy. |
| **Stochastic random-walk + commit** | `src/bibs.py`, `src/sampling.py` | The code was fully deterministic; Chapter 2 actually intends a **Top-K sampled random walk** so each run differs. Added `stochastic`, `top_k`, `temperature`, `random_seed`, and `commit_beams` (write the chosen beam paths instead of discarding them). Deterministic default unchanged. Why: deliver run-to-run novelty/diversity and let the beams actually reach the output. |
| **Micro-repair** | `src/repair.py` | A guarded, post-hoc local-reordering pass (the methodology's "micro phase" that only existed as a prototype in `main.py`). It reorders small windows around the worst transitions, **only accepting moves that reduce transition cost and don't raise arc RMSE**, keeping the playlist exact-once. Method-agnostic (works on any constructor's output). Why: directly attacks the transition/coherence losses. |
| **Bottleneck demotion** | `src/bibs.py` | `bottleneck_mode = "score"` (default) or `"eligibility"` (removes the direct score discount; bottlenecks then only shape candidate exposure). Why: the bottleneck term can pull in bad-transition tracks. *Finding: it's already negligible (avg ≈ −0.02), so eligibility is nearly a no-op.* |
| **Parallel solving** | `src/parallel_solver.py` | A **level-synchronous** scheduler: after the first split both halves run "at the same time", then all four, etc. A barrier + **arbitration rule** resolves conflicts (if two intervals want the same song, the one where it has the lower step-score keeps it; the loser re-picks). Reproducible per seed, uniqueness guaranteed. Why: realize the owner's parallelism vision safely (free-running threads would be irreproducible and Python's GIL gives no speedup anyway). |
| **Baseline ladder** | `src/baselines/` | Six reference methods so each ingredient can be isolated: random, transition-only greedy, Hungarian arc-assignment (the arc lower bound), forward beam, MM bidirectional beam, and **Flexer interpolation** (start/end acoustic interpolation from Flexer 2008). Why: a proper ablation makes the write-up defensible. |
| **New metrics** | `src/evaluator.py` | Added (never replaced the originals): worst-case transition **p90/p95/max**, **DTW** shape distance, **ratio-adherence** RMSE (Flexer), and run-to-run **diversity**. Why: mean transition hides "leftovers"; stochastic methods need diversity; DTW captures shape-following that arc_rmse can't. |
| **Experiments** | `src/experiments/{ladder,search_weight_sweep,runner,explain}.py` | `ladder.py` runs every method on one pool; `search_weight_sweep.py` tunes BIBS knobs (replaces the **broken** legacy `bibs_config_sweep.py`); `runner.py` is the reusable engine (multi-genre pools, genre similarity, arc shapes); `explain.py` holds all formula/metric documentation. |
| **Dashboard** | `dashboard.py` | A Streamlit app to drive everything visually (see §6). |

---

## 4. Key empirical findings (pop, pool 150, seed 42 unless noted)

- **Original BIBS is dominated** by Greedy and even by the plain MM beam on transition cost
  and coherence (total ~152 vs Greedy ~116). The recursion as built isn't earning its place.
- **Arc vs transition are in sharp tension.** Hungarian arc-assignment hits arc_rmse 0.068
  (the lower bound) but transition ~344; transition-greedy is the opposite (~112 / arc 0.225).
  Balanced methods cluster at arc ~0.19–0.22 / transition ~112–122. **Non-linear arcs widen
  this tension** — the regime where global planning should pay off.
- **Repair is the cleanest win**: BIBS total 152→147, p95 2.29→1.92, coherence 0.772→0.803,
  at no arc cost. It topped the config sweep.
- **Stochastic mode gives high diversity** (0.89–0.93 on 150 tracks) at similar mean quality —
  the intended novelty/quality trade-off. So the honest success story is *arc adherence +
  bounded worst-case transitions + diversity*, **not** beating deterministic Greedy on mean
  transition (Greedy is the deterministic transition optimum by construction).
- **The bottleneck term is empirically negligible** — the candidate orchestrator already
  pre-filters to low-arc (= low-bottleneck) tracks.

---

## 5. How to set up and run

The project sits deep inside OneDrive, which can trip Windows' 260-char path limit. Put the
**virtual environment in a short path** to avoid it.

```powershell
# Create the venv at a short location (avoids the long-path error)
python -m venv C:\venvs\plo
C:\venvs\plo\Scripts\Activate.ps1

# Install + run from inside new_method_try
cd "C:\Users\yuval\OneDrive\מסמכים\BGU\6th_semester\Project\playlist_optimizer\playlist_optimizer\playlist_optimizer\new_method_try"
python -m pip install --upgrade pip
pip install -r requirements.txt          # numpy, pandas, scipy, streamlit, altair
streamlit run dashboard.py               # opens http://localhost:8501
```

(Alternative: enable long paths as admin — `Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1` — then a `.venv` inside the project works.)

Command-line experiments use the same venv:

```powershell
python -m src.experiments.ladder              # ablation across all methods
python -m src.experiments.search_weight_sweep # BIBS knob tuning
```

---

## 6. The dashboard, tab by tab

Configure in the **sidebar**, then click **Run experiments**:
- **Playlist source:** all genres (random mix) / similar genres (auto data-driven groups of
  2–3) / genres you choose.
- **Number of playlists** and **length**.
- **Arc shape:** random per playlist, or choose shape(s) (one picked at random per playlist).
- Start/end tracks are chosen automatically (generic low-EV start, high-EV end) per playlist.

Tabs:
1. **Run & Results** — mean metrics per method across all generated playlists, plus
   per-playlist tables. Every baseline and BIBS variant in one place.
2. **Playlist Examples** — pick a playlist; see its start/end and arc shape; a **chart of the
   ordering's EV trajectory vs the target arc**; and an expander per method with the full
   ordered tracklist (name, artist, EV, camelot, tempo).
3. **BIBS Variability** — pick a pool, run stochastic BIBS N times, see the **diversity
   score**, per-run metrics, and the first positions of several runs side-by-side to confirm
   it genuinely orders differently each run.
4. **Formulas** — every cost function (EV, arc, transition components, bottleneck, beam /
   anchor / base scores, repair, arbitration) with current weights and provenance.
5. **Metric Definitions** — how each reported metric is computed.
6. **Genre Explorer** — tracks-per-genre chart/table and a similar-genres lookup.

---

## 7. How to read the metrics

- **arc_rmse** (lower better): how tightly the ordering follows the target arc. *Ignore it for
  `random` and `transition_greedy`* (they don't optimize the arc); `arc_assignment` is the
  best-possible value (lower bound).
- **total / average_transition_cost** (lower better): overall smoothness.
- **global_coherence** (higher better, ~0→1): how much smoother adjacent songs are than random
  pairs from the pool.
- **p90 / p95 / max_transition** (lower better): the *worst* seams — the "leftovers" test.
  A method can have a fine average but bad p95 (one jarring jump).
- **dtw_shape** (lower better): does the ordering follow the arc *shape* even if slightly
  shifted (complements arc_rmse, which is strict position-by-position).
- **ratio_adherence_rmse** (lower better): how well the order interpolates acoustically from
  the start song to the end song (Flexer).
- **diversity** (0→1, higher = more varied): only meaningful for stochastic methods across
  multiple runs.

Rule of thumb: judge arc-optimizing methods on **arc_rmse + p95 + coherence + diversity**
together, not on mean transition alone.

---

## 8. What's intentionally unchanged & open decisions

- **Unchanged:** the entire original project (`src/`, `main.py`, `outputs/`, the academic
  docs). EV_score weights are frozen (learning them against the internal metrics would be
  circular). The original `main.py` diagnostics block was left in place.
- **Doc/code mismatches** (texture distance, tonality, bottleneck definition, the Chapter 2
  "random walk" narrative, etc.) are catalogued in `new_method_try/README.md` for you to fold
  into the write-up later — not edited here.
- **Open decision — arc-vs-transition balance under non-linear arcs:** recommended approach is
  high arc weight at *anchor* selection (realize the shape) with transition-dominated *base
  case + repair* (clean local seams); settle the exact balance empirically using the dashboard
  across arc shapes rather than guessing.
- **Next experiment:** run the ladder/dashboard across `double_peak` / `log_rise` arcs and
  multiple seeds — that's where BIBS's global structure should most clearly beat Greedy.

---

## 9. What we can still tune to improve

A crucial distinction first:

- **Search weights are safe to tune against the metrics** (beam/anchor/base/repair weights,
  beam width, sampling temperature, graph/candidate sizes). They change *how we search*, not
  *what we measure*.
- **Cost-definition weights change the ruler** (EV_score weights, the arc shape, and the
  transition `α/β/γ`). Tuning these to lower a metric can be circular/overfitting — change
  them for *musical* reasons and report them, don't optimize them against arc_rmse/coherence.

### A. BIBS search weights (safe; biggest near-term lever) — `BIBSConfig`
- `anchor_arc_weight` (1.0), `anchor_transition_weight` (1.2), `anchor_balance_weight` (0.3):
  raise arc weight to realize non-linear shapes more tightly; raise transition/balance to
  smooth the anchor seams. Recommended starting experiment: arc-heavy anchors + transition-heavy
  base.
- `base_arc_weight` (0.3) vs `base_transition_weight` (1.5): the base case is the step that
  actually reaches the output — push transition higher here to cut local roughness.
- `bottleneck_weight` / `anchor_bottleneck_weight` (0.6): currently near-inert; either drop to
  ~0 (use `bottleneck_mode="eligibility"`) or, if you want it to matter, also widen candidate
  exposure so bottleneck tracks actually compete.

### B. Beam & stochastic sampling (safe) — `BIBSConfig`
- `beam_width` (4): wider beams (8–16) explore more; diminishing returns and slower.
- `beam_length` (3): chunking granularity for expansion.
- `top_k` (5) and `temperature` (0.15): the novelty/quality dial. Lower temperature → closer to
  deterministic (less diversity, usually lower transition); higher → more diversity. Tune to the
  diversity level you want while watching p95.
- `commit_beams` (off): try on — it turns BIBS into a committed meet-in-the-middle; compare to
  the recursive default in the ladder.

### C. Repair (safe) — `RepairConfig`
- `window_radius` (2) / `max_window_size` (6): larger windows fix more but cost factorial time.
- `max_windows` (20): how many worst seams to attempt.
- `arc_guard` (0.02): how much arc RMSE you'll trade for smoother transitions (raise it to let
  repair help more aggressively; lower it to protect the arc).
- Idea: iterate repair to convergence, or add **Or-opt / 2-opt** (move single tracks / swap
  pairs across the whole playlist), not just local windows.

### D. Candidate exposure & graph recall (safe) — `CandidateOrchestratorConfig`, `TransitionGraphConfig`
- `max_candidates` (60): if BIBS is missing good options, raise it.
- Graph `threshold_percentile` (0.15), `min/max_out_degree` (5/40): looser thresholds / higher
  degree caps increase recall (more, better neighbors available) at some speed cost. Worth
  checking graph recall@10 on multi-genre pools where neighborhoods are sparser.

### E. Base case size (safe) — `BIBSConfig`
- `base_case_size` (4) / `base_case_max_candidates` (12): larger base cases optimize bigger
  windows by brute force (better local order) but grow factorially — keep modest.

### F. Start/end selection (musical) — `StartEndSelectionConfig`
- `min_ev_gap` (0.35): too high causes failures on narrow similar-genre pools (the dashboard's
  occasional error). Lower it (e.g., 0.2) for small/homogeneous pools, or scale it to the pool's
  EV spread.
- `start_quantile`/`end_quantile`, `transition_potential_weight`: bias endpoints toward
  well-connected songs to ease the whole ordering.

### G. Arc shape params (musical / definitional) — `TargetArcConfig`
- `ev_low_quantile`/`ev_high_quantile` (0.05/0.95): how much of the pool's EV range the arc
  spans; tighter quantiles → gentler arcs (easier to follow), wider → more dramatic (harder).
- `snap_fraction` (0.15): how quickly the arc bends to hit the fixed endpoints.
- Shape-specific knobs (`double_peak_sigma`, `log_rise_k`, `wave_cycles`) shape the curve.
- Consider new shapes (e.g., "hero's journey") if musically motivated.

### H. Transition cost `α/β/γ` (definitional — tune cautiously) — `compute_transition_cost_matrix`
- `α`=1.0 rhythm, `β`=0.4 tonality, `γ`=0.6 texture, `meter_penalty`=0.25. These define what
  "smooth" means; adjust for musical fidelity (e.g., weight harmony more for DJ-style mixing),
  then report — don't fit them to the metrics.

### I. EV_score (frozen by policy) — `EnergyValenceConfig`
- Keep the official definition fixed. The only sanctioned experiments: a **no-tempo** variant
  (tempo is double-counted with the rhythm cost), and—if ever needed—learning weights against an
  **external** target (human energy labels / MPD order) with held-out validation, never against
  the internal metrics.

### J. Parallel solver (safe) — `ParallelSolverConfig`
- Same arc/transition/bottleneck weights as BIBS, plus stochastic options. It currently uses a
  simpler (beamless) anchor score; adding a small lookahead would likely improve its quality.

### K. Structural / algorithmic ideas (bigger bets)
- **Fold Flexer's ratio into the objective** as an extra `C_ratio[track, pos]` cost dimension
  (start→end acoustic interpolation) with its own weight — already scaffolded as a future role.
- **Anneal / local-search the whole order** after construction (simulated annealing or large-
  neighborhood search guarded by arc), beyond windowed repair.
- **Multi-objective reporting** (arc vs transition Pareto front) instead of a single composite,
  so the trade-off is explicit per arc shape.
- **Per-shape weight presets** — the best arc/transition balance likely differs for `linear`
  vs `double_peak`; learn a small preset per shape via the (safe) search-weight sweep.

### L. Experiment coverage (do this regardless)
- Run the ladder/dashboard across **multiple seeds, genres, lengths, and all arc shapes**;
  report **mean ± std and best-of-N** (especially for stochastic methods) rather than single
  runs. This is what turns "looks better" into a defensible result.

---

See also: `new_method_try/README.md` (concise module map + mismatch notes) and the plan file
the work followed.
