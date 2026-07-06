# new_method_try — experimental playlist-ordering improvements

This folder is an **isolated copy** of the working pipeline plus new, configurable methods.
The original project (`../src`, `../main.py`, `../outputs`) is untouched and still works; all
experimentation happens here. The shared dataset `../data/spotify_tracks.csv` is read-only.

```powershell
pip install -r requirements.txt           # numpy, pandas, scipy
python main.py                            # single run (linear arc = identical to original)
python -m src.experiments.ladder          # ablation ladder across all methods
python -m src.experiments.search_weight_sweep   # core BIBSConfig sweep (replaces broken legacy)
```
> On Windows consoles set `PYTHONIOENCODING=utf-8` (the inherited diagnostics print Hebrew text).

## What was added (all additive; defaults reproduce the original exactly)

| Area | File | Notes |
|---|---|---|
| Non-linear target arcs | `src/arcs.py` | shapes: `linear` (default, byte-identical), `double_peak`, `log_rise`, `inverted_parabola`, `wave`. Pool-range (Q05–Q95) normalization + endpoint snapping. Selected via `TARGET_ARC_SHAPE` in `main.py`. |
| Stochastic random-walk + commit | `src/bibs.py`, `src/sampling.py` | `BIBSConfig.stochastic`, `top_k`, `temperature`, `random_seed`, `commit_beams`. Deterministic default unchanged. |
| Bottleneck demotion | `src/bibs.py` | `BIBSConfig.bottleneck_mode = "score"` (default) or `"eligibility"`. |
| Micro-repair | `src/repair.py` | method-agnostic windowed reorder; exact-once + arc-guarded; only accepts transition-reducing moves. |
| Parallel solving | `src/parallel_solver.py` | level-synchronous (round-based) anchor placement with barrier + arbitration (lowest step-score wins a contested track). Reproducible per seed. |
| Baselines | `src/baselines/` | random, transition-only greedy, Hungarian arc-assignment (arc lower bound), forward beam, MM bidirectional beam, Flexer interpolation. |
| New metrics | `src/evaluator.py` | `dtw_shape_distance`, `transition_percentiles` (p90/p95/max), `ratio_adherence_rmse`, `playlist_diversity`. Original metrics kept unchanged. |
| Experiments | `src/experiments/` | `ladder.py`, `search_weight_sweep.py`. Output under `outputs/results/{ladder,sweep}/`. |

## Key empirical findings (pop, pool 150, seed 42)

- **The original deterministic BIBS is dominated** by simple greedy and even the plain
  MM beam on transition cost and coherence (total ~152 vs greedy ~116). Confirmed across
  the full ladder.
- **arc vs transition are in sharp tension:** Hungarian arc-assignment reaches arc_rmse 0.068
  (lower bound) but transition ~344; transition-greedy is the opposite (~112 transition).
  Balanced methods cluster at arc ~0.19–0.22 / transition ~112–122.
- **Repair helps BIBS** (total 152→147, p95 2.29→1.92, coherence 0.772→0.803) at no arc cost.
- **Stochastic mode delivers high diversity** (0.89–0.93 mean pairwise position-disagreement)
  with similar mean quality — the intended novelty/quality trade-off.
- **The bottleneck term is already negligible** (avg anchor bottleneck component ≈ −0.02), so
  `eligibility` mode is effectively a no-op on this pool: the candidate orchestrator already
  pre-filters to low-arc (= low-bottleneck) candidates.
- **DTW shape** and **ratio-adherence** cleanly separate arc-following methods (assignment,
  Flexer) from transition-following ones — useful complementary metrics.

Because the methods are stochastic, treat multi-seed mean ± best-of-N as primary; BIBS success
should be argued via arc adherence + bounded worst-case transitions + diversity, not by beating
deterministic greedy on mean transition.

## Doc/code mismatches to fold into the academic write-up (not yet edited there)

These describe the *running* code vs the methodology/Chapter 2 docs; the code is the better
source of truth except where noted:

1. **Texture distance** — methodology says squared Euclidean; code uses Euclidean (`np.linalg.norm`),
   which matches Bittner. Update the doc.
2. **Tonality** — methodology describes a 3D circle-of-fifths prism; with this dataset the code
   path that actually runs is Camelot sector + mode distance (the prism branch is dormant). Update
   the doc to describe Camelot as primary.
3. **Bottleneck score** — methodology §3.3 says `BS(s) = min_t C_arc`; code uses the mean of the 5
   lowest C_arc (more robust). Update the doc; also note the term is empirically negligible.
4. **Target arc** — docs mention double-peak / ramp arcs but the original code was linear-only;
   now implemented in `src/arcs.py`.
5. **EV_score** — equal weights (all 1.0) and tempo is double-counted (also in transition rhythm
   cost). Frozen as the official definition; a no-tempo variant is available for experiments.
   Do not learn EV weights against the internal metrics (circular).
6. **Method narrative** — Chapter 2 describes a "Context-Aware Random Walk + forward Beam Search +
   Top-K sampling"; the methodology doc + code implement recursive meet-in-the-middle BIBS. The
   stochastic mode added here reconciles the random-walk/Top-K intent with the BIBS structure.
   Align Chapter 2 to the stochastic recursive BIBS that exists.
7. **Audio-frame transition cost** — the formulas doc's Bittner Algorithm-1 (timbre/chroma/
   loudness/vocalness/drop/section over an n-beat overlap) needs audio and is out of scope; the
   current feature-level `C_trans` is the faithful adaptation. Keep; document audio version as future work.
8. **Legacy `src/bibs_config_sweep.py`** references deleted config fields and crashes; it is
   replaced here by `src/experiments/search_weight_sweep.py` for the current `BIBSConfig`.
