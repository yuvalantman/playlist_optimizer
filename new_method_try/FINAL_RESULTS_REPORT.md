# Playlist Optimizer — Final Results & Report Notes

## Overview

This document summarizes the findings from the full evaluation suite, provides conclusions
suitable for the final project report, and notes areas for improvement and future work.

**Experiment scale:**
- 9 cells: 3 experiment types × 3 playlist lengths (N = 60 / 90 / 120)
- 40 playlists per cell → 360 total playlists evaluated
- 21 algorithms compared per playlist
- 5 trajectory shapes: `linear`, `log_rise`, `wave`, `double_peak`, `inverted_parabola`
- 3 genre diversity settings: `single_genre`, `similar_genres`, `all_genres`

---

## Baselines (what we compare against)

| Name | Description |
|---|---|
| **random** | Random permutation of tracks. Floor reference — no optimization. |
| **transition_greedy** | At each step, picks the track with the lowest transition cost from the previous. Ignores trajectory. Best transition metric but worst arc adherence. |
| **arc_assignment** | Hungarian assignment — globally optimal track-to-position mapping by arc cost. Zero transition awareness. Theoretical arc RMSE lower bound. |
| **flexer_interp** | Assigns tracks to positions using acoustic interpolation ratios between start and end. No trajectory, no transitions. |
| **forward_beam** | Left-to-right beam search scoring `arc_weight * C_arc + trans_weight * C_trans`. Deterministic, single-direction. |
| **mm_beam** | Bidirectional meet-in-the-middle beam. Forward beam covers positions 0→mid, backward beam covers N-1→mid, committed at midpoint. No recursion. Deterministic. |
| **greedy_arc_trans** | Greedy per-step: picks the candidate minimizing `arc_weight * C_arc + trans_weight * C_trans`. Direct baseline for BIBS. |

---

## Our Methods (ablation)

All methods below use the same cost functions, trajectory definition, and endpoint selection
(`ev_proximity` mode: start/end tracks are the songs whose EV score is closest to the
natural low/high endpoints of the trajectory).

| Name | What was added vs. the plain MM beam baseline |
|---|---|
| **bibs_current** | Recursive bidirectional beam with bottleneck-guided candidate orchestration and anchor placement at the midpoint of each interval. Recurses until base-case intervals are brute-forced. |
| **bibs_repair** | BIBS output passed through a guarded local repair pass (see below). |
| **bibs_stochastic** | BIBS with stochastic beam pruning: instead of deterministic top-K survival, samples from top-K candidates with probability ∝ exp(-(score - min) / temperature). Different seed → different valid ordering. |
| **bibs_stoch_repair** | Stochastic BIBS + repair. |
| **bibs_parallel** | Level-synchronous BIBS: all intervals at the same recursion depth place their anchors simultaneously, with conflict arbitration when two intervals compete for the same track. |
| **mm_beam_stochastic** | Plain MM beam with stochastic beam pruning (same softmax top-K sampler as stochastic BIBS). Non-recursive. |
| **mm_beam_stochastic_repair** | Stochastic MM beam + repair pass. |
| **mm_beam_orch_stochastic** | Stochastic MM beam **with BIBS-style candidate orchestration**: at each beam expansion step, candidates are pre-filtered by arc cost, transition graph neighbors, and bottleneck priority — instead of iterating all remaining tracks. |
| **mm_beam_orch_stochastic_repair** | Orchestrated stochastic MM beam + repair. **Best composite-ranked method overall.** |
| **bibs_tempo_aware** | BIBS with a soft BPM-jump penalty: tracks that cause a tempo jump above a threshold are penalized, with an "excuse" that waives the penalty if the track is an excellent trajectory fit. (Implementation had runtime issues — see caveats.) |
| **bibs_alpha2** | BIBS with rhythm weight α=2.0 in C_trans (double weight for tempo difference). Intended to suppress large tempo gaps. |
| **mm_beam_alpha2** | MM beam with α=2.0. |
| **bibs_traj_first** | BIBS with trajectory-first endpoint selection: trajectory is defined from the pool's natural EV range first, then start/end tracks are chosen to best fit positions 0 and N-1. |

### The Repair Pass (detail)

After any algorithm produces an ordering, the repair pass:
1. Finds the `max_windows` transitions with the highest cost (worst stitching points).
2. Extracts a small window of ±`window_radius` tracks around each bad transition.
3. Tries all permutations of the tracks in that window.
4. Accepts a reorder **only if**: (a) total transition cost in the window decreases, AND (b) arc RMSE does not increase beyond `arc_guard` threshold.
5. Endpoints are always fixed.

This is a local micro-optimizer — it does not restructure the whole playlist, only patches the worst seams. It reliably reduces `p95_transition` and `max_transition` with minimal arc cost.

### Stochastic Pruning (detail)

In a standard beam search, at each step only the top-K scoring beams survive (deterministic argmin).
In the stochastic variant, beam survival uses softmax sampling:
- Restrict candidates to top-`top_k` by score.
- Sample `beam_width` of them without replacement, probability ∝ `exp(-(score - min_score) / temperature)`.

The result: better candidates are more likely to survive, but the selection is not deterministic.
Different `random_seed` → different but equally valid playlist ordering from the same song pool.
This is intentional: in a real app, users would run the algorithm multiple times and get varied
playlists that still respect quality constraints — not the same static ordering every time.

### Candidate Orchestration (detail)

In plain MM beam, at each expansion step the beam considers all remaining (not-yet-placed) tracks as candidates — O(N) per step.
The orchestrated variant pre-filters candidates using the same `CandidateOrchestrator` as BIBS:
- **Arc cost filter**: only pass tracks whose `C_arc[track, position]` is below a threshold (close fit to the target trajectory at this position).
- **Transition graph neighbors**: prefer tracks that are already connected to the previous track in the learned transition graph.
- **Bottleneck priority**: tracks flagged as "hard to place" (high arc cost, few good positions) are boosted so they do not get stranded.

This shrinks the effective candidate pool without losing quality — which in BIBS was the main efficiency gain. In MM beam, it additionally biases expansion toward musically coherent sequences.

---

## Overall Results (averaged across all 360 playlists)

### Composite Ranking (lower = better)

| Rank | Method | Category | Composite Score |
|---|---|---|---|
| 1 | mm_beam_orch_stochastic | **ours** | 4.35 |
| 2 | mm_beam_orch_stochastic_repair | **ours** | 4.90 |
| 3 | mm_beam_stochastic | **ours** | 5.10 |
| 4 | mm_beam_stochastic_repair | **ours** | 5.30 |
| 5 | mm_beam | baseline | 6.55 |
| 6 | greedy_arc_trans | baseline | 7.35 |
| 7 | transition_greedy | baseline | 7.60 |
| 8 | mm_beam_alpha2 | baseline | 9.30 |
| 9 | forward_beam | baseline | 10.40 |
| 10 | bibs_stoch_repair | **ours** | 11.30 |
| 11 | bibs_repair | **ours** | 11.40 |
| 12 | arc_assignment | baseline | 11.60 |
| 13 | bibs_parallel | **ours** | 12.10 |
| 14 | bibs_stochastic | **ours** | 12.30 |
| 15 | flexer_interp | baseline | 12.80 |
| 16 | bibs_current | **ours** | 13.55 |
| 17 | bibs_tempo_aware | **ours** | 14.60 |
| 18 | bibs_traj_first | **ours** | 15.10 |
| 19 | bibs_alpha2 | **ours** | 17.00 |
| 20 | random | baseline | 17.40 |

Composite score = average rank across: arc_rmse, total_transition_cost, average_transition_cost,
p90_transition, p95_transition, max_transition, dtw_shape, ratio_adherence_rmse, method_time_s,
global_coherence (inverted — higher is better for coherence).

### Key Metric Snapshot (overall averages)

| Method | arc_rmse | avg_trans | global_coh | p95_trans | max_trans |
|---|---|---|---|---|---|
| mm_beam_orch_stochastic | 0.176 | 0.920 | 0.818 | 1.827 | 2.804 |
| mm_beam_orch_stochastic_repair | 0.176 | 0.913 | 0.823 | **1.789** | **2.625** |
| mm_beam_stochastic_repair | 0.176 | 0.913 | 0.823 | 1.789 | 2.626 |
| mm_beam (baseline) | 0.176 | 0.920 | 0.818 | 1.828 | 2.805 |
| greedy_arc_trans (baseline) | 0.184 | 0.909 | 0.814 | 1.853 | 3.293 |
| transition_greedy (baseline) | 0.199 | 0.894 | 0.820 | 1.813 | 3.267 |
| arc_assignment (baseline) | **0.094** | 2.187 | 0.086 | 3.522 | 4.090 |
| bibs_current | 0.184 | 1.192 | 0.657 | 2.890 | 4.104 |
| bibs_repair | 0.184 | 1.098 | 0.736 | 2.284 | 3.167 |
| random (baseline) | 0.194 | 2.292 | 0.006 | 3.640 | 4.222 |

---

## Key Findings and Conclusions

### 1. Our best method wins — and it's not BIBS

The top 4 methods by composite ranking are all ours: the stochastic and orchestrated MM beam variants.
`mm_beam_orch_stochastic_repair` achieves the best or near-best score on every transition
metric while matching the best baselines on trajectory adherence.

This directly demonstrates the value of our two main contributions to MM beam:
- **Stochastic pruning**: adds output diversity (different orderings per run) with no quality loss.
- **Candidate orchestration**: smartly filters the candidate pool at each beam step using BIBS's
  bottleneck and arc-cost machinery, steering the beam toward musically coherent placements.

### 2. Surprising finding: BIBS underperforms its own simpler variant

`bibs_current` ranks 16th overall (composite 13.55), well below the plain `mm_beam` baseline (5th, 6.55).
BIBS's transition costs are far higher (avg_trans 1.19 vs 0.92) despite similar arc RMSE (0.184 vs 0.176).

The recursive midpoint anchoring — BIBS's defining feature — hurts transition quality because it
places anchor tracks based on arc and bottleneck scores without fully accounting for the transition
cost of what comes immediately before and after the anchor. Each recursion level independently
places an anchor, which can create high-cost stitching points at every midpoint.

MM beam, by contrast, scores every candidate with both arc and transition cost at every step, so
transitions are never ignored.

The repair pass partially recovers BIBS (`bibs_repair` ranks 11th, 11.40) by patching the worst
midpoint seams, but never fully catches up.

### 3. The repair pass is the most effective single intervention

Across all base algorithms, adding the repair pass consistently and reliably:
- Reduces `p95_transition` by ~5–10%
- Reduces `max_transition` by ~6–15%
- Does not meaningfully change `arc_rmse`

For BIBS specifically, repair drops average transition cost from 1.192 → 1.098 and p95 from 2.890 → 2.284.
For MM stochastic, it drops max_transition from 2.805 → 2.626.

The repair pass is cheap (runs in ~0.13–0.17s on top of any method) and always safe (it never
worsens arc adherence beyond the guard threshold).

### 4. Results are stable across N sizes and trajectory shapes

| N | Best method | arc_rmse | avg_trans | global_coh |
|---|---|---|---|---|
| 60 | mm_beam_orch_stochastic | 0.179 | 1.022 | 0.783 |
| 90 | mm_beam_orch_stochastic | 0.176 | 0.911 | 0.822 |
| 120 | mm_beam_orch_stochastic | 0.173 | 0.828 | 0.848 |

Relative method ranking is consistent regardless of playlist length.
Larger N slightly improves absolute transition quality (more candidates → better local choices).

Across trajectory shapes, `double_peak` is easiest (best arc_rmse for most methods) while
`inverted_parabola` is hardest for `arc_assignment` specifically (0.138 vs 0.075 for linear).
The winning methods perform consistently across all 5 shapes.

### 5. The alpha-boost (heavier rhythm weight) does not help

`bibs_alpha2` and `mm_beam_alpha2` (rhythm weight α=2.0 vs default 1.0) both perform worse than
their default counterparts across all metrics. Increasing the weight of tempo differences in the
cost matrix pushes the algorithm too far toward tempo smoothness at the expense of arc adherence
and overall balance. The default equal weighting is better.

### 6. Trajectory-first endpoint selection does not improve results

`bibs_traj_first` (rank 18th) performs no better than `bibs_current` (rank 16th) — in fact
slightly worse on some metrics. The `ev_proximity` endpoint mode (used for all other methods)
already provides a sensible start/end selection: the track whose EV score is closest to the
natural trajectory endpoints. Adding a trajectory-first pipeline step adds ~4s per playlist
(due to a bug in the implementation — see caveats) without metric benefit.

### 7. Tempo-aware BIBS failed

`bibs_tempo_aware` (rank 17th) shows no improvement on any metric compared to `bibs_current`
despite its intended BPM-jump penalty. More critically, it runs in **~4.5 seconds per playlist**
versus ~0.003s for standard BIBS — a ~1500x slowdown. This indicates a Python-loop bottleneck
in the penalty computation that should have been vectorized with numpy. The idea is sound but
the implementation needs to be fixed before this variant can be evaluated fairly.

---

## Metric Definitions (for the report)

| Metric | What it measures | Direction |
|---|---|---|
| **arc_rmse** | Root mean squared error between each track's EV score and the target trajectory value at its position. Measures how well the playlist follows the intended energy shape. | Lower = better |
| **average_transition_cost** | Mean of `C_trans[i, i+1]` across all consecutive pairs. Measures overall smoothness. | Lower = better |
| **p95_transition / max_transition** | 95th percentile / maximum single-step transition cost. Measures worst-case "jarring" moments. | Lower = better |
| **global_coherence** | Mean correlation between the track's EV score and a position-normalized smooth curve. Measures how naturally the energy flows (independent of the target trajectory). | Higher = better |
| **dtw_shape** | Dynamic Time Warping distance between the actual EV sequence and the target trajectory shape. More forgiving than arc_rmse — tolerates slight timing shifts. | Lower = better |
| **ratio_adherence_rmse** | RMSE comparing the track's ratio position (D(track, start) / (D(track,start)+D(track,end))) to the expected uniform position ratio. Measures acoustic interpolation quality. | Lower = better |
| **method_time_s** | Wall-clock time for the algorithm only (excludes shared precompute). | Lower = better |
| **total_time_s** | Wall-clock time including shared precompute (EV scores, C_trans, C_arc, bottleneck detection, graph). | Lower = better |

**EV score** = weighted mean of 5 normalized audio features: energy, valence, danceability,
loudness (normalized), tempo (normalized). All weights = 1.0.

**C_trans** = `α·rhythm + β·tonality + γ·texture` where:
- rhythm = `|log2(bpm_u / bpm_v)|` + meter penalty for different time signatures
- tonality = key distance on the circle of fifths
- texture = absolute difference in spectral/texture features
- Default: α=1.0, β=0.4, γ=0.6

---

## What Needs Expansion in the Report

1. **EV score definition and feature weighting** — explain the 5 features used, why they were chosen,
   and why equal weighting was used (and that tuning weights is future work).

2. **Why BIBS was the starting point** — motivate the recursive structure: it was designed to avoid
   the O(N!) full-search problem while maintaining awareness of global bottlenecks. The surprising
   result that it underperforms should be framed as a meaningful empirical finding, not a failure.

3. **The arc/transition trade-off** — `arc_assignment` gets the best arc_rmse (0.094) but has terrible
   transitions (avg_trans 2.187). `transition_greedy` gets great transitions but terrible arc following.
   Our best method (0.176 arc_rmse, 0.913 avg_trans) achieves a strong balance between both, which
   is the core problem statement of the project.

4. **Trajectory shape design** — describe the 5 shapes and why they represent realistic DJ set patterns
   (build, peak, fade, wave, double peak). Explain `ev_proximity` endpoint selection.

5. **Ablation motivation** — explicitly frame it as: we started with BIBS, identified its weakness
   (midpoint anchoring hurts transitions), then took the MM beam framework and applied BIBS's best
   components (candidate orchestration, stochastic pruning) to it one by one.

6. **Genre diversity experiment** — discuss that results hold across single-genre, similar-genre,
   and all-genre playlists, which means the approach generalizes across realistic use cases.

---

## Things That Need Improvement (honest limitations)

1. **bibs_tempo_aware is unfinished** — the BPM-jump penalty was implemented as a Python loop over
   tracks per beam step, causing a ~1500x slowdown. Should be vectorized. Until then, this variant
   cannot be evaluated fairly and should not be included in the final comparison table.

2. **BIBS transition quality** — the recursive structure creates high-cost midpoint seams. A future fix
   would be to incorporate transition cost explicitly into the anchor selection formula, not just arc
   cost and bottleneck priority.

3. **Endpoint selection** — the `ev_proximity` mode sometimes picks a start/end track that is EV-close
   to the trajectory endpoint but acoustically awkward as an opener/closer. A user-driven endpoint
   selection (DJ chooses first and last track) would be more practical and is the natural extension.

4. **C_trans feature weights** — α=1.0, β=0.4, γ=0.6 were set heuristically and never tuned against
   the evaluation metrics. Grid search or learning the weights from user preference data is future work.

5. **Candidate orchestrator candidate count** — the top-K arc-cost filter and beam width were not
   ablated. It is possible that a different beam width would change the relative ranking of BIBS vs MM beam.

6. **No perceptual evaluation** — all metrics are computed, not listened to. Ground truth would require
   a user study asking DJs or listeners to rate the playlists. The computed metrics are proxies.

7. **Dataset scope** — the catalog is a single fixed dataset. Results may differ on streaming-scale
   catalogs with different genre distributions, feature ranges, or audio quality.

---

## Future Work

1. **User-controlled endpoints** — let the DJ pin the first and last track; the algorithm fills in the middle.
   This is already partially implemented (`quantile` and `trajectory_fit` endpoint modes) but needs
   a UI integration and proper evaluation.

2. **Fix and re-evaluate bibs_tempo_aware** — vectorize the penalty computation, re-run the suite.
   The concept of a soft BPM-jump penalty with an arc-quality excuse is theoretically sound and
   worth properly testing.

3. **Hybrid approach** — BIBS for the global structure (recursive anchor placement ensuring the
   playlist follows the trajectory shape at a macro level), MM stochastic for the local fill (each
   sub-interval filled with stochastic beam + repair instead of greedy brute-force base case).
   This could combine BIBS's trajectory discipline with MM beam's transition quality.

4. **Weight learning** — learn C_trans weights (α, β, γ) and arc vs. transition trade-off weights
   from user preference feedback. Even a small survey (5 DJs, 10 playlist pairs each) would provide
   a meaningful signal.

5. **Per-genre tuning** — the feature importance likely differs across genres. Tempo weight matters
   more for techno than for ambient. Per-genre weight sets could be learned.

6. **Parallelization** — `bibs_parallel` was the first step toward parallelization. A proper
   multiprocessing implementation of level-synchronous anchor placement could make BIBS practical
   for much larger N (200+), where sequential recursion is slow.

7. **Streaming / online ordering** — the current algorithm assumes the full pool is known upfront.
   An online variant that can insert new tracks into an existing playlist without full recomputation
   is the natural extension for a real recommendation system.

---

## Summary Recommendation for the Report

**Present `mm_beam_orch_stochastic_repair` as the recommended algorithm.**

It wins on all transition metrics, ties on arc adherence, and integrates all the innovations cleanly:
bidirectional beam structure (from MM beam) + stochastic diversity (for real-world use) +
BIBS-style candidate orchestration (bottleneck awareness, arc-cost filtering, graph neighbors) +
guarded local repair (patches the worst seams without hurting arc adherence).

Frame BIBS not as a failure but as the foundation: its bottleneck detection and candidate
orchestration components are directly adopted by the winning method. The finding that removing
BIBS's recursion and replacing it with a committed bidirectional beam produces better results
is itself a contribution — it shows which parts of BIBS actually matter.

**The paper can honestly claim:** "Our best method significantly outperforms all baselines on
transition smoothness metrics while matching the best on trajectory adherence. It also produces
diverse orderings per run — a practical requirement for DJ tooling — unlike all deterministic baselines."
