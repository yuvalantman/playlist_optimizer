"""Human-readable explanations of every formula, cost function, and metric.

Consumed by the Streamlit dashboard (Formulas and Metric Definitions tabs). Kept in one
place so the explanations stay in sync with the code that actually runs.
"""

# Each entry: (title, markdown body). Weights shown are the current defaults.
FORMULAS: list[tuple[str, str]] = [
    (
        "EV_score (Energy-Valence score)",
        r"""
**What:** one scalar per track summarizing its energy/mood.

$$EV(s) = \frac{w_E\,\hat{E} + w_V\,\hat{V} + w_D\,\hat{D} + w_L\,\hat{L} + w_T\,\hat{T}}{w_E+w_V+w_D+w_L+w_T}$$

where $\hat{\cdot}$ are min-max normalized features over the pool: energy, valence,
danceability, loudness, tempo.

**Weights (current):** all equal to **1.0** (`EnergyValenceConfig`).
**Type:** heuristic / definitional. **Frozen** — not learned against the metrics
(that would be circular, since the arc and arc_rmse are built from EV).
**Note:** tempo appears here *and* in the transition rhythm cost (double-counted); a
no-tempo variant is available for experiments.
""",
    ),
    (
        "Target arc  A_EV(t)",
        r"""
**What:** the desired EV trajectory across playlist positions $t$.

Shapes (over normalized progress $u=t/(L-1)$, then mapped to the pool's robust EV range
$[Q_{05}, Q_{95}]$, then endpoints snapped to the fixed start/end EV):
- **linear:** $u$  (identical to the original pipeline)
- **log_rise:** $\log(1+ku)/\log(1+k)$, $k=9$ (fast rise, then plateau)
- **inverted_parabola:** $4u(1-u)$ (single mid peak)
- **double_peak:** two Gaussian bumps at $u=0.3,0.7$ (peak–dip–peak)
- **wave:** rising-floor oscillation

**Type:** linear is from the methodology; the non-linear shapes are new (Chapter 2 names
"double peak"/"ramp" arcs). Normalization is per-playlist because EV has no absolute scale.
""",
    ),
    (
        "Arc cost  C_arc(s, t)",
        r"""
**What:** how well track $s$ fits position $t$.

$$C_{arc}(s,t) = \lvert EV(s) - A_{EV}(t) \rvert$$

Lower = better. **Type:** methodology. Stays as-is.
""",
    ),
    (
        "Transition cost  C_trans(u, v)",
        r"""
**What:** how smooth it is to go from track $u$ to track $v$.

$$C_{trans}(u,v) = \alpha\, d_{rhythm} + \beta\, d_{tonality} + \gamma\, d_{texture}$$

- **Weights:** $\alpha=1.0$ (rhythm), $\beta=0.4$ (tonality), $\gamma=0.6$ (texture).
- $d_{rhythm} = \lvert \log_2(tempo_v/tempo_u) \rvert + 0.25\cdot[\text{meter mismatch}]$
- $d_{tonality} =$ Camelot sector distance + mode distance (0/1). *(Code uses Camelot here;
  the circle-of-fifths "prism" only runs when Camelot columns are absent.)*
- $d_{texture} =$ Euclidean distance over 8 normalized audio features
  (danceability, energy, loudness, speechiness, acousticness, instrumentalness,
  liveness, valence). *(Methodology text says squared; code uses plain Euclidean.)*

**Type:** feature-level adaptation of Bittner et al. (2017). The audio-frame version
(timbre/chroma/vocalness/cue points) needs raw audio and is out of scope.
""",
    ),
    (
        "Bottleneck scores",
        r"""
**What:** which tracks/positions are hard to place.

- **Song bottleneck** $BS(s)$ = mean of the **5 lowest** $C_{arc}(s,\cdot)$ across positions
  (high = even its best slots fit poorly). *(Methodology §3.3 says $\min_t C_{arc}$; code uses
  mean-of-5, which is more robust.)*
- A track is a **bottleneck** if $BS(s) \ge$ the **90th percentile** (adaptive, per-playlist).
- **Candidate set** for a position = its 20 lowest-$C_{arc}$ tracks.

**Empirical note:** as wired, the bottleneck term barely affects placement (avg ≈ −0.02),
because the candidate orchestrator already surfaces low-arc (= low-bottleneck) tracks.
`bottleneck_mode="eligibility"` removes the direct score discount entirely.
""",
    ),
    (
        "Beam step score (BIBS)",
        r"""
**What:** scores a candidate when extending a partial beam toward a position $t$.

$$\text{step} = w_{arc}\,\widetilde{C_{arc}}(x,t) + w_{tr}\,\widetilde{C_{trans}}(\text{prev},x)
- w_{bn}\,\widetilde{BS}(x)$$

with clipping ($\widetilde{C_{arc}}\in[0,1]$, $\widetilde{C_{trans}}\in[0,2.5]$,
$\widetilde{BS}\in[0,1]$). **Weights:** $w_{arc}=1.0$, $w_{tr}=1.0$, $w_{bn}=0.6$.
**Stochastic mode:** instead of keeping the top-`beam_width` by score, survivors are
**sampled** with probability $\propto e^{-\text{score}/\tau}$ over the top-$k$
(`top_k=5`, `temperature=0.15`). **Type:** heuristic; tunable (search weights).
""",
    ),
    (
        "Anchor score (meet-in-the-middle)",
        r"""
**What:** scores the midpoint anchor that splits an interval, combining the forward and
backward beam costs with the anchor's own arc fit, its two seam transitions, bottleneck
bonus, and a left/right balance term.

**Weights:** anchor_arc=1.0, anchor_transition=1.2, anchor_bottleneck=0.6,
anchor_balance=0.3. Deterministic mode takes the min; stochastic mode samples among the
top-$k$ anchors. **Type:** the novel BIBS component (recursive local meet-in-the-middle).
""",
    ),
    (
        "Base-case ordering (micro phase)",
        r"""
**What:** for small intervals ($\le 4$ empty slots), brute-force the best local
permutation, **transition-dominated**.

$$\text{score} = w^{base}_{arc}\sum C_{arc} + w^{base}_{tr}\sum C_{trans} - w_{bn}\sum BS$$

**Weights:** base_arc=0.3, base_transition=1.5 (transition dominates), bottleneck=0.6.
**Type:** methodology micro phase. This is the step that genuinely reaches the output.
""",
    ),
    (
        "Micro-repair (post-pass)",
        r"""
**What:** after a playlist is built, locally reorder small windows around the worst
transitions; accept a reorder only if it **reduces total transition cost** and does not
raise arc RMSE beyond a guard ($+0.02$), keeping endpoints fixed and the playlist
exact-once. **Type:** new; method-agnostic (works on any constructor's output).
""",
    ),
    (
        "Parallel arbitration rule",
        r"""
**What:** in the level-synchronous parallel solver, all intervals at one depth propose an
anchor from the shared pool; if two want the same track, the interval where it yields the
**lower step score** keeps it (ties by index), the loser re-proposes, repeat until no
conflicts. Guarantees uniqueness and is reproducible per seed.
""",
    ),
]


METRIC_EXPLANATIONS: dict[str, str] = {
    "arc_rmse": r"""
**Arc adherence (RMSE).** How closely the playlist follows the target EV arc.
$$\text{arc\_rmse} = \sqrt{\tfrac{1}{L}\sum_t C_{arc}(p_t, t)^2}$$
Lower is better. Ignore it for methods that don't optimize the arc
(random, transition-only greedy).
""",
    "total_transition_cost": r"""
**Total transition cost (TS).** Sum of $C_{trans}$ over consecutive pairs.
$$\sum_{i} C_{trans}(p_i, p_{i+1})$$ Lower is better.
""",
    "average_transition_cost": r"""
**Average transition cost.** Total transition cost divided by $L-1$. Lower is better.
""",
    "global_coherence": r"""
**Global coherence (COH).** Sequential smoothness relative to the whole pool's diversity.
$$COH = 1 - \frac{\text{mean adjacent } C_{trans}^2}{\text{mean all-pairs } C_{trans}^2}$$
Near 1 = much smoother than random; near 0 = shuffle-like. Higher is better.
""",
    "p90_transition": "**p90 transition.** 90th-percentile transition cost — only ~10% "
    "of seams are worse. Worst-case smoothness. Lower is better.",
    "p95_transition": "**p95 transition.** 95th-percentile transition cost — the worst "
    "~5% of seams (the 'leftovers' test). Lower is better.",
    "max_transition": "**max transition.** The single worst transition in the playlist. "
    "Lower is better.",
    "dtw_shape": r"""
**DTW shape distance.** Dynamic Time Warping between the playlist's EV trajectory and the
target arc. Unlike arc_rmse (strict position-by-position), DTW allows small time shifts,
so it measures whether the playlist follows the arc *shape* even if slightly misaligned.
Auxiliary metric only (never used in optimization). Lower is better.
""",
    "ratio_adherence_rmse": r"""
**Ratio adherence (Flexer).** RMSE between each track's acoustic relative position
$pos(i)=D(i,\text{start})/(D(i,\text{start})+D(i,\text{end}))$ and its target $t/(L-1)$.
Measures how well the order interpolates acoustically from the start song to the end song.
Lower is better.
""",
    "diversity": r"""
**Diversity.** Mean pairwise position-disagreement across multiple stochastic runs, in
$[0,1]$. 0 = identical orderings every run; near 1 = highly varied. Higher = more novelty.
""",
}
