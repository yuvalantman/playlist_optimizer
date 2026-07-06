# Extracted Formulas and Functions — Source-by-Source Reference

This document compiles all formulas and functions extracted from the reviewed sources, organized by source paper. For each formula: the mathematical expression (in plain-text/Unicode notation), its purpose, and how it is used/calculated, including any specific parameters mentioned in the original text.

> **Note:** The source "Computational Architecture and Algorithmic Processing" was excluded per the original query constraints. Additionally, "Cue_Point_Estimation_using_Object_Detection" contains no formal mathematical equations — only neural network architectural descriptions — and is therefore not included below.

---

## 1. Automatic Playlist Sequencing and Transitions (2017)

### Crossfade Time-Stretch Calculation

```
d_out = ((N − n) / N) * d1 + (n / N) * d2
```

**Explanation & Usage:** This calculates the duration of the *n*-th beat in an *N*-beat transition when trying to sync two tracks of different tempos. `d1` is the duration of the beat in track 1, and `d2` is the duration of the beat in track 2. Track 1 is time-stretched by a factor of `d1/d_out`, and track 2 by `d2/d_out`.

### Transition Cost Matrices (Algorithm 1)

**Timbre:**
```
Λ_T[i, j] ← norm(T1[i : i+n] − T2[j : j+n])
```

**Chroma (Pitch):**
```
Λ_C[i, j] ← norm(C1[i : i+n] − C2[j : j+n])
```

**Loudness:**
```
Λ_ℓ[i, j] ← avg(2 − (ℓ1[i : i+n] + ℓ2[j : j+n]))
```

**Vocalness:**
```
Λ_v[i, j] ← avg(v1[i : i+n]) + avg(v2[j : j+n])
```

**Drop Points:**
```
Λ_D[i, j] ← 1{i+n ∉ D1} + 1{j+n ∉ D2}
```

**Section Boundaries:**
```
Λ_S[i, j] ← 1{i+n ∉ S1} + 1{j+n ∉ S2}
```

**Combined Cost:**
```
Λ ← weightedAvg(Λ_T, Λ_C, Λ_ℓ, Λ_v, Λ_D, Λ_S)
```

**Explanation & Usage:** These formulas evaluate the pair-wise local compatibility between an outgoing track 1 and an incoming track 2 over an overlapping transition region of *n* beats. They penalize transitions that don't match in timbre or chroma, have low loudness, overlap vocals, or fail to end on drop/section boundaries. The final transition point is chosen by taking the `argmin` (minimum cost entry) of the combined weighted matrix Λ.

---

## 2. Automated Generation of Music Playlists: Survey and Experiments

### Hit Rate

```
HitRate(Train, Test) = (1 / |Test|) * Σ_{(h,t)∈Test} 1_R(h)(t)
```

**Explanation & Usage:** Used to evaluate playlist generation models by hiding a track *t* from a history *h* and measuring if the algorithm successfully guesses the hidden item within its recommendation list `R(h)`. `1_R(h)(t)` is an indicator function that equals 1 if the track is in the recommended list.

### Average Log-Likelihood (ALL)

```
ALL(Train, Test) = (1 / |Test|) * Σ_{(h,t)∈Test} log( P_Train(t | h) )
```

**Explanation & Usage:** Evaluates the overall quality of algorithms based on the probability `P_Train` they assign to generating the true next track *t* given history *h*.

### k-Nearest Neighbors (kNN) Score & Similarity

**Similarity:**
```
sim(h, p) = |h ∩ p| / sqrt(|h| * |p|)
```

**Score:**
```
score_kNN(t, h) = Σ_{n∈N_h} sim(h, n) * 1_n(t)
```

**Explanation & Usage:** Ranks next-track candidates by finding playlists *n* (from nearest neighbors `N_h`) that are similar to the current history *h* using a binary cosine similarity.

### Frequent Patterns Score

```
score_pattern(t, h) = Σ_{ω∈Ω} confidence(ω → t)
```

**Explanation & Usage:** Scores a track *t* based on Association Rules or Sequential Patterns where Ω is the set of all possible rule antecedents in history *h*, and the score is the sum of the conditional probabilities (confidence) of those rules.

### Same Artists – Greatest Hits (SAGH)

```
score_SAGH(t, h) = counts(t) * 1_h(a)
```

**Explanation & Usage:** Scores tracks strictly by their popularity (`counts(t)`) *only* if their artist *a* is already present in the playlist history (`1_h(a) = 1`).

### Collocated Artists – Greatest Hits (CAGH)

**Artist Similarity:**
```
sim_a(a, b) = Σ_p (1_p(a) * 1_p(b)) / sqrt( Σ_p 1_p(a) * Σ_p 1_p(b) )
```

**Score:**
```
score_CAGH(t, h) = Σ_{b∈A_h} sim_a(a, b) * counts(t)
```

**Explanation & Usage:** Extends SAGH by scoring a track *t* (by artist *a*) based on its overall popularity multiplied by how frequently artist *a* is collocated in playlists with any artist *b* already in the history `A_h`.

### Mixture Model Linear Normalization

```
p_M(t | h) = score_M(t, h) / Σ_i score_M(i, h)
```

**Explanation & Usage:** Converts raw scores from algorithms into probabilities so they can be smoothed with a uniform distribution for ALL evaluation (avoiding zero probabilities).

---

## 3. From Raw Audio to a Seamless Mix: Creating an Automated DJ System for Drum and Bass

### Onset Detection Function (ODF) & Beat Tracking

**Melflux:**
```
Γ(m) = Σ_{k=1}^{40} HWR( X_mel40(m, k) − X_mel40(m−1, k) )
```

**Running Mean** *(parameter: Q = 16 frames)*:
```
Γ̄(m) = mean_{m−Q/2 ≤ q ≤ m+Q/2} { Γ(q) }
```

**Half-Wave Rectified (HWR) ODF:**
```
Γ_HWR(m) = HWR( Γ(m) − Γ̄(m) )
```

**Autocorrelation:**
```
A_Γ(l) = (1/M) * Σ_m Γ_HWR(m) * Γ_HWR(l + m)
```

**Tempo Curve:**
```
B(τ) = (1/N) * Σ_n A(n * L_τ)
```

**Phase Curve:**
```
Φ(φ) = (1/N) * Σ_n Γ_HWR(n * L_τ̂ + L_φ)
```

**Explanation & Usage:** Detects the exact position of beats in the audio. It measures spectral difference, removes the running mean, and uses autocorrelation to find the optimal beat period `τ̂` and beat phase `φ̂` by locating the peaks (`argmax`).

### Beat Location Mappings

**Seconds:**
```
t(m)_beat = m * τ̂ + φ̂
```

**Frames:**
```
L(m)_beat = round( m * L_τ̂ + L_φ̂ )
```

**Samples:**
```
N(m)_beat = round( m * N_τ̂ + N_φ̂ )
```

**Explanation & Usage:** Translates the *m*-th estimated beat into absolute seconds, spectral frames, and raw audio samples.

### Downbeat Tracking Features (Isolated and Contextual)

**Loudness Difference:**
```
X_loud^ctxt(m, l) = X_loud(m + l) − X_loud(m)
```

**Mel Spectrum Difference:**
```
X_mel^ctxt(m, l, k) = X_mel(m + l, k) − X_mel(m, k)
```

**ODF Correlation:**
```
X_odf,corr^(ctxt,i)(m, l) = Σ_{k=0}^{L_τ̂ − 1} Γ^(i)( L(m+l)_beat + k ) * Γ^(i)( L(m)_beat + k )
```

**Explanation & Usage:** Features fed into a logistic regression classifier to predict if a beat is the 1st, 2nd, 3rd, or 4th in a measure. Contextual features explicitly look at the differences between the current beat *m* and a beat *l* steps ahead.

### Structural Segmentation

**Cosine Distance** *(used for MFCC vectors)*:
```
d_cos(u, v) = 1 − ( u · v / (||u||₂ * ||v||₂) )
```

**Checkerboard Kernel:**
```
K(k, l) = 1   if k*l ≥ 0
K(k, l) = −1  if k*l < 0
```

**Novelty Curve** *(parameter: K = 64 frames)*:
```
Γ^(SSM, i)(m) = | Σ_{k=−K/2}^{K/2} Σ_{l=−K/2}^{K/2} K(k, l) * S_i(m+k, m+l) |
```

**Combined Novelty:**
```
Γ^(SSM)(m) = sqrt( Γ^(SSM, RMS)(m) * Γ^(SSM, MFCC)(m) )
```

**Explanation & Usage:** Convolves a checkerboard kernel across a Self-Similarity Matrix (SSM) of MFCCs and RMS energy to find structural boundaries in the track (like drops or breakdowns). The geometric mean is used so a peak only occurs if *both* RMS and MFCC novelty agree.

### Style Descriptor & Target Next Song Prediction

**Style Descriptor:**
```
v = PCA_3(X_style)
```
*(where `X_style` includes spectral contrast means, variances, and 1st-order diffs)*

**Next Song Target** *(parameters: α = 0.4, β = −0.1)*:
```
v̂_next = α * v_centroid + (1 − α) * ( β * v_prev + (1 − β) * v_cur )
```

**Explanation & Usage:** Simulates a DJ's style flow. It projects songs into a 3D PCA space. The target for the next song balances staying close to the playlist's core mood (`v_centroid`) while ensuring a deliberate stylistic progression away from the previous song.

---

## 4. Playlist Generation Using Start and End Songs

### Kullback-Leibler Divergence (Symmetric Approximation)

```
D_KL(p, q) = Tr(Σ_p⁻¹ Σ_q) + Tr(Σ_q⁻¹ Σ_p) + Tr( (Σ_p⁻¹ + Σ_q⁻¹)(μ_p − μ_q)(μ_q − μ_p)ᵀ )
```

**Explanation & Usage:** Computes the acoustic distance between two songs represented as single Gaussians (G1) with mean μ and covariance Σ over 20 MFCCs.

### Divergence Ratio and Interpolation

**Ratio** *(distance ratio of candidate track `i` to start song `s` vs. end song `e`)*:
```
R(i) = D_KL(i, s) / D_KL(i, e)
```

**Step Width** *(for p target positions)*:
```
step = ( R(s) − R(e) ) / (p + 1)
```

**Ideal Position** *(for the j-th target slot)*:
```
R̂(j) = R(s) + j * step
```

**Selection** *(track minimizing distance to the ideal ratio position)*:
```
S_j = argmin_i | R̂(j) − R(i) |
```

**Explanation & Usage:** Filters candidate songs by calculating how acoustically close they are to the start song (s) versus the end song (e). It then defines *p* perfectly evenly-spaced target positions along this ratio gradient, and assigns the real track `S_j` that minimizes the distance to the target position.

---

## 5. The Impact of Playlist Characteristics on Coherence in User-Curated Music Playlists

### Variance and Sequential Variance

**Population Variance:**
```
σ² = (1/n²) * Σ_{i=1}^{n} Σ_{j=i+1}^{n} d(x_i, x_j)²
```

**Sequential Variance:**
```
s⃗² = (1 / 2n) * Σ_{i=1}^{n−1} d(x_i, x_{i+1})²
```

**Explanation & Usage:** Population variance maps the overall diversity (all pairwise distances) in the playlist. Sequential variance measures the diversity only between *adjacent* tracks in the sequence.

### Playlist Coherence

```
coh = 1 − (s⃗² / σ²)
    = 1 − [ n² * Σ_{i=1}^{n−1} d(x_i, x_{i+1})² ] / [ 2(n−1) * Σ_{i=1}^{n} Σ_{j=i+1}^{n} d(x_i, x_j)² ]
```

**Explanation & Usage:** The central metric of the paper. It evaluates how smooth a playlist is structurally. It is bounded between −1 (abrupt changes) and 1 (perfectly smooth transitions) by comparing sequential variance to total population variance.

### Artist TF-IDF and Distance

**TF-IDF:**
```
tf(t, d) = f_{t,d} / max{ f_{t',d} }

idf(t, D) = log( N / |{ d ∈ D : t ∈ d }| )
```

**Artist Set Distance:**
```
d_artists(A_i, A_j) = (1 / (|A_i| * |A_j|)) * Σ_{a∈A_i} Σ_{b∈A_j} ( 1 − (m_a · m_b) / (||m_a|| * ||m_b||) )
```

**Explanation & Usage:** To prevent highly popular artists from biasing coherence, artists are weighted via TF-IDF in a playlist-artist matrix *m*. The distance between two tracks' artist sets is the average pairwise cosine distance.

### Bayesian Transition Probability

*(Dirichlet prior parameter: α = 1e-5)*

```
P(t_i | t_j) = (n_ij + α) / ( Σ_k n_ik + α )
```

**Explanation & Usage:** Estimates the probability that track `t_i` is followed by `t_j`. A Dirichlet prior parameter α = 1e-5 is used to smooth the probabilities to prevent overfitting on rare tracks.

### Optimization Errors (Greedy Rearrangement)

**Coherence Error:**
```
ε_coh = Σ_f ( E(coh_f) − coh_f(p) )²
```

**Transition Error:**
```
ε_trans(i) = Σ_f ( E(d_f) − d_f(t_i, t_{i+1}) )²
```

**Transition Improvement:**
```
Δ_trans(i) = Σ_f [ (E(d_f) − d_f(t'_i, t'_{i+1}))² − (E(d_f) − d_f(t_i, t_{i+1}))² ]
```

**Explanation & Usage:** Used by a greedy algorithm to reorder tracks. It minimizes the squared difference between the *target* coherence/distance expected for a playlist (based on its attributes) and the *actual* coherence/distance.

---

## 6. Automatic Cue Detection

### Data Normalization

```
z_i = (x_i − min(x)) / (max(x) − min(x))
```

**Explanation & Usage:** Min-Max scales the extracted audio feature arrays before sending them into the novelty detection algorithms.

### Phase Offset Identification (Weight Function)

```
g(k) := Σ_f  p-th_root[ (1 / (N/8)) * Σ_j  nov_f(b_{k+8j})²  ]      for k ∈ [0, 1, ..., 7]
```

**Explanation & Usage:** Finds the starting offset for structural musical periods (which occur every 4 bars). It maximizes the sum of weighted novelty values (`nov_f`) over all features across strong beats *b*, looking specifically for the shift *k* that best aligns with structural boundaries.

> **Note on missing information:** The original source expresses the inner term with a *p*-th root (`p√(...)`), but does not specify the value of *p* in the extracted text. It is reproduced here as "p-th_root[...]" rather than guessed at, to avoid introducing inaccurate information.

---

## 7. Bidirectional MM Search

### Priority Metric (Forward Search)

```
pr_F(n) = max( f_F(n), 2 * g_F(n) )
```

**Explanation & Usage:** The central function of the "Meet in the Middle" (MM) bidirectional algorithm. By factoring in `2 * g_F(n)`, it forces the priority of nodes expanding past the optimal midpoint to exceed the optimal solution cost, guaranteeing that the forward and backward searches meet strictly in the middle.

### Stopping Condition

```
Stop when:  U ≤ max( C, fmin_F, fmin_B, gmin_F + gmin_B + ε )
```

**Explanation & Usage:** `U` is the cost of the best solution found so far. ε is the cost of the cheapest edge. The algorithm safely terminates when the known solution `U` is less than or equal to any of the provided theoretical lower bounds.

---

## 8. Liebman — DJ-MC

### Expected Transition Reward

```
E[ R_t( (a_1, ..., a_{t−1}), a_t ) ] = Σ_{i=1}^{t−1} (1 / i²) * r_t(a_{t−i}, a_t)
```

**Explanation & Usage:** Calculates the listener's enjoyment of the transition to a new song `a_t`. The `1/i²` term models "memory decay," asserting that a song played *i* steps ago has a decaying probability of impacting the current transition reward.

### Linear Reward Models

**Song Reward:**
```
R_s(a) = φ_s(u) · θ_s(a)
```

**Transition Reward:**
```
R_t(a_i, a_j) = φ_t(u) · θ_t(a_i, a_j)
```

**Explanation & Usage:** Linear approximations using sparse binary feature vectors θ mapped to 10-percentile bins, dotted against user-specific preference weights φ.

### Model Update & Credit Assignment

**Magnitude/Direction** *(reward r, historical average r̄)*:
```
r_incr = log( r / r̄ )
```

**Song Weight:**
```
w_s = R_s(a_i) / ( R_s(a_i) + R_t(a_{i−1}, a_i) )
```

**Transition Weight:**
```
w_t = R_t(a_{i−1}, a_i) / ( R_s(a_i) + R_t(a_{i−1}, a_i) )
```

**Update** *(attenuating learning rate 1/(i+1))*:
```
φ_s = ( i / (i+1) ) * φ_s + ( 1 / (i+1) ) * θ_s * w_s * r_incr
```

**Explanation & Usage:** When the user provides a reward *r* (with historical average r̄), the system decides how much credit belongs to the individual song vs. the transition (via Maximum Likelihood estimation `w_s`, `w_t`). It then applies a Temporal Difference-style update with an attenuating learning rate `1/(i+1)`.

---

## 9. Spotify Playlist Optimization (Reinforcement Learning)

### Simulated User Models

**Sequential Model (SWM):**
```
p( y^(0,...,T) ) = Π_t  p( y_t | y_0, ..., y_{t−1}, i_0, ..., i_t, u )
```

**Non-Sequential Model (CWM):**
```
p( y^(0,...,T) ) = Π_t  p( y_t | i_t, u )
```

**Explanation & Usage:** Generative environment models that simulate a user's probability of skipping/completing a track `y_t`. The sequential model utilizes an LSTM structure to factor in previous track outcomes, whereas the CWM treats the response to the item features `i_t` and context `u` independently.

### Action-Head Deep Q-Network (AH-DQN)

**Bellman Equation:**
```
Q(s_t, a_t) = r(s_t, a_t) + γ * max_a Q(s_{t+1}, a)
```

**Policy:**
```
π_AH(s_t) = argmax_a Q(s_t, a)
```

**Explanation & Usage:** Evaluates the Quality (Q) of choosing action `a_t` in state `s_t`. Unlike standard DQNs where the output layer is mapped exactly to fixed discrete actions, this action-head ingests both the state *and* the item features of available tracks, allowing it to predict Q-values for dynamic, changing candidate pools dynamically.

---

## Summary: Notation Key

| Symbol | Meaning |
|---|---|
| `Σ` (capital sigma, as operator) | Summation |
| `Σ` (capital sigma, as matrix) | Covariance matrix (context-dependent — see source 4) |
| `Π` | Product |
| `Tr(...)` | Matrix trace |
| `argmin` / `argmax` | The input that minimizes / maximizes a function |
| `1_X(y)` or `1{condition}` | Indicator function — 1 if condition/membership holds, else 0 |
| `\|\|x\|\|` or `\|\|x\|\|₂` | (Euclidean / L2) norm of vector x |
| `x · y` | Dot product |
| `γ` | Discount factor (reinforcement learning) |
| `α`, `β` | Weighting/interpolation parameters (meaning is source-specific) |
| `φ`, `θ` | Preference weight vector / feature vector (DJ-MC, source 8) |
| `HWR(x)` | Half-wave rectification: `max(x, 0)` |
| `round(x)` | Nearest-integer rounding (original notation used a rounding bracket) |

---

## Notes on Completeness

- All formulas from the original extraction are represented above; none were omitted.
- LaTeX constructs (`\frac`, `\sum`, `\sqrt`, subscripts/superscripts, etc.) have been converted into linear plain-text/Unicode notation for portability and to avoid rendering issues outside LaTeX-aware viewers.
- Where the source text itself was ambiguous or incomplete — specifically the unspecified exponent *p* in the Phase Offset Identification formula (Source 6) — this has been flagged explicitly rather than silently resolved, so no information has been invented.
- Two sources mentioned in the original query (the Computational Architecture paper and the Cue Point Estimation paper) contain no extractable formulas and were correctly excluded, consistent with the original document.
