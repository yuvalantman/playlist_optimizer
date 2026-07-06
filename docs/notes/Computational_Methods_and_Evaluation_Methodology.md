# Computational Methods and Evaluation Methodology

**Authors:** Yuval Antman (212054167), Ohad Motola (209232420), Shelly Kritsberg (323866285), Tomer Filo (206969750)

---

## 1. System Objectives, Computational Architecture, and Information Flow

The proposed system is designed to construct a complete playlist from a fixed and predefined set of tracks, where every track must be placed exactly once. The goal of the system is to find a feasible, low-cost ordering that follows a predefined energy and mood arc as closely as possible, while also reducing the transition cost between consecutive tracks.

This creates two main requirements:
- **Macro level:** the sequence should match the intended energy and valence development across the playlist positions.
- **Micro level:** the transitions between an outgoing track and an incoming track should be musically coherent, mainly in terms of tempo and harmonic compatibility.

Since the set of tracks is closed, the system must also address the **"leftovers" problem**: difficult-to-place tracks cannot be ignored, and if they are left until the end, they may damage the quality of the final sequence.

### Computational Architecture Overview

The computational architecture of the system is built as a hybrid process that combines:
- pre-computation
- bottleneck identification
- recursive bidirectional search
- local micro-optimization

**Information entering the system:**
- the track pool
- the desired playlist arc
- the musical features of each track, including energy, valence, BPM, and musical key

**Processing flow:**

1. The first stage evaluates the fit between every track and every possible playlist position by calculating the **arc cost**.
2. Based on this calculation, the system identifies **bottleneck tracks** — tracks that are hard to place because even their best position still produces a relatively high distance from the target arc.
3. In parallel, the system computes **transition costs** between relevant candidate track pairs, using tempo and harmonic distance.
4. After this preparation stage, the system applies a **bottleneck guided recursive bidirectional search**. Instead of building the playlist only from beginning to end, the algorithm works from the boundaries of a segment toward an anchor position, while giving priority to difficult bottleneck tracks when they fit the relevant area of the arc. This reflects the main logic of the architecture: the system prioritizes the placement of the most restrictive tracks first, before arranging the more flexible tracks around them.
5. Once an anchor track is placed, it is removed from the available pool, and the segment is split into smaller sub-segments. The same process is then repeated recursively.
6. When the remaining gaps are small, the system switches to a **micro-level stage**: it evaluates a limited set of candidate arrangements for each gap, keeping up to **K** suitable candidate tracks per position, and selects the arrangement with the lowest local transition cost among the considered candidates.
7. If some tracks do not appear in any candidate set, they are assigned to their K most suitable positions so they are not left outside the search.

**Output:** a complete ordered playlist that uses all input tracks, aims to reduce deviation from the predefined energy and mood arc, and attempts to maintain smooth tempo- and harmony-based transitions throughout the sequence.

---

## System Architecture Diagram (Summary of Visual Overview)

**Title:** Concurrent Bottleneck-Guided Recursive Bidirectional Beam Search
*Simplified system architecture for playlist sequencing*

**Goal:** Build an ordered playlist that follows a target Energy–Valence arc while keeping transitions smooth and using every track exactly once.

### Stage 1 — Inputs & Preparation
- **Song Pool & Features:** Tracks with metadata and musical features.
- **Feature Normalization:** Scale features to comparable ranges.
- **Target Energy–Valence Arc:** Desired emotional trajectory across playlist positions.

### Stage 2 — Global Scoring
- **Arc Cost Matrix:** Compute how well each song fits each playlist position.
- **Bottleneck Detection:** Find hard-to-place songs and difficult positions.
- **Transition Cost:** Measure tempo / harmony compatibility between song pairs.

### Stage 3 — Search Core
- **Concurrent Bidirectional Beam Search:** Search from both ends of an interval toward the middle while prioritizing bottleneck tracks. (Diagram shows: Start (1) → ... → M (meeting point) ← ... ← End (L))

### Stage 4 — Anchor & Recursion
- **Anchor Selection:** Choose the best middle anchor using arc fit, transition quality, and bottleneck priority (arc fit + transition smoothness + bottleneck priority).
- **Recursive Split:** Split the interval around the anchor and solve sub-intervals recursively.

### Stage 5 — Base Case Optimization
- **Base-Case Matching:** For small intervals, assign remaining songs to remaining positions while minimizing local transition cost.

### Stage 6 — Output & Evaluation
- **Final Ordered Playlist:** Complete playlist of length L (positions 1, 2, 3, 4, ..., L).
- **Arc Error:** Deviation from the target Energy–Valence arc.
- **Transition Coherence:** Smoothness and compatibility of consecutive songs.
- **Coverage:** All tracks used exactly once.

**Key idea:** Place difficult songs early, preserve the global energy arc, and refine local transitions at the end.

---

## 2. Module-Level Design: Semantics, Inputs, Outputs, and Processing Methods

The following segments delineate the internal operational logic of the Concurrent Bottleneck-Guided Recursive Bidirectional Beam Search architecture. This section deconstructs the specific data inputs, resulting outputs, and the mathematical processing techniques utilized by every module to ensure the track pool aligns with the desired target parameters.

### 2.1 Formal Problem Definition

The core challenge is framed as a constrained sequencing problem, where the objective is to synthesize a seamless and coherent musical journey.

**System Inputs:**
- A collection of **N** tracks, each strictly characterized by identifiers (id, name, artist) and a suite of **13 musical features**, including duration, energy, valence, and tempo (BPM).
- Designated starting and ending tracks (`s_start` and `s_end`).
- A specific playlist length **L**.
- Global parameters required to synthesize the target Energy-Valence Arc.

**System Outputs:**
- A complete, ordered sequence **P** of length L that optimizes track-to-track fluidity while adhering to the specified emotional narrative.

---

### 2.2 Global Narrative: Energy-Valence Arc Cost Module

The Energy-Valence Arc serves as the blueprint for the intended emotional progression and development across the entire playlist.

**Module Inputs:**
- Total sequence length L.
- Specified arc parameters for the initial and final positions.
- Macro-level musical attributes for the track pool, such as energy, valence, danceability, and loudness.

**Module Outputs:**
- An integrated Energy-Valence Score (`EV_score`) assigned to every track.
- An Arc Cost Matrix (`C_arc`) quantifying the compatibility between each track and every possible sequence position.

**Processing Method:**

The system derives a composite EV_score for each track via a weighted summation of its primary musical features:

```
EV_Score(S) = w_E * energy + w_V * valence + w_D * danceability + w_L * norm_loudness + w_T * norm_tempo
```

The target emotional progression is defined as a continuous linear function over the playlist duration:

```
A_EV(t) = EV_Score_Start + progress(t) * (EV_Score_End − EV_Score_Start)
```

Relative progress at position t is normalized as follows:

```
progress(t) = (t − 1) / (L − 1)
```

The Arc Cost, measuring the macro-level distance between a track and its target position, is computed as:

```
C_arc(s, t) = | EV_Score(s) − A_EV(t) |
```

Lower values in this matrix signify a superior alignment with the intended global narrative flow.

---

### 2.3 Bottleneck Identification and Search Filtering Module

To mitigate the Leftovers Problem, this module identifies tracks and positions that are musically restrictive, ensuring these "bottlenecks" are prioritized early in the sequencing process.

**Module Inputs:**
- The pre-computed Arc Cost Matrix (`C_arc`).

**Module Outputs:**
- Track-specific (`BS_song`) and position-specific (`BS_loc`) Bottleneck Scores.
- Pruned Candidate Sets for every position in the playlist.

**Processing Method:**
- **Song Bottleneck Score (`BS_song`):** Derived by averaging the `k_5` lowest `C_arc` values for track s. High scores indicate tracks that are globally difficult to fit and require early placement.
- **Location Bottleneck Score (`BS_loc`):** Calculated as the average of the `k_2` lowest `C_arc` values for position t. This identifies positions with few suitable candidates, prompting the system to generate filtered Candidate Sets to streamline the search.

---

### 2.4 Local Fluidity: Transition Cost Module

This module quantifies the micro-level compatibility (`C_trans`) between consecutive tracks, ensuring musical coherence in terms of rhythm, tonality, and texture.

**Module Inputs:**
- Micro-level features for tracks u and v: musical key, mode, meter, and acoustic profiles.
- Tempo values for both candidates.

**Module Outputs:**
- A numerical cost penalty (`C_trans`) representing the quality of the transition.

**Processing Method:**

The total transition cost is decomposed into three musical dimensions:

```
C_trans(u, v) = (α * d_rhythm) + (β * d_tonality) + (γ * d_texture)
```

**Rhythmic Distance (`d_rhythm`):** Integrates a logarithmic tempo distance with a boolean penalty for mismatched time signatures.

```
d_tempo = | log2(tempo_v / tempo_u) |
d_rhythm = d_tempo + (0 if meters match else weight)
```

**Harmonic Distance (`d_tonality`):** Employs a 3D Euclidean distance on a circle-of-fifths based prism, ensuring harmonic compatibility between keys and modes (major/minor).

```
d_sector = min(|key_v − key_u|, 12 − |key_v − key_u|)
d_mode = 0 if mode_v == mode_u else 1
d_tonality = d_sector + d_mode
```

**Textural Distance (`d_texture`):** Calculates the geometric distance between acoustic profiles (e.g., acousticness, instrumentalness).

```
vector_u = [danceability_u, energy_u, norm_loudness_u, speechiness_u, acousticness_u, instrumentalness_u, liveness_u, valence_u]
vector_v = [danceability_v, energy_v, norm_loudness_v, speechiness_v, acousticness_v, instrumentalness_v, liveness_v, valence_v]

d_texture = sum over i of (vector_v[i] − vector_u[i])^2
```

(Note: This is a sum of squared differences across the feature vectors, i.e. squared Euclidean distance, with i ranging over the length of vector_v.)

---

### 2.5 Core Engine: Recursive Bidirectional Search and Anchor Locking

This central module recursively subdivides the playlist intervals, expanding inward from established boundaries toward a central anchor point.

**Module Inputs:**
- A target interval `[start, end]` with fixed boundary tracks.
- Pre-calculated matrices (`C_arc`, `C_trans`) and Bottleneck Scores.

**Module Outputs:**
- A designated "Anchor" track locked at midpoint **M**.
- Resulting sub-intervals `[start, M]` and `[M, end]` for subsequent recursion.

**Processing Method:**

The system performs a simultaneous forward and backward search. Candidates are evaluated using a composite heuristic score:

```
StepScore(x, prev, t) = (a * C_arc) + (b * C_trans) − (c * BS_song)
```

By incorporating a negative `BS_song` term, the algorithm is incentivized to place difficult tracks earlier. Inspired by the MM algorithm (Holte et al., 2016), the system applies a priority penalty that encourages searches to meet at the midpoint. While the beam search nature of the system precludes a strict mathematical guarantee of optimality, this mechanism effectively directs convergence toward the midpoint M. Once the searches meet, an anchor is selected and the segment is bifurcated.

---

### 2.6 Final Resolution: Base-Case Local Matching Phase

The concluding module resolves all remaining positions once recursive segments reach a minimal threshold.

**Module Inputs:**
- Sub-intervals of length 3 or 4 and all unplaced tracks.

**Module Outputs:**
- The final, fully sequenced musical tracklist.

**Processing Method:**

When intervals reach the base-case size, recursion terminates. The system evaluates all valid permutations for the remaining unassigned slots. In this phase, the weight of the global arc cost is minimized in favor of local transition smoothness (`C_trans`). This ensures that the final transitions in the sequence are musically flawless and harmonically aligned, while guaranteeing that every track in the original pool is successfully placed exactly once.

---

### 2.7 Computational Representation (Pseudocode)

The following pseudocode represents the algorithmic architecture of the system:

```
Algorithm: Concurrent Bottleneck-Guided Recursive Bidirectional Beam Search

Input: S (Pool), L (Length), s_start, s_end, A_EV(t) (Target Arc), k-Parameters
Output: P (Ordered Playlist)

Initialization:
    P[1] = s_start; P[L] = s_end
    Available_S = S - {s_start, s_end}
    Compute C_arc(s, t) for all s, t
    Compute BS_song(s) and BS_loc(t)
    Active_Intervals = queue{[1, L]}

Main Recursion Loop:
    While Active_Intervals has segments > 3:
        For each [start, end] in parallel:
            M = floor((start + end) / 2)
            Forward_Beams = expand(start to M-1, width=k1, score=StepScore_f)
            Backward_Beams = expand(end to M+1, width=k1, score=StepScore_b)
            Anchor_x = Select track with min AnchorScore at M
            P[M] = Anchor_x; Remove Anchor_x from Available_S
            Enqueue [start, M] and [M, end]

Final Matching Phase:
    For remaining small segments:
        Evaluate permutations to minimize total C_trans
        Fill remaining positions in P

Return P
```

---

## 3. Relationship to Methods Presented in the Background Chapter

### 3.1 Methods Presented in the Background That We Chose Not to Use

#### 3.1.1 Greedy Algorithms

Greedy algorithms were presented in the background chapter as a simple and fast approach that selects, at each step, the transition with the lowest local cost. We rejected them as a primary method due to the well-documented Leftovers Problem. Bittner et al. (2017) explicitly noted that greedy sequencing (HAM-1, HAM-2) leaves hard-to-fit tracks for the tail of the playlist, creating musically disjointed conclusions. In our system, the greedy approach will serve exclusively as a baseline in the comparative evaluation experiment, to empirically demonstrate that our method resolves the problem greedy algorithms inherently create.

#### 3.1.2 Constraint Satisfaction Programming (CSP)

CSP was presented as a method that allows defining structural constraints on the playlist, such as a gradual energy build-up, a fixed sequence length, or harmonic compatibility requirements. We chose not to adopt it as our primary framework because hard, pre-defined constraints do not provide a sufficiently precise response to our problem. The balance between following the target energy arc, ensuring smooth musical transitions, and placing difficult tracks optimally is dynamic and continuous in nature, not binary. Interestingly, our approach can be understood as a form of **soft CSP**: rather than enforcing rigid boolean constraints, we implement a continuous numerical cost function with dynamically adjusted weights depending on the algorithmic phase.

#### 3.1.3 Reinforcement Learning (RL)

Reinforcement Learning (RL) frameworks, such as DJ-MC (Liebman et al., 2015) and Spotify's AH-DQN (Tomasi et al., 2023), have been proposed for music sequencing, where an agent learns to select tracks dynamically based on the cumulative state of the sequence. We rejected this approach for our project primarily due to a misalignment in application. RL models are fundamentally designed for interactive environments that rely on continuous, iterative user feedback in real time to update their policy. Conversely, our project focuses on an offline sequencing task, where a predefined, static playlist must be optimized in a single pass without ongoing user interaction. Furthermore, solving the sequential decision-making problem (MDP) under an RL framework introduces immense state-space and computational complexity, which is unnecessary for our objective of generating a direct, calculated layout based purely on the acoustic graph characteristics of the given tracks.

---

### 3.2 Methods We Use and How Exactly We Apply Them

#### 3.2.1 Transition Cost Function (Bittner et al., 2017)

The DJ-transition feature space defined and validated by Bittner et al. (2017) is adopted directly as our micro-level compatibility module. The transition cost between an outgoing track u and an incoming track v is defined as:

```
C_trans(u, v) = α · |log2(BPM_v / BPM_u)| + β · min(|h_u − h_v|, 12 − |h_u − h_v|)
```

The logarithmic BPM term captures the perceptual nature of tempo perception, and the circular Camelot Wheel distance captures harmonic compatibility. As a core innovation and departure from their static framework, we introduce a **dynamic weight mechanism** over this function. Our novel use of this function is that it does not operate uniformly throughout the algorithm: during macro-level phases the cost function is dominated by Arc Cost (distance from the target energy arc), whereas during micro-level phases the weight shifts entirely to `C_trans`, mimicking a human DJ's focus on beat-matching and harmonic smoothness at the moment of transition.

#### 3.2.2 Bidirectional Heuristic Search — Meet-in-the-Middle (Holte et al., 2016)

The structural backbone of our architecture adapts the Meet-in-the-Middle (MM) principle formalized by Holte et al. (2016). In its original formulation, the MM algorithm executes a single, global bidirectional search from fixed starting and goal states until the frontiers collapse at a global cost midpoint. As a key architectural innovation, we depart from this single-stage global execution and instead apply the Meet-in-the-Middle framework **locally and recursively**. At each recursive call, rather than searching across the entire playlist duration, a localized interval `[start, end]` is defined, and two simultaneous searches expand inward exclusively toward the midpoint of that specific sub-segment. This localized adaptation enables the algorithm to firmly preserve the global energy arc through recursively locked anchor points, while independently optimizing local transitions within isolated sub-segments. Finally, whereas the original MM algorithm guarantees mathematical path optimality in explicit graphs, we adapt its bidirectional search pattern to a combinatorial sequencing space governed by musical heuristics and item-uniqueness constraints, where we prioritize heuristic fluidity over a provably shortest path.

#### 3.2.3 Beam Search

Beam Search was introduced in the background chapter as an "RL-lite" alternative (Liebman et al., 2015) for sequential decision making with look ahead. We embed it within every level of the recursive bidirectional search: instead of maintaining a single optimal path at each step, the system keeps **B parallel candidate sequences** (beam width B). This provides diversity and robustness to local cost minima. Our use differs from the standard linear application of Beam Search: it is embedded within each recursion level and directed toward a target anchor point determined by the Bottleneck Prioritization Heuristic described below, rather than scanning linearly from start to end.

#### 3.2.4 Top-K Candidate Filtering (Micro-Level)

For each small interval at the base case of the recursion (typically 3–4 positions), the algorithm maintains a set of up to **K** candidate songs per position that satisfy an energy threshold relative to the target arc. All permutations within each interval are then evaluated and the arrangement minimizing total `C_trans` is selected. To prevent any song from being unplaceable, a fallback mechanism ensures that songs not appearing in any Top-K set are assigned to their K most compatible positions. A constraint prevents scenarios where more positions exist than valid candidates, ensuring the final assignment is always feasible.

---

### 3.3 Innovation: The Bottleneck-Guided Recursive Bidirectional Beam Search

The central innovation of our system is not the invention of an entirely new algorithm, but rather the integration of several well-known methods into a novel architecture that produces results none of them achieve individually. Bidirectional Search, Beam Search, and recursive decomposition are each known independently; their combination for the specific problem of closed-catalog playlist sequencing with a predefined energy arc and fixed start/end tracks constitutes our primary contribution.

#### The Bottleneck Prioritization Heuristic (novel component)

This is the most novel element of our design, absent from the background chapter. Before the search begins, the system pre-computes a Bottleneck Score `BS(s)` for every track s in the pool, defined as the minimum Arc Cost across all possible playlist positions:

```
BS(s) = min_t [ C_arc(s, t) ]
```

Tracks with a high `BS(s)` are classified as bottlenecks — even in their best possible position, they barely fit the desired narrative. At each recursive level, the anchor chosen is not the geometric midpoint of the interval but the optimal position of the most constrained unplaced track. By locking in the most difficult songs first, the algorithm ensures that only the most flexible, easily mixable tracks remain for the final micro-level stages. This constitutes a direct, mathematically-grounded solution to the Leftovers Problem documented in the existing literature.

#### Dynamic Weight Inversion Between Macro and Micro Phases (novel application)

The same cost function infrastructure operates in two distinct modes:
- **During macro-level recursion:** the energy arc component (γ, λ weights) is dominant, ensuring global narrative coherence.
- **During micro-level base-case optimization:** the transition cost component (δ weight) is dominant, ensuring flawless beat-matching and harmonic smoothness.

This dynamic inversion of weights within a single unified cost function, applied differently by phase, was not described in any of the methods surveyed in the background chapter and represents a contribution of our system design.

---

## 4. Evaluation Methodology and Success Criteria

To evaluate our proposed Bottleneck-Guided Recursive Bidirectional Beam Search algorithm, we will compare its performance to a greedy sequencing baseline. The purpose of the evaluation is to determine whether our method can generate playlists that follow a predefined emotional arc, maintain smooth transitions between songs, maximize macro-level sequence coherence, and ensure structural stability at recursive bidirectional midpoints.

### Metric 1: Arc Adherence (RMSE)

Measured using the Root Mean Square Error (RMSE) between the target Energy-Valence arc and the sequence produced by the generated playlist. This metric measures how closely the generated playlist follows the desired emotional progression. Lower RMSE values indicate that the sequence stays closer to the target Energy-Valence arc throughout the playlist.

To compute this metric, we first calculate the Arc Cost between a song and a specific position in the playlist:

```
C_arc(s, t) = |EV_score(s) − A_EV(t)|
```

where s represents a candidate song and t represents a position in the playlist. `EV_score(s)` is the integrated Energy-Valence score of song s, while `A_EV(t)` is the target Energy-Valence value at position t according to the desired emotional arc. A lower Arc Cost means that the song is a better fit for that position in the playlist.

After calculating the Arc Cost for all positions, the overall RMSE is computed as:

```
RMSE = sqrt[ (1/L) * Σ (C_arc(p_t, t))^2 ]
```

where `p_t` is the song assigned to position t and L is the playlist length. RMSE summarizes the Arc Cost values across the entire playlist, providing a single measure of how closely the generated sequence follows the target Energy-Valence arc.

### Metric 2: Transition Smoothness (TS)

This metric evaluates the quality of transitions between consecutive songs throughout the playlist. It is based on two musical properties: tempo similarity and harmonic compatibility. Large BPM changes or transitions between distant musical keys increase the transition cost, while songs with similar tempo and harmonically compatible keys produce lower costs.

The transition cost between two consecutive tracks is defined as:

```
C_trans(u, v) = (α * d_rhythm) + (β * d_tonality) + (γ * d_texture)
```

where u and v represent two consecutive songs in the playlist. The parameters α, β, and γ are weighting factors establishing the optimization priorities across rhythmic, harmonic, and acoustic profiles.

- **Rhythmic Distance (`d_rhythm`):** captures human tempo perception on a base-2 logarithmic scale and appends a static structural penalty for meter variations.
- **Harmonic Distance (`d_tonality`):** employs a 3D Euclidean distance mapping keys and modes across a circle-of-fifths based prism to penalize tonal clashes while maintaining equidistant adjacent steps.
- **Textural Distance (`d_texture`):** measures the geometric distance between explicit acoustic characteristics, including acousticness and instrumentalness, to preserve timbral continuity.

Lower combined transition costs indicate structurally smoother boundaries. (All formulas are explained in the transition module under Section 2.4.)

The total Transition Smoothness score is calculated as:

```
TS = Σ C_trans(p_i, p_{i+1})
```

for all consecutive track pairs in the playlist, where `p_i` and `p_{i+1}` denote neighboring songs in the generated sequence. This metric aggregates the transition costs across the entire playlist and provides a single measure of overall transition quality.

### Metric 3: Global Coherence (COH)

This metric measures the structural narrative and flow of the playlist by analyzing the relationship between local sequencing choices and the overall diversity of the track pool. In many traditional sequencing methods, local choices are made in isolation, leading to a phenomenon where the playlist lacks a cohesive progression or suffers from a "tilting effect" where musical styles shift abruptly. The Global Coherence metric mathematically quantifies whether a playlist flows like a carefully curated set or behaves like a random shuffle.

To quantify this effect, we define the Playlist Coherence Score (COH) using a normalized ratio of sequential variance to population variance:

```
COH = 1 − ( (n * Σ[C_trans(x_i, x_{i+1})^2]) / (2 * Σ[Σ[C_trans(x_i, x_j)^2]]) )
```

In this equation:
- **n** denotes the total number of tracks in the sequence.
- The **numerator** calculates the sequential variance based on the transition cost between consecutive songs `x_i` and `x_{i+1}`.
- The **denominator** calculates the total population variance across all unique track combinations `x_i` and `x_j` in the pool.

The transition cost uses the exact same multi-feature distance function defined in the transition framework (Section 2.4), combining tempo differences, harmonic compatibility, and timbral properties.

This metric evaluates the macro-level consistency of the sequencing.
- A Coherence Score near **0** indicates a completely random or shuffled arrangement where local transitions are no smoother than random track pairs.
- A **positive score (closer to 1)** proves a highly coherent flow, demonstrating that the local variation between adjacent tracks is significantly smaller than the global diversity of the entire pool.

By utilizing Global Coherence as a core metric, we can directly verify if our Concurrent Bottleneck-Guided Recursive Bidirectional Beam Search successfully establishes a unified, structured musical narrative across the entire playlist.

### Metric 4: Anchor Alignment Cost (AC)

This objective metric explicitly evaluates the structural integrity of the midpoints selected during the meet-in-the-middle recursive strategy. Because the algorithm grows sequences bidirectionally from both an initial start song and a final end song to bridge at a targeted midpoint index `M = floor(L/2)`, it is critical to evaluate if these structural junction pillars balance global targeting and local flow without introducing acoustic drift or jarring boundaries.

For any designated midpoint slot M, the Anchor Cost evaluates the candidate song `x_M` against its target position on the reference curve and its immediate neighbor context:

```
AC(x_M) = w_arc * |EV_score(x_M) − A_EV(M)|^2 + w_transition * [C_trans(x_{M−1}, x_M) + C_trans(x_M, x_{M+1})]
```

Here, `w_arc` and `w_transition` represent dynamic balancing weights used during the recursive search layers. By explicitly tracking the Anchor Alignment Cost across the core midpoints of the playlist, we can verify that our concurrent bottleneck-guided selection successfully anchors the playlist's core structure without creating localized bottlenecks where the forward and backward beams meet.

### Metric 5: Human Satisfaction Evaluation (HSS)

While the previous metrics provide objective measures of playlist quality, it is also important to evaluate how the generated playlists are perceived by human listeners. Therefore, a Human Satisfaction Evaluation will be included as a complementary metric.

**Experimental design:**
- Participants will listen to playlists generated by three different methods:
  1. The proposed Bottleneck-Guided Recursive Bidirectional Beam Search algorithm
  2. A Greedy Baseline
  3. A Random Baseline
- The playlists will be presented in a **blind evaluation setting** so that listeners are unaware of which algorithm generated each playlist.
- After listening, participants will rate each playlist on a **five-point Likert scale** according to their overall satisfaction with the listening experience.
- Participants will also be asked to **rank the three playlists** from most preferred to least preferred.

This allows measurement of both overall satisfaction and direct preference between the competing sequencing methods.

The average Human Satisfaction Score (HSS) is defined as:

```
HSS = (1/N) * Σ_{i=1}^{N} r_i
```

where N is the number of participants and `r_i` is the satisfaction rating assigned by participant i.

In addition to the HSS, the ranking results will be analyzed by calculating the **percentage of participants who ranked each playlist generation method in first place**. While HSS measures the average level of satisfaction with a playlist, the ranking analysis captures direct preference between the competing methods. Together, these measures complement the objective metrics by assessing whether improvements in arc adherence, transition smoothness, global coherence, and structural midpoint transitions are perceived by real listeners.

**Success criterion (human evaluation):** The proposed method will be considered successful if it achieves higher HSS values and is ranked first more frequently than both the Greedy Baseline and the Random Baseline.

### Experimental Setup

For the experiment itself, multiple collections of electronic music tracks containing BPM, harmonic key, energy, and valence features will be used. For each collection, a target emotional arc will be defined in advance. Playlists will then be generated using both the greedy baseline and the proposed algorithm under identical conditions. The resulting playlists will be evaluated using the metrics described above.

### Overall Success Criteria

The proposed method will be considered successful if it achieves:
- **Low RMSE values** (strong arc adherence)
- **Low overall transition costs** (smooth transitions)
- **Higher Global Coherence scores**
- **Stabilized Anchor Alignment costs**

compared to the greedy baseline. Achieving these results would indicate that the framework successfully balances global adherence to the desired emotional arc with fine-grained local transition quality, while creating an acoustic narrative that avoids structural fragmentation at the search junctions.

---

## Summary Reference: Key Symbols and Definitions

| Symbol | Meaning |
|---|---|
| `N` | Number of tracks in the pool |
| `L` | Playlist length (number of positions) |
| `s_start`, `s_end` | Designated starting and ending tracks |
| `EV_score(s)` | Composite Energy-Valence score for track s |
| `A_EV(t)` | Target Energy-Valence value at position t |
| `progress(t)` | Normalized relative progress at position t, `(t-1)/(L-1)` |
| `C_arc(s, t)` | Arc Cost: distance between track s's EV score and target arc value at position t |
| `BS_song(s)` | Song Bottleneck Score: average of k_5 lowest C_arc values for track s |
| `BS_loc(t)` | Location Bottleneck Score: average of k_2 lowest C_arc values for position t |
| `C_trans(u, v)` | Transition Cost between outgoing track u and incoming track v |
| `d_rhythm` | Rhythmic distance component (tempo + meter mismatch penalty) |
| `d_tonality` | Harmonic distance component (key/mode distance) |
| `d_texture` | Textural distance component (acoustic feature vector distance) |
| `α, β, γ` | Weighting factors for d_rhythm, d_tonality, d_texture in C_trans |
| `M` | Midpoint/anchor position of a recursive interval, `floor((start+end)/2)` |
| `StepScore(x, prev, t)` | Composite heuristic score during bidirectional search |
| `a, b, c` | Weights in StepScore for C_arc, C_trans, and BS_song respectively |
| `K` | Max number of candidate tracks retained per position in base-case matching |
| `B` | Beam width (number of parallel candidate sequences kept in beam search) |
| `BS(s)` | Bottleneck Score: minimum Arc Cost across all positions for track s |
| `RMSE` | Root Mean Square Error — Arc Adherence metric |
| `TS` | Transition Smoothness — summed transition cost across the playlist |
| `COH` | Global Coherence Score |
| `AC(x_M)` | Anchor Alignment Cost at midpoint M |
| `w_arc, w_transition` | Dynamic balancing weights in Anchor Alignment Cost |
| `HSS` | Human Satisfaction Score (average Likert rating) |

---

## Key Cited Methods/References (from Section 3)

- **Bittner et al. (2017)** — DJ-transition feature space; basis for the transition cost function (`C_trans`); also documented the Leftovers Problem in greedy sequencing (HAM-1, HAM-2).
- **Holte et al. (2016)** — Meet-in-the-Middle (MM) bidirectional heuristic search algorithm; basis for the recursive bidirectional search core engine.
- **Liebman et al. (2015)** — DJ-MC; Reinforcement Learning framework for music sequencing (considered and rejected); also introduced Beam Search as an "RL-lite" alternative.
- **Tomasi et al. (2023)** — Spotify's AH-DQN; another RL framework for sequencing (considered and rejected).
