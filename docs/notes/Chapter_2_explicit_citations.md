# Chapter 2: Background & Literature Review

## Optimizing Playlists Flow and Sequencing
**A model to find the best path across a set of songs while maintaining a predefined "energetic arc" and ensuring high quality local transitions**

**Authors:** Yuval Antman, Ohad Motola, Shelly Kritsberg, Tomer Filo

---

## Gap Analysis & Proposed Solution: What We Do Differently

Most traditional music recommendation systems (like Collaborative Filtering or Content-Based Filtering) focus on recommending individual tracks based on semantic similarity or past user interactions, often ignoring the temporal context and the holistic listening experience. While automated DJ systems and playlist generators have been explored in the past, they typically suffer from fundamental limitations: they either focus strictly on local transition mechanics without a long-term structural goal, or they rely on generic Reinforcement Learning (RL) aiming to minimize track skips.

As Spotify Researchers (Tomasi et al. 2023, Abstract section) note, conventional techniques frequently suffer from a **"misalignment between offline model objectives and online user satisfaction metrics"** [1], failing to guarantee a cohesive musical narrative.

Our project introduces a significant innovation by formulating playlist generation not as a static, computationally heavy graph-solving problem, but as a **dynamic, probabilistic sequential selection process (Context-Aware Random Walk) guided by Beam Search lookahead**. Rather than solving an entire track pool upfront (like we initially thought would be the correct approach), our system computes real-time transition costs, incorporating acoustic and harmonic disparities (using the Camelot Wheel), relative to a predefined global **"Energy Arc"**. By sampling from **Top-K candidates** at each step rather than acting purely greedily, our system injects controlled randomness (Novelty/Diversity) while using lookahead planning to ensure global coherence.

---

## 1. In-Depth Understanding of Existing Methods and Tools

The challenge of automated playlist generation has been approached from several distinct angles in the Music Information Retrieval (MIR) and Recommender Systems (RS) domains.

### Similarity-Based and Sequence Models

Early research primarily relied on audio similarity and metadata to generate playlists.

- **Bonnin et al. (2014) [2]** — mentions the use of GPS location of users as metadata *(page 8, in the table of additional information used)*.
- **Flexer et al. (2008) [3]** — demonstrated the generation of playlists by defining a start and end song, using spectral similarity (specifically, **Kullback-Leibler divergence between Mel Frequency Cepstrum Coefficients, or MFCCs**) to create a smooth, interpolating transition between the two anchor points.

However, relying solely on similarity often results in overly homogeneous playlists that lack the dynamic variation required for an engaging listening experience *(Bonnin et al. (2014) [2], page 8, under section 4.1)*. Also, they discuss generating a playlist, whereas we discuss the situation where a playlist is given and we want to find fitting energetic sequences within it.

Moreover, **Flexer et al. [3]** *(page 177, under Table 8)* note that finding a proper transition can fail if the database lacks tracks that fit logically between two highly contrasting anchor songs.

### Graph Theory and Combinatorial Optimization

A more sophisticated approach models track sequencing as a graph traversal problem.

- **Bittner et al. (2017) [4]** *(page 3, under section 3.1.2)* — framed the optimal ordering of a playlist as finding the **Shortest Hamiltonian Path** in a complete weighted graph. In their model, edge weights are determined by Euclidean distances in a feature space that includes tempo (mapped logarithmically), key (mapped via the circle of fifths), and timbre *(section 3.1.1 in Bittner et al.)*.

Because finding an exact Hamiltonian path is **NP-complete**, greedy algorithms (e.g., HAM-1, HAM-2) and Traveling Salesman Problem (TSP) approximations are widely used to minimize overall transition costs. By combining this macro-level sequencing with micro-level cue point selection at major structural boundaries, they demonstrated that beat-aligned transitions strongly enhance listener satisfaction and mask otherwise abrupt changes *(under section 4, Transitions)*.

However, a major vulnerability in myopic greedy selection is what is known as the **"Greedy Trap"** (or **"leftovers" problem**). Bittner et al. explicitly documented this limitation, noting that "an undesirable artifact is the presence of poor track pairings at the tail of the sequencing for HAM-1" [4]. This highlights the strict necessity for algorithms that can look ahead rather than just taking the immediate lowest cost.

### Reinforcement Learning (RL) and Markov Decision Processes (MDP)

RL has been utilized to adapt to listener preferences dynamically.

- **Liebman et al. (2015) [5]** *(page 591, in their Abstract and Part 2, Reinforcement Learning Framework)* — introduced **DJ-MC**, an agent that models playlist recommendation as an episodic MDP, learning user preferences over both individual songs and song-to-song transitions by representing songs through rich spectral and rhythmic descriptors *(under section 3, Modeling)*.
- **Spotify's AH-DQN** — Recently, Spotify developed an **Action-Head Deep Q-Network (AH-DQN)** trained in a simulated environment to optimize user satisfaction metrics (like completion rates) by sequentially selecting tracks [1] *(under the Abstract section on page 4948, and "contribution" in the Introduction section on page 4949)*.

These models highlight the absolute importance of sequential awareness, proving mathematically that the pleasure derived from a song is directly affected by its relative position in a sequence *(Liebman et al., under the Introduction section)*.

While deep RL can be complex to train, planning algorithms derived from this domain (such as **Beam Search** or **Monte Carlo Tree Search**) provide powerful **"RL-lite"** alternatives *(Liebman et al., under section 5, DJ-MC)*. They evaluate partial sequences and look several steps ahead to avoid dead ends, proving highly effective for sequential decision making.

### DJ Automation and Intra-Track Analysis

Creating a seamless mix requires analyzing the internal structure of songs to find "switch points" or cue points.

- **Zehren et al. (2022) [6]** *(under sections Rule-Based Approach and Novelty Detection, pages 70–76)* — used expert DJ rules and statistical novelty detection algorithms (like **Foote's checkerboard kernel**) to identify structural boundaries, such as downbeats at the start of a musical period, suitable for crossfading.
- **Vande Veire and De Bie (2018) [7]** — built a complete open-source auto-DJ system for **Drum and Bass** that automates beat-matching, downbeat tracking, and applies specific crossfade profiles based on structural segmentation *(as explained in their Abstract and methodology sections — 4 in total, and specifically sections 4.3 and 4.1, on pages 1 and 11–17)*.

---

## 2. Intelligent Comparison: Evolution and Differences Between Methods

The evolution of playlist generation moves from static similarity retrieval to sequential optimization, and recently toward adaptive learning (RL).

### Graph/Optimization vs. Reinforcement Learning

Graph-based approaches (like TSP or Constraint Satisfaction Programming) are highly explainable and allow developers to enforce strict musical rules, such as preventing dissonant harmonic clashes or mandating a specific tempo trajectory [4] *(specifying what we talk about on page 1, in the Introduction and Related Work sections, and then on page 5, in sections 4.2 and 4.3, and page 6, Conclusions)*. They also naturally support the incorporation of explicit mathematical constraints, which is highly beneficial when crafting a deliberate musical journey.

However, calculating the cost matrix for large catalogs is computationally expensive — **O(N²)** *(Bittner et al., page 3, section 3.1.2)*.

In contrast, RL models (like DJ-MC [5]) excel at personalizing content dynamically based on user feedback (skips/listens) [1][5] *(in Liebman et al., mentioned on page 598, figure 4 explanation and section 9, and on page 595, section 5.3; in Tomasi et al., on page 4951, section 4.1)*. Yet, RL models often act myopically (nearsighted) and struggle to enforce strict long-term structural narratives, such as the classic **"Hero's Journey"** or **"Double Peak"** energy arcs used by professional DJs to prevent listener ear fatigue.

By adopting a dynamic pipeline, our system calculates cost functions in real-time, allowing the track pool to be updated on the fly while following the "Energy Arc" and having the start and end songs as context. This mirrors a real live DJ environment, where a DJ responds to the crowd's energy sequentially rather than having the entire night perfectly mapped out in advance.

### Inter-track vs. Intra-track Optimization

While traditional Recommender Systems focus entirely on **inter-track mixability** (which song should play next), automated DJ research highlights the absolute necessity of **intra-track mixability** (where exactly the crossfade should occur).

Zehren et al. mention the other sources we cite as examples for this case [6] *(page 69, under Related Work)*, noting that switch points profoundly impact the continuous listening experience *(under Abstract)*. Studies show that misaligned beats or transitioning during vocals are primary causes of perceived poor quality in automated mixes, alongside incompatible timbral shifts — **Bittner et al. (2017) [4]** *(pages 5–6, under section 4.3, figure 5, figure 6, table 2, table 3, and the conclusions section)*.

### Optimization vs. Diversity

Greedy algorithms maximize mathematical similarity but destroy musical novelty. By incorporating a probabilistic Random Walk that samples from the **Top-K lowest-cost tracks**, developers can balance optimization with critical musical diversity, preventing the algorithm from getting stuck in repetitive loops.

---

## 3. Comparing Existing Solutions to Our Proposed Product

Existing commercial and academic systems largely fail to integrate the macro-level planning of a DJ set with the micro-level execution of smooth audio transitions. For example, Spotify's default shuffle or basic recommendation algorithms can create jarring, disjointed experiences because they evaluate tracks independently.

While advanced RL models aim to maximize listen time [1] *(Tomasi et al., page 4952, under Reward and User Simulator, and page 4954, under section 5.2)*, they do not explicitly plan an **"Energy Arc"** (e.g., Ramp-Up for a warmup set, or Rollercoaster for a peak-time party).

Our proposed product tackles this directly through a **Context-Aware Random Walk with Beam Search**. At each time step *t*, the system computes a cost function *J* for all remaining tracks in the pool, evaluating their harmonic compatibility and proximity to the target energy curve.

To prevent the **"Greedy Trap"** observed in previous literature [4] *(Bittner et al., under section 3.1.2)*, we utilize **Bi-directional Beam Search** to evaluate the next several tracks (lookahead paths) before making a selection, choosing the shortest path among future trajectories.

We will rigorously measure the success of this model using two primary metrics:

1. **The convergence to the energy arc** (measured via RMSE between the target curve and the generated sequence), and
2. **Sequential Coherence.**

As **Schweiger et al. (2025) [8]** *(Abstract and Introduction)* highlight, traditional coherence metrics are flawed; therefore, we will apply their formal definition for measuring playlist coherence "based on the sequential ordering of tracks", mathematically evaluating the order-dependent smoothness of BPM and harmonic transitions over time.

---

## 4. Relevant Papers, Tools, and Resources for Citation & Extraction

To build and evaluate our system, we would utilize the following existing tools and datasets:

### Audio Feature Extraction Tools
*(for future work and not the first part of the research we are doing in the scope of the course)*

- **Librosa** — The standard Python library for Music Information Retrieval (MIR). Excellent for extracting Mel-spectrograms, MFCCs, beat tracking, and onset detection.
- **Essentia** — A highly optimized C++ library with Python bindings. It is heavily recommended over Librosa for processing large datasets quickly (up to twice as fast for tasks like FFT) and extracting advanced psychoacoustic features.

### Open Datasets
*(what we will actually use in our project)*

- **FMA (Free Music Archive) & MTG-Jamendo** — Due to recent closures of Spotify's Audio Features API, these open datasets are critical. They provide hundreds of thousands of tracks with pre-computed acoustic features, tags, and completely open audio files suitable for rendering crossfades.
- **Million Playlist Dataset (MPD)** — Useful for analyzing human curation patterns and sequence coherence. Also mentioned to be used in Schweiger et al. (2025) [8] research on user-curated playlists.
- **UnmixDB** — Contains ground-truth annotations of DJ mixes, cue points, and beat tracking, which is essential if you plan to implement automatic crossfading and switch-point detection. Also mentioned to be used in Zehren et al. (2022) [6].

---

## Works Cited

| # | Citation |
|---|---|
| [1] | F. Tomasi, J. Cauteruccio, S. Kanoria, K. Ciosek, M. Rinaldi and A. Z. Dai, "Automatic Music Playlist Generation via Simulation-based Reinforcement Learning," 2023. |
| [2] | G. Bonnin and D. Jannach, "Automated generation of music playlists: Survey and experiments," 2014. |
| [3] | A. Flexer, D. Schnitzer, M. Gasser and G. Widmer, "Playlist Generation Using Start and End Songs," 2008. |
| [4] | R. M. Bittner, M. Gu, G. Hernandez, E. J. Humphrey, T. Jehan, P. H. McCurry and N. Montecchio, "Automatic Playlist Sequencing and Transitions," 2016. |
| [5] | E. Liebman, M. Saar-Tsechansky and P. Stone, "DJ-MC: A Reinforcement-Learning Agent for Music Playlist Recommendation," 2015. |
| [6] | M. Zehren, M. Alunno and P. Bientinesi, "Automatic Detection of Cue Points for the Emulation of DJ Mixing," 2022. |
| [7] | L. V. Veire and T. D. Bie, "From raw audio to a seamless mix: creating an automated DJ system for Drum and Bass," 2018. |
| [8] | H. Schweiger, E. Parada-Cabaleiro and M. Schedl, "The impact of playlist characteristics on coherence in user-curated music playlists," 2025. |

---

## Summary Reference: Key Concepts by Source

| Source | Core Contribution | Key Method/Concept |
|---|---|---|
| Tomasi et al. (2023) [1] | Spotify AH-DQN; simulation-based RL for playlist generation | Action-Head Deep Q-Network; reward/user simulator; offline-vs-online objective misalignment |
| Bonnin & Jannach (2014) [2] | Survey of automated playlist generation methods | Use of metadata (incl. GPS location); critique of over-homogeneous playlists |
| Flexer et al. (2008) [3] | Playlist generation between fixed start/end songs | Spectral similarity via KL divergence between MFCCs |
| Bittner et al. (2017/2016) [4] | Automatic playlist sequencing and transitions | Shortest Hamiltonian Path framing; HAM-1/HAM-2 greedy algorithms; "Greedy Trap"/leftovers problem; beat-aligned transitions |
| Liebman et al. (2015) [5] | DJ-MC reinforcement learning agent | Episodic MDP; spectral/rhythmic song descriptors; Beam Search / Monte Carlo Tree Search as "RL-lite" |
| Zehren et al. (2022) [6] | Cue point detection for DJ mixing emulation | Rule-based approach; novelty detection (Foote's checkerboard kernel); UnmixDB dataset |
| Vande Veire & De Bie (2018) [7] | Open-source auto-DJ system for Drum and Bass | Beat-matching, downbeat tracking, structural segmentation-based crossfade profiles |
| Schweiger et al. (2025) [8] | Coherence in user-curated playlists | Formal sequential-ordering-based coherence metric; Million Playlist Dataset (MPD) |

---

## Key Terms Glossary

| Term | Definition (as used in this chapter) |
|---|---|
| **Energy Arc** | A predefined target trajectory of energy/intensity across a playlist (e.g., Ramp-Up, Rollercoaster/Double Peak). |
| **Camelot Wheel** | A harmonic mixing notation system used to assess key compatibility between tracks. |
| **Context-Aware Random Walk** | Our proposed dynamic, probabilistic sequential track-selection process guided by lookahead. |
| **Beam Search** | A lookahead search strategy that evaluates multiple partial sequences ("beams") at once rather than committing greedily at each step. |
| **Top-K Sampling** | Selecting candidates probabilistically from the K lowest-cost options at each step, rather than always picking the single lowest-cost option, to preserve novelty/diversity. |
| **Greedy Trap / Leftovers Problem** | The failure mode where greedy, locally-optimal track selection leaves poorly-fitting tracks stranded at the end of a sequence. |
| **Shortest Hamiltonian Path** | A graph-theory framing of playlist sequencing where the goal is to visit every track (node) exactly once at minimum total transition cost. |
| **MFCC (Mel Frequency Cepstrum Coefficients)** | Spectral audio features used to measure timbral/spectral similarity between tracks. |
| **Cue Points / Switch Points** | Specific intra-track timestamps identified as optimal points for crossfading or transitioning between songs. |
| **AH-DQN (Action-Head Deep Q-Network)** | Spotify's deep reinforcement learning architecture for sequential playlist track selection. |
