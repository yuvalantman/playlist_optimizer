# Playlist Optimizer - Project Structure and Algorithm Flow

## 1. High-Level Goal

This project builds a complete ordered playlist from a fixed pool of Spotify tracks.

The playlist must satisfy a closed-pool constraint:

- Every track in the selected pool appears exactly once.
- No track is duplicated.
- No track is missing.
- The first track is fixed.
- The last track is fixed.

The playlist should also satisfy musical and structural goals:

- It should follow a global energy/valence trajectory from the selected start track to the selected end track.
- Consecutive songs should have coherent transitions.
- Rhythm, tonality, texture, Camelot compatibility, and audio-feature similarity should influence transition cost.
- The system compares a simple Greedy baseline against BIBS, the main recursive search algorithm.

The project is built around two central cost matrices:

- `C_arc`: track-position cost. It measures how well a track fits a playlist position.
- `C_trans`: track-to-track transition cost. It measures how smooth or compatible it is to transition from one track into another.

BIBS tries to combine:

- global arc structure from `C_arc`
- local transition quality from `C_trans`
- bottleneck handling for difficult-to-place tracks
- candidate pruning through a transition graph and candidate orchestrator

The main question the code answers is:

```text
Given N tracks, a fixed start, and a fixed end,
what ordering uses every track exactly once while balancing arc fit and transition quality?
```

## 2. Developer Quick Start

### Install Dependencies

From the project root:

```powershell
pip install -r requirements.txt
```

If you use a virtual environment, activate it first.

### Run the Main Pipeline

From the project root:

```powershell
python main.py
```

### Required Input File

The main input dataset is:

```text
data/spotify_tracks.csv
```

The file must contain the columns used throughout the project, including track metadata, tempo, audio features, and harmonic features such as key/mode or Camelot fields.

### Output Locations

Generated outputs are written under:

```text
outputs/playlists/
outputs/results/
```

The output writer saves:

- ordered Greedy playlist CSVs
- ordered BIBS playlist CSVs
- metric comparison CSVs

Some experiment utilities also create additional result files.

### Main Run Settings

The primary run settings are defined near the top of `main.py`:

```python
RUN_EXPERIMENTS = False
RUN_BIBS_SWEEP = False
POOL_SIZE = 150
EXPERIMENT_POOL_SIZE = 150
GENRE = "pop"
RANDOM_SEED = 42
```

Change these in `main.py` when you want to run a different pool size, genre, or random seed.

## 3. Directory Structure

Current important project files and folders:

```text
playlist_optimizer/
  main.py
  README.md
  requirements.txt
  PROJECT_STRUCTURE_AND_FLOW.md
  data/
    spotify_tracks.csv
  outputs/
    playlists/
    results/
  sources/
    research/source PDFs
  src/
    __init__.py
    bibs.py
    bibs_config_sweep.py
    bottleneck_detector.py
    candidate_orchestrator.py
    cost_functions.py
    data_preparation.py
    evaluator.py
    experiment_runner.py
    feature_engineering.py
    greedy_baseline.py
    output_writer.py
    transition_graph.py
```

### `main.py`

Responsible for:

- Running the full single-run pipeline.
- Loading and preparing the track pool.
- Computing `EV_score`, `C_arc`, and `C_trans`.
- Selecting fixed start/end tracks.
- Detecting bottlenecks.
- Building the transition graph.
- Running Greedy.
- Running BIBS.
- Evaluating both playlists.
- Printing the active diagnostic dashboard.
- Saving playlist and metric outputs.

Main inputs:

- `data/spotify_tracks.csv`
- constants at the top of `main.py`
- configuration objects from `src/`

Main outputs:

- `greedy_playlist`
- `bibs_playlist`
- metric dictionaries
- diagnostics dictionaries
- optional CSV outputs

Important data structures created:

- `track_pool`: cleaned sampled track DataFrame
- `c_trans`: transition cost matrix
- `target_arc`: desired EV arc
- `c_arc`: track-position arc cost matrix
- `bottleneck_results`: bottleneck detector output dictionary
- `graph_data`: `TransitionGraphData`
- `orchestrator`: `CandidateOrchestrator`

Where it is called from:

- It is the command-line entrypoint. Run it with `python main.py`.

Developer changes:

- Change pool size, genre, and seed here.
- Add or remove experiment dashboards here.
- Do not put core algorithm logic here unless it is explicitly experimental or diagnostic.

### `data/`

Contains input datasets.

Important file:

- `data/spotify_tracks.csv`

Responsible for:

- Providing the source track metadata and audio features.

Expected by:

- `src/data_preparation.py`

Developer changes:

- Replace or update the CSV to run on a different dataset.
- Make sure required columns remain available.

### `outputs/`

Contains generated output files.

Important subfolders:

- `outputs/playlists/`
- `outputs/results/`

Responsible for:

- Storing generated playlist CSVs.
- Storing metrics, experiment results, and diagnostics.

Created/used by:

- `src/output_writer.py`
- `main.py`
- experiment utilities

Developer changes:

- Change output naming or saved columns in `src/output_writer.py`.

### `src/`

Contains the implementation modules.

The core modules are:

- `src/data_preparation.py`
- `src/feature_engineering.py`
- `src/cost_functions.py`
- `src/transition_graph.py`
- `src/bottleneck_detector.py`
- `src/candidate_orchestrator.py`
- `src/greedy_baseline.py`
- `src/bibs.py`
- `src/evaluator.py`
- `src/output_writer.py`

## 4. Main Execution Flow

The normal single-run pipeline in `main.py` is:

```text
main.py
-> load tracks
-> clean/preprocess data
-> build fixed track pool
-> compute EV_score
-> compute C_trans
-> select/fix start and end tracks
-> compute target energy/valence arc
-> compute C_arc
-> detect bottlenecks
-> build transition graph
-> create candidate orchestrator
-> run Greedy baseline
-> run BIBS
-> evaluate both playlists
-> print diagnostics
-> export results if enabled
```

### Step 1: Load Tracks

What happens:

- `TrackPoolBuilder` loads `data/spotify_tracks.csv`.
- It checks that required columns exist.

Responsible module:

- `src/data_preparation.py`

Function/class:

- `TrackPoolBuilder.load()`

Receives:

- CSV path from `TrackPoolConfig`

Returns:

- raw pandas DataFrame

Passed forward:

- DataFrame to `TrackPoolBuilder.build()`

### Step 2: Build Fixed Track Pool

What happens:

- Filters by genre.
- Converts tempo to numeric.
- Removes invalid tempo rows.
- Removes duplicate track IDs if configured.
- Removes duplicate track-name/artist pairs if configured.
- Samples exactly `POOL_SIZE` tracks using `RANDOM_SEED`.
- Resets DataFrame index.

Responsible module:

- `src/data_preparation.py`

Function/class:

- `TrackPoolBuilder.build()`

Receives:

- `TrackPoolConfig`

Returns:

- `track_pool`

Configuration values:

- `csv_path`
- `genre`
- `pool_size`
- `random_seed`
- `min_tempo`
- `max_tempo`
- `remove_duplicate_track_ids`
- `remove_duplicate_songs`

Developer changes:

- Change filtering and sampling rules here.
- Add new validity filters here if they are about track pool construction.

### Step 3: Compute `EV_score`

What happens:

- Audio features are normalized.
- A weighted average creates one `EV_score` per track.

Responsible module:

- `src/feature_engineering.py`

Function:

- `compute_ev_score(df, config)`

Receives:

- `track_pool`
- `EnergyValenceConfig`

Returns:

- copy of `track_pool` with an `EV_score` column

Configuration values:

- `energy_weight`
- `valence_weight`
- `danceability_weight`
- `loudness_weight`
- `tempo_weight`

Developer changes:

- Change EV-score definition here.
- Add/remove features from the EV score here.

### Step 4: Compute `C_trans`

What happens:

- Computes pairwise transition cost for every ordered source-target pair.
- Combines rhythm, tonality, and texture.

Responsible module:

- `src/cost_functions.py`

Function:

- `compute_transition_cost_matrix(df, alpha=1.0, beta=0.4, gamma=0.6, ...)`

Receives:

- track pool DataFrame

Returns:

- square NumPy matrix `c_trans`

Passed to:

- start/end selection
- transition graph builder
- Greedy
- BIBS
- evaluator

Developer changes:

- Change transition formula in `src/cost_functions.py`.
- Keep `compute_transition_component_matrices()` consistent with `compute_transition_cost_matrix()`.

### Step 5: Select Start and End Tracks

What happens:

- Selects a lower-EV start track.
- Selects a higher-EV end track.
- Enforces a minimum EV gap.
- Uses transition potential when `C_trans` is available.

Responsible module:

- `src/feature_engineering.py`

Function:

- `select_start_end_tracks(df, c_trans, config)`

Receives:

- track pool with `EV_score`
- optional `c_trans`
- `StartEndSelectionConfig`

Returns:

- `(start_index, end_index)`

Configuration values:

- `min_ev_gap`
- `preferred_ev_gap`
- `start_quantile`
- `end_quantile`
- `transition_potential_k`
- `transition_potential_weight`
- `random_seed`

Developer changes:

- Change endpoint policy here.

### Step 6: Compute Target Arc and `C_arc`

What happens:

- Builds a linear target arc from start EV to end EV.
- Computes absolute deviation of every track from every playlist position.

Responsible module:

- `src/feature_engineering.py`

Functions:

- `create_target_arc_from_tracks(df, start_index, end_index)`
- `compute_arc_cost_matrix(df, target_arc)`

Receives:

- track pool with `EV_score`
- fixed start/end indices
- target arc

Returns:

- `target_arc`
- `c_arc`

Main data structure:

```text
C_arc[track_index, playlist_position]
```

Lower means the track better fits that position.

Developer changes:

- Change the target arc shape here.
- Change arc cost definition here.

### Step 7: Detect Bottlenecks

What happens:

- Scores tracks by how hard they are to place on the arc.
- Scores positions by how hard they are to fill.
- Builds candidate sets for each playlist position.

Responsible module:

- `src/bottleneck_detector.py`

Function/class:

- `BottleneckDetector.detect(c_arc)`

Receives:

- `c_arc`

Returns:

- `bottleneck_results`

Important keys:

- `song_bottleneck_scores`
- `location_bottleneck_scores`
- `bottleneck_track_indices`
- `candidate_sets`

Developer changes:

- Change bottleneck scoring here.
- Change candidate-set size here.

### Step 8: Build Transition Graph

What happens:

- Builds a sparse directed graph from `C_trans`.
- Low-cost transition edges are kept.
- Fallback edges are added to guarantee minimum degree.

Responsible module:

- `src/transition_graph.py`

Function/class:

- `TransitionGraphBuilder.build_transition_graph_data(c_trans)`

Receives:

- `c_trans`
- optional tonality/rhythm constraint matrices

Returns:

- `TransitionGraphData`

Important fields:

- `outgoing_neighbors`
- `incoming_neighbors`
- `outgoing_neighbors_with_costs`
- `incoming_neighbors_with_costs`
- `edge_metadata`
- `diagnostics`

Developer changes:

- Change thresholding, min degree, max degree, or fallback behavior here.

### Step 9: Create Candidate Orchestrator

What happens:

- Creates an object that can build candidate sets for different BIBS decision types.

Responsible module:

- `src/candidate_orchestrator.py`

Class:

- `CandidateOrchestrator`

Receives:

- `CandidateOrchestratorConfig`

Returns:

- candidate lists during BIBS execution

Developer changes:

- Change candidate exposure here.
- Tune which sources are prioritized here.

### Step 10: Run Greedy

What happens:

- Greedy starts at `start_index`.
- Repeatedly picks the lowest local-cost next track.
- Holds `end_index` until the final position.

Responsible module:

- `src/greedy_baseline.py`

Function/class:

- `GreedyPlaylistBaseline.generate(...)`

Receives:

- `c_arc`
- `c_trans`
- `start_index`
- `end_index`
- optional transition graph

Returns:

- `greedy_playlist`

Developer changes:

- Change local greedy score here.
- Change graph use behavior here.

### Step 11: Run BIBS

What happens:

- BIBS recursively splits the playlist interval.
- Selects midpoint anchors.
- Expands forward/backward beams.
- Solves small intervals by local permutation.

Responsible module:

- `src/bibs.py`

Function/class:

- `BIBS.generate(...)`

Receives:

- `c_arc`
- `c_trans`
- `bottleneck_results`
- `graph_data`
- `candidate_orchestrator`
- `target_arc`
- `track_pool`
- `start_index`
- `end_index`

Returns:

- `bibs_playlist`

Developer changes:

- Change recursive structure, beam scoring, anchor selection, or base-case logic here.

### Step 12: Evaluate

What happens:

- Validates exact-once playlists.
- Computes official metrics.

Responsible module:

- `src/evaluator.py`

Function/class:

- `PlaylistEvaluator.evaluate(playlist, c_arc, c_trans)`

Returns:

- metric dictionary

Metrics:

- `arc_rmse`: lower is better
- `total_transition_cost`: lower is better
- `average_transition_cost`: lower is better
- `global_coherence`: higher is better

Developer changes:

- Add or change metrics here.

### Step 13: Export Results

What happens:

- Saves playlists and metric comparisons to CSV.

Responsible module:

- `src/output_writer.py`

Function/class:

- `OutputWriter.save_single_run_outputs(...)`

Returns:

- dictionary of resolved output paths

Developer changes:

- Change output columns or file naming here.

## 5. Core Modules

### `src/data_preparation.py`

Purpose:

- Build a fixed, reproducible track pool from the Spotify CSV.

Main classes:

#### `TrackPoolConfig`

Configuration dataclass.

Fields:

- `csv_path`
- `genre`
- `pool_size`
- `random_seed`
- `min_tempo`
- `max_tempo`
- `remove_duplicate_track_ids`
- `remove_duplicate_songs`

#### `TrackPoolBuilder`

Builds the pool.

Main methods:

##### `__init__(config)`

Receives:

- `TrackPoolConfig`

Creates/modifies:

- stores config
- validates config

##### `load()`

Receives:

- no runtime arguments

Uses:

- `self.config.csv_path`

Returns:

- raw DataFrame

Validates:

- file exists
- required columns include `track_id`, `track_genre`, `tempo`, `track_name`, `artists`

Called from:

- `TrackPoolBuilder.build()`

##### `build()`

Receives:

- no runtime arguments

Returns:

- cleaned and sampled `track_pool`

Creates/modifies:

- local DataFrame copy
- numeric tempo column
- duplicate-filtered DataFrame
- sampled DataFrame with reset index

Called from:

- `main.py`

Developer improvement points:

- Add additional validity filters.
- Add support for multiple genres.
- Add feature completeness checks.
- Add deterministic sorting before sampling if needed.

### `src/feature_engineering.py`

Purpose:

- Build derived features and arc costs.

Main configuration classes:

#### `EnergyValenceConfig`

Controls feature weights for `EV_score`.

Fields:

- `energy_weight`
- `valence_weight`
- `danceability_weight`
- `loudness_weight`
- `tempo_weight`

#### `StartEndSelectionConfig`

Controls start/end selection.

Fields:

- `min_ev_gap`
- `preferred_ev_gap`
- `start_quantile`
- `end_quantile`
- `transition_potential_k`
- `transition_potential_weight`
- `random_seed`

#### `TempoEnvelopeConfig`

Controls diagnostic tempo envelope margins.

Main functions:

##### `normalize_column(column)`

Receives:

- pandas Series

Returns:

- min-max normalized Series

Used by:

- `normalize_columns()`
- `compute_ev_score()`

##### `normalize_columns(df, columns)`

Receives:

- DataFrame
- list of column names

Returns:

- DataFrame copy with selected columns normalized

##### `compute_ev_score(df, config)`

Receives:

- DataFrame with audio feature columns
- `EnergyValenceConfig`

Returns:

- DataFrame copy with `EV_score`

Creates/modifies:

- normalized intermediate DataFrame
- `EV_score` column

Called from:

- `main.py`

Developer improvement points:

- Change the EV definition.
- Add genre-specific EV weighting.
- Remove tempo from EV if tempo should only affect transition costs.

##### `select_start_end_tracks(df, c_trans, config)`

Receives:

- DataFrame with `EV_score`
- optional `c_trans`
- `StartEndSelectionConfig`

Returns:

- `(start_index, end_index)`

Uses:

- EV quantiles
- minimum EV gap
- preferred EV gap
- transition potential from `c_trans`

Called from:

- `main.py`

Developer improvement points:

- Change endpoint policy.
- Add user-specified start/end.
- Penalize endpoints with poor graph degree.

##### `create_target_arc_from_tracks(df, start_index, end_index)`

Receives:

- DataFrame with `EV_score`
- start index
- end index

Returns:

- linear NumPy target arc

Called from:

- `main.py`

##### `compute_arc_cost_matrix(df, target_arc)`

Receives:

- DataFrame with `EV_score`
- target arc sequence

Returns:

- `c_arc`

How `C_arc` is computed:

```text
C_arc[track, position] = abs(EV_score(track) - target_arc[position])
```

Lower is better.

Called from:

- `main.py`

Developer improvement points:

- Replace absolute distance with squared distance.
- Add asymmetric penalties for energy drops.
- Add tempo arc cost if global tempo shape becomes part of methodology.

### `src/cost_functions.py`

Purpose:

- Compute `C_trans` and transition diagnostics.

Main functions:

##### `tempo_distance(tempo_u, tempo_v)`

Receives:

- two positive tempos

Returns:

- `abs(log2(tempo_v / tempo_u))`

Used by:

- `transition_cost()`
- conceptually by matrix computation

##### `camelot_distance(h_u, h_v)`

Receives:

- two Camelot numbers from 1 to 12

Returns:

- circular distance on the Camelot wheel

Example:

- 12 to 1 has distance 1.

##### `camelot_tonality_distance(sector_u, mode_u, sector_v, mode_v)`

Receives:

- Camelot number and mode for source and target

Returns:

```text
sector distance + mode distance
```

Where:

- sector distance is circular from 0 to 6
- mode distance is 0 for same A/B mode, 1 otherwise

##### `compute_transition_cost_matrix(df, alpha, beta, gamma, meter_penalty, mode_height)`

Receives:

- track pool DataFrame
- weights for rhythm, tonality, texture

Returns:

- `c_trans`

How `C_trans` is computed:

```text
C_trans(u, v)
  = alpha * d_rhythm(u, v)
  + beta * d_tonality(u, v)
  + gamma * d_texture(u, v)
```

Rhythm:

- Uses tempo log-ratio:

```text
abs(log2(tempo_v / tempo_u))
```

- Adds meter penalty when `time_signature` exists and differs.

Tonality:

- If `camelot_number` and `camelot_mode` exist, uses Camelot.
- Else if `key` and `mode` exist, uses circle-of-fifths prism distance.
- Else if only `camelot_number` exists, uses circular Camelot number distance.

Texture:

- Uses available audio features:
  - `danceability`
  - `energy`
  - `loudness`
  - `speechiness`
  - `acousticness`
  - `instrumentalness`
  - `liveness`
  - `valence`

The texture features are normalized before Euclidean distance is computed.

Called from:

- `main.py`

Used by:

- transition graph
- Greedy
- BIBS
- evaluator

##### `compute_transition_component_matrices(df, ...)`

Receives:

- same inputs as `compute_transition_cost_matrix()`

Returns:

- dictionary with component matrices:
  - `rhythm_costs`
  - `tonality_costs`
  - `texture_costs`
  - `weighted_rhythm_costs`
  - `weighted_tonality_costs`
  - `weighted_texture_costs`
  - `total_costs`

Developer improvement points:

- Change transition formula here.
- Keep component-matrix reconstruction equal to `C_trans`.
- Change Camelot priority here if harmonic methodology changes.
- Add new transition components only if all downstream diagnostics are updated.

### `src/transition_graph.py`

Purpose:

- Build sparse directed graph views from dense `C_trans`.

Main classes:

#### `TransitionGraphConfig`

Important fields:

- `threshold_percentile`
- `absolute_threshold`
- `min_out_degree`
- `min_in_degree`
- `max_out_degree`
- `max_in_degree`
- `exclude_self_transitions`
- `add_fallback_edges`
- `top_m_neighbors`
- `max_tonality_distance`
- `max_rhythm_distance`

#### `TransitionGraphData`

Stores:

- `outgoing_neighbors`
- `incoming_neighbors`
- `outgoing_neighbors_with_costs`
- `incoming_neighbors_with_costs`
- `edge_metadata`
- `threshold_used`
- `diagnostics`

#### `TransitionGraphBuilder`

Main methods:

##### `build_transition_graph_data(c_trans, tonality_distances=None, rhythm_distances=None)`

Receives:

- dense `c_trans`
- optional constraint matrices

Returns:

- `TransitionGraphData`

Creates:

- threshold edges
- fallback edges
- outgoing and incoming neighbor dictionaries
- graph diagnostics

How graph construction works:

1. Validate `c_trans`.
2. Build possible directed edges.
3. Compute threshold from percentile or absolute threshold.
4. Add edges whose cost is below threshold.
5. Respect optional rhythm/tonality constraints.
6. Enforce degree caps.
7. Add fallback edges to satisfy minimum in/out degree.
8. Sort neighbor lists by cost.

Called from:

- `main.py`
- BIBS if no graph is provided

Used by:

- Greedy
- CandidateOrchestrator
- BIBS

Developer improvement points:

- Tune sparsity here.
- Change fallback strategy here.
- Add harmonic or rhythm graph constraints here.
- Increase graph recall by changing threshold/min degree/max degree.

### `src/bottleneck_detector.py`

Purpose:

- Identify hard-to-place tracks and hard-to-fill positions.

Main classes:

#### `BottleneckConfig`

Fields:

- `song_k_best`
- `location_k_best`
- `candidate_set_size`
- `bottleneck_percentile`

#### `BottleneckDetector`

Main methods:

##### `compute_song_bottleneck_scores(c_arc)`

Receives:

- `c_arc`

Returns:

- one score per track

How it works:

- For each track, take its `song_k_best` lowest arc costs.
- Average them.
- Higher means harder to place.

##### `compute_location_bottleneck_scores(c_arc)`

Receives:

- `c_arc`

Returns:

- one score per playlist position

How it works:

- For each position, take the `location_k_best` lowest track costs.
- Average them.
- Higher means the position is harder to fill.

##### `get_bottleneck_track_indices(song_scores)`

Receives:

- song bottleneck scores

Returns:

- indices at or above configured percentile

##### `build_candidate_sets(c_arc)`

Receives:

- `c_arc`

Returns:

- dictionary mapping each position to its best arc-fit candidates

##### `detect(c_arc)`

Receives:

- `c_arc`

Returns:

- full bottleneck result dictionary

Called from:

- `main.py`

Used by:

- `CandidateOrchestrator`
- `BIBS`

Developer improvement points:

- Change what "hard to place" means here.
- Tune bottleneck percentile here.
- Tune per-position candidate set size here.

### `src/candidate_orchestrator.py`

Purpose:

- Build candidate pools for BIBS.

Main classes:

#### `CandidateOrchestratorConfig`

Important fields:

- `max_candidates`
- `graph_priority`
- `candidate_set_priority`
- `bottleneck_priority`
- `fallback_priority`
- `use_bottlenecks`
- `use_location_bottlenecks`
- `graph_left_weight`
- `graph_right_weight`
- `bridge_balance_weight`
- `balance_weight`
- `arc_preview_weight`
- `bottleneck_preview_weight`

#### `CandidateOrchestrator`

Main methods:

##### `build_forward_candidates(...)`

Receives:

- last placed track
- target position
- available tracks
- graph data
- bottleneck results
- `c_arc`

Returns:

- candidate track list for forward beam expansion

Candidate sources:

- outgoing graph neighbors
- candidate set for the position
- bottleneck tracks
- arc fallback

##### `build_backward_candidates(...)`

Receives:

- next boundary track
- target position
- available tracks
- graph data
- bottleneck results
- `c_arc`

Returns:

- candidate track list for backward beam expansion

Candidate sources:

- incoming graph neighbors
- candidate set for the position
- bottleneck tracks
- arc fallback

##### `build_anchor_candidates(...)`

Receives:

- left boundary track
- right boundary track
- midpoint
- available tracks
- graph data
- bottleneck results
- `c_arc`
- `c_trans`

Returns:

- candidate anchor tracks

Candidate sources appear to include:

- graph-side candidates
- candidate-set candidates
- bottleneck candidates
- bridge candidates
- fallback candidates

##### `build_base_case_candidates(...)`

Receives:

- interval boundaries
- positions
- available tracks
- graph data
- bottleneck results
- `c_arc`

Returns:

- candidate tracks for local base-case permutation

Called from:

- `src/bibs.py`

Developer improvement points:

- Change candidate exposure here.
- Add or remove candidate sources.
- Increase `max_candidates` if BIBS is missing good options.
- Change bridge candidate logic if interval seams are poor.

### `src/greedy_baseline.py`

Purpose:

- Build a simple exact-once playlist baseline.

Main classes:

#### `GreedyBaselineConfig`

Fields:

- `arc_weight`
- `transition_weight`
- `use_transition_graph`

#### `GreedyPlaylistBaseline`

Main methods:

##### `generate(c_arc, c_trans, start_index=None, end_index=None, transition_graph=None)`

Receives:

- `c_arc`
- `c_trans`
- optional fixed start/end
- optional transition graph

Returns:

- complete playlist list

Creates/modifies:

- `available_tracks` set
- `playlist` list

How Greedy selects the next track:

1. Start with all tracks available.
2. Remove fixed end track until the end.
3. Place fixed start track first.
4. At each next position:
   - if graph is enabled, try available outgoing graph neighbors
   - otherwise use all available tracks
   - score candidates by arc cost plus transition cost
   - choose the minimum score
5. Append fixed end track.

Local score:

```text
score(track, position)
  = arc_weight * C_arc[track, position]
  + transition_weight * C_trans[previous_track, track]
```

Called from:

- `main.py`

Developer improvement points:

- Change local score here.
- Disable or change graph use here.
- Add diagnostics for greedy choices here.

### `src/bibs.py`

Purpose:

- Implement BIBS, the main recursive search algorithm.

Main classes:

#### `BIBSConfig`

Important fields:

- `beam_width`: number of partial beam paths retained
- `beam_length`: chunk size for beam expansion progress
- `base_case_size`: recursion stopping threshold
- `max_recursion_depth`
- `arc_weight`
- `transition_weight`
- `bottleneck_weight`
- `anchor_arc_weight`
- `anchor_transition_weight`
- `anchor_bottleneck_weight`
- `anchor_balance_weight`
- `base_arc_weight`
- `base_transition_weight`
- `base_case_max_candidates`

#### `BIBS`

Main public methods:

##### `generate(...)`

Receives:

- `c_arc`
- `c_trans`
- `bottleneck_results`
- optional `graph_data`
- optional `candidate_orchestrator`
- optional `target_arc`
- `track_pool`
- `start_index`
- `end_index`

Returns:

- complete BIBS playlist

Creates/modifies:

- internal diagnostics
- anchor history
- decision trace
- recursion trace
- beam trace
- anchor selection trace
- base case trace

Called from:

- `main.py`
- experiment/tuning code in `main.py`

##### `get_diagnostics()`

Receives:

- no arguments

Returns:

- dictionary containing counters and trace rows

Important internal methods:

##### `_solve_interval(...)`

Purpose:

- Recursively solve interval `[left_pos, right_pos]`.

Receives:

- playlist list
- interval boundaries
- available tracks
- cost matrices
- bottleneck results
- graph data
- candidate orchestrator
- target arc
- track pool
- recursion depth

Behavior:

- If interval has no empty positions, return.
- If interval is small enough, call `_fill_base_case()`.
- Otherwise:
  - compute midpoint
  - expand forward beam toward midpoint
  - expand backward beam toward midpoint
  - select anchor
  - lock anchor
  - recursively solve left interval
  - recursively solve right interval

##### `_expand_beam(...)`

Purpose:

- Expand partial paths position by position.

Uses:

- `beam_width` to prune surviving paths
- `beam_length` to process positions in chunks
- candidate orchestrator to get possible next tracks

Returns:

- list of `_BeamItem` paths

##### `_select_anchor(...)`

Purpose:

- Choose a midpoint anchor track.

Receives:

- interval boundary positions/tracks
- midpoint
- forward beams
- backward beams
- available tracks
- `c_arc`
- `c_trans`
- bottleneck results
- graph data
- candidate orchestrator

Returns:

- selected anchor
- left neighbor
- right neighbor
- score
- selection details

Anchor score combines:

- anchor arc component
- left transition component
- right transition component
- bottleneck bonus
- balance component
- beam path components


##### `_fill_base_case(...)`

Purpose:

- Solve small intervals by local permutation.

Receives:

- small interval boundaries
- available tracks
- cost matrices
- bottleneck results
- graph data
- candidate orchestrator
- target arc
- track pool

Behavior:

- Build candidate pool.
- Try permutations for empty positions.
- Score each sequence.
- Select best local sequence.
- Assign tracks to playlist.
- Remove assigned tracks from `available_tracks`.

Base-case score:

```text
base_arc_weight * sum(C_arc)
+ base_transition_weight * sum(C_trans along local path)
+ bottleneck bonus
```

Developer improvement points:

- Change recursive splitting in `_solve_interval()`.
- Change beam scoring in `_expand_beam()`.
- Change anchor scoring in `_select_anchor()`.
- Change local permutation scoring in `_base_case_score()`.
- Add final repair logic here if methodology requires it.

### `src/evaluator.py`

Purpose:

- Validate and evaluate playlists.

Main class:

#### `PlaylistEvaluator`

Main methods:

##### `validate_playlist(playlist, number_of_tracks)`

Receives:

- playlist list
- expected number of tracks

Returns:

- nothing

Raises:

- `ValueError` if length is wrong, track indices are invalid, or duplicates exist

Exact-once validation checks:

- playlist length equals number of tracks
- every item is a valid integer track index
- number of unique tracks equals number of tracks

##### `compute_arc_rmse(playlist, c_arc)`

Returns:

- root mean square of assigned `C_arc` costs

Lower is better.

##### `compute_total_transition_cost(playlist, c_trans)`

Returns:

- sum of transition costs along the playlist

Lower is better.

##### `get_transition_costs_by_position(playlist, c_trans)`

Returns:

- list of transition costs for each consecutive playlist pair

##### `compute_global_coherence(playlist, c_trans)`

Returns:

- coherence score relative to the full population of transition costs

Higher is better.

##### `evaluate(playlist, c_arc, c_trans)`

Returns:

- dictionary with:
  - `arc_rmse`
  - `total_transition_cost`
  - `average_transition_cost`
  - `global_coherence`

Called from:

- `main.py`

Developer improvement points:

- Add new metrics here.
- Change metric definitions here.
- Keep metric direction clear: lower or higher.

### `src/output_writer.py`

Purpose:

- Save playlists and metrics to CSV files.

Main class:

#### `OutputWriter`

Main methods:

##### `save_playlist(playlist, track_pool, output_path)`

Receives:

- playlist list
- track pool DataFrame
- output path

Writes:

- ordered playlist CSV

Validates:

- exact length
- no duplicates
- valid track indices

##### `save_metric_comparison(greedy_metrics, bibs_metrics, output_path, anchor_alignment_cost=None)`

Receives:

- Greedy metrics
- BIBS metrics
- output path
- optional anchor alignment cost

Writes:

- metric comparison CSV

##### `save_single_run_outputs(...)`

Receives:

- Greedy playlist
- BIBS playlist
- track pool
- metrics
- run name

Writes:

- Greedy playlist CSV
- BIBS playlist CSV
- metric comparison CSV

Returns:

- dictionary of output paths

Developer improvement points:

- Change saved playlist columns here.
- Change output file naming here.
- Add diagnostic exports here.

### `src/experiment_runner.py`

Purpose:

- Run repeated experiments across genres and seeds.

Main classes:

- `ExperimentConfig`
- `ExperimentRunner`

This appears to support batch evaluation beyond one `main.py` run.

Developer improvement points:

- Change experiment grid behavior here.
- Add aggregate metrics here.

### `src/bibs_config_sweep.py`

Purpose:

- Run BIBS configuration sweeps.

This appears to support tuning different BIBS configurations without changing defaults.

Developer improvement points:

- Add new sweep dimensions here.
- Add ranking logic here.

## 6. How BIBS Works

Simple explanation:

BIBS builds the playlist by repeatedly choosing important midpoint tracks, then recursively filling the intervals on both sides.

Instead of asking "what is the next track?", BIBS asks:

```text
What track should anchor the middle of this interval?
```

Then it solves the left and right halves.

Conceptual flow:

1. Start with the full interval from first position to last position.
2. Find the midpoint.
3. Use forward beam search from the left boundary.
4. Use backward beam search from the right boundary.
5. Choose an anchor for the midpoint.
6. Lock the anchor in place.
7. Recursively solve the left interval.
8. Recursively solve the right interval.
9. When intervals become small, solve them using base-case local permutations.
10. Return a complete playlist with every track exactly once.

Technical flow:

```text
playlist[0] = start_index
playlist[-1] = end_index
available_tracks = all other tracks

solve_interval(0, N - 1):
  if small:
    fill_base_case()
  else:
    midpoint = floor((left + right) / 2)
    forward_beams = expand from left toward midpoint
    backward_beams = expand from right toward midpoint
    anchor = select_anchor(midpoint)
    playlist[midpoint] = anchor
    remove anchor from available_tracks
    solve_interval(left, midpoint)
    solve_interval(midpoint, right)
```

Data moving through BIBS:

- `playlist`: partially filled output list
- `available_tracks`: unplaced tracks
- `C_arc`: track-position fit
- `C_trans`: transition quality
- `graph_data`: sparse transition graph
- `bottleneck_results`: hard-track and candidate-set information
- `candidate_orchestrator`: candidate source combiner
- beam paths: partial paths from each boundary
- anchors: midpoint tracks

### Forward and Backward Beams

Forward beam:

- Starts at the left boundary.
- Expands toward `midpoint - 1`.
- Uses outgoing-style candidates.

Backward beam:

- Starts at the right boundary.
- Expands toward `midpoint + 1`.
- Uses incoming-style candidates.

Each beam item stores:

- path
- used tracks
- last track
- score
- score components

`beam_width`:

- Controls how many partial paths survive after pruning.

`beam_length`:

- Controls chunked expansion progress.

### Anchor Selection

Anchor selection uses:

- candidate anchors
- forward beam frontier
- backward beam frontier
- midpoint arc cost
- transition cost into and out of the anchor
- bottleneck bonus
- balance between left and right seam costs

The selected anchor is locked into the playlist.

### Base Case

When the interval is small:

- BIBS stops recursion.
- It builds a local candidate pool.
- It tries permutations of candidate tracks.
- It chooses the best local ordering.

This keeps small intervals transition-smooth while still considering arc fit.

## 7. How Greedy Works

Greedy builds left to right.

Flow:

1. Start with `start_index`.
2. Remove `end_index` from available tracks until the final position.
3. At each position, choose the next track with the lowest local score.
4. Append `end_index` last.

Local score:

```text
arc_weight * C_arc[track, position]
+ transition_weight * C_trans[previous_track, track]
```

If graph mode is enabled:

- Greedy first checks outgoing graph neighbors from the previous track.
- If any are available, it chooses only among those graph candidates.
- If no graph candidates are available, it falls back to the full available set.

Why Greedy is useful:

- It is simple.
- It is deterministic.
- It is strong locally.
- It gives BIBS a meaningful baseline.

Greedy limitations:

- No recursive planning.
- No midpoint anchors.
- No explicit bottleneck strategy.
- Can make locally good choices that cause later constraints.

## 8. Greedy vs BIBS

### Greedy

- Builds left to right.
- Optimizes one next step at a time.
- Strong local transition quality.
- Uses `C_arc` and `C_trans` directly.
- Can use transition graph neighbors.
- Less global planning.

### BIBS

- Splits the playlist recursively.
- Uses midpoint anchors.
- Uses forward and backward beams.
- Uses bottleneck scores.
- Uses local base-case permutations.
- More global than Greedy.
- More complex and not guaranteed to beat Greedy without tuning.

## 9. Important Data Structures

### Track Pool

A pandas DataFrame where each row is one track.

Important columns include:

- `track_id`
- `track_name`
- `artists`
- `track_genre`
- `tempo`
- `energy`
- `valence`
- `danceability`
- `loudness`
- `key`
- `mode`
- `camelot`
- `camelot_number`
- `camelot_mode`
- `EV_score`

### `C_arc`

A NumPy matrix:

```text
C_arc[track_index, playlist_position]
```

Lower means better fit.

### `C_trans`

A NumPy matrix:

```text
C_trans[source_track, target_track]
```

Lower means smoother transition.

### Transition Graph

Sparse graph built from `C_trans`.

Contains:

- outgoing neighbors
- incoming neighbors
- edge costs
- graph diagnostics

### Bottleneck Results

Dictionary containing:

- `song_bottleneck_scores`
- `location_bottleneck_scores`
- `bottleneck_track_indices`
- `candidate_sets`

### Playlist

A list of track indices:

```text
[start_track, ..., end_track]
```

It must contain every track exactly once.

## 10. Where to Change What

Use this map when deciding where a change belongs.

### Change playlist size, genre, or seed

File:

- `main.py`

Change:

- `POOL_SIZE`
- `GENRE`
- `RANDOM_SEED`
- `RUN_EXPERIMENTS`
- `RUN_BIBS_SWEEP`

### Change EV arc

File:

- `src/feature_engineering.py`

Change:

- `create_target_arc_from_tracks()`
- `create_linear_target_arc()`
- endpoint logic if arc depends on start/end selection

### Change `C_arc`

File:

- `src/feature_engineering.py`

Change:

- `compute_arc_cost_matrix()`

### Change `C_trans`

File:

- `src/cost_functions.py`

Change:

- `compute_transition_cost_matrix()`
- `compute_transition_component_matrices()`
- helper distance functions

### Change transition graph behavior

File:

- `src/transition_graph.py`

Change:

- `TransitionGraphConfig`
- `build_transition_graph_data()`

### Change candidate exposure

File:

- `src/candidate_orchestrator.py`

Change:

- `CandidateOrchestratorConfig`
- candidate-building methods

### Change bottleneck logic

File:

- `src/bottleneck_detector.py`

Change:

- `BottleneckConfig`
- song/location bottleneck score methods
- candidate-set construction

### Change Greedy

File:

- `src/greedy_baseline.py`

Change:

- `GreedyBaselineConfig`
- `_choose_track()`
- `generate()`

### Change BIBS

File:

- `src/bibs.py`

Change:

- `BIBSConfig`
- `_solve_interval()`
- `_expand_beam()`
- `_select_anchor()`
- `_fill_base_case()`
- `_base_case_score()`

### Change metrics

File:

- `src/evaluator.py`

Change:

- `evaluate()`
- metric helper methods
- validation methods if exact-once rules change

### Change output files

File:

- `src/output_writer.py`

Change:

- playlist columns
- metric comparison rows
- output file naming

## 11. Outputs and Diagnostics

The project can write:

- Greedy playlist CSV
- BIBS playlist CSV
- metric comparison CSV
- experiment results
- diagnostics

Important output locations:

- `outputs/playlists/`
- `outputs/results/`

The current project includes diagnostics for investigating:

- anchor choice
- beam/path construction
- bottleneck weighting
- local interval seams
- candidate exposure
- transition graph coverage

These diagnostics should usually stay close to the code path they inspect unless they become stable project features.

## 12. Practical Mental Model

The project can be understood as a sequence of transformations:

```text
Spotify CSV
-> fixed clean track pool
-> EV score
-> C_trans
-> start/end tracks
-> target arc
-> C_arc
-> bottleneck results
-> transition graph
-> Greedy playlist
-> BIBS playlist
-> evaluation metrics
-> diagnostics and outputs
```

The central comparison is:

- Greedy asks: "What is the best next track right now?"
- BIBS asks: "What anchor should structure this interval, and how should both sides be solved?"

That difference is the heart of the project.
