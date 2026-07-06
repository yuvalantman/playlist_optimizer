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
│   ├── proposal_and_