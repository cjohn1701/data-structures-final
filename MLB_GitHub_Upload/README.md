# Shared teammate experience and MLB team performance

Research question: How accurately can prior teammate network density, average prior shared seasons, and full-season roster size predict regular-season win percentage for MLB team-seasons from 2005–2024?

This is a **retrospective analysis**, not a preseason forecast or a causal study. Full-season appearances identify the target roster. Payroll and player quality are omitted.

## Run

Use Python 3.12. From the project root:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows instead: .venv\Scripts\activate
python -m pip install -r requirements.txt
python src/analyze.py
```

Open `notebooks/mlb_analysis.ipynb` in Jupyter or VS Code and run all cells to reproduce and display the same outputs. The notebook calls the source script rather than duplicating its implementation.

## Files

- `src/analyze.py`: data acquisition, feature construction, quality assertions, models, validation, sensitivity checks, and charts.
- `data/raw/`: input files are downloaded automatically and checked against their recorded SHA-256 hashes.
- `data/processed/team_seasons.csv`: 600 rows, 10 columns; three predictors.
- `outputs/`: tables, eight figures, test predictions, data hashes, and quality checks.
- `SOURCE_VERIFICATION.md`: ten scholarly sources, claims, verification links, and access limitations.

## Data provenance

Original database: [SABR Lahman](https://sabr.org/lahman-database/). Distribution: [Rdatasets](https://github.com/vincentarelbundock/Rdatasets). Tables: Appearances and Teams. Downloaded November 18, 2026. The script pins revision `f91e6ed4824ad5cdb75c3a5495d228294d91a184` and checks SHA-256 hashes. Inputs contain 2025, but evaluation ends in 2024 to preserve the earlier study scope. See `outputs/data_manifest.json`. Raw source data retain their upstream terms; no new license over them is asserted.

## Measurement and validation

AL/NL pair history begins in 1995. Pair connections are unordered, deduplicated, and counted at most once per historical year. Every target-year team is scored before that year's pair history is updated. Density is the fraction of all possible current-roster pairs with any past shared season. Tenure is average previous shared seasons over **all** possible pairs, including zero-history pairs. Participant count is not a direct turnover measure.

Train: 2005–2019 (450 rows). Test: 2020–2024 (150 rows). Earlier expanding validation folds end in 2013, 2016, and 2019. The later period was already examined in the preliminary worksheet, so this rerun is not a new untouched holdout. Standardization is fitted inside each Ridge training pipeline. Fixed original model settings are retained. Post-holdout ablations are exploratory.

## Main results

| Approach | MAE pp | RMSE pp | R² |
|---|---:|---:|---:|
| Training mean | 6.87 | 8.50 | approximately 0 |
| Linear | 6.74 | 8.12 | 0.087 |
| Ridge | 6.74 | 8.12 | 0.087 |
| Random forest | 6.86 | 8.32 | 0.041 |

All values are computed, not simulated. The gain is small. Historical co-membership cannot directly measure chemistry. Paired season-level bootstrap uncertainty uses only five observed years and is illustrative. The study does not support employment or roster decisions.

## AI assistance and contributions

ChatGPT helped with ensuring code was as efficient as possible and assisted with source verification and visuals. Reported group contributions are distinguished from AI work. 
