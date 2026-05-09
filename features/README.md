# Features

This directory holds extracted feature artefacts produced by the notebook so that the modelling cells can be re-run without regenerating the upstream simulation each time.

Generated files:

- `dataset.csv` — Module E training data: per `(state, action)` features and binary `label` (top-fraction by teacher signal). Produced by `build_ml_dataset` in `notebooks/BlockBlast.ipynb`.
- `blockblast_model.pkl` — Pickled `dict` containing the trained Decision Tree, Random Forest, and the canonical `feature_columns` list. Loaded with `joblib.load`.
- `outputs/bayes/` — Module D rollout dataset and CPD CSVs (`bayes_dataset.csv`, `cpd_valid_move_level.csv`, `cpd_stuck_risk.csv`, `bayes_stuckrisk_distribution.csv`, `bayes_coverage_summary.csv`).

These files are produced when the corresponding notebook cells run; they are not committed by default (see `.gitignore`).
