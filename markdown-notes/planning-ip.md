## What the Inference Pipeline Actually Has to Do

It's the simplest of the three pipelines. The hard work is already done — the fitted preprocessor and best model are saved on disk. Inference just has to:

1. Load `X_unknown.csv`
2. Apply `engineer_features()` — **the exact same function from `features.py`**
3. Load `best_model.joblib` — one file that contains both preprocessor + classifier
4. Call `.predict()` — the Pipeline handles preprocessing automatically
5. Format and save `submission.csv` in the Kaggle format (`ID` + `label`)

***

## Why It's So Simple

Because of the decisions made earlier. The `best_model.joblib` is a full sklearn `Pipeline` — not just the classifier. So calling `pipeline.predict(X_unknown)` automatically runs:

```
X_unknown (raw)
    → preprocessor.transform()   ← uses X_train statistics, not refitted
    → classifier.predict()
    → predictions
```

No manual scaling, no manual encoding, nothing. All of that is encapsulated inside the saved Pipeline object.

***

## What Goes in `inference.py`

Three focused functions:

```
inference.py
│
├── load_unknown_data()   ← loads X_unknown.csv, returns DataFrame
│
├── predict()             ← applies engineer_features(), loads pipeline,
│                           returns array of predictions
│
└── make_submission()     ← formats predictions into Kaggle CSV
                            (ID + label columns, correct format)
```

***

## Key Design Decisions

**`engineer_features()` is imported from `features.py` — not reimplemented.** This is the single most important point. If you rewrite it here, the two versions will eventually diverge and your predictions will be wrong. One function, two callers.

**`best_model.joblib` is the only model file needed.** The preprocessor is already inside it. The Inference Pipeline does not touch `preprocessor.joblib` directly.

**Prediction format matters.** The assignment specifies columns `ID` and `label`. The `X_unknown.csv` has an `id` column — we need to preserve it before `engineer_features()` drops it, then attach it back to the predictions.

**Predict class labels, not probabilities.** Kaggle submission requires `0` or `1`, not a probability score. So we use `.predict()`, not `.predict_proba()`.

***

## The Tricky Part — `id` Column Preservation

`engineer_features()` drops the `id` column (it's a pure row index). But we need `id` in the submission file. The solution:

```
Load X_unknown.csv
    ↓
Extract id column BEFORE calling engineer_features()
    ↓
engineer_features(X_unknown)   ← drops id, creates ratios
    ↓
pipeline.predict()
    ↓
Combine: id + predictions → submission.csv
```

This is handled inside `make_submission()` — the caller never has to think about it.

***

## The Notebook Flow

```
03_inference_pipeline.ipynb

Load X_unknown.csv
    ↓
make_submission(
    unknown_path  = "../data/raw/X_unknown.csv",
    model_path    = "../models/best_model.joblib",
    output_path   = "../outputs/submission.csv"
)
    ↓
Verify: print first 10 rows of submission.csv
Verify: value_counts() of label column
```

The notebook is essentially 5 lines of code — all logic lives in `inference.py`.

***

## What `inference.py` Does NOT Need

- No `ColumnTransformer` — already inside the saved Pipeline
- No `StandardScaler` — already inside the saved Pipeline
- No `OneHotEncoder` — already inside the saved Pipeline
- No `GridSearchCV` — training is done
- No validation metrics — no ground truth labels exist for `X_unknown`

***

