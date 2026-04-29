> *"Usually it's not a case of having a 'better' classifier that will produce good results. Rather, it's a case of identifying or generating good features."*

That one sentence is the most important line in the entire brief. EDA is where you find and generate those features. [sites.gatech](https://sites.gatech.edu/omscs7641/2026/01/26/eda-for-cs7641/)

## But Add One Important Nuance

Don't treat EDA as a **one-time gate** you must complete perfectly before touching any model code. The professional reality is: [vskumar](https://vskumar.blog/2025/03/14/the-role-of-exploratory-data-analysis-eda-in-ml-model-design/)

```
EDA → Preprocessing decisions → Train baseline model
         ↑                              ↓
         └──── New insights from model ←┘
               (feature importance, error analysis)
```

Your first EDA will be incomplete — that's normal. Model results will send you back to EDA with better questions. Plan for **at least two passes**.

## What Your EDA Must Specifically Answer for This Assignment

These are the questions whose answers directly drive every downstream decision:

| EDA Question | Why It Matters |
|---|---|
| What's the class balance of `loan_paid_back`? | If imbalanced → drives metric choice (ROC-AUC over accuracy) and `class_weight='balanced'` |
| Which columns have missing values and how many? | Determines imputation strategy per column |
| Which features are numeric vs categorical? | Drives your `ColumnTransformer` design |
| Are any numeric features heavily skewed? | May need log transform before scaling |
| Are any features correlated with each other? | Multicollinearity — may drop or combine |
| Are any features correlated with the target? | Identifies your strongest predictors |
| Are there any obvious data quality issues? | Wrong dtypes, nonsense values, duplicates |

These 7 questions are the minimum — answer all of them and your Criterion 3 justification practically writes itself.

## What EDA Mistakes Actually Cost You

The costly mistakes aren't missed visualizations — they're: [scribd](https://www.scribd.com/document/721126230/Exploratory-Data-Analysis-in-ML)

- **Fitting the scaler on the full dataset** before splitting → data leakage → inflated metrics that collapse on `X_unknown`
- **Ignoring class imbalance** → model predicts majority class always → looks 80% accurate but is useless
- **Not checking dtypes** → a numeric column stored as `object` silently breaks your `ColumnTransformer`
- **Missing the `loan_paid_back` distribution** → wrong evaluation metric chosen for the whole project

So yes — thorough EDA, but focused on these concrete questions rather than exhaustive plotting for its own sake.

## Revised FTI Plan (Post-EDA)

### Feature Pipeline → `01_feature_pipeline.ipynb` + `features.py`

```
Load trainData.csv
    ↓
engineer_features()          ← drop id, monthly_income, installment
                               create credit_utilization, loan_to_income
    ↓
split_data()                 ← stratified 70/15/15 train/val/test
    ↓
build_preprocessor()         ← fit on X_train ONLY
    ├── numeric: log1p(current_balance, total_credit_limit, annual_income)
    │            → StandardScaler
    ├── numeric (no log): debt_to_income_ratio, credit_score, interest_rate, age...
    │            → StandardScaler
    └── categorical: gender, marital_status, education_level,
                     employment_status, loan_purpose, loan_term
                     → OneHotEncoder
    ↓
Save to disk:
    ├── data/processed/X_train.csv, X_val.csv, X_test.csv
    ├── data/processed/y_train.csv, y_val.csv, y_test.csv
    └── models/preprocessor.joblib    ← FITTED on X_train only
```

### Training Pipeline → `02_training_pipeline.ipynb` + `training.py`

```
Load X_train, y_train from data/processed/
Load preprocessor.joblib
    ↓
get_models(preprocessor)     ← 7 sklearn Pipelines with param grids
    ↓
evaluate_all()               ← GridSearchCV + StratifiedKFold(5)
                               scoring = roc_auc
    ↓
Compare all models           ← results table: model, CV ROC-AUC, best params
    ↓
Evaluate best model on X_val ← confusion matrix, F1, precision, recall
    ↓
Save to disk:
    └── models/best_model.joblib
```

### Inference Pipeline → `03_inference_pipeline.ipynb` + `inference.py`

```
Load X_unknown.csv
Load preprocessor.joblib     ← same fitted object from Feature Pipeline
Load best_model.joblib
    ↓
engineer_features(X_unknown) ← same function as Feature Pipeline
    ↓
predict()                    ← model.predict() — preprocessor is inside pipeline
    ↓
Save to disk:
    └── outputs/submission.csv  ← ID + label columns for Kaggle
```

## The One Rule That Ties Everything Together

The preprocessor **must flow** from Feature Pipeline → Inference Pipeline via `models/preprocessor.joblib`. This is the architectural guarantee that prevents train-serving skew: [blog.det](https://blog.det.life/building-machine-learning-pipelines-with-the-fti-architecture-a-practical-step-by-step-guide-179acfb3da14?gi=be240f266379)

```
trainData.csv ──→ Feature Pipeline ──→ preprocessor.joblib ──→ Training Pipeline
                                              ↓
X_unknown.csv ──→ Inference Pipeline ←────────┘
```

Never refit the preprocessor on `X_unknown`. The fitted scalers and encoders from `X_train` must be reused as-is — otherwise your test predictions are computed on a different scale than your training was done on.

## What Goes in `src/` vs Notebooks

| Code | Location | Reason |
|---|---|---|
| `engineer_features()`, `build_preprocessor()`, `split_data()` | `features.py` | Called by both Feature and Inference notebooks — must be identical |
| `get_models()`, `evaluate_all()` | `training.py` | Reusable across experiments |
| `make_submission()` | `inference.py` | Clean separation of concerns |
| EDA cells, results tables, plots, markdown notes | Notebooks only | Exploratory — not reusable code |

You're ready to start writing `features.py`. Want to go through it function by function?