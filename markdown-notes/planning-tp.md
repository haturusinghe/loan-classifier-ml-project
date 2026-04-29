
## Overall Structure of `training.py`

Four functions, each with a single responsibility:

```
get_models(preprocessor)   → dict of {name: sklearn Pipeline}
get_param_grids()          → dict of {name: param_grid for GridSearchCV}
train_all_models()         → runs GridSearchCV on all models, returns results
evaluate_model()           → computes metrics on val/test set for one model
```

The notebook calls these in sequence — it doesn't contain logic, just orchestration.

***

## The 7 Models

| # | Model | Why It's Here |
|---|---|---|
| 1 | **Logistic Regression** | Linear baseline — interpretable coefficients, strong for well-separated classes |
| 2 | **Decision Tree** | Fully interpretable, shows decision rules — good for oral defence |
| 3 | **Random Forest** | Ensemble of trees — handles non-linearity, gives feature importance |
| 4 | **K-Nearest Neighbors** | Instance-based — no assumptions about distribution |
| 5 | **SVM** | Margin-based — effective in high dimensions after OHE encoding |
| 6 | **Naive Bayes** | Probabilistic baseline — very fast, good sanity check |
| 7 | **Gradient Boosting** | Strong performer — XGBoost or sklearn's GradientBoostingClassifier |

Models 1–6 are the core workshop requirement. Model 7 is the "expert use" differentiator the assignment rubric specifically rewards. [towardsdatascience](https://towardsdatascience.com/top-machine-learning-algorithms-for-classification-2197870ff501/)

***

## Hyperparameters to Tune Per Model

| Model | Parameters | Why These |
|---|---|---|
| **LR** | `C`, `class_weight` | C controls regularisation strength; class_weight handles 80/20 imbalance |
| **DT** | `max_depth`, `min_samples_split`, `min_samples_leaf`, `class_weight` | Prevents overfitting; leaf size controls tree complexity |
| **RF** | `n_estimators`, `max_depth`, `max_features`, `class_weight` | More trees = less variance; max_features controls diversity |
| **KNN** | `n_neighbors`, `weights`, `metric` | k controls bias-variance; distance weighting handles imbalance |
| **SVM** | `C`, `kernel`, `gamma`, `class_weight` | Kernel choice is critical; C controls margin softness |
| **NB** | `var_smoothing` | Only tunable parameter — controls numerical stability |
| **GBM** | `n_estimators`, `learning_rate`, `max_depth`, `subsample` | Classic GBM tradeoff: more trees + lower LR = better generalisation |

***

## Evaluation Strategy

This is important because of the **80/20 class imbalance**. Accuracy alone is meaningless — a model predicting class 1 every time would score 80% accuracy while being completely useless.

**Primary scoring for GridSearchCV: `roc_auc`**
ROC-AUC measures how well the model separates the two classes regardless of the imbalance. [builtin](https://builtin.com/data-science/supervised-machine-learning-classification)

**Full evaluation metrics per model:**
- ROC-AUC — primary ranking metric
- F1-score (macro) — balances precision and recall across both classes
- Precision and Recall — separately for class 0 (the minority, harder to predict correctly)
- Confusion matrix — shows exactly where each model makes mistakes

**Cross-validation setup: `StratifiedKFold(n_splits=5)`**
Stratified ensures each fold preserves the 80/20 ratio, so CV scores are comparable across models.

***

## Class Imbalance Handling

For models that support it, we set `class_weight='balanced'` — sklearn automatically weights class 0 (minority, didn't repay) higher to compensate for its lower frequency: [geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/top-machine-learning-algorithms-for-classification/)

```
class_weight='balanced' adds weight:
    class 0 (20%) → weighted 4× more heavily
    class 1 (80%) → weighted 1×
```

This makes the model pay more attention to correctly classifying the harder minority class. Models that support this: LR, DT, RF, SVM.

KNN and NB don't have `class_weight` — they handle it differently (KNN via distance weighting, NB via prior probabilities).

***

## What Gets Saved

```
models/
├── preprocessor.joblib        ← from Feature Pipeline (already done)
├── results_table.csv          ← all models compared by metric
└── best_model.joblib          ← full sklearn Pipeline (preprocessor + classifier)
```

The `best_model.joblib` is a complete sklearn Pipeline containing both the fitted preprocessor and the fitted classifier — so the Inference Pipeline only needs to load one file and call `.predict()`.

***

## The Notebook Flow

```
02_training_pipeline.ipynb

Load X_train, X_val, X_test, y_* from data/processed/
Load preprocessor from models/preprocessor.joblib
    ↓
models   = get_models(preprocessor)
grids    = get_param_grids()
    ↓
results  = train_all_models(models, grids, X_train, y_train)
    → For each model: GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(5), scoring='roc_auc')
    → Fits on X_train, returns best params + CV score
    ↓
Display results table (all 7 models sorted by ROC-AUC)
    ↓
Evaluate top 2-3 models on X_val
    → Confusion matrix, F1, precision, recall per model
    ↓
Select best model — justify choice in markdown cell
    ↓
Final evaluation on X_test (done ONCE, at the very end)
    ↓
Save best_model.joblib
```

The key discipline: **`X_test` is only touched once** — after the best model is already selected using `X_val`. This ensures the test score is a genuine unseen estimate, not an optimistic one from repeated evaluation. [joneshshrestha](https://joneshshrestha.com/blog/16-machine-learning-pipelines/)

Good constraint to set upfront — it changes the design meaningfully. Here's the plan:

## The Core Design Principle — Data-Driven, Not Logic-Driven

The extensibility problem with most training pipelines is that models and their param grids get hardcoded inside function logic. Adding a new model means editing function internals — risky and messy.

The solution is a **registry pattern**: models and their configs live as plain data (dicts). Every function just iterates over whatever is in the registry. Adding a new model means adding one entry to the registry — zero changes to any function. [artiba](https://www.artiba.org/blog/best-practices-for-designing-an-efficient-machine-learning-pipeline)

```
# Adding a new model in future = just this:
register_model(
    "XGBoost",
    XGBClassifier(...),
    {"classifier__n_estimators": [100, 200], ...}
)
# Every other function automatically picks it up
```

***

## Structure of `training.py`

Five components:

```
training.py
│
├── ModelRegistry          ← stores {name: (pipeline, param_grid)} pairs
│   ├── register()         ← add a model + its param grid
│   ├── get_pipeline()     ← retrieve one pipeline by name
│   └── get_all()          ← retrieve all registered models
│
├── build_model_registry() ← creates registry pre-loaded with all 7 models
│
├── train_all_models()     ← runs GridSearchCV on every model in registry
│
├── evaluate_model()       ← full metrics for one fitted model on a split
│
└── compare_models()       ← produces ranked results DataFrame
```

***

## How Each Model's Pipeline is Built

Each model gets wrapped in a `sklearn.pipeline.Pipeline` with the preprocessor as the first step. The param grid keys use the `classifier__` prefix — this is sklearn's convention for targeting a specific step in a pipeline with GridSearchCV:

```
Pipeline([
    ("preprocessor", preprocessor),   ← step name: "preprocessor"
    ("classifier",   LogisticRegression())  ← step name: "classifier"
])

# Param grid targets the classifier step using double underscore:
{"classifier__C": [0.01, 0.1, 1, 10]}
```

Because **every model uses the same step name `"classifier"`**, the param grid format is consistent across all models — which is exactly what makes iteration clean.

***

## The 7 Models and Their Param Grids (preview)

| Model | Key Params to Tune | Notes |
|---|---|---|
| `LogisticRegression` | `C`, `class_weight` | `max_iter=1000` — OHE expanded features need more iterations |
| `DecisionTreeClassifier` | `max_depth`, `min_samples_split`, `min_samples_leaf`, `class_weight` | Small grid — DTs overfit fast |
| `RandomForestClassifier` | `n_estimators`, `max_depth`, `max_features`, `class_weight` | Wider grid — benefits from more tuning |
| `KNeighborsClassifier` | `n_neighbors`, `weights`, `metric` | No class_weight — use `weights='distance'` instead |
| `SVC` | `C`, `kernel`, `gamma`, `class_weight` | `probability=True` needed for ROC-AUC scoring |
| `GaussianNB` | `var_smoothing` | Log-spaced values — only one param |
| `GradientBoostingClassifier` | `n_estimators`, `learning_rate`, `max_depth`, `subsample` | No class_weight — handle imbalance via `subsample` |

***

## Evaluation Metrics Strategy

Because of the 80/20 imbalance, we track multiple metrics, not just accuracy:

```
For each model:
    CV ROC-AUC          ← primary ranking metric (from GridSearchCV)
    Val ROC-AUC         ← on held-out val set
    Val F1 (macro)      ← balances performance across both classes
    Val Precision (0)   ← how precise is it when predicting "didn't repay"
    Val Recall (0)      ← how many actual defaulters does it catch
    Val Accuracy        ← shown last — least informative given imbalance
```

The output of `compare_models()` is a clean DataFrame sorted by CV ROC-AUC — this becomes the comparison table in your report.

***

## Extensibility in Practice

Extending the registry later looks like this — nothing else changes:

```python
from xgboost import XGBClassifier

registry.register(
    name="XGBoost",
    estimator=XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    ),
    param_grid={
        "classifier__n_estimators": [100, 200, 300],
        "classifier__learning_rate": [0.05, 0.1, 0.2],
        "classifier__max_depth": [3, 5, 7],
        "classifier__subsample": [0.8, 1.0],
    }
)

# Then just re-run train_all_models() — XGBoost is included automatically
```

No changes to `train_all_models()`, `evaluate_model()`, or the notebook logic. The registry decouples the *what to train* from the *how to train it*. [artiba](https://www.artiba.org/blog/best-practices-for-designing-an-efficient-machine-learning-pipeline)

