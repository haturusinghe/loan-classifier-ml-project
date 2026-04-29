## The Problem It Solves

Your dataset has two types of columns that ML models fundamentally can't handle raw:

- **Numeric columns at wildly different scales** — `credit_score` ranges 373–850, `annual_income` ranges 6,000–400,000. A model treating these as raw numbers would think income is 500× more important than credit score, just because its numbers are bigger
- **Categorical columns stored as text** — `employment_status` contains "Employed", "Unemployed" etc. A model can't do math on strings

A **preprocessor** is a single object that fixes both problems in one step. [paubox](https://www.paubox.com/blog/what-is-the-preprocessing-pipeline)

## What It Actually Does to Your Data

```
Raw row:
annual_income=400000, credit_score=620, employment_status="Unemployed"

After preprocessor:
annual_income_scaled=-0.23, credit_score_scaled=-0.84,
emp_Employed=0, emp_Retired=0, emp_Student=0, emp_Unemployed=1, emp_Self=0
```

It **scales** numeric columns (so all features live on the same scale) and **encodes** categorical columns into numbers (so models can do math on them).

## How sklearn Builds It — ColumnTransformer

sklearn's `ColumnTransformer` lets you apply different transformations to different columns simultaneously: [geeksforgeeks](https://www.geeksforgeeks.org/blogs/machine-learning-pipeline/)

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

preprocessor = ColumnTransformer([
    # Apply log1p then scale these (heavily skewed)
    ('log_scale', Pipeline([
        ('log', FunctionTransformer(np.log1p)),
        ('scaler', StandardScaler())
    ]), ['annual_income', 'current_balance', 'total_credit_limit']),

    # Just scale these (already symmetric)
    ('scale', StandardScaler(),
        ['credit_score', 'debt_to_income_ratio', 'interest_rate', 'age',
         'loan_amount', 'delinquency_history', 'num_of_open_accounts']),

    # Encode these (categorical)
    ('encode', OneHotEncoder(handle_unknown='ignore'),
        ['employment_status', 'gender', 'marital_status',
         'education_level', 'loan_purpose', 'loan_term'])
])
```

## The Critical Rule — Fit Once, Transform Many Times

This is where it connects to your FTI pipeline. The preprocessor has two modes: [lakefs](https://lakefs.io/blog/data-preprocessing-in-machine-learning/)

```
preprocessor.fit(X_train)      ← LEARNS the statistics from training data
                                  e.g., mean=43537, std=28684 for annual_income

preprocessor.transform(X)      ← APPLIES those learned statistics to any data
```

**Why this matters for your FTI architecture:**

```
Feature Pipeline:
    preprocessor.fit(X_train)           ← learns from training data
    preprocessor.transform(X_train)     ← transforms training set
    preprocessor.transform(X_val)       ← same stats applied to val
    preprocessor.transform(X_test)      ← same stats applied to test
    joblib.dump(preprocessor, 'models/preprocessor.joblib')

Inference Pipeline:
    preprocessor = joblib.load('models/preprocessor.joblib')
    preprocessor.transform(X_unknown)   ← same stats applied to Kaggle data
```

If you called `fit()` again on `X_val` or `X_unknown`, the mean and std would be different numbers — your model would receive data on a completely different scale than it was trained on, and predictions would be wrong.

## How It Fits Into the Training Pipeline

In sklearn, the preprocessor gets **wrapped inside a Pipeline object together with the model**:

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),    ← transforms features first
    ('classifier', RandomForestClassifier())  ← then model sees clean data
])

# This single call does both steps:
rf_pipeline.fit(X_train, y_train)

# And prediction does both steps automatically:
rf_pipeline.predict(X_unknown)
```

So in your training pipeline, you build one Pipeline object per model (LR, DT, RF, KNN, SVM, NB) — each containing the same preprocessor + a different classifier. GridSearchCV then tunes each one. The best Pipeline object is what gets saved to `models/best_model.joblib` and loaded in the Inference Pipeline.


## What Data Leakage Is

Leakage is when your model indirectly "sees" val/test data during training, making its performance look better than it really is. It's dangerous because your Kaggle score and oral defence results will be unreliable — you'll think your model is better than it actually is. [ibm](https://www.ibm.com/think/topics/data-leakage-machine-learning)

## The 3 Places Leakage Can Happen in Your Pipeline

**1 — Splitting AFTER preprocessing (most common mistake)**

```python
# ❌ WRONG — scaler learns mean/std from entire dataset including val/test
scaler.fit(df[num_cols])
X_scaled = scaler.transform(df[num_cols])
X_train, X_val = train_test_split(X_scaled)

# ✅ CORRECT — split first, then fit scaler only on X_train
X_train, X_val = train_test_split(df)
scaler.fit(X_train[num_cols])          # learns only from training data
X_train_scaled = scaler.transform(X_train[num_cols])
X_val_scaled = scaler.transform(X_val[num_cols])  # applies same stats
```

The scaler learns `mean` and `std` from whatever you call `.fit()` on. If that includes val/test rows, those splits have already influenced the transformation — that's leakage. [scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html)

**2 — Feature engineering BEFORE splitting**

```python
# ❌ WRONG — ratio computed on full dataset before split
df['credit_utilization'] = df['current_balance'] / df['total_credit_limit']
X_train, X_val = train_test_split(df)

# ✅ CORRECT — engineer_features() is safe here ONLY because
# it's doing row-wise arithmetic (no statistics learned from data)
# credit_utilization = current_balance / total_credit_limit
# This doesn't learn anything from other rows — it's fine pre-split
```

Actually `engineer_features()` is safe to call before splitting in your case because `credit_utilization` and `loan_to_income` are pure row-level calculations — no mean, min, max, or any aggregate statistic from other rows is involved. The dangerous operations are ones that learn population statistics — scaling, encoding frequencies, imputing with mean, etc. [airbyte](https://airbyte.com/data-engineering-resources/what-is-data-leakage)

**3 — Leakage inside GridSearchCV (the sneaky one)**

```python
# ❌ WRONG — preprocessor fitted on full X_train before CV
#            so each fold's val split has already influenced the scaler
preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)
GridSearchCV(RandomForestClassifier(), param_grid).fit(X_train_processed, y_train)

# ✅ CORRECT — wrap preprocessor inside sklearn Pipeline
#             GridSearchCV then re-fits the preprocessor on each fold's
#             training portion automatically
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])
GridSearchCV(pipeline, param_grid).fit(X_train, y_train)
```

This is the subtlest one. When GridSearchCV does 5-fold cross-validation, it creates 5 different train/val splits internally. If the preprocessor was already fitted on all of `X_train`, the statistics from each fold's internal val portion have already leaked into the scaler. Wrapping everything in a `Pipeline` fixes this automatically — sklearn re-fits the preprocessor on each fold's training portion only. [leeroopedia](https://leeroopedia.com/index.php/Heuristic:Scikit_learn_Scikit_learn_Data_Leakage_Prevention)

## Your Safe Order of Operations

```
Raw trainData.csv
    ↓
engineer_features()          ← row-wise only, safe before split
    ↓
train_test_split(stratify=y) ← SPLIT FIRST — before any fitting
    ↓
    ├── X_train, y_train
    ├── X_val, y_val
    └── X_test, y_test
         ↓
Pipeline([preprocessor, model]).fit(X_train, y_train)   ← fit on X_train only
         ↓
pipeline.predict(X_val)      ← transform uses X_train statistics
pipeline.predict(X_test)     ← same
pipeline.predict(X_unknown)  ← same
```

## How sklearn Pipeline Protects You Automatically

The single biggest leakage protection is just using `sklearn.pipeline.Pipeline`. It enforces the correct behaviour mechanically: [sklearn](https://sklearn.org/stable/common_pitfalls.html)

```python
pipeline.fit(X_train, y_train)
# Internally does:
#   preprocessor.fit_transform(X_train)  ← fits on X_train
#   model.fit(X_train_transformed, y_train)

pipeline.predict(X_val)
# Internally does:
#   preprocessor.transform(X_val)        ← only transforms, never fits
#   model.predict(X_val_transformed)
```

You physically cannot call `.fit()` on val/test data by accident — the Pipeline object simply doesn't do it during `.predict()`. This is exactly why the plan wraps every model inside a Pipeline with the preprocessor attached.