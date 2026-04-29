## What Data Leakage Actually Means Here

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