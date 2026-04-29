"""
features.py
Feature Pipeline functions for the loan classification project.
Follows FTI architecture — all functions here are shared between
the Feature Pipeline (01) and the Inference Pipeline (03).

Safe order of operations (enforced by caller):
    load_raw_data()
        → engineer_features()      # row-wise only — safe before split
        → split_data()             # split BEFORE any fitting
        → build_preprocessor()     # returns unfitted ColumnTransformer
                                   # fit on X_train only in the notebook
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split

# ── Constants ────────────────────────────────────────────────────────────────

TARGET = "loan_paid_back"

# Skewness > 1.0 from EDA Q4 — apply log1p before scaling
LOG_SCALE_COLS = [
    "annual_income",       # skew 2.343
    "current_balance",     # skew 2.849
    "total_credit_limit",  # skew 2.533
]

# Skewness between -1.0 and 1.0 — scale only
SCALE_COLS = [
    "credit_score",           # skew -0.076
    "debt_to_income_ratio",   # skew  0.781
    "interest_rate",          # skew  0.028
    "age",                    # skew  0.016
    "loan_amount",            # skew  0.251
    "delinquency_history",    # skew  0.826
    "num_of_open_accounts",   # skew  0.448
    "credit_utilization",     # engineered ratio — bounded, low skew expected
    "loan_to_income",         # engineered ratio — bounded, low skew expected
]

# Categorical — OneHotEncoder
# loan_term included here: only 2 values (36/60), treated as categorical
CATEGORICAL_COLS = [
    "employment_status",
    "gender",
    "marital_status",
    "education_level",
    "loan_purpose",
    "loan_term",
]


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw CSV and return as DataFrame."""
    df = pd.read_csv(path)
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop redundant columns and create new ratio features.

    All operations here are row-wise arithmetic — no statistics
    are learned from the data, so this is safe to call before splitting.

    Drops:
        - id              : row index, zero predictive value
        - monthly_income  : perfect duplicate of annual_income (Q5 corr = 1.0)
        - installment     : mathematically derived from loan_amount (Q5 corr = 0.944)
        - num_of_delinquencies : near-duplicate of delinquency_history (Q5 corr = 0.903)

    Creates:
        - credit_utilization : current_balance / total_credit_limit
          Signal hidden in ratio — raw columns showed near-zero target corr
          due to skewness (Q4), but their ratio captures credit stress directly.

        - loan_to_income : loan_amount / annual_income
          Loan size relative to income is more predictive than raw loan_amount
          (Q6 target corr = 0.000 for loan_amount alone).
    """
    df = df.copy()

    # Drop redundant / non-predictive columns
    cols_to_drop = ["id", "monthly_income", "installment", "num_of_delinquencies"]
    df = df.drop(columns=cols_to_drop)

    # Engineer ratio features
    df["credit_utilization"] = df["current_balance"] / df["total_credit_limit"]
    df["loan_to_income"] = df["loan_amount"] / df["annual_income"]

    # loan_term: cast to string so ColumnTransformer treats it as categorical
    df["loan_term"] = df["loan_term"].astype(str)

    return df


# ── Data Splitting ────────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, ...]:
    """
    Stratified train / val / test split.

    Stratify on target to preserve the 80/20 class ratio in every split.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Then split remaining into train / val
    # Adjust val_size relative to the remaining portion
    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=adjusted_val_size,
        stratify=y_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Preprocessor ─────────────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Build and return an UNFITTED ColumnTransformer.

    Three transformer branches:
        log_scale  : log1p → StandardScaler  (heavily right-skewed features)
        scale      : StandardScaler only     (already symmetric features)
        encode     : OneHotEncoder           (categorical features)

    IMPORTANT: Call preprocessor.fit(X_train) in the notebook — never fit
    on val, test, or X_unknown. The fitted object must be saved to disk and
    reloaded in the Inference Pipeline to prevent train-serving skew.
    """
    # Branch 1 — log1p then scale (skewness > 1.0)
    log_scale_transformer = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, validate=True)),
        ("scaler", StandardScaler()),
    ])

    # Branch 2 — scale only (skewness between -1.0 and 1.0)
    scale_transformer = StandardScaler()

    # Branch 3 — one-hot encode categorical columns
    # handle_unknown='ignore' silently ignores unseen categories in X_unknown
    # sparse_output=False returns a dense array (easier to work with in notebooks)
    encode_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("log_scale", log_scale_transformer, LOG_SCALE_COLS),
            ("scale",     scale_transformer,     SCALE_COLS),
            ("encode",    encode_transformer,    CATEGORICAL_COLS),
        ],
        remainder="drop",  # drop any column not explicitly listed above
    )

    return preprocessor


# ── Feature Group Accessors ───────────────────────────────────────────────────

def get_feature_groups() -> dict[str, list[str]]:
    """
    Return all feature group definitions in one place.
    Useful for inspection and debugging in notebooks.
    """
    return {
        "log_scale": LOG_SCALE_COLS,
        "scale": SCALE_COLS,
        "categorical": CATEGORICAL_COLS,
        "target": [TARGET],
        "dropped": ["id", "monthly_income", "installment", "num_of_delinquencies"],
        "engineered": ["credit_utilization", "loan_to_income"],
    }