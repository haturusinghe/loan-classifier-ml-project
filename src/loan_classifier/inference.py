"""
inference.py
Inference Pipeline for the loan classification project.
Follows FTI architecture — load X_unknown, apply same feature engineering
as Feature Pipeline, predict using saved best model, output Kaggle submission.

Key guarantee: engineer_features() is imported from features.py — the exact
same function used in the Feature Pipeline. Never reimplement it here.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline

from loan_classifier.features import engineer_features


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_unknown_data(path: str) -> pd.DataFrame:
    """
    Load X_unknown.csv and return as DataFrame.

    Args:
        path: path to X_unknown.csv

    Returns:
        Raw DataFrame — not yet engineered or preprocessed
    """
    df = pd.read_csv(path)
    print(f"Loaded X_unknown: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(
    df_unknown: pd.DataFrame,
    model_path: str,
) -> tuple[np.ndarray, pd.Series]:
    """
    Apply feature engineering and predict class labels for X_unknown.

    The saved Pipeline (best_model.joblib) contains both the fitted
    preprocessor and the fitted classifier — .predict() runs both steps
    automatically using X_train statistics. Nothing is refitted here.

    Args:
        df_unknown: raw X_unknown DataFrame (output of load_unknown_data)
        model_path: path to best_model.joblib

    Returns:
        predictions: np.ndarray of 0/1 labels
        ids:         pd.Series of original id values (preserved before drop)
    """
    # Preserve id BEFORE engineer_features() drops it
    if "id" not in df_unknown.columns:
        raise ValueError("X_unknown must contain an 'id' column for submission")
    ids = df_unknown["id"].copy()

    # Apply identical feature engineering as Feature Pipeline
    # engineer_features() is row-wise arithmetic — safe to call on any split
    df_engineered = engineer_features(df_unknown)

    # Load full Pipeline (preprocessor + classifier bundled together)
    pipeline: Pipeline = joblib.load(model_path)
    print(f"Loaded model from: {model_path}")
    print(f"Model type: {type(pipeline.named_steps['classifier']).__name__}")

    # Predict class labels (0 or 1) — not probabilities
    predictions = pipeline.predict(df_engineered)
    print(f"Predictions complete: {len(predictions)} rows")
    print(f"Class distribution → 0: {(predictions == 0).sum()}  1: {(predictions == 1).sum()}")

    return predictions, ids


# ── Submission Builder ────────────────────────────────────────────────────────

def make_submission(
    unknown_path: str,
    model_path:   str,
    output_path:  str,
) -> pd.DataFrame:
    """
    End-to-end function: load → engineer → predict → save submission CSV.

    Kaggle requires exactly two columns: 'ID' and 'label'.
    Column names are case-sensitive — 'ID' (uppercase) must match
    sample_submission.csv exactly.

    Args:
        unknown_path: path to X_unknown.csv
        model_path:   path to best_model.joblib
        output_path:  path to write submission CSV (e.g. '../outputs/submission.csv')

    Returns:
        submission DataFrame — also saved to output_path
    """
    # Load raw unknown data
    df_unknown = load_unknown_data(unknown_path)

    # Engineer features + predict
    predictions, ids = predict(df_unknown, model_path)

    # Build submission DataFrame in Kaggle format
    submission = pd.DataFrame({
        "ID":    ids.values,
        "label": predictions,
    })

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved → {output_path}")
    print(f"Shape: {submission.shape}")

    return submission


# ── Validation Helper ─────────────────────────────────────────────────────────

def validate_submission(submission: pd.DataFrame) -> None:
    """
    Run basic sanity checks on the submission DataFrame before uploading.

    Checks:
        - Correct column names ('ID' and 'label')
        - No missing values
        - Labels are only 0 or 1
        - No duplicate IDs
    """
    print("\n── Submission Validation ──────────────────────────")

    # Column names
    expected_cols = {"ID", "label"}
    actual_cols   = set(submission.columns)
    if actual_cols != expected_cols:
        raise ValueError(f"Wrong columns: got {actual_cols}, expected {expected_cols}")
    print("✓ Column names correct: ID, label")

    # Missing values
    nulls = submission.isnull().sum().sum()
    if nulls > 0:
        raise ValueError(f"Found {nulls} missing values in submission")
    print("✓ No missing values")

    # Label values
    invalid_labels = ~submission["label"].isin([0, 1])
    if invalid_labels.any():
        raise ValueError(f"Invalid labels found: {submission.loc[invalid_labels, 'label'].unique()}")
    print(f"✓ Labels are valid: 0={( submission['label'] == 0).sum()}  1={(submission['label'] == 1).sum()}")

    # Duplicate IDs
    dupes = submission["ID"].duplicated().sum()
    if dupes > 0:
        raise ValueError(f"Found {dupes} duplicate IDs in submission")
    print("✓ No duplicate IDs")

    print("── All checks passed ──────────────────────────────\n")