"""
training.py
Training Pipeline functions for the loan classification project.
Follows FTI architecture — load processed features, train all models,
compare, select best.

Extensibility: add new models via registry.register() in the notebook
or in build_model_registry(). Zero changes needed to any other function.
"""

import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass, field
from typing import Any

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# ── Constants ─────────────────────────────────────────────────────────────────

CV_FOLDS    = 5
CV_SCORING  = "roc_auc"
RANDOM_SEED = 42


# ── Model Registry ────────────────────────────────────────────────────────────

@dataclass
class ModelEntry:
    """A single registered model — its sklearn Pipeline and param grid."""
    name:       str
    pipeline:   Pipeline
    param_grid: dict[str, Any]


class ModelRegistry:
    """
    Stores (pipeline, param_grid) pairs keyed by model name.

    Extensibility: call registry.register() to add any new model.
    All downstream functions (train_all_models, compare_models) iterate
    over self._entries — they pick up new models automatically.
    """

    def __init__(self):
        self._entries: dict[str, ModelEntry] = {}

    def register(
        self,
        name: str,
        estimator: Any,
        param_grid: dict[str, Any],
        preprocessor: Any,
    ) -> None:
        """
        Register a model.

        Args:
            name:         display name used in results tables
            estimator:    unfitted sklearn-compatible classifier
            param_grid:   GridSearchCV param grid — keys must use
                          'classifier__' prefix to target the pipeline step
            preprocessor: fitted ColumnTransformer from Feature Pipeline
        """
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier",   estimator),
        ])
        self._entries[name] = ModelEntry(
            name=name,
            pipeline=pipeline,
            param_grid=param_grid,
        )

    def get(self, name: str) -> ModelEntry:
        if name not in self._entries:
            raise KeyError(f"Model '{name}' not in registry. Available: {list(self._entries)}")
        return self._entries[name]

    def get_all(self) -> dict[str, ModelEntry]:
        return dict(self._entries)

    def list_models(self) -> list[str]:
        return list(self._entries.keys())


# ── Registry Builder ──────────────────────────────────────────────────────────

def build_model_registry(preprocessor: Any) -> ModelRegistry:
    """
    Create a ModelRegistry pre-loaded with all 7 classifiers.

    To add a new model later — call registry.register() after this
    function returns. Nothing else needs to change.

    class_weight='balanced' compensates for the 80/20 class imbalance
    by weighting the minority class (loan not repaid) more heavily during
    training. Applied to all models that support it.

    Args:
        preprocessor: fitted ColumnTransformer saved by Feature Pipeline

    Returns:
        ModelRegistry with 7 models registered
    """
    registry = ModelRegistry()

    # ── 1. Logistic Regression ────────────────────────────────────────────────
    registry.register(
        name="LogisticRegression",
        estimator=LogisticRegression(
            max_iter=1000,       # OHE-expanded features need more iterations
            random_state=RANDOM_SEED,
        ),
        param_grid={
            "classifier__C":            [0.01, 0.1, 1.0, 10.0, 100.0],
            "classifier__class_weight": [None, "balanced"],
            "classifier__solver":       ["lbfgs", "liblinear"],
        },
        preprocessor=preprocessor,
    )

    # ── 2. Decision Tree ──────────────────────────────────────────────────────
    registry.register(
        name="DecisionTree",
        estimator=DecisionTreeClassifier(random_state=RANDOM_SEED),
        param_grid={
            "classifier__max_depth":        [3, 5, 7, 10, None],
            "classifier__min_samples_split": [2, 10, 20],
            "classifier__min_samples_leaf":  [1, 5, 10],
            "classifier__class_weight":      [None, "balanced"],
        },
        preprocessor=preprocessor,
    )

    # ── 3. Random Forest ──────────────────────────────────────────────────────
    registry.register(
        name="RandomForest",
        estimator=RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
        param_grid={
            "classifier__n_estimators":  [100, 200, 300],
            "classifier__max_depth":     [5, 10, 20, None],
            "classifier__max_features":  ["sqrt", "log2"],
            "classifier__class_weight":  [None, "balanced"],
        },
        preprocessor=preprocessor,
    )

    # ── 4. K-Nearest Neighbors ────────────────────────────────────────────────
    # No class_weight support — use weights='distance' to partially
    # compensate for imbalance via proximity weighting
    registry.register(
        name="KNN",
        estimator=KNeighborsClassifier(n_jobs=-1),
        param_grid={
            "classifier__n_neighbors": [3, 5, 7, 11, 15],
            "classifier__weights":     ["uniform", "distance"],
            "classifier__metric":      ["euclidean", "manhattan"],
        },
        preprocessor=preprocessor,
    )

    # ── 5. Support Vector Machine ─────────────────────────────────────────────
    # probability=True required for roc_auc scoring
    registry.register(
        name="SVM",
        estimator=SVC(probability=True, random_state=RANDOM_SEED),
        param_grid={
            "classifier__C":            [0.1, 1.0, 10.0],
            "classifier__kernel":       ["rbf", "linear"],
            "classifier__gamma":        ["scale", "auto"],
            "classifier__class_weight": [None, "balanced"],
        },
        preprocessor=preprocessor,
    )

    # ── 6. Naive Bayes ────────────────────────────────────────────────────────
    # var_smoothing controls numerical stability — log-spaced values
    # are appropriate since the parameter spans many orders of magnitude
    registry.register(
        name="NaiveBayes",
        estimator=GaussianNB(),
        param_grid={
            "classifier__var_smoothing": np.logspace(-12, -6, 7).tolist(),
        },
        preprocessor=preprocessor,
    )

    # ── 7. Gradient Boosting ──────────────────────────────────────────────────
    # No class_weight — subsample + learning_rate together control overfitting
    # Lower learning_rate + more estimators generally outperforms the inverse
    registry.register(
        name="GradientBoosting",
        estimator=GradientBoostingClassifier(random_state=RANDOM_SEED),
        param_grid={
            "classifier__n_estimators":  [100, 200, 300],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__max_depth":     [3, 5, 7],
            "classifier__subsample":     [0.7, 0.8, 1.0],
        },
        preprocessor=preprocessor,
    )

    return registry


# ── Training ──────────────────────────────────────────────────────────────────

def train_all_models(
    registry: ModelRegistry,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = CV_FOLDS,
    scoring:  str = CV_SCORING,
    n_jobs:   int = -1,
    verbose:  int = 1,
) -> dict[str, GridSearchCV]:
    """
    Run GridSearchCV for every model in the registry.

    Preprocessor is inside each pipeline — GridSearchCV re-fits it on
    each fold's training portion automatically, preventing leakage.

    Args:
        registry:  ModelRegistry with all models registered
        X_train:   raw (unprocessed) training features
        y_train:   training labels
        cv_folds:  number of StratifiedKFold splits
        scoring:   GridSearchCV scoring metric
        n_jobs:    parallelism (-1 = all cores)
        verbose:   GridSearchCV verbosity level

    Returns:
        dict of {model_name: fitted GridSearchCV object}
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    fitted_searches = {}

    for name, entry in registry.get_all().items():
        print(f"\n{'─' * 50}")
        print(f"Training: {name}")
        print(f"{'─' * 50}")

        search = GridSearchCV(
            estimator=entry.pipeline,
            param_grid=entry.param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            refit=True,       # refit best model on full X_train after CV
            return_train_score=True,
        )
        search.fit(X_train, y_train)

        print(f"Best CV {scoring}: {search.best_score_:.4f}")
        print(f"Best params: {search.best_params_}")

        fitted_searches[name] = search

    return fitted_searches


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    search:   GridSearchCV,
    X:        pd.DataFrame,
    y:        pd.Series,
    split_name: str = "val",
) -> dict[str, Any]:
    """
    Compute full evaluation metrics for a fitted GridSearchCV on one split.

    Args:
        search:      fitted GridSearchCV (best estimator is used)
        X:           feature DataFrame (val or test)
        y:           true labels
        split_name:  label for display only ('val' or 'test')

    Returns:
        dict with all metrics — suitable for building a results DataFrame
    """
    best = search.best_estimator_
    y_pred      = best.predict(X)
    y_pred_prob = best.predict_proba(X)[:, 1]

    metrics = {
        "split":          split_name,
        "roc_auc":        round(roc_auc_score(y, y_pred_prob), 4),
        "f1_macro":       round(f1_score(y, y_pred, average="macro"), 4),
        "f1_class0":      round(f1_score(y, y_pred, average=None)[0], 4),
        "f1_class1":      round(f1_score(y, y_pred, average=None)[1], 4),
        "precision_0":    round(precision_score(y, y_pred, pos_label=0), 4),
        "recall_0":       round(recall_score(y, y_pred, pos_label=0), 4),
        "precision_1":    round(precision_score(y, y_pred, pos_label=1), 4),
        "recall_1":       round(recall_score(y, y_pred, pos_label=1), 4),
        "accuracy":       round(accuracy_score(y, y_pred), 4),
        "confusion_matrix": confusion_matrix(y, y_pred),
    }
    return metrics


# ── Comparison Table ──────────────────────────────────────────────────────────

def compare_models(
    fitted_searches: dict[str, GridSearchCV],
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate all fitted models on the validation set and return a
    ranked comparison DataFrame.

    This is the primary tool for selecting the best model.
    X_test is NOT touched here — it is only used for final evaluation
    after the best model is chosen.

    Args:
        fitted_searches: output of train_all_models()
        X_val:           validation features
        y_val:           validation labels

    Returns:
        DataFrame sorted descending by val_roc_auc
    """
    rows = []
    for name, search in fitted_searches.items():
        metrics = evaluate_model(search, X_val, y_val, split_name="val")
        rows.append({
            "model":          name,
            "cv_roc_auc":     round(search.best_score_, 4),
            "val_roc_auc":    metrics["roc_auc"],
            "val_f1_macro":   metrics["f1_macro"],
            "val_f1_class0":  metrics["f1_class0"],
            "val_precision_0": metrics["precision_0"],
            "val_recall_0":   metrics["recall_0"],
            "val_accuracy":   metrics["accuracy"],
            "best_params":    search.best_params_,
        })

    results_df = pd.DataFrame(rows).sort_values("val_roc_auc", ascending=False)
    results_df = results_df.reset_index(drop=True)
    return results_df


# ── Persistence ───────────────────────────────────────────────────────────────

def save_best_model(
    fitted_searches: dict[str, GridSearchCV],
    model_name: str,
    path: str,
) -> None:
    """
    Save the best estimator (full Pipeline) from a named GridSearchCV.

    Saves the complete sklearn Pipeline (preprocessor + classifier) so the
    Inference Pipeline only needs to load one file and call .predict().

    Args:
        fitted_searches: output of train_all_models()
        model_name:      name as registered in the ModelRegistry
        path:            file path e.g. '../models/best_model.joblib'
    """
    best_pipeline = fitted_searches[model_name].best_estimator_
    joblib.dump(best_pipeline, path)
    print(f"Saved: {model_name} → {path}")


def save_results_table(results_df: pd.DataFrame, path: str) -> None:
    """Save the comparison results DataFrame to CSV."""
    results_df.drop(columns=["best_params"]).to_csv(path, index=False)
    print(f"Saved results table → {path}")


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(search, X, y, title: str = "ROC Curve"):
    """
    Plot ROC curve for the best estimator in a fitted GridSearchCV.

    Works for any classifier that implements predict_proba() or decision_function().
    """
    best = search.best_estimator_

    # Get predicted probabilities or decision scores
    if hasattr(best, "predict_proba"):
        y_scores = best.predict_proba(X)[:, 1]
    else:
        # e.g. some SVM configs — decision_function returns signed distance
        y_scores = best.decision_function(X)

    fpr, tpr, _ = roc_curve(y, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(search, X, y, title: str = "Confusion Matrix"):
    """
    Plot confusion matrix heatmap for a fitted GridSearchCV.
    """
    best = search.best_estimator_
    y_pred = best.predict(X)

    cm = confusion_matrix(y, y_pred)
    labels = ["Not repaid (0)", "Repaid (1)"]

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.show()


from sklearn.metrics import classification_report

def classification_report_markdown(search, X, y) -> str:
    """
    Generate a classification_report and return it as a Markdown table string.

    Useful for pasting directly into the assignment report.
    """
    best = search.best_estimator_
    y_pred = best.predict(X)

    report_dict = classification_report(y, y_pred, output_dict=True, zero_division=0)

    # Build Markdown table manually
    headers = ["Class", "Precision", "Recall", "F1-score", "Support"]
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]

    for label in ["0", "1", "macro avg", "weighted avg"]:
        metrics = report_dict[label]
        line = "| {cls} | {p:.3f} | {r:.3f} | {f1:.3f} | {s:.0f} |".format(
            cls=label,
            p=metrics["precision"],
            r=metrics["recall"],
            f1=metrics["f1-score"],
            s=metrics["support"],
        )
        lines.append(line)

    markdown = "\n".join(lines)
    return markdown