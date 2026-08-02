"""
Model evaluation, cross-validation benchmarking, locked test set evaluation, and model persistence.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
SCORING = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted",
}


def run_cross_validation(
    models: Dict[str, Pipeline],
    splits: Dict[str, Dict[str, pd.DataFrame]],
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Run 5-fold Stratified Cross-Validation on the training splits of all datasets.

    Parameters
    ----------
    models : Dict[str, Pipeline]
        Dictionary of unfitted classifier pipelines.
    splits : Dict[str, Dict[str, pd.DataFrame]]
        Dictionary containing train and test DataFrames for each dataset.
    n_splits : int, optional
        Number of CV folds, defaults to 5.
    random_state : int, optional
        CV shuffle seed, defaults to 42.

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing CV metrics (mean ± SD) sorted by macro F1 score.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_rows: List[Dict[str, Any]] = []

    for dataset_id, part in splits.items():
        train_df = part["train"]
        X_train = train_df["text"]
        y_train = train_df["label"]

        for model_name, pipeline in models.items():
            scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=SCORING,
                n_jobs=-1,
                error_score="raise",
            )
            cv_rows.append(
                {
                    "dataset_id": dataset_id,
                    "model": model_name,
                    "accuracy_mean": float(scores["test_accuracy"].mean()),
                    "accuracy_sd": float(scores["test_accuracy"].std()),
                    "macro_f1_mean": float(scores["test_macro_f1"].mean()),
                    "macro_f1_sd": float(scores["test_macro_f1"].std()),
                    "weighted_f1_mean": float(scores["test_weighted_f1"].mean()),
                    "weighted_f1_sd": float(scores["test_weighted_f1"].std()),
                }
            )

    results_df = pd.DataFrame(cv_rows)
    results_df = results_df.sort_values(
        ["dataset_id", "macro_f1_mean"], ascending=[True, False]
    ).reset_index(drop=True)
    return results_df


def evaluate_locked_test(
    model: Pipeline, test_df: pd.DataFrame
) -> Tuple[Dict[str, float], pd.DataFrame, np.ndarray, List[str], np.ndarray]:
    """
    Perform a single locked evaluation of a fitted model on the locked test split.

    Parameters
    ----------
    model : Pipeline
        Fitted scikit-learn Pipeline.
    test_df : pd.DataFrame
        Locked test partition DataFrame.

    Returns
    -------
    Tuple[Dict[str, float], pd.DataFrame, np.ndarray, List[str], np.ndarray]
        - summary dict: overall accuracy, macro F1, weighted F1
        - report_df: per-class precision, recall, F1, support table
        - confusion matrix array (counts)
        - class label list
        - prediction vector
    """
    X_test = test_df["text"]
    y_test = test_df["label"]
    pred = model.predict(X_test)

    summary = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
    }

    report_dict = classification_report(y_test, pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).T

    labels = sorted(y_test.unique().tolist())
    cm = confusion_matrix(y_test, pred, labels=labels)
    return summary, report_df, cm, labels, pred


def fit_and_save_best_models(
    cv_results: pd.DataFrame,
    models: Dict[str, Pipeline],
    splits: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: Path = Path("outputs"),
) -> Tuple[Dict[str, Pipeline], pd.DataFrame]:
    """
    Select best model per dataset by CV macro F1, fit on full training set, evaluate on locked test set,
    and persist fitted pipelines with joblib.

    Parameters
    ----------
    cv_results : pd.DataFrame
        Sorted cross-validation results DataFrame.
    models : Dict[str, Pipeline]
        Registry of unfitted pipeline objects.
    splits : Dict[str, Dict[str, pd.DataFrame]]
        Splits dictionary.
    output_dir : Path, optional
        Output directory path, defaults to 'outputs'.

    Returns
    -------
    Tuple[Dict[str, Pipeline], pd.DataFrame]
        Map of dataset_id to fitted Pipeline, and locked test results DataFrame.
    """
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    selected_models: Dict[str, Pipeline] = {}
    test_rows: List[Dict[str, Any]] = []

    for dataset_id, part in splits.items():
        ranked = cv_results[cv_results["dataset_id"] == dataset_id]
        best_name = ranked.iloc[0]["model"]

        model = clone(models[best_name])
        model.fit(part["train"]["text"], part["train"]["label"])
        selected_models[dataset_id] = model

        summary, report_df, cm, labels, pred = evaluate_locked_test(model, part["test"])
        test_rows.append(
            {
                "dataset_id": dataset_id,
                "selected_model": best_name,
                **summary,
            }
        )

        joblib.dump(model, models_dir / f"{dataset_id}_{best_name}.joblib")

    test_results_df = pd.DataFrame(test_rows)
    return selected_models, test_results_df
