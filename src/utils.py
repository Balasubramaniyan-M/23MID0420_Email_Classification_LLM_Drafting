"""
Utility functions for reproducibility seeding, stratified train/test splitting, split manifest export,
and core visualization plots.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def set_seed(seed: int = RANDOM_STATE) -> None:
    """
    Set random seed across NumPy and random modules for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Random seed value, defaults to 42.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)


def make_stratified_split(
    df: pd.DataFrame, test_size: float = 0.20, random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a locked stratified train/test split (80/20) for a given dataset DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset DataFrame.
    test_size : float, optional
        Fraction for test split, defaults to 0.20.
    random_state : int, optional
        Random state seed, defaults to 42.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (train_df, test_df) with reset indices.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_split_manifest(
    splits: Dict[str, Dict[str, pd.DataFrame]], output_path: Path
) -> Dict[str, Any]:
    """
    Extract email IDs from train/test splits and save to split_manifest.json.

    Parameters
    ----------
    splits : Dict[str, Dict[str, pd.DataFrame]]
        Splits dictionary mapping dataset_id to {'train': train_df, 'test': test_df}.
    output_path : Path
        Target JSON file path.

    Returns
    -------
    Dict[str, Any]
        Manifest dictionary.
    """
    manifest = {
        dataset_id: {
            "train_ids": part["train"]["email_id"].tolist(),
            "test_ids": part["test"]["email_id"].tolist(),
        }
        for dataset_id, part in splits.items()
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def plot_class_distribution(
    datasets: Dict[str, pd.DataFrame], save_path: Optional[Path] = None
) -> None:
    """
    Plot bar charts of class distributions for all loaded datasets.

    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
        Map of dataset_id to DataFrame.
    save_path : Optional[Path], optional
        Path to save plot image.
    """
    fig, axes = plt.subplots(1, len(datasets), figsize=(12, 4))
    if len(datasets) == 1:
        axes = [axes]

    for idx, (ds_id, df) in enumerate(datasets.items()):
        counts = df["label"].value_counts()
        axes[idx].bar(counts.index, counts.values, color="teal", alpha=0.85)
        axes[idx].set_title(f"Class Distribution: {ds_id}")
        axes[idx].set_ylabel("Count")
        axes[idx].tick_params(axis="x", rotation=30)
        axes[idx].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_cv_macro_f1(
    cv_results: pd.DataFrame, save_path: Optional[Path] = None
) -> None:
    """
    Plot cross-validation macro F1 comparison with error bars across models and datasets.

    Parameters
    ----------
    cv_results : pd.DataFrame
        DataFrame of cross-validation results.
    save_path : Optional[Path], optional
        Path to save plot image.
    """
    datasets = cv_results["dataset_id"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(12, 5))
    if len(datasets) == 1:
        axes = [axes]

    for idx, ds_id in enumerate(datasets):
        ds_data = cv_results[cv_results["dataset_id"] == ds_id]
        axes[idx].bar(
            ds_data["model"],
            ds_data["macro_f1_mean"],
            yerr=ds_data["macro_f1_sd"],
            capsize=5,
            color="steelblue",
            alpha=0.85,
        )
        axes[idx].set_title(f"5-Fold CV Macro F1: {ds_id}")
        axes[idx].set_ylabel("Macro F1 (Mean ± SD)")
        axes[idx].set_ylim(0, 1.05)
        axes[idx].tick_params(axis="x", rotation=35)
        axes[idx].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_confusion_matrix_heatmap(
    cm: np.ndarray, labels: List[str], title: str = "Confusion Matrix", save_path: Optional[Path] = None
) -> None:
    """
    Plot confusion matrix heatmap with raw counts.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix array.
    labels : List[str]
        List of class label strings.
    title : str, optional
        Title of the plot.
    save_path : Optional[Path], optional
        Path to save plot image.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.8)
    fig.colorbar(cax)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), va="center", ha="center", color="black", fontsize=11
            )

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="left")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title, pad=20)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()
