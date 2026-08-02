"""
Dataset registry, schema validation, and unified loading module.
Enforces the mandatory unified schema across all laboratory datasets.
"""

from pathlib import Path
from typing import Dict, Any, Set
import pandas as pd

REQUIRED_COLUMNS: Set[str] = {"email_id", "subject", "body", "label"}

DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "business_intent": {
        "path": Path("data") / "business_email_intent.csv",
        "task": "multiclass_intent",
        "label_space": ["request", "meeting", "complaint", "information", "urgent_action", "spam"],
    },
    "enron_spam": {
        "path": Path("data") / "enron_spam.csv",
        "task": "binary_spam",
        "label_space": ["legitimate", "spam"],
    },
}


def load_dataset(dataset_id: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load, validate, and preprocess a dataset according to the unified schema.

    Parameters
    ----------
    dataset_id : str
        Unique string identifier for the dataset (e.g., 'business_intent', 'enron_spam').
    config : Dict[str, Any]
        Configuration dictionary containing file path and task metadata.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns, normalized text fields, derived combined text,
        and computed character lengths.

    Raises
    ------
    FileNotFoundError
        If the configured CSV file does not exist.
    ValueError
        If required schema columns are missing from the loaded CSV.
    """
    path: Path = Path(config["path"])
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file for '{dataset_id}': {path}")

    df = pd.read_csv(path)
    missing_cols = REQUIRED_COLUMNS.difference(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset '{dataset_id}' fails schema validation. Missing columns: {sorted(missing_cols)}"
        )

    df = df.copy()
    df["dataset_id"] = dataset_id
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    
    # Unified text derivation: Rule 5
    df["text"] = (
        "subject: " + df["subject"].str.strip() + "\nbody: " + df["body"].str.strip()
    )
    df["text_length"] = df["text"].str.len()
    return df


def load_all_datasets(configs: Dict[str, Dict[str, Any]] = DATASET_CONFIGS) -> Dict[str, pd.DataFrame]:
    """
    Load all registered datasets into a dictionary mapped by dataset_id.

    Parameters
    ----------
    configs : Dict[str, Dict[str, Any]], optional
        Dataset configuration map, defaults to DATASET_CONFIGS.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Map of dataset_id to validated DataFrame.
    """
    return {ds_id: load_dataset(ds_id, cfg) for ds_id, cfg in configs.items()}
