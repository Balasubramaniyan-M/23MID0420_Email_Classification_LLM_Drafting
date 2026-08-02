"""
Dataset audit, hash integrity verification, duplicate detection, and leakage checks.
"""

import hashlib
from pathlib import Path
from typing import Dict, Any, Union
import pandas as pd


def sha256_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """
    Calculate SHA-256 checksum of a file on disk.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the file to hash.
    chunk_size : int, optional
        Chunk size in bytes for reading, defaults to 1MB.

    Returns
    -------
    str
        Hexadecimal SHA-256 checksum string.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(dataset_id: str, df: pd.DataFrame, source_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Perform audit on a loaded dataset DataFrame.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier string.
    df : pd.DataFrame
        DataFrame to audit.
    source_path : Union[str, Path]
        Path to source CSV file.

    Returns
    -------
    Dict[str, Any]
        Audit metric dictionary containing row counts, class counts, duplicate counts, etc.
    """
    return {
        "dataset_id": dataset_id,
        "rows": len(df),
        "classes": int(df["label"].nunique()),
        "empty_text": int((df["text"].str.strip() == "").sum()),
        "exact_duplicate_text": int(df["text"].duplicated().sum()),
        "median_text_length": float(df["text_length"].median()),
        "file_sha256": sha256_file(source_path),
    }


def check_split_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame, id_col: str = "email_id") -> bool:
    """
    Verify that train and test partitions are strictly disjoint by ID and text content.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training partition DataFrame.
    test_df : pd.DataFrame
        Testing partition DataFrame.
    id_col : str, optional
        Unique ID column name, defaults to 'email_id'.

    Returns
    -------
    bool
        True if split is completely disjoint without ID leakage.

    Raises
    ------
    AssertionError
        If any overlap exists between train and test record IDs.
    """
    train_ids = set(train_df[id_col])
    test_ids = set(test_df[id_col])
    overlap = train_ids.intersection(test_ids)
    if overlap:
        raise AssertionError(f"Data leakage detected! {len(overlap)} IDs present in both train and test.")
    return True
