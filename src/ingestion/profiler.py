"""
Layer 1: Data Ingestion and Profiling
--------------------------------------
Reads the uploaded CSV file and builds a compact
JSON profile that is sent to the LLM for planning.
"""

import pandas as pd
import numpy as np
import json


def load_csv(file) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    Handles multiple encodings automatically.
    """
    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    for enc in encodings:
        try:
            df = pd.read_csv(file, encoding=enc)
            return df
        except Exception:
            continue
    raise ValueError("Could not read the CSV file. Please check the file format.")


def profile_dataframe(df: pd.DataFrame) -> dict:
    """
    Build a compact JSON profile of the DataFrame.
    This profile is sent to the LLM so it can plan the analysis.
    
    Returns a dictionary with:
    - shape: rows and columns
    - columns: name, type, nulls, unique values, sample values
    - numeric_summary: basic stats for numeric columns
    - categorical_summary: top values for categorical columns
    """
    profile = {
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1])
        },
        "columns": [],
        "numeric_summary": {},
        "categorical_summary": {},
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_pct": round(df[col].isnull().mean() * 100, 2),
            "unique_count": int(df[col].nunique()),
            "sample_values": df[col].dropna().head(3).tolist()
        }

        # Always convert to plain strings to avoid PyArrow mixed-type errors
        col_info["sample_values"] = [str(v) for v in col_info["sample_values"]]
        profile["columns"].append(col_info)

        # Numeric summary
        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            profile["numeric_summary"][col] = {
                "mean":   round(float(desc["mean"]), 4) if not np.isnan(desc["mean"]) else None,
                "std":    round(float(desc["std"]),  4) if not np.isnan(desc["std"])  else None,
                "min":    round(float(desc["min"]),  4),
                "25pct":  round(float(desc["25%"]),  4),
                "median": round(float(desc["50%"]),  4),
                "75pct":  round(float(desc["75%"]),  4),
                "max":    round(float(desc["max"]),  4),
            }

        # Categorical summary
        elif df[col].dtype == object or str(df[col].dtype) == "category":
            top = df[col].value_counts().head(5)
            profile["categorical_summary"][col] = {
                k: int(v) for k, v in top.items()
            }

    return profile


def profile_to_text(profile: dict) -> str:
    """
    Convert profile dict to a readable text summary
    for the LLM prompt.
    """
    lines = []
    lines.append(f"Dataset Shape: {profile['shape']['rows']} rows × {profile['shape']['columns']} columns")
    lines.append(f"Total Missing Values: {profile['missing_total']}")
    lines.append(f"Duplicate Rows: {profile['duplicate_rows']}")
    lines.append("")
    lines.append("COLUMNS:")
    for col in profile["columns"]:
        lines.append(
            f"  - {col['name']} | type: {col['dtype']} | "
            f"nulls: {col['null_pct']}% | unique: {col['unique_count']} | "
            f"sample: {col['sample_values']}"
        )
    lines.append("")
    lines.append("NUMERIC STATISTICS:")
    for col, stats in profile["numeric_summary"].items():
        lines.append(
            f"  {col}: mean={stats['mean']}, std={stats['std']}, "
            f"min={stats['min']}, median={stats['median']}, max={stats['max']}"
        )
    lines.append("")
    lines.append("TOP CATEGORICAL VALUES:")
    for col, vals in profile["categorical_summary"].items():
        lines.append(f"  {col}: {vals}")

    return "\n".join(lines)
