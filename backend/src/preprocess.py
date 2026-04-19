"""
preprocess.py
─────────────
Handles all data cleaning, missing-value treatment, encoding, and
outlier removal.  Called by both train.py and predict.py.
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_RAW, DATA_PROC,
    TARGET_COL, DROP_COLS, GRADE_MAP,
    CATEGORICAL_COLS, NUMERICAL_COLS
)


# ─── Loading ─────────────────────────────────────────────────────────────────

def load_raw(path: str = DATA_RAW) -> pd.DataFrame:
    """Load raw CSV and return a DataFrame."""
    df = pd.read_csv(path, low_memory=False)
    print(f"[preprocess] Loaded {df.shape[0]:,} rows × {df.shape[1]} cols from {path}")
    return df


# ─── Cleaning ────────────────────────────────────────────────────────────────

def drop_irrelevant(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that add no predictive value."""
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def fix_typos(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise categorical values with known typos."""
    if "term" in df.columns:
        df["term"] = df["term"].str.strip().str.lower()
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
      - Numerical  → mean
      - Categorical → mode
    """
    for col in NUMERICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    IQR-based outlier removal on heavily skewed numeric cols.
    Only applied when the target column is present (training time).
    """
    skewed = ["revol_util"]
    for col in skewed:
        if col not in df.columns:
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    if "total_rec_late_fee" in df.columns:
        cap = df["total_rec_late_fee"].quantile(0.989)
        df = df[df["total_rec_late_fee"] < cap]

    return df


# ─── Encoding ────────────────────────────────────────────────────────────────

def encode_grade(df: pd.DataFrame) -> pd.DataFrame:
    """Map letter grade (A–G) to numeric 7–1."""
    if "grade" in df.columns:
        df["grade"] = df["grade"].map(GRADE_MAP)
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    return pd.get_dummies(df, columns=cols)


# ─── Feature Engineering ─────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derived features that improve model performance.

    emi_to_income  — estimated monthly EMI burden relative to income.
                     Uses od_ratio as a proxy for obligation-to-debt.
    credit_risk_score — weighted composite of the strongest default signals
                        (grade, dti, revol_util).  Higher = riskier.
    """
    if "annual_inc" in df.columns and "dti" in df.columns:
        # Monthly debt payment estimated from DTI × monthly income
        monthly_inc = df["annual_inc"] / 12
        df["emi_to_income"] = (df["dti"] / 100) * monthly_inc / (monthly_inc + 1e-9)

    if all(c in df.columns for c in ["grade", "dti", "revol_util"]):
        # grade already numeric 1–7; invert so higher grade → lower risk
        grade_risk = (8 - df["grade"]) / 7          # 0 (A) … 1 (G)
        dti_norm   = df["dti"].clip(0, 50) / 50      # 0 … 1
        ru_norm    = df["revol_util"].clip(0, 100) / 100
        df["credit_risk_score"] = (0.4 * grade_risk + 0.35 * dti_norm + 0.25 * ru_norm)

    return df


# ─── Full pipeline ────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline.

    Parameters
    ----------
    df       : raw DataFrame
    training : if True, removes outliers and keeps the target column
    """
    df = df.copy()
    df = drop_irrelevant(df)
    df = fix_typos(df)
    df = handle_missing(df)
    if training:
        df = remove_outliers(df)
    df = encode_grade(df)
    df = add_features(df)
    df = one_hot_encode(df)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df


def preprocess_and_save(raw_path: str = DATA_RAW,
                        out_path: str = DATA_PROC) -> pd.DataFrame:
    """Convenience wrapper: load → preprocess → save to disk."""
    df_raw  = load_raw(raw_path)
    df_proc = preprocess(df_raw, training=True)
    df_proc.to_csv(out_path, index=False)
    print(f"[preprocess] Saved processed data → {out_path}  shape={df_proc.shape}")
    return df_proc


if __name__ == "__main__":
    preprocess_and_save()
