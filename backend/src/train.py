"""
train.py
────────
Trains two models and saves them as .pkl pipelines:

  1. Default-risk model   → predicts bad_loan (0/1)
  2. Loan-approval model  → predicts loan_approved (0/1)
     (approval label is derived from the rules in config.py)

Run:
    python src/train.py
"""

import os, sys
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_RAW, MODEL_DIR,
    DEFAULT_MODEL_PATH, APPROVAL_MODEL_PATH,
    TARGET_COL, CATEGORICAL_COLS, NUMERICAL_COLS,
    GRADE_MAP, DROP_COLS, APPROVAL_RULES,
    RANDOM_STATE, TEST_SIZE, SMOTE_STRATEGY, RF_PARAM_GRID
)
from src.preprocess import load_raw, preprocess

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score, confusion_matrix
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# ─── Helpers ─────────────────────────────────────────────────────────────────

def build_sklearn_pipeline(numerical_cols, categorical_cols):
    """
    Build a reusable sklearn ColumnTransformer + RandomForest pipeline.
    Preprocessing is done inside the pipeline so predict() works on raw input.
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  MinMaxScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer,  numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])
    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote",  SMOTE(sampling_strategy=SMOTE_STRATEGY, random_state=RANDOM_STATE)),
        ("model",  RandomForestClassifier(random_state=RANDOM_STATE)),
    ])
    return pipeline


def evaluate(y_true, y_pred, y_proba, label="Model"):
    """Print a standard evaluation block."""
    print(f"\n{'='*50}")
    print(f"  {label}  Evaluation")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=["No","Yes"]))
    print(f"  F1 Score : {f1_score(y_true, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_true, y_proba):.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")


# ─── Default-risk model ───────────────────────────────────────────────────────

def train_default_model(df_raw: pd.DataFrame) -> None:
    """Train bad_loan classifier and save pipeline."""
    print("\n[train] ── Training Default-Risk Model ──")

    # Light preprocessing (grade map + typo fix) — no one-hot yet (pipeline handles it)
    df = df_raw.copy()
    for col in DROP_COLS:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    df["term"] = df["term"].str.strip().str.lower()
    df["grade"] = df["grade"].map(GRADE_MAP)
    df.dropna(subset=[TARGET_COL], inplace=True)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    # Identify actual columns present
    num_cols = [c for c in NUMERICAL_COLS if c in X.columns] + ["grade"]
    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipe = build_sklearn_pipeline(num_cols, cat_cols)

    # Hyperparameter search
    search = RandomizedSearchCV(
        pipe, RF_PARAM_GRID,
        n_iter=8, cv=3, scoring="f1",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    y_pred  = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    evaluate(y_test, y_pred, y_proba, label="Default-Risk")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best, DEFAULT_MODEL_PATH)
    print(f"[train] Saved default model → {DEFAULT_MODEL_PATH}")


# ─── Loan-approval model ──────────────────────────────────────────────────────

def create_approval_label(df: pd.DataFrame) -> pd.Series:
    """
    Rule-based approval label derived from APPROVAL_RULES in config.
    1 = eligible for loan, 0 = not eligible.
    """
    rules = APPROVAL_RULES
    approved = (
        (df["dti"].fillna(999)          <= rules["max_dti"])         &
        (df["annual_inc"].fillna(0)     >= rules["min_annual_inc"])  &
        (df["grade"].map(GRADE_MAP).fillna(0) >= rules["min_grade"]) &
        (df["revol_util"].fillna(999)   <= rules["max_revol_util"])
    )
    return approved.astype(int)


def train_approval_model(df_raw: pd.DataFrame) -> None:
    """Train loan-approval classifier and save pipeline."""
    print("\n[train] ── Training Loan-Approval Model ──")

    df = df_raw.copy()
    for col in DROP_COLS:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    df["term"] = df["term"].str.strip().str.lower()

    # Create approval label
    df["loan_approved"] = create_approval_label(df)
    df["grade"] = df["grade"].map(GRADE_MAP)

    X = df.drop(columns=[TARGET_COL, "loan_approved"])
    y = df["loan_approved"].astype(int)

    num_cols = [c for c in NUMERICAL_COLS if c in X.columns] + ["grade"]
    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipe = build_sklearn_pipeline(num_cols, cat_cols)

    search = RandomizedSearchCV(
        pipe, RF_PARAM_GRID,
        n_iter=8, cv=3, scoring="f1",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    y_pred  = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    evaluate(y_test, y_pred, y_proba, label="Loan-Approval")

    joblib.dump(best, APPROVAL_MODEL_PATH)
    print(f"[train] Saved approval model → {APPROVAL_MODEL_PATH}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def train_all():
    df_raw = pd.read_csv(DATA_RAW, low_memory=False)
    train_default_model(df_raw)
    train_approval_model(df_raw)
    print("\n[train] ✓ Both models trained and saved.")


if __name__ == "__main__":
    train_all()
