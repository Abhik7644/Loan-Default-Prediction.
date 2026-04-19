"""
evaluate.py
───────────
Compare multiple classifiers side-by-side and produce evaluation plots.
Run standalone after training to audit model performance.

    python src/evaluate.py
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_RAW, MODEL_DIR, DEFAULT_MODEL_PATH,
    TARGET_COL, CATEGORICAL_COLS, NUMERICAL_COLS,
    GRADE_MAP, DROP_COLS, RANDOM_STATE, TEST_SIZE
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    accuracy_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# ─── Data prep (mirrors train.py lightweight prep) ────────────────────────────

def _prepare_data():
    df = pd.read_csv(DATA_RAW, low_memory=False)
    for col in DROP_COLS:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    df["term"]  = df["term"].str.strip().str.lower()
    df["grade"] = df["grade"].map(GRADE_MAP)
    df.dropna(subset=[TARGET_COL], inplace=True)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    num_cols = [c for c in NUMERICAL_COLS if c in X.columns] + ["grade"]
    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]

    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=y), num_cols, cat_cols


def _build_pipe(clf, num_cols, cat_cols):
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="mean")),
                          ("sc",  MinMaxScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                sparse_output=False))]), cat_cols),
    ])
    return ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote",  SMOTE(sampling_strategy="minority", random_state=RANDOM_STATE)),
        ("model",  clf),
    ])


# ─── Model comparison ─────────────────────────────────────────────────────────

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=500, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE),
}


def compare_models(save_dir: str = MODEL_DIR) -> pd.DataFrame:
    """Train each classifier and return a comparison DataFrame."""
    (X_train, X_test, y_train, y_test), num_cols, cat_cols = _prepare_data()

    rows = []
    fitted = {}

    for name, clf in CLASSIFIERS.items():
        print(f"[evaluate] Fitting {name}…")
        pipe = _build_pipe(clf, num_cols, cat_cols)
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        rows.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall":    round(recall_score(y_test, y_pred),    4),
            "F1":        round(f1_score(y_test, y_pred),        4),
            "ROC-AUC":   round(roc_auc_score(y_test, y_proba),  4),
        })
        fitted[name] = (pipe, y_test, y_proba)

    results = pd.DataFrame(rows).sort_values("F1", ascending=False)
    print("\n── Model Comparison ──")
    print(results.to_string(index=False))

    _plot_roc(fitted, save_dir)
    _plot_comparison_bar(results, save_dir)
    return results


# ─── Plots ────────────────────────────────────────────────────────────────────

def _plot_roc(fitted: dict, save_dir: str):
    plt.figure(figsize=(8, 6))
    for name, (pipe, y_test, y_proba) in fitted.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "roc_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[evaluate] ROC plot saved → {path}")


def _plot_comparison_bar(results: pd.DataFrame, save_dir: str):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    x = np.arange(len(results))
    width = 0.15

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, results[m], width, label=m)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(results["Model"], rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Comparison")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[evaluate] Bar chart saved → {path}")


def plot_confusion_matrix(model_path: str = DEFAULT_MODEL_PATH,
                          save_dir: str = MODEL_DIR):
    """Load the saved best model and plot its confusion matrix."""
    if not os.path.exists(model_path):
        print(f"[evaluate] Model not found: {model_path}")
        return

    (X_train, X_test, y_train, y_test), _, _ = _prepare_data()
    model   = joblib.load(model_path)
    y_pred  = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — Best Default Model")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[evaluate] Confusion matrix saved → {path}")


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    compare_models()
    plot_confusion_matrix()
