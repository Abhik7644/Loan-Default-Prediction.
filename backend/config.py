import os

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_RAW   = os.path.join(BASE_DIR, "data", "raw", "dataset.csv")
DATA_PROC  = os.path.join(BASE_DIR, "data", "processed", "processed.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
DEFAULT_MODEL_PATH  = os.path.join(MODEL_DIR, "default_pipeline.pkl")
APPROVAL_MODEL_PATH = os.path.join(MODEL_DIR, "approval_pipeline.pkl")

# ─── Feature definitions ────────────────────────────────────────────────────
TARGET_COL = "bad_loan"

CATEGORICAL_COLS = ["term", "home_ownership", "purpose"]
ORDINAL_COLS     = ["grade"]           # encoded as 1–7
NUMERICAL_COLS   = [
    "annual_inc", "short_emp", "emp_length_num",
    "dti", "last_delinq_none", "revol_util",
    "total_rec_late_fee", "od_ratio"
]

GRADE_MAP = {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}

DROP_COLS = ["id", "last_major_derog_none"]

# ─── Loan-approval thresholds (rule-based pre-filter) ───────────────────────
APPROVAL_RULES = {
    "max_dti":          40.0,     # debt-to-income ratio ceiling
    "min_annual_inc":   15000.0,  # minimum annual income (₹ or $)
    "min_grade":        2,        # grade F or above (G=1 is rejected)
    "max_revol_util":   90.0,     # revolving utilisation ceiling (%)
}

# Risk-score thresholds (probability of default → verdict)
RISK_THRESHOLDS = {
    "low":    0.30,   # p < 30%  → Low Risk   → Approve
    "medium": 0.55,   # 30–55%   → Medium Risk → Approve with conditions
    # above 55%       → High Risk → Reject
}

# ─── Model hyperparams ──────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
SMOTE_STRATEGY = "minority"

RF_PARAM_GRID = {
    "model__n_estimators":    [100, 300],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf":  [1, 2],
    "model__max_depth":       [None, 20],
}
