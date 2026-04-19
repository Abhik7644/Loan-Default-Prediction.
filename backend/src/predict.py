"""
predict.py
──────────
Two-stage prediction pipeline:

  Stage 1 — Loan Approval   : Is the applicant eligible?
  Stage 2 — Default Risk     : If approved, what's the default probability?

Returns a structured verdict dict with all details.

Usage (standalone):
    python src/predict.py
"""

import os, sys
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEFAULT_MODEL_PATH, APPROVAL_MODEL_PATH,
    RISK_THRESHOLDS, APPROVAL_RULES, GRADE_MAP
)


# ─── Model loading (lazy, cached) ────────────────────────────────────────────

_default_model  = None
_approval_model = None


def _load_models():
    global _default_model, _approval_model
    if _default_model is None:
        if not os.path.exists(DEFAULT_MODEL_PATH):
            raise FileNotFoundError(
                f"Default model not found at {DEFAULT_MODEL_PATH}. "
                "Run: python src/train.py"
            )
        _default_model = joblib.load(DEFAULT_MODEL_PATH)
    if _approval_model is None:
        if not os.path.exists(APPROVAL_MODEL_PATH):
            raise FileNotFoundError(
                f"Approval model not found at {APPROVAL_MODEL_PATH}. "
                "Run: python src/train.py"
            )
        _approval_model = joblib.load(APPROVAL_MODEL_PATH)


# ─── Risk scoring ─────────────────────────────────────────────────────────────

def _risk_label(prob: float) -> str:
    if prob < RISK_THRESHOLDS["low"]:
        return "Low Risk"
    elif prob < RISK_THRESHOLDS["medium"]:
        return "Medium Risk"
    else:
        return "High Risk"


def _risk_score(prob: float) -> int:
    """Convert default probability to a 0–100 risk score (higher = riskier)."""
    return int(round(prob * 100))


# ─── EMI feasibility (rule-based) ────────────────────────────────────────────

def check_emi_feasibility(annual_inc: float, dti: float,
                          loan_amount: float = None,
                          term_months: int = 36,
                          interest_rate: float = 12.0) -> dict:
    """
    Estimate whether the applicant can afford the monthly EMI.

    Returns a dict with:
        feasible     : bool
        monthly_inc  : float
        current_emi_burden : float (existing obligations from DTI)
        proposed_emi : float | None  (if loan_amount provided)
        recommended_max_loan : float
    """
    monthly_inc = annual_inc / 12
    current_burden = (dti / 100) * monthly_inc      # existing monthly debt

    # Standard affordability: total obligations ≤ 50% of monthly income
    available_for_emi = (0.50 * monthly_inc) - current_burden

    result = {
        "monthly_inc":        round(monthly_inc, 2),
        "current_emi_burden": round(current_burden, 2),
        "available_for_emi":  round(max(available_for_emi, 0), 2),
    }

    if loan_amount is not None:
        r = (interest_rate / 100) / 12
        emi = loan_amount * r * (1 + r)**term_months / ((1 + r)**term_months - 1)
        result["proposed_emi"] = round(emi, 2)
        result["feasible"]     = emi <= available_for_emi
    else:
        result["feasible"] = available_for_emi > 0

    # Recommend max loan based on available EMI capacity
    r = (interest_rate / 100) / 12
    if available_for_emi > 0 and r > 0:
        max_loan = available_for_emi * ((1 + r)**term_months - 1) / (r * (1 + r)**term_months)
    else:
        max_loan = 0.0
    result["recommended_max_loan"] = round(max(max_loan, 0), 2)

    return result


# ─── Loan amount suggestion ───────────────────────────────────────────────────

def suggest_loan_amount(annual_inc: float, dti: float,
                        requested_amount: float,
                        term_months: int = 36,
                        interest_rate: float = 12.0) -> dict:
    """
    If the requested loan is too large, suggest a safer amount.
    """
    emi_info = check_emi_feasibility(
        annual_inc, dti, requested_amount, term_months, interest_rate
    )
    if emi_info.get("feasible", True):
        return {"suggestion": "requested_amount_is_feasible",
                "recommended": requested_amount}
    return {
        "suggestion": "reduce_loan_amount",
        "recommended": emi_info["recommended_max_loan"],
        "reason": (
            f"Your estimated monthly EMI would be "
            f"{emi_info['proposed_emi']:.0f}, but only "
            f"{emi_info['available_for_emi']:.0f} is available after "
            f"existing obligations. Suggested max loan: "
            f"{emi_info['recommended_max_loan']:.0f}"
        )
    }


# ─── Main predict function ────────────────────────────────────────────────────

def predict(applicant: dict, loan_amount: float = None,
            term_months: int = 36) -> dict:
    """
    Full two-stage prediction for a single applicant.

    Parameters
    ----------
    applicant   : dict with keys matching dataset feature names (raw values,
                  e.g. grade='A', term=' 36 months')
    loan_amount : optional; used for EMI feasibility check
    term_months : loan term in months (36 or 60)

    Returns
    -------
    dict with keys:
        approved          : bool
        approval_prob     : float
        default_risk      : str  ("Low Risk" / "Medium Risk" / "High Risk")
        default_prob      : float
        risk_score        : int  (0–100)
        verdict           : str  (human-readable final decision)
        emi_info          : dict (if loan_amount provided)
        loan_suggestion   : dict (if loan_amount provided)
        reasons           : list[str]  (key factors)
    """
    _load_models()

    df = pd.DataFrame([applicant])

    # Normalise grade (A–G string → int 1–7)
    if "grade" in df.columns:
        df["grade"] = df["grade"].astype(str).map(GRADE_MAP)

    # Fix term formatting
    if "term" in df.columns:
        df["term"] = df["term"].str.strip().str.lower()

    # Reorder columns to match training-time feature order
    approval_cols = list(_approval_model.named_steps["preprocessor"].feature_names_in_)
    default_cols  = list(_default_model.named_steps["preprocessor"].feature_names_in_)
    df_approval = df[approval_cols]
    df_default  = df[default_cols]

    # ── Stage 1: Approval ────────────────────────────────────────────────────
    approval_prob = _approval_model.predict_proba(df_approval)[0][1]
    approved      = bool(_approval_model.predict(df_approval)[0])

    reasons = []
    a = applicant

    # Rule-based reason generation
    grade_num = GRADE_MAP.get(str(a.get("grade", "G")).upper(), 1)
    if grade_num < APPROVAL_RULES["min_grade"]:
        reasons.append(f"Grade {a.get('grade')} is below minimum acceptable grade.")
    if float(a.get("dti", 0)) > APPROVAL_RULES["max_dti"]:
        reasons.append(f"DTI {a.get('dti')} exceeds maximum allowed ({APPROVAL_RULES['max_dti']}).")
    if float(a.get("annual_inc", 0)) < APPROVAL_RULES["min_annual_inc"]:
        reasons.append(f"Annual income {a.get('annual_inc')} is below minimum ({APPROVAL_RULES['min_annual_inc']}).")
    if float(a.get("revol_util", 0)) > APPROVAL_RULES["max_revol_util"]:
        reasons.append(f"Revolving utilisation {a.get('revol_util')}% exceeds limit ({APPROVAL_RULES['max_revol_util']}%).")

    result = {
        "approved":      approved,
        "approval_prob": round(float(approval_prob), 4),
        "reasons":       reasons,
    }

    if not approved:
        result["verdict"]      = "❌ Loan Not Approved — applicant does not meet eligibility criteria."
        result["default_risk"] = "N/A"
        result["default_prob"] = None
        result["risk_score"]   = None
        return result

    # ── Stage 2: Default Risk ────────────────────────────────────────────────
    default_prob = _default_model.predict_proba(df_default)[0][1]
    risk_label   = _risk_label(default_prob)
    score        = _risk_score(default_prob)

    result["default_prob"] = round(float(default_prob), 4)
    result["default_risk"] = risk_label
    result["risk_score"]   = score

    # Final verdict
    if risk_label == "Low Risk":
        result["verdict"] = "✅ Approved — Low default risk. Eligible for standard loan terms."
    elif risk_label == "Medium Risk":
        result["verdict"] = "⚠️  Conditionally Approved — Moderate risk. Consider higher interest rate or shorter term."
    else:
        result["verdict"] = "❌ Rejected — High default risk despite meeting basic eligibility."

    # ── Optional EMI & amount checks ─────────────────────────────────────────
    if loan_amount is not None:
        result["emi_info"] = check_emi_feasibility(
            float(a.get("annual_inc", 0)),
            float(a.get("dti", 0)),
            loan_amount, term_months
        )
        result["loan_suggestion"] = suggest_loan_amount(
            float(a.get("annual_inc", 0)),
            float(a.get("dti", 0)),
            loan_amount, term_months
        )

    return result


# ─── CLI demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Sample applicant — mirrors the first row of the dataset
    sample = {
        "grade":               "A",
        "annual_inc":          100000.0,
        "short_emp":           1,
        "emp_length_num":      1,
        "home_ownership":      "RENT",
        "dti":                 26.27,
        "purpose":             "credit_card",
        "term":                " 36 months",
        "last_delinq_none":    1,
        "revol_util":          43.2,
        "total_rec_late_fee":  0.0,
        "od_ratio":            0.1606,
    }

    result = predict(sample, loan_amount=25000, term_months=36)
    print("\n── Prediction Result ──")
    print(json.dumps(result, indent=2))
