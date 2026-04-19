"""
tests/test_predict.py
─────────────────────
Unit tests for the prediction and preprocessing logic.
Run: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from src.preprocess import fix_typos, handle_missing, encode_grade, add_features
from src.predict    import check_emi_feasibility, suggest_loan_amount, _risk_label, _risk_score
from config         import RISK_THRESHOLDS


# ─── Preprocess tests ────────────────────────────────────────────────────────

def test_fix_typos_term():
    df = pd.DataFrame({"term": [" 36 Months", " 60 months", " 36 months"]})
    result = fix_typos(df)
    assert all(result["term"] == result["term"].str.lower())


def test_handle_missing_numerical():
    df = pd.DataFrame({"annual_inc": [50000, None, 30000], "dti": [10, 20, None]})
    result = handle_missing(df)
    assert result["annual_inc"].isnull().sum() == 0
    assert result["dti"].isnull().sum() == 0


def test_encode_grade():
    df = pd.DataFrame({"grade": ["A", "B", "G"]})
    result = encode_grade(df)
    assert result.loc[0, "grade"] == 7
    assert result.loc[1, "grade"] == 6
    assert result.loc[2, "grade"] == 1


def test_add_features_columns():
    df = pd.DataFrame({
        "annual_inc": [60000],
        "dti":        [20],
        "grade":      [5],
        "revol_util": [40],
    })
    result = add_features(df)
    assert "emi_to_income" in result.columns
    assert "credit_risk_score" in result.columns


# ─── EMI feasibility tests ───────────────────────────────────────────────────

def test_emi_feasibility_affordable():
    result = check_emi_feasibility(
        annual_inc=120000, dti=10, loan_amount=10000, term_months=36
    )
    assert "feasible" in result
    assert result["monthly_inc"] == pytest.approx(10000.0, rel=0.01)


def test_emi_feasibility_not_affordable():
    result = check_emi_feasibility(
        annual_inc=20000, dti=45, loan_amount=50000, term_months=36
    )
    assert result["feasible"] == False


def test_suggest_loan_reduces_for_high_amount():
    result = suggest_loan_amount(
        annual_inc=24000, dti=40, requested_amount=100000
    )
    assert result["suggestion"] == "reduce_loan_amount"
    assert result["recommended"] < 100000


def test_suggest_loan_ok_for_small_amount():
    result = suggest_loan_amount(
        annual_inc=200000, dti=5, requested_amount=5000
    )
    assert result["suggestion"] == "requested_amount_is_feasible"


# ─── Risk scoring tests ───────────────────────────────────────────────────────

def test_risk_label_low():
    assert _risk_label(RISK_THRESHOLDS["low"] - 0.01) == "Low Risk"

def test_risk_label_medium():
    assert _risk_label(RISK_THRESHOLDS["low"] + 0.01) == "Medium Risk"

def test_risk_label_high():
    assert _risk_label(RISK_THRESHOLDS["medium"] + 0.01) == "High Risk"

def test_risk_score_range():
    for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
        s = _risk_score(p)
        assert 0 <= s <= 100
