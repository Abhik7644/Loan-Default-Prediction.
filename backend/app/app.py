"""
app/app.py
──────────
Streamlit web application for the Loan Default Prediction system.

Features:
  • Applicant input form
  • Stage 1 — Loan Eligibility check
  • Stage 2 — Default Risk prediction with risk score
  • EMI feasibility check with loan amount suggestion
  • Feature importance sidebar

Run:
    streamlit run app/app.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DEFAULT_MODEL_PATH, APPROVAL_MODEL_PATH,
    GRADE_MAP, RISK_THRESHOLDS
)
from src.predict import predict, check_emi_feasibility


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Loan Prediction System",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Loan Prediction System")
st.markdown(
    "**Two-stage pipeline:** Eligibility Check → Default Risk Assessment"
)
st.divider()


# ─── Sidebar — applicant inputs ───────────────────────────────────────────────

st.sidebar.header("📋 Applicant Details")

grade        = st.sidebar.selectbox("Credit Grade", ["A","B","C","D","E","F","G"])
annual_inc   = st.sidebar.number_input("Annual Income ($)", min_value=0.0,
                                        value=60000.0, step=1000.0)
emp_length   = st.sidebar.slider("Employment Length (years)", 0, 10, 5)
short_emp    = st.sidebar.selectbox("Employed < 1 Year?", [0, 1],
                                     format_func=lambda x: "Yes" if x else "No")
home_own     = st.sidebar.selectbox("Home Ownership",
                                     ["RENT", "OWN", "MORTGAGE"])
dti          = st.sidebar.slider("Debt-to-Income Ratio (%)", 0.0, 60.0, 15.0, 0.1)
purpose      = st.sidebar.selectbox("Loan Purpose", [
    "credit_card","debt_consolidation","home_improvement",
    "major_purchase","medical","other","small_business",
    "vacation","car","house","moving","wedding",
])
term         = st.sidebar.selectbox("Loan Term", [" 36 months", " 60 months"])
last_delinq  = st.sidebar.selectbox("Previous Delinquency?", [0, 1],
                                     format_func=lambda x: "Yes" if x else "No")
revol_util   = st.sidebar.slider("Revolving Utilisation (%)", 0.0, 100.0, 40.0, 0.1)
late_fee     = st.sidebar.number_input("Total Late Fees Received", 0.0, 5000.0, 0.0)
od_ratio     = st.sidebar.slider("OD Ratio", 0.0, 2.0, 0.5, 0.01)

st.sidebar.divider()
st.sidebar.subheader("💰 Loan Request (optional)")
loan_amount  = st.sidebar.number_input("Requested Loan Amount ($)",
                                        min_value=0.0, value=0.0, step=500.0)
term_months  = int(term.strip().split()[0])

predict_btn  = st.sidebar.button("🔍 Predict", use_container_width=True)


# ─── Prediction ───────────────────────────────────────────────────────────────

if predict_btn:
    models_ready = (
        os.path.exists(DEFAULT_MODEL_PATH) and
        os.path.exists(APPROVAL_MODEL_PATH)
    )

    if not models_ready:
        st.error(
            "⚠️ Models not found. Please run `python src/train.py` first "
            "to train and save the models."
        )
        st.stop()

    applicant = {
        "grade":              grade,
        "annual_inc":         annual_inc,
        "short_emp":          short_emp,
        "emp_length_num":     emp_length,
        "home_ownership":     home_own,
        "dti":                dti,
        "purpose":            purpose,
        "term":               term,
        "last_delinq_none":   last_delinq,
        "revol_util":         revol_util,
        "total_rec_late_fee": late_fee,
        "od_ratio":           od_ratio,
    }

    result = predict(
        applicant,
        loan_amount=loan_amount if loan_amount > 0 else None,
        term_months=term_months,
    )

    # ── Layout: 3 columns ──────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1.2, 1.2, 1.6])

    with col1:
        st.subheader("Stage 1 — Eligibility")
        if result["approved"]:
            st.success("✅ Eligible for Loan")
        else:
            st.error("❌ Not Eligible")

        st.metric("Approval Confidence",
                  f"{result['approval_prob']*100:.1f}%")

        if result["reasons"]:
            st.warning("**Issues flagged:**")
            for r in result["reasons"]:
                st.markdown(f"- {r}")
        else:
            st.info("No eligibility issues found.")

    with col2:
        st.subheader("Stage 2 — Default Risk")
        if result["default_prob"] is not None:
            prob = result["default_prob"]
            score = result["risk_score"]
            risk  = result["default_risk"]

            colour = {"Low Risk": "green", "Medium Risk": "orange",
                      "High Risk": "red"}.get(risk, "grey")
            st.markdown(
                f"<h3 style='color:{colour}'>{risk}</h3>",
                unsafe_allow_html=True
            )
            st.metric("Default Probability", f"{prob*100:.1f}%")
            st.metric("Risk Score (0–100)", score)

            # Simple risk bar
            fig, ax = plt.subplots(figsize=(4, 0.5))
            ax.barh([""], [score], color=colour, height=0.4)
            ax.barh([""], [100 - score], left=[score],
                    color="#e0e0e0", height=0.4)
            ax.set_xlim(0, 100)
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.info("Stage 2 skipped — applicant not eligible.")

    with col3:
        st.subheader("📝 Verdict")
        st.markdown(f"### {result['verdict']}")

        if "emi_info" in result:
            st.divider()
            st.subheader("💳 EMI Feasibility")
            ei = result["emi_info"]
            c1, c2 = st.columns(2)
            c1.metric("Monthly Income",  f"${ei['monthly_inc']:,.0f}")
            c2.metric("Existing Burden", f"${ei['current_emi_burden']:,.0f}/mo")
            c1.metric("Available for EMI", f"${ei['available_for_emi']:,.0f}/mo")
            if "proposed_emi" in ei:
                c2.metric("Proposed EMI", f"${ei['proposed_emi']:,.0f}/mo")
                if ei["feasible"]:
                    st.success("✅ EMI is affordable.")
                else:
                    st.warning("⚠️ EMI may strain finances.")

        if "loan_suggestion" in result:
            ls = result["loan_suggestion"]
            if ls["suggestion"] == "reduce_loan_amount":
                st.warning(
                    f"💡 **Suggested loan:** ${ls['recommended']:,.0f}  \n"
                    f"{ls['reason']}"
                )
            else:
                st.success("✅ Requested loan amount is financially feasible.")

    st.divider()
    with st.expander("🔎 Full prediction details (JSON)"):
        st.json(result)

else:
    st.info(
        "👈 Fill in the applicant details on the left and click **Predict**."
    )
    st.markdown("""
    ### How it works
    1. **Stage 1 — Eligibility**: Checks grade, DTI, income, and revolving
       utilisation against bank policy thresholds.
    2. **Stage 2 — Default Risk**: If eligible, a Random Forest model estimates
       the probability of default. The result is a risk score (0–100) and a
       Low / Medium / High risk label.
    3. **EMI Feasibility**: If a loan amount is entered, the system checks whether
       the monthly EMI is affordable and suggests a safer amount if needed.
    """)
