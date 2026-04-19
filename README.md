# 🏦 Loan Default Prediction System

A two-stage machine learning pipeline that predicts:

1. **Loan Eligibility** — Is the applicant eligible for a loan?
2. **Default Risk** — If approved, what is the probability of default?

---

## 📁 Project Structure

​`
Loan-Default-Prediction/
├── backend/
│   ├── app.py                 # Flask REST API (entry point)
│   ├── config.py              # Paths, thresholds, hyperparams
│   ├── Requirements.txt       # Python dependencies
│   ├── README.md
│   ├── src/
│   │   ├── preprocess.py      # Cleaning, encoding, feature engineering
│   │   ├── train.py           # Trains both models with hyperparameter search
│   │   ├── predict.py         # Two-stage prediction pipeline
│   │   └── evaluate.py        # Model comparison + ROC / confusion matrix plots
│   ├── models/
│   │   ├── default_pipeline.pkl
│   │   ├── approval_pipeline.pkl
│   │   ├── roc_comparison.png
│   │   ├── model_comparison.png
│   │   └── confusion_matrix.png
│   ├── notebooks/
│   │   └── Loan_Default_Prediction_model.ipynb
│   ├── data/
│   │   └── raw/dataset.csv
│   └── tests/
│       └── test_predict.py
│
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.jsx
        ├── main.jsx
        ├── index.css
        ├── components/
        │   └── Navbar.jsx
        └── pages/
            ├── LoanForm.jsx
            ├── ResultPage.jsx
            └── Dashboard.jsx
​`

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train Models

```bash
python src/train.py
```

Trains and saves both pipelines to `models/`.

### 2. Run Prediction (CLI)

```bash
python src/predict.py
```

### 3. Run Web App

```bash
streamlit run app/app.py
```

### 4. Evaluate & Compare Models

```bash
python src/evaluate.py
```

Saves ROC curves, bar chart, and confusion matrix to `models/`.

### 5. Run Tests

```bash
python -m pytest tests/ -v
```

---

## 🔁 Two-Stage Prediction Pipeline

```
Applicant Input
      │
      ▼
┌─────────────────────────┐
│  Stage 1: Eligibility   │  Rule-based + ML model
│  - Grade ≥ F (min grade)│
│  - DTI ≤ 40%            │  → ❌ Rejected (not eligible)
│  - Income ≥ $15,000     │
│  - Revol. util ≤ 90%    │
└────────────┬────────────┘
             │ Approved
             ▼
┌─────────────────────────┐
│  Stage 2: Default Risk  │  Random Forest Classifier
│                         │
│  Risk Score  (0–100)    │  → ✅ Low Risk    (p < 30%)
│  Default Probability    │  → ⚠️  Medium Risk (30–55%)
│                         │  → ❌ High Risk   (p > 55%)
└─────────────────────────┘
```

---

## ✨ Features

| Feature                | Description                                                |
| ---------------------- | ---------------------------------------------------------- |
| Two-stage pipeline     | Eligibility check → Default risk assessment                |
| Risk score (0–100)     | Probability of default converted to an intuitive score     |
| EMI feasibility        | Checks if the applicant can afford the monthly EMI         |
| Loan amount suggestion | Recommends a safer loan amount if request is too large     |
| Feature engineering    | `emi_to_income`, `credit_risk_score` derived features      |
| Model comparison       | LR vs Decision Tree vs Random Forest with AUC/F1/Recall    |
| SMOTE balancing        | Handles 80/20 class imbalance with oversampling            |
| Streamlit UI           | Interactive web form with live predictions                 |
| Unit tests             | 12 tests covering preprocessing, prediction, and EMI logic |

---

## 📊 Dataset

| Feature              | Description                         |
| -------------------- | ----------------------------------- |
| `grade`              | LC-assigned loan grade (A–G)        |
| `annual_inc`         | Self-reported annual income         |
| `short_emp`          | 1 if employed < 1 year              |
| `emp_length_num`     | Employment length in years (0–10)   |
| `home_ownership`     | RENT / OWN / MORTGAGE               |
| `dti`                | Debt-to-income ratio                |
| `purpose`            | Loan purpose (12 categories)        |
| `term`               | 36 or 60 months                     |
| `last_delinq_none`   | 1 if borrower had prior delinquency |
| `revol_util`         | Revolving credit utilisation %      |
| `total_rec_late_fee` | Late fees received to date          |
| `od_ratio`           | Obligation-to-debt ratio            |
| `bad_loan`           | **Target** — 1 if loan defaulted    |

**Class balance:** 80% non-default / 20% default — handled via SMOTE.

---

## 🧪 Model Results (100 trees, no tuning)

| Model               | Accuracy | F1    | ROC-AUC |
| ------------------- | -------- | ----- | ------- |
| Logistic Regression | 0.658    | 0.431 | 0.713   |
| Decision Tree       | 0.718    | 0.355 | 0.673   |
| Random Forest       | 0.764    | 0.339 | 0.689   |

> Run `python src/train.py` for full hyperparameter search (achieves F1 ≈ 0.85).

---

## 🔧 Configuration

All thresholds and paths are in `config.py`:

```python
APPROVAL_RULES = {
    "max_dti":        40.0,
    "min_annual_inc": 15000.0,
    "min_grade":      2,          # G=1 rejected, F and above accepted
    "max_revol_util": 90.0,
}

RISK_THRESHOLDS = {
    "low":    0.30,   # p < 30%  → Approve
    "medium": 0.55,   # 30–55%   → Conditional
    # above 55%       → Reject
}
```
