# 🏦 Loan Default Prediction System

A two-stage machine learning pipeline that predicts:
1. **Loan Eligibility** — Is the applicant eligible for a loan?
2. **Default Risk** — If approved, what is the probability of default?

---

## 📁 Project Structure

```
Loan-Default-Prediction/
│
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── Requirements.txt
│   ├── README.md
│   │
│   ├── src/
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   │
│   ├── models/
│   │   ├── default_pipeline.pkl
│   │   ├── approval_pipeline.pkl
│   │   ├── roc_comparison.png
│   │   ├── model_comparison.png
│   │   └── confusion_matrix.png
│   │
│   ├── notebooks/
│   │   └── Loan_Default_Prediction_model.ipynb
│   │
│   ├── data/
│   │   └── raw/
│   │       └── dataset.csv
│   │
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
```

---

## ⚙️ Setup

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r Requirements.txt
```

Train the models (only needed once):

```bash
python src/train.py
```

Start the Flask API:

```bash
python app.py
```

API runs at: **http://localhost:5000**

---

### Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

UI runs at: **http://localhost:5173**

---

## 🔁 Two-Stage Prediction Pipeline

```
Applicant Input
      │
      ▼
┌─────────────────────────┐
│  Stage 1: Eligibility   │  Rule-based + ML model
│  - Grade >= F           │
│  - DTI <= 40%           │  →  Rejected (not eligible)
│  - Income >= $15,000    │
│  - Revol. util <= 90%   │
└────────────┬────────────┘
             │ Approved
             ▼
┌─────────────────────────┐
│  Stage 2: Default Risk  │  Random Forest Classifier
│                         │
│  Risk Score  (0-100)    │  →  Low Risk    (p < 30%)
│  Default Probability    │  →  Medium Risk (30-55%)
│                         │  →  High Risk   (p > 55%)
└─────────────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---|---|
| Two-stage pipeline | Eligibility check then default risk assessment |
| Risk score (0-100) | Probability of default as an intuitive score |
| EMI feasibility | Checks if applicant can afford the monthly EMI |
| Loan amount suggestion | Recommends a safer loan amount if request is too large |
| Feature engineering | emi_to_income and credit_risk_score derived features |
| Model comparison | LR vs Decision Tree vs Random Forest with AUC/F1/Recall |
| SMOTE balancing | Handles 80/20 class imbalance with oversampling |
| React frontend | Interactive web form with live predictions |
| Flask REST API | Clean API endpoints consumed by the React frontend |
| Unit tests | 12 tests covering preprocessing, prediction and EMI logic |

---

## 📊 Dataset

| Feature | Description |
|---|---|
| grade | LC-assigned loan grade (A-G) |
| annual_inc | Self-reported annual income |
| short_emp | 1 if employed less than 1 year |
| emp_length_num | Employment length in years (0-10) |
| home_ownership | RENT / OWN / MORTGAGE |
| dti | Debt-to-income ratio |
| purpose | Loan purpose (12 categories) |
| term | 36 or 60 months |
| last_delinq_none | 1 if borrower had prior delinquency |
| revol_util | Revolving credit utilisation % |
| total_rec_late_fee | Late fees received to date |
| od_ratio | Obligation-to-debt ratio |
| bad_loan | Target — 1 if loan defaulted |

**Class balance:** 80% non-default / 20% default — handled via SMOTE.

---

## 🧪 Model Results

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.658 | 0.431 | 0.713 |
| Decision Tree | 0.718 | 0.355 | 0.673 |
| Random Forest | 0.764 | 0.371 | 0.702 |

---

## 🔌 API Reference

### POST /api/predict

**Request:**
```json
{
  "grade": "B",
  "annual_inc": 60000,
  "short_emp": 0,
  "emp_length_num": 5,
  "home_ownership": "RENT",
  "dti": 15.0,
  "purpose": "debt_consolidation",
  "term": " 36 months",
  "last_delinq_none": 1,
  "revol_util": 40.0,
  "total_rec_late_fee": 0.0,
  "od_ratio": 0.5,
  "loan_amount": 25000,
  "term_months": 36
}
```

**Response:**
```json
{
  "approved": true,
  "approval_prob": 0.98,
  "default_risk": "Low Risk",
  "default_prob": 0.12,
  "risk_score": 12,
  "verdict": "Approved - Low default risk.",
  "reasons": [],
  "emi_info": {
    "monthly_inc": 5000.0,
    "current_emi_burden": 750.0,
    "available_for_emi": 1750.0,
    "proposed_emi": 830.36,
    "feasible": true,
    "recommended_max_loan": 52800.0
  }
}
```

### GET /api/health
```json
{ "status": "ok", "message": "Loan Prediction API is running" }
```

### GET /api/model-stats
Returns model comparison metrics and dataset info.

---

## 🚀 Common Issues

| Problem | Fix |
|---|---|
| CORS error in browser | Make sure Flask is running on port 5000 |
| Cannot connect to backend | Run python app.py inside backend/ |
| ModuleNotFoundError | Run pip install -r Requirements.txt with venv active |
| Model not found | Run python src/train.py first |
| npm not found | Install Node.js from https://nodejs.org |