"""
app.py  —  Flask REST API for Loan Prediction System
-----------------------------------------------------
Endpoints:
  POST /api/predict        — two-stage prediction
  GET  /api/health         — health check
  GET  /api/model-stats    — model evaluation metrics
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os, sys, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predict import predict

app = Flask(__name__)
CORS(app)   # allow React frontend on localhost:5173


# ─── Health ──────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Loan Prediction API is running"})


# ─── Main prediction endpoint ─────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def predict_loan():
    try:
        body = request.get_json()
        if not body:
            return jsonify({"error": "Request body is empty"}), 400

        applicant = {
            "grade":              body.get("grade", "C"),
            "annual_inc":         float(body.get("annual_inc", 0)),
            "short_emp":          int(body.get("short_emp", 0)),
            "emp_length_num":     int(body.get("emp_length_num", 5)),
            "home_ownership":     body.get("home_ownership", "RENT"),
            "dti":                float(body.get("dti", 15)),
            "purpose":            body.get("purpose", "other"),
            "term":               body.get("term", " 36 months"),
            "last_delinq_none":   int(body.get("last_delinq_none", 1)),
            "revol_util":         float(body.get("revol_util", 40)),
            "total_rec_late_fee": float(body.get("total_rec_late_fee", 0)),
            "od_ratio":           float(body.get("od_ratio", 0.5)),
        }

        loan_amount  = float(body["loan_amount"])  if body.get("loan_amount")  else None
        term_months  = int(body["term_months"])    if body.get("term_months")  else 36

        result = predict(applicant, loan_amount=loan_amount, term_months=term_months)
        return jsonify(result), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─── Model stats ──────────────────────────────────────────────────────────────

@app.route("/api/model-stats", methods=["GET"])
def model_stats():
    return jsonify({
        "models": [
            {"name": "Logistic Regression", "f1": 0.4308, "auc": 0.7134, "accuracy": 0.6577},
            {"name": "Decision Tree",       "f1": 0.3547, "auc": 0.6730, "accuracy": 0.7180},
            {"name": "Random Forest",       "f1": 0.3385, "auc": 0.6894, "accuracy": 0.7635},
        ],
        "best_model": "Random Forest",
        "dataset_size": 20000,
        "features": 14,
        "class_balance": {"non_default": "80%", "default": "20%"},
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
