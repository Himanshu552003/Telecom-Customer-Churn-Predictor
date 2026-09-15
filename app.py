"""
Flask app serving the telecom churn prediction model.

Endpoints:
  GET  /            - simple HTML form for manual testing
  POST /predict      - JSON API: takes raw customer details, returns
                        churn probability + prediction

Local dev:
    python app.py
Then visit http://127.0.0.1:5000/

Production (e.g. Render/Railway):
    gunicorn app:app
"""

import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

sys.path.append(str(Path(__file__).parent / "src"))
from preprocessing import engineer_features  # noqa: E402

MODEL_PATH = "models/churn_model.joblib"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

REQUIRED_FIELDS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

EXAMPLE_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 171.0,
}


def predict_from_payload(payload: dict):
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    row = pd.DataFrame([payload])
    row = engineer_features(row)

    proba = float(model.predict_proba(row)[:, 1][0])
    prediction = "Yes" if proba >= 0.5 else "No"
    return {"churn_probability": round(proba, 4), "churn_prediction": prediction}


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", example=EXAMPLE_PAYLOAD)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        result = predict_from_payload(payload)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        form = request.form.to_dict()
        payload = {
            "gender": form["gender"],
            "SeniorCitizen": int(form["SeniorCitizen"]),
            "Partner": form["Partner"],
            "Dependents": form["Dependents"],
            "tenure": int(form["tenure"]),
            "PhoneService": form["PhoneService"],
            "MultipleLines": form["MultipleLines"],
            "InternetService": form["InternetService"],
            "OnlineSecurity": form["OnlineSecurity"],
            "OnlineBackup": form["OnlineBackup"],
            "DeviceProtection": form["DeviceProtection"],
            "TechSupport": form["TechSupport"],
            "StreamingTV": form["StreamingTV"],
            "StreamingMovies": form["StreamingMovies"],
            "Contract": form["Contract"],
            "PaperlessBilling": form["PaperlessBilling"],
            "PaymentMethod": form["PaymentMethod"],
            "MonthlyCharges": float(form["MonthlyCharges"]),
            "TotalCharges": float(form["TotalCharges"]),
        }
        result = predict_from_payload(payload)
        return render_template("index.html", example=payload, result=result)
    except Exception as e:
        return render_template("index.html", example=EXAMPLE_PAYLOAD, error=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
