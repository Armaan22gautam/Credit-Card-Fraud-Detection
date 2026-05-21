"""
Credit Card Fraud Detection – Flask Web Application
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ─── Global State ────────────────────────────────────────────────────────────
MODELS = {}
EVALUATION = {}
SAMPLE_TRANSACTIONS = {}
DATASET_STATS = {}
FEATURE_COLS = None   # will be set after loading data

# ─── Startup Loading ─────────────────────────────────────────────────────────

def load_all():
    global MODELS, EVALUATION, SAMPLE_TRANSACTIONS, DATASET_STATS, FEATURE_COLS

    print("[startup] Loading evaluation results …")
    eval_path = os.path.join("models", "evaluation_results.json")
    with open(eval_path, "r") as f:
        EVALUATION = json.load(f)

    print("[startup] Loading trained models …")
    model_files = {
        "logistic_regression": "models/logistic_regression.pkl",
        "decision_tree":       "models/decision_tree.pkl",
        "k_nearest_neighbors": "models/k_nearest_neighbors.pkl",
    }
    for name, path in model_files.items():
        if not os.path.exists(path):
            alt = path.replace("k_nearest_neighbors", "k-nearest_neighbors")
            path = alt if os.path.exists(alt) else path
        with open(path, "rb") as f:
            MODELS[name] = pickle.load(f)
        print(f"  ✓ {name}")

    print("[startup] Loading dataset …")
    df = pd.read_csv("data/creditcard.csv")

    # Feature columns used by the models (post-preprocessing)
    # V1-V28 + scaled_amount + scaled_time
    from sklearn.preprocessing import StandardScaler
    scaler_a = StandardScaler()
    scaler_t = StandardScaler()
    scaled_amount = scaler_a.fit_transform(df[["Amount"]]).flatten()
    scaled_time   = scaler_t.fit_transform(df[["Time"]]).flatten()

    v_cols = [f"V{i}" for i in range(1, 29)]
    FEATURE_COLS = ["scaled_amount", "scaled_time"] + v_cols

    df["scaled_amount"] = scaled_amount
    df["scaled_time"]   = scaled_time

    # Dataset stats
    fraud   = int(df["Class"].sum())
    total   = int(len(df))
    normal  = total - fraud
    DATASET_STATS = {
        "total":          total,
        "fraud":          fraud,
        "normal":         normal,
        "fraud_pct":      round(fraud / total * 100, 4),
        "features":       ["Time", "Amount"] + v_cols,
        "time_range_hrs": round(float(df["Time"].max()) / 3600, 1),
        "amount_max":     round(float(df["Amount"].max()), 2),
        "amount_mean":    round(float(df["Amount"].mean()), 2),
    }

    # Sample transactions – 10 normal + 10 fraudulent
    normal_sample = (
        df[df["Class"] == 0]
        .sample(10, random_state=42)
        .reset_index(drop=True)
    )
    fraud_sample = (
        df[df["Class"] == 1]
        .sample(10, random_state=42)
        .reset_index(drop=True)
    )

    def row_to_dict(row):
        return {col: float(row[col]) for col in FEATURE_COLS}

    SAMPLE_TRANSACTIONS = {
        "normal": [
            {"label": f"Normal #{i+1}", "class": 0,
             "amount": round(float(r["Amount"]), 2),
             "features": row_to_dict(r)}
            for i, r in normal_sample.iterrows()
        ],
        "fraud": [
            {"label": f"Fraud #{i+1}", "class": 1,
             "amount": round(float(r["Amount"]), 2),
             "features": row_to_dict(r)}
            for i, r in fraud_sample.iterrows()
        ],
    }
    print("[startup] All resources loaded. Ready!")


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dataset-stats")
def api_dataset_stats():
    return jsonify(DATASET_STATS)


@app.route("/api/sample-transactions")
def api_sample_transactions():
    return jsonify(SAMPLE_TRANSACTIONS)


@app.route("/api/model-performance")
def api_model_performance():
    display_names = {
        "logistic_regression": "Logistic Regression",
        "decision_tree":       "Decision Tree",
        "k_nearest_neighbors": "KNN",
    }
    perf = {}
    for key, data in EVALUATION.items():
        perf[display_names.get(key, key)] = {
            "accuracy":  data["accuracy"],
            "precision": data["precision"],
            "recall":    data["recall"],
            "f1_score":  data["f1_score"],
            "roc_auc":   data["roc_auc"],
            "confusion_matrix": data["confusion_matrix"],
            "roc_curve": data["roc_curve"],
            "pr_curve":  data["pr_curve"],
        }
    return jsonify(perf)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True)
    features = payload.get("features", {})

    try:
        X = np.array([[features[col] for col in FEATURE_COLS]])
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400

    predictions = {}
    display_names = {
        "logistic_regression": "Logistic Regression",
        "decision_tree":       "Decision Tree",
        "k_nearest_neighbors": "KNN",
    }
    for key, model in MODELS.items():
        y_pred = int(model.predict(X)[0])
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[0][1])
        else:
            scores = model.decision_function(X)[0]
            prob = float(1 / (1 + np.exp(-scores)))

        predictions[display_names[key]] = {
            "prediction": y_pred,
            "probability": round(prob, 6),
        }

    # Ensemble average
    probs = [v["probability"] for v in predictions.values()]
    ensemble_prob = round(sum(probs) / len(probs), 6)
    ensemble_pred = 1 if ensemble_prob >= 0.5 else 0

    return jsonify({
        "models": predictions,
        "ensemble": {
            "prediction": ensemble_pred,
            "probability": ensemble_prob,
        }
    })


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_all()
    app.run(debug=False, port=5000, use_reloader=False)
