from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer

app = Flask(__name__)

# Load the trained model, scaler, and feature names
model = joblib.load("best_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

@app.route("/")
def index():
    return render_template("index-professional.html")

@app.route("/loan_prediction.html")
def loan_prediction_page():
    return render_template("loan_prediction.html")

@app.route("/visualizations.html")
def visualizations_page():
    return render_template("visualizations.html")

@app.route("/emi.html")
def emi_page():
    return render_template("emi.html")

@app.route("/predict", methods=["POST"])
def predict_with_feature_importance():
    try:
        data = request.get_json()

        # Parse input
        input_features = {
            'income_annum': [float(data.get('income', 0))],
            'cibil_score': [float(data.get('cibil', 0))],
            'loan_amount': [float(data.get('loan_amount', 0))],
            'loan_term': [float(data.get('loan_term', 0))],
            'residential_assets_value': [float(data.get('residential_assets_value', 0))],
            'commercial_assets_value': [float(data.get('commercial_assets_value', 0))],
            'luxury_assets_value': [float(data.get('luxury_assets_value', 0))],
            'bank_asset_value': [float(data.get('bank_asset_value', 0))],
            'no_of_dependents': [int(data.get('dependents', 0))],
            'education': [int(data.get('education', 0))],
            'self_employed': [int(data.get('self_employed', 0))],
            'gender': [int(data.get('gender', 0))],
            'married': [int(data.get('married', 0))],
        }

        # Feature Engineering
        features_df = pd.DataFrame(input_features)
        features_df["debt_to_income_ratio"] = features_df["loan_amount"] / (features_df["income_annum"] + 1)
        features_df["asset_to_loan_ratio"] = (
            features_df["residential_assets_value"] + features_df["commercial_assets_value"] + features_df["luxury_assets_value"]
        ) / (features_df["loan_amount"] + 1)
        features_df["asset_to_income_ratio"] = (
            features_df["residential_assets_value"] + features_df["commercial_assets_value"] + features_df["luxury_assets_value"]
        ) / (features_df["income_annum"] + 1)

        # Replace infinite and missing values
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_df.fillna(0, inplace=True)

        # Ensure DataFrame contains only feature_names
        features_df = features_df[feature_names]

        # Scale features
        features_scaled = scaler.transform(features_df)

        # Predict
        threshold = 0.6
        prediction_proba = model.predict_proba(features_scaled)[0][1]
        message = "Loan Approved" if prediction_proba >= threshold else "Loan Not Approved"

        # LIME Explanation
        explainer = LimeTabularExplainer(
            training_data=scaler.transform(pd.DataFrame(features_df, columns=feature_names)),
            feature_names=feature_names,
            class_names=["Not Approved", "Approved"],
            discretize_continuous=True
        )
        explanation = explainer.explain_instance(features_scaled[0], model.predict_proba)
        lime_explanation = [(feature, round(weight, 2)) for feature, weight in explanation.as_list()]
        lime_message = ", ".join([f"{f}: {w}" for f, w in lime_explanation])
        
        print({
            "message": message,
            "prediction_probability": prediction_proba,
            "lime_explanation": lime_explanation,
            "lime_message": lime_message
            
        })

        return jsonify({
            "message": message,
            "prediction_probability": prediction_proba,
            "lime_explanation": lime_explanation,
            "lime_message": lime_message
        })

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=False)
