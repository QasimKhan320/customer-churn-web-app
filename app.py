
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the saved Assignment 1 model and preprocessing information.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "churn_model.joblib"

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
scaler = artifact["scaler"]
feature_columns = artifact["feature_columns"]
binary_cols = artifact["binary_cols"]
service_cols = artifact["service_cols"]
nominal_cols = artifact["nominal_cols"]
numeric_cols = artifact["numeric_cols"]
model_name = artifact["model_name"]


def preprocess_customer(customer_data):
    """Transform raw form data into the format expected by the model."""

    input_df = pd.DataFrame([customer_data]).copy()

    # Convert Yes/No binary fields into 1/0.
    for col in binary_cols:
        input_df[col] = input_df[col].map({"Yes": 1, "No": 0})

    # Treat unavailable telecom services as No.
    for col in service_cols:
        input_df[col] = (
            input_df[col]
            .replace({
                "No internet service": "No",
                "No phone service": "No",
            })
            .map({"Yes": 1, "No": 0})
        )

    # One-hot encode the multi-category fields.
    input_df = pd.get_dummies(
        input_df,
        columns=nominal_cols,
        drop_first=True,
        dtype=int,
    )

    # Match the exact columns and order used during training.
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0,
    )

    # Use the scaler fitted during Assignment 1.
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    return input_df


def validate_customer_data(form):
    """Read and validate the submitted customer information."""

    required_fields = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges",
    ]

    customer = {}

    for field in required_fields:
        value = form.get(field, "").strip()

        if not value:
            raise ValueError(f"Please provide a value for {field}.")

        customer[field] = value

    # Validate numeric fields.
    try:
        customer["SeniorCitizen"] = int(customer["SeniorCitizen"])
        customer["tenure"] = int(customer["tenure"])
        customer["MonthlyCharges"] = float(customer["MonthlyCharges"])
        customer["TotalCharges"] = float(customer["TotalCharges"])
    except ValueError:
        raise ValueError("Please enter valid numeric values.")

    if customer["SeniorCitizen"] not in (0, 1):
        raise ValueError("Senior Citizen must be 0 or 1.")

    if not 0 <= customer["tenure"] <= 72:
        raise ValueError("Tenure must be between 0 and 72 months.")

    if customer["MonthlyCharges"] < 0 or customer["TotalCharges"] < 0:
        raise ValueError("Charges cannot be negative.")

    # Validate categorical choices on the server as well.
    allowed_values = {
        "gender": ["Female", "Male"],
        "Partner": ["Yes", "No"],
        "Dependents": ["Yes", "No"],
        "PhoneService": ["Yes", "No"],
        "MultipleLines": ["Yes", "No", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["Yes", "No", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No internet service"],
        "DeviceProtection": ["Yes", "No", "No internet service"],
        "TechSupport": ["Yes", "No", "No internet service"],
        "StreamingTV": ["Yes", "No", "No internet service"],
        "StreamingMovies": ["Yes", "No", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    }

    for field, choices in allowed_values.items():
        if customer[field] not in choices:
            raise ValueError(f"Invalid selection for {field}.")

    return customer


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        try:
            customer = validate_customer_data(request.form)
            processed_customer = preprocess_customer(customer)

            prediction = int(model.predict(processed_customer)[0])
            churn_probability = float(
                model.predict_proba(processed_customer)[0, 1]
            )

            result = {
                "prediction": (
                    "Likely to Churn" if prediction == 1
                    else "Not Likely to Churn"
                ),
                "probability": round(churn_probability * 100, 2),
                "model_name": model_name,
            }

        except ValueError as exc:
            error = str(exc)

        except Exception:
            app.logger.exception("Customer churn prediction failed.")
            error = (
                "The prediction could not be completed. "
                "Check the application terminal for details."
            )

    return render_template(
        "index.html",
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
