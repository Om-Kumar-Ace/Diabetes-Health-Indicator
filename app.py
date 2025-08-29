import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from io import BytesIO

# Safe model loader
def load_pickle_from_url(url, label):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pickle.load(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to fetch {label}: {e}")
    except ModuleNotFoundError as e:
        st.error(f"❌ Missing module while loading {label}: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected error loading {label}: {e}")
    return None

# Load model and scaler
model_url = "https://github.com/Om-Kumar-Ace/Diabetes-Health-Indicator/raw/main/logistic_model.pkl"
scaler_url = "https://github.com/Om-Kumar-Ace/Diabetes-Health-Indicator/raw/main/scaler.pkl"

model = load_pickle_from_url(model_url, "model")
scaler = load_pickle_from_url(scaler_url, "scaler")

# Validate successful loading
if model is None or scaler is None:
    st.stop()

# Define expected feature order
try:
    feature_order = scaler.feature_names_in_
except AttributeError:
    st.error("❌ Scaler object missing 'feature_names_in_' attribute.")
    st.stop()

# UI
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter patient health indicators to predict diabetes risk.")

# Input form
with st.form("prediction_form"):
    inputs = {}

    inputs["Sex"] = st.radio("Sex", ["Male", "Female"])
    inputs["Sex"] = 1 if inputs["Sex"] == "Male" else 0

    inputs["Age"] = st.number_input("Age", min_value=0, max_value=120, step=1)
    inputs["BMI"] = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, step=0.1)
    inputs["GenHlth"] = st.slider("General Health (1 = Excellent, 5 = Poor)", min_value=1, max_value=5)

    inputs["HighChol"] = st.radio("Do you have high cholesterol?", ["No", "Yes"])
    inputs["HighChol"] = 1 if inputs["HighChol"] == "Yes" else 0

    inputs["Smoker"] = st.radio("Have you smoked at least 100 cigarettes in your life?", ["No", "Yes"])
    inputs["Smoker"] = 1 if inputs["Smoker"] == "Yes" else 0

    inputs["PhysActivity"] = st.radio("Physical activity in past 30 days (excluding job)?", ["No", "Yes"])
    inputs["PhysActivity"] = 1 if inputs["PhysActivity"] == "Yes" else 0

    inputs["Fruits"] = st.radio("Consume fruit 1+ times per day?", ["No", "Yes"])
    inputs["Fruits"] = 1 if inputs["Fruits"] == "Yes" else 0

    inputs["Veggies"] = st.radio("Consume vegetables 1+ times per day?", ["No", "Yes"])
    inputs["Veggies"] = 1 if inputs["Veggies"] == "Yes" else 0

    inputs["DiffWalk"] = st.radio("Do you have difficulty walking?", ["No", "Yes"])
    inputs["DiffWalk"] = 1 if inputs["DiffWalk"] == "Yes" else 0

    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    try:
        input_df = pd.DataFrame([inputs])[feature_order]
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

        if prediction == 0:
            st.success("🎉 Prediction: Non-Diabetic")
            st.info(f"Probability of diabetes: {probability:.2f}")
            st.balloons()
        else:
            st.warning("⚠️ Prediction: Diabetic")
            st.info(f"Probability of diabetes: {probability:.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# Footer
st.markdown("---")
st.caption("⚠️ This prediction is based on statistical modeling and may not be fully accurate. Please consult a medical professional for a proper diagnosis.")
