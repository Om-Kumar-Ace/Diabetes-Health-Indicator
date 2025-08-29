import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define expected feature order
feature_order = scaler.feature_names_in_

st.title("🩺 Diabetes Prediction App")
st.markdown("Enter patient health indicators to predict diabetes risk.")

# Input form
with st.form("prediction_form"):
    inputs = {}

    # Sex: 1 = Male, 0 = Female
    inputs["Sex"] = st.radio("Sex", ["Male", "Female"])
    inputs["Sex"] = 1 if inputs["Sex"] == "Male" else 0

    # Age
    inputs["Age"] = st.number_input("Age", min_value=0, max_value=120, step=1)

    # BMI
    inputs["BMI"] = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, step=0.1)

    # GenHlth
    inputs["GenHlth"] = st.slider("General Health (1 = Excellent, 5 = Poor)", min_value=1, max_value=5)

    # HighChol
    inputs["HighChol"] = st.radio("Do you have high cholesterol?", ["No", "Yes"])
    inputs["HighChol"] = 1 if inputs["HighChol"] == "Yes" else 0

    # Smoker
    inputs["Smoker"] = st.radio("Have you smoked at least 100 cigarettes in your life?", ["No", "Yes"])
    inputs["Smoker"] = 1 if inputs["Smoker"] == "Yes" else 0

    # PhysActivity
    inputs["PhysActivity"] = st.radio("Physical activity in past 30 days (excluding job)?", ["No", "Yes"])
    inputs["PhysActivity"] = 1 if inputs["PhysActivity"] == "Yes" else 0

    # Fruits
    inputs["Fruits"] = st.radio("Consume fruit 1+ times per day?", ["No", "Yes"])
    inputs["Fruits"] = 1 if inputs["Fruits"] == "Yes" else 0

    # Veggies
    inputs["Veggies"] = st.radio("Consume vegetables 1+ times per day?", ["No", "Yes"])
    inputs["Veggies"] = 1 if inputs["Veggies"] == "Yes" else 0

    # DiffWalk
    inputs["DiffWalk"] = st.radio("Do you have difficulty walking?", ["No", "Yes"])
    inputs["DiffWalk"] = 1 if inputs["DiffWalk"] == "Yes" else 0

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([inputs])[feature_order]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.success(f"Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
    st.info(f"Probability of diabetes: {probability:.2f}")
