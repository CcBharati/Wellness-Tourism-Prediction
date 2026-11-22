import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Visit with Us — Wellness Package Predictor", layout="wide")
st.title("Visit with Us — Wellness Tourism Package Predictor")

MODEL_PATHS = ["models/best_model.pkl", "models/model.pkl", "models/model.joblib"]

@st.cache_resource
def load_model():
    for p in MODEL_PATHS:
        if os.path.exists(p):
            return joblib.load(p), p
    return None, None

model, model_path = load_model()
if model is None:
    st.warning("Model file not found in models/. Add a trained model artifact to enable predictions.")

age = st.number_input("Age", 18, 100, 35)
monthly_income = st.number_input("MonthlyIncome", 1000, 500000, 25000, step=500)
passport = st.selectbox("Passport", [0, 1])
pitch_satisfaction = st.slider("PitchSatisfactionScore", 1, 5, 3)
num_followups = st.number_input("NumberOfFollowups", 0, 15, 3)
duration_of_pitch = st.number_input("DurationOfPitch", 1, 180, 15)

df = pd.DataFrame([{
    "Age": age,
    "MonthlyIncome": monthly_income,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction,
    "NumberOfFollowups": num_followups,
    "DurationOfPitch": duration_of_pitch,
}])

if st.button("Predict ProdTaken"):
    if model is not None:
        pred = int(model.predict(df)[0])
        st.success(f"Predicted ProdTaken: {pred}")
