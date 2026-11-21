import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Visit with Us - Wellness Tourism Package Predictor",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ Visit with Us: Wellness Tourism Package Predictor")
st.write(
    "Enter customer profile and sales interaction details to predict whether "
    "the customer will purchase the Wellness Tourism Package (`ProdTaken`)."
)

MODEL_PATH = "best_wellness_model.joblib"  # Update filename if your committed model has a different name


@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        st.error(f"Model file '{path}' was not found in the repository root.")
        return None
    return joblib.load(path)


model = load_model(MODEL_PATH)

st.sidebar.header("Customer & Interaction Inputs")

with st.sidebar.expander("Customer Profile", expanded=True):
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced", "Unmarried"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("MonthlyIncome", min_value=0.0, value=25000.0, step=1000.0)
    city_tier = st.selectbox("CityTier", [1, 2, 3])

with st.sidebar.expander("Trip & Household Details", expanded=False):
    num_persons_visiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=15, value=2, step=1)
    num_children_visiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=10, value=0, step=1)
    num_trips = st.number_input("NumberOfTrips", min_value=0, max_value=50, value=3, step=1)
    preferred_property_star = st.selectbox("PreferredPropertyStar", [3, 4, 5])
    passport = st.selectbox("Passport (0: No, 1: Yes)", [0, 1])
    own_car = st.selectbox("OwnCar (0: No, 1: Yes)", [0, 1])

with st.sidebar.expander("Sales Interaction Details", expanded=False):
    typeof_contact = st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox("ProductPitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration_of_pitch = st.number_input("DurationOfPitch (mins)", min_value=1, max_value=240, value=15, step=1)
    pitch_satisfaction_score = st.slider("PitchSatisfactionScore", min_value=1, max_value=5, value=3)
    num_followups = st.number_input("NumberOfFollowups", min_value=0, max_value=20, value=3, step=1)

# Save user inputs into a DataFrame (excluding CustomerID and target ProdTaken)
input_df = pd.DataFrame(
    [
        {
            "Age": age,
            "TypeofContact": typeof_contact,
            "CityTier": city_tier,
            "DurationOfPitch": duration_of_pitch,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": num_persons_visiting,
            "NumberOfFollowups": num_followups,
            "ProductPitched": product_pitched,
            "PreferredPropertyStar": preferred_property_star,
            "MaritalStatus": marital_status,
            "NumberOfTrips": num_trips,
            "Passport": passport,
            "PitchSatisfactionScore": pitch_satisfaction_score,
            "OwnCar": own_car,
            "NumberOfChildrenVisiting": num_children_visiting,
            "Designation": designation,
            "MonthlyIncome": monthly_income,
        }
    ]
)

st.subheader("Input DataFrame")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict Purchase (`ProdTaken`)"):
    if model is not None:
        try:
            pred = int(model.predict(input_df)[0])
            proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

            if pred == 1:
                st.success("Prediction: Likely to Purchase (`ProdTaken = 1`)")
            else:
                st.info("Prediction: Unlikely to Purchase (`ProdTaken = 0`)")

            if proba is not None:
                st.metric("Purchase Probability", f"{proba * 100:.2f}%")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
