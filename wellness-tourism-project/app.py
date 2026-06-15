import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Wellness Tourism Predictor", page_icon="🌍", layout="wide")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="arjunmathuram/wellness-tourism-model",
        filename="model.pkl"
    )
    return joblib.load(model_path)

model = load_model()

st.title("Wellness Tourism Package Predictor")
st.success("Model loaded successfully.")

st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration Of Pitch", min_value=1, value=15)
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female", "Fe Male"])
    typeofcontact = st.selectbox("Type Of Contact", ["Company Invited", "Self Enquiry"])

with col2:
    number_of_person_visiting = st.number_input("Number Of Person Visiting", min_value=1, value=2)
    number_of_followups = st.number_input("Number Of Followups", min_value=0, value=2)
    productpitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    preferred_property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
    maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    number_of_trips = st.number_input("Number Of Trips", min_value=0, value=2)

with col3:
    passport = st.selectbox("Passport", [0, 1])
    pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    own_car = st.selectbox("Own Car", [0, 1])
    number_of_children_visiting = st.number_input("Number Of Children Visiting", min_value=0, value=0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=0, value=30000)

input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeofcontact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": productpitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": maritalstatus,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

if st.button("Predict Package Purchase"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"Customer is likely to purchase the package. Probability: {probability:.2%}")
    else:
        st.warning(f"Customer is not likely to purchase the package. Probability: {probability:.2%}")

    st.subheader("Input Data")
    st.dataframe(input_df)
