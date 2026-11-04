import os
import streamlit as st
import pandas as pd
import joblib



model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
THRESHOLD = 0.45

@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

st.title("Wellness Tourism Package Purchase Predication App")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.
Enter the customer and interaction data below to get a prediction.customer and interaction
""")

Age                      = st.slider("Age", 18, 70, 30)
TypeofContact            = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
CityTier                 = st.selectbox("City tier", [1, 2, 3])
Occupation               = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business"])
Gender                   = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus            = st.selectbox("Marital status", ["Married", "Divorced", "Unmarried", "Single"])
MonthlyIncome            = st.number_input("Monthly income", 1000.0, 30000.0)
ProductPitched           = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
NumberOfPersonVisiting   = st.slider("Number of Persons Visiting", 1, 5, 2)
NumberOfChildrenVisiting = st.slider("Number of Children Visited", 0, 5, 1)
PreferredPropertyStar    = st.selectbox("Preferred property star", [3, 4, 5])
NumberOfTrips            = st.slider("Number of Trips", 1, 20, 3)
Passport                 = st.selectbox("Has passport?", ["No", "Yes"])
OwnCar                   = st.selectbox("Owns a car?", ["No", "Yes"])


input_data = pd.DataFrame([{
    "Age": Age,
    "Type of Contact": TypeofContact,
    "City Tier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "Number of Person Visiting": NumberOfPersonVisiting,
    "Product Pitched": ProductPitched,
    "Preferred Property Star": PreferredPropertyStar,
    "Marital Status": MaritalStatus,
    "Number of Trips": NumberOfTrips,
    "Passport": 1 if Passport=='Yes' else 0,
    "Own Car": 1 if OwnCar == 'Yes' else 0,
    "Number of Children Visiting": NumberOfChildrenVisiting,
    "Monthly Income": MonthlyIncome,
        
}])

if st.button("Predict Purchase"):
    proba = model.predict_proba(input_data)[0, 1]
    st.subheader("Prediction result")
    if proba >= THRESHOLD:
        st.success(f"Likely to purchase (probability {proba:.1%}). Worth contacting.")
    else:
        st.info(f"Unlikely to purchase (probability {proba:.1%}).")

