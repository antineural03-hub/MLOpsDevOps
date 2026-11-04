import os
import streamlit as st
import pandas as pd
import joblib



model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

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
    "Marital Status": MaritalStatus,
    "Monthly Income": MonthlyIncome,
    "Product Pitched": ProductPitched,
    "Number of Person Visiting": NumberOfPersonVisiting,
    "Number of Children Visiting": NumberOfChildrenVisiting,
    "Preferred Property Star": PreferredPropertyStar,
    "Number of Trips": NumberOfTrips,
    "Passport": Passport,
    "Own Car": 1 if OwnCar == 'Yes' else 0
}])

if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Yes-Purchased" if prediction == 1 else "No-Not Purchased"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
