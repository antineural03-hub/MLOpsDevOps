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

Age                      = st.number_input("Age", 18, 61, 36, 1)
TypeofContact            = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
CityTier                 = st.selectbox("City tier", [1, 2, 3])
Occupation               = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business"])
Gender                   = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus            = st.selectbox("Marital status", ["Married", "Divorced", "Unmarried", "Single"])
MonthlyIncome            = st.number_input("Monthly income", 1000.0, 100000.0, 22418.0, 500.0)
ProductPitched           = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
NumberOfPersonVisiting   = st.number_input("Number of persons visiting", 1, 5, 3)
NumberOfChildrenVisiting = st.number_input("Children under 5 visiting", 0, 3, 1)
PreferredPropertyStar    = st.selectbox("Preferred property star", [3, 4, 5])
NumberOfTrips            = st.number_input("Trips per year", 1, 22, 3)
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
    "Own Car": OwnCar
}])

if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Yes-Purchased" if prediction == 1 else "No-Not Purchased"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
