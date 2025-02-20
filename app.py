import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title of the web app
st.title("Diabetes Risk Prediction App")

# User input fields
st.sidebar.header("Enter Patient Details:")

pregnancies = st.sidebar.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=200, step=1)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=150, step=1)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
insulin = st.sidebar.number_input("Insulin Level", min_value=0, max_value=900, step=1)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
diabetes_pedigree = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
age = st.sidebar.number_input("Age", min_value=1, max_value=120, step=1)

# Create a button for prediction
if st.sidebar.button("Predict"):
    # Convert inputs into a NumPy array and scale
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    
    # Display the result
    if prediction[0] == 1:
        st.error("The model predicts that you are at **high risk** of diabetes.")
    else:
        st.success("The model predicts that you are **not at risk** of diabetes.")

