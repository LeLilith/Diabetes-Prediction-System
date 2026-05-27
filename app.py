import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set the configuration for the web page
st.set_page_config(page_title="Diabetes Prediction System", layout="centered")

# --- Load the Model and Scaler ---
# The @st.cache_resource decorator ensures Streamlit only loads these large files once
@st.cache_resource
def load_components():
    rf_model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('diabetes_scaler.pkl')
    return rf_model, scaler

model, scaler = load_components()

# --- Build the UI ---
st.title("🩺 AI-Based Diabetes Prediction System")
st.write("Enter the patient's medical details below to predict the likelihood of diabetes.")
st.markdown("---")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Level (IU/mL)", min_value=0, max_value=1000, value=79)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.000, max_value=3.000, value=0.500, format="%.3f")
    age = st.number_input("Age", min_value=1, max_value=120, value=33)

st.markdown("---")

# --- Prediction Logic ---
if st.button("Predict Diabetes Risk", type="primary", use_container_width=True):
    
    # 1. Gather the inputs into a dataframe (must match the exact order of the training data)
    input_data = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness, 
        insulin, bmi, dpf, age
    ]], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])
    
    # 2. Scale the input data using the exact same scaler from training
    input_scaled = scaler.transform(input_data)
    
    # 3. Get the prediction and the probability score
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # 4. Display the results
    st.subheader("Prediction Results")
    
    if prediction == 1:
        st.error(f"⚠️ **Result: Diabetic**")
        st.write(f"The model predicts this patient has diabetes with a **{probability * 100:.1f}% risk score.**")
        st.info("Recommendation: Please consult a healthcare professional for a formal diagnosis and preventive care.")
    else:
        st.success(f"✅ **Result: Non-Diabetic**")
        st.write(f"The model predicts this patient does not have diabetes. (Risk score: **{probability * 100:.1f}%**)")