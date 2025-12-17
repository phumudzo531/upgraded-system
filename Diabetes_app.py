import streamlit as st
import pandas as pd
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load("diabetes_app_new.pkl")

model = load_model()

# App title and intro
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("Diabetes Prediction App")
st.write("Fill in the details below to predict the likelihood of diabetes.")

# Input form
with st.form("Prediction form"):
    st.subheader("Enter Patient Data")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.slider("Pregnancies", 0, 10, step=1)
        glucose = st.slider("Glucose", 0, 200, 120)
        blood_pressure = st.slider("Blood Pressure", 0, 140, 70)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)

    with col2:
        skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
        insulin = st.slider("Insulin", 0, 900, 80)
        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5
        )
        age = st.selectbox("Age", options=[i for i in range(18, 81)], index=2)

    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    input_data = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ],
    )

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("Prediction: Positive for Diabetes")
    else:
        st.success("Prediction: Negative for Diabetes")