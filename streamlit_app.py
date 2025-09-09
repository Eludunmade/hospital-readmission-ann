import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Hospital Readmission Predictor", page_icon="🏥")

st.title("🏥 Predicting Hospital Readmission (Psychiatric Patients)")
st.write("Enter patient details to estimate the probability of readmission. "
         "Inputs are limited to the 7 features used for training.")

# --- Load ANN model ---
@st.cache_resource
def load_ann(path: str):
    return load_model(path)

MODEL_PATH = "hospital_readmission_model.h5"   # or .h5 if that's what you saved
model = load_ann(MODEL_PATH)

# --- Define preprocessing pipeline (rebuilt instead of loading joblib) ---
@st.cache_resource
def build_pipeline():
    numeric_features = ["Age", "BMI", "NumberOfPreviousAdmissions", "LengthOfStay"]
    categorical_features = ["MedicationAdherence", "Diagnosis", "SocialSupport"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Wrap in pipeline so it behaves like your old one
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    return pipeline

pipeline = build_pipeline()

st.sidebar.header("Patient Input")

# --- Collect inputs (only 7 features) ---
age = st.sidebar.number_input("Age", min_value=10, max_value=120, value=30, step=1)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=22.5, step=0.1)
num_admissions = st.sidebar.number_input("Number of Previous Admissions", min_value=0, max_value=50, value=1, step=1)
length_stay = st.sidebar.number_input("Length of Stay (days)", min_value=0, max_value=365, value=7, step=1)

medication_adherence = st.sidebar.selectbox("Medication Adherence", ["Low", "Medium", "High"])
diagnosis = st.sidebar.selectbox("Diagnosis", ["Schizophrenia", "Bipolar Disorder", "Depression", "Other"])
social_support = st.sidebar.selectbox("Social Support", ["Low", "Moderate", "High"])

# --- Build input DataFrame in SAME order as training ---
input_df = pd.DataFrame({
    "Age": [age],
    "BMI": [bmi],
    "NumberOfPreviousAdmissions": [num_admissions],
    "LengthOfStay": [length_stay],
    "MedicationAdherence": [medication_adherence],
    "Diagnosis": [diagnosis],
    "SocialSupport": [social_support],
})

st.subheader("📝 Input Preview")
st.dataframe(input_df, use_container_width=True)

# --- Train pipeline on some dummy data (so encoder knows categories) ---
# 👇 You must replace with the categories/data you trained on!
dummy_data = pd.DataFrame({
    "Age": [30, 45, 60],
    "BMI": [22.5, 28.0, 35.0],
    "NumberOfPreviousAdmissions": [1, 2, 3],
    "LengthOfStay": [7, 14, 21],
    "MedicationAdherence": ["Low", "Medium", "High"],
    "Diagnosis": ["Schizophrenia", "Bipolar Disorder", "Depression"],
    "SocialSupport": ["Low", "Moderate", "High"],
})

pipeline.fit(dummy_data)   # fit only once

# --- Preprocess input ---
try:
    input_processed = pipeline.transform(input_df)
except Exception as e:
    st.error(f"Preprocessing failed: {e}")
    st.stop()

# --- Make prediction ---
if st.button("🔮 Predict Readmission Risk"):
    try:
        prob = float(model.predict(input_processed)[0][0])
        st.write(f"**Predicted probability of readmission:** {prob:.3f}")
        if prob >= 0.5:
            st.error("⚠️ High Risk of Readmission")
        else:
            st.success("✅ Low Risk of Readmission")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
