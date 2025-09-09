import streamlit as st
import pandas as pd
import joblib
import json
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Hospital Readmission Predictor", page_icon="🏥")

# --- Load model ---
@st.cache_resource
def load_ann(path: str):
    return load_model(path)

# --- Load preprocessing pipeline ---
@st.cache_resource
def load_pipeline(path: str):
    return joblib.load(path)

# --- Load feature names ---
def load_features(path: str):
    with open(path, "r") as f:
        return json.load(f)

MODEL_PATH = "hospital_readmission_model.h5"
PIPELINE_PATH = "pipeline.pkl"
FEATURES_PATH = "feature_names.json"

# Load everything
model = load_ann(MODEL_PATH)
pipeline = load_pipeline(PIPELINE_PATH)
expected_cols = load_features(FEATURES_PATH)

st.title("🏥 Predicting Hospital Readmission (Psychiatric Patients)")
st.write("Fill in the patient details below:")

# --- Collect inputs (raw values, before encoding) ---
age = st.sidebar.number_input("Age", 10, 120, 30)
bmi = st.sidebar.number_input("BMI", 10.0, 60.0, 22.5)
num_admissions = st.sidebar.number_input("Previous Admissions", 0, 50, 1)
length_stay = st.sidebar.number_input("Length of Stay (days)", 0, 365, 7)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
ethnicity = st.sidebar.selectbox("Ethnicity", ["White", "Black", "Asian", "Other"])
diagnosis = st.sidebar.selectbox("Diagnosis", ["Schizophrenia", "Bipolar Disorder", "Depression", "Other"])
medication_adherence = st.sidebar.selectbox("Medication Adherence", ["Low", "Medium", "High"])
social_support = st.sidebar.selectbox("Social Support", ["Low", "Moderate", "High"])

# --- Build input DataFrame (raw format, matches training df before pipeline) ---
input_df = pd.DataFrame({
    "Age": [age],
    "BMI": [bmi],
    "NumberOfPreviousAdmissions": [num_admissions],
    "LengthOfStay": [length_stay],
    "Gender": [gender],
    "Ethnicity": [ethnicity],
    "Diagnosis": [diagnosis],
    "MedicationAdherence": [medication_adherence],
    "SocialSupport": [social_support]
})

st.subheader("📝 Input Preview")
st.dataframe(input_df, use_container_width=True)

# --- Preprocess with pipeline ---
try:
    input_processed = pipeline.transform(input_df)
    input_final = pd.DataFrame(input_processed, columns=expected_cols)
except Exception as e:
    st.error(f"⚠️ Preprocessing failed: {e}")
    st.stop()

# --- Predict ---
if st.button("🔮 Predict Readmission Risk"):
    try:
        prob = float(model.predict(input_final)[0][0])
        st.write(f"**Predicted probability of readmission:** {prob:.3f}")
        if prob >= 0.5:
            st.error("⚠️ High Risk of Readmission")
        else:
            st.success("✅ Low Risk of Readmission")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
