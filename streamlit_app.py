import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Hospital Readmission Predictor", page_icon="🏥")

st.title("🏥 Predicting Hospital Readmission (Psychiatric Patients)")
st.write("Enter patient details to estimate the probability of readmission. "
         "Inputs are limited to the 7 features used for training.")

# --- Load pipeline (preprocessing + ANN model) ---
@st.cache_resource
def load_pipeline(path: str):
    return joblib.load(path)

PIPELINE_PATH = "readmission_pipeline.pkl"
pipeline = load_pipeline(PIPELINE_PATH)

st.sidebar.header("Patient Input")

# --- Collect inputs (only 7 features) ---
age = st.sidebar.number_input("Age", min_value=10, max_value=120, value=30, step=1)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=22.5, step=0.1)
num_admissions = st.sidebar.number_input("Number of Previous Admissions", min_value=0, max_value=50, value=1, step=1)
length_stay = st.sidebar.number_input("Length of Stay (days)", min_value=0, max_value=365, value=7, step=1)

# ⚠️ Make sure categories match exactly what you used in training!
medication_adherence = st.sidebar.selectbox("Medication Adherence", ["Poor", "Average", "Good"])
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

# --- Predict directly with pipeline ---
if st.button("🔮 Predict Readmission Risk"):
    try:
        prob = pipeline.predict_proba(input_df)[0][1]  # probability of class=1
        st.write(f"**Predicted probability of readmission:** {prob:.3f}")
        if prob >= 0.5:
            st.error("⚠️ High Risk of Readmission")
        else:
            st.success("✅ Low Risk of Readmission")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
