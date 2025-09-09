import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

st.set_page_config(page_title="Hospital Readmission Predictor", page_icon="🏥")

st.title("🏥 Predicting Hospital Readmission (Psychiatric Patients)")
st.write("Enter patient details to estimate the probability of readmission. "
         "Make sure the features here match the exact order used during training.")

# --- Load model ---
@st.cache_resource
def load_ann(path: str):
    return load_model(path)

# --- Load scaler (optional) ---
@st.cache_resource
def load_scaler(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning("Scaler not found or could not be loaded. Proceeding without scaling.")
        return None

MODEL_PATH = "hospital_readmission_model.h5"
SCALER_PATH = "scaler.pkl"

model = load_ann(MODEL_PATH)
scaler = load_scaler(SCALER_PATH)

st.sidebar.header("Patient Input")

# 🔁 TODO: Replace these with your REAL training features (same order!)
age = st.sidebar.number_input("Age", min_value=10, max_value=120, value=30, step=1)
previous_admissions = st.sidebar.number_input("Previous Admissions", min_value=0, max_value=100, value=1, step=1)
medication_count = st.sidebar.number_input("Current Medication Count", min_value=0, max_value=50, value=2, step=1)
length_of_stay = st.sidebar.number_input("Length of Last Stay (days)", min_value=0, max_value=365, value=7, step=1)

# Build a DataFrame so column order is explicit and readable
input_df = pd.DataFrame({
    "Age": [age],
    "Previous_Admissions": [previous_admissions],
    "Medication_Count": [medication_count],
    "Length_of_Stay": [length_of_stay],
})

st.subheader("📝 Input Preview")
st.dataframe(input_df, use_container_width=True)

# Keep a copy of the raw (unscaled) inputs for download or auditing
raw_features = input_df.copy()

# Scale if scaler is available
if scaler is not None:
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        st.error(f"Scaler failed on the provided inputs: {e}")
        st.stop()
else:
    # If no scaler, pass raw values (ONLY if your model was trained without scaling)
    input_scaled = input_df.values

if st.button("🔮 Predict Readmission Risk"):
    try:
        prob = float(model.predict(input_scaled)[0][0])
        st.write(f"**Predicted probability of readmission:** {prob:.3f}")
        if prob >= 0.5:
            st.error("⚠️ High Risk of Readmission")
        else:
            st.success("✅ Low Risk of Readmission")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

st.caption("Tip: Ensure the feature list, their order, and preprocessing (scaling/encoding) "
           "match exactly what you used during training.")
