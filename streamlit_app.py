import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# ------------------ Streamlit Config ------------------
st.set_page_config(page_title="Hospital Readmission Predictor", page_icon="🏥")

st.title("🏥 Predicting Hospital Readmission (Psychiatric Patients)")
st.write("Enter patient details to estimate the probability of readmission. "
         "Inputs are transformed automatically to match training preprocessing.")

# ------------------ Load Model & Pipeline ------------------
@st.cache_resource
def load_ann(path: str):
    return load_model(path)

@st.cache_resource
def load_pipeline(path: str):
    return joblib.load(path)

MODEL_PATH = "hospital_readmission_model.h5"   # or .keras
PIPELINE_PATH = "pipeline.pkl"                 # preprocessing pipeline

model = load_ann(MODEL_PATH)
pipeline = load_pipeline(PIPELINE_PATH)

# ------------------ Sidebar Inputs ------------------
st.sidebar.header("Patient Input")

age = st.sidebar.number_input("Age", min_value=10, max_value=120, value=30, step=1)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=22.5, step=0.1)
num_admissions = st.sidebar.number_input("Number of Previous Admissions", min_value=0, max_value=50, value=1, step=1)
length_stay = st.sidebar.number_input("Length of Stay (days)", min_value=0, max_value=365, value=7, step=1)

gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
ethnicity = st.sidebar.selectbox("Ethnicity", ["White", "Black", "Asian", "Hispanic", "Other"])
education = st.sidebar.selectbox("Education Level", ["None", "Primary", "Secondary", "Tertiary"])
smoking = st.sidebar.selectbox("Smoking", ["Yes", "No"])
alcohol = st.sidebar.selectbox("Alcohol Consumption", ["Yes", "No"])
physical_activity = st.sidebar.selectbox("Physical Activity", ["Low", "Moderate", "High"])
diet_quality = st.sidebar.selectbox("Diet Quality", ["Poor", "Average", "Good"])
medication_adherence = st.sidebar.selectbox("Medication Adherence", ["Poor", "Average", "Good"])
diagnosis = st.sidebar.selectbox("Diagnosis", ["Schizophrenia", "Bipolar Disorder", "Depression", "Other"])
suicidal = st.sidebar.selectbox("Suicidal Ideation", ["Yes", "No"])
homeless = st.sidebar.selectbox("Homeless", ["Yes", "No"])
social_support = st.sidebar.selectbox("Social Support", ["Low", "Moderate", "High"])
functional_impairment = st.sidebar.selectbox("Functional Impairment", ["Yes", "No"])
cognitive_impairment = st.sidebar.selectbox("Cognitive Impairment", ["Yes", "No"])

# ------------------ Build DataFrame ------------------
# Must match your training dataset column names & order
input_df = pd.DataFrame({
    "Age": [age],
    "BMI": [bmi],
    "NumberOfPreviousAdmissions": [num_admissions],
    "LengthOfStay": [length_stay],
    "Gender": [gender],
    "Ethnicity": [ethnicity],
    "EducationLevel": [education],
    "Smoking": [smoking],
    "AlcoholConsumption": [alcohol],
    "PhysicalActivity": [physical_activity],
    "DietQuality": [diet_quality],
    "MedicationAdherence": [medication_adherence],
    "Diagnosis": [diagnosis],
    "SuicidalIdeation": [suicidal],
    "Homeless": [homeless],
    "SocialSupport": [social_support],
    "FunctionalImpairment": [functional_impairment],
    "CognitiveImpairment": [cognitive_impairment],
})

st.subheader("📝 Input Preview")
st.dataframe(input_df, use_container_width=True)

# ------------------ Preprocessing ------------------
try:
    input_processed = pipeline.transform(input_df)   # expands → 28 features
except Exception as e:
    st.error(f"Preprocessing failed: {e}")
    st.stop()

# ------------------ Prediction ------------------
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
