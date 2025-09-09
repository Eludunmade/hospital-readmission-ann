import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# -------------------------------
# Load model + pipeline
# -------------------------------
@st.cache_resource
def load_ann(path: str):
    return load_model(path)

@st.cache_resource
def load_pipeline(path: str):
    return joblib.load(path)

MODEL_PATH = "hospital_readmission_model.h5"   # must be uploaded
PIPELINE_PATH = "pipeline.pkl"                 # must be uploaded

model = load_ann(MODEL_PATH)
pipeline = load_pipeline(PIPELINE_PATH)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Hospital Readmission Predictor", page_icon="🏥")
st.title("🏥 Predicting Hospital Readmission (Psychiatric Patients)")
st.write("Fill in patient details to estimate the probability of readmission.")

# -------------------------------
# Collect User Input (must match training features)
# -------------------------------
age = st.number_input("Age", min_value=0, max_value=120, value=45)
gender = st.selectbox("Gender", ["Male", "Female"])
ethnicity = st.selectbox("Ethnicity", ["White", "Black", "Asian", "Hispanic", "Other"])
education = st.selectbox("Education Level", ["None", "Primary", "Secondary", "Tertiary"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
smoking = st.selectbox("Smoking", ["Yes", "No"])
alcohol = st.selectbox("Alcohol Consumption", ["Yes", "No"])
physical_activity = st.selectbox("Physical Activity", ["Low", "Medium", "High"])
diet_quality = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
medication_adherence = st.selectbox("Medication Adherence", ["Poor", "Average", "Good"])
previous_admissions = st.number_input("Number of Previous Admissions", min_value=0, max_value=50, value=0)
length_of_stay = st.number_input("Length of Stay (days)", min_value=1, max_value=365, value=5)
diagnosis = st.selectbox("Diagnosis", ["Depression", "Bipolar", "Schizophrenia", "Anxiety", "Other"])
suicidal_ideation = st.selectbox("Suicidal Ideation", ["Yes", "No"])
homeless = st.selectbox("Homeless", ["Yes", "No"])
social_support = st.selectbox("Social Support", ["Poor", "Average", "Good"])
functional_impairment = st.selectbox("Functional Impairment", ["None", "Mild", "Moderate", "Severe"])
cognitive_impairment = st.selectbox("Cognitive Impairment", ["None", "Mild", "Moderate", "Severe"])

# -------------------------------
# Build input DataFrame
# -------------------------------
input_df = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Ethnicity": ethnicity,
    "EducationLevel": education,
    "BMI": bmi,
    "Smoking": smoking,
    "AlcoholConsumption": alcohol,
    "PhysicalActivity": physical_activity,
    "DietQuality": diet_quality,
    "MedicationAdherence": medication_adherence,
    "NumberOfPreviousAdmissions": previous_admissions,
    "LengthOfStay": length_of_stay,
    "Diagnosis": diagnosis,
    "SuicidalIdeation": suicidal_ideation,
    "Homeless": homeless,
    "SocialSupport": social_support,
    "FunctionalImpairment": functional_impairment,
    "CognitiveImpairment": cognitive_impairment
}])

st.subheader("📝 Input Preview")
st.dataframe(input_df, use_container_width=True)

# -------------------------------
# Preprocess + Predict
# -------------------------------
if st.button("🔮 Predict Readmission Risk"):
    try:
        # Preprocess
        input_processed = pipeline.transform(input_df)

        # Predict
        prob = float(model.predict(input_processed)[0][0])

        st.write(f"**Predicted probability of readmission:** {prob:.3f}")
        if prob >= 0.5:
            st.error("⚠️ High Risk of Readmission")
        else:
            st.success("✅ Low Risk of Readmission")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
