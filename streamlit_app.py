import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# -------------------------
# Load trained ANN model
# -------------------------
MODEL_PATH = "hospital_readmission_model.h5"

@st.cache_resource
def load_ann(path: str):
    return load_model(path)

model = load_ann(MODEL_PATH)

# -------------------------
# Define all features (must match training)
# -------------------------
numeric_cols = [
    "Age", "BMI", "NumberOfPreviousAdmissions", "LengthOfStay"
]

categorical_cols = [
    "Gender", "Ethnicity", "Diagnosis",
    "MedicationAdherence", "SocialSupport",
    # 👉 add the rest of the categorical features you trained with
    "MaritalStatus", "EmploymentStatus", "EducationLevel",
    "SubstanceAbuseHistory", "FamilyHistory", "TreatmentType",
    "FollowUpPlan", "HousingStatus", "IncomeLevel", "InsuranceStatus",
    "SuicidalIdeation", "ViolentBehavior", "CognitiveImpairment",
    "SleepPattern", "PhysicalHealthCondition", "TherapyAttendance",
    "CrisisHistory", "LegalIssues", "CommunitySupport"
]

# Combine everything
all_features = numeric_cols + categorical_cols

# -------------------------
# Define preprocessing pipeline (no pickle)
# -------------------------
pipeline = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# -------------------------
# Streamlit UI
# -------------------------
st.title("🏥 Psychiatric Hospital Readmission Prediction")

# Numeric inputs
age = st.number_input("Age", 18, 100, 40)
bmi = st.number_input("BMI", 10.0, 50.0, 22.5)
num_admissions = st.number_input("Number of Previous Admissions", 0, 20, 1)
length_stay = st.number_input("Length of Stay (days)", 1, 365, 10)

# Categorical inputs
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
ethnicity = st.selectbox("Ethnicity", ["White", "Black", "Asian", "Hispanic", "Other"])
diagnosis = st.selectbox("Diagnosis", ["Depression", "Schizophrenia", "Bipolar", "Anxiety", "Other"])
medication_adherence = st.selectbox("Medication Adherence", ["Good", "Poor"])
social_support = st.selectbox("Social Support", ["Strong", "Weak", "None"])

# Fill in all extra categorical features with drop-downs or radios
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Student", "Retired"])
education_level = st.selectbox("Education Level", ["Primary", "Secondary", "Tertiary", "None"])
substance_abuse = st.selectbox("Substance Abuse History", ["Yes", "No"])
family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
treatment_type = st.selectbox("Treatment Type", ["Inpatient", "Outpatient"])
follow_up_plan = st.selectbox("Follow-Up Plan", ["Yes", "No"])
housing_status = st.selectbox("Housing Status", ["Stable", "Unstable", "Homeless"])
income_level = st.selectbox("Income Level", ["Low", "Medium", "High"])
insurance_status = st.selectbox("Insurance Status", ["Insured", "Uninsured"])
suicidal_ideation = st.selectbox("Suicidal Ideation", ["Yes", "No"])
violent_behavior = st.selectbox("Violent Behavior", ["Yes", "No"])
cognitive_impairment = st.selectbox("Cognitive Impairment", ["Yes", "No"])
sleep_pattern = st.selectbox("Sleep Pattern", ["Normal", "Disturbed"])
physical_health = st.selectbox("Physical Health Condition", ["Good", "Poor"])
therapy_attendance = st.selectbox("Therapy Attendance", ["Regular", "Irregular"])
crisis_history = st.selectbox("Crisis History", ["Yes", "No"])
legal_issues = st.selectbox("Legal Issues", ["Yes", "No"])
community_support = st.selectbox("Community Support", ["Strong", "Weak", "None"])

# -------------------------
# Create input DataFrame
# -------------------------
input_df = pd.DataFrame([{
    "Age": age,
    "BMI": bmi,
    "NumberOfPreviousAdmissions": num_admissions,
    "LengthOfStay": length_stay,
    "Gender": gender,
    "Ethnicity": ethnicity,
    "Diagnosis": diagnosis,
    "MedicationAdherence": medication_adherence,
    "SocialSupport": social_support,
    "MaritalStatus": marital_status,
    "EmploymentStatus": employment_status,
    "EducationLevel": education_level,
    "SubstanceAbuseHistory": substance_abuse,
    "FamilyHistory": family_history,
    "TreatmentType": treatment_type,
    "FollowUpPlan": follow_up_plan,
    "HousingStatus": housing_status,
    "IncomeLevel": income_level,
    "InsuranceStatus": insurance_status,
    "SuicidalIdeation": suicidal_ideation,
    "ViolentBehavior": violent_behavior,
    "CognitiveImpairment": cognitive_impairment,
    "SleepPattern": sleep_pattern,
    "PhysicalHealthCondition": physical_health,
    "TherapyAttendance": therapy_attendance,
    "CrisisHistory": crisis_history,
    "LegalIssues": legal_issues,
    "CommunitySupport": community_support
}])

# -------------------------
# Predict button
# -------------------------
if st.button("🔮 Predict Readmission"):
    try:
        # Dummy fit for pipeline structure
        pipeline.fit(input_df)  # In production, replace with training data
        input_processed = pipeline.transform(input_df)

        prob = float(model.predict(input_processed)[0][0])
        result = "⚠️ High Risk of Readmission" if prob > 0.5 else "✅ Low Risk of Readmission"

        st.write(f"**Prediction:** {result}")
        st.write(f"**Probability of readmission:** {prob:.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
