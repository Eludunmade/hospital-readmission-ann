import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
        .main {
            background-color: #f9fafc;
            padding: 20px;
            border-radius: 15px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            font-family: 'Helvetica Neue', sans-serif;
        }
        h2 {
            color: #34495e;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .stButton button {
            background: linear-gradient(135deg, #3498db, #2ecc71);
            color: white;
            border-radius: 12px;
            height: 3em;
            width: 100%;
            font-weight: bold;
        }
        .stButton button:hover {
            background: linear-gradient(135deg, #2ecc71, #3498db);
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.title("🏥 Hospital Readmission Prediction App")
st.write("Predict whether a psychiatric patient is likely to be readmitted using 28 clinical and demographic features.")

# Sidebar
st.sidebar.header("⚙️ Input Features")
st.sidebar.write("Fill in the details below:")

# Collect inputs
feature_names = [
    f"Feature {i+1}" for i in range(28)  # Replace with actual column names if available
]

user_data = {}
for feature in feature_names:
    user_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

# Convert inputs into DataFrame
input_df = pd.DataFrame([user_data])

# Scale input
scaled_input = scaler.transform(input_df)

# Predict
if st.button("🔮 Predict"):
    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ The model predicts this patient **is likely to be readmitted**.\n\n Probability: **{prob:.2f}**")
    else:
        st.success(f"✅ The model predicts this patient **is not likely to be readmitted**.\n\n Probability: **{prob:.2f}**")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Built with ❤️ using Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)
