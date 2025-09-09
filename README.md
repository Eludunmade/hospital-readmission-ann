# Hospital Readmission Predictor (Streamlit + TensorFlow)

This is a minimal Streamlit app that loads a trained Keras model (`hospital_readmission_model.h5`) and an optional scaler (`scaler.pkl`) to predict psychiatric patient readmission risk.

## Files you need in the same folder
- `streamlit_app.py` (this app)
- `hospital_readmission_model.h5` (saved from your notebook)
- `scaler.pkl` (optional, saved with `joblib.dump(scaler, "scaler.pkl")`)

## How to run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Saving your model and scaler in the notebook
```python
# Save model
ann.save("hospital_readmission_model.h5")

# Save scaler (if you used one)
import joblib
joblib.dump(scaler, "scaler.pkl")
```

## IMPORTANT
The input features (names and order) in `streamlit_app.py` **must match** what you used during training. Edit the sidebar inputs to reflect your real features.
