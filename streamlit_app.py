# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Hospital Readmission Predictor", page_icon="🏥")

# -------------------------
# Helpers to introspect pipeline
# -------------------------
def get_preprocessor_from_pipeline(pipeline):
    # pipeline might be a sklearn Pipeline (with step named 'preprocessor')
    if hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
        return pipeline.named_steps["preprocessor"]
    # or pipeline might *be* the preprocessor (ColumnTransformer)
    return pipeline

def get_raw_columns_and_cats(preprocessor):
    """
    Returns:
      - raw_cols: list of raw column names (the columns the preprocessor expects)
      - categorical_map: dict {col_name: [cat1, cat2, ...]} for categorical columns (if available)
    """
    raw_cols = []
    categorical_map = {}

    # preprocessor.transformers_ is a list of (name, transformer, columns)
    for name, transformer, cols in preprocessor.transformers_:
        # ignore remainder
        if name == "remainder":
            continue
        if cols == "drop" or cols is None:
            continue

        # normalize cols into list of str
        if isinstance(cols, (list, tuple, np.ndarray)):
            cols_list = list(cols)
        else:
            # sometimes cols can be a string or slice; try to handle
            cols_list = [cols]

        # accumulate raw feature names
        raw_cols.extend(cols_list)

        # If transformer is a OneHotEncoder (or has categories_), capture categories
        try:
            # Some transformers (like OneHotEncoder) have attribute categories_
            if hasattr(transformer, "categories_") and hasattr(transformer, "get_feature_names_out"):
                # transformer.categories_ is a list aligned with cols_list
                for col_name, cats in zip(cols_list, transformer.categories_):
                    # ensure string categories
                    cats = [str(c) for c in cats]
                    categorical_map[col_name] = cats
        except Exception:
            # ignore if introspection fails
            pass

    return raw_cols, categorical_map

# -------------------------
# Load model + pipeline
# -------------------------
@st.cache_resource
def load_pipeline(path: str):
    return joblib.load(path)

@st.cache_resource
def load_ann(path: str):
    return load_model(path)

PIPELINE_PATH = "pipeline.pkl"           # must be the pipeline you saved (ColumnTransformer or Pipeline)
MODEL_PATH = "hospital_readmission_model.h5"

try:
    pipeline = load_pipeline(PIPELINE_PATH)
except Exception as e:
    st.error(f"Failed to load pipeline at '{PIPELINE_PATH}': {e}")
    st.stop()

try:
    model = load_ann(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load ANN model at '{MODEL_PATH}': {e}")
    st.stop()

preprocessor = get_preprocessor_from_pipeline(pipeline)

# Extract the raw column names and categories (if available)
try:
    raw_cols, categorical_map = get_raw_columns_and_cats(preprocessor)
except Exception as e:
    st.error(f"Failed to introspect the preprocessing pipeline: {e}")
    st.stop()

if not raw_cols:
    st.error("Could not determine the raw input columns from the pipeline. Make sure the saved pipeline contains a fitted ColumnTransformer with 'transformers_'.")
    st.stop()

st.title("🏥 Hospital Readmission Predictor (Auto UI from pipeline)")
st.write("This app generates inputs from the preprocessing pipeline used during training. Fill all fields and press Predict.")

# -------------------------
# Dynamic UI creation
# -------------------------
st.sidebar.header("Patient input (auto-generated)")

user_input = {}
for col in raw_cols:
    # Decide input type: if col exists in categorical_map -> selectbox; else number_input (float)
    if col in categorical_map:
        choices = categorical_map[col]
        # put a friendly default
        default = choices[0] if len(choices) > 0 else ""
        user_input[col] = st.sidebar.selectbox(col, choices, index=0)
    else:
        # numeric — use float input, but allow integers if you prefer
        # Provide reasonable defaults; user can override.
        # You can tweak min/max/default as needed.
        user_input[col] = st.sidebar.number_input(col, value=0.0 if col.lower() != "age" else 45.0)

# Preview DataFrame (single row)
input_df = pd.DataFrame([user_input], columns=raw_cols)
st.subheader("📝 Input preview")
st.dataframe(input_df, use_container_width=True)

# -------------------------
# Transform & predict
# -------------------------
if st.button("🔮 Predict Readmission Risk"):
    try:
        # transform using pipeline (pipeline could wrap the preprocessor, or be a preprocessor itself)
        # If pipeline is a full Pipeline with model inside, adjust accordingly (we assume pipeline is preprocessor)
        if hasattr(pipeline, "predict") and not hasattr(preprocessor, "transform"):
            # In the rare case pipeline itself is (preprocessor + model) used previously, handle carefully
            st.warning("Loaded pipeline appears to contain a model; ensure you load only the preprocessing pipeline separately if possible.")
        input_processed = pipeline.transform(input_df)  # this should produce shape (1, 28)

        # confirm shape vs model expected input
        expected = None
        try:
            # Keras models have input_shape or model.input_shape
            expected = model.input_shape[-1]
        except Exception:
            expected = None

        if expected is not None and input_processed.shape[1] != expected:
            st.error(f"Transformed feature count ({input_processed.shape[1]}) does not match model expected input size ({expected}).\n"
                     "This means the pipeline or model you saved during training differ from the ones you loaded here.")
            st.stop()

        # Predict
        prob = float(model.predict(input_processed)[0][0])
        st.write(f"**Predicted probability of readmission:** {prob:.3f}")
        if prob >= 0.5:
            st.error("⚠️ High Risk of Readmission")
        else:
            st.success("✅ Low Risk of Readmission")

        # show final transformed feature count and (optionally) names if you want
        st.caption(f"Transformed shape: {input_processed.shape}")

        # Try to show encoded feature names (best-effort)
        try:
            if hasattr(preprocessor, "get_feature_names_out"):
                # Attempt to compute final encoded feature names
                try:
                    encoded_feature_names = preprocessor.get_feature_names_out(raw_cols)
                except Exception:
                    # fallback: ask each transformer for its names
                    encoded_feature_names = None
                if encoded_feature_names is not None:
                    st.write("Final transformed feature names (first 50):")
                    st.write(list(encoded_feature_names)[:50])
        except Exception:
            pass

    except Exception as e:
        st.error(f"Prediction failed: {e}")
