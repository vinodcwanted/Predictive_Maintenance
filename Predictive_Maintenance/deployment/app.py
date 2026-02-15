import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ---------------------------
# Download + load model (PUBLIC repo, so no token needed)
# ---------------------------
model_path = hf_hub_download(
    repo_id="vinodcwanted/Predictive-Maintenance",
    filename="best_engine_xgb_model.joblib"
)
model = joblib.load(model_path)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Predictive Maintenance - Engine Condition Prediction")
st.write(
    "Enter the engine sensor values below. The model will predict **Engine Condition** (0/1)."
)

# ---------------------------
# Collect user input (6 parameters)
# ---------------------------
engine_rpm = st.number_input("Engine rpm", min_value=0.0, value=700.0, step=1.0)
lub_oil_pressure = st.number_input("Lub oil pressure", min_value=0.0, value=2.5, step=0.01)
fuel_pressure = st.number_input("Fuel pressure", min_value=0.0, value=12.0, step=0.01)
coolant_pressure = st.number_input("Coolant pressure", min_value=0.0, value=3.0, step=0.01)
lub_oil_temp = st.number_input("lub oil temp", min_value=0.0, value=80.0, step=0.1)
coolant_temp = st.number_input("Coolant temp", min_value=0.0, value=82.0, step=0.1)

# Build input DataFrame (MUST match training column names exactly)
input_data = pd.DataFrame([{
    "Engine rpm": engine_rpm,
    "Lub oil pressure": lub_oil_pressure,
    "Fuel pressure": fuel_pressure,
    "Coolant pressure": coolant_pressure,
    "lub oil temp": lub_oil_temp,
    "Coolant temp": coolant_temp
}])

# ---------------------------
# Prediction
# ---------------------------
classification_threshold = 0.50

if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = int(prediction_proba >= classification_threshold)

    # Labels + comments
    if prediction == 1:
        condition_label = "Faulty"
        comment = "⚠️ Engine is at risk of failure. Please schedule maintenance soon."
        st.error(comment)
    else:
        condition_label = "Normal"
        comment = "✅ Operation is normal. No need to worry."
        st.success(comment)

    st.write(f"**Predicted probability (Engine Condition = 1 / Faulty):** {prediction_proba:.4f}")
    st.write(f"**Engine Condition  :** {prediction} (**{condition_label}**)")
