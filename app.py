from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import load_model

MODEL_PATH = Path(__file__).resolve().parent / "saved_model.pkl"

st.set_page_config(
    page_title="Train Journey Time Prediction",
    page_icon="🚆"
)

st.title("🚆 Train Journey Time Prediction System")

st.write("Enter the journey details below.")

distance = st.number_input(
    "Total Distance (km)",
    min_value=0.0,
    value=100.0
)

stops = st.number_input(
    "Number of Stops",
    min_value=1,
    value=5
)

if st.button("Predict Journey Time"):

    model = load_model(MODEL_PATH)

    input_df = pd.DataFrame({
        "Distance": [distance],
        "Stops": [stops]
    })

    prediction = model.predict(input_df)

    st.success(
        f"Predicted Journey Duration: {prediction[0]:.2f} minutes"
    )