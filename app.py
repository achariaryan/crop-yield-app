# -*- coding: utf-8 -*-
"""Streamlit app for crop yield prediction (deployable version)."""
import os
import time

import numpy as np
import pandas as pd
import streamlit as st

from model_pipeline import predict_yield, train_model

# Optional fallback if the host ever runs low on memory: add
# RF_N_ESTIMATORS = "150" to the app's Secrets (settings page) and redeploy.
try:
    if "RF_N_ESTIMATORS" in st.secrets:
        os.environ["RF_N_ESTIMATORS"] = str(st.secrets["RF_N_ESTIMATORS"])
except Exception:
    pass

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
)


@st.cache_resource
def get_bundle():
    """Train once per server start, then cache for every rerun."""
    return train_model()


with st.spinner(
    "Training model — first load takes about a minute, "
    "after that everything is instant..."
):
    bundle = get_bundle()

kpis = bundle["kpis"]
df = bundle["raw_df"]

st.title("🌾 Agricultural Crop Yield Prediction")

st.write("Machine Learning based crop yield prediction system")

st.divider()


# KPI SECTION

st.subheader("📊 Project KPIs")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Yield Planning Accuracy", f"{kpis['yield_accuracy']:.2f}%")
col2.metric("R² Score", f"{kpis['r2']:.4f}")
col3.metric("MAE (t/ha)", f"{kpis['mae']:.4f}")
col4.metric("Missing Value Rate", f"{kpis['missing_rate']:.2f}%")
col5.metric("Dataset Records", f"{len(df):,}")

st.divider()


# INPUT SECTION

st.subheader("🌱 Enter Agricultural Information")

col1, col2 = st.columns(2)

with col1:
    crop = st.selectbox("Crop", sorted(df["Crop"].unique()))
    state = st.selectbox("State", sorted(df["State"].unique()))
    season = st.selectbox("Season", sorted(df["Season"].unique()))
    year = st.number_input(
        "Future Crop Year", min_value=2021, max_value=2035,
        value=2026, step=1,
    )

with col2:
    area = st.number_input("Area (hectares)", min_value=0.01, value=100.0)
    rainfall = st.number_input(
        "Annual Rainfall (mm)", min_value=0.0,
        value=float(df["Annual_Rainfall"].median()),
    )
    fertilizer = st.number_input(
        "Fertilizer (kg)", min_value=0.0,
        value=float(df["Fertilizer"].median()),
    )
    pesticide = st.number_input(
        "Pesticide (kg)", min_value=0.0,
        value=float(df["Pesticide"].median()),
    )

st.divider()


# PREDICTION

if st.button("🌾 Predict Crop Yield", use_container_width=True):
    start_time = time.perf_counter()
    prediction = predict_yield(bundle, dict(
        crop=crop, state=state, season=season, year=year,
        area=area, rainfall=rainfall, fertilizer=fertilizer,
        pesticide=pesticide,
    ))
    latency = time.perf_counter() - start_time

    st.success(f"🌾 Predicted Crop Yield: {prediction:.4f} tonnes per hectare (t/ha)")
    st.info(f"⚡ Prediction Latency: {latency:.4f} seconds")

st.divider()


# KPI DETAILS

st.subheader("📋 KPI Details")

kpi_table = pd.DataFrame({
    "KPI": [
        "Yield Planning Accuracy", "R² Score", "MAE",
        "Missing Value Rate", "Prediction Latency",
    ],
    "Category": [
        "Business KPI", "ML / Model KPI", "ML / Model KPI",
        "Data Quality KPI", "Product / Engineering KPI",
    ],
    "Measurement": [
        "100 - (MAE / Mean Actual Yield × 100)",
        "Coefficient of determination",
        "Mean Absolute Error (t/ha)",
        "(Missing Values / Total Values) × 100",
        "Prediction end time - start time",
    ],
    "Target": ["≥ 85%", "Higher is better", "Lower is better", "0%", "≤ 2 seconds"],
    "Actual Result": [
        f"{kpis['yield_accuracy']:.2f}%",
        f"{kpis['r2']:.4f}",
        f"{kpis['mae']:.4f}",
        f"{kpis['missing_rate']:.2f}%",
        "Measured during prediction",
    ],
})

st.dataframe(kpi_table, use_container_width=True, hide_index=True)

st.caption(
    "Final Model: Random Forest Regression (log target, 300 trees) | "
    "Production excluded to prevent potential data leakage | "
    "Yields in tonnes per hectare (t/ha); area in hectares; "
    "rainfall in mm; fertilizer & pesticide in kg"
)
