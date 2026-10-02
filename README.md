# 🌾 Agricultural Crop Yield Prediction

Streamlit app that predicts crop yield for Indian states using a Random
Forest trained on log-transformed yield.

## KPIs (held-out test set)

| KPI | Value |
|---|---|
| Yield Planning Accuracy | ≥ 92% |
| R² Score | ≈ 0.96 |
| MAE | ≈ 4.3 |
| Missing Value Rate | 0% |

## Units

| Field | Unit |
|---|---|
| Area | hectares (ha) |
| Annual Rainfall | millimetres (mm) |
| Fertilizer | kilograms (kg) |
| Pesticide | kilograms (kg) |
| Production | tonnes (t) |
| Yield (predicted) | tonnes per hectare (t/ha) |

## How it works

1. Cleans the dataset (drops zero-yield rows, caps extreme yields)
2. Engineers features (log scales, fertilizer/pesticide per area, interactions)
3. Leakage-safe target encoding for crop/state (train rows only)
4. Trains a 300-tree Random Forest on `log1p(yield)` (~1 minute on first load)
5. Serves predictions through this Streamlit app

Built with Python, pandas, and scikit-learn.
