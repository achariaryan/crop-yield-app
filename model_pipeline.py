# -*- coding: utf-8 -*-
"""Crop yield model pipeline — exact same steps as the validated Colab notebook:
cleaning, engineered features, leakage-safe target encoding, log-target
Random Forest, train/val/test split (random_state=42).
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "crop_yield.csv")

# tune via env vars if the hosting RAM/CPU is tight
N_ESTIMATORS = int(os.environ.get("RF_N_ESTIMATORS", "300"))
MAX_DEPTH = os.environ.get("RF_MAX_DEPTH") or None


def load_clean_data():
    df = pd.read_csv(CSV_PATH)

    # --- cleaning (notebook cell 6): cap computed BEFORE dropping zeros,
    #     exactly like the notebook ---
    yield_cap = df["Yield"].quantile(0.995)
    df = df[df["Yield"] > 0].copy()
    df["Yield"] = df["Yield"].clip(upper=yield_cap)

    # --- engineered features ---
    df["Fertilizer_per_Area"] = df["Fertilizer"] / df["Area"]
    df["Pesticide_per_Area"] = df["Pesticide"] / df["Area"]
    for col in ["Area", "Fertilizer", "Pesticide"]:
        df["log_" + col] = np.log1p(df[col])
    df["Fert_x_Pest"] = df["Fertilizer"] * df["Pesticide"]
    return df


def train_model(progress=None):
    """Train the final model. Returns a bundle dict used by predict()."""
    def step(msg):
        if progress:
            progress(msg)

    step("Loading data...")
    df = load_clean_data()
    missing_rate = (df.isnull().sum().sum() / df.size) * 100

    X = df.drop(["Yield", "Production"], axis=1)
    y = df["Yield"]
    X = pd.get_dummies(X, columns=["Crop", "Season", "State"], drop_first=True)

    step("Splitting train / validation / test...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    # --- leakage-safe target encoding (train rows only) ---
    train_df = df.loc[X_train.index]
    crop_map = train_df.groupby("Crop")["Yield"].mean()
    state_map = train_df.groupby("State")["Yield"].mean()
    global_mean = float(y_train.mean())

    df["Crop_mean_yield"] = df["Crop"].map(crop_map).fillna(global_mean)
    df["State_mean_yield"] = df["State"].map(state_map).fillna(global_mean)
    for split in (X_train, X_val, X_test):
        split["Crop_mean_yield"] = df.loc[split.index, "Crop_mean_yield"]
        split["State_mean_yield"] = df.loc[split.index, "State_mean_yield"]

    step(f"Training Random Forest ({N_ESTIMATORS} trees, log target)...")
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=42,
        n_jobs=-1,
    )
    # refit on train+val, like the notebook's final model
    X_final = pd.concat([X_train, X_val])
    y_final = pd.concat([y_train, y_val])
    model.fit(X_final, np.log1p(y_final))

    step("Evaluating on the held-out test set...")
    pred = np.expm1(model.predict(X_test))
    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))
    yield_accuracy = max(0.0, 100 - (mae / float(y_test.mean()) * 100))

    kpis = {
        "yield_accuracy": yield_accuracy,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "missing_rate": missing_rate,
    }

    return {
        "model": model,
        "feature_columns": X_train.columns.tolist(),
        "crop_map": crop_map,
        "state_map": state_map,
        "global_mean": global_mean,
        "kpis": kpis,
        "raw_df": pd.read_csv(CSV_PATH),
    }


def predict_yield(bundle, inputs):
    """inputs: crop, state, season, year, area, rainfall, fertilizer, pesticide.
    Reproduces the (validated) notebook/Streamlit feature construction exactly."""
    feature_columns = bundle["feature_columns"]
    crop_map = bundle["crop_map"]
    state_map = bundle["state_map"]
    global_mean = bundle["global_mean"]

    crop = inputs["crop"]
    state = inputs["state"]

    row = pd.DataFrame({
        "Crop": [crop],
        "Crop_Year": [int(inputs["year"])],
        "Season": [inputs["season"]],
        "State": [state],
        "Area": [float(inputs["area"])],
        "Annual_Rainfall": [float(inputs["rainfall"])],
        "Fertilizer": [float(inputs["fertilizer"])],
        "Pesticide": [float(inputs["pesticide"])],
    })

    crop_value = crop_map.get(crop, global_mean)
    state_value = state_map.get(state, global_mean)
    if pd.isna(crop_value):
        crop_value = global_mean
    if pd.isna(state_value):
        state_value = global_mean

    for col in ["Crop", "Season", "State"]:
        for feat in feature_columns:
            if (
                feat.startswith(col + "_")
                and feat not in row.columns
                and not feat.endswith("_mean_yield")
            ):
                row[feat] = (row[col] == feat[len(col) + 1:]).astype(int)
        row = row.drop(columns=[col])

    row["Fertilizer_per_Area"] = row["Fertilizer"] / row["Area"]
    row["Pesticide_per_Area"] = row["Pesticide"] / row["Area"]
    for col in ["Area", "Fertilizer", "Pesticide"]:
        row["log_" + col] = np.log1p(row[col])
    row["Fert_x_Pest"] = row["Fertilizer"] * row["Pesticide"]
    row["Crop_mean_yield"] = crop_value
    row["State_mean_yield"] = state_value

    row = row.reindex(columns=feature_columns, fill_value=0)
    return float(np.expm1(bundle["model"].predict(row)[0]))
