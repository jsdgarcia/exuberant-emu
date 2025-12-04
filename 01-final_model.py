import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline


# ============================================================
# --- Load data ---
# ============================================================
pickle_path = 'df_merged_fin.pkl'
print(f"Loading data from {pickle_path}...")

try:
    df_merged = pd.read_pickle(pickle_path)
except Exception:
    with open(pickle_path, 'rb') as f:
        df_merged = pickle.load(f)

print(f"Data loaded. Shape: {df_merged.shape}")

# ============================================================
# --- Build features (corrected duration column) ---
# ============================================================

df_merged["segment_duration_min"] = df_merged["seg_duration_hrs"] * 60

df_merged["mean_groundspeed"] = (
    df_merged["seg_flight_distance_km"] /
    df_merged["seg_duration_hrs"].replace(0, np.nan)
)

df_merged["mean_altitude"] = (df_merged["seg_altitude_max"] + df_merged["seg_altitude_min"]) / 2
df_merged["altitude_variation"] = df_merged["seg_altitude_diff"]
df_merged["mean_vertical_rate"] = df_merged["seg_avg_rate_of_climb"]
df_merged["std_vertical_rate"] = df_merged["seg_sd_rate_of_climb"]
df_merged["distance_km"] = df_merged["seg_flight_distance_km"].clip(lower=1.0)
df_merged["mean_mach"] = 0.0
df_merged["elevation_diff"] = df_merged["destination_elevation"] - df_merged["origin_elevation"]

from sklearn.preprocessing import LabelEncoder
df_merged["aircraft_code"] = LabelEncoder().fit_transform(df_merged["aircraft_type"].astype(str))


# ============================================================
# --- Define features ---
# ============================================================
categorical_features = ["aircraft_type"]
numeric_features = [
    "segment_duration_min",
    "mean_groundspeed",
    "distance_km",
    "mean_altitude",
    "altitude_variation",
    "mean_vertical_rate",
    "std_vertical_rate",
    "mean_mach",
    "elevation_diff",
    "aircraft_code"
]

target = "fuel_kg"

# ============================================================
# --- Split into train and rank datasets ---
# ============================================================
df_train = df_merged[df_merged["seg_dataset"] == "train"].copy()
df_rank  = df_merged[df_merged["seg_dataset"] == "final"].copy()   

print("Train shape:", df_train.shape)
print("Rank shape:", df_rank.shape)


df_train = df_train.replace([np.inf, -np.inf], np.nan)
df_train = df_train.dropna(subset=[target] + numeric_features)

X = df_train[categorical_features + numeric_features]
y = df_train[target]

print("Final training shape:", X.shape)

# ============================================================
# --- Preprocessing + LightGBM ---
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

model = LGBMRegressor(
    boosting_type="gbdt",
    n_estimators=5000,
    learning_rate=0.015,
    max_depth=-1,
    num_leaves=768,
    max_bin=512,
    subsample=0.7,
    colsample_bytree=0.7,
    feature_fraction_bynode=0.8,
    reg_alpha=0.4,
    reg_lambda=2.0,
    min_child_samples=50,
    min_child_weight=1,
    min_split_gain=0.01,
    extra_trees=True,
    n_jobs=-1,
    random_state=42,
    force_col_wise=True
)

# ============================================================
# --- Train-test split ---
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

print("\n🚀 Training LightGBM model...")
pipeline.fit(X_train, y_train)
print("Model training complete.")

# ============================================================
# --- Evaluate ---
# ============================================================
y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n📊 RMSE on test set: {rmse:.2f} kg")

# ============================================================
# --- Predict on rank dataset ---
# ============================================================
print("\nPredicting on final dataset...")
X_rank = df_rank[categorical_features + numeric_features]
y_rank_pred = pipeline.predict(X_rank)

df_rank_out = df_rank[['idx', 'flight_id', 'start', 'end']].copy()
df_rank_out['fuel_kg'] = y_rank_pred

# ============================================================
# --- Save predictions ---
# ============================================================
csv_path = Path("fuel_rank_predictions.csv")
parquet_path = Path("exuberant-emu_final.parquet")

df_rank_out.to_csv(csv_path, index=False)
df_rank_out.to_parquet(parquet_path, index=False)

print(f"\n✅ Rank predictions saved to: {csv_path.resolve()}")
print(f"✅ Submission parquet saved to: {parquet_path.resolve()}")
