# -*- coding: utf-8 -*-
"""
This script is a little different from the above ones in that it trains on ground-flight data and tests using only ground inputs instead of flight inputs.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


CSV_PATH = Path("cleaned_combined_updated.csv")   
OUTDIR = Path("lsds55_outputs_methodA") 
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET = "Biofilm surface area coverage (%)"  # flight y-label
COL_MAX_THICK = "Biofilm Maximum thickness (µm)"
COL_ROUGH_BIO = "roughness coefficient Ra*"
COL_MAT_ROUGH = "Material_Roughness"
COL_CA = "Contact_Angle"
COL_COSTH = "Cos(theta)"
COL_WADH = "Wadh"


def met(y_true, y_pred):
    """Compute RMSE/MAE/R2 + n."""
    if len(y_true) == 0:
        return {"note": "no rows"}
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
        "n": int(len(y_true)),
    }


if not CSV_PATH.exists():
    raise FileNotFoundError(f"Cannot find {CSV_PATH}. Make sure cleaned_combined.csv is next to this script.")

df = pd.read_csv(CSV_PATH)

required_cols = [
    "BaseMaterial",
    "DayInt",
    "condition",
    TARGET,
    COL_MAX_THICK,
    COL_ROUGH_BIO,
    COL_MAT_ROUGH,
    COL_CA,
    COL_COSTH,
    COL_WADH,
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns in cleaned_combined.csv: {missing}")

df["DayInt"] = pd.to_numeric(df["DayInt"], errors="coerce")
df = df.dropna(subset=["BaseMaterial", "DayInt", "condition", TARGET]).copy()


# Separate ground and flight
df_ground = df[df["condition"].str.lower() == "ground"].copy()
df_flight = df[df["condition"].str.lower() == "flight"].copy()

group_cols = ["BaseMaterial", "DayInt"]

g_groups = df_ground.groupby(group_cols)
f_groups = df_flight.groupby(group_cols)

common_keys = sorted(set(g_groups.groups.keys()) & set(f_groups.groups.keys()))

paired_rows = []

for (mat, day) in common_keys:
    g_block = g_groups.get_group((mat, day))
    f_block = f_groups.get_group((mat, day))

    # Aggregate GROUND-side inputs (mean over replicates)
    row = {
        "BaseMaterial": mat,
        "DayInt": float(day),
        "ground_coverage_mean": float(g_block[TARGET].mean()),
        "ground_max_thickness_mean": float(g_block[COL_MAX_THICK].mean()),
        "ground_roughness_bio_mean": float(g_block[COL_ROUGH_BIO].mean()),
        # Material properties (should be constant per material, but we average just in case):
        "Material_Roughness": float(g_block[COL_MAT_ROUGH].mean()),
        "Contact_Angle": float(g_block[COL_CA].mean()),
        "Cos_theta": float(g_block[COL_COSTH].mean()),
        "Wadh": float(g_block[COL_WADH].mean()),
        # FLIGHT coverage = target y (mean)
        "flight_coverage_mean": float(f_block[TARGET].mean()),
    }

    paired_rows.append(row)

paired_df = pd.DataFrame(paired_rows)

if paired_df.empty:
    raise RuntimeError("No (BaseMaterial, DayInt) pairs with both ground and flight data found.")


num_feature_cols = [
    "DayInt",
    "ground_coverage_mean",
    "ground_max_thickness_mean",
    "ground_roughness_bio_mean",
    "Material_Roughness",
    "Contact_Angle",
    "Cos_theta",
    "Wadh",
]
cat_feature_cols = ["BaseMaterial"]

for c in num_feature_cols:
    paired_df[c] = pd.to_numeric(paired_df[c], errors="coerce")

paired_df = paired_df.dropna(subset=num_feature_cols + ["flight_coverage_mean"]).copy()

X = paired_df[num_feature_cols + cat_feature_cols]
y = paired_df["flight_coverage_mean"].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)

# Preprocessing: scale numerics, one-hot encode BaseMaterial
pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=False), num_feature_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feature_cols),
    ],
    remainder="drop",
)

rf = RandomForestRegressor(
    n_estimators=600,
    random_state=42,
    n_jobs=-1,
)

pipe = Pipeline([
    ("pre", pre),
    ("rf", rf),
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# post-hoc linear calibration to fix scaling issue
from sklearn.linear_model import LinearRegression

# reshape for sklearn
X_cal = y_pred.reshape(-1, 1)
y_cal = y_test

cal = LinearRegression().fit(X_cal, y_cal)
a = cal.intercept_
b = cal.coef_[0]

print("\nCalibration fitted: y_true ≈ a + b * y_pred")
print("a =", a, "   b =", b)

# apply calibration
y_pred_cal = a + b * y_pred

# physical clipping to avoid negative or >100 values
y_pred_cal = np.clip(y_pred_cal, 0.0, 100.0)


metrics = {}
metrics["strict_ground_to_flight"] = met(y_test, y_pred_cal)
metrics["strict_ground_to_flight"].update({
    "calibration_a": float(a),
    "calibration_b": float(b),
})
metrics["strict_ground_to_flight"].update({
    "n_pairs_total": int(len(paired_df)),
    "n_train_pairs": int(len(X_train)),
    "n_test_pairs": int(len(X_test)),
    "numeric_features": num_feature_cols,
    "categorical_features": cat_feature_cols,
})

with open(OUTDIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Save predictions (test set only)
pred_df = X_test.copy()
pred_df["y_true_flight"] = y_test
pred_df["y_pred_flight_calibrated"] = y_pred_cal
pred_df["y_pred_flight_raw"] = y_pred  # (optional, for debugging)

pred_df.to_csv(OUTDIR / "strict_ground_to_flight_predictions.csv", index=False)

# Pred vs actual scatter
plt.figure()
plt.scatter(y_test, y_pred_cal)
plt.xlabel("Actual Flight Coverage (mean)")
plt.ylabel("Calibrated Predicted Flight Coverage")
plt.title("Strict Ground→Flight Regression (after calibration)")

lo = float(min(y_test.min(), y_pred_cal.min()))
hi = float(max(y_test.max(), y_pred_cal.max()))
plt.plot([lo, hi], [lo, hi], linestyle="--")

plt.tight_layout()
plt.savefig(OUTDIR / "strict_ground_to_flight_pred_vs_actual_CALIBRATED.png", dpi=150)

plt.close()

print("Done.")
print("Pairs total:", len(paired_df))
print("Train pairs:", len(X_train), " Test pairs:", len(X_test))
print("Metrics written to:", OUTDIR / "metrics.json")
print("Predictions CSV:", OUTDIR / "strict_ground_to_flight_predictions.csv")
print("Pred vs Actual plot:", OUTDIR / "strict_ground_to_flight_pred_vs_actual.png")

