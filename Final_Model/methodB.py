
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

OUTDIR = Path("lsds55_outputs_expertA_methodB")
TARGET = "Biofilm surface area coverage (%)"
CLEAN_CSV = Path("cleaned_combined.csv")

def met(y_true, y_pred):
    if len(y_true) == 0:
        return {"note": "no rows"}
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
        "n": int(len(y_true)),
    }

df = pd.read_csv(CLEAN_CSV)

# Make sure required columns exist; if not, stop early
required_cols = {"BaseMaterial", "DayInt", "condition", TARGET}
missing = required_cols - set(df.columns)
if missing:
    raise RuntimeError(f"Missing required columns in cleaned_combined.csv: {missing}")

# Drop rows without essential info
df = df.dropna(subset=["BaseMaterial", "DayInt", "condition", TARGET]).copy()
df["condition_bin"] = (df["condition"] == "flight").astype(int)

# Numeric + categorical setup
num_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
if TARGET in num_candidates:
    num_candidates.remove(TARGET)

# Keep DayInt and condition_bin explicitly at the end for readability
num_cols = [c for c in num_candidates if c not in ["DayInt", "condition_bin"]]
for c in ["DayInt", "condition_bin"]:
    if c in df.columns and c not in num_cols:
        num_cols.append(c)

cat_cols = [c for c in ["BaseMaterial", "condition"] if c in df.columns]

pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

rf_base = RandomForestRegressor(random_state=42, n_estimators=500, n_jobs=-1)

param_grid = {
    "model__max_depth": [3, 5, None],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": [None],
}

pipe = Pipeline([("pre", pre), ("model", rf_base)])
cv = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
X_all = df[num_cols + cat_cols]
y_all = df[TARGET].values

cv.fit(X_all, y_all)
best_model = cv.best_estimator_
print("Best Expert A params:", cv.best_params_)

# Add Expert A predictions to df
df["expertA_pred"] = best_model.predict(X_all)

plots_dir = OUTDIR
plots_dir.mkdir(exist_ok=True)

Y_MIN, Y_MAX = 0.0, 100.0  # standardized axis for coverage
materials = sorted(df["BaseMaterial"].dropna().unique())

traj_rows = []

for mat in materials:
    mat_sub = df[df["BaseMaterial"] == mat].copy()
    g_sub = mat_sub[mat_sub["condition"] == "ground"]
    f_sub = mat_sub[mat_sub["condition"] == "flight"]

    # Need >=2 ground days for linear trajectory
    if g_sub["DayInt"].nunique() < 2:
        continue

    Xg = g_sub[["DayInt"]].values
    yg = g_sub[TARGET].values
    lin = LinearRegression().fit(Xg, yg)

    slope = float(lin.coef_[0])
    intercept = float(lin.intercept_)

    # Ground growth rate (Day1 -> Day3) for Method B
    growth_rate = np.nan
    if (g_sub["DayInt"] == 1).any() and (g_sub["DayInt"] == 3).any():
        cov_d1 = g_sub.loc[g_sub["DayInt"] == 1, TARGET].mean()
        cov_d3 = g_sub.loc[g_sub["DayInt"] == 3, TARGET].mean()
        growth_rate = (cov_d3 - cov_d1) / 2.0

    pred_day3_A = float(lin.predict(np.array([[3]]))[0])

    pred_day3_B = np.nan
    actual_flight_d3 = np.nan
    if not f_sub.empty:
        f_d1 = f_sub[f_sub["DayInt"] == 1]
        f_d3 = f_sub[f_sub["DayInt"] == 3]
        if not f_d3.empty:
            actual_flight_d3 = float(f_d3[TARGET].mean())
        if not f_d1.empty and not np.isnan(growth_rate):
            pred_day3_B = float(f_d1[TARGET].mean() + 2.0 * growth_rate)

    traj_rows.append({
        "BaseMaterial": mat,
        "ground_lin_slope": slope,
        "ground_lin_intercept": intercept,
        "ground_growth_rate_per_day": growth_rate,
        "predicted_flight_day3_methodA_ground_line": pred_day3_A,
        "predicted_flight_day3_methodB_flightD1_plus_ground_rate": pred_day3_B,
        "actual_flight_day3": actual_flight_d3,
    })

    # Ground
    if not g_sub.empty:
        g_mean = (
            g_sub.groupby("DayInt")
            .agg(actual=(TARGET, "mean"), pred=("expertA_pred", "mean"))
            .reset_index()
        )
    else:
        g_mean = None

    # Flight
    if not f_sub.empty:
        f_mean = (
            f_sub.groupby("DayInt")
            .agg(actual=(TARGET, "mean"), pred=("expertA_pred", "mean"))
            .reset_index()
        )
    else:
        f_mean = None

    plt.figure()

    # Ground actual scatter
    plt.scatter(g_sub["DayInt"], g_sub[TARGET],
                label="Ground (actual)", color="tab:blue")

    # Ground linear fit (old "Ground fit")
    xs = np.array([[1], [2], [3]])
    ys = lin.predict(xs)
    plt.plot(xs.flatten(), ys, linestyle="--",
             label="Ground fit (linear)", color="tab:blue")

    # Flight actual scatter
    if not f_sub.empty:
        plt.scatter(f_sub["DayInt"], f_sub[TARGET],
                    label="Flight (actual)", color="tab:orange")

    # Old trajectory predictions (Method A & B)
    if not np.isnan(pred_day3_B):
        plt.scatter([3], [pred_day3_B], marker="x",
                    label="Pred Flight D3 (Method B)", color="tab:green")
    plt.scatter([3], [pred_day3_A], marker="^",
                label="Pred Flight D3 (Method A)", color="tab:red")

    # Expert A ground predictions (day-wise mean)
    if g_mean is not None and len(g_mean) > 0:
        plt.plot(
            g_mean["DayInt"],
            g_mean["pred"],
            "-.",
            color="tab:purple",
            label="Ground Expert A (mean pred)",
        )

    # Expert A flight predictions (day-wise mean)
    if f_mean is not None and len(f_mean) > 0:
        plt.plot(
            f_mean["DayInt"],
            f_mean["pred"],
            "-.",
            color="tab:brown",
            label="Flight Expert A (mean pred)",
        )

    plt.xlabel("Day")
    plt.ylabel("Biofilm surface area coverage (%)")
    plt.title(f"Trajectory with Expert A (BaseMaterial): {mat}")
    plt.xlim(0.8, 3.2)
    plt.xticks([1, 2, 3])
    plt.ylim(Y_MIN, Y_MAX)
    plt.legend()
    plt.tight_layout()

    safe_name = "".join(ch if ch.isalnum() else "_" for ch in mat).strip("_")
    plt.savefig(plots_dir / f"trajectory_expertA_{safe_name}.png", dpi=150)
    plt.close()

traj_df = pd.DataFrame(traj_rows)
traj_df.to_csv(plots_dir / "trajectory_expertA_summary.csv", index=False)

print(f"Done. Expert A trajectory plots saved in: {plots_dir}")
