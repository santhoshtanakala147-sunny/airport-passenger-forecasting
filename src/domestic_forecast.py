import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")


# ============================================================
# 1. LOAD DOMESTIC DATA
# ============================================================

DATA_FILE = "data/domestic_data.xlsx"

df = pd.read_excel(DATA_FILE, sheet_name="data")

print("=" * 70)
print("DOMESTIC AIRPORT PASSENGER FORECASTING")
print("=" * 70)

print(f"Dataset shape : {df.shape}")
print(f"Years covered : {sorted(df['Year'].unique())}")
print(
    f"Unique routes : "
    f"{df[['Origin', 'Dest']].drop_duplicates().shape[0]}"
)

if 2025 in df["Year"].unique():
    print(
        f"2025 months   : "
        f"{sorted(df[df['Year'] == 2025]['Month'].unique())}"
    )


# ============================================================
# 2. CLEAN NUMERICAL DATA
# ============================================================

num_cols = [
    "Pax From Origin",
    "Pax To Origin",
    "Freight From Origin",
    "Frieght To Origin",
    "Mail From Origin",
    "Mail To Origin"
]

for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).fillna(0)


# Sort initial data
df = df.sort_values(
    ["Year", "Month"]
).reset_index(drop=True)


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

# ------------------------------------------------------------
# Passenger ratio
# ------------------------------------------------------------

df["pax_ratio"] = (
    df["Pax To Origin"] /
    df["Pax From Origin"].replace(0, np.nan)
).fillna(1.0).clip(0, 10)


# ------------------------------------------------------------
# Freight ratio
# ------------------------------------------------------------

df["freight_ratio"] = (
    df["Frieght To Origin"] /
    df["Freight From Origin"].replace(0, np.nan)
).fillna(1.0).clip(0, 20)


# ------------------------------------------------------------
# Mail ratio
# ------------------------------------------------------------

df["mail_ratio"] = (
    df["Mail To Origin"] /
    df["Mail From Origin"].replace(0, np.nan)
).fillna(1.0).clip(0, 20)


# ------------------------------------------------------------
# Total freight
# ------------------------------------------------------------

df["total_freight"] = (
    df["Freight From Origin"] +
    df["Frieght To Origin"]
)


# ------------------------------------------------------------
# Seasonal features
# ------------------------------------------------------------

df["month_sin"] = np.sin(
    2 * np.pi * df["Month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["Month"] / 12
)


# ============================================================
# 4. ROUTE-WISE LAG FEATURES
# ============================================================

df = df.sort_values(
    ["Origin", "Dest", "Year", "Month"]
).reset_index(drop=True)


# Previous month passenger traffic
df["lag_1_pax"] = (
    df.groupby(["Origin", "Dest"])["Pax From Origin"]
    .shift(1)
)


# Same month previous year
df["lag_12_pax"] = (
    df.groupby(["Origin", "Dest"])["Pax From Origin"]
    .shift(12)
)


# Year-over-year growth
df["yoy_growth"] = (
    (
        df["Pax From Origin"] - df["lag_12_pax"]
    )
    /
    df["lag_12_pax"].replace(0, np.nan)
).fillna(0).clip(-1, 5)


# Return chronological order
df = df.sort_values(
    ["Year", "Month"]
).reset_index(drop=True)


# ============================================================
# 5. DEFINE TARGET AND FEATURES
# ============================================================

target = "Pax From Origin"

cat_features = [
    "Origin",
    "Dest"
]

num_features = [
    "Year",
    "Month",

    "Freight From Origin",
    "Mail From Origin",

    "pax_ratio",
    "freight_ratio",
    "mail_ratio",

    "total_freight",

    "month_sin",
    "month_cos",

    "lag_12_pax",
    "lag_1_pax",

    "yoy_growth"
]

all_features = num_features + cat_features


# ============================================================
# 6. REMOVE INITIAL LAG NaN ROWS
# ============================================================

df_model = df.dropna(
    subset=["lag_12_pax", "lag_1_pax"]
).reset_index(drop=True)

print()
print(
    f"Rows after dropping lag NaNs : "
    f"{df_model.shape[0]}"
)

print(
    f"Features used                : "
    f"{len(all_features)}"
)


# ============================================================
# 7. TIME-BASED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

n = len(df_model)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df_model.iloc[:train_end]
val_df = df_model.iloc[train_end:val_end]
test_df = df_model.iloc[val_end:]


X_train = train_df[all_features]
y_train = train_df[target]

X_val = val_df[all_features]
y_val = val_df[target]

X_test = test_df[all_features]
y_test = test_df[target]


print()
print("=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    f"Train : {len(X_train)} rows "
    f"({train_df['Year'].min()} – "
    f"{train_df['Year'].max()})"
)

print(
    f"Val   : {len(X_val)} rows "
    f"({val_df['Year'].min()} – "
    f"{val_df['Year'].max()})"
)

print(
    f"Test  : {len(X_test)} rows "
    f"({test_df['Year'].min()} – "
    f"{test_df['Year'].max()})"
)


# ============================================================
# 8. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            num_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            cat_features
        )
    ]
)


# ============================================================
# 9. DEFINE MODELS
# ============================================================

models = {

    "Ridge":
        Ridge(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=200,
            random_state=42
        ),

    "XGBoost":
        XGBRegressor(
            n_estimators=200,
            random_state=42,
            verbosity=0
        ),

    "LightGBM":
        LGBMRegressor(
            n_estimators=200,
            random_state=42,
            verbose=-1
        )
}


# ============================================================
# 10. MODEL COMPARISON
# ============================================================

results = {}
trained_pipes = {}

print()
print("=" * 75)
print(
    f"{'Model':<22}"
    f"{'Val R2':>10}"
    f"{'Val MAE':>14}"
    f"{'Val RMSE':>14}"
    f"{'Val MSE':>18}"
)
print("=" * 75)


for name, model in models.items():

    pipe = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "regressor",
                model
            )
        ]
    )

    # Train only on training set
    pipe.fit(
        X_train,
        y_train
    )

    # Validation prediction
    y_val_pred = pipe.predict(X_val)

    r2 = r2_score(
        y_val,
        y_val_pred
    )

    mae = mean_absolute_error(
        y_val,
        y_val_pred
    )

    mse = mean_squared_error(
        y_val,
        y_val_pred
    )

    rmse = np.sqrt(mse)

    results[name] = {
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }

    trained_pipes[name] = pipe

    print(
        f"{name:<22}"
        f"{r2:>10.4f}"
        f"{mae:>14.1f}"
        f"{rmse:>14.1f}"
        f"{mse:>18.1f}"
    )


print("=" * 75)


# ============================================================
# 11. SELECT BEST MODEL
# ============================================================

results_df = (
    pd.DataFrame(results)
    .T
    .sort_values(
        "R2",
        ascending=False
    )
)

best_model_name = results_df.index[0]

print()
print(
    f"Best model by Validation R² : "
    f"{best_model_name}"
)

print(
    f"Validation R²               : "
    f"{results_df.loc[best_model_name, 'R2']:.4f}"
)


# ============================================================
# 12. HYPERPARAMETER TUNING
# ============================================================

param_grids = {

    "Ridge": {
        "regressor__alpha":
        [0.1, 1, 10, 100]
    },

    "Random Forest": {
        "regressor__n_estimators":
        [100, 200],

        "regressor__max_depth":
        [None, 10, 20]
    },

    "Gradient Boosting": {
        "regressor__n_estimators":
        [100, 200],

        "regressor__learning_rate":
        [0.05, 0.1]
    },

    "XGBoost": {
        "regressor__n_estimators":
        [100, 200],

        "regressor__learning_rate":
        [0.05, 0.1],

        "regressor__max_depth":
        [4, 6]
    },

    "LightGBM": {
        "regressor__n_estimators":
        [100, 200],

        "regressor__learning_rate":
        [0.05, 0.1],

        "regressor__num_leaves":
        [31, 63]
    }
}


print()
print("=" * 70)
print(
    f"Tuning {best_model_name}"
)
print("=" * 70)


tscv = TimeSeriesSplit(
    n_splits=5
)


grid_search = GridSearchCV(
    estimator=trained_pipes[best_model_name],
    param_grid=param_grids[best_model_name],
    cv=tscv,
    scoring="r2",
    n_jobs=-1
)


# IMPORTANT:
# Hyperparameter tuning uses TRAIN data only
grid_search.fit(
    X_train,
    y_train
)


tuned_model = (
    grid_search.best_estimator_
)


print(
    "Best parameters :",
    grid_search.best_params_
)


# ============================================================
# 13. TUNED MODEL — VALIDATION
# ============================================================

y_val_tuned = tuned_model.predict(
    X_val
)

val_r2 = r2_score(
    y_val,
    y_val_tuned
)

val_mae = mean_absolute_error(
    y_val,
    y_val_tuned
)

val_mse = mean_squared_error(
    y_val,
    y_val_tuned
)

val_rmse = np.sqrt(
    val_mse
)


print()
print("=" * 70)
print("TUNED MODEL — VALIDATION")
print("=" * 70)

print(
    f"Val R²   : {val_r2:.4f}"
)

print(
    f"Val MAE  : {val_mae:.2f}"
)

print(
    f"Val MSE  : {val_mse:.2f}"
)

print(
    f"Val RMSE : {val_rmse:.2f}"
)


# ============================================================
# 14. FINAL TEST EVALUATION
# ============================================================

y_test_pred = tuned_model.predict(
    X_test
)

test_r2 = r2_score(
    y_test,
    y_test_pred
)

test_mae = mean_absolute_error(
    y_test,
    y_test_pred
)

test_mse = mean_squared_error(
    y_test,
    y_test_pred
)

test_rmse = np.sqrt(
    test_mse
)


print()
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

print(
    f"Test R²   : {test_r2:.4f}"
)

print(
    f"Test MAE  : {test_mae:.2f}"
)

print(
    f"Test MSE  : {test_mse:.2f}"
)

print(
    f"Test RMSE : {test_rmse:.2f}"
)


# ============================================================
# 15. RETRAIN MODEL ON COMPLETE DATA
# ============================================================

print()
print(
    "Retraining final model on complete dataset..."
)

tuned_model.fit(
    df_model[all_features],
    df_model[target]
)

final_model = tuned_model

print("Retraining completed.")


# ============================================================
# 16. PREPARE ROUTES
# ============================================================

routes = (
    df[
        ["Origin", "Dest"]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)


# ============================================================
# 17. ROUTE MONTHLY STATISTICS
# ============================================================

route_monthly_stats = (
    df
    .groupby(
        ["Origin", "Dest", "Month"]
    )
    .agg(
        avg_freight_from=(
            "Freight From Origin",
            "mean"
        ),

        avg_mail_from=(
            "Mail From Origin",
            "mean"
        ),

        avg_pax_ratio=(
            "pax_ratio",
            "mean"
        ),

        avg_freight_ratio=(
            "freight_ratio",
            "mean"
        ),

        avg_mail_ratio=(
            "mail_ratio",
            "mean"
        ),

        avg_total_freight=(
            "total_freight",
            "mean"
        ),

        avg_pax=(
            "Pax From Origin",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# 18. HISTORICAL 2025 AND 2024 DATA
# ============================================================

pax_2025 = (
    df[df["Year"] == 2025]
    .set_index(
        ["Origin", "Dest", "Month"]
    )["Pax From Origin"]
    .to_dict()
)


pax_2024 = (
    df[df["Year"] == 2024]
    .set_index(
        ["Origin", "Dest", "Month"]
    )["Pax From Origin"]
    .to_dict()
)


# ============================================================
# 19. FUTURE PERIODS
# ============================================================

if 2025 in df["Year"].unique():

    months_in_2025 = sorted(
        df[
            df["Year"] == 2025
        ]["Month"].unique()
    )

else:
    months_in_2025 = []


months_to_predict_2025 = [
    m for m in range(1, 13)
    if m not in months_in_2025
]


future_periods = (
    [(2025, m)
     for m in months_to_predict_2025]
    +
    [(2026, m)
     for m in range(1, 13)]
)


print()
print("=" * 70)
print("FUTURE FORECAST")
print("=" * 70)

print(
    "2025 remaining months :",
    months_to_predict_2025
)

print(
    "2026 : January – December"
)


# ============================================================
# 20. MONTH NAMES
# ============================================================

month_names = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec"
}


# ============================================================
# 21. CHAINED FUTURE FORECASTING
# ============================================================

predicted_pool = {}

all_preds = []


for year, month in future_periods:

    month_rows = []

    for _, route in routes.iterrows():

        origin = route["Origin"]
        dest = route["Dest"]

        stats = route_monthly_stats[
            (route_monthly_stats["Origin"] == origin)
            &
            (route_monthly_stats["Dest"] == dest)
            &
            (route_monthly_stats["Month"] == month)
        ]

        if stats.empty:
            continue

        s = stats.iloc[0]


        # ----------------------------------------------------
        # lag_12
        # ----------------------------------------------------

        prev_year = year - 1

        if (
            (origin, dest, month) in pax_2025
            and prev_year == 2025
        ):

            lag_12 = pax_2025[
                (origin, dest, month)
            ]

        elif (
            (origin, dest, month) in pax_2024
            and prev_year == 2024
        ):

            lag_12 = pax_2024[
                (origin, dest, month)
            ]

        elif (
            origin,
            dest,
            prev_year,
            month
        ) in predicted_pool:

            lag_12 = predicted_pool[
                (
                    origin,
                    dest,
                    prev_year,
                    month
                )
            ]

        else:

            lag_12 = s["avg_pax"]


        # ----------------------------------------------------
        # lag_1
        # ----------------------------------------------------

        if month == 1:

            prev_m = 12
            prev_y = year - 1

        else:

            prev_m = month - 1
            prev_y = year


        if (
            (origin, dest, prev_m) in pax_2025
            and prev_y == 2025
        ):

            lag_1 = pax_2025[
                (origin, dest, prev_m)
            ]

        elif (
            origin,
            dest,
            prev_y,
            prev_m
        ) in predicted_pool:

            lag_1 = predicted_pool[
                (
                    origin,
                    dest,
                    prev_y,
                    prev_m
                )
            ]

        else:

            lag_1 = pax_2025.get(
                (origin, dest, prev_m),
                pax_2024.get(
                    (origin, dest, prev_m),
                    s["avg_pax"]
                )
            )


        # ----------------------------------------------------
        # YoY growth
        # ----------------------------------------------------

        if lag_12 > 0:

            yoy = (
                lag_1 - lag_12
            ) / lag_12

        else:

            yoy = 0.0


        yoy = float(
            np.clip(
                yoy,
                -1,
                5
            )
        )


        # ----------------------------------------------------
        # CREATE FUTURE ROW
        # ----------------------------------------------------

        month_rows.append({

            "Year": year,

            "Month": month,

            "Month Name":
                month_names[month],

            "Origin": origin,

            "Dest": dest,

            "Freight From Origin":
                s["avg_freight_from"],

            "Mail From Origin":
                s["avg_mail_from"],

            "pax_ratio":
                s["avg_pax_ratio"],

            "freight_ratio":
                s["avg_freight_ratio"],

            "mail_ratio":
                s["avg_mail_ratio"],

            "total_freight":
                s["avg_total_freight"],

            "month_sin":
                np.sin(
                    2 * np.pi * month / 12
                ),

            "month_cos":
                np.cos(
                    2 * np.pi * month / 12
                ),

            "lag_12_pax":
                lag_12,

            "lag_1_pax":
                lag_1,

            "yoy_growth":
                yoy
        })


    if not month_rows:
        continue


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    m_df = pd.DataFrame(
        month_rows
    )


    m_pred = final_model.predict(
        m_df[all_features]
    )


    # No negative passengers
    m_pred = np.clip(
        m_pred,
        0,
        None
    )


    # --------------------------------------------------------
    # STORE PREDICTIONS
    # --------------------------------------------------------

    for i, row in m_df.iterrows():

        predicted_pool[
            (
                row["Origin"],
                row["Dest"],
                year,
                month
            )
        ] = m_pred[i]


    m_df[
        "Predicted Pax From Origin"
    ] = m_pred.astype(int)


    all_preds.append(
        m_df
    )


    print(
        f"{month_names[month]} {year} "
        f"→ {len(m_df)} routes predicted"
    )


# ============================================================
# 22. COMBINE FORECAST RESULTS
# ============================================================

future_df = pd.concat(
    all_preds,
    ignore_index=True
)


future_df = future_df.sort_values(
    [
        "Origin",
        "Dest",
        "Year",
        "Month"
    ]
).reset_index(drop=True)


print()
print("=" * 70)
print("FORECAST SUMMARY")
print("=" * 70)

print(
    f"Total predictions : "
    f"{len(future_df)}"
)

print(
    f"Routes covered    : "
    f"{future_df[['Origin', 'Dest']].drop_duplicates().shape[0]}"
)


# ============================================================
# 23. SAMPLE FORECAST
# ============================================================

print()
print("--- Sample: HYDERABAD → DELHI (2026) ---")


sample = future_df[
    (future_df["Dest"] == "DELHI")
    &
    (future_df["Year"] == 2026)
][
    [
        "Month Name",
        "Predicted Pax From Origin"
    ]
].reset_index(drop=True)


if not sample.empty:
    print(
        sample.to_string(
            index=False
        )
    )
else:
    print(
        "No Hyderabad → Delhi "
        "forecast available."
    )


# ============================================================
# 24. SAVE FORECAST TO EXCEL
# ============================================================

OUTPUT_FILE = (
    "results/"
    "domestic_predictions_2025_2026.xlsx"
)


with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    # All route predictions
    out = future_df[
        [
            "Year",
            "Month",
            "Month Name",
            "Origin",
            "Dest",
            "Predicted Pax From Origin"
        ]
    ]

    out.to_excel(
        writer,
        sheet_name="All Routes",
        index=False
    )


    # Route-wise pivot table
    pivot = future_df.pivot_table(
        index=[
            "Origin",
            "Dest"
        ],

        columns="Month Name",

        values="Predicted Pax From Origin",

        aggfunc="sum"
    )


    ordered = [
        month_names[m]
        for m in range(1, 13)
        if month_names[m]
        in pivot.columns
    ]


    pivot = pivot[
        ordered
    ]


    pivot[
        "Annual Total"
    ] = pivot.sum(axis=1)


    pivot.reset_index().to_excel(
        writer,
        sheet_name="Pivot by Route",
        index=False
    )


print()
print(
    f"Saved: {OUTPUT_FILE}"
)


# ============================================================
# 25. SAVE MODEL
# ============================================================

MODEL_FILE = (
    "models/"
    "domestic_pax_model.joblib"
)


joblib.dump(
    final_model,
    MODEL_FILE
)


print(
    f"Model saved: {MODEL_FILE}"
)


# ============================================================
# 26. CREATE RESULTS DIRECTORY
# ============================================================

import os

os.makedirs(
    "results",
    exist_ok=True
)

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# 27. GRAPH 1 — MODEL COMPARISON
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)


rp = results_df.sort_values(
    "R2"
)


ax.barh(
    rp.index,
    rp["R2"]
)


for i, value in enumerate(
    rp["R2"]
):

    ax.text(
        value + 0.01,
        i,
        f"{value:.4f}",
        va="center"
    )


ax.set_xlabel(
    "R² Score"
)

ax.set_ylabel(
    "Model"
)

ax.set_title(
    "Model Comparison — Domestic Routes"
)

ax.set_xlim(
    0,
    1.05
)


plt.tight_layout()

plt.savefig(
    "results/domestic_model_comparison.png",
    dpi=150
)

plt.show()


# ============================================================
# 28. GRAPH 2 — ACTUAL VS PREDICTED
# ============================================================

val_delhi = val_df[
    val_df["Dest"] == "DELHI"
].copy()


if not val_delhi.empty:

    pred_val = final_model.predict(
        val_delhi[all_features]
    )


    fig, ax = plt.subplots(
        figsize=(14, 6)
    )


    ax.plot(
        range(len(val_delhi)),
        val_delhi[target].values,
        label="Actual",
        marker="o",
        markersize=4
    )


    ax.plot(
        range(len(val_delhi)),
        pred_val,
        label="Predicted",
        linestyle="--",
        marker="s",
        markersize=4
    )


    ax.set_title(
        "Actual vs Predicted — "
        "HYDERABAD → DELHI"
    )

    ax.set_xlabel(
        "Validation Samples"
    )

    ax.set_ylabel(
        "Passengers"
    )


    ax.legend()


    plt.tight_layout()


    plt.savefig(
        "results/domestic_actual_vs_predicted.png",
        dpi=150
    )


    plt.show()


# ============================================================
# 29. GRAPH 3 — VALIDATION VS TEST
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


metrics = [
    ("R² Score", val_r2, test_r2),
    ("MAE", val_mae, test_mae),
    ("RMSE", val_rmse, test_rmse)
]


for i, (
    label,
    val_value,
    test_value
) in enumerate(metrics):

    axes[i].bar(
        [
            "Validation",
            "Test"
        ],
        [
            val_value,
            test_value
        ]
    )


    axes[i].set_title(
        label
    )


    axes[i].text(
        0,
        val_value,
        f"{val_value:.2f}",
        ha="center",
        va="bottom"
    )


    axes[i].text(
        1,
        test_value,
        f"{test_value:.2f}",
        ha="center",
        va="bottom"
    )


plt.suptitle(
    "Validation vs Test Metrics — Tuned Domestic Model"
)


plt.tight_layout()


plt.savefig(
    "results/domestic_val_vs_test.png",
    dpi=150
)


plt.show()


# ============================================================
# 30. GRAPH 4 — 2026 FORECAST
# ============================================================

destinations = [
    "DELHI",
    "MUMBAI",
    "BENGALURU"
]


for dest in destinations:

    fore = future_df[
        (future_df["Dest"] == dest)
        &
        (future_df["Year"] == 2026)
    ].sort_values(
        "Month"
    )


    if fore.empty:
        continue


    # Historical last 24 months
    hist = (
        df[
            df["Dest"] == dest
        ]
        .groupby(
            ["Year", "Month"]
        )["Pax From Origin"]
        .sum()
        .reset_index()
        .sort_values(
            ["Year", "Month"]
        )
        .tail(24)
    )


    if hist.empty:
        continue


    fig, ax = plt.subplots(
        figsize=(14, 5)
    )


    # Historical data
    ax.plot(
        range(len(hist)),
        hist["Pax From Origin"].values,
        marker="o",
        markersize=3,
        label="Historical"
    )


    # Forecast
    forecast_x = range(
        len(hist) - 1,
        len(hist) + len(fore)
    )


    forecast_y = [
        hist["Pax From Origin"].values[-1]
    ] + list(
        fore[
            "Predicted Pax From Origin"
        ]
    )


    ax.plot(
        forecast_x,
        forecast_y,
        linestyle="--",
        marker="s",
        markersize=4,
        label="2026 Forecast"
    )


    # Forecast start
    ax.axvline(
        x=len(hist) - 1,
        linestyle=":"
    )


    ax.set_title(
        f"Passenger Forecast — "
        f"HYDERABAD → {dest}"
    )


    ax.set_xlabel(
        "Months"
    )


    ax.set_ylabel(
        "Passengers"
    )


    ax.legend()


    plt.tight_layout()


    plt.savefig(
        f"results/"
        f"domestic_forecast_"
        f"{dest.lower()}.png",
        dpi=150
    )


    plt.show()


# ============================================================
# 31. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("DOMESTIC FORECASTING COMPLETED")
print("=" * 70)

print(
    f"Train rows : {len(X_train)}"
)

print(
    f"Val rows   : {len(X_val)}"
)

print(
    f"Test rows  : {len(X_test)}"
)

print(
    f"Best model : {best_model_name}"
)

print(
    f"Test R²    : {test_r2:.4f}"
)

print(
    f"Predictions: {len(future_df)}"
)

print(
    "Routes     : "
    f"{future_df[['Origin', 'Dest']].drop_duplicates().shape[0]}"
)

print()
print("✓ Domestic forecasting completed successfully.")