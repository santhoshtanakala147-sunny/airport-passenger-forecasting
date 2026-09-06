import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')


# =============================================================
# 1. LOAD DATA
# =============================================================

df = pd.read_excel('international (1).xlsx')

df.drop(columns=['Source.Name'], errors='ignore', inplace=True)

num_cols = ['Pax From Origin', 'Pax To Origin',
            'Freight From Origin', 'Frieght To Origin']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df = df.sort_values(['Year', 'Quarter']).reset_index(drop=True)

print(f"Dataset shape  : {df.shape}")
print(f"Years covered  : {sorted(df['Year'].unique())}")
print(f"Unique routes  : {df[['Origin','Dest']].drop_duplicates().shape[0]}")
print(f"Destinations   : {sorted(df['Dest'].unique())}")
print(f"2025 quarters  : {sorted(df[df['Year']==2025]['Quarter'].unique())}")
print(f"Zero-pax rows  : {(df['Pax From Origin']==0).sum()}")


# =============================================================
# 2. FEATURE ENGINEERING
# =============================================================

df['pax_ratio'] = (
    df['Pax To Origin'] / df['Pax From Origin'].replace(0, np.nan)
).fillna(1.0).clip(0, 10)

df['freight_ratio'] = (
    df['Frieght To Origin'] / df['Freight From Origin'].replace(0, np.nan)
).fillna(1.0).clip(0, 20)

df['total_freight'] = df['Freight From Origin'] + df['Frieght To Origin']

df['quarter_sin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
df['quarter_cos'] = np.cos(2 * np.pi * df['Quarter'] / 4)

# COVID flag: 2020 Q2 onward through 2021 Q4
df['is_covid'] = (
    ((df['Year'] == 2020) & (df['Quarter'] >= 2)) |
    (df['Year'] == 2021)
).astype(int)

df = df.sort_values(['Origin', 'Dest', 'Year', 'Quarter']).reset_index(drop=True)

df['lag_4_pax']     = df.groupby(['Origin', 'Dest'])['Pax From Origin'].shift(4)
df['lag_1_pax']     = df.groupby(['Origin', 'Dest'])['Pax From Origin'].shift(1)
df['lag_4_freight'] = df.groupby(['Origin', 'Dest'])['Freight From Origin'].shift(4)

df['yoy_growth'] = (
    (df['Pax From Origin'] - df['lag_4_pax']) /
    df['lag_4_pax'].replace(0, np.nan)
).fillna(0).clip(-1, 5)

df = df.sort_values(['Year', 'Quarter']).reset_index(drop=True)


# =============================================================
# 3. DEFINE FEATURES & TARGET
# =============================================================

target       = 'Pax From Origin'
cat_features = ['Origin', 'Dest']
num_features = [
    'Year', 'Quarter',
    'Freight From Origin',
    'pax_ratio', 'freight_ratio',
    'total_freight',
    'quarter_sin', 'quarter_cos',
    'is_covid',
    'lag_4_pax', 'lag_1_pax', 'lag_4_freight',
    'yoy_growth',
]
all_features = num_features + cat_features

# Drop rows with NaN lags (first year per route)
df_model = df.dropna(subset=['lag_4_pax', 'lag_1_pax']).reset_index(drop=True)

print(f"\nRows after dropping lag NaNs : {df_model.shape[0]}")
print(f"Features used                : {len(all_features)}")


# =============================================================
# 4. TRAIN / VALIDATION / TEST SPLIT  (70% / 15% / 15%)
#    Strictly time-based — no shuffling
# =============================================================

n         = len(df_model)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

train_df = df_model.iloc[:train_end]
val_df   = df_model.iloc[train_end:val_end]
test_df  = df_model.iloc[val_end:]

X_train, y_train = train_df[all_features], train_df[target]
X_val,   y_val   = val_df[all_features],   val_df[target]
X_test,  y_test  = test_df[all_features],  test_df[target]

print(f"\nTrain : {len(X_train)} rows  "
      f"({train_df['Year'].min()} Q{train_df['Quarter'].min()} – "
      f"{train_df['Year'].max()} Q{train_df['Quarter'].max()})")
print(f"Val   : {len(X_val)}  rows  "
      f"({val_df['Year'].min()} Q{val_df['Quarter'].min()} – "
      f"{val_df['Year'].max()} Q{val_df['Quarter'].max()})")
print(f"Test  : {len(X_test)}  rows  "
      f"({test_df['Year'].min()} Q{test_df['Quarter'].min()} – "
      f"{test_df['Year'].max()} Q{test_df['Quarter'].max()})")


# =============================================================
# 5. PREPROCESSOR
# =============================================================

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
])


# =============================================================
# 6. MODEL COMPARISON  (evaluated on VALIDATION set)
#    Test set is kept completely untouched until Section 8
# =============================================================

tscv = TimeSeriesSplit(n_splits=5)

models = {
    'Ridge':             Ridge(),
    'Random Forest':     RandomForestRegressor(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
    'XGBoost':           XGBRegressor(n_estimators=200, random_state=42, verbosity=0),
    'LightGBM':          LGBMRegressor(n_estimators=200, random_state=42, verbose=-1),
}

results       = []
trained_pipes = {}

print("\n" + "="*70)
print(f"{'Model':<22} {'Val R2':>8} {'Val MAE':>10} {'Val RMSE':>10} {'Val MSE':>14}")
print("="*70)

for name, model in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # Fit on TRAIN only
    pipe.fit(X_train, y_train)

    # Evaluate on VALIDATION set
    y_val_pred = pipe.predict(X_val)

    r2   = r2_score(y_val, y_val_pred)
    mae  = mean_absolute_error(y_val, y_val_pred)
    mse  = mean_squared_error(y_val, y_val_pred)
    rmse = mse ** 0.5

    results.append({'Model': name, 'Val R2': r2, 'Val MAE': mae,
                    'Val MSE': mse, 'Val RMSE': rmse})
    trained_pipes[name] = pipe

    print(f"{name:<22} {r2:>8.4f} {mae:>10.1f} {rmse:>10.1f} {mse:>14.1f}")

print("="*70)

results_df      = pd.DataFrame(results).sort_values('Val R2', ascending=False)
best_model_name = results_df.iloc[0]['Model']
print(f"\nBest model (by Val R²): {best_model_name}  "
      f"(Val R² = {results_df.iloc[0]['Val R2']:.4f})")


# =============================================================
# 7. HYPERPARAMETER TUNING  (GridSearchCV on TRAIN only)
# =============================================================

param_grids = {
    'Ridge':             {'regressor__alpha': [0.1, 1, 10, 100]},
    'Random Forest':     {'regressor__n_estimators': [100, 200],
                          'regressor__max_depth': [None, 10, 20]},
    'Gradient Boosting': {'regressor__n_estimators': [100, 200],
                          'regressor__learning_rate': [0.05, 0.1]},
    'XGBoost':           {'regressor__n_estimators': [100, 200],
                          'regressor__learning_rate': [0.05, 0.1],
                          'regressor__max_depth': [4, 6]},
    'LightGBM':          {'regressor__n_estimators': [100, 200],
                          'regressor__learning_rate': [0.05, 0.1],
                          'regressor__num_leaves': [31, 63]},
}

print(f"\nTuning {best_model_name} with GridSearchCV (fitted on Train only)...")

grid_search = GridSearchCV(
    trained_pipes[best_model_name],
    param_grids[best_model_name],
    cv=tscv, scoring='r2', n_jobs=-1
)
grid_search.fit(X_train, y_train)   # ← TRAIN only

tuned_model = grid_search.best_estimator_
print(f"Best params : {grid_search.best_params_}")

# Validation metrics after tuning
y_val_tuned = tuned_model.predict(X_val)
print(f"\n--- Tuned Model — Validation Set ---")
print(f"Val R²   : {r2_score(y_val, y_val_tuned):.4f}")
print(f"Val MAE  : {mean_absolute_error(y_val, y_val_tuned):.2f}")
print(f"Val MSE  : {mean_squared_error(y_val, y_val_tuned):.2f}")
print(f"Val RMSE : {mean_squared_error(y_val, y_val_tuned)**0.5:.2f}")


# =============================================================
# 8. FINAL EVALUATION ON TEST SET  (done only ONCE)
# =============================================================

y_test_pred = tuned_model.predict(X_test)

print(f"\n--- Final Evaluation — Test Set (unseen) ---")
print(f"Test R²   : {r2_score(y_test, y_test_pred):.4f}")
print(f"Test MAE  : {mean_absolute_error(y_test, y_test_pred):.2f}")
print(f"Test MSE  : {mean_squared_error(y_test, y_test_pred):.2f}")
print(f"Test RMSE : {mean_squared_error(y_test, y_test_pred)**0.5:.2f}")


# =============================================================
# 9. RETRAIN ON FULL DATA (train+val+test) FOR FORECASTING
# =============================================================

print("\nRetraining on full dataset for future prediction...")
tuned_model.fit(df_model[all_features], df_model[target])
final_model = tuned_model
print("Done.")


# =============================================================
# 10. FUTURE PREDICTION — 2025 Q4 + ALL 2026
#     Chained lag strategy (quarter by quarter per route)
# =============================================================

routes = df[['Origin', 'Dest']].drop_duplicates().reset_index(drop=True)

route_quarterly_stats = df.groupby(['Origin', 'Dest', 'Quarter']).agg(
    avg_freight_from  = ('Freight From Origin', 'mean'),
    avg_pax_ratio     = ('pax_ratio',           'mean'),
    avg_freight_ratio = ('freight_ratio',        'mean'),
    avg_total_freight = ('total_freight',        'mean'),
    avg_pax           = ('Pax From Origin',      'mean'),
    avg_freight4      = ('lag_4_freight',        'mean'),
).reset_index()

pax_2025 = (
    df[df['Year'] == 2025]
    .set_index(['Origin', 'Dest', 'Quarter'])['Pax From Origin']
    .to_dict()
)
pax_2024 = (
    df[df['Year'] == 2024]
    .set_index(['Origin', 'Dest', 'Quarter'])['Pax From Origin']
    .to_dict()
)

quarters_in_2025         = sorted(df[df['Year'] == 2025]['Quarter'].unique())
quarters_to_predict_2025 = [q for q in range(1, 5) if q not in quarters_in_2025]

future_periods = (
    [(2025, q) for q in quarters_to_predict_2025] +
    [(2026, q) for q in range(1, 5)]
)

print(f"\nPredicting: 2025 Q{quarters_to_predict_2025} + all 4 quarters of 2026")

quarter_names  = {1:'Q1 (Jan-Mar)', 2:'Q2 (Apr-Jun)',
                  3:'Q3 (Jul-Sep)', 4:'Q4 (Oct-Dec)'}
predicted_pool = {}
all_preds      = []

for year, quarter in future_periods:
    quarter_rows = []

    for _, route in routes.iterrows():
        origin, dest = route['Origin'], route['Dest']

        stats = route_quarterly_stats[
            (route_quarterly_stats['Origin']  == origin) &
            (route_quarterly_stats['Dest']    == dest)   &
            (route_quarterly_stats['Quarter'] == quarter)
        ]
        if stats.empty:
            continue
        s = stats.iloc[0]

        # lag_4: same quarter previous year
        prev_year = year - 1
        if (origin, dest, quarter) in pax_2025 and prev_year == 2025:
            lag_4 = pax_2025[(origin, dest, quarter)]
        elif (origin, dest, quarter) in pax_2024 and prev_year == 2024:
            lag_4 = pax_2024[(origin, dest, quarter)]
        elif (origin, dest, prev_year, quarter) in predicted_pool:
            lag_4 = predicted_pool[(origin, dest, prev_year, quarter)]
        else:
            lag_4 = s['avg_pax']

        # lag_1: previous quarter (chained)
        if quarter == 1:
            prev_q, prev_y = 4, year - 1
        else:
            prev_q, prev_y = quarter - 1, year

        if (origin, dest, prev_q) in pax_2025 and prev_y == 2025:
            lag_1 = pax_2025[(origin, dest, prev_q)]
        elif (origin, dest, prev_y, prev_q) in predicted_pool:
            lag_1 = predicted_pool[(origin, dest, prev_y, prev_q)]
        else:
            lag_1 = pax_2025.get((origin, dest, prev_q),
                    pax_2024.get((origin, dest, prev_q), s['avg_pax']))

        yoy = float(np.clip((lag_1 - lag_4) / lag_4 if lag_4 > 0 else 0.0, -1, 5))

        f4 = s['avg_freight4']
        if isinstance(f4, float) and np.isnan(f4):
            f4 = s['avg_freight_from']

        quarter_rows.append({
            'Year':                  year,
            'Quarter':               quarter,
            'Quarter Label':         quarter_names[quarter],
            'Origin':                origin,
            'Dest':                  dest,
            'Freight From Origin':   s['avg_freight_from'],
            'pax_ratio':             s['avg_pax_ratio'],
            'freight_ratio':         s['avg_freight_ratio'],
            'total_freight':         s['avg_total_freight'],
            'quarter_sin':           np.sin(2 * np.pi * quarter / 4),
            'quarter_cos':           np.cos(2 * np.pi * quarter / 4),
            'is_covid':              0,
            'lag_4_pax':             lag_4,
            'lag_1_pax':             lag_1,
            'lag_4_freight':         f4,
            'yoy_growth':            yoy,
        })

    if not quarter_rows:
        continue

    q_df   = pd.DataFrame(quarter_rows)
    q_pred = final_model.predict(q_df[all_features]).clip(0)

    for i, row in q_df.iterrows():
        predicted_pool[(row['Origin'], row['Dest'], year, quarter)] = q_pred[i]

    q_df['Predicted Pax From Origin'] = q_pred.astype(int)
    all_preds.append(q_df)
    print(f"  {year} {quarter_names[quarter]} → {len(q_df)} routes predicted")

future_df = pd.concat(all_preds, ignore_index=True)
future_df = future_df.sort_values(['Origin', 'Dest', 'Year', 'Quarter']).reset_index(drop=True)

print(f"\nTotal predictions : {len(future_df)}")
print(f"Routes covered    : {future_df[['Origin','Dest']].drop_duplicates().shape[0]}")

print("\n--- Sample: HYDERABAD → DUBAI (2026) ---")
sample = future_df[
    (future_df['Dest'] == 'DUBAI') & (future_df['Year'] == 2026)
][['Quarter Label', 'Predicted Pax From Origin']].reset_index(drop=True)
print(sample.to_string(index=False))


# =============================================================
# 11. SAVE OUTPUTS
# =============================================================

with pd.ExcelWriter('international_predictions_2025_2026.xlsx', engine='openpyxl') as writer:

    out = future_df[['Year', 'Quarter', 'Quarter Label', 'Origin', 'Dest',
                      'Predicted Pax From Origin']]
    out.to_excel(writer, sheet_name='All Routes', index=False)

    future_df['Period'] = future_df['Year'].astype(str) + ' ' + future_df['Quarter Label']
    pivot = future_df.pivot_table(
        index=['Origin', 'Dest'],
        columns='Period',
        values='Predicted Pax From Origin',
        aggfunc='sum'
    )
    annual = future_df[future_df['Year'] == 2026].groupby(
        ['Origin', 'Dest'])['Predicted Pax From Origin'].sum()
    pivot['Annual 2026 Total'] = annual
    pivot.reset_index().to_excel(writer, sheet_name='Pivot by Route', index=False)

print("\nSaved: international_predictions_2025_2026.xlsx")

joblib.dump(final_model, 'international_pax_model.joblib')
print("Model saved: international_pax_model.joblib")


# =============================================================
# 12. PLOTS
# =============================================================

# Plot 1: Model comparison (Val R²) + Actual vs Predicted (Val set)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

rp = results_df.sort_values('Val R2')
axes[0].barh(rp['Model'], rp['Val R2'], color='steelblue')
for i, (_, row) in enumerate(rp.iterrows()):
    axes[0].text(row['Val R2'] + 0.003, i, f"{row['Val R2']:.4f}", va='center', fontsize=9)
axes[0].set_xlabel("R² Score (Validation Set)")
axes[0].set_title("Model Comparison — International Routes")
axes[0].set_xlim(0, 1.1)

val_dubai = val_df[val_df['Dest'] == 'DUBAI'].copy()
if not val_dubai.empty:
    pred_val = final_model.predict(val_dubai[all_features])
    axes[1].plot(range(len(val_dubai)), val_dubai[target].values,
                 label='Actual', color='steelblue', marker='o', markersize=5)
    axes[1].plot(range(len(val_dubai)), pred_val,
                 label='Predicted', color='orange', linestyle='--', marker='s', markersize=5)
    axes[1].set_title("Actual vs Predicted — HYDERABAD → DUBAI (Val Set)")
    axes[1].set_xlabel("Validation samples (quarters)")
    axes[1].set_ylabel("Passengers")
    axes[1].legend()

plt.tight_layout()
plt.savefig('international_model_evaluation.png', dpi=150)
plt.show()

# Plot 2: Validation vs Test metrics side by side
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))
metric_labels = ['R² Score', 'MAE (passengers)', 'RMSE (passengers)']
val_vals  = [r2_score(y_val, y_val_tuned),
             mean_absolute_error(y_val, y_val_tuned),
             mean_squared_error(y_val, y_val_tuned)**0.5]
test_vals = [r2_score(y_test, y_test_pred),
             mean_absolute_error(y_test, y_test_pred),
             mean_squared_error(y_test, y_test_pred)**0.5]

for i, (ml, vv, tv) in enumerate(zip(metric_labels, val_vals, test_vals)):
    axes2[i].bar(['Validation', 'Test'], [vv, tv], color=['steelblue', 'coral'])
    axes2[i].set_title(ml)
    for j, v in enumerate([vv, tv]):
        axes2[i].text(j, v * 1.01, f"{v:.2f}", ha='center', fontsize=10)
    axes2[i].set_ylim(0, max(vv, tv) * 1.2)

plt.suptitle(f"Validation vs Test Metrics — {best_model_name} (Tuned)", fontsize=13)
plt.tight_layout()
plt.savefig('international_val_vs_test.png', dpi=150)
plt.show()

# Plot 3: Historical + forecast for top route
for dest in ['DUBAI', 'SINGAPORE', 'DOHA']:
    fore = future_df[(future_df['Dest'] == dest) & (future_df['Year'] == 2026)].sort_values('Quarter')
    if fore.empty:
        continue
    hist = (df[df['Dest'] == dest]
            .groupby(['Year','Quarter'])['Pax From Origin'].sum()
            .reset_index().sort_values(['Year','Quarter']).tail(20))

    fig3, ax3 = plt.subplots(figsize=(14, 5))
    ax3.plot(range(len(hist)), hist['Pax From Origin'].values,
             color='steelblue', marker='o', markersize=4, label='Historical (last 5 yrs)')
    ax3.plot(range(len(hist)-1, len(hist)+len(fore)),
             [hist['Pax From Origin'].values[-1]] + list(fore['Predicted Pax From Origin']),
             color='orange', linestyle='--', marker='s', markersize=5, label='2026 Forecast')
    ax3.axvline(x=len(hist)-1, color='red', linestyle=':', alpha=0.6, label='Forecast start')
    ax3.set_title(f"Passenger Forecast — HYDERABAD → {dest}")
    ax3.set_ylabel("Passengers")
    ax3.set_xlabel("Quarters")
    ax3.legend()
    plt.tight_layout()
    plt.savefig(f'international_forecast_{dest.lower()}.png', dpi=150)
    plt.show()
    break

print("\n✓ Done!")
print(f"  Train rows : {len(X_train)}")
print(f"  Val rows   : {len(X_val)}")
print(f"  Test rows  : {len(X_test)}")
print(f"  Predictions: {len(future_df)} ({future_df[['Origin','Dest']].drop_duplicates().shape[0]} routes)")
