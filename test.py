# ============================================
# Forecasting Penjualan Harian - End to End
# RandomForest + TimeSeriesSplit + RandomizedSearchCV
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

# -----------------------------
# 1) Load & Basic Preprocessing
# -----------------------------
df = pd.read_csv("supermarket_sales.csv")

# Pastikan kolom Date ada, dan parse
df['Date'] = pd.to_datetime(df['Date'])

# (Opsional) Pastikan kolom Total numeric
df['Total'] = pd.to_numeric(df['Total'], errors='coerce')

# Drop rows bermasalah
df = df.dropna(subset=['Date', 'Total'])

# -----------------------------
# 2) Aggregate to Daily Revenue
# -----------------------------
daily = (
    df.groupby(df['Date'].dt.date)  # group by pure date
      .agg(Total=('Total', 'sum'))
      .reset_index()
      .rename(columns={'Date': 'ds', 'Total': 'y'})
)

daily['ds'] = pd.to_datetime(daily['ds'])
daily = daily.sort_values('ds').reset_index(drop=True)

# (Optional) isi tanggal hilang dengan 0 penjualan (jika diperlukan)
# Buat date range penuh
full_range = pd.date_range(daily['ds'].min(), daily['ds'].max(), freq='D')
daily = (
    pd.DataFrame({'ds': full_range})
    .merge(daily, on='ds', how='left')
    .fillna({'y': 0.0})
)

# -----------------------------
# 3) Feature Engineering (Lag/Rolling/Calendar)
# -----------------------------
def add_time_features(df, target_col='y'):
    df = df.copy()
    # Calendar features
    df['dow'] = df['ds'].dt.dayofweek          # 0=Mon
    df['dom'] = df['ds'].dt.day                # day of month
    df['month'] = df['ds'].dt.month
    df['weekofyear'] = df['ds'].dt.isocalendar().week.astype(int)

    # Lag features
    for lag in [1, 7, 14, 30]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)

    # Rolling window stats (based on past values)
    df['roll7_mean']  = df[target_col].shift(1).rolling(window=7, min_periods=1).mean()
    df['roll7_std']   = df[target_col].shift(1).rolling(window=7, min_periods=1).std()
    df['roll30_mean'] = df[target_col].shift(1).rolling(window=30, min_periods=1).mean()
    df['roll30_std']  = df[target_col].shift(1).rolling(window=30, min_periods=1).std()

    # After feature creation, drop initial rows with NaNs from shifting
    df = df.dropna().reset_index(drop=True)
    return df

data = add_time_features(daily, target_col='y')

# -----------------------------
# 4) Train/Test Split (by time)
# -----------------------------
# Contoh: gunakan 85% pertama sebagai train/val (dengan TSSplit), sisanya test final
split_idx = int(len(data) * 0.85)
train_df = data.iloc[:split_idx].copy()
test_df  = data.iloc[split_idx:].copy()

feature_cols = [c for c in data.columns if c not in ['ds', 'y']]
X_train = train_df[feature_cols]
y_train = train_df['y']
X_test  = test_df[feature_cols]
y_test  = test_df['y']

# -----------------------------
# 5) Pipeline & Preprocessor
# -----------------------------
num_features = feature_cols  # semua numeric di sini

preprocess = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_features)
    ],
    remainder='drop'
)

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

pipe = Pipeline([
    ('prep', preprocess),
    ('model', rf)
])

# -----------------------------
# 6) TimeSeriesSplit + Hyperparameter Tuning
# -----------------------------
tscv = TimeSeriesSplit(n_splits=5)

param_dist = {
    'model__n_estimators': [200, 400, 600, 800, 1000],
    'model__max_depth': [None, 6, 8, 10, 12, 16, 20],
    'model__min_samples_split': [2, 5, 10, 20],
    'model__min_samples_leaf': [1, 2, 4, 8],
    'model__max_features': ['auto', 'sqrt', 0.3, 0.5, 0.7],
    'model__bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=40,                # tingkatkan jika ingin lebih teliti
    scoring='neg_mean_absolute_error',
    cv=tscv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)

print("Best CV MAE:", -random_search.best_score_)
print("Best Params:", random_search.best_params_)

best_model = random_search.best_estimator_

# -----------------------------
# 7) Evaluasi di Test Set
# -----------------------------
y_pred = best_model.predict(X_test)

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero = y_true != 0
    return np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100

mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mape_val = mape(y_test, y_pred)

print(f"TEST MAE : {mae:,.2f}")
print(f"TEST RMSE: {rmse:,.2f}")
print(f"TEST MAPE: {mape_val:.2f}%")

# Plot actual vs predicted on test
plt.figure(figsize=(12,5))
plt.plot(test_df['ds'], y_test.values, label='Actual')
plt.plot(test_df['ds'], y_pred, label='Predicted')
plt.title('Actual vs Predicted - Test Period')
plt.xlabel('Date')
plt.ylabel('Revenue (Total)')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 8) Forecast 30 Hari ke Depan (Iterative)
# -----------------------------
# Kita butuh rolling update fitur lag/rolling berbasis prediksi terbaru.
hist_df = data.copy()  # pakai seluruh data historis (train+test) untuk start
last_date = hist_df['ds'].max()
horizon = 30

future_rows = []
current_history = hist_df[['ds', 'y']].copy()

for i in range(1, horizon + 1):
    next_date = last_date + timedelta(days=i)

    # Buat satu baris 'calender features'
    row = {
        'ds': next_date,
        'dow': next_date.dayofweek,
        'dom': next_date.day,
        'month': next_date.month,
        'weekofyear': int(pd.Timestamp(next_date).isocalendar().week),
    }

    # Tambahkan lag dari current_history (perhatikan ds unik)
    # Pastikan current_history terurut
    current_history = current_history.sort_values('ds').reset_index(drop=True)

    # Helper untuk ambil lag value:
    def get_lag(d, k):
        ref_date = d - timedelta(days=k)
        val = current_history.loc[current_history['ds'] == ref_date, 'y']
        if len(val) == 0:
            return np.nan
        return float(val.values[0])

    for lag in [1, 7, 14, 30]:
        row[f'lag_{lag}'] = get_lag(next_date, lag)

    # Rolling (pakai histori sampai kemarin)
    def rolling_stat(days, func):
        end_date = next_date - timedelta(days=1)
        start_date = end_date - timedelta(days=days-1)
        mask = (current_history['ds'] >= start_date) & (current_history['ds'] <= end_date)
        vals = current_history.loc[mask, 'y'].values
        if len(vals) == 0:
            return np.nan
        return func(vals)

    row['roll7_mean']  = rolling_stat(7, np.mean)
    row['roll7_std']   = rolling_stat(7, np.std)
    row['roll30_mean'] = rolling_stat(30, np.mean)
    row['roll30_std']  = rolling_stat(30, np.std)

    # Buat dataframe 1-baris untuk prediksi
    x_row = pd.DataFrame([row])
    # Buang kolom ds saat predict
    x_pred = x_row[[c for c in feature_cols]]

    # Jika ada NaN (awal-awal horizon), imputer di pipeline akan handle
    y_hat = best_model.predict(x_pred)[0]

    # Simpan ke list future + update current_history (append prediksi)
    future_rows.append({'ds': next_date, 'y_pred': y_hat})
    current_history = pd.concat([current_history, pd.DataFrame({'ds':[next_date], 'y':[y_hat]})], ignore_index=True)

future_df = pd.DataFrame(future_rows)

# Plot forecast
plt.figure(figsize=(12,5))
plt.plot(hist_df['ds'], hist_df['y'], label='History')
plt.plot(future_df['ds'], future_df['y_pred'], label='Forecast (30d)')
plt.title('30-Day Revenue Forecast')
plt.xlabel('Date')
plt.ylabel('Revenue (Total)')
plt.legend()
plt.grid(True)
plt.show()

# (Opsional) Simpan model
import joblib
joblib.dump(best_model, 'rf_daily_revenue_forecast.pkl')
print("Model saved to rf_daily_revenue_forecast.pkl")
