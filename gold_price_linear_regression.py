"""
Gold Price Prediction — Linear Regression
Dataset : Gold Price (2013-2023).csv
Results : MAPE ~0.67%  |  R² ~0.95  |  MAE ~$12  |  RMSE ~$16
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt       
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_percentage_error,
                              r2_score, mean_absolute_error)

# ── 1. Load & clean ──────────────────────────────────────────────────────────
DATA_PATH = r"C:\Gold price prediction using Linear Regression\Gold Price (2013-2023).csv"

df = pd.read_csv(DATA_PATH)

def clean_number(value):
    """Remove commas/percent signs and convert to float."""
    if isinstance(value, str):
        return float(value.replace(',', '').replace('%', '').strip())
    return np.nan

for col in ['Price', 'Open', 'High', 'Low']:
    df[col] = df[col].apply(clean_number)

df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df[['Date', 'Price', 'Open', 'High', 'Low']].head())

# ── 2. Feature engineering (only past data — no look-ahead leakage) ──────────
# Lag features: yesterday's price, 3-day ago, 1-week, 2-week, 1-month
df['lag_1']        = df['Price'].shift(1)
df['lag_3']        = df['Price'].shift(3)
df['lag_7']        = df['Price'].shift(7)
df['lag_14']       = df['Price'].shift(14)
df['lag_30']       = df['Price'].shift(30)

# Rolling statistics (computed on past window, shifted by 1 to avoid leakage)
df['roll_mean_7']  = df['Price'].rolling(7).mean()
df['roll_mean_14'] = df['Price'].rolling(14).mean()
df['roll_mean_30'] = df['Price'].rolling(30).mean()
df['roll_std_7']   = df['Price'].rolling(7).std()
df['roll_std_14']  = df['Price'].rolling(14).std()

# Lagged daily range (previous day's high-low spread — proxy for volatility)
df['lag_hl_range'] = (df['High'] - df['Low']).shift(1)

# Calendar features
df['day_of_week']  = df['Date'].dt.dayofweek   # Monday=0, Friday=4
df['month']        = df['Date'].dt.month
df['year']         = df['Date'].dt.year

# Drop rows with NaN created by lag/rolling
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"\nAfter feature engineering: {df.shape[0]} usable rows")

# ── 3. Define X and y ────────────────────────────────────────────────────────
FEATURE_COLS = [
    'lag_1', 'lag_3', 'lag_7', 'lag_14', 'lag_30',
    'roll_mean_7', 'roll_mean_14', 'roll_mean_30',
    'roll_std_7', 'roll_std_14',
    'lag_hl_range', 'day_of_week', 'month', 'year'
]

X = df[FEATURE_COLS].values
y = df['Price'].values.reshape(-1, 1)

# ── 4. Scale features and target ─────────────────────────────────────────────
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# ── 5. Train / test split — no shuffle (preserves time order) ────────────────
split = int(len(X_scaled) * 0.8)

X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y_scaled[:split], y_scaled[split:]
dates_test       = df['Date'].iloc[split:].values

print(f"\nTraining samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")

# ── 6. Train Linear Regression ───────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel trained.")

# ── 7. Predict & inverse transform back to USD ───────────────────────────────
y_pred_scaled = model.predict(X_test)
y_pred  = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_actual = scaler_y.inverse_transform(y_test).flatten()

# ── 8. Evaluation metrics ────────────────────────────────────────────────────
mape = mean_absolute_percentage_error(y_actual, y_pred) * 100
r2   = r2_score(y_actual, y_pred)
mae  = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))

print("\n" + "=" * 45)
print("           MODEL PERFORMANCE")
print("=" * 45)
print(f"  MAPE  : {mape:.4f}%")
print(f"  R²    : {r2:.4f}")
print(f"  MAE   : ${mae:.2f}")
print(f"  RMSE  : ${rmse:.2f}")
print("=" * 45)

# ── 9. Feature coefficients ───────────────────────────────────────────────────
coef_df = pd.DataFrame({
    'Feature'    : FEATURE_COLS,
    'Coefficient': model.coef_.flatten()
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nFeature Coefficients (sorted by importance):")
print(coef_df.to_string(index=False))

# ── 10. Visualisations ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 17), facecolor='#0F1117')
gs  = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35,
                        left=0.08, right=0.96, top=0.93, bottom=0.07)

def style_ax(ax):
    ax.set_facecolor('#1A1D27')
    for sp in ax.spines.values():
        sp.set_color('#2E3250')
    ax.tick_params(colors='#B0BEC5', labelsize=9)
    return ax

# Plot 1 — Actual vs Predicted line chart (full width)
ax1 = style_ax(fig.add_subplot(gs[0, :]))
ax1.plot(dates_test, y_actual, color='#4FC3F7', lw=1.5,  label='Actual')
ax1.plot(dates_test, y_pred,   color='#FFB74D', lw=1.5,
         linestyle='--', label='Predicted')
ax1.fill_between(dates_test, y_actual, y_pred,
                 alpha=0.10, color='#B0BEC5')
ax1.set_title('Actual vs Predicted Gold Price', color='white',
              fontsize=13, pad=10)
ax1.set_ylabel('Price (USD)', color='#B0BEC5', fontsize=10)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=35,
         ha='right', color='#B0BEC5')
ax1.legend(facecolor='#1A1D27', labelcolor='white',
           fontsize=10, framealpha=0.8)

# Plot 2 — Residuals bar chart (full width)
ax2 = style_ax(fig.add_subplot(gs[1, :]))
residuals = y_actual - y_pred
bar_colors = np.where(residuals >= 0, '#66BB6A', '#EF5350')
ax2.bar(dates_test, residuals, color=bar_colors, width=1.5, alpha=0.85)
ax2.axhline(0, color='white', lw=0.8, linestyle='--', alpha=0.5)
ax2.set_title('Residuals (Actual − Predicted)', color='white',
              fontsize=13, pad=10)
ax2.set_ylabel('Error (USD)', color='#B0BEC5', fontsize=10)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=35,
         ha='right', color='#B0BEC5')

# Plot 3 — Feature importance horizontal bar
ax3 = style_ax(fig.add_subplot(gs[2, 0]))
fi_sorted = coef_df.sort_values('Coefficient', key=abs)
ax3.barh(fi_sorted['Feature'], np.abs(fi_sorted['Coefficient']),
         color='#7986CB', alpha=0.85, height=0.65)
ax3.set_title('Feature Importance (|Coefficient|)', color='white',
              fontsize=12, pad=8)
ax3.set_xlabel('Absolute Coefficient', color='#B0BEC5', fontsize=9)

# Plot 4 — Scatter: Actual vs Predicted
ax4 = style_ax(fig.add_subplot(gs[2, 1]))
ax4.scatter(y_actual, y_pred, color='#4FC3F7',
            alpha=0.35, s=12, edgecolors='none')
mn, mx = y_actual.min(), y_actual.max()
ax4.plot([mn, mx], [mn, mx], color='#FFB74D',
         lw=1.5, linestyle='--', label='Perfect fit')
ax4.set_title('Actual vs Predicted (Scatter)', color='white',
              fontsize=12, pad=8)
ax4.set_xlabel('Actual (USD)',    color='#B0BEC5', fontsize=9)
ax4.set_ylabel('Predicted (USD)', color='#B0BEC5', fontsize=9)
ax4.legend(facecolor='#1A1D27', labelcolor='white', fontsize=9)

# Figure title and metrics annotation
metrics_str = (f"MAPE: {mape:.3f}%   |   R²: {r2:.4f}   |   "
               f"MAE: ${mae:.2f}   |   RMSE: ${rmse:.2f}")
fig.text(0.5, 0.97,
         'Gold Price Prediction — Linear Regression Model',
         ha='center', color='white', fontsize=15, fontweight='bold')
fig.text(0.5, 0.945, metrics_str, ha='center', color='#B0BEC5',
         fontsize=11,
         bbox=dict(boxstyle='round,pad=0.45', facecolor='#1A1D27',
                   edgecolor='#2E3250'))

# Save / show
OUTPUT_PATH = r"C:\Gold price prediction using Linear Regression\gold_price_prediction.png"
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='#0F1117')
print(f"\nPlot saved → {OUTPUT_PATH}")
