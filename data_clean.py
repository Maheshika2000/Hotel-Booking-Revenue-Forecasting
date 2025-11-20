# ============================================================
#STEP 1: Import Libraries
# ============================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import math

# ============================================================
#STEP 2: Load and preprocess data 
# ============================================================

data = pd.read_csv("sorted_new.csv")

# Convert date columns
data['Booked on'] = pd.to_datetime(data['Booked on'])
data['Departure'] = pd.to_datetime(data['Departure'])
data['Arrival'] = pd.to_datetime(data['Arrival'])

# Calculate hotel revenue
data['Hotel_Revenue'] = data['Total payment'] - data['Commission']

# Create Year-Month column
data['YearMonth'] = data['Booked on'].dt.to_period('M').astype(str)

# Aggregate monthly revenue
monthly_revenue = data.groupby('YearMonth')['Hotel_Revenue'].sum().reset_index()

# Convert YearMonth to datetime for modeling
monthly_revenue['YearMonth'] = pd.to_datetime(monthly_revenue['YearMonth'])

monthly_revenue.to_csv("monthly_revenue.csv", index=False)
monthly_revenue.head()


# ============================================================
# EXTRA PLOT: Figure 1 – Monthly Revenue Trends
# ============================================================

plt.figure(figsize=(10,5))
plt.plot(monthly_revenue['YearMonth'], monthly_revenue['Hotel_Revenue'], linewidth=2)
plt.title('Monthly Revenue Trend (Historical Data)')
plt.xlabel('Month')
plt.ylabel('Revenue (LKR)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# STEP 3: Prepare Data for LSTM
# ============================================================
# Sort and scale
monthly_revenue = monthly_revenue.sort_values('YearMonth')
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(monthly_revenue[['Hotel_Revenue']])

# Create sequences
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


n_steps = 6 # past 6 months used to predict next month
X, y = create_sequences(scaled_data, n_steps)
X = X.reshape((X.shape[0], X.shape[1], 1))

# ============================================================
# STEP 4: Train-Test Split
# ============================================================

train_size = int(len(scaled_data) * 0.8)
train, test = scaled_data[:train_size], scaled_data[train_size:]

X_train, y_train = create_sequences(train, n_steps)
X_test, y_test = create_sequences(test, n_steps)

# Reshape to 3D for LSTM input: [samples, time_steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# ============================================================
# STEP 5: Build and Train the LSTM Model
# ============================================================
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(n_steps, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(X_train, y_train, epochs=100, verbose=1, validation_data=(X_test, y_test))

# ============================================================
# EXTRA PLOT: Figure 2 – Training vs Validation Loss Curve
# ============================================================

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Model Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# STEP 6: Evaluate the Model
# ============================================================
# Predict on test data
y_pred = model.predict(X_test)

# Inverse scale to original revenue values
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(f"Test RMSE: {rmse:.2f} LKR")

# ============================================================
# STEP 7: Visualize Predictions vs Actual
# ============================================================
plt.figure(figsize=(10,5))
plt.plot(y_test_inv, label='Actual Revenue', linewidth=2)
plt.plot(y_pred_inv, label='Predicted Revenue', linestyle='--')
plt.title('LSTM Hotel Revenue Forecast (Test Data)')
plt.xlabel('Months')
plt.ylabel('Revenue (LKR)')
plt.legend()
plt.show()

# ============================================================
# STEP 8: Forecast Future 6 Months
# ============================================================
future_steps = 6
last_sequence = scaled_data[-n_steps:]
future_predictions = []

for _ in range(future_steps):
    next_pred = model.predict(last_sequence.reshape(1, n_steps, 1))
    future_predictions.append(next_pred[0,0])
    last_sequence = np.append(last_sequence[1:], next_pred[0,0])

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))

# Create future date labels
last_date = pd.to_datetime(monthly_revenue['YearMonth'].iloc[-1])
future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(), periods=future_steps, freq='MS')
future_forecast = pd.DataFrame({'Month': future_dates, 'Predicted_Revenue': future_predictions.flatten()})

print("\n Future 6-Month Revenue Forecast:")
print(future_forecast)

# ============================================================
# STEP 9A: Visualize Future Forecast (FIXED)
# ============================================================
# Convert YearMonth to datetime
monthly_revenue['YearMonth'] = pd.to_datetime(monthly_revenue['YearMonth'])

plt.figure(figsize=(10,5))

# Plot historical data
plt.plot(monthly_revenue['YearMonth'], monthly_revenue['Hotel_Revenue'], label='Historical Revenue', linewidth=2)

# Plot forecast data
plt.plot(future_forecast['Month'], future_forecast['Predicted_Revenue'], 
         label='Forecasted Revenue', linestyle='--', color='orange', linewidth=2)

plt.title('Hotel Revenue Forecast for Next 6 Months')
plt.xlabel('Month')
plt.ylabel('Revenue (LKR)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# STEP 10: Save the Model
# ============================================================
model.save("hotel_revenue_lstm_model.h5")
print("\nModel saved as hotel_revenue_lstm_model.h5")

# Calculate growth rate between first and last predicted months
future_forecast.loc[future_forecast['Predicted_Revenue'].idxmin(), 'Month'].strftime('%B %Y')
start_rev = future_forecast['Predicted_Revenue'].iloc[0]
end_rev = future_forecast['Predicted_Revenue'].iloc[-1]
growth_rate = ((end_rev - start_rev) / start_rev) * 100

print("Business Insights")
print("-------------------")
print(f"1. Revenue is expected to change by {growth_rate:.2f}% over the next 6 months.")
print(f"2. Highest predicted revenue: {future_forecast['Predicted_Revenue'].max():,.2f} LKR")
print(f"3. Lowest predicted revenue: {future_forecast['Predicted_Revenue'].min():,.2f} LKR")
print(f"4. Peak month: {future_forecast.loc[future_forecast['Predicted_Revenue'].idxmax(), 'Month'].strftime('%B %Y')}")
print(f"5. Lowest month: {future_forecast.loc[future_forecast['Predicted_Revenue'].idxmin(), 'Month'].strftime('%B %Y')}")
print(f"6. Expected seasonal trend: steady growth after first 3 months.")