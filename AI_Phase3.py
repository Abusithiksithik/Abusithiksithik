
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the ROC registration dataset (replace 'dataset.csv' with your dataset)
data = pd.read_csv('dataset.csv')

# Data preprocessing (assumes a date column and registration count column)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Time Series Decomposition
result = seasonal_decompose(data['Registrations'], model='additive')
result.plot()
plt.show()

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Time Series Forecasting using Holt-Winters Exponential Smoothing
model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=12)
model_fit = model.fit(optimized=True, use_brute=True)
forecast = model_fit.forecast(len(test))

# Evaluate the model
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Visualize the forecast
plt.plot(train, label='Training Data')
plt.plot(test, label='Testing Data')
plt.plot(forecast, label='Forecast')
plt.legend()
plt.show()
```

In this code:

1. We load the ROC registration dataset (replace 'dataset.csv' with your actual dataset).
2. Preprocess the data to have a date column and a registration count column.
3. Perform time series decomposition to understand trends, seasonality, and residuals.
4. Split the data into training and testing sets.
5. Use Holt-Winters Exponential Smoothing for time series forecasting and evaluate the model's performance.

Please note that this is a basic example. A real project would require more advanced time series modeling, feature engineering, and comprehensive documentation. Additionally, this code assumes a specific structure of your dataset, so you should adapt it to your data's format and requirements.
