import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the absolute path to the data file
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
data_path = os.path.join(root_dir, "SDG_Indicator_3.2.1.csv")

# Load the data
df = pd.read_csv(data_path,
                 sep=';',  # Use semicolon as separator
                 on_bad_lines='skip',
                 encoding='utf-8')

# Select and rename relevant columns (using column indices)
df = df.iloc[:, [3, 6, 7, 8]]  # SeriesCode, GeoAreaName, TimePeriod, Value columns
df.columns = ["SeriesCode", "GeoAreaName", "TimePeriod", "Value"]

# Filter for Germany, remove NaN values, and keep only SH_DYN_IMRT values
df = df[df["GeoAreaName"] == "Germany"].dropna()
df = df[df["SeriesCode"] == "SH_DYN_IMRT"]  # Filter for infant mortality rate indicator
df = df.rename(columns={"TimePeriod": "Year"}).sort_values("Year")

# Print the data info
print("\nData sample:")
print(df[["Year", "Value"]].head())
print("\nValue statistics:")
print(df["Value"].describe())

df = df[["Year", "Value"]].set_index("Year")

# Prepare data for Prophet
df_prophet = df.reset_index().rename(columns={"Year": "ds", "Value": "y"})
# Convert years to datetime
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'].astype(str), format='%Y')

# Create and fit the model with some constraints
model = Prophet(
    growth='linear',
    changepoint_prior_scale=0.05,  # More conservative trend changes
    seasonality_mode='multiplicative'
)
model.fit(df_prophet)

# Make future predictions (5 years into the future)
future = model.make_future_dataframe(periods=5, freq='Y')
forecast = model.predict(future)

# Plot the results with proper y-axis limits
plt.figure(figsize=(12, 6))
plt.plot(df_prophet['ds'], df_prophet['y'], 'b.', label='Actual')
plt.plot(forecast['ds'], forecast['yhat'], 'r-', label='Prediction')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                 color='r', alpha=0.2, label='Confidence Interval')

plt.title('Infant Mortality Rate Forecast for Germany')
plt.xlabel('Year')
plt.ylabel('Infant Deaths per 1,000 Live Births')

# Set y-axis limits to reasonable range
ymin = min(min(df_prophet['y']), forecast['yhat_lower'].min()) * 0.9
ymax = max(max(df_prophet['y']), forecast['yhat_upper'].max()) * 1.1
plt.ylim(ymin, ymax)

plt.legend()
plt.grid(True)
plt.show()
