import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load and preprocess data with semicolon separator
df = pd.read_csv("SDG_Indicator_3.2.1.csv", 
                 sep=';',  # Use semicolon as separator
                 on_bad_lines='skip',
                 encoding='utf-8')

# Select and rename relevant columns
df = df.iloc[:, [3, 6, 7, 8]]  # SeriesCode, GeoAreaName, TimePeriod, Value columns
df.columns = ["SeriesCode", "GeoAreaName", "TimePeriod", "Value"]

# Filter for Germany and infant mortality rate
df = df[df["GeoAreaName"] == "Germany"].dropna()
df = df[df["SeriesCode"] == "SH_DYN_IMRT"]  # Filter for infant mortality rate indicator
df = df.rename(columns={"TimePeriod": "Year"}).sort_values("Year")

# Print original data to show all three values per year
print("\nOriginaldaten (3 Werte pro Jahr):")
years = df["Year"].unique()
for year in years[:5]:  # Show first 5 years as example
    year_data = df[df["Year"] == year]
    print(f"\nJahr {year}:")
    print(year_data[["Year", "Value"]].to_string(index=False))

# Convert Year to datetime
df['Year'] = pd.to_datetime(df['Year'].astype(str), format='%Y')

# Calculate point estimates and uncertainty intervals
def calculate_estimates(group):
    values = group['Value'].values
    # Calculate mean and standard deviation
    mean = np.mean(values)
    std = np.std(values)
    # Calculate confidence interval
    lower_bound = mean - 1.96 * std  # 95% confidence interval
    upper_bound = mean + 1.96 * std
    return pd.Series({
        'Punktschätzung': mean,
        'Untere Schätzung': lower_bound,
        'Obere Schätzung': upper_bound
    })

# Calculate estimates for each year
df_pivot = df.groupby('Year').apply(calculate_estimates)

print("\nDaten nach Schätzungstyp (mit Unsicherheitsintervallen):")
print(df_pivot.head())

# Use point estimate for ARIMA modeling
ts = df_pivot['Punktschätzung']

# Split data into training (80%) and test (20%) sets
train_size = int(len(ts) * 0.8)
train = ts[:train_size]
test = ts[train_size:]

print("\nTrainings- und Testdaten Split:")
print(f"Trainingsdaten: {train.index.min().year} bis {train.index.max().year}")
print(f"Testdaten: {test.index.min().year} bis {test.index.max().year}")

# Fit ARIMA model
model = ARIMA(train, order=(1,1,1))
model_fit = model.fit()

# Generate future dates for prediction
future_dates = pd.date_range(start=test.index[0], 
                           end=ts.index[-1] + pd.DateOffset(years=5), 
                           freq='Y')

# Make predictions
predictions = model_fit.forecast(steps=len(future_dates))
predictions.index = future_dates

# Calculate RMSE for the test period
rmse = np.sqrt(mean_squared_error(test, predictions[:len(test)]))
print(f"\nRMSE für Testperiode: {rmse:.4f}")

# Plot the results
plt.figure(figsize=(12, 6))

# Plot all three estimates
plt.plot(df_pivot.index, df_pivot['Untere Schätzung'], 'b--', alpha=0.3, label='Untere Schätzung')
plt.plot(df_pivot.index, df_pivot['Punktschätzung'], 'b.', label='Punktschätzung')
plt.plot(df_pivot.index, df_pivot['Obere Schätzung'], 'b--', alpha=0.3, label='Obere Schätzung')

# Fill between upper and lower estimates
plt.fill_between(df_pivot.index, 
                 df_pivot['Untere Schätzung'],
                 df_pivot['Obere Schätzung'],
                 alpha=0.1,
                 color='blue')

# Plot individual data points for each year
for year in df['Year'].unique():
    year_data = df[df['Year'] == year]
    plt.scatter([year] * 3, year_data['Value'], 
                color='red', alpha=0.5, s=30, 
                label='Tatsächliche Werte' if year == df['Year'].unique()[0] else "")

# Split training and test data visualization
train_mask = df_pivot.index <= train.index[-1]
test_mask = df_pivot.index > train.index[-1]

# Highlight training and test periods
plt.axvline(x=train.index[-1], color='gray', linestyle='--', alpha=0.5)
plt.text(train.index[-1], plt.ylim()[1], 'Train-Test Split', 
         rotation=90, va='top', ha='right')

# Plot predictions
plt.plot(predictions.index, predictions, 'r-', label='ARIMA Vorhersage')

plt.title('Säuglingssterblichkeitsrate in Deutschland (ARIMA Prognose)')
plt.xlabel('Jahr')
plt.ylabel('Todesfälle pro 1.000 Lebendgeburten')

# Format x-axis to show years
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
plt.xticks(rotation=45)

# Set y-axis limits
ymin = min(df_pivot['Untere Schätzung'].min(), predictions.min(), df['Value'].min()) * 0.9
ymax = max(df_pivot['Obere Schätzung'].max(), predictions.max(), df['Value'].max()) * 1.1
plt.ylim(ymin, ymax)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print model summary
print("\nARIMA Modell Zusammenfassung:")
print(model_fit.summary()) 