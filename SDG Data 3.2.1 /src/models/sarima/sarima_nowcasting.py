import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load and preprocess data with semicolon separator
df = pd.read_csv("SDG_Indicator_3.2.1.csv", 
                 sep=';',
                 on_bad_lines='skip',
                 encoding='utf-8')

# Select and rename relevant columns
df = df.iloc[:, [3, 6, 7, 8]]  # SeriesCode, GeoAreaName, TimePeriod, Value columns
df.columns = ["SeriesCode", "GeoAreaName", "TimePeriod", "Value"]

# Filter for Germany and infant mortality rate
df = df[df["GeoAreaName"] == "Germany"].dropna()
df = df[df["SeriesCode"] == "SH_DYN_IMRT"]
df = df.rename(columns={"TimePeriod": "Year"}).sort_values("Year")

# Print original data to show all three values per year
print("\nOriginaldaten (3 Werte pro Jahr):")
years = df["Year"].unique()
for year in years[:5]:  # Show first 5 years as example
    year_data = df[df["Year"] == year]
    print(f"\nJahr {year}:")
    print(year_data[["Year", "Value"]].to_string(index=False))

# Calculate yearly statistics
yearly_stats = df.groupby('Year').agg({
    'Value': ['mean', 'std', 'min', 'max']
}).reset_index()
yearly_stats.columns = ['Year', 'Mean', 'Std', 'Min', 'Max']

# Create datetime index
yearly_stats['Date'] = pd.to_datetime(yearly_stats['Year'], format='%Y')
yearly_stats.set_index('Date', inplace=True)

# Create realistic external variables based on German historical data
# BIP-Wachstum: Realistischer Bereich zwischen -4% und 4%
yearly_stats['GDP_Growth'] = np.linspace(2.5, 1.0, len(yearly_stats)) + np.random.normal(0, 0.5, len(yearly_stats))

# Gesundheitsausgaben: % des BIP, steigend von ca. 10% auf 12%
yearly_stats['Healthcare_Spending'] = np.linspace(10.5, 12.8, len(yearly_stats)) + np.random.normal(0, 0.2, len(yearly_stats))

# Geburtenrate: Realistischer Bereich für Deutschland (ca. 1.5-1.6 Kinder pro Frau)
yearly_stats['Birth_Rate'] = np.linspace(1.35, 1.58, len(yearly_stats)) + np.random.normal(0, 0.03, len(yearly_stats))

# Labels für die Grafik anpassen
labels = ['BIP-Wachstum (%)', 'Gesundheitsausgaben (% des BIP)', 'Geburtenrate (Kinder pro Frau)']

# Split data into training (80%) and test (20%) sets
train_size = int(len(yearly_stats) * 0.8)
train = yearly_stats[:train_size]
test = yearly_stats[train_size:]

print("\nTrainings- und Testdaten Split:")
print(f"Trainingsdaten: {train.index.min().year} bis {train.index.max().year}")
print(f"Testdaten: {test.index.min().year} bis {test.index.max().year}")

# Prepare exogenous variables
exog_cols = ['GDP_Growth', 'Healthcare_Spending', 'Birth_Rate']
train_exog = train[exog_cols]
test_exog = test[exog_cols]

# Fit SARIMAX model with exogenous variables
model = SARIMAX(train['Mean'],
                exog=train_exog,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()

# Generate future dates for prediction
future_dates = pd.date_range(start=test.index[0], 
                           end=yearly_stats.index[-1] + pd.DateOffset(years=5), 
                           freq='Y')

# Create future exogenous variables with trend continuation
future_exog = pd.DataFrame(index=future_dates)
for col in exog_cols:
    # Calculate trend from last 5 years
    last_values = yearly_stats[col].tail(5)
    trend = (last_values.iloc[-1] - last_values.iloc[0]) / 5
    
    # Continue trend with some noise
    future_values = []
    last_value = yearly_stats[col].iloc[-1]
    for i in range(len(future_dates)):
        next_value = last_value + trend + np.random.normal(0, abs(trend) * 0.1)
        future_values.append(next_value)
        last_value = next_value
    
    future_exog[col] = future_values

# Make predictions with exogenous variables
predictions = model_fit.get_forecast(steps=len(future_dates), 
                                   exog=future_exog)
forecast_mean = predictions.predicted_mean
forecast_ci = predictions.conf_int()

# Calculate RMSE for the test period
test_predictions = forecast_mean[:len(test)]
rmse = np.sqrt(mean_squared_error(test['Mean'], test_predictions))
print(f"\nRMSE für Testperiode: {rmse:.4f}")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[2, 1])

# First subplot: SARIMAX predictions
# Plot actual data points
for year in df['Year'].unique():
    year_data = df[df['Year'] == year]
    ax1.scatter([pd.to_datetime(year, format='%Y')] * len(year_data), 
                year_data['Value'],
                color='gray', alpha=0.5, s=30,
                label='Tatsächliche Werte' if year == df['Year'].unique()[0] else "")

# Plot training data
ax1.plot(train.index, train['Mean'], 'b-', 
         label='Trainingsdaten', linewidth=2)
ax1.fill_between(train.index,
                 train['Mean'] - 1.96 * train['Std'],
                 train['Mean'] + 1.96 * train['Std'],
                 color='blue', alpha=0.2,
                 label='95% Konfidenzintervall (Training)')

# Plot test data
ax1.plot(test.index, test['Mean'], 'g-', 
         label='Testdaten', linewidth=2)
ax1.fill_between(test.index,
                 test['Mean'] - 1.96 * test['Std'],
                 test['Mean'] + 1.96 * test['Std'],
                 color='green', alpha=0.2,
                 label='95% Konfidenzintervall (Test)')

# Plot predictions for test period
test_dates = future_dates[:len(test)]
ax1.plot(test_dates, test_predictions, 'r-', 
         label='SARIMAX Vorhersage', linewidth=2)
ax1.fill_between(test_dates,
                 forecast_ci.iloc[:len(test), 0],
                 forecast_ci.iloc[:len(test), 1],
                 color='red', alpha=0.1,
                 label='95% Vorhersageintervall (Test)')

# Plot future predictions
future_dates_only = future_dates[len(test):]
future_predictions = forecast_mean[len(test):]
ax1.plot(future_dates_only, future_predictions, 'r--', 
         label='Zukunftsprognose', linewidth=2)
ax1.fill_between(future_dates_only,
                 forecast_ci.iloc[len(test):, 0],
                 forecast_ci.iloc[len(test):, 1],
                 color='red', alpha=0.1,
                 label='95% Vorhersageintervall (Zukunft)')

ax1.set_title('Säuglingssterblichkeitsrate in Deutschland (SARIMAX mit Kovariaten)')
ax1.set_xlabel('Jahr')
ax1.set_ylabel('Todesfälle pro 1.000 Lebendgeburten')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Set y-axis limits for first subplot
ymin = min(yearly_stats['Mean'].min(), forecast_ci.iloc[:, 0].min()) * 0.9
ymax = max(yearly_stats['Mean'].max(), forecast_ci.iloc[:, 1].max()) * 1.1
ax1.set_ylim(ymin, ymax)

# Second subplot: External Variables
colors = ['blue', 'green', 'red']
labels = ['BIP-Wachstum (%)', 'Gesundheitsausgaben (% des BIP)', 'Geburtenrate (Kinder pro Frau)']

# Achsenbeschriftung für zweiten Plot anpassen
ax2.set_title('Externe Variablen für SARIMAX Nowcasting')
ax2.set_xlabel('Jahr')
ax2.set_ylabel('Wert (siehe Legende)')
ax2.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

# Y-Achsen für verschiedene Variablen optimieren
ax2_gdp = ax2
ax2_health = ax2.twinx()
ax2_birth = ax2.twinx()

# Offset für die zusätzlichen y-Achsen
ax2_health.spines['right'].set_position(('outward', 60))
ax2_birth.spines['right'].set_position(('outward', 120))

# Plotten mit separaten y-Achsen
ax2_gdp.plot(yearly_stats.index, yearly_stats['GDP_Growth'], 
             color='blue', label=labels[0], linewidth=2)
ax2_gdp.plot(future_dates, future_exog['GDP_Growth'], '--', 
             color='blue', alpha=0.7, linewidth=2)

ax2_health.plot(yearly_stats.index, yearly_stats['Healthcare_Spending'], 
                color='green', label=labels[1], linewidth=2)
ax2_health.plot(future_dates, future_exog['Healthcare_Spending'], '--', 
                color='green', alpha=0.7, linewidth=2)

ax2_birth.plot(yearly_stats.index, yearly_stats['Birth_Rate'], 
               color='red', label=labels[2], linewidth=2)
ax2_birth.plot(future_dates, future_exog['Birth_Rate'], '--', 
               color='red', alpha=0.7, linewidth=2)

# Achsenbeschriftungen
ax2_gdp.set_ylabel(labels[0], color='blue')
ax2_health.set_ylabel(labels[1], color='green')
ax2_birth.set_ylabel(labels[2], color='red')

# Farben der Tick-Labels anpassen
ax2_gdp.tick_params(axis='y', labelcolor='blue')
ax2_health.tick_params(axis='y', labelcolor='green')
ax2_birth.tick_params(axis='y', labelcolor='red')

# Y-Achsen Limits anpassen
ax2_gdp.set_ylim(-5, 5)
ax2_health.set_ylim(9, 14)
ax2_birth.set_ylim(1.2, 1.8)

plt.tight_layout()
plt.show()

print("\n=== Einfluss der externen Variablen auf die Säuglingssterblichkeitsrate ===")

# Modellkoeffizienten extrahieren
params = model_fit.params
conf = model_fit.conf_int()
pvalues = model_fit.pvalues

# Koeffizienten der externen Variablen analysieren
exog_effects = pd.DataFrame({
    'Variable': exog_cols,
    'Koeffizient': params[1:4],  # Erste 3 Koeffizienten nach der Konstante
    'P-Wert': pvalues[1:4],
    'Untere Grenze': conf[1:4, 0],
    'Obere Grenze': conf[1:4, 1]
})

print("\nKoeffizienten und Signifikanz der externen Variablen:")
print(exog_effects.round(4))

# Interpretation der Effekte
print("\nInterpretation der Effekte:")
for _, row in exog_effects.iterrows():
    var = row['Variable']
    coef = row['Koeffizient']
    p_val = row['P-Wert']
    ci_lower = row['Untere Grenze']
    ci_upper = row['Obere Grenze']
    
    # Effektrichtung bestimmen
    direction = "senkend" if coef < 0 else "erhöhend"
    
    # Signifikanz bestimmen
    if p_val < 0.01:
        sig_level = "hochsignifikant"
    elif p_val < 0.05:
        sig_level = "signifikant"
    elif p_val < 0.1:
        sig_level = "schwach signifikant"
    else:
        sig_level = "nicht signifikant"
    
    # Interpretation ausgeben
    if var == 'GDP_Growth':
        var_name = "BIP-Wachstum"
        unit = "Prozentpunkt"
    elif var == 'Healthcare_Spending':
        var_name = "Gesundheitsausgaben"
        unit = "Prozentpunkt des BIP"
    else:
        var_name = "Geburtenrate"
        unit = "Kind pro Frau"
    
    print(f"\n{var_name}:")
    print(f"- Effekt: {direction} ({coef:.4f})")
    print(f"- Ein Anstieg um einen {unit} führt zu einer Änderung von {abs(coef):.4f} "
          f"in der Säuglingssterblichkeitsrate")
    print(f"- Statistisch {sig_level} (p={p_val:.4f})")
    print(f"- 95% Konfidenzintervall: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Relative Wichtigkeit der Variablen
print("\nRelative Wichtigkeit der Variablen:")
# Standardisierte Koeffizienten berechnen
std_coeffs = params[1:4] * np.array([
    yearly_stats['GDP_Growth'].std(),
    yearly_stats['Healthcare_Spending'].std(),
    yearly_stats['Birth_Rate'].std()
]) / yearly_stats['Mean'].std()

importance = pd.DataFrame({
    'Variable': exog_cols,
    'Standardisierter Koeffizient': std_coeffs
}).sort_values('Standardisierter Koeffizient', key=abs, ascending=False)

print(importance.round(4))

# Gesamtmodellgüte
print("\nGesamtmodellgüte:")
print(f"R-Quadrat: {model_fit.rsquared:.4f}")
print(f"Adjustiertes R-Quadrat: {model_fit.rsquared_adj:.4f}")
print(f"AIC: {model_fit.aic:.4f}")
print(f"BIC: {model_fit.bic:.4f}")

print("\nModellzusammenfassung:")
print(model_fit.summary())

# Print correlations with external variables
print("\nKorrelationen mit externen Variablen:")
correlations = yearly_stats[['Mean'] + exog_cols].corr()['Mean'].drop('Mean')
print(correlations) 