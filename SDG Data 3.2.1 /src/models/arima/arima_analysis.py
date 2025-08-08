import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os

# Lade CSV mit low_memory=False Option
df = pd.read_csv("data/processed/all_goals_combined.csv", 
                 sep=';', 
                 encoding='utf-8',
                 low_memory=False)

# Nur benötigte Spalten
df = df[['GeoAreaName', 'TimePeriod', 'Value']]
df = df.rename(columns={'TimePeriod': 'Year'})
df['Year'] = pd.to_datetime(df['Year'].astype(str), format='%Y')

# Erstelle Verzeichnis
os.makedirs("results/arima_by_country", exist_ok=True)

# Für jedes Land analysieren
for country in sorted(df['GeoAreaName'].unique()):
    print(f"\nAnalysiere {country}...")

    country_data = df[df['GeoAreaName'] == country]

    # Mittelwert + Standardabweichung pro Jahr
    grouped = country_data.groupby('Year').agg(['mean', 'std'])['Value']
    if grouped.shape[0] < 6:  # Minimum Datenjahre für Modellierung
        print(f"Überspringe {country} (nicht genug Daten)")
        continue

    grouped['lower'] = grouped['mean'] - 1.96 * grouped['std']
    grouped['upper'] = grouped['mean'] + 1.96 * grouped['std']

    ts = grouped['mean']
    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]

    print(f"Train: {train.index.min().year}–{train.index.max().year}, Test: {test.index.min().year}–{test.index.max().year}")

    try:
        model = ARIMA(train, order=(1,1,1))
        model_fit = model.fit()

        future_dates = pd.date_range(start=test.index[0], end=ts.index[-1] + pd.DateOffset(years=5), freq='Y')
        predictions = model_fit.forecast(steps=len(future_dates))
        predictions.index = future_dates

        rmse = np.sqrt(mean_squared_error(test, predictions[:len(test)]))
        print(f"RMSE: {rmse:.4f}")

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(grouped.index, grouped['mean'], 'b.', label='Mittelwert')
        plt.fill_between(grouped.index, grouped['lower'], grouped['upper'], color='blue', alpha=0.2, label='95% CI')
        plt.axvline(x=train.index[-1], color='gray', linestyle='--', alpha=0.5)
        plt.plot(predictions.index, predictions, 'r-', label='ARIMA Prognose')
        plt.scatter(country_data['Year'], country_data['Value'], color='red', alpha=0.3, s=30, label='Originalwerte')

        plt.title(f'SDG Index Score – {country}')
        plt.xlabel('Jahr')
        plt.ylabel('SDG Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'results/arima_by_country/{country}_sdg_analysis.png')
        plt.close()

        with open(f'results/arima_by_country/{country}_summary.txt', 'w') as f:
            f.write(f"ARIMA Modellzusammenfassung für {country}\n")
            f.write("=" * 50 + "\n\n")
            f.write(str(model_fit.summary()))
            f.write(f"\n\nRMSE: {rmse:.4f}")

    except Exception as e:
        print(f"Fehler bei {country}: {e}")
