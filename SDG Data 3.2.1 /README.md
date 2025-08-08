# Zeitreihenanalyse der Säuglingssterblichkeitsrate (SDG 3.2.1)

## Zusammenfassung
Dieser Prototyp untersucht die Entwicklung der Säuglingssterblichkeitsrate im Rahmen der UN Sustainable Development Goals (SDG 3.2.1). Durch die Anwendung verschiedener Zeitreihenmodelle werden Trends identifiziert und Prognosen erstellt.

## Methodologie

Das Projekt implementiert drei komplementäre Ansätze:

### 1. Prophet-Modell ([Code](src/models/prophet/prophet_analysis.py))
- Facebook's Prophet für robuste Zeitreihenprognosen
- Berücksichtigung von Trends und Saisonalität
- Automatische Erkennung von Ausreißern

![Prophet Prognose](results/figures/Prophet%20Mortality%20Rate%20Prognose.png)
*Abbildung 1: Prophet-Modell Vorhersage mit Konfidenzintervallen*

### 2. ARIMA-Modell ([Code](src/models/arima/arima_analysis.py))
- Klassische Zeitreihenanalyse
- Integration von Autoregression und Moving Average
- Differenzierung zur Stationaritätsherstellung

![ARIMA Prognose](results/figures/ARIMA%20Mortality%20Rate%20Prognose.png)
*Abbildung 2: ARIMA-Modell Vorhersage und Trendanalyse*

### 3. SARIMA-Modell ([Code](src/models/sarima/sarima_nowcasting.py))
- Erweiterung des ARIMA-Modells um saisonale Komponenten
- Berücksichtigung externer Variablen
- Verbesserte Genauigkeit durch Saisonalitätsmodellierung

![SARIMA Analyse](results/figures/SARIMA%20Mortality%20Rate.png)
*Abbildung 3: SARIMA-Modell mit saisonaler Komponente*

## Hauptergebnisse

1. **Trendanalyse**
   - Signifikanter Abwärtstrend in der globalen Säuglingssterblichkeitsrate
   - Regionale Unterschiede in der Entwicklungsgeschwindigkeit
   - Identifikation von Schlüsselfaktoren für Verbesserungen

2. **Modellvergleich**
   - Prophet: Beste Performance bei langfristigen Trends
   - ARIMA: Gute Kurzfristprognosen
   - SARIMA: Präziseste Vorhersagen bei saisonalen Mustern

3. **Zukunftsprognosen**
   - Fortsetzung des positiven Trends erwartet
   - Regionale Unterschiede bleiben bestehen
   - Potenzial für weitere Verbesserungen identifiziert

## Projektstruktur

```
.
├── data/               # Datendateien
│   └── raw/           # Originaldaten (SDG 3.2.1)
├── results/           # Analyseergebnisse
│   ├── figures/       # Visualisierungen
│   └── reports/       # Detaillierte Berichte
└── src/               # Quellcode
    └── models/        # Modellimplementierungen
        ├── prophet/   # Prophet-Modell
        ├── arima/     # ARIMA-Modell
        └── sarima/    # SARIMA-Modell
```

## Technische Details

### Datenquelle
- SDG Indicator 3.2.1 Dataset
- Zeitraum: 2000-2022
- Globale Abdeckung mit Ländergranularität

### Implementierung
- Python 3.8+
- Hauptbibliotheken:
  - Prophet
  - statsmodels (ARIMA/SARIMA)
  - pandas
  - numpy
  - matplotlib

## Installation

```bash
# Repository klonen
git clone https://github.com/StethDeloche/SDG-Forecasting.git
cd SDG-Forecasting

# Abhängigkeiten installieren
pip install -r requirements.txt

# Modelle ausführen
python src/models/prophet/prophet_analysis.py
python src/models/arima/arima_analysis.py
python src/models/sarima/sarima_nowcasting.py
```

## Weitere Ressourcen

- [Prophet Dokumentation](https://facebook.github.io/prophet/)
- [ARIMA Methodik](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [SDG 3.2.1 Beschreibung](https://sdg-tracker.org/good-health#3.2.1)

## Kontakt

Für Fragen oder Anmerkungen stehe ich gerne zur Verfügung 