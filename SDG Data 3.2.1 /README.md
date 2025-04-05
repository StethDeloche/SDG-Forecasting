# SDG 3.2.1 - Säuglingssterblichkeitsrate Analyse und Prognose

Dieser Prototyp analysiert und prognostiziert die Säuglingssterblichkeitsrate in Deutschland basierend auf den SDG-Indikatoren.

## Projektstruktur

```
SDG_Data_3.2.1/
├── data/                      # Rohdaten und verarbeitete Daten
│   ├── raw/                   # Original-Datensätze
│   └── processed/             # Aufbereitete Datensätze
│
├── src/                       # Quellcode
│   ├── preprocessing/         # Datenaufbereitung
│   ├── models/               # Verschiedene Modellimplementierungen
│   │   ├── prophet/          # Prophet-Modelle
│   │   ├── arima/           # ARIMA-Modelle
│   │   ├── sarima/          # SARIMA-Modelle
│   │   └── ml/              # Machine Learning Modelle
│   └── utils/                # Hilfsfunktionen
│
├── notebooks/                # Jupyter Notebooks für Analyse
├── results/                  # Ergebnisse und Visualisierungen
│   ├── figures/             # Grafiken
│   └── reports/             # Berichte
└── config/                  # Konfigurationsdateien
```

## Modelle

Das Projekt implementiert verschiedene Zeitreihenmodelle zur Prognose der Säuglingssterblichkeitsrate:

1. **Prophet**: Facebook's Prophet für Zeitreihenprognosen
2. **ARIMA**: Autoregressive Integrated Moving Average
3. **SARIMA**: Seasonal ARIMA mit externen Variablen
4. **Machine Learning**: Random Forest und andere ML-Modelle

### Modellbeschreibungen

#### Prophet
Prophet ist ein von Facebook entwickeltes Prognosemodell, das besonders gut mit saisonalen Mustern und Feiertagen umgehen kann. Es ist robust gegenüber fehlenden Daten und Ausreißern. Prophet verwendet ein additives Modell, das Trend, Saisonalität und Feiertagseffekte kombiniert.

#### ARIMA
ARIMA (Autoregressive Integrated Moving Average) ist ein klassisches Zeitreihenmodell, das drei Komponenten kombiniert:
- Autoregression (AR): Nutzt vergangene Werte
- Integration (I): Differenziert die Zeitreihe für Stationarität
- Moving Average (MA): Modelliert Fehlerterme

#### SARIMA
SARIMA erweitert ARIMA um saisonale Komponenten. Es ist besonders nützlich für Daten mit regelmäßigen saisonalen Mustern (z.B. jährliche Schwankungen). Zusätzlich können externe Variablen (Kovariaten) einbezogen werden, um die Prognosegenauigkeit zu verbessern.

#### Machine Learning (Random Forest)
Der Random Forest Ansatz nutzt mehrere Entscheidungsbäume für die Prognose. Er kann komplexe nicht-lineare Beziehungen modellieren und ist robust gegenüber Ausreißern. Im Kontext der Zeitreihenprognose werden historische Werte und zusätzliche Features als Eingabevariablen verwendet.

## Installation

1. Klonen Sie das Repository
2. Installieren Sie die Abhängigkeiten:
   ```bash
   pip install -r requirements.txt
   ```

## Verwendung

Jedes Modell kann separat ausgeführt werden:

```bash
# SARIMA Nowcasting
python src/models/sarima/sarima_nowcasting.py

# ARIMA Analyse
python src/models/arima/arima_analysis.py

# Prophet Analyse
python src/models/prophet/prophet_analysis.py

# Machine Learning Analyse
python src/models/ml/ml_analysis.py
```

## Ergebnisse

Die Ergebnisse der Analysen finden sich im `results/` Verzeichnis:
- Grafiken in `results/figures/`
- Berichte in `results/reports/` 