import pandas as pd

# Beispiel: Datei laden (ersetze den Pfad mit deinem Dateinamen)
df = pd.read_csv("SDG_Indicator_3.2.1.csv")
df

# 👁 Überblick über die Daten
print(df.head())

# 🧹 Nur relevante Spalten auswählen
df = df[["GeoAreaName", "TimePeriod", "Value"]]

# 📆 Index = Jahr, sortieren
df = df.rename(columns={"TimePeriod": "Year"}).sort_values("Year")
df.set_index("Year", inplace=True)

# 📊 Werttyp ändern
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

# 🔍 Nur relevante Daten anzeigen
print(df.head())

# 📊 Werttyp ändern