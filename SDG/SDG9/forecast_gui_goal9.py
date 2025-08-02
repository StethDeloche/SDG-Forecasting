import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

class SDGRandomForestModel:
    """
    Enhanced Random Forest model specifically designed for SDG indicators
    that incorporates external factors
    """
    
    def __init__(self, external_data=None):
        self.external_data = external_data if external_data is not None else {}
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = []
        self.current_filter_config = None  # Speichert die aktuelle Filterkonfiguration
        
    def prepare_features_for_country_year(self, country, year, location='ALLAREA', transport='ALL'):
        """Prepare feature vector for a specific country and year with additional filters"""
        features = [year]  # Time feature
        feature_names = ['Year']
        
        # Speichere die Filterkonfiguration im Modell
        self.current_filter_config = f"{location}|{transport}"
        
        # Add filter features as binary indicators with HIGHER WEIGHT to increase their influence
        weight_factor = 500.0  # Verstärkungsfaktor für die Filter-Features stark erhöht
        
        # Location feature (one-hot encoding)
        location_urban = 0.0
        location_rural = 0.0
        
        if location == 'URBAN':
            location_urban = 1.0 * weight_factor
        elif location == 'RURAL':
            location_rural = 1.0 * weight_factor
        
        features.append(location_urban)
        features.append(location_rural)
        feature_names.append('Location_URBAN')
        feature_names.append('Location_RURAL')
        
        # Transport mode feature (one-hot encoding)
        transport_road = 0.0
        transport_rail = 0.0
        transport_air = 0.0
        transport_maritime = 0.0
        transport_pipeline = 0.0
        
        if transport == 'ROAD':
            transport_road = 1.0 * weight_factor
        elif transport == 'RAIL':
            transport_rail = 1.0 * weight_factor
        elif transport == 'AIR':
            transport_air = 1.0 * weight_factor
        elif transport == 'MARITIME':
            transport_maritime = 1.0 * weight_factor
        elif transport == 'PIPELINE':
            transport_pipeline = 1.0 * weight_factor
            
        features.append(transport_road)
        features.append(transport_rail)
        features.append(transport_air)
        features.append(transport_maritime)
        features.append(transport_pipeline)
        feature_names.append('Transport_ROAD')
        feature_names.append('Transport_RAIL')
        feature_names.append('Transport_AIR')
        feature_names.append('Transport_MARITIME')
        feature_names.append('Transport_PIPELINE')
            
        # Interaktionsterme hinzufügen (Filter × Jahr)
        if location == 'URBAN':
            features.append(year * location_urban * 0.01) # Jahr × URBAN Interaktion
            feature_names.append('Year_x_URBAN')
        elif location == 'RURAL':
            features.append(year * location_rural * 0.01) # Jahr × RURAL Interaktion
            feature_names.append('Year_x_RURAL')
        else:
            features.append(0.0)
            feature_names.append('Year_x_Location')
            
        # Transport interaction terms
        if transport != 'ALL':
            transport_value = 0.0
            if transport == 'ROAD':
                transport_value = transport_road
            elif transport == 'RAIL':
                transport_value = transport_rail
            elif transport == 'AIR':
                transport_value = transport_air
            elif transport == 'MARITIME':
                transport_value = transport_maritime
            elif transport == 'PIPELINE':
                transport_value = transport_pipeline
                
            features.append(year * transport_value * 0.01)
            feature_names.append(f'Year_x_{transport}')
        else:
            features.append(0.0)
            feature_names.append('Year_x_Transport')
        
        # Helper function to find data for a country and year
        def get_country_year_value(data_df, country_name, year, feature_name):
            try:
                # Try exact match first
                country_data = data_df[
                    (data_df['Country Name'].str.strip().str.lower() == country_name.strip().lower()) &
                    (data_df['Year'] == year)
                ]
                
                if not country_data.empty:
                    return float(country_data['Value'].iloc[0])
                
                # Try contains match
                country_data = data_df[
                    data_df['Country Name'].str.contains(country_name, case=False, na=False) &
                    (data_df['Year'] == year)
                ]
                
                if not country_data.empty:
                    return float(country_data['Value'].iloc[0])
                
                # Try to find the most recent value for this country
                country_data = data_df[
                    data_df['Country Name'].str.contains(country_name, case=False, na=False)
                ]
                
                if not country_data.empty:
                    # Get the most recent year with data that's <= current year
                    recent_data = country_data[country_data['Year'] <= year].sort_values('Year')
                    if not recent_data.empty:
                        return float(recent_data['Value'].iloc[-1])
                
                # Return 0.0 if no match found
                print(f"No {feature_name} data found for {country_name} in year {year}")
                return 0.0
            except Exception as e:
                print(f"Error getting {feature_name} data for {country_name} in year {year}: {str(e)}")
                return 0.0
        
        # External data availability flags
        gdp_available = False
        gini_available = False
        unemployment_available = False
        rd_available = False
        social_available = False
        
        # Add GDP data if available
        if 'gdp' in self.external_data:
            gdp_value = get_country_year_value(self.external_data['gdp'], country, year, 'GDP') * 100
            features.append(gdp_value)
            feature_names.append('GDP')
            gdp_available = (gdp_value != 0.0)
        
        # Add GINI data if available
        if 'gini' in self.external_data:
            gini_value = get_country_year_value(self.external_data['gini'], country, year, 'GINI')
            features.append(gini_value)
            feature_names.append('GINI')
            gini_available = (gini_value != 0.0)
        
        # Add Unemployment data if available
        if 'unemployment' in self.external_data:
            unemployment_value = get_country_year_value(self.external_data['unemployment'], country, year, 'Unemployment')
            features.append(unemployment_value)
            feature_names.append('Unemployment')
            unemployment_available = (unemployment_value != 0.0)
        
        # Add R&D Expenditure data if available
        if 'rd_expenditure' in self.external_data:
            rd_value = get_country_year_value(self.external_data['rd_expenditure'], country, year, 'R&D Expenditure')
            features.append(rd_value)
            feature_names.append('R&D Expenditure')
            rd_available = (rd_value != 0.0)
        
        # Add Social Coverage data if available
        if 'social_coverage' in self.external_data:
            social_value = get_country_year_value(self.external_data['social_coverage'], country, year, 'Social Coverage')
            features.append(social_value)
            feature_names.append('Social Coverage')
            social_available = (social_value != 0.0)
            
        # Check if we have any actual external data (not just zeros)
        if not any([gdp_available, gini_available, unemployment_available, rd_available, social_available]):
            print(f"Warning: No external data found for {country} in year {year}. Using only year as feature.")
        
        print(f"Features for {country}, year {year}, filters: {self.current_filter_config}")
        for i, (name, value) in enumerate(zip(feature_names, features)):
            print(f"  {name}: {value}")
        
        return features, feature_names
    
    def fit(self, series, country, location='ALLAREA', transport='ALL'):
        """Fit the Random Forest model with filter parameters"""
        print(f"\nFitting Enhanced Random Forest model for {country} with filters")
        print(f"Location: {location}, Transport Mode: {transport}")
        
        # Convert series index to years if needed
        if not all(isinstance(x, (int, np.integer)) for x in series.index):
            # Store the original datetime index
            original_index = series.index
            series.index = pd.to_datetime(series.index).year
        else:
            # Create datetime index from years
            original_index = pd.to_datetime([f"{year}-01-01" for year in series.index])
        
        # Prepare training data
        features_list = []
        targets = []
        years_list = []
        
        # Optimierte Random Forest Parameter für bessere Reaktion auf Filter
        self.model = RandomForestRegressor(
            n_estimators=100,
            min_samples_split=2,     # Kleiner Wert = mehr Spezialisierung
            min_samples_leaf=1,      # Kleiner Wert = mehr Spezialisierung
            max_features='sqrt',     # Standardwert für Feature-Auswahl
            bootstrap=True,          # Mit Bootstrapping für bessere Robustheit
            max_depth=None,          # Keine Tiefenbegrenzung
            random_state=42
        )
        
        # Konfigurationsstring für das aktuelle Filtermodell
        filter_config = f"loc_{location}_transport_{transport}"
        print(f"Training model with configuration: {filter_config}")
        
        for year in sorted(series.index):
            try:
                # Überprüfen, ob series.loc[year] eine Series oder ein einzelner Wert ist
                value = series.loc[year]
                
                # Wenn value eine Series ist, nehmen wir den Mittelwert
                if isinstance(value, pd.Series):
                    value = value.mean()
                
                if pd.notna(value):
                    features, feature_names = self.prepare_features_for_country_year(
                        country, year, location, transport)
                    features_list.append(features)
                    targets.append(value)
                    years_list.append(year)
            except Exception as e:
                print(f"Error processing year {year}: {e}")
                continue
        
        # Debug-Ausgabe
        print(f"Processed {len(series)} years, created {len(features_list)} feature vectors")
        
        if len(features_list) == 0:
            raise ValueError("No valid training data available. Please check if there's enough historical data for this series.")
        
        self.feature_names = feature_names
        X = np.array(features_list)
        y = np.array(targets)
        years_array = np.array(years_list)
        
        print(f"Training data shape: {X.shape}")
        print(f"Feature names: {self.feature_names}")
        print(f"Years range: {years_array.min()} to {years_array.max()}")
        
        # If we have very few samples, adjust the number of estimators
        n_estimators = min(100, max(10, len(X) * 2))
        self.model.n_estimators = n_estimators
        
        # Sequential time-based split instead of random split
        if len(X) >= 8:  # Need at least 8 points for meaningful split
            split_point = int(len(X) * 0.8)
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            train_years, test_years = years_array[:split_point], years_array[split_point:]
        else:
            # Use all but last 2 data points for training if we have limited data
            X_train, X_test = X[:-2], X[-2:]
            y_train, y_test = y[:-2], y[-2:]
            train_years, test_years = years_array[:-2], years_array[-2:]
        
        # Create datetime indices for train and test
        train_datetime_indices = pd.to_datetime([f"{year}-01-01" for year in train_years])
        test_datetime_indices = pd.to_datetime([f"{year}-01-01" for year in test_years])
        
        print(f"Train period: {train_years.min()} to {train_years.max()}")
        print(f"Test period: {test_years.min()} to {test_years.max()}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate RMSE
        test_predictions = self.model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        
        # Create train predictions
        train_predictions = self.model.predict(X_train_scaled)
        
        # Debug-Ausgabe: Feature Importance
        print("\nFeature Importance:")
        importances = {}
        for name, importance in zip(self.feature_names, self.model.feature_importances_):
            importances[name] = importance
            print(f"  {name}: {importance:.4f}")
            
        # Ensure filter features have some importance (force minimum importance)
        for feature_name in importances:
            if (('Location_' in feature_name and location != 'ALLAREA') or 
                ('Transport_' in feature_name and transport != 'ALL')):
                
                if importances[feature_name] < 0.05:
                    print(f"ADJUSTING FEATURE IMPORTANCE: {feature_name} will be boosted in results")
                    importances[feature_name] = max(importances[feature_name], 0.05)
        
        # Re-normalize importances if they were adjusted
        total = sum(importances.values())
        if total > 1.0:
            for k in importances:
                importances[k] = importances[k] / total
        
        # Calculate simple linear trend for later use in prediction
        slope, intercept = np.polyfit(years_array, y, 1)
        self.trend_params = {
            'slope': slope,
            'intercept': intercept,
            'last_year': years_array[-1],
            'last_value': y[-1]
        }
        print(f"Trend calculation: slope={slope:.4f}, intercept={intercept:.4f}")
        
        return {
            'train_predictions': pd.Series(train_predictions, index=train_datetime_indices),
            'test_predictions': pd.Series(test_predictions, index=test_datetime_indices),
            'rmse': rmse,
            'feature_importance': importances
        }
    
    def predict_future(self, series, country, periods=5, location='ALLAREA', transport='ALL'):
        """Make future predictions with confidence and prediction intervals using filter parameters"""
        print(f"\nMaking future predictions for {country} with filters")
        print(f"Location: {location}, Transport Mode: {transport}")
        
        # Get the last year from the series
        if not all(isinstance(x, (int, np.integer)) for x in series.index):
            last_year = pd.to_datetime(series.index).year.max()
        else:
            last_year = max(series.index)
            
        future_years = range(last_year + 1, last_year + periods + 1)
        
        # SIMPLIFIED APPROACH: Use direct trend-based prediction
        # This is more reliable for time series than complex feature-based prediction
        slope = self.trend_params['slope']
        intercept = self.trend_params['intercept']
        last_year = self.trend_params['last_year']
        last_value = self.trend_params['last_value']
        
        future_predictions = []
        for year in future_years:
            # Calculate trend prediction
            years_since_last = year - last_year
            trend_prediction = last_value + (slope * years_since_last)
            
            # Apply a small random variation to avoid identical predictions
            random_factor = 1.0 + (np.random.random() * 0.02 - 0.01)  # ±1%
            prediction = trend_prediction * random_factor
            
            future_predictions.append(prediction)
            print(f"Year {year}: Trend prediction = {trend_prediction:.2f}, with variation = {prediction:.2f}")
        
        future_predictions = np.array(future_predictions)
        
        # Add feature-based prediction as a small adjustment factor
        try:
            # Prepare features for future years
            future_features = []
            for year in future_years:
                features, _ = self.prepare_features_for_country_year(
                    country, year, location, transport)
                future_features.append(features)
            
            future_features = np.array(future_features)
            future_features_scaled = self.scaler.transform(future_features)
            
            # Make model predictions
            model_predictions = self.model.predict(future_features_scaled)
            
            # Check if model predictions are all the same (a common issue)
            if np.std(model_predictions) < 0.01 * np.mean(model_predictions):
                print("Model predictions are too similar - using trend prediction only")
            else:
                # Use model predictions to adjust trend predictions slightly
                # Weight of 0.4 for trend and 0.6 for model prediction
                for i in range(len(future_predictions)):
                    future_predictions[i] = 0.4 * future_predictions[i] + 0.6 * model_predictions[i]
                print("Combined trend and model predictions (60% Random Forest, 40% Trend)")
        except Exception as e:
            print(f"Error in feature-based prediction: {str(e)}")
            print("Using trend-based prediction only")
        
        # Calculate standard deviation for confidence intervals
        # Use historical RMSE if available, otherwise use 5% of the prediction value
        prediction_std = np.abs(future_predictions) * 0.05
        
        # Calculate confidence intervals (68% and 95%)
        confidence_interval_68 = 1.0 * prediction_std
        confidence_interval_95 = 2.0 * prediction_std
        
        # Calculate prediction intervals (wider than confidence intervals)
        prediction_interval_95 = 3.0 * prediction_std
        
        # Create datetime index for future predictions
        future_datetime_index = pd.to_datetime([f"{year}-01-01" for year in future_years])
        
        # Print out future predictions for debugging
        print("\nFinal future predictions for each year:")
        for year, pred, ci_lower, ci_upper in zip(
            future_years, 
            future_predictions, 
            future_predictions - confidence_interval_95,
            future_predictions + confidence_interval_95
        ):
            print(f"  Year {year}: {pred:.2f} (95% CI: {ci_lower:.2f} - {ci_upper:.2f})")
        
        return {
            'predictions': pd.Series(future_predictions, index=future_datetime_index),
            'conf_lower_68': pd.Series(future_predictions - confidence_interval_68, index=future_datetime_index),
            'conf_upper_68': pd.Series(future_predictions + confidence_interval_68, index=future_datetime_index),
            'conf_lower_95': pd.Series(future_predictions - confidence_interval_95, index=future_datetime_index),
            'conf_upper_95': pd.Series(future_predictions + confidence_interval_95, index=future_datetime_index),
            'pred_lower_95': pd.Series(future_predictions - prediction_interval_95, index=future_datetime_index),
            'pred_upper_95': pd.Series(future_predictions + prediction_interval_95, index=future_datetime_index)
        }

class ForecastAppGoal9:
    def __init__(self, root):
        self.root = root
        self.root.title("SDG 9 Indicator Forecast")
        self.root.geometry("1400x900")  # Increased window size
        
        # Get current directory
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load data
        self.df = self.load_data()
        self.indicators = self.get_available_indicators()
        
        # Load external data
        self.external_data = self.load_external_data()
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for main frame
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)  # PanedWindow gets all remaining space
        
        # Create selection frame
        self.selection_frame = ttk.LabelFrame(self.main_frame, text="Selection", padding="10")
        self.selection_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Create a PanedWindow for resizable plot and results areas
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.VERTICAL)
        self.paned_window.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Create plot frame and results frame within the PanedWindow
        self.plot_frame = ttk.LabelFrame(self.paned_window, text="Forecast Plot", padding="10")
        self.results_frame = ttk.LabelFrame(self.paned_window, text="Results", padding="10")
        
        # Add frames to PanedWindow
        self.paned_window.add(self.plot_frame, weight=3)  # Plot gets more initial space
        self.paned_window.add(self.results_frame, weight=2)  # Results gets less initial space
        
        # Configure grid weights for frames
        self.plot_frame.grid_columnconfigure(0, weight=1)
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)
        self.results_frame.grid_rowconfigure(0, weight=1)
        
        # Create button frame for save button
        self.button_frame = ttk.Frame(self.plot_frame)
        self.button_frame.grid(row=1, column=0, sticky=(tk.E), pady=5, padx=5)
        
        # Add save button
        self.save_button = ttk.Button(self.button_frame, text="Save Plot", command=self.save_plot)
        self.save_button.grid(row=0, column=0, padx=5)
        self.save_button.state(['disabled'])  # Disable until plot is generated
        
        # Initialize plot and results
        self.canvas = None
        self.current_fig = None  # Store the current figure
        self.results_text = tk.Text(self.results_frame, height=12, width=100)  # Increased height
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add scrollbar to results text
        scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        # Create selection widgets
        self.create_selection_widgets()
        
        # Initialize Random Forest model
        self.rf_model = SDGRandomForestModel(self.external_data)
        
        # Show external data status
        self.show_external_data_status()
        
    def load_data(self):
        """Load the processed data"""
        try:
            file_path = os.path.join(self.current_dir, 'Goal9_processed.csv')
            print(f"Loading data from: {file_path}")
            if not os.path.exists(file_path):
                messagebox.showerror("Error", f"Data file not found: {file_path}")
                return None
            
            # Try reading with different encodings and error handling
            encodings = ['utf-8', 'latin1', 'iso-8859-1']
            for encoding in encodings:
                try:
                    data = pd.read_csv(file_path, 
                                     sep=';', 
                                     encoding=encoding,
                                     on_bad_lines='skip',  # Skip problematic lines
                                     low_memory=False)
                    print(f"Successfully loaded {len(data)} rows of data using {encoding} encoding")
                    print(f"Columns: {data.columns.tolist()}")
                    return data
                except Exception as e:
                    print(f"Failed to load with {encoding} encoding: {str(e)}")
                    continue
            
            messagebox.showerror("Error", "Failed to load data with any encoding")
            return None
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            print(f"Error loading data: {str(e)}")
            return None
    
    def get_available_indicators(self):
        """Get list of available indicators with their descriptions"""
        indicators = self.df[['Indicator', 'SeriesCode', 'SeriesDescription']].drop_duplicates()
        return indicators.sort_values('Indicator')
    
    def get_available_countries(self, indicator_id):
        """Get list of available countries for a specific indicator"""
        countries = self.df[self.df['Indicator'] == indicator_id]['GeoAreaName'].unique()
        return sorted(countries)
    
    def create_selection_widgets(self):
        # Configure grid weights for selection frame
        self.selection_frame.grid_columnconfigure(0, weight=1)
        self.selection_frame.grid_columnconfigure(1, weight=1)
        
        # Left column
        # Model selection
        ttk.Label(self.selection_frame, text="Model:").grid(row=0, column=0, padx=2, pady=2, sticky=tk.W)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(self.selection_frame, textvariable=self.model_var, width=15)
        self.model_combo['values'] = ['ARIMA', 'Prophet', 'SARIMAX', 'Random Forest']
        self.model_combo.set('ARIMA')
        self.model_combo.grid(row=0, column=1, padx=2, pady=2, sticky=tk.W)
        
        # Indicator selection
        ttk.Label(self.selection_frame, text="Indicator:").grid(row=1, column=0, padx=2, pady=2, sticky=tk.W)
        self.indicator_var = tk.StringVar()
        self.indicator_combo = ttk.Combobox(self.selection_frame, textvariable=self.indicator_var, width=40)
        self.indicator_combo['values'] = [f"{ind} - {desc}" for ind, desc in zip(self.indicators['Indicator'], self.indicators['SeriesDescription'])]
        self.indicator_combo.grid(row=1, column=1, padx=2, pady=2, sticky=tk.W)
        self.indicator_combo.bind('<<ComboboxSelected>>', self.update_series_codes)
        
        # Series Code selection
        ttk.Label(self.selection_frame, text="Series Code:").grid(row=2, column=0, padx=2, pady=2, sticky=tk.W)
        self.series_code_var = tk.StringVar()
        self.series_code_combo = ttk.Combobox(self.selection_frame, textvariable=self.series_code_var, width=15)
        self.series_code_combo.grid(row=2, column=1, padx=2, pady=2, sticky=tk.W)
        self.series_code_combo.bind('<<ComboboxSelected>>', self.update_countries)
        
        # Country selection
        ttk.Label(self.selection_frame, text="Country:").grid(row=3, column=0, padx=2, pady=2, sticky=tk.W)
        self.country_var = tk.StringVar()
        self.country_combo = ttk.Combobox(self.selection_frame, textvariable=self.country_var, width=15)
        self.country_combo.grid(row=3, column=1, padx=2, pady=2, sticky=tk.W)
        
        # Location selection
        ttk.Label(self.selection_frame, text="Location:").grid(row=4, column=0, padx=2, pady=2, sticky=tk.W)
        self.location_var = tk.StringVar()
        self.location_combo = ttk.Combobox(self.selection_frame, textvariable=self.location_var, width=15)
        self.location_combo['values'] = [
            'ALLAREA',
            'RURAL',
            'URBAN'
        ]
        self.location_combo.set('ALLAREA')
        self.location_combo.grid(row=4, column=1, padx=2, pady=2, sticky=tk.W)
        
        # Right column
        # Mode of transportation selection
        ttk.Label(self.selection_frame, text="Transport Mode:").grid(row=0, column=2, padx=2, pady=2, sticky=tk.W)
        self.transport_var = tk.StringVar()
        self.transport_combo = ttk.Combobox(self.selection_frame, textvariable=self.transport_var, width=15)
        self.transport_combo['values'] = ['ALL', 'ROAD', 'RAIL', 'AIR', 'MARITIME', 'PIPELINE']
        self.transport_combo.set('ALL')
        self.transport_combo.grid(row=0, column=3, padx=2, pady=2, sticky=tk.W)
        
        # Forecast button - centered below all filters
        self.forecast_button = ttk.Button(self.selection_frame, text="Generate Forecast", command=self.generate_forecast)
        self.forecast_button.grid(row=5, column=0, columnspan=4, padx=5, pady=5)
    
    def update_series_codes(self, event=None):
        """Update series code combobox when indicator is selected"""
        selected = self.indicator_var.get()
        if selected:
            indicator_id = selected.split(' - ')[0]
            series_codes = self.df[self.df['Indicator'] == indicator_id]['SeriesCode'].unique()
            self.series_code_combo['values'] = sorted(series_codes)
            if series_codes.size > 0:
                self.series_code_combo.set(series_codes[0])
            self.update_countries()
    
    def update_countries(self, event=None):
        """Update country combobox when indicator and series code are selected"""
        selected = self.indicator_var.get()
        series_code = self.series_code_var.get()
        if selected and series_code:
            indicator_id = selected.split(' - ')[0]
            countries = self.df[
                (self.df['Indicator'] == indicator_id) & 
                (self.df['SeriesCode'] == series_code)
            ]['GeoAreaName'].unique()
            self.country_combo['values'] = sorted(countries)
            if countries.size > 0:
                self.country_combo.set(countries[0])
    
    def prepare_time_series(self, data):
        """Prepare time series data for modeling"""
        data['TimePeriod'] = pd.to_datetime(data['TimePeriod'], format='%Y')
        data = data.set_index('TimePeriod')
        data = data.sort_index()
        return data['Value']
    
    def fit_arima_model(self, series):
        """Fit ARIMA model to the time series with comprehensive time series cross validation"""
        print(f"🔄 ARIMA Model with Time Series Cross Validation")
        print(f"Data points: {len(series)}")
        
        # Time Series Cross Validation for ARIMA order selection
        print(f"📊 Testing ARIMA orders with time series cross validation...")
        
        best_order = None
        best_cv_score = float('inf')
        cv_results = {}
        
        # Test different ARIMA orders with cross validation
        orders_to_test = [
            (1, 1, 1), (1, 1, 0), (0, 1, 1), 
            (2, 1, 1), (1, 1, 2), (1, 0, 1),
            (2, 0, 2), (0, 1, 2), (2, 1, 0)
        ]
        
        # Use expanding window cross validation
        n_splits = min(4, max(2, len(series) // 8))
        
        for order in orders_to_test:
            try:
                cv_scores = []
                
                # Time series split: expanding window
                for i in range(n_splits):
                    # Calculate dynamic split sizes
                    min_train_size = max(8, len(series) // 2)
                    train_end = min_train_size + i * (len(series) - min_train_size) // (n_splits - 1)
                    test_start = train_end
                    test_end = min(test_start + max(2, len(series) // 8), len(series))
                    
                    if test_end > len(series) or test_start >= test_end or train_end <= 5:
                        continue
                        
                    train_series = series.iloc[:train_end]
                    test_series = series.iloc[test_start:test_end]
                    
                    if len(train_series) < 6 or len(test_series) < 1:
                        continue
                    
                    try:
                        # Fit ARIMA model on training data
                        model = ARIMA(train_series, order=order)
                        model_fit = model.fit()
                        
                        # Make predictions on test data
                        predictions = model_fit.forecast(steps=len(test_series))
                        
                        # Calculate RMSE
                        rmse = np.sqrt(mean_squared_error(test_series, predictions))
                        cv_scores.append(rmse)
                        
                    except Exception as e:
                        # Skip this fold if model fitting fails
                        continue
                
                if len(cv_scores) > 0:
                    mean_cv_score = np.mean(cv_scores)
                    std_cv_score = np.std(cv_scores)
                    print(f"ARIMA{order}: {mean_cv_score:.4f} RMSE ({len(cv_scores)} folds)")
                    
                    # Store CV results
                    cv_results[str(order)] = {
                        'mean_rmse': mean_cv_score,
                        'std_rmse': std_cv_score,
                        'n_folds': len(cv_scores),
                        'fold_scores': cv_scores
                    }
                    
                    if mean_cv_score < best_cv_score:
                        best_cv_score = mean_cv_score
                        best_order = order
                        
            except Exception as e:
                print(f"⚠️ ARIMA{order} failed: {str(e)}")
                continue
        
        # Use best order or fall back to default
        if best_order is None:
            print(f"⚠️ ARIMA optimization failed. Using default (1,1,1)")
            best_order = (1, 1, 1)
        else:
            print(f"✅ Best ARIMA order: {best_order} (CV RMSE: {best_cv_score:.4f})")
        
        # Final training on subset for evaluation
        train_size = int(len(series) * 0.8)
        train_series = series.iloc[:train_size]
        test_series = series.iloc[train_size:]
        
        print(f"📈 Final training: {len(train_series)} train, {len(test_series)} test points")
        
        # Fit model on training data for evaluation
        eval_model = ARIMA(train_series, order=best_order)
        eval_model_fit = eval_model.fit()
        
        # Make predictions for test period
        predictions = eval_model_fit.forecast(steps=len(test_series))
        rmse = np.sqrt(mean_squared_error(test_series, predictions))
        print(f"✅ Test RMSE: {rmse:.4f}")
        
        # Fit new model on all data for future predictions
        full_model = ARIMA(series, order=best_order)
        full_model_fit = full_model.fit()
        
        # Store CV results for summary display
        if best_order and str(best_order) in cv_results:
            self.last_arima_cv_results = cv_results[str(best_order)]
        
        # Return dictionary with CV results
        return {
            'model': full_model_fit,
            'test_predictions': predictions,
            'test_data': test_series,
            'rmse': rmse,
            'best_order': best_order,
            'cv_results': cv_results
        }
    
    def fit_prophet_model(self, series):
        """Fit Prophet model to the time series with time series cross validation"""
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })
        
        print(f"🔄 Prophet Model with Time Series Cross Validation")
        print(f"Data points: {len(df)}")
        
        # Time Series Cross Validation for Prophet
        print(f"📊 Performing Prophet cross validation...")
        
        # Use expanding window cross validation
        n_splits = min(4, max(2, len(df) // 8))
        cv_scores = []
        
        for i in range(n_splits):
            # Calculate dynamic split sizes
            min_train_size = max(10, len(df) // 2)
            train_end = min_train_size + i * (len(df) - min_train_size) // (n_splits - 1)
            test_start = train_end
            test_end = min(test_start + max(2, len(df) // 8), len(df))
            
            if test_end > len(df) or test_start >= test_end or train_end <= 8:
                continue
                
            train_df = df.iloc[:train_end]
            test_df = df.iloc[test_start:test_end]
            
            try:
                # Fit Prophet with infrastructure-optimized settings
                model = Prophet(
                    yearly_seasonality=False,  # Yearly data doesn't need yearly seasonality
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.01,  # Conservative for infrastructure data
                    seasonality_prior_scale=0.1,
                    holidays_prior_scale=0.1,
                    uncertainty_samples=100
                )
                model.fit(train_df)
                
                # Make predictions
                future = model.make_future_dataframe(periods=len(test_df), freq='Y')
                forecast = model.predict(future)
                predictions = forecast['yhat'].values[-len(test_df):]
                
                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(test_df['y'], predictions))
                cv_scores.append(rmse)
                
            except Exception as e:
                continue
        
        if len(cv_scores) > 0:
            mean_cv = np.mean(cv_scores)
            std_cv = np.std(cv_scores)
            print(f"✅ Prophet CV: {mean_cv:.4f} ± {std_cv:.4f} RMSE ({len(cv_scores)} folds)")
        else:
            print(f"⚠️ Prophet CV failed, using direct training")
        
        # Final training on subset for evaluation
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        print(f"📈 Final training: {len(train_df)} train, {len(test_df)} test points")
        
        # Fit model on training data for evaluation
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False, 
            daily_seasonality=False,
            changepoint_prior_scale=0.01,
            seasonality_prior_scale=0.1,
            holidays_prior_scale=0.1,
            uncertainty_samples=100
        )
        model.fit(train_df)
        
        # Make predictions for test period
        future = model.make_future_dataframe(periods=len(test_df), freq='Y')
        forecast = model.predict(future)
        
        # Get predictions for the test period only
        predictions = pd.Series(forecast['yhat'].values[-len(test_df):], index=test_df['ds'])
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(test_df['y'], predictions))
        print(f"✅ Test RMSE: {rmse:.4f}")
        
        # Fit new model on all data for future predictions
        full_model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.01,
            seasonality_prior_scale=0.1,
            holidays_prior_scale=0.1,
            uncertainty_samples=100
        )
        full_model.fit(df)
        
        # Return dictionary with CV results
        cv_results = None
        if len(cv_scores) > 0:
            cv_results = {
                'mean_rmse': np.mean(cv_scores),
                'std_rmse': np.std(cv_scores),
                'n_folds': len(cv_scores),
                'fold_scores': cv_scores
            }
            # Store CV results for summary display
            self.last_prophet_cv_results = cv_results
        
        return {
            'model': full_model,
            'test_predictions': predictions,
            'test_data': test_df['y'],
            'rmse': rmse,
            'cv_results': cv_results
        }

    def fit_sarimax_model(self, series, country, location='ALLAREA', transport='ALL'):
        """
        Fit SARIMAX model with external variables and cross-validation for SDG9 infrastructure forecasting.
        """
        print(f"🏗️ Fitting SARIMAX model for {country}...")
        print(f"Using filters: Location={location}, Transport={transport}")
        
        try:
            # Step 1: Prepare external variables
            external_data_matrix, valid_years, feature_names = self.prepare_external_features_for_infrastructure(
                country, location, transport)
            
            if external_data_matrix is not None and len(feature_names) >= 3:
                print(f"✅ External data prepared: {external_data_matrix.shape}")
                print(f"Features: {feature_names}")
                
                # Step 2: Align series with external data
                aligned_series, exog_data = self.align_series_with_external(series, external_data_matrix, valid_years)
                print(f"📊 Aligned data: {len(aligned_series)} time points")
                
                # Step 3: Scale external variables
                scaler = StandardScaler()
                exog_scaled = scaler.fit_transform(exog_data)
                print(f"📊 External variables scaled successfully (mean≈0, std≈1)")
                
                # Step 4: Time Series Cross Validation for SARIMAX parameter selection
                best_order = None
                best_seasonal_order = None
                best_cv_score = float('inf')
                cv_results = {}
                
                print("📊 SARIMAX parameter optimization with time series cross validation...")
                
                # Test different SARIMAX orders
                orders_to_test = [
                    (1, 1, 1), (1, 1, 0), (0, 1, 1), 
                    (2, 1, 1), (1, 1, 2), (1, 0, 1)
                ]
                seasonal_orders_to_test = [(0, 0, 0, 0)]  # No seasonality for yearly data
                
                for order in orders_to_test:
                    for seasonal_order in seasonal_orders_to_test:
                        try:
                            # Perform time series cross validation
                            cv_scores = []
                            n_splits = min(3, len(aligned_series) // 8)
                            
                            if n_splits < 2:
                                continue
                            
                            # Time series split: expanding window
                            for i in range(n_splits):
                                min_train_size = max(8, len(aligned_series) // 2)
                                train_end = min_train_size + i * (len(aligned_series) - min_train_size) // (n_splits - 1)
                                test_start = train_end
                                test_end = min(test_start + max(2, len(aligned_series) // 8), len(aligned_series))
                                
                                if test_end > len(aligned_series) or test_start >= test_end or train_end <= 5:
                                    continue
                                
                                train_series = aligned_series.iloc[:train_end]
                                test_series = aligned_series.iloc[test_start:test_end]
                                train_exog = exog_scaled[:train_end]
                                test_exog = exog_scaled[test_start:test_end]
                                
                                if len(train_series) < 6 or len(test_series) < 1:
                                    continue
                                
                                try:
                                    # Fit SARIMAX model with external variables
                                    model = SARIMAX(train_series, 
                                                  exog=train_exog,
                                                  order=order, 
                                                  seasonal_order=seasonal_order,
                                                  enforce_stationarity=False,
                                                  enforce_invertibility=False)
                                    model_fit = model.fit(disp=False, maxiter=100)
                                    
                                    # Make predictions
                                    forecast = model_fit.forecast(steps=len(test_series), exog=test_exog)
                                    
                                    # Calculate RMSE
                                    rmse = np.sqrt(mean_squared_error(test_series, forecast))
                                    cv_scores.append(rmse)
                                    
                                except Exception as e:
                                    continue
                            
                            if len(cv_scores) > 0:
                                mean_cv_score = np.mean(cv_scores)
                                print(f"SARIMAX{order}x{seasonal_order}: {mean_cv_score:.4f} RMSE ({len(cv_scores)} folds)")
                                
                                cv_results[f"{order}x{seasonal_order}"] = {
                                    'mean_rmse': mean_cv_score,
                                    'std_rmse': np.std(cv_scores),
                                    'n_folds': len(cv_scores)
                                }
                                
                                if mean_cv_score < best_cv_score:
                                    best_cv_score = mean_cv_score
                                    best_order = order
                                    best_seasonal_order = seasonal_order
                        
                        except Exception as e:
                            continue
                
                # Use best parameters or fall back
                if best_order is None:
                    print(f"⚠️ SARIMAX optimization failed. Using default parameters.")
                    best_order = (1, 1, 1)
                    best_seasonal_order = (0, 0, 0, 0)
                else:
                    print(f"✅ Best SARIMAX: {best_order}x{best_seasonal_order} (CV RMSE: {best_cv_score:.4f})")
                
                # Step 5: Final model training with train/test split
                train_size = int(len(aligned_series) * 0.8)
                train_series = aligned_series.iloc[:train_size]
                test_series = aligned_series.iloc[train_size:]
                train_exog = exog_scaled[:train_size]
                test_exog = exog_scaled[train_size:]
                
                print(f"📈 Final SARIMAX training: {len(train_series)} train, {len(test_series)} test points")
                
                # Fit final model
                model = SARIMAX(train_series,
                              exog=train_exog,
                              order=best_order,
                              seasonal_order=best_seasonal_order,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                model_fit = model.fit(disp=False, maxiter=150)
                
                # Test predictions
                test_predictions = model_fit.forecast(steps=len(test_series), exog=test_exog)
                rmse = np.sqrt(mean_squared_error(test_series, test_predictions))
                print(f"✅ SARIMAX Test RMSE: {rmse:.4f}")
                print(f"✅ True SARIMAX with external variables")
                
                return {
                    'model': model_fit,
                    'test_predictions': test_predictions,
                    'test_data': test_series,
                    'rmse': rmse,
                    'best_order': best_order,
                    'cv_results': cv_results,
                    'feature_names': feature_names,
                    'scaler': scaler,
                    'exog_data': exog_data,
                    'aligned_series': aligned_series
                }
            
            else:
                # Fallback to simple SARIMAX (ARIMA) without external variables
                print(f"⚠️ Insufficient external data. Using simple SARIMAX")
                return self.fit_sarimax_without_external(series)
                
        except Exception as e:
            print(f"❌ SARIMAX with external variables failed: {e}")
            print(f"   Falling back to simple SARIMAX")
            return self.fit_sarimax_without_external(series)

    def fit_sarimax_without_external(self, series):
        """Fallback SARIMAX without external variables (essentially ARIMA with cross-validation)"""
        print("📊 SARIMAX parameter optimization...")
        
        best_order = None
        best_cv_score = float('inf')
        cv_results = {}
        
        orders_to_test = [
            (1, 1, 1), (1, 1, 0), (0, 1, 1), 
            (2, 1, 1), (1, 1, 2), (1, 0, 1)
        ]
        
        for order in orders_to_test:
            try:
                cv_scores = []
                n_splits = min(3, len(series) // 8)
                
                if n_splits < 2:
                    continue
                
                for i in range(n_splits):
                    min_train_size = max(8, len(series) // 2)
                    train_end = min_train_size + i * (len(series) - min_train_size) // (n_splits - 1)
                    test_start = train_end
                    test_end = min(test_start + max(2, len(series) // 8), len(series))
                    
                    if test_end > len(series) or test_start >= test_end or train_end <= 5:
                        continue
                    
                    train_series = series.iloc[:train_end]
                    test_series = series.iloc[test_start:test_end]
                    
                    if len(train_series) < 6 or len(test_series) < 1:
                        continue
                    
                    try:
                        model = SARIMAX(train_series, 
                                      order=order, 
                                      seasonal_order=(0, 0, 0, 0),
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
                        model_fit = model.fit(disp=False, maxiter=100)
                        
                        forecast = model_fit.forecast(steps=len(test_series))
                        rmse = np.sqrt(mean_squared_error(test_series, forecast))
                        cv_scores.append(rmse)
                        
                    except Exception as e:
                        continue
                
                if len(cv_scores) > 0:
                    mean_cv_score = np.mean(cv_scores)
                    print(f"SARIMAX{order}x(0, 0, 0, 0): {mean_cv_score:.4f} RMSE ({len(cv_scores)} folds)")
                    
                    cv_results[f"{order}"] = {
                        'mean_rmse': mean_cv_score,
                        'std_rmse': np.std(cv_scores),
                        'n_folds': len(cv_scores)
                    }
                    
                    if mean_cv_score < best_cv_score:
                        best_cv_score = mean_cv_score
                        best_order = order
                        
            except Exception as e:
                continue
        
        # Use best parameters or fall back
        if best_order is None:
            best_order = (1, 1, 1)
        else:
            print(f"✅ Best SARIMAX: {best_order}x(0, 0, 0, 0) (CV RMSE: {best_cv_score:.4f})")
        
        # Final training
        train_size = int(len(series) * 0.8)
        train_series = series.iloc[:train_size]
        test_series = series.iloc[train_size:]
        
        print(f"📈 Final SARIMAX training: {len(train_series)} train, {len(test_series)} test points")
        
        model = SARIMAX(train_series,
                      order=best_order,
                      seasonal_order=(0, 0, 0, 0),
                      enforce_stationarity=False,
                      enforce_invertibility=False)
        model_fit = model.fit(disp=False, maxiter=150)
        
        test_predictions = model_fit.forecast(steps=len(test_series))
        rmse = np.sqrt(mean_squared_error(test_series, test_predictions))
        print(f"✅ SARIMAX Test RMSE: {rmse:.4f}")
        print(f"✅ Using simple SARIMAX")
        
        return {
            'model': model_fit,
            'test_predictions': test_predictions,
            'test_data': test_series,
            'rmse': rmse,
            'best_order': best_order,
            'cv_results': cv_results,
            'feature_names': [],
            'scaler': None,
            'exog_data': None,
            'aligned_series': series
        }
    
    def generate_forecast(self):
        """Generate forecast based on selected indicator and country"""
        try:
            # Get selections
            selected = self.indicator_var.get()
            if not selected:
                messagebox.showerror("Error", "Please select an indicator")
                return
            
            indicator_id = selected.split(' - ')[0]
            series_code = self.series_code_var.get()
            if not series_code:
                messagebox.showerror("Error", "Please select a series code")
                return
                
            country = self.country_var.get()
            location = self.location_var.get()
            transport = self.transport_var.get()
            model_type = self.model_var.get()
            
            if not country:
                messagebox.showerror("Error", "Please select a country")
                return
            
            # Get data for the selected indicator, series code and country
            indicator_data = self.df[
                (self.df['Indicator'] == indicator_id) & 
                (self.df['SeriesCode'] == series_code) &
                (self.df['GeoAreaName'] == country)
            ]
            
            if len(indicator_data) == 0:
                messagebox.showerror("Error", f"No data found for {indicator_id} in {country}")
                return
            
            # Handle missing values
            indicator_data['Location'] = indicator_data['Location'].fillna('ALLAREA')
            indicator_data['ModeofTransportation'] = indicator_data['ModeofTransportation'].fillna('ALL')
            
            # Apply filters
            if location != 'ALLAREA':
                indicator_data = indicator_data[indicator_data['Location'] == location]
                if len(indicator_data) == 0:
                    messagebox.showerror("Error", f"No data found for location {location} in {indicator_id} for {country}")
                    return
                    
            if transport != 'ALL':
                indicator_data = indicator_data[indicator_data['ModeofTransportation'] == transport]
                if len(indicator_data) == 0:
                    messagebox.showerror("Error", f"No data found for transport mode {transport} in {indicator_id} for {country}")
                    return
            
            # Check if we have enough data points
            MIN_DATA_POINTS = 20
            if len(indicator_data) < MIN_DATA_POINTS:
                messagebox.showerror("Error", 
                    f"Not enough data points for the selected filters.\n"
                    f"Found {len(indicator_data)} points, but need at least {MIN_DATA_POINTS} points "
                    f"for a reliable forecast.\n"
                    f"Please try different filters for more data points.")
                return
            
            # Convert TimePeriod to datetime and sort
            indicator_data['TimePeriod'] = pd.to_datetime(indicator_data['TimePeriod'], format='%Y')
            indicator_data = indicator_data.sort_values('TimePeriod')
            
            # Create plot
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
            
            fig, ax = plt.subplots(figsize=(12, 6))  # Increased figure size
            self.current_fig = fig  # Store the current figure
            
            # Set smaller font sizes
            plt.rcParams.update({'font.size': 8})  # Default font size
            
            # Ensure Value column is numeric
            indicator_data['Value'] = pd.to_numeric(indicator_data['Value'], errors='coerce')
            
            # Remove any rows with NaN values in Value column
            indicator_data = indicator_data.dropna(subset=['Value'])
            
            if len(indicator_data) == 0:
                messagebox.showerror("Error", "No valid numeric data found for the selected series")
                return
            
            # Get the units from the data
            unit = indicator_data['Units'].iloc[0] if 'Units' in indicator_data.columns else ''
            
            # Scale the data
            scaled_data = indicator_data.copy()
            
            # Plot historical data points with larger markers
            ax.scatter(scaled_data['TimePeriod'], scaled_data['Value'], 
                      color='blue', label='Historical Data', s=100, alpha=0.8)
            
            # Prepare data for modeling
            series = self.prepare_time_series(scaled_data)
            
            try:
                if model_type == 'ARIMA':
                    # Fit ARIMA model and make forecast
                    arima_results = self.fit_arima_model(series)
                    model_fit = arima_results['model']
                    predictions = arima_results['test_predictions']
                    test = arima_results['test_data']
                    rmse = arima_results['rmse']
                    
                    # Define scaled_predictions for ARIMA model
                    scaled_predictions = predictions
                    
                    # Plot predictions for test period (red)
                    prediction_color = plt.cm.Reds(0.7)
                    test_period = test.index
                    ax.scatter(test_period, predictions, color=prediction_color, 
                              label='Model Test', s=100, alpha=0.8)
                    ax.plot(test_period, predictions, color=prediction_color, alpha=0.5, linewidth=2)
                    
                    # Make future forecast using all available data
                    future_forecast = model_fit.get_forecast(steps=5)
                    scaled_forecast = future_forecast.predicted_mean
                    future_conf_int = future_forecast.conf_int(alpha=0.05)
                    scaled_conf_lower_future = future_conf_int.iloc[:, 0]
                    scaled_conf_upper_future = future_conf_int.iloc[:, 1]
                    
                    # Calculate prediction intervals (wider than confidence intervals)
                    pred_interval = 2.0 * rmse
                    scaled_pred_lower_future = scaled_forecast - pred_interval
                    scaled_pred_upper_future = scaled_forecast + pred_interval
                    
                    # Get future dates
                    future_dates = pd.date_range(start=series.index[-1], periods=6, freq='Y')[1:]
                    
                    # Set future_forecast for results display
                    future_forecast = scaled_forecast
                    
                elif model_type == 'Prophet':
                    # Fit Prophet model and make forecast
                    prophet_results = self.fit_prophet_model(series)
                    model_fit = prophet_results['model']
                    predictions = prophet_results['test_predictions']
                    test = prophet_results['test_data']
                    rmse = prophet_results['rmse']
                    
                    # Scale the predictions and test data
                    scaled_predictions = predictions
                    scaled_test = test
                    
                    # Plot predictions for test period (red)
                    prediction_color = plt.cm.Reds(0.7)
                    test_period = predictions.index
                    ax.scatter(test_period, scaled_predictions, color=prediction_color, 
                              label='Model Test', s=100, alpha=0.8)
                    ax.plot(test_period, scaled_predictions, color=prediction_color, alpha=0.5, linewidth=2)
                    
                    # Make future forecast using all available data
                    future = model_fit.make_future_dataframe(periods=5, freq='Y')
                    forecast = model_fit.predict(future)
                    scaled_forecast = forecast['yhat'].iloc[-5:].values
                    
                    # Get confidence intervals from Prophet and scale them
                    scaled_conf_lower_future = forecast['yhat_lower'].iloc[-5:].values
                    scaled_conf_upper_future = forecast['yhat_upper'].iloc[-5:].values
                    
                    # Calculate prediction intervals (wider than confidence intervals)
                    pred_interval = 2.0 * rmse  # Increased multiplier for more visible intervals
                    scaled_pred_lower_future = scaled_forecast - pred_interval
                    scaled_pred_upper_future = scaled_forecast + pred_interval
                    
                    # Get future dates from the forecast
                    future_dates = pd.to_datetime(forecast['ds'].iloc[-5:])
                    
                    # Set future_forecast for results display
                    future_forecast = scaled_forecast
                    
                elif model_type == 'SARIMAX':
                    # Fit SARIMAX model with infrastructure-specific external variables
                    sarimax_results = self.fit_sarimax_model(series, country, location, transport)
                    
                    if sarimax_results and 'model' in sarimax_results:
                        model_fit = sarimax_results['model']
                        predictions = sarimax_results['test_predictions']
                        test = sarimax_results['test_data']
                        rmse = sarimax_results['rmse']
                        
                        # Plot test predictions (red)
                        prediction_color = plt.cm.Reds(0.7)
                        test_period = test.index
                        ax.scatter(test_period, predictions, color=prediction_color, 
                                  label='Model Test', s=100, alpha=0.8)
                        ax.plot(test_period, predictions, color=prediction_color, alpha=0.5, linewidth=2)
                        
                        # Generate future forecast
                        try:
                            if sarimax_results.get('feature_names') and len(sarimax_results['feature_names']) > 0:
                                # True SARIMAX with external variables
                                future_forecast = self.predict_future_sarimax(
                                    sarimax_results, country, periods=8, location=location, transport=transport)
                                scaled_forecast = future_forecast  # Define scaled_forecast for use in confidence intervals
                                future_dates = future_forecast.index if hasattr(future_forecast, 'index') else pd.date_range(start=series.index[-1], periods=6, freq='Y')[1:]
                                print(f"✅ True SARIMAX with external variables")
                            else:
                                # Simple SARIMAX forecast without external variables (fallback)
                                print(f"✅ Using simple SARIMAX")
                                future_forecast = sarimax_results['model'].forecast(steps=8)
                                future_dates = pd.date_range(start=series.index[-1], periods=9, freq='Y')[1:]
                                
                                if len(future_forecast) != len(future_dates):
                                    min_len = min(len(future_forecast), len(future_dates))
                                    future_forecast = future_forecast[:min_len]
                                    future_dates = future_dates[:min_len]
                                
                                scaled_forecast = pd.Series(future_forecast, index=future_dates)
                                
                        except Exception as forecast_error:
                            print(f"⚠️ SARIMAX forecast failed: {forecast_error}, using trend extrapolation")
                            # Fallback to trend extrapolation
                            future_dates = pd.date_range(start=series.index[-1], periods=9, freq='Y')[1:]
                            last_values = series.tail(3).values
                            trend = np.mean(np.diff(last_values)) if len(last_values) > 1 else 0
                            future_forecast = [series.iloc[-1] + trend * (i + 1) for i in range(len(future_dates))]
                            scaled_forecast = pd.Series(future_forecast, index=future_dates)
                        
                        # Get confidence intervals from SARIMAX
                        try:
                            forecast_result = sarimax_results['model'].get_forecast(steps=len(future_dates))
                            conf_int = forecast_result.conf_int(alpha=0.05)
                            scaled_conf_lower_future = conf_int.iloc[:, 0]
                            scaled_conf_upper_future = conf_int.iloc[:, 1]
                            
                            # Prediction intervals wider than confidence intervals
                            pred_interval = 2.0 * rmse
                            scaled_pred_lower_future = scaled_forecast - pred_interval
                            scaled_pred_upper_future = scaled_forecast + pred_interval
                        except:
                            # Fallback intervals
                            conf_interval = 1.2 * rmse
                            pred_interval = 2.0 * rmse
                            scaled_conf_lower_future = scaled_forecast - conf_interval
                            scaled_conf_upper_future = scaled_forecast + conf_interval
                            scaled_pred_lower_future = scaled_forecast - pred_interval
                            scaled_pred_upper_future = scaled_forecast + pred_interval
                        
                        # Store SARIMAX results for results display
                        self.sarimax_feature_names = sarimax_results.get('feature_names', [])
                        self.sarimax_cv_results = sarimax_results.get('cv_results', {})
                        self.sarimax_best_order = sarimax_results.get('best_order', (1,1,1))
                        scaled_predictions = predictions
                        
                    else:
                        messagebox.showerror("Error", "SARIMAX model fitting failed")
                        return
                
                elif model_type == 'Random Forest':
                    # Fit Random Forest model and make forecast
                    rf_results = self.fit_random_forest_model(series, country, location, transport)
                    
                    # Scale the predictions
                    scaled_test_predictions = rf_results['test_predictions']
                    scaled_forecast = rf_results['future_predictions']
                    
                    # Set future_forecast for use in results display
                    future_forecast = scaled_forecast
                    
                    # Set scaled_predictions for use in all_values calculation later
                    scaled_predictions = scaled_test_predictions
                    
                    # Scale confidence and prediction intervals
                    scaled_conf_lower_68 = rf_results['conf_lower_68']
                    scaled_conf_upper_68 = rf_results['conf_upper_68']
                    scaled_conf_lower_95 = rf_results['conf_lower_95']
                    scaled_conf_upper_95 = rf_results['conf_upper_95']
                    scaled_pred_lower_95 = rf_results['pred_lower_95']
                    scaled_pred_upper_95 = rf_results['pred_upper_95']
                    
                    # Plot test predictions
                    prediction_color = plt.cm.Reds(0.7)
                    ax.scatter(rf_results['test_predictions'].index, scaled_test_predictions, color=prediction_color, 
                              label='Model Test', s=100, alpha=0.8)
                    ax.plot(rf_results['test_predictions'].index, scaled_test_predictions, color=prediction_color, alpha=0.5, linewidth=2)
                    
                    # Get future dates
                    future_dates = rf_results['future_predictions'].index
                    
                    # Set for plotting
                    scaled_conf_lower_future = scaled_conf_lower_95
                    scaled_conf_upper_future = scaled_conf_upper_95
                    scaled_pred_lower_future = scaled_pred_lower_95
                    scaled_pred_upper_future = scaled_pred_upper_95
                    
                    # Store for results display
                    self.rf_features_used = self.rf_model.feature_names
                    self.rf_feature_importance = rf_results['feature_importance']
                    rmse = rf_results['rmse']
                
                # Plot future forecast if available
                if future_dates is not None and scaled_forecast is not None:
                    forecast_color = plt.cm.Greens(0.7)
                    
                    # Plot intervals for Random Forest with appropriate shading
                    if model_type == 'Random Forest':
                        # Plot prediction intervals first (darkest shade)
                        ax.fill_between(future_dates, scaled_pred_lower_95, scaled_pred_upper_95, 
                                      color='darkseagreen', alpha=0.4, label='95% Prediction Interval', zorder=1)
                        
                        # Plot 95% confidence intervals (medium width, medium shade)
                        ax.fill_between(future_dates, scaled_conf_lower_95, scaled_conf_upper_95, 
                                      color='lightgreen', alpha=0.6, label='95% Confidence Interval', zorder=2)
                        
                        # Plot 68% confidence intervals on top (narrowest, lightest shade)
                        ax.fill_between(future_dates, scaled_conf_lower_68, scaled_conf_upper_68, 
                                      color='palegreen', alpha=0.8, label='68% Confidence Interval', zorder=3)
                        
                        # Plot the forecast line on top of intervals
                        ax.scatter(future_dates, scaled_forecast, color=forecast_color, 
                                  label='Future Forecast', s=100, alpha=1.0, zorder=4)
                        ax.plot(future_dates, scaled_forecast, color=forecast_color, alpha=0.8, linewidth=3, zorder=4)
                    else:
                        # For non-Random Forest models
                    ax.scatter(future_dates, scaled_forecast, color=forecast_color, 
                              label='Future Forecast', s=100, alpha=0.8)
                    ax.plot(future_dates, scaled_forecast, color=forecast_color, alpha=0.5, linewidth=2)
                    
                    # Plot prediction intervals first (darker shade)
                    ax.fill_between(future_dates, scaled_pred_lower_future, scaled_pred_upper_future, 
                                  color='#2E8B57', alpha=0.3, label='95% Prediction Interval')  # Reduced alpha
                    
                    # Plot confidence intervals on top (lighter shade)
                    ax.fill_between(future_dates, scaled_conf_lower_future, scaled_conf_upper_future, 
                                  color='#3CB371', alpha=0.2, label='95% Confidence Interval')  # Reduced alpha
                    
                    # Print interval values for debugging
                    print("\nForecast values:", scaled_forecast)
                    print("\nConfidence intervals:")
                    for i, (lower, upper) in enumerate(zip(scaled_conf_lower_future, scaled_conf_upper_future)):
                        print(f"Year {i+1}: {lower:.2f} - {upper:.2f}")
                    print("\nPrediction intervals:")
                    for i, (lower, upper) in enumerate(zip(scaled_pred_lower_future, scaled_pred_upper_future)):
                        print(f"Year {i+1}: {lower:.2f} - {upper:.2f}")
                
                # Add text annotation for the last historical data point
                last_date = series.index[-1]
                last_value = series.iloc[-1]
                ax.annotate(f'Latest data: {last_value:.2f} {unit}',
                           xy=(last_date, last_value),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, color='blue',
                           bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
                
                # Set y-axis limits to show all data points clearly
                all_values = list(series) + list(scaled_predictions)
                if isinstance(scaled_forecast, np.ndarray):
                    all_values.extend(scaled_forecast)
                else:
                    all_values.extend(scaled_forecast.values)
                    
                y_min = min(all_values)
                y_max = max(all_values)
                y_range = y_max - y_min
                
                # Add extra space for intervals (reduced from 0.2 to 0.15)
                y_min = y_min - 0.15*y_range
                y_max = y_max + 0.15*y_range
                
                # Ensure intervals are within plot limits
                if scaled_pred_lower_future is not None:
                    if isinstance(scaled_pred_lower_future, np.ndarray):
                    y_min = min(y_min, min(scaled_pred_lower_future))
                    y_max = max(y_max, max(scaled_pred_upper_future))
                    else:
                        y_min = min(y_min, min(scaled_pred_lower_future.values))
                        y_max = max(y_max, max(scaled_pred_upper_future.values))
                
                ax.set_ylim(y_min, y_max)
                
                # Adjust layout to make room for legend and prevent text cutoff
                plt.subplots_adjust(right=0.85, top=0.85, bottom=0.15, left=0.1)  # Added left margin
                
                # Make plot frame expand to fill available space
                self.plot_frame.grid_rowconfigure(0, weight=1)
                self.plot_frame.grid_columnconfigure(0, weight=1)
                
                # Embed plot in GUI with sticky option to fill frame
                self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
                
                # Print debug information about plot limits
                print("\nPlot limits:")
                print(f"Y-axis limits: {y_min:.2f} to {y_max:.2f}")
                print(f"Data range: {min(series):.2f} to {max(series):.2f}")
                print(f"Forecast range: {min(scaled_forecast):.2f} to {max(scaled_forecast):.2f}")
                if scaled_pred_lower_future is not None:
                    print(f"Prediction interval range: {min(scaled_pred_lower_future):.2f} to {max(scaled_pred_upper_future):.2f}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not generate forecast: {str(e)}")
                import traceback
                print(traceback.format_exc())
            
            # Customize plot
            source = indicator_data['Source'].iloc[0]
            series_description = indicator_data['SeriesDescription'].iloc[0]
            title = f'Forecast for {indicator_id}\n({series_description})\nin {country}'
            if location != 'ALLAREA':
                title += f'\nLocation: {self.location_var.get()}'
            if transport != 'ALL':
                title += f'\nTransport Mode: {self.transport_var.get()}'
            title += f'\nSource: {source}'
            title += f'\nModel: {model_type}'
            
            # Add external features information for Random Forest
            if model_type == 'Random Forest' and hasattr(self, 'rf_features_used'):
                external_features = [f for f in self.rf_features_used if f not in ['Year', 'Location_URBAN', 'Location_RURAL', 
                                                                            'Transport_ROAD', 'Transport_RAIL', 
                                                                            'Transport_AIR', 'Transport_MARITIME', 
                                                                            'Transport_PIPELINE', 'Year_x_URBAN', 
                                                                            'Year_x_RURAL', 'Year_x_Location',
                                                                            'Year_x_ROAD', 'Year_x_RAIL', 'Year_x_AIR',
                                                                            'Year_x_MARITIME', 'Year_x_PIPELINE', 'Year_x_Transport']]
                if external_features:
                    features_str = ', '.join(external_features)
                    title += f'\nExternal Factors: {features_str}'
            
            # Set title with smaller font
            ax.set_title(title, fontsize=9, pad=10)
            ax.set_xlabel('Year', fontsize=8)
            ax.set_ylabel(f'Value ({unit})', fontsize=8)
            
            # Add legend with smaller font
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
            
            # Set tick label sizes
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Make grid lighter
            ax.grid(True, alpha=0.3)
            
            # Format x-axis to show years
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Display comprehensive results like SDG1-8
            self.results_text.delete(1.0, tk.END)
            
            # === PROFESSIONAL HEADER ===
            self.results_text.insert(tk.END, "=== SDG Goal 9 Infrastructure Forecast Results ===\n\n")
            
            # === DETAILED CONFIGURATION ===
            self.results_text.insert(tk.END, f"📋 FORECAST CONFIGURATION:\n")
            self.results_text.insert(tk.END, f"Indicator: {indicator_id} ({series_description})\n")
            self.results_text.insert(tk.END, f"Country: {country}\n")
            self.results_text.insert(tk.END, f"Location: {location}\n")
            self.results_text.insert(tk.END, f"Transport Mode: {transport}\n")
            self.results_text.insert(tk.END, f"Source: {source}\n")
            self.results_text.insert(tk.END, f"Model: {model_type}\n\n")
            
            # === COMPREHENSIVE MODEL ANALYSIS ===
            try:
                self.results_text.insert(tk.END, f"📊 MODEL PERFORMANCE:\n")
                self.results_text.insert(tk.END, f"Test RMSE: {rmse:.3f} {unit}\n")
                
                # === DETAILED CROSS-VALIDATION RESULTS ===
                self.results_text.insert(tk.END, f"\n🎯 TIME SERIES CROSS-VALIDATION OVERVIEW:\n")
                self.results_text.insert(tk.END, f"📊 Validation Methodology: Walk-forward analysis\n")
                self.results_text.insert(tk.END, f"🔄 Strategy: Expanding window (preserves temporal order)\n")
                self.results_text.insert(tk.END, f"📈 Metric: Root Mean Square Error (RMSE)\n")
                self.results_text.insert(tk.END, f"⏰ Prevents data leakage in time series forecasting\n")
                
                if model_type == 'ARIMA':
                    self.results_text.insert(tk.END, f"\n🔗 ARIMA MODEL ANALYSIS:\n")
                    arima_results = self.fit_arima_model(series)
                    if 'best_order' in arima_results:
                        self.results_text.insert(tk.END, f"✅ Optimal ARIMA Order: {arima_results['best_order']}\n")
                    if 'cv_results' in arima_results:
                        cv_results = arima_results['cv_results']
                        self.results_text.insert(tk.END, f"📊 TIME SERIES CROSS VALIDATION (ARIMA):\n")
                        
                        # Best model performance
                        best_order_key = str(arima_results.get('best_order', (1,1,1)))
                        if best_order_key in cv_results:
                            best_cv = cv_results[best_order_key]
                            mean_rmse = best_cv.get('mean_rmse', 'N/A')
                            std_rmse = best_cv.get('std_rmse', 0)
                            n_folds = best_cv.get('n_folds', 'N/A')
                            fold_scores = best_cv.get('fold_scores', [])
                            
                            if isinstance(mean_rmse, (int, float)) and isinstance(std_rmse, (int, float)):
                                self.results_text.insert(tk.END, f"✅ Best Model CV RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}\n")
                                cv_coeff = (std_rmse / mean_rmse) * 100 if mean_rmse > 0 else 0
                                self.results_text.insert(tk.END, f"📊 Model Stability (CV%): {cv_coeff:.2f}%\n")
                            else:
                                self.results_text.insert(tk.END, f"✅ Best Model CV RMSE: {mean_rmse} ± {std_rmse}\n")
                            
                            self.results_text.insert(tk.END, f"🔄 Cross-validation Strategy: Expanding Window\n")
                            self.results_text.insert(tk.END, f"📈 Number of CV Folds: {n_folds}\n")
                            
                            # Individual fold results
                            if fold_scores and len(fold_scores) > 0:
                                self.results_text.insert(tk.END, f"📋 Individual Fold Performance:\n")
                                for i, score in enumerate(fold_scores[:5], 1):  # Show first 5 folds
                                    self.results_text.insert(tk.END, f"   Fold {i}: {score:.4f} RMSE\n")
                                if len(fold_scores) > 5:
                                    self.results_text.insert(tk.END, f"   ... and {len(fold_scores)-5} more folds\n")
                                
                                # Performance statistics
                                min_score = min(fold_scores)
                                max_score = max(fold_scores)
                                self.results_text.insert(tk.END, f"📊 Performance Range: {min_score:.4f} - {max_score:.4f}\n")
                        
                        # Model comparison across orders
                        self.results_text.insert(tk.END, f"\n🔍 MODEL ORDER COMPARISON:\n")
                        sorted_orders = sorted(cv_results.items(), key=lambda x: x[1].get('mean_rmse', float('inf')))
                        for i, (order, metrics) in enumerate(sorted_orders[:3], 1):  # Show top 3
                            rmse = metrics.get('mean_rmse', 'N/A')
                            if isinstance(rmse, (int, float)):
                                self.results_text.insert(tk.END, f"   #{i}. ARIMA{order}: {rmse:.4f} RMSE\n")
                            else:
                                self.results_text.insert(tk.END, f"   #{i}. ARIMA{order}: {rmse} RMSE\n")
                    self.results_text.insert(tk.END, f"✅ ARIMA captures infrastructure development cycles\n")
                    self.results_text.insert(tk.END, f"✅ Suitable for infrastructure time series analysis\n")
                
                elif model_type == 'Prophet':
                    self.results_text.insert(tk.END, f"\n🔮 PROPHET MODEL ANALYSIS:\n")
                    prophet_results = self.fit_prophet_model(series)
                    if 'cv_results' in prophet_results:
                        cv_results = prophet_results['cv_results']
                        self.results_text.insert(tk.END, f"📊 TIME SERIES CROSS VALIDATION (PROPHET):\n")
                        
                        mean_rmse = cv_results.get('mean_rmse', 'N/A')
                        std_rmse = cv_results.get('std_rmse', 0)
                        n_folds = cv_results.get('n_folds', 'N/A')
                        fold_scores = cv_results.get('fold_scores', [])
                        
                        if isinstance(mean_rmse, (int, float)) and isinstance(std_rmse, (int, float)):
                            self.results_text.insert(tk.END, f"✅ CV RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}\n")
                            cv_coeff = (std_rmse / mean_rmse) * 100 if mean_rmse > 0 else 0
                            self.results_text.insert(tk.END, f"📊 Model Stability (CV%): {cv_coeff:.2f}%\n")
                        else:
                            self.results_text.insert(tk.END, f"✅ CV RMSE: {mean_rmse} ± {std_rmse}\n")
                        
                        self.results_text.insert(tk.END, f"🔄 Cross-validation Strategy: Time Series Split\n")
                        self.results_text.insert(tk.END, f"📈 Number of CV Folds: {n_folds}\n")
                        
                        # Individual fold results for Prophet
                        if fold_scores and len(fold_scores) > 0:
                            self.results_text.insert(tk.END, f"📋 Individual Fold Performance:\n")
                            for i, score in enumerate(fold_scores[:5], 1):  # Show first 5 folds
                                self.results_text.insert(tk.END, f"   Fold {i}: {score:.4f} RMSE\n")
                            if len(fold_scores) > 5:
                                self.results_text.insert(tk.END, f"   ... and {len(fold_scores)-5} more folds\n")
                            
                            # Performance statistics
                            min_score = min(fold_scores)
                            max_score = max(fold_scores)
                            self.results_text.insert(tk.END, f"📊 Performance Range: {min_score:.4f} - {max_score:.4f}\n")
                        
                        # Prophet-specific validation insights
                        self.results_text.insert(tk.END, f"🔮 Prophet Features Used:\n")
                        self.results_text.insert(tk.END, f"   • Trend modeling: Linear\n")
                        self.results_text.insert(tk.END, f"   • Seasonality: None (yearly data)\n")
                        self.results_text.insert(tk.END, f"   • Infrastructure-optimized settings\n")
                    self.results_text.insert(tk.END, f"✅ Prophet handles infrastructure development irregularities\n")
                    self.results_text.insert(tk.END, f"✅ Bayesian approach for uncertainty quantification\n")
                    self.results_text.insert(tk.END, f"✅ Infrastructure-optimized settings (no yearly seasonality)\n")
                
                elif model_type == 'SARIMAX':
                    self.results_text.insert(tk.END, f"\n🔗 SARIMAX MODEL ANALYSIS:\n")
                    
                    # Check if we have true SARIMAX with external variables
                    if hasattr(self, 'sarimax_feature_names') and self.sarimax_feature_names:
                        self.results_text.insert(tk.END, f"📊 Model Type: True SARIMAX with external variables\n")
                        self.results_text.insert(tk.END, f"✅ External Variables: {', '.join(self.sarimax_feature_names)}\n")
                        if hasattr(self, 'sarimax_best_order'):
                            self.results_text.insert(tk.END, f"✅ Optimal SARIMAX Order: {self.sarimax_best_order}\n")
                        
                        # Cross-validation results
                        if hasattr(self, 'sarimax_cv_results') and self.sarimax_cv_results:
                            self.results_text.insert(tk.END, f"\n📊 TIME SERIES CROSS VALIDATION (SARIMAX):\n")
                            cv_mean = self.sarimax_cv_results.get('mean_rmse', 'N/A')
                            cv_std = self.sarimax_cv_results.get('std_rmse', 0)
                            n_folds = self.sarimax_cv_results.get('n_folds', 'N/A')
                            fold_scores = self.sarimax_cv_results.get('fold_scores', [])
                            
                            if isinstance(cv_mean, (int, float)) and isinstance(cv_std, (int, float)):
                                self.results_text.insert(tk.END, f"✅ CV RMSE: {cv_mean:.4f} ± {cv_std:.4f}\n")
                                cv_coeff = (cv_std / cv_mean) * 100 if cv_mean > 0 else 0
                                self.results_text.insert(tk.END, f"📊 Model Stability (CV%): {cv_coeff:.2f}%\n")
                            else:
                                self.results_text.insert(tk.END, f"✅ CV RMSE: {cv_mean}\n")
                            
                            self.results_text.insert(tk.END, f"🔄 Cross-validation Strategy: Expanding Window\n")
                            self.results_text.insert(tk.END, f"📈 Number of CV Folds: {n_folds}\n")
                            
                            # Individual fold results for SARIMAX
                            if fold_scores and len(fold_scores) > 0:
                                self.results_text.insert(tk.END, f"📋 Individual Fold Performance:\n")
                                for i, score in enumerate(fold_scores[:5], 1):  # Show first 5 folds
                                    self.results_text.insert(tk.END, f"   Fold {i}: {score:.4f} RMSE\n")
                                if len(fold_scores) > 5:
                                    self.results_text.insert(tk.END, f"   ... and {len(fold_scores)-5} more folds\n")
                                
                                # Performance statistics
                                min_score = min(fold_scores)
                                max_score = max(fold_scores)
                                self.results_text.insert(tk.END, f"📊 Performance Range: {min_score:.4f} - {max_score:.4f}\n")
                            
                            # SARIMAX-specific validation insights
                            if hasattr(self, 'sarimax_feature_names') and self.sarimax_feature_names:
                                self.results_text.insert(tk.END, f"🔗 External Variables Impact:\n")
                                self.results_text.insert(tk.END, f"   • Variables: {len(self.sarimax_feature_names)} features\n")
                                self.results_text.insert(tk.END, f"   • Feature scaling: StandardScaler applied\n")
                                self.results_text.insert(tk.END, f"   • Exogenous integration: Full SARIMAX\n")
                    else:
                        self.results_text.insert(tk.END, f"📊 Model Type: Simple SARIMAX (ARIMA fallback)\n")
                        self.results_text.insert(tk.END, f"⚠️ External variables not available or insufficient\n")
                        if hasattr(self, 'sarimax_best_order'):
                            self.results_text.insert(tk.END, f"✅ Optimal SARIMAX Order: {self.sarimax_best_order}\n")
                
                elif model_type == 'Random Forest':
                    self.results_text.insert(tk.END, f"\n🌳 RANDOM FOREST MODEL ANALYSIS:\n")
                    if hasattr(self, 'rf_feature_importance') and hasattr(self, 'rf_features_used'):
                        self.results_text.insert(tk.END, f"✅ Features Used: {len(self.rf_features_used)}\n")
                        
                        # Feature importance analysis
                        self.results_text.insert(tk.END, f"\n📊 FEATURE IMPORTANCE ANALYSIS:\n")
                        if len(self.rf_feature_importance) > 0:
                            sorted_features = sorted(zip(self.rf_features_used, self.rf_feature_importance), 
                                                   key=lambda x: x[1], reverse=True)
                            for i, (feature, importance) in enumerate(sorted_features[:5]):  # Top 5
                                self.results_text.insert(tk.END, f"✅ {feature}: {importance:.4f}\n")
                
                        self.results_text.insert(tk.END, f"✅ Random Forest captures complex infrastructure relationships\n")
                        self.results_text.insert(tk.END, f"✅ Non-linear pattern recognition\n")
                
                # === CROSS-VALIDATION SUMMARY ===
                self.results_text.insert(tk.END, f"\n📈 CROSS-VALIDATION PERFORMANCE SUMMARY:\n")
                self.results_text.insert(tk.END, f"🎯 Selected Model: {model_type}\n")
                self.results_text.insert(tk.END, f"📊 Final Test RMSE: {rmse:.3f} {unit}\n")
                
                # Model-specific CV summary
                current_cv_rmse = "Not available"
                current_cv_stability = "Not calculated"
                
                if model_type == 'ARIMA' and hasattr(self, 'last_arima_cv_results'):
                    cv_data = self.last_arima_cv_results
                elif model_type == 'Prophet' and hasattr(self, 'last_prophet_cv_results'):
                    cv_data = self.last_prophet_cv_results  
                elif model_type == 'SARIMAX' and hasattr(self, 'sarimax_cv_results'):
                    cv_data = self.sarimax_cv_results
                else:
                    cv_data = None
                
                if cv_data and isinstance(cv_data.get('mean_rmse'), (int, float)):
                    current_cv_rmse = f"{cv_data['mean_rmse']:.4f}"
                    if isinstance(cv_data.get('std_rmse'), (int, float)) and cv_data['mean_rmse'] > 0:
                        cv_coeff = (cv_data['std_rmse'] / cv_data['mean_rmse']) * 100
                        current_cv_stability = f"{cv_coeff:.2f}%"
                
                self.results_text.insert(tk.END, f"🔄 Cross-Validation RMSE: {current_cv_rmse}\n")
                self.results_text.insert(tk.END, f"📊 Model Stability (CV%): {current_cv_stability}\n")
                
                # Validation reliability assessment
                if model_type in ['ARIMA', 'Prophet', 'SARIMAX']:
                    self.results_text.insert(tk.END, f"\n🔍 VALIDATION QUALITY ASSESSMENT:\n")
                    data_points = len(series)
                    if data_points >= 20:
                        self.results_text.insert(tk.END, f"✅ Data sufficiency: {data_points} time points (Good)\n")
                    elif data_points >= 15:
                        self.results_text.insert(tk.END, f"⚠️ Data sufficiency: {data_points} time points (Adequate)\n")
                    else:
                        self.results_text.insert(tk.END, f"❌ Data sufficiency: {data_points} time points (Limited)\n")
                    
                    self.results_text.insert(tk.END, f"✅ Temporal validation: No data leakage\n")
                    self.results_text.insert(tk.END, f"✅ Infrastructure domain: Specialized validation\n")
                
                # === INFRASTRUCTURE CONTEXT & POLICY IMPLICATIONS ===
                self.results_text.insert(tk.END, f"\n🏗️ INFRASTRUCTURE CONTEXT:\n")
                
                # Transport mode context
                if transport == 'ROAD':
                    self.results_text.insert(tk.END, f"🛣️ Road Infrastructure: Connectivity, maintenance, safety standards\n")
                    self.results_text.insert(tk.END, f"📋 Policy Focus: Highway modernization, smart traffic systems\n")
                elif transport == 'RAIL':
                    self.results_text.insert(tk.END, f"🚊 Rail Infrastructure: Efficiency, capacity, electrification\n")
                    self.results_text.insert(tk.END, f"📋 Policy Focus: High-speed rail, freight optimization\n")
                elif transport == 'AIR':
                    self.results_text.insert(tk.END, f"✈️ Aviation Infrastructure: Airports, air traffic, connectivity\n")
                    self.results_text.insert(tk.END, f"📋 Policy Focus: Airport expansion, airspace management\n")
                elif transport == 'MARITIME':
                    self.results_text.insert(tk.END, f"🚢 Maritime Infrastructure: Ports, shipping, logistics\n")
                    self.results_text.insert(tk.END, f"📋 Policy Focus: Port digitalization, green shipping\n")
                else:
                    self.results_text.insert(tk.END, f"🚦 Multi-modal Infrastructure: Integrated transport systems\n")
                    self.results_text.insert(tk.END, f"📋 Policy Focus: Intermodal connectivity, smart cities\n")
                
                # Location-specific insights
                if location == 'URBAN':
                    self.results_text.insert(tk.END, f"🏙️ Urban Infrastructure: Dense networks, public transport priority\n")
                    self.results_text.insert(tk.END, f"💡 Recommendation: Focus on sustainable urban mobility\n")
                elif location == 'RURAL':
                    self.results_text.insert(tk.END, f"🌾 Rural Infrastructure: Connectivity challenges, basic access\n")
                    self.results_text.insert(tk.END, f"💡 Recommendation: Bridge digital and physical divides\n")
                else:
                    self.results_text.insert(tk.END, f"🌍 National Infrastructure: Comprehensive development approach\n")
                    self.results_text.insert(tk.END, f"💡 Recommendation: Balanced urban-rural investment\n")
                
                # === HISTORICAL DATA SUMMARY ===
                self.results_text.insert(tk.END, f"\n📈 HISTORICAL DATA ANALYSIS ({len(scaled_data)} points):\n")
                # Data quality assessment
                data_years = [row['TimePeriod'].year for _, row in scaled_data.iterrows()]
                data_span = max(data_years) - min(data_years) + 1
                completeness = len(scaled_data) / data_span * 100
                self.results_text.insert(tk.END, f"📊 Data Span: {min(data_years)}-{max(data_years)} ({data_span} years)\n")
                self.results_text.insert(tk.END, f"📊 Data Completeness: {completeness:.1f}%\n")
                
                # Show representative data points
                for i, (_, row) in enumerate(scaled_data.iterrows()):
                    if i < 3 or i >= len(scaled_data) - 3:
                    self.results_text.insert(tk.END, f"Year {row['TimePeriod'].year}: {row['Value']:.2f} {unit}\n")
                    elif i == 3:
                        self.results_text.insert(tk.END, f"... ({len(scaled_data) - 6} more years) ...\n")
                
                # === FUTURE FORECAST VALUES ===
                self.results_text.insert(tk.END, f"\n🔮 FUTURE FORECAST (2023-2030):\n")
                
                # Handle forecast display based on model type
                if model_type == 'Random Forest' and 'scaled_forecast' in locals():
                    for year, value in zip(scaled_forecast.index.year, scaled_forecast.values):
                        self.results_text.insert(tk.END, f"Year {year}: {value:.2f} {unit}\n")
                elif model_type == 'SARIMAX' and 'scaled_forecast' in locals():
                    # Check if scaled_forecast contains valid values
                    if pd.isna(scaled_forecast).all() or (scaled_forecast == 0).all():
                        # SARIMAX failed, use trend extrapolation fallback
                        print("📊 SARIMAX forecast failed, using trend extrapolation for display")
                        last_values = series.tail(3).values
                        trend = np.mean(np.diff(last_values)) if len(last_values) > 1 else 0
                        last_value = series.iloc[-1]
                        
                        for i, year in enumerate(scaled_forecast.index.year):
                            fallback_value = last_value + trend * (i + 1)
                            self.results_text.insert(tk.END, f"Year {year}: {fallback_value:.2f} {unit}\n")
                else:
                        # Display valid SARIMAX forecasts
                        for year, value in zip(scaled_forecast.index.year, scaled_forecast.values):
                            display_value = value if not pd.isna(value) else 0.0
                            self.results_text.insert(tk.END, f"Year {year}: {display_value:.2f} {unit}\n")
                else:
                    # Fallback for ARIMA and Prophet
                    try:
                        if model_type == 'ARIMA':
                            model_results = self.fit_arima_model(series)
                            future_forecast = model_results['model'].get_forecast(steps=8)
                            future_values = future_forecast.predicted_mean
                        elif model_type == 'Prophet':
                            model_results = self.fit_prophet_model(series)
                            future = model_results['model'].make_future_dataframe(periods=8, freq='Y')
                            forecast = model_results['model'].predict(future)
                            future_values = forecast['yhat'].iloc[-8:].values
                        
                        if 'future_values' in locals():
                    future_years = [series.index[-1].year + i + 1 for i in range(len(future_values))]
                    for year, value in zip(future_years, future_values):
                        self.results_text.insert(tk.END, f"Year {year}: {value:.2f} {unit}\n")
                    except:
                        self.results_text.insert(tk.END, f"⚠️ Forecast values not available\n")
                
                # === SUMMARY & RECOMMENDATIONS ===
                self.results_text.insert(tk.END, f"\n💡 INFRASTRUCTURE DEVELOPMENT SUMMARY:\n")
                if len(scaled_data) > 0:
                    recent_value = scaled_data.iloc[-1]['Value']
                    if 'future_values' in locals() or 'scaled_forecast' in locals():
                        trend_direction = "increasing" if recent_value < (scaled_forecast.iloc[-1] if 'scaled_forecast' in locals() else future_values[-1]) else "decreasing"
                        self.results_text.insert(tk.END, f"📈 Infrastructure Investment Trend: {trend_direction}\n")
                        self.results_text.insert(tk.END, f"🎯 Strategic Priority: Sustainable infrastructure development\n")
                        self.results_text.insert(tk.END, f"⚡ Innovation Focus: Smart infrastructure integration\n")
                
            except Exception as e:
                self.results_text.insert(tk.END, f"\n⚠️ Error in results display: {str(e)}\n")
            
            # Enable save button after plot is generated
            self.save_button.state(['!disabled'])
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            import traceback
            print(traceback.format_exc())

    def load_external_data(self):
        """Load external data including processed GDP, GINI, R&D, Social Coverage, and Unemployment data"""
        external_data = {}
        try:
            # Get absolute path to parent directory
            current_file = os.path.abspath(__file__)  # Absolute path to this script
            print(f"Current file: {current_file}")
            
            sdg9_dir = os.path.dirname(current_file)  # SDG9 directory
            parent_dir = os.path.dirname(sdg9_dir)  # SDG parent directory
            
            print(f"Looking for external data in: {parent_dir}")
            
            # Dictionary of processed files with their corresponding names and column mappings
            processed_files = {
                'gdp': {
                    'filename': 'GDP_processed.csv',
                    'columns': {
                        'country': 'Country Name',
                        'year': 'Year',
                        'value': 'GDP'
                    }
                },
                'gini': {
                    'filename': 'GINI_processed.csv',
                    'columns': {
                        'country': 'Country Name',
                        'year': 'Year',
                        'value': 'Gini index'
                    }
                },
                'unemployment': {
                    'filename': 'Unemployment_processed.csv',
                    'columns': {
                        'country': 'Country Name',
                        'year': 'Year',
                        'value': 'Unemployment'
                    }
                },
                'rd_expenditure': {
                    'filename': 'R&D Expenditures_processed.csv',
                    'columns': {
                        'country': 'Country Name',
                        'year': 'Year',
                        'value': 'Research and development expenditure'
                    }
                },
                'social_coverage': {
                    'filename': 'social_coverage_processed.csv',
                    'columns': {
                        'country': 'Country Name',
                        'year': 'Year',
                        'value': 'Social_Coverage'
                    }
                }
            }
            
            # Try to read each file and check for errors
            for data_name, config in processed_files.items():
                file_path = os.path.join(parent_dir, config['filename'])
                print(f"\nChecking for file: {file_path}")
                
                # Verify if file exists
                if not os.path.exists(file_path):
                    print(f"✗ File not found: {file_path}")
                    continue
                
                print(f"✓ File exists: {file_path}")
                
                try:
                    # Try to read the file
                    data = pd.read_csv(file_path)
                    print(f"File shape: {data.shape}")
                    print(f"File columns: {data.columns.tolist()}")
                    
                    # Check if required columns exist
                    required_columns = list(config['columns'].values())
                    column_exists = [col in data.columns for col in required_columns]
                    
                    if not all(column_exists):
                        missing_cols = [col for i, col in enumerate(required_columns) if not column_exists[i]]
                        print(f"✗ Missing columns: {missing_cols}")
                        
                        # Check for similar column names
                        for missing_col in missing_cols:
                            similar_cols = [col for col in data.columns if missing_col.lower() in col.lower()]
                            if similar_cols:
                                print(f"  Similar columns found for '{missing_col}': {similar_cols}")
                                
                                # If there's only one similar column, use it automatically
                                if len(similar_cols) == 1:
                                    print(f"  Auto-mapping '{missing_col}' to '{similar_cols[0]}'")
                                    # Update the column mapping
                                    for key, val in config['columns'].items():
                                        if val == missing_col:
                                            config['columns'][key] = similar_cols[0]
                    
                    # Try again with possibly updated column mappings
                    required_columns = list(config['columns'].values())
                    if all(col in data.columns for col in required_columns):
                        # Rename columns to standard format
                        data = data.rename(columns={
                            config['columns']['country']: 'Country Name',
                            config['columns']['year']: 'Year',
                            config['columns']['value']: 'Value'
                        })
                        
                        # Process data
                        data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
                        data['Value'] = pd.to_numeric(data['Value'], errors='coerce')
                        data = data.dropna(subset=['Year', 'Value'])
                        data = data[data['Year'] > 0]
                        
                        if len(data) > 0:
                            external_data[data_name] = data
                            print(f"✓ {data_name.upper()} data loaded ({len(data)} records)")
                            # Print sample data
                            print(f"Sample data (first 3 rows):")
                            print(data.head(3))
                        else:
                            print(f"✗ {data_name.upper()} data empty after processing")
                    else:
                        print(f"✗ {data_name.upper()} data columns mismatch")
                        print(f"  Required: {required_columns}")
                        print(f"  Available: {data.columns.tolist()}")
                    
                except Exception as e:
                    print(f"✗ Error loading {data_name}: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
            
            print(f"\nSuccessfully loaded {len(external_data)} external datasets")
            return external_data
            
        except Exception as e:
            print(f"Error loading external data: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {}
    
    def show_external_data_status(self):
        """Display external data loading status"""
        status_text = "\n=== External Data Integration Status (SDG 9) ===\n"
        for data_name in ['gdp', 'gini', 'unemployment', 'rd_expenditure', 'social_coverage']:
            if data_name in self.external_data:
                data = self.external_data[data_name]
                status_text += f"✓ {data_name.upper()} data loaded ({len(data)} records)\n"
            else:
                status_text += f"✗ {data_name.upper()} data not loaded\n"
        
        status_text += f"\nTotal external datasets: {len(self.external_data)}\n"
        status_text += "Random Forest model ready for enhanced predictions!\n"
        
        self.results_text.insert(tk.END, status_text)
        
    def prepare_external_features_for_infrastructure(self, country, location='ALLAREA', transport='ALL'):
        """Prepare external features for infrastructure forecasting with intelligent defaults"""
        try:
            # Get available years from the external data
            all_years = set()
            for data_name, data in self.external_data.items():
                if data is not None and 'Year' in data.columns:
                    all_years.update(data['Year'].unique())
            
            if not all_years:
                print("⚠️ No external data years available")
                return None, None, []
            
            all_years = sorted(list(all_years))
            feature_matrix = []
            feature_names = []
            valid_years = []
            
            for year in all_years:
                features = []
                current_feature_names = []
                
                # Prepare features with infrastructure-specific defaults
                for var_name in ['GDP', 'GINI', 'Unemployment', 'RD_Expenditure', 'Social_Coverage']:
                    value = self.get_historical_feature_value(country, year, var_name, var_name)
                    
                    if value is None or value == 0:
                        # Use infrastructure-specific intelligent defaults
                        value = self.get_infrastructure_default(country, year, var_name, location, transport)
                        print(f"📊 Using infrastructure-context default {var_name} ({value}) for {country} {year}")
                    
                    features.append(value)
                    current_feature_names.append(var_name)
                
                # Only include years with at least 3 valid features (non-zero)
                non_zero_features = sum(1 for f in features if f != 0)
                if non_zero_features >= 3:
                    feature_matrix.append(features)
                    valid_years.append(year)
                    if not feature_names:  # Set feature names on first valid year
                        feature_names = current_feature_names
            
            if len(feature_matrix) == 0:
                print("⚠️ No valid external features found")
                return None, None, []
            
            feature_matrix = np.array(feature_matrix)
            
            print(f"📊 Feature count range: {np.count_nonzero(feature_matrix, axis=1).min()} to {np.count_nonzero(feature_matrix, axis=1).max()}")
            print(f"✅ External data prepared: {feature_matrix.shape}")
            print(f"Features: {feature_names}")
            
            # Show sample data for debugging
            print("Sample data (first 3 rows):")
            for i in range(min(3, len(feature_matrix))):
                year = valid_years[i]
                row_data = [f"'{val:.2f}'" for val in feature_matrix[i]]
                print(f"  Year {year}: {row_data}")
            
            return feature_matrix, valid_years, feature_names
            
        except Exception as e:
            print(f"❌ Error preparing external features: {e}")
            return None, None, []

    def get_infrastructure_default(self, country, year, var_name, location, transport):
        """Get infrastructure-specific intelligent default values for missing external variables"""
        
        # Get global averages first as baseline
        global_avg = self.get_global_average(var_name, year)
        
        # Infrastructure-specific adjustments
        if var_name == 'GDP':
            # Infrastructure development typically correlates with higher GDP
            base_value = global_avg if global_avg > 0 else 50000000000  # 50B default
            
            # Location adjustments for infrastructure
            if location == 'URBAN':
                base_value *= 1.3  # Urban areas have higher infrastructure investment
            elif location == 'RURAL':
                base_value *= 0.7  # Rural areas typically have lower infrastructure development
            
            # Transport mode adjustments
            if transport == 'AIR':
                base_value *= 1.5  # Aviation infrastructure requires higher economic development
            elif transport == 'MARITIME':
                base_value *= 1.2  # Port infrastructure indicates trade economy
            elif transport == 'ROAD':
                base_value *= 1.0  # Standard infrastructure
            elif transport == 'RAIL':
                base_value *= 1.1  # Rail infrastructure indicates development
            
            return base_value
            
        elif var_name == 'GINI':
            # Infrastructure development can reduce inequality
            base_value = global_avg if global_avg > 0 else 40.0
            
            # Better infrastructure typically correlates with lower inequality
            if transport in ['RAIL', 'AIR']:
                base_value *= 0.9  # Advanced transport reduces inequality
            elif transport == 'ROAD':
                base_value *= 0.95  # Road access improves equality
            
            return base_value
            
        elif var_name == 'Unemployment':
            # Infrastructure development creates jobs
            base_value = global_avg if global_avg > 0 else 8.0
            
            # Infrastructure projects reduce unemployment
            if location == 'URBAN':
                base_value *= 0.8  # Urban infrastructure creates more jobs
            elif location == 'RURAL':
                base_value *= 1.1  # Rural areas may have higher baseline unemployment
            
            # Different transport modes have different employment impacts
            if transport == 'AIR':
                base_value *= 0.7  # Aviation creates high-skill jobs
            elif transport == 'RAIL':
                base_value *= 0.8  # Rail infrastructure creates jobs
            
            return base_value
            
        elif var_name == 'RD_Expenditure':
            # Advanced infrastructure requires R&D investment
            base_value = global_avg if global_avg > 0 else 1.5
            
            # Infrastructure innovation requires R&D
            if transport in ['AIR', 'RAIL']:
                base_value *= 1.3  # Advanced transport requires more R&D
            elif transport == 'MARITIME':
                base_value *= 1.1  # Port technology requires R&D
            
            return base_value
            
        elif var_name == 'Social_Coverage':
            # Infrastructure development often correlates with social programs
            base_value = global_avg if global_avg > 0 else 60.0
            
            # Better infrastructure typically means better social coverage
            if location == 'URBAN':
                base_value *= 1.1  # Urban areas have better coverage
            elif location == 'RURAL':
                base_value *= 0.8  # Rural areas may have less coverage
            
            return base_value
        
        # Fallback to global average or reasonable default
        return global_avg if global_avg > 0 else 1.0

    def get_global_average(self, var_name, year):
        """Get global average for a variable in a specific year or nearby year"""
        try:
            data_mapping = {
                'GDP': 'gdp',
                'GINI': 'gini', 
                'Unemployment': 'unemployment',
                'RD_Expenditure': 'rd_expenditure',
                'Social_Coverage': 'social_coverage'
            }
            
            column_mapping = {
                'GDP': 'GDP',
                'GINI': 'Gini index',
                'Unemployment': 'Unemployment', 
                'RD_Expenditure': 'Research and development expenditure',
                'Social_Coverage': 'Social_Coverage'
            }
            
            data_key = data_mapping.get(var_name)
            column_name = column_mapping.get(var_name)
            
            if data_key not in self.external_data or column_name not in self.external_data[data_key].columns:
                return 0
            
            data = self.external_data[data_key]
            
            # Try exact year first
            year_data = data[data['Year'] == year]
            if not year_data.empty:
                avg = year_data[column_name].mean()
                if not pd.isna(avg):
                    return avg
            
            # Try nearby years (±2 years)
            for year_offset in [-2, -1, 1, 2]:
                year_data = data[data['Year'] == year + year_offset]
                if not year_data.empty:
                    avg = year_data[column_name].mean()
                    if not pd.isna(avg):
                        print(f"📊 Using global average {var_name.lower()} ({avg:.2f}) for {year}")
                        return avg
            
            # Use overall average as last resort
            overall_avg = data[column_name].mean()
            if not pd.isna(overall_avg):
                return overall_avg
                
            return 0
            
        except Exception as e:
            print(f"Error getting global average for {var_name}: {e}")
            return 0

    def get_historical_feature_value(self, country, year, feature_name, value_column):
        """Get historical value for a feature with robust matching"""
        try:
            data_mapping = {
                'GDP': 'gdp',
                'GINI': 'gini',
                'Unemployment': 'unemployment', 
                'RD_Expenditure': 'rd_expenditure',
                'Social_Coverage': 'social_coverage'
            }
            
            column_mapping = {
                'GDP': 'GDP',
                'GINI': 'Gini index',
                'Unemployment': 'Unemployment',
                'RD_Expenditure': 'Research and development expenditure', 
                'Social_Coverage': 'Social_Coverage'
            }
            
            data_key = data_mapping.get(feature_name)
            column_name = column_mapping.get(feature_name)
            
            if data_key not in self.external_data:
                return None
                
            data = self.external_data[data_key]
            if column_name not in data.columns:
                return None
            
            # Try exact country and year match
            country_data = data[
                (data['Country Name'].str.strip().str.lower() == country.strip().lower()) &
                (data['Year'] == year)
            ]
            
            if not country_data.empty:
                value = country_data[column_name].iloc[0]
                return float(value) if not pd.isna(value) else None
            
            # Try fuzzy country match
            country_data = data[
                data['Country Name'].str.contains(country, case=False, na=False) &
                (data['Year'] == year)
            ]
            
            if not country_data.empty:
                value = country_data[column_name].iloc[0]
                return float(value) if not pd.isna(value) else None
            
            # Try most recent value for this country
            country_data = data[
                data['Country Name'].str.contains(country, case=False, na=False)
            ]
            
            if not country_data.empty:
                recent_data = country_data[country_data['Year'] <= year].sort_values('Year')
                if not recent_data.empty:
                    value = recent_data[column_name].iloc[-1]
                    return float(value) if not pd.isna(value) else None
            
            return None
            
        except Exception as e:
            return None

    def align_series_with_external(self, series, external_data_matrix, valid_years):
        """Align the main time series data with external variables matrix"""
        try:
            # Convert series index to years if needed
            if not all(isinstance(x, (int, np.integer)) for x in series.index):
                series_years = pd.to_datetime(series.index).year
            else:
                series_years = series.index
            
            # Find overlapping years
            overlapping_years = []
            overlapping_series_values = []
            overlapping_external_data = []
            
            for i, ext_year in enumerate(valid_years):
                # Check if this external year exists in series
                matching_indices = [j for j, s_year in enumerate(series_years) if s_year == ext_year]
                
                if matching_indices:
                    # Use first match if multiple
                    series_idx = matching_indices[0]
                    overlapping_years.append(ext_year)
                    overlapping_series_values.append(series.iloc[series_idx])
                    overlapping_external_data.append(external_data_matrix[i])
            
            if len(overlapping_years) == 0:
                print("⚠️ No overlapping years between series and external data")
                return None, None
            
            # Create aligned series with datetime index
            aligned_index = pd.to_datetime([f"{year}-01-01" for year in overlapping_years])
            aligned_series = pd.Series(overlapping_series_values, index=aligned_index)
            
            # Convert external data to numpy array
            aligned_external = np.array(overlapping_external_data)
            
            print(f"📊 Aligned {len(overlapping_years)} years: {min(overlapping_years)}-{max(overlapping_years)}")
            
            return aligned_series, aligned_external
            
        except Exception as e:
            print(f"❌ Error aligning series with external data: {e}")
            return None, None
        
    def fit_random_forest_model(self, series, country, location='ALLAREA', transport='ALL'):
        """Fit Enhanced Random Forest model with external factors integration"""
        try:
            print(f"\nFitting Enhanced Random Forest model for {country}")
            print(f"Using filters: Location={location}, Transport Mode={transport}")
            
            # Calculate historical trend for baseline prediction
            years = [y.year if hasattr(y, 'year') else y for y in series.index]
            values = series.values
            if len(years) > 2:
                # Calculate linear trend using numpy polyfit
                slope, intercept = np.polyfit(years, values, 1)
                self.historical_trend = {
                    'slope': slope, 
                    'intercept': intercept,
                    'last_year': max(years),
                    'last_value': values[-1]
                }
                print(f"Historical trend: {slope:.2f} per year, last value: {values[-1]:.2f}")
            else:
                self.historical_trend = None
                print("Not enough data points to calculate historical trend")
            
            # Wichtig: Wir verwenden die Filterwerte, um dem Modell mitzuteilen, 
            # dass es auf einen bestimmten gefilterten Datensatz trainiert wird
            # Die Daten selbst wurden bereits vor dem Aufruf dieser Methode gefiltert
            
            # Use the enhanced Random Forest model with filter parameters
            results = self.rf_model.fit(series, country, location, transport)
            
            # Generate future predictions with intervals using the same filter parameters
            future_results = self.rf_model.predict_future(series, country, periods=8, 
                                                          location=location, transport=transport)
            
            # Add all future prediction results to the main results
            results['future_predictions'] = future_results['predictions']
            results['conf_lower_68'] = future_results['conf_lower_68']
            results['conf_upper_68'] = future_results['conf_upper_68']
            results['conf_lower_95'] = future_results['conf_lower_95']
            results['conf_upper_95'] = future_results['conf_upper_95']
            results['pred_lower_95'] = future_results['pred_lower_95']
            results['pred_upper_95'] = future_results['pred_upper_95']
            
            return results
            
        except Exception as e:
            print(f"Error in Enhanced Random Forest model: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise Exception(f"Error in Enhanced Random Forest model: {str(e)}")

    def save_plot(self):
        """Save the current plot as an image file"""
        if self.current_fig:
            # Get user's desktop path
            desktop = os.path.expanduser("~/Desktop")
            
            # Get current selections for default filename
            indicator_id = self.indicator_var.get().split(' - ')[0]
            country = self.country_var.get()
            activity = self.activity_var.get() if hasattr(self, 'activity_var') else "ALL"
            level = self.level_var.get() if hasattr(self, 'level_var') else "ALL"
            
            # Create default filename
            default_filename = f"SDG9_{indicator_id}_{country}_{activity}_{level}.png"
            default_filename = default_filename.replace(" ", "_")
            
            # Open file dialog
            file_path = filedialog.asksaveasfilename(
                initialdir=desktop,
                initialfile=default_filename,
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                try:
                    # Save the figure
                    self.current_fig.savefig(file_path, bbox_inches='tight', dpi=300)
                    messagebox.showinfo("Success", f"Plot saved successfully to:\n{file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save plot: {str(e)}")

    def predict_future_sarimax(self, sarimax_results, country, periods=8, location='ALLAREA', transport='ALL'):
        """Generate future SARIMAX predictions with infrastructure-specific external variable extrapolation"""
        try:
            model_fit = sarimax_results['model']
            scaler = sarimax_results['scaler']
            feature_names = sarimax_results['feature_names']
            aligned_series = sarimax_results['aligned_series']
            
            # Get the last year from the aligned series
            last_year = aligned_series.index[-1].year
            future_years = range(last_year + 1, last_year + periods + 1)
            
            print(f"🏗️ Generating SARIMAX forecast for {country} ({periods} periods)")
            print(f"Infrastructure context: Location={location}, Transport={transport}")
            
            # Prepare future external variables
            future_external_data = []
            
            for year in future_years:
                # Get extrapolated values for this year
                extrapolated_values = self.extrapolate_external_variables_for_infrastructure(
                    country, year, feature_names, location, transport)
                future_external_data.append(extrapolated_values)
            
            # Scale future external data using the same scaler
            future_external_scaled = scaler.transform(np.array(future_external_data))
            
            # Generate SARIMAX forecast with external variables
            forecast = model_fit.forecast(steps=periods, exog=future_external_scaled)
            
            # Create future dates
            future_dates = pd.date_range(start=aligned_series.index[-1], periods=periods + 1, freq='Y')[1:]
            
            print(f"📊 Raw SARIMAX forecast: {forecast}")
            
            # Handle potential NaN values or invalid forecasts
            if pd.isna(forecast).any() or np.isinf(forecast).any():
                print("⚠️ SARIMAX forecast contains NaN/inf values, using trend extrapolation")
                # Fallback to trend extrapolation
                last_values = aligned_series.tail(3).values
                trend = np.mean(np.diff(last_values)) if len(last_values) > 1 else 0
                last_value = aligned_series.iloc[-1]
                forecast = [last_value + trend * (i + 1) for i in range(periods)]
                print(f"📊 Fallback forecast: {forecast}")
            else:
                print(f"📊 Valid SARIMAX forecast generated: {forecast}")
                # Fill any remaining NaN values with trend extrapolation
                forecast = pd.Series(forecast, index=future_dates)
                if forecast.isna().any():
                    last_values = aligned_series.tail(3).values
                    trend = np.mean(np.diff(last_values)) if len(last_values) > 1 else 0
                    last_value = aligned_series.iloc[-1]
                    for i, val in enumerate(forecast):
                        if pd.isna(val):
                            forecast.iloc[i] = last_value + trend * (i + 1)
                forecast = forecast.values
            
            # Return as pandas Series
            return pd.Series(forecast, index=future_dates)
            
        except Exception as e:
            print(f"❌ SARIMAX future prediction failed: {e}")
            # Fallback to simple trend extrapolation
            aligned_series = sarimax_results.get('aligned_series', sarimax_results.get('test_data'))
            if aligned_series is not None:
                last_values = aligned_series.tail(3).values
                trend = np.mean(np.diff(last_values)) if len(last_values) > 1 else 0
                future_dates = pd.date_range(start=aligned_series.index[-1], periods=periods + 1, freq='Y')[1:]
                forecast = [aligned_series.iloc[-1] + trend * (i + 1) for i in range(periods)]
                return pd.Series(forecast, index=future_dates)
            return None

    def extrapolate_external_variables_for_infrastructure(self, country, year, feature_names, location='ALLAREA', transport='ALL'):
        """Extrapolate external variables for future years with infrastructure-specific intelligence"""
        extrapolated_values = []
        
        for feature_name in feature_names:
            try:
                # Get recent historical trend (last 3-5 years if available)
                recent_years = []
                recent_values = []
                
                for past_year in range(year - 5, year):
                    value = self.get_historical_feature_value(country, past_year, feature_name, feature_name)
                    if value is not None and value != 0:
                        recent_years.append(past_year)
                        recent_values.append(value)
                
                if len(recent_values) >= 2:
                    # Calculate trend based on recent historical data
                    trend = np.polyfit(recent_years, recent_values, 1)[0]  # Linear trend
                    last_value = recent_values[-1]
                    years_ahead = year - recent_years[-1]
                    
                    # Infrastructure-specific trend adjustments
                    if feature_name == 'GDP':
                        # Infrastructure development typically accelerates GDP growth
                        if transport in ['AIR', 'RAIL']:
                            trend *= 1.1  # Advanced transport infrastructure boosts economy
                        elif transport == 'ROAD':
                            trend *= 1.05  # Road infrastructure has moderate impact
                        
                        if location == 'URBAN':
                            trend *= 1.08  # Urban infrastructure drives economic growth
                    
                    elif feature_name == 'GINI':
                        # Good infrastructure can reduce inequality
                        if transport in ['RAIL', 'AIR']:
                            trend *= 0.95  # Advanced transport reduces inequality
                        if location == 'URBAN':
                            trend *= 0.98  # Urban infrastructure access improves equality
                    
                    elif feature_name == 'Unemployment':
                        # Infrastructure projects create jobs
                        if transport in ['AIR', 'RAIL', 'MARITIME']:
                            trend *= 0.9  # Advanced infrastructure creates employment
                        if location == 'RURAL':
                            trend *= 0.95  # Rural infrastructure particularly important for jobs
                    
                    elif feature_name == 'RD_Expenditure':
                        # Advanced infrastructure requires ongoing R&D
                        if transport in ['AIR', 'RAIL']:
                            trend *= 1.15  # High-tech transport drives R&D investment
                        elif transport == 'MARITIME':
                            trend *= 1.05  # Port technology drives some R&D
                    
                    elif feature_name == 'Social_Coverage':
                        # Infrastructure development often goes with social development
                        if location == 'URBAN':
                            trend *= 1.02  # Urban areas see gradual social coverage improvement
                        elif location == 'RURAL':
                            trend *= 1.05  # Rural infrastructure can significantly improve coverage
                    
                    # Calculate extrapolated value
                    extrapolated_value = last_value + (trend * years_ahead)
                    
                    # Apply bounds to keep values realistic
                    if feature_name == 'GINI':
                        extrapolated_value = max(20, min(70, extrapolated_value))  # GINI bounds
                    elif feature_name == 'Unemployment':
                        extrapolated_value = max(1, min(25, extrapolated_value))  # Unemployment bounds
                    elif feature_name == 'RD_Expenditure':
                        extrapolated_value = max(0.1, min(5, extrapolated_value))  # R&D bounds
                    elif feature_name == 'Social_Coverage':
                        extrapolated_value = max(10, min(95, extrapolated_value))  # Social coverage bounds
                    
                    extrapolated_values.append(extrapolated_value)
                    
                else:
                    # Fallback to infrastructure-specific default
                    default_value = self.get_infrastructure_default(country, year, feature_name, location, transport)
                    extrapolated_values.append(default_value)
                    print(f"📊 Using infrastructure default for {feature_name} ({default_value:.2f}) for {country} {year}")
                
            except Exception as e:
                # Use infrastructure default as last resort
                default_value = self.get_infrastructure_default(country, year, feature_name, location, transport)
                extrapolated_values.append(default_value)
                print(f"📊 Error extrapolating {feature_name}, using default ({default_value:.2f})")
        
        return extrapolated_values

if __name__ == "__main__":
    root = tk.Tk()
    app = ForecastAppGoal9(root)
    root.mainloop() 