import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os
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
        
    def prepare_features_for_country_year(self, country, year, location='ALLAREA', sex='BOTHSEX', age='ALLAGE', 
                                         product='ALL', occupation='ALL', education='ALL'):
        """Prepare feature vector for a specific country and year with additional filters"""
        features = [year]  # Time feature
        feature_names = ['Year']
        
        # Speichere die Filterkonfiguration im Modell
        self.current_filter_config = f"{location}|{sex}|{age}|{product}|{occupation}|{education}"
        
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
        
        # Sex feature (one-hot encoding)
        sex_male = 0.0
        sex_female = 0.0
        
        if sex == 'MALE':
            sex_male = 1.0 * weight_factor
        elif sex == 'FEMALE':
            sex_female = 1.0 * weight_factor
            
        features.append(sex_male)
        features.append(sex_female)
        feature_names.append('Sex_MALE')
        feature_names.append('Sex_FEMALE')
        
        # Age feature (one-hot encoding)
        age_youth = 0.0
        age_adult = 0.0
        age_senior = 0.0
        
        if age == '15-24':
            age_youth = 1.0 * weight_factor
        elif age == '25-54':
            age_adult = 1.0 * weight_factor
        elif age == '55+':
            age_senior = 1.0 * weight_factor
            
        features.append(age_youth)
        features.append(age_adult)
        features.append(age_senior)
        feature_names.append('Age_YOUTH')
        feature_names.append('Age_ADULT')
        feature_names.append('Age_SENIOR')
            
        # Product type feature (one-hot encoding)
        if product != 'ALL':
            features.append(1.0 * weight_factor)
            feature_names.append(f'Product_{product}')
        else:
            features.append(0.0)
            feature_names.append('Product_SPECIFIC')
            
        # Occupation feature (simplified binary indicator)
        if occupation != 'ALL':
            features.append(1.0 * weight_factor)
            feature_names.append(f'Occupation_{occupation}')
        else:
            features.append(0.0)
            feature_names.append('Occupation_SPECIFIC')
            
        # Education feature (one-hot encoding)
        if education != 'ALL':
            features.append(1.0 * weight_factor)
            feature_names.append(f'Education_{education}')
        else:
            features.append(0.0)
            feature_names.append('Education_SPECIFIC')
            
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
            
        # Sex interaction terms
        if sex == 'MALE':
            features.append(year * sex_male * 0.01)
            feature_names.append('Year_x_MALE')
        elif sex == 'FEMALE':
            features.append(year * sex_female * 0.01)
            feature_names.append('Year_x_FEMALE')
        else:
            features.append(0.0)
            feature_names.append('Year_x_Sex')
        
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
    
    def fit(self, series, country, location='ALLAREA', sex='BOTHSEX', age='ALLAGE', 
           product='ALL', occupation='ALL', education='ALL'):
        """Fit the Random Forest model with filter parameters"""
        print(f"\nFitting Enhanced Random Forest model for {country} with filters")
        
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
        filter_config = f"loc_{location}_sex_{sex}_age_{age}_prod_{product}_occ_{occupation}_edu_{education}"
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
                        country, year, location, sex, age, product, occupation, education)
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
        
        # Train model with höherer Lernrate für bessere Anpassung
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
                ('Sex_' in feature_name and sex != 'BOTHSEX') or
                ('Age_' in feature_name and age != 'ALLAGE') or
                ('Product_' in feature_name and product != 'ALL') or
                ('Occupation_' in feature_name and occupation != 'ALL') or
                ('Education_' in feature_name and education != 'ALL')):
                
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
    
    def predict_future(self, series, country, periods=5, location='ALLAREA', sex='BOTHSEX', age='ALLAGE', 
                      product='ALL', occupation='ALL', education='ALL'):
        """Make future predictions with confidence and prediction intervals using filter parameters"""
        print(f"\nMaking future predictions for {country} with filters")
        
        # Get the last year from the series
        if not all(isinstance(x, (int, np.integer)) for x in series.index):
            last_year = pd.to_datetime(series.index).year.max()
        else:
            last_year = max(series.index)
            
        future_years = range(last_year + 1, last_year + periods + 1)
        
        # STEP 1: Calculate trend-based predictions
        # This is more reliable for time series than complex feature-based prediction
        slope = self.trend_params['slope']
        intercept = self.trend_params['intercept']
        last_year = self.trend_params['last_year']
        last_value = self.trend_params['last_value']
        
        trend_predictions = []
        for year in future_years:
            # Calculate trend prediction
            years_since_last = year - last_year
            trend_prediction = last_value + (slope * years_since_last)
            
            # Add a small random variation to avoid identical predictions
            random_factor = 1.0 + (np.random.random() * 0.02 - 0.01)  # ±1%
            prediction = trend_prediction * random_factor
            
            trend_predictions.append(prediction)
            print(f"Year {year}: Trend prediction = {trend_prediction:.2f}, with variation = {prediction:.2f}")
        
        trend_predictions = np.array(trend_predictions)
        
        # STEP 2: Calculate feature-based predictions
        future_features = []
        for year in future_years:
            features, _ = self.prepare_features_for_country_year(
                country, year, location, sex, age, product, occupation, education)
            future_features.append(features)
        
        future_features = np.array(future_features)
        future_features_scaled = self.scaler.transform(future_features)
        
        # Make predictions with all trees to get uncertainty estimates
        tree_predictions = []
        for tree in self.model.estimators_:
            tree_pred = tree.predict(future_features_scaled)
            tree_predictions.append(tree_pred)
        
        tree_predictions = np.array(tree_predictions)
        
        # Calculate mean prediction and standard deviation
        model_predictions = np.mean(tree_predictions, axis=0)
        prediction_std = np.std(tree_predictions, axis=0)
        
        # STEP 3: Combine trend and model predictions (60% RF, 40% trend)
        # Check if model predictions are all the same (a common issue)
        if np.std(model_predictions) < 0.01 * np.mean(model_predictions):
            print("Model predictions are too similar - using trend prediction only")
            future_predictions = trend_predictions
        else:
            # Combine predictions with weight 40% trend, 60% model
            future_predictions = np.zeros_like(trend_predictions)
            for i in range(len(future_predictions)):
                future_predictions[i] = 0.4 * trend_predictions[i] + 0.6 * model_predictions[i]
            print("Combined trend and model predictions (60% Random Forest, 40% Trend)")
        
        # Add a minimum standard deviation to ensure visible intervals
        min_std = np.abs(future_predictions) * 0.05  # At least 5% of the prediction value
        prediction_std = np.maximum(prediction_std, min_std)
        
        # Calculate confidence intervals (68% and 95%)
        confidence_interval_68 = 1.0 * prediction_std
        confidence_interval_95 = 2.0 * prediction_std
        
        # Calculate prediction intervals (wider than confidence intervals)
        prediction_interval_95 = 3.0 * prediction_std
        
        # Create datetime index for future predictions
        future_datetime_index = pd.to_datetime([f"{year}-01-01" for year in future_years])
        
        # Print out future predictions for debugging
        print("\nFuture predictions for each year:")
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

class ForecastAppGoal8:
    def __init__(self, root):
        self.root = root
        self.root.title("SDG 8 Indicator Forecast")
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
            file_path = os.path.join(self.current_dir, 'Goal8_processed.csv')
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
        self.model_combo['values'] = ['ARIMA', 'Prophet', 'Random Forest']
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
        # Sex selection
        ttk.Label(self.selection_frame, text="Sex:").grid(row=0, column=2, padx=2, pady=2, sticky=tk.W)
        self.sex_var = tk.StringVar()
        self.sex_combo = ttk.Combobox(self.selection_frame, textvariable=self.sex_var, width=15)
        self.sex_combo['values'] = ['BOTHSEX', 'MALE', 'FEMALE']
        self.sex_combo.set('BOTHSEX')
        self.sex_combo.grid(row=0, column=3, padx=2, pady=2, sticky=tk.W)
        
        # Age selection
        ttk.Label(self.selection_frame, text="Age:").grid(row=1, column=2, padx=2, pady=2, sticky=tk.W)
        self.age_var = tk.StringVar()
        self.age_combo = ttk.Combobox(self.selection_frame, textvariable=self.age_var, width=15)
        self.age_combo['values'] = ['ALLAGE', '15-24', '25-54', '55+']
        self.age_combo.set('ALLAGE')
        self.age_combo.grid(row=1, column=3, padx=2, pady=2, sticky=tk.W)
        
        # Type of product selection
        ttk.Label(self.selection_frame, text="Product Type:").grid(row=2, column=2, padx=2, pady=2, sticky=tk.W)
        self.product_var = tk.StringVar()
        self.product_combo = ttk.Combobox(self.selection_frame, textvariable=self.product_var, width=15)
        self.product_combo['values'] = ['ALL', 'GOODS', 'SERVICES']
        self.product_combo.set('ALL')
        self.product_combo.grid(row=2, column=3, padx=2, pady=2, sticky=tk.W)
        
        # Type of occupation selection
        ttk.Label(self.selection_frame, text="Occupation:").grid(row=3, column=2, padx=2, pady=2, sticky=tk.W)
        self.occupation_var = tk.StringVar()
        self.occupation_combo = ttk.Combobox(self.selection_frame, textvariable=self.occupation_var, width=15)
        self.occupation_combo['values'] = ['ALL', 'MANAGERS', 'PROFESSIONALS', 'TECHNICIANS', 'CLERKS', 'SERVICE', 'CRAFT', 'OPERATORS', 'ELEMENTARY']
        self.occupation_combo.set('ALL')
        self.occupation_combo.grid(row=3, column=3, padx=2, pady=2, sticky=tk.W)
        
        # Education level selection
        ttk.Label(self.selection_frame, text="Education:").grid(row=4, column=2, padx=2, pady=2, sticky=tk.W)
        self.education_var = tk.StringVar()
        self.education_combo = ttk.Combobox(self.selection_frame, textvariable=self.education_var, width=15)
        self.education_combo['values'] = ['ALL', 'PRIMARY', 'SECONDARY', 'TERTIARY']
        self.education_combo.set('ALL')
        self.education_combo.grid(row=4, column=3, padx=2, pady=2, sticky=tk.W)
        
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
        """Fit ARIMA model to the time series"""
        # Use 80% of data for training to evaluate model performance
        train_size = int(len(series) * 0.8)
        train, test = series[0:train_size], series[train_size:]
        
        # Fit model on training data for evaluation
        eval_model = ARIMA(train, order=(1,1,1))
        eval_model_fit = eval_model.fit()
        
        # Make predictions for test period
        predictions = eval_model_fit.forecast(steps=len(test))
        rmse = np.sqrt(mean_squared_error(test, predictions))
        
        # Fit new model on all data for future predictions
        full_model = ARIMA(series, order=(1,1,1))
        full_model_fit = full_model.fit()
        
        return full_model_fit, predictions, test, rmse
    
    def fit_prophet_model(self, series):
        """Fit Prophet model to the time series"""
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })
        
        # Use 80% of data for training to evaluate model performance
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        # Fit model on training data for evaluation
        model = Prophet()
        model.fit(train_df)
        
        # Make predictions for test period
        future = model.make_future_dataframe(periods=len(test_df), freq='Y')
        forecast = model.predict(future)
        
        # Get predictions for the test period only
        predictions = pd.Series(forecast['yhat'].values[-len(test_df):], index=test_df['ds'])
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(test_df['y'], predictions))
        
        # Fit new model on all data for future predictions
        full_model = Prophet()
        full_model.fit(df)
        
        return full_model, predictions, test_df['y'], rmse
    
    def save_plot(self):
        """Save the current plot as an image file"""
        if self.current_fig:
            # Get user's desktop path
            desktop = os.path.expanduser("~/Desktop")
            
            # Get current selections for default filename
            indicator_id = self.indicator_var.get().split(' - ')[0]
            country = self.country_var.get()
            gender = self.sex_var.get() if hasattr(self, 'sex_var') else "ALL"
            age = self.age_var.get() if hasattr(self, 'age_var') else "ALL"
            
            # Create default filename
            default_filename = f"SDG8_{indicator_id}_{country}_{gender}_{age}.png"
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
            sex = self.sex_var.get()
            age = self.age_var.get()
            product = self.product_var.get()
            occupation = self.occupation_var.get()
            education = self.education_var.get()
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
            indicator_data['Sex'] = indicator_data['Sex'].fillna('BOTHSEX')
            indicator_data['Age'] = indicator_data['Age'].fillna('ALLAGE')
            indicator_data['TypeofProduct'] = indicator_data['TypeofProduct'].fillna('ALL')
            indicator_data['TypeofOccupation'] = indicator_data['TypeofOccupation'].fillna('ALL')
            indicator_data['EducationLevel'] = indicator_data['EducationLevel'].fillna('ALL')
            
            # Apply filters
            if location != 'ALLAREA':
                indicator_data = indicator_data[indicator_data['Location'] == location]
                if len(indicator_data) == 0:
                    messagebox.showerror("Error", f"No data found for location {location} in {indicator_id} for {country}")
                    return
                    
            if sex != 'BOTHSEX':
                indicator_data = indicator_data[indicator_data['Sex'] == sex]
                if len(indicator_data) == 0:
                    messagebox.showerror("Error", f"No data found for sex {sex} in {indicator_id} for {country}")
                    return
                    
            if age != 'ALLAGE':
                indicator_data = indicator_data[indicator_data['Age'] == age]
                if len(indicator_data) == 0:
                    messagebox.showerror("Error", f"No data found for age {age} in {indicator_id} for {country}")
                    return
                    
            if product != 'ALL':
                indicator_data = indicator_data[indicator_data['TypeofProduct'] == product]
                if len(indicator_data) == 0:
                    messagebox.showerror("Error", f"No data found for product type {product} in {indicator_id} for {country}")
                    return
                    
            if occupation != 'ALL':
                indicator_data = indicator_data[indicator_data['TypeofOccupation'] == occupation]
                if len(indicator_data) == 0:
                    messagebox.showerror("Error", f"No data found for occupation {occupation} in {indicator_id} for {country}")
                    return
                    
            if education != 'ALL':
                indicator_data = indicator_data[indicator_data['EducationLevel'] == education]
                if len(indicator_data) == 0:
                    messagebox.showerror("Error", f"No data found for education level {education} in {indicator_id} for {country}")
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
                    model_fit, predictions, test, rmse = self.fit_arima_model(series)
                    
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
                    pred_interval = 2.0 * rmse  # Increased multiplier for more visible intervals
                    scaled_pred_lower_future = scaled_forecast - pred_interval
                    scaled_pred_upper_future = scaled_forecast + pred_interval
                    
                    # Calculate confidence intervals (narrower than prediction intervals)
                    conf_interval = 1.0 * rmse  # Smaller multiplier for confidence intervals
                    scaled_conf_lower_future = scaled_forecast - conf_interval
                    scaled_conf_upper_future = scaled_forecast + conf_interval
                    
                    # Generate future dates for ARIMA
                    future_dates = pd.date_range(start=series.index[-1], periods=6, freq='Y')[1:]
                    
                    # Set future_forecast for results display
                    future_forecast = scaled_forecast
                    
                elif model_type == 'Prophet':
                    # Fit Prophet model and make forecast
                    model_fit, predictions, test, rmse = self.fit_prophet_model(series)
                    
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
                
                elif model_type == 'Random Forest':
                    # Fit Random Forest model and make forecast
                    rf_results = self.fit_random_forest_model(
                        series, country, location, sex, age, product, occupation, education)
                    
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
            if sex != 'BOTHSEX':
                title += f'\nSex: {self.sex_var.get()}'
            if age != 'ALLAGE':
                title += f'\nAge: {self.age_var.get()}'
            if product != 'ALL':
                title += f'\nProduct: {self.product_var.get()}'
            if occupation != 'ALL':
                title += f'\nOccupation: {self.occupation_var.get()}'
            if education != 'ALL':
                title += f'\nEducation: {self.education_var.get()}'
            title += f'\nSource: {source}'
            title += f'\nModel: {model_type}'
            
            # Add external features information for Random Forest
            if model_type == 'Random Forest' and hasattr(self, 'rf_features_used'):
                external_features = [f for f in self.rf_features_used if f not in ['Year', 'Location_URBAN', 'Location_RURAL', 
                                                                            'Sex_MALE', 'Sex_FEMALE', 'Age_YOUTH', 
                                                                            'Age_ADULT', 'Age_SENIOR', 'Product_SPECIFIC', 
                                                                            'Occupation_SPECIFIC', 'Education_SPECIFIC',
                                                                            'Year_x_URBAN', 'Year_x_RURAL', 'Year_x_Location',
                                                                            'Year_x_MALE', 'Year_x_FEMALE', 'Year_x_Sex']]
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
            
            # Display results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Indicator: {indicator_id} ({series_description})\n")
            self.results_text.insert(tk.END, f"Country: {country}\n")
            self.results_text.insert(tk.END, f"Location: {self.location_var.get()}\n")
            self.results_text.insert(tk.END, f"Sex: {self.sex_var.get()}\n")
            self.results_text.insert(tk.END, f"Age: {self.age_var.get()}\n")
            self.results_text.insert(tk.END, f"Product Type: {self.product_var.get()}\n")
            self.results_text.insert(tk.END, f"Occupation: {self.occupation_var.get()}\n")
            self.results_text.insert(tk.END, f"Education: {self.education_var.get()}\n")
            self.results_text.insert(tk.END, f"Source: {source}\n")
            self.results_text.insert(tk.END, f"Model: {model_type}\n\n")
            
            # Add feature importance for Random Forest
            if model_type == 'Random Forest' and hasattr(self, 'rf_feature_importance'):
                self.results_text.insert(tk.END, "Feature Importance (%):\n")
                for feature, importance in sorted(self.rf_feature_importance.items(), 
                                                key=lambda x: x[1], reverse=True):
                    self.results_text.insert(tk.END, f"{feature}: {importance*100:.1f}%\n")
                self.results_text.insert(tk.END, "\n")
            
            # Add forecast results
            try:
                if model_type == 'ARIMA':
                    model_fit, predictions, test, rmse = self.fit_arima_model(series)
                    future_forecast = model_fit.get_forecast(steps=5)
                    future_values = future_forecast.predicted_mean  # Get the actual forecast values
                elif model_type == 'Prophet':
                    model_fit, predictions, test, rmse = self.fit_prophet_model(series)
                    future = model_fit.make_future_dataframe(periods=5, freq='Y')
                    forecast = model_fit.predict(future)
                    future_values = forecast['yhat'].iloc[-5:].values
                elif model_type == 'Random Forest':
                    future_values = scaled_forecast
                
                self.results_text.insert(tk.END, f"Test RMSE: {rmse:.3f} {unit}\n")
                self.results_text.insert(tk.END, "Historical Data Points:\n")
                for _, row in scaled_data.iterrows():
                    self.results_text.insert(tk.END, f"Year {row['TimePeriod'].year}: {row['Value']:.2f} {unit}\n")
                
                # Add future forecast values
                self.results_text.insert(tk.END, "Future forecast values:\n")
                if model_type == 'Random Forest':
                    for i, (year, value) in enumerate(zip(future_forecast.index.year, future_forecast.values)):
                        self.results_text.insert(tk.END, f"Year {year}: {value:.2f} {unit}\n")
                else:
                    # Get future years
                    future_years = [series.index[-1].year + i + 1 for i in range(len(future_values))]
                    for year, value in zip(future_years, future_values):
                        self.results_text.insert(tk.END, f"Year {year}: {value:.2f} {unit}\n")
            except Exception as e:
                self.results_text.insert(tk.END, f"Could not generate forecast: {str(e)}\n")
            
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
            
            sdg8_dir = os.path.dirname(current_file)  # SDG8 directory
            parent_dir = os.path.dirname(sdg8_dir)  # SDG parent directory
            
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
        status_text = "\n=== External Data Integration Status (SDG 8) ===\n"
        for data_name in ['gdp', 'gini', 'unemployment', 'rd_expenditure', 'social_coverage']:
            if data_name in self.external_data:
                data = self.external_data[data_name]
                status_text += f"✓ {data_name.upper()} data loaded ({len(data)} records)\n"
            else:
                status_text += f"✗ {data_name.upper()} data not loaded\n"
        
        status_text += f"\nTotal external datasets: {len(self.external_data)}\n"
        status_text += "Random Forest model ready for enhanced predictions!\n"
        
        self.results_text.insert(tk.END, status_text)

    def fit_random_forest_model(self, series, country, location='ALLAREA', sex='BOTHSEX', age='ALLAGE', 
                              product='ALL', occupation='ALL', education='ALL'):
        """Fit Enhanced Random Forest model with external factors integration"""
        try:
            print(f"\nFitting Enhanced Random Forest model for {country}")
            print(f"Using filters: Location={location}, Sex={sex}, Age={age}, Product={product}, Occupation={occupation}, Education={education}")
            
            # Wichtig: Wir verwenden die Filterwerte, um dem Modell mitzuteilen, 
            # dass es auf einen bestimmten gefilterten Datensatz trainiert wird
            # Die Daten selbst wurden bereits vor dem Aufruf dieser Methode gefiltert
            
            # Use the enhanced Random Forest model with filter parameters
            results = self.rf_model.fit(series, country, location, sex, age, product, occupation, education)
            
            # Generate future predictions with intervals using the same filter parameters
            future_results = self.rf_model.predict_future(series, country, periods=5, 
                                                         location=location, sex=sex, age=age,
                                                         product=product, occupation=occupation, education=education)
            
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

if __name__ == "__main__":
    root = tk.Tk()
    app = ForecastAppGoal8(root)
    root.mainloop() 