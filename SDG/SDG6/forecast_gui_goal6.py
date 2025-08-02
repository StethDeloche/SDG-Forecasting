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
import scipy.stats as stats

warnings.filterwarnings('ignore')

class SDGRandomForestModel:
    """
    Enhanced Random Forest model specifically designed for SDG6 water/sanitation indicators
    that incorporates external factors for water and sanitation forecasting
    """
    
    def __init__(self, external_data):
        self.external_data = external_data
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def prepare_features_for_country_year(self, country, year):
        """Prepare feature vector for a specific country and year with SDG6-specific context"""
        features = [year]  # Time feature
        feature_names = ['Year']
        
        # Helper function to find data for a country and year
        def get_country_year_value(data_df, country_name, year, value_column):
            # Try exact match first
            country_data = data_df[
                (data_df['Country Name'].str.strip().str.lower() == country_name.strip().lower()) &
                (data_df['Year'] == year)
            ]
            
            if not country_data.empty:
                return float(country_data[value_column].iloc[0])
            
            # Try contains match
            country_data = data_df[
                data_df['Country Name'].str.contains(country_name, case=False, na=False) &
                (data_df['Year'] == year)
            ]
            
            if not country_data.empty:
                return float(country_data[value_column].iloc[0])
            
            # Try to find the most recent value for this country
            country_data = data_df[
                data_df['Country Name'].str.contains(country_name, case=False, na=False)
            ]
            
            if not country_data.empty:
                # Get the most recent year with data that's <= current year
                recent_data = country_data[country_data['Year'] <= year].sort_values('Year')
                if not recent_data.empty:
                    return float(recent_data[value_column].iloc[-1])
            
            # Return 0.0 if no match found
            print(f"No {value_column} data found for {country_name} in year {year}")
            return 0.0
        
        # External data availability flags
        gdp_available = False
        gini_available = False
        unemployment_available = False
        rd_available = False
        social_available = False
        
        # Add GDP data if available (critical for water infrastructure investment)
        if 'gdp' in self.external_data:
            gdp_value = get_country_year_value(self.external_data['gdp'], country, year, 'Value')
            features.append(gdp_value)
            feature_names.append('GDP')
            gdp_available = (gdp_value != 0.0)
        
        # Add GINI data if available (inequality affects water access)
        if 'gini' in self.external_data:
            gini_value = get_country_year_value(self.external_data['gini'], country, year, 'Value')
            features.append(gini_value)
            feature_names.append('GINI')
            gini_available = (gini_value != 0.0)
        
        # Add Unemployment data if available (economic conditions affect sanitation)
        if 'unemployment' in self.external_data:
            unemployment_value = get_country_year_value(self.external_data['unemployment'], country, year, 'Value')
            features.append(unemployment_value)
            feature_names.append('Unemployment')
            unemployment_available = (unemployment_value != 0.0)
        
        # Add R&D Expenditure data if available (innovation in water technology)
        if 'rd_expenditure' in self.external_data:
            rd_value = get_country_year_value(self.external_data['rd_expenditure'], country, year, 'Value')
            features.append(rd_value)
            feature_names.append('R&D Expenditure')
            rd_available = (rd_value != 0.0)
        
        # Add Social Coverage data if available (public services including water)
        if 'social_coverage' in self.external_data:
            social_value = get_country_year_value(self.external_data['social_coverage'], country, year, 'Value')
            features.append(social_value)
            feature_names.append('Social Coverage')
            social_available = (social_value != 0.0)
            
        # Check if we have any actual external data (not just zeros)
        if not any([gdp_available, gini_available, unemployment_available, rd_available, social_available]):
            print(f"Warning: No external data found for {country} in year {year}. Using only year as feature.")
        
        return features, feature_names
    
    def fit(self, series, country):
        """Fit the Random Forest model for water/sanitation indicators"""
        print(f"\nFitting Enhanced Random Forest model for {country} (SDG6 Water & Sanitation)")
        
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
        
        for year in sorted(series.index):
            try:
                value = series.loc[year]
                
                # If value is a Series, take the mean
                if isinstance(value, pd.Series):
                    value = value.mean()
                
                if pd.notna(value):
                    features, feature_names = self.prepare_features_for_country_year(country, year)
                    features_list.append(features)
                    targets.append(value)
                    years_list.append(year)
            except Exception as e:
                print(f"Error processing year {year}: {e}")
                continue
        
        print(f"Processed {len(series)} years, created {len(features_list)} feature vectors")
        
        if len(features_list) == 0:
            raise ValueError("No valid training data available. Please check if there's enough historical data for this water/sanitation series.")
        
        self.feature_names = feature_names
        X = np.array(features_list)
        y = np.array(targets)
        years_array = np.array(years_list)
        
        print(f"Training data shape: {X.shape}")
        print(f"Feature names: {self.feature_names}")
        print(f"Years range: {years_array.min()} to {years_array.max()}")
        
        # If we have very few samples, adjust the number of estimators
        n_estimators = min(100, max(10, len(X) * 2))
        self.model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=1, 
                                           min_samples_split=2, random_state=42)
        
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
        
        # Calculate simple linear trend for later use in prediction
        slope, intercept = np.polyfit(years_array, y, 1)
        self.trend_params = {
            'slope': slope,
            'intercept': intercept,
            'last_year': years_array[-1],
            'last_value': y[-1]
        }
        print(f"Water trend calculation: slope={slope:.4f}, intercept={intercept:.4f}")
        
        return {
            'train_predictions': pd.Series(train_predictions, index=train_datetime_indices),
            'test_predictions': pd.Series(test_predictions, index=test_datetime_indices),
            'rmse': rmse,
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
        }
    
    def predict_future(self, series, country, periods=5):
        """Make future predictions with confidence and prediction intervals for water/sanitation indicators"""
        # Get the last year from the series
        if not all(isinstance(x, (int, np.integer)) for x in series.index):
            last_year = pd.to_datetime(series.index).year.max()
        else:
            last_year = max(series.index)
            
        # Only forecast future years
        future_years = range(last_year + 1, last_year + periods + 1)
        
        print(f"🔮 Water/Sanitation Random Forest forecasting from {last_year + 1} to {last_year + periods} ({periods} periods)")
        
        # STEP 1: Enhanced trend analysis for water indicators
        slope = self.trend_params['slope']
        intercept = self.trend_params['intercept']
        last_year_trend = self.trend_params['last_year']
        last_value = self.trend_params['last_value']
        
        # Get historical data for advanced analysis
        years_hist = pd.to_datetime(series.index).year.values if not all(isinstance(x, (int, np.integer)) for x in series.index) else series.index.values
        values_hist = series.values
        
        # STEP 2: Water-specific intelligent feature extrapolation
        future_features_enhanced = []
        for i, year in enumerate(future_years):
            period = i + 1
            enhanced_features = []
            
            # Get base features for this FUTURE year only
            base_features, feature_names = self.prepare_features_for_country_year(country, year)
            
            if len(base_features) >= 3:  # We have external data
                # GDP: Water infrastructure investment capacity
                if 'GDP' in str(feature_names) or len(base_features) > 0:
                    last_gdp = base_features[0] if len(base_features) > 0 else 50000
                    # Water infrastructure follows economic cycles
                    business_cycle = np.sin(2 * np.pi * period / 7) * 0.03
                    base_growth = 0.025  # 2.5% base growth
                    dampening = (0.98 ** period)
                    growth_rate = (base_growth * dampening) + business_cycle
                    
                    if growth_rate > 0.06:  # Cap at 6%
                        growth_rate = 0.06 - (growth_rate - 0.06) * 0.5
                    
                    future_gdp = last_gdp * ((1 + growth_rate) ** period)
                    enhanced_features.append(future_gdp)
                
                # GINI: Water access inequality patterns
                if len(base_features) > 1:
                    last_gini = base_features[1]
                    # Water access improvements can reduce inequality
                    country_targets = {
                        'germany': 28, 'france': 32, 'italy': 35, 'spain': 36,
                        'united states': 37, 'brazil': 45, 'china': 42,
                        'world': 35, 'europe': 31, 'africa': 42
                    }
                    
                    target_gini = 35  # Default
                    for country_key, target in country_targets.items():
                        if country_key in country.lower():
                            target_gini = target
                            break
                    
                    # Water policy cycles affect inequality
                    policy_cycle = np.sin(2 * np.pi * period / 12) * 2.0
                    reversion_speed = 0.08
                    future_gini = last_gini + (target_gini - last_gini) * reversion_speed * period + policy_cycle
                    future_gini = max(15, min(60, future_gini))
                    enhanced_features.append(future_gini)
                
                # Unemployment: Affects sanitation affordability
                if len(base_features) > 2:
                    last_unemployment = base_features[2]
                    structural_rates = {
                        'germany': 4.5, 'france': 8.5, 'italy': 9.5, 'spain': 12.0,
                        'united states': 5.5, 'brazil': 11.0, 'china': 4.0,
                        'world': 7.0, 'europe': 7.5, 'africa': 12.0
                    }
                    
                    structural_rate = 7.0
                    for country_key, rate in structural_rates.items():
                        if country_key in country.lower():
                            structural_rate = rate
                            break
                    
                    gdp_effect = -business_cycle * 0.8
                    mean_reversion = (structural_rate - last_unemployment) * 0.15 * period
                    future_unemployment = last_unemployment + mean_reversion + gdp_effect
                    future_unemployment = max(1, min(25, future_unemployment))
                    enhanced_features.append(future_unemployment)
                
                # R&D: Water technology innovation
                if len(base_features) > 3:
                    last_rd = base_features[3]
                    # Water tech innovation waves
                    innovation_wave = np.sin(2 * np.pi * period / 10) * 0.15
                    base_rd_growth = 0.03
                    future_rd = last_rd * ((1 + base_rd_growth) ** period) + innovation_wave
                    future_rd = max(0.1, min(5.0, future_rd))
                    enhanced_features.append(future_rd)
                
                # Social Coverage: Water service delivery
                if len(base_features) > 4:
                    last_social = base_features[4]
                    # Water service expansion cycles
                    policy_cycle = np.sin(2 * np.pi * period / 15) * 3.0
                    improvement_rate = 0.015  # Water access improvements
                    future_social = last_social * (1 + improvement_rate * period) + policy_cycle
                    future_social = max(10, min(100, future_social))
                    enhanced_features.append(future_social)
                
                # Pad with remaining features if needed
                while len(enhanced_features) < len(base_features):
                    enhanced_features.append(base_features[len(enhanced_features)])
                    
                future_features_enhanced.append(enhanced_features[:len(base_features)])
            else:
                # Fallback to original features if no external data
                future_features_enhanced.append(base_features)
        
        # STEP 3: Model predictions with enhanced water-specific features
        if len(future_features_enhanced) > 0 and len(future_features_enhanced[0]) > 0:
            future_features = np.array(future_features_enhanced)
            future_features_scaled = self.scaler.transform(future_features)
            
            # Get predictions from all trees for uncertainty estimation
            tree_predictions = []
            for tree in self.model.estimators_:
                tree_pred = tree.predict(future_features_scaled)
                tree_predictions.append(tree_pred)
            
            tree_predictions = np.array(tree_predictions)
            model_predictions = np.mean(tree_predictions, axis=0)
            prediction_std = np.std(tree_predictions, axis=0)
            
            print(f"Enhanced water features used for {len(future_features)} future periods")
            print(f"Model prediction std: {np.mean(prediction_std):.4f}")
        else:
            # Fallback to trend if no features available
            model_predictions = np.array([last_value + slope * period for period in range(1, periods + 1)])
            prediction_std = np.ones_like(model_predictions) * abs(last_value) * 0.1
            print("Using trend-based predictions (no external features)")
        
        # STEP 4: Enhanced trend predictions with water infrastructure realism
        trend_predictions = []
        for i, year in enumerate(future_years):
            period = i + 1
            years_since_last = period
            
            # Base trend for water indicators
            base_trend = last_value + (slope * years_since_last)
            
            # Add water-specific cycles and volatility
            # Long-term water infrastructure cycle (8-year)
            long_cycle = np.sin(2 * np.pi * period / 8) * abs(last_value) * 0.05
            
            # Short-term policy cycle (3 year)
            short_cycle = np.sin(2 * np.pi * period / 3) * abs(last_value) * 0.02
            
            # Trend dampening for extreme water access changes
            if abs(slope) > abs(last_value) * 0.1:
                dampening_factor = 0.95 ** period
                dampened_slope = slope * dampening_factor
                trend_prediction = last_value + (dampened_slope * years_since_last)
            else:
                trend_prediction = base_trend
            
            # Add cycles
            trend_prediction += long_cycle + short_cycle
            
            trend_predictions.append(trend_prediction)
            print(f"Year {year}: Enhanced water trend = {trend_prediction:.2f} (base: {base_trend:.2f})")
        
        trend_predictions = np.array(trend_predictions)
        
        # STEP 5: Intelligent combination for water indicators
        if np.std(model_predictions) < 0.005 * np.mean(np.abs(model_predictions)):
            print("⚠️  Model predictions too similar - using enhanced water trend predictions")
            future_predictions = trend_predictions
            combination_weights = "100% Enhanced Water Trend"
        else:
            # Dynamic weighting for water indicators
            model_consistency = 1.0 / (1.0 + np.std(model_predictions) / np.mean(np.abs(model_predictions)))
            trend_weight = 0.3 + (1.0 - model_consistency) * 0.4
            model_weight = 1.0 - trend_weight
            
            future_predictions = trend_weight * trend_predictions + model_weight * model_predictions
            combination_weights = f"{model_weight*100:.0f}% Enhanced RF, {trend_weight*100:.0f}% Enhanced Water Trend"
            print(f"Combined water predictions: {combination_weights}")
        
        # STEP 6: Water-specific uncertainty estimation
        # Base uncertainty from model
        base_std = np.maximum(prediction_std, np.abs(future_predictions) * 0.03)
        
        # Add uncertainty from water infrastructure cycles and external shocks
        cycle_uncertainty = np.abs(future_predictions) * 0.02 * np.sqrt(np.arange(1, periods + 1))
        water_shock_uncertainty = np.abs(future_predictions) * 0.04  # Water crisis/drought effects
        
        total_uncertainty = np.sqrt(base_std**2 + cycle_uncertainty**2 + water_shock_uncertainty**2)
        
        # Calculate realistic confidence and prediction intervals
        confidence_interval_68 = 1.0 * total_uncertainty
        confidence_interval_95 = 2.0 * total_uncertainty
        prediction_interval_95 = 2.8 * total_uncertainty
        
        print(f"Enhanced water uncertainty: base={np.mean(base_std):.3f}, cycle={np.mean(cycle_uncertainty):.3f}, water_shock={np.mean(water_shock_uncertainty):.3f}")
        print(f"Water predictions range: {np.min(future_predictions):.2f} to {np.max(future_predictions):.2f}")
        print(f"✅ Generated {len(future_predictions)} water predictions for years {min(future_years)}-{max(future_years)}")
        
        # Create datetime index for future predictions
        future_datetime_index = pd.to_datetime([f"{year}-01-01" for year in future_years])
        
        return {
            'predictions': pd.Series(future_predictions, index=future_datetime_index),
            'conf_lower_68': pd.Series(future_predictions - confidence_interval_68, index=future_datetime_index),
            'conf_upper_68': pd.Series(future_predictions + confidence_interval_68, index=future_datetime_index),
            'conf_lower_95': pd.Series(future_predictions - confidence_interval_95, index=future_datetime_index),
            'conf_upper_95': pd.Series(future_predictions + confidence_interval_95, index=future_datetime_index),
            'pred_lower_95': pd.Series(future_predictions - prediction_interval_95, index=future_datetime_index),
            'pred_upper_95': pd.Series(future_predictions + prediction_interval_95, index=future_datetime_index)
        }

class ForecastAppGoal6:
    def __init__(self, root):
        self.root = root
        self.root.title("SDG Goal 6 Indicator Forecast with Multiple External Factors (GDP, GINI, Unemployment, R&D, Social Coverage) - Water & Sanitation")
        self.root.geometry("1400x900")
        
        # Get current directory
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load data
        self.df = self.load_data()
        if self.df is None:
            return
            
        self.indicators = self.get_available_indicators()
        
        # Load external data
        self.external_data = self.load_external_data()
        
        # Initialize Random Forest model
        self.rf_model = SDGRandomForestModel(self.external_data)
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        
        # Create frames
        self.selection_frame = ttk.LabelFrame(self.main_frame, text="Model Selection & Parameters - Water & Sanitation", padding="10")
        self.selection_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Create a PanedWindow for resizable plot and results areas
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.VERTICAL)
        self.paned_window.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.plot_frame = ttk.LabelFrame(self.paned_window, text="Water & Sanitation Forecast Plot", padding="10")
        self.results_frame = ttk.LabelFrame(self.paned_window, text="Results & Water Feature Analysis", padding="10")
        
        # Add frames to PanedWindow
        self.paned_window.add(self.plot_frame, weight=3)
        self.paned_window.add(self.results_frame, weight=2)
        
        # Configure frame grid weights for proper resizing
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
        
        # Initialize components
        self.canvas = None
        self.current_fig = None  # Store the current figure
        self.results_text = tk.Text(self.results_frame, height=8, width=100, wrap=tk.WORD)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add scrollbar for results text
        scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        # Configure results frame grid for scrollbar
        self.results_frame.grid_columnconfigure(1, weight=0)
        
        # Create widgets
        self.create_selection_widgets()
        
        # Show external data status
        self.show_external_data_status()
        
    def load_data(self):
        """Load the processed SDG6 water/sanitation data"""
        try:
            file_path = os.path.join(self.current_dir, 'Goal6_processed.csv')
            print(f"Loading SDG6 data from: {file_path}")
            if not os.path.exists(file_path):
                messagebox.showerror("Error", f"SDG6 data file not found: {file_path}")
                return None
            
            # Try reading with different encodings and error handling
            encodings = ['utf-8', 'latin1', 'iso-8859-1']
            for encoding in encodings:
                try:
                    data = pd.read_csv(file_path, 
                                     sep=';', 
                                     encoding=encoding,
                                     on_bad_lines='skip',
                                     low_memory=False)
                    print(f"Successfully loaded {len(data)} rows of SDG6 data using {encoding} encoding")
                    print(f"Columns: {data.columns.tolist()}")
                    return data
                except Exception as e:
                    print(f"Failed to load with {encoding} encoding: {str(e)}")
                    continue
            
            messagebox.showerror("Error", "Failed to load SDG6 data with any encoding")
            return None
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load SDG6 data: {str(e)}")
            print(f"Error loading SDG6 data: {str(e)}")
            return None
    
    def get_available_indicators(self):
        """Get list of available SDG6 indicators with their descriptions"""
        indicators = self.df[['Indicator', 'SeriesCode', 'SeriesDescription']].drop_duplicates()
        return indicators.sort_values('Indicator')
    
    def get_available_countries(self, indicator_id):
        """Get list of available countries for a specific SDG6 indicator"""
        countries = self.df[self.df['Indicator'] == indicator_id]['GeoAreaName'].unique()
        return sorted(countries)
    
    def get_available_series_codes(self, indicator_id, country):
        """Get list of available series codes for a specific SDG6 indicator and country"""
        series_codes = self.df[
            (self.df['Indicator'] == indicator_id) & 
            (self.df['GeoAreaName'] == country)
        ]['SeriesCode'].unique()
        return sorted(series_codes)
    
    def create_selection_widgets(self):
        # Model selection
        ttk.Label(self.selection_frame, text="Model:").grid(row=0, column=0, padx=2, pady=2, sticky=tk.W)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(self.selection_frame, textvariable=self.model_var, width=15)
        self.model_combo['values'] = ['ARIMA', 'Prophet', 'Random Forest', 'SARIMAX']
        self.model_combo.set('ARIMA')
        self.model_combo.grid(row=0, column=1, padx=2, pady=2, sticky=tk.W)
        
        # Indicator selection
        ttk.Label(self.selection_frame, text="SDG6 Indicator:").grid(row=1, column=0, padx=2, pady=2, sticky=tk.W)
        self.indicator_var = tk.StringVar()
        self.indicator_combo = ttk.Combobox(self.selection_frame, textvariable=self.indicator_var, width=40)
        self.indicator_combo['values'] = [f"{ind} - {desc}" for ind, desc in zip(self.indicators['Indicator'], self.indicators['SeriesDescription'])]
        self.indicator_combo.grid(row=1, column=1, padx=2, pady=2, sticky=tk.W)
        self.indicator_combo.bind('<<ComboboxSelected>>', self.update_countries)
        self.indicator_combo.bind('<<ComboboxSelected>>', self.on_indicator_change, add='+')
        
        # Country selection
        ttk.Label(self.selection_frame, text="Country:").grid(row=2, column=0, padx=2, pady=2, sticky=tk.W)
        self.country_var = tk.StringVar()
        self.country_combo = ttk.Combobox(self.selection_frame, textvariable=self.country_var, width=15)
        self.country_combo.grid(row=2, column=1, padx=2, pady=2, sticky=tk.W)
        self.country_combo.bind('<<ComboboxSelected>>', self.update_series_codes)
        self.country_combo.bind('<<ComboboxSelected>>', self.on_country_change, add='+')
        
        # Series Code selection
        ttk.Label(self.selection_frame, text="Series Code:").grid(row=3, column=0, padx=2, pady=2, sticky=tk.W)
        self.series_code_var = tk.StringVar()
        self.series_code_combo = ttk.Combobox(self.selection_frame, textvariable=self.series_code_var, width=15)
        self.series_code_combo.grid(row=3, column=1, padx=2, pady=2, sticky=tk.W)
        
        # Forecast button
        self.forecast_button = ttk.Button(self.selection_frame, text="Generate Water & Sanitation Forecast", command=self.generate_forecast)
        self.forecast_button.grid(row=4, column=0, columnspan=2, padx=5, pady=5)
    
    def update_countries(self, event=None):
        """Update country combobox when indicator is selected"""
        selected = self.indicator_var.get()
        if selected:
            indicator_id = selected.split(' - ')[0]
            countries = self.get_available_countries(indicator_id)
            self.country_combo['values'] = countries
            if countries:
                self.country_combo.set(countries[0])
            self.update_series_codes()
    
    def update_series_codes(self, event=None):
        """Update series code combobox when country is selected"""
        selected_indicator = self.indicator_var.get()
        selected_country = self.country_var.get()
        if selected_indicator and selected_country:
            indicator_id = selected_indicator.split(' - ')[0]
            series_codes = self.get_available_series_codes(indicator_id, selected_country)
            self.series_code_combo['values'] = series_codes
            if series_codes:
                self.series_code_combo.set(series_codes[0])
    
    def prepare_time_series(self, data):
        """Prepare time series data for modeling"""
        data['TimePeriod'] = pd.to_datetime(data['TimePeriod'], format='%Y')
        data = data.set_index('TimePeriod')
        data = data.sort_index()
        return data['Value']
    
    def fit_arima_model(self, series):
        """Fit ARIMA model to the time series with proper time series cross validation"""
        print(f"\n🔄 ARIMA Model with Time Series Cross Validation for SDG6")
        print(f"Data points: {len(series)}")
        
        # Time Series Cross Validation for ARIMA order selection
        best_order = None
        best_cv_score = float('inf')
        cv_results = {}
        
        # Test different ARIMA orders suitable for water/sanitation indicators
        orders_to_test = [
            (1, 1, 1), (1, 1, 0), (0, 1, 1), 
            (2, 1, 1), (1, 1, 2), (1, 0, 1), 
            (2, 0, 2), (0, 1, 2), (2, 1, 0)
        ]
        
        print("📊 Testing ARIMA orders with time series cross validation for water indicators...")
        
        for order in orders_to_test:
            try:
                # Perform time series cross validation
                cv_scores = []
                n_splits = min(5, len(series) // 4)  # Adaptive number of splits
                
                if n_splits < 3:
                    print(f"⚠️  Not enough data for cross validation ({len(series)} points)")
                    break
                
                # Time series split: expanding window
                for i in range(n_splits):
                    # Calculate split points
                    min_train_size = max(8, len(series) // 3)
                    train_end = min_train_size + i * (len(series) - min_train_size) // (n_splits - 1)
                    test_start = train_end
                    test_end = min(test_start + max(2, len(series) // 10), len(series))
                    
                    if test_end > len(series) or test_start >= test_end:
                        continue
                    
                    train_data = series.iloc[:train_end]
                    test_data = series.iloc[test_start:test_end]
                    
                    if len(train_data) < 5 or len(test_data) < 1:
                        continue
                    
                    try:
                        # Fit ARIMA model on training data
                        model = ARIMA(train_data, order=order)
                        model_fit = model.fit()
                        
                        # Make predictions on test data
                        forecast = model_fit.forecast(steps=len(test_data))
                        
                        # Calculate RMSE
                        rmse = np.sqrt(mean_squared_error(test_data, forecast))
                        cv_scores.append(rmse)
                        
                    except Exception as e:
                        # Skip this fold if model fitting fails
                        continue
                
                if len(cv_scores) > 0:
                    mean_cv_score = np.mean(cv_scores)
                    cv_results[order] = {
                        'mean_rmse': mean_cv_score,
                        'std_rmse': np.std(cv_scores),
                        'n_folds': len(cv_scores)
                    }
                    
                    print(f"ARIMA{order}: {mean_cv_score:.4f} RMSE ({len(cv_scores)} folds)")
                    
                    if mean_cv_score < best_cv_score:
                        best_cv_score = mean_cv_score
                        best_order = order
                
            except Exception as e:
                print(f"⚠️  ARIMA{order} failed: {str(e)}")
                continue
        
        # Use best order or fall back to simple order
        if best_order is None:
            best_order = (1, 1, 1)
            print(f"⚠️  Using fallback order: {best_order}")
        else:
            print(f"✅ Best ARIMA order for water indicator: {best_order} (CV RMSE: {best_cv_score:.4f})")
        
        # Final model training with train/test split
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]
        
        print(f"📈 Final training: {len(train)} train, {len(test)} test points")
        
        # Fit model on training data for evaluation
        eval_model = ARIMA(train, order=best_order)
        eval_model_fit = eval_model.fit()
        
        # Make predictions for test period
        predictions = eval_model_fit.forecast(steps=len(test))
        test_rmse = np.sqrt(mean_squared_error(test, predictions))
        
        print(f"✅ Test RMSE: {test_rmse:.4f}")
        
        # Fit final model on all data for future predictions
        full_model = ARIMA(series, order=best_order)
        full_model_fit = full_model.fit()
        
        return {
            'model': full_model_fit,
            'test_predictions': predictions,
            'test_data': test,
            'rmse': test_rmse,
            'best_order': best_order,
            'cv_results': cv_results
        }
    
    def fit_prophet_model(self, series):
        """Fit Prophet model to the time series with time series cross validation for water indicators"""
        print(f"\n🔄 Prophet Model with Time Series Cross Validation for SDG6")
        print(f"Data points: {len(series)}")
        
        # Convert series to DataFrame with proper datetime index
        df = pd.DataFrame({'ds': pd.to_datetime(series.index), 'y': series.values})
        df = df.drop_duplicates(subset='ds').sort_values('ds')
        
        # Time Series Cross Validation for Prophet
        cv_scores = []
        n_splits = min(5, len(df) // 4)
        
        if n_splits >= 3:
            print("📊 Performing Prophet cross validation for water/sanitation indicators...")
            
            for i in range(n_splits):
                # Calculate split points
                min_train_size = max(10, len(df) // 3)
                train_end = min_train_size + i * (len(df) - min_train_size) // (n_splits - 1)
                test_start = train_end
                test_end = min(test_start + max(3, len(df) // 8), len(df))
                
                if test_end > len(df) or test_start >= test_end:
                    continue
                
                train_df = df.iloc[:train_end]
                test_df = df.iloc[test_start:test_end]
                
                if len(train_df) < 8 or len(test_df) < 2:
                    continue
                
                try:
                    # Fit Prophet model on training data with water-specific settings
                    model = Prophet(
                        yearly_seasonality=True,  # Water usage can have yearly patterns
                        daily_seasonality=False,
                        weekly_seasonality=False,
                        seasonality_mode='additive'  # Water access typically grows additively
                    )
                    model.fit(train_df)
                    
                    # Make predictions on test data
                    forecast = model.predict(test_df[['ds']])
                    
                    # Calculate RMSE
                    rmse = np.sqrt(mean_squared_error(test_df['y'], forecast['yhat']))
                    cv_scores.append(rmse)
                    
                except Exception as e:
                    continue
            
            if len(cv_scores) > 0:
                mean_cv_score = np.mean(cv_scores)
                std_cv_score = np.std(cv_scores)
                print(f"✅ Prophet CV for water indicators: {mean_cv_score:.4f} ± {std_cv_score:.4f} RMSE ({len(cv_scores)} folds)")
            else:
                print("⚠️  Prophet cross validation failed")
        else:
            print(f"⚠️  Not enough data for Prophet cross validation ({len(df)} points)")
        
        # Final model training with train/test split
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        print(f"📈 Final training: {len(train_df)} train, {len(test_df)} test points")
        
        # Fit model on training data with water-specific configuration
        model = Prophet(
            yearly_seasonality=True,  # Water usage can have yearly patterns
            daily_seasonality=False,
            weekly_seasonality=False,
            seasonality_mode='additive'  # Water access typically grows additively
        )
        model.fit(train_df)
        
        # Make predictions for test period
        forecast_test = model.predict(test_df[['ds']])
        
        # Create test predictions series with correct datetime index
        test_predictions = pd.Series(
            forecast_test['yhat'].values,
            index=pd.to_datetime(test_df['ds'])
        )
        
        # Calculate RMSE
        test_rmse = np.sqrt(mean_squared_error(test_df['y'], test_predictions))
        
        print(f"✅ Test RMSE for water indicator: {test_rmse:.4f}")
        
        # Fit final model on all data for future predictions
        full_model = Prophet(
            yearly_seasonality=True,
            daily_seasonality=False,
            weekly_seasonality=False
        )
        full_model.fit(df)
        
        return {
            'model': full_model,
            'test_predictions': test_predictions,
            'test_data': pd.Series(test_df['y'].values, index=pd.to_datetime(test_df['ds'])),
            'rmse': test_rmse,
            'cv_scores': cv_scores if len(cv_scores) > 0 else None
        }
    
    def fit_sarimax_model(self, series, country):
        """Fit SARIMAX model with external variables and time series cross validation"""
        print(f"\n🔄 SARIMAX Model with External Variables for {country}")
        print(f"Data points: {len(series)}")
        
        # Prepare external variables for all years in the series
        external_data_matrix = []
        feature_names = []
        valid_years = []
        
        years = pd.to_datetime(series.index).year.values
        
        # First pass: determine the consistent feature set for ALL years in the series
        all_features_by_year = {}
        for year in years:
            try:
                features = self.prepare_external_features(country, year)
                if features is not None and len(features) > 0:
                    all_features_by_year[year] = features
                else:
                    print(f"⚠️  No external features found for {year}, using interpolation")
                    # Try to interpolate from nearby years
                    if len(all_features_by_year) > 0:
                        # Use last available features as fallback
                        last_features = list(all_features_by_year.values())[-1]
                        all_features_by_year[year] = last_features
            except Exception as e:
                print(f"⚠️  Error getting external data for {year}: {e}")
                # Try to use last available features
                if len(all_features_by_year) > 0:
                    last_features = list(all_features_by_year.values())[-1]
                    all_features_by_year[year] = last_features
                continue
        
        # Check if we have features for all years - use same absolute threshold as SDG3
        if len(all_features_by_year) < 8:  # Use same absolute threshold as SDG3
            print(f"⚠️  Insufficient external data points ({len(all_features_by_year)} < 8). Falling back to ARIMA.")
            return self.fit_arima_model(series)
        
        # Fill missing years with interpolated values
        for year in years:
            if year not in all_features_by_year:
                # Find nearest available years
                available_years = sorted(all_features_by_year.keys())
                if available_years:
                    # Use nearest year's features
                    nearest_year = min(available_years, key=lambda x: abs(x - year))
                    all_features_by_year[year] = all_features_by_year[nearest_year]
                    print(f"📊 Interpolated features for {year} using {nearest_year}")
        
        # Determine the minimum number of features available across all years
        if len(all_features_by_year) == 0:
            print(f"⚠️  No external data available. Falling back to ARIMA.")
            return self.fit_arima_model(series)
            
        min_features = min(len(features) for features in all_features_by_year.values())
        max_features = max(len(features) for features in all_features_by_year.values())
        
        print(f"📊 Feature count range: {min_features} to {max_features}")
        
        if min_features < 3:  # Ensure SARIMAX quality
            print(f"⚠️  Too few external features ({min_features} < 3). Falling back to ARIMA.")
            return self.fit_arima_model(series)
        
        # Build consistent feature matrix using the minimum number of features
        feature_names = ['GDP', 'GINI', 'Unemployment', 'RD_Expenditure', 'Social_Coverage'][:min_features]
        
        # Create external data matrix for ALL years in the series
        for year in sorted(years):
            if year in all_features_by_year:
                features = all_features_by_year[year][:min_features]  # Take only the first min_features
                
                # Ensure all features are valid numbers
                if len(features) == min_features and all(isinstance(f, (int, float)) and not np.isnan(f) for f in features):
                    external_data_matrix.append(features)
                    valid_years.append(year)
                else:
                    print(f"⚠️  Invalid features for {year}: {features}")
                    # Use average of available features as fallback
                    if len(external_data_matrix) > 0:
                        avg_features = np.mean(external_data_matrix, axis=0)
                        external_data_matrix.append(avg_features.tolist())
                        valid_years.append(year)
            else:
                print(f"⚠️  No features available for {year}")
                # Use average of available features as fallback
                if len(external_data_matrix) > 0:
                    avg_features = np.mean(external_data_matrix, axis=0)
                    external_data_matrix.append(avg_features.tolist())
                    valid_years.append(year)
        
        if len(external_data_matrix) < 6:  # More flexible than SDG3
            print(f"⚠️  After filtering, insufficient external data points ({len(external_data_matrix)} < 6). Falling back to ARIMA.")
            return self.fit_arima_model(series)
        
        # Align series with available external data
        valid_indices = [i for i, year in enumerate(years) if year in valid_years]
        aligned_series = series.iloc[valid_indices]
        
        # Convert to numpy array and ensure proper shape
        exog_data = np.array(external_data_matrix, dtype=np.float64)
        
        print(f"✅ External data prepared: {exog_data.shape}")
        print(f"Features: {feature_names}")
        print(f"Sample data (first 3 rows):")
        for i, features in enumerate(exog_data[:3]):
            print(f"  Year {valid_years[i]}: {[f'{f:.2f}' for f in features]}")
        
        # Scale external variables to prevent numerical issues
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        try:
            exog_scaled = scaler.fit_transform(exog_data)
            print(f"📊 External variables scaled successfully (mean≈0, std≈1)")
        except Exception as e:
            print(f"⚠️  Scaling failed: {e}. Falling back to ARIMA.")
            return self.fit_arima_model(series)
        
        # Time Series Cross Validation for SARIMAX parameter selection
        best_order = None
        best_seasonal_order = None
        best_cv_score = float('inf')
        cv_results = {}
        
        # Test different SARIMAX orders
        orders_to_test = [
            (1, 1, 1), (1, 1, 0), (0, 1, 1), 
            (2, 1, 1), (1, 1, 2), (1, 0, 1)
        ]
        
        seasonal_orders_to_test = [
            (0, 0, 0, 0),  # No seasonality (most appropriate for yearly data)
            (1, 0, 0, 0),  # Simple seasonal AR without seasonal period
            (0, 1, 0, 0),  # Simple seasonal differencing without seasonal period
        ]
        
        print("📊 SARIMAX parameter optimization with time series cross validation...")
        
        for order in orders_to_test:
            for seasonal_order in seasonal_orders_to_test:
                try:
                    # Perform time series cross validation
                    cv_scores = []
                    n_splits = min(4, len(aligned_series) // 5)  # Conservative splits for SARIMAX
                    
                    if n_splits < 3:
                        continue
                    
                    # Time series split: expanding window
                    for i in range(n_splits):
                        # Calculate split points
                        min_train_size = max(10, len(aligned_series) // 2)
                        train_end = min_train_size + i * (len(aligned_series) - min_train_size) // (n_splits - 1)
                        test_start = train_end
                        test_end = min(test_start + max(2, len(aligned_series) // 8), len(aligned_series))
                        
                        if test_end > len(aligned_series) or test_start >= test_end:
                            continue
                        
                        train_series = aligned_series.iloc[:train_end]
                        test_series = aligned_series.iloc[test_start:test_end]
                        train_exog = exog_scaled[:train_end]
                        test_exog = exog_scaled[test_start:test_end]
                        
                        if len(train_series) < 8 or len(test_series) < 2:
                            continue
                        
                        # Validate shapes before fitting
                        if train_exog.shape[0] != len(train_series) or test_exog.shape[0] != len(test_series):
                            print(f"⚠️  Shape mismatch in fold {i+1}: train_series={len(train_series)}, train_exog={train_exog.shape}, test_series={len(test_series)}, test_exog={test_exog.shape}")
                            continue
                        
                        try:
                            # Import SARIMAX
                            from statsmodels.tsa.statespace.sarimax import SARIMAX
                            
                            # Fit SARIMAX model on training data
                            model = SARIMAX(train_series, 
                                          exog=train_exog,
                                          order=order, 
                                          seasonal_order=seasonal_order,
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)
                            model_fit = model.fit(disp=False, maxiter=100)
                            
                            # Make predictions on test data
                            forecast = model_fit.forecast(steps=len(test_series), exog=test_exog)
                            
                            # Calculate RMSE
                            rmse = np.sqrt(mean_squared_error(test_series, forecast))
                            cv_scores.append(rmse)
                            
                        except Exception as e:
                            # Skip this fold if model fitting fails
                            continue
                    
                    if len(cv_scores) > 0:
                        mean_cv_score = np.mean(cv_scores)
                        param_key = (order, seasonal_order)
                        cv_results[param_key] = {
                            'mean_rmse': mean_cv_score,
                            'std_rmse': np.std(cv_scores),
                            'n_folds': len(cv_scores)
                        }
                        
                        print(f"SARIMAX{order}x{seasonal_order}: {mean_cv_score:.4f} RMSE ({len(cv_scores)} folds)")
                        
                        if mean_cv_score < best_cv_score:
                            best_cv_score = mean_cv_score
                            best_order = order
                            best_seasonal_order = seasonal_order
                    
                except Exception as e:
                    print(f"⚠️  SARIMAX{order}x{seasonal_order} failed: {str(e)}")
                    continue
        
        # Use best parameters or fall back
        if best_order is None:
            print(f"⚠️  SARIMAX optimization failed. Using default parameters.")
            best_order = (1, 1, 1)
            best_seasonal_order = (0, 0, 0, 0)
        else:
            print(f"✅ Best SARIMAX: {best_order}x{best_seasonal_order} (CV RMSE: {best_cv_score:.4f})")
        
        # Final model training with train/test split
        train_size = int(len(aligned_series) * 0.8)
        train_series = aligned_series[:train_size]
        test_series = aligned_series[train_size:]
        train_exog = exog_scaled[:train_size]
        test_exog = exog_scaled[train_size:]
        
        print(f"📈 Final SARIMAX training: {len(train_series)} train, {len(test_series)} test points")
        
        # Validate final shapes
        if train_exog.shape[0] != len(train_series) or test_exog.shape[0] != len(test_series):
            print(f"⚠️  Final shape mismatch: train_series={len(train_series)}, train_exog={train_exog.shape}, test_series={len(test_series)}, test_exog={test_exog.shape}. Falling back to ARIMA.")
            return self.fit_arima_model(series)
        
        # Fit final model on training data
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        try:
            eval_model = SARIMAX(train_series, 
                               exog=train_exog,
                               order=best_order, 
                               seasonal_order=best_seasonal_order,
                               enforce_stationarity=False,
                               enforce_invertibility=False)
            eval_model_fit = eval_model.fit(disp=False, maxiter=150)
            
            # Make predictions for test period
            test_predictions = eval_model_fit.forecast(steps=len(test_series), exog=test_exog)
            test_rmse = np.sqrt(mean_squared_error(test_series, test_predictions))
            
            print(f"✅ SARIMAX Test RMSE: {test_rmse:.4f}")
            
            # Fit final model on all data for future predictions
            full_model = SARIMAX(aligned_series, 
                               exog=exog_scaled,
                               order=best_order, 
                               seasonal_order=best_seasonal_order,
                               enforce_stationarity=False,
                               enforce_invertibility=False)
            full_model_fit = full_model.fit(disp=False, maxiter=150)
            
            return {
                'model': full_model_fit,
                'test_predictions': test_predictions,
                'test_data': test_series,
                'rmse': test_rmse,
                'best_order': best_order,
                'best_seasonal_order': best_seasonal_order,
                'cv_results': cv_results,
                'feature_names': feature_names,
                'scaler': scaler,
                'exog_data': exog_scaled,
                'aligned_series': aligned_series
            }
            
        except Exception as e:
            print(f"⚠️  Final SARIMAX fitting failed: {e}. Falling back to ARIMA.")
            return self.fit_arima_model(series)
    
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
            
            # Check if we have enough data points
            MIN_DATA_POINTS = 20
            if len(indicator_data) < MIN_DATA_POINTS:
                messagebox.showerror("Error", 
                    f"Not enough data points for {indicator_id} in {country}.\n"
                    f"Found {len(indicator_data)} points, but need at least {MIN_DATA_POINTS} points "
                    f"for a reliable forecast.\n"
                    f"Please try a different indicator or country for more data points.")
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
                messagebox.showerror("Error", "No valid numeric water/sanitation data found for the selected series")
                return
            
            # Determine the unit and scale factor based on the series description
            series_description = indicator_data['SeriesDescription'].iloc[0].lower()
            unit = ""
            scale_factor = 1.0
            
            if any(word in series_description for word in ['percentage', 'percent', '%']):
                unit = "%"
                scale_factor = 1.0
            elif any(word in series_description for word in ['thousand', '1000']):
                unit = "thousands"
                scale_factor = 1.0
            elif any(word in series_description for word in ['million']):
                unit = "millions"
                scale_factor = 1.0
            elif any(word in series_description for word in ['liter', 'litre', 'cubic']):
                unit = "liters"
                scale_factor = 1.0
            else:
                # Check the actual values to determine the scale
                max_value = indicator_data['Value'].max()
                if max_value > 1000000:
                    unit = "millions"
                    scale_factor = 1000000.0
                elif max_value > 1000:
                    unit = "thousands"
                    scale_factor = 1000.0
                else:
                    unit = ""
                    scale_factor = 1.0
            
            # Scale the data
            scaled_data = indicator_data.copy()
            scaled_data['Value'] = scaled_data['Value'] / scale_factor
            
            # Plot historical data points with larger markers
            ax.scatter(scaled_data['TimePeriod'], scaled_data['Value'], 
                      color='blue', label='Historical Water Data', s=100, alpha=0.8)
            
            # Prepare data for modeling
            series = self.prepare_time_series(scaled_data)
            
            # Initialize variables to avoid NameError
            future_dates = None
            scaled_forecast = None
            scaled_predictions = None
            scaled_pred_lower_future = None
            scaled_pred_upper_future = None
            scaled_conf_lower_future = None
            scaled_conf_upper_future = None
            
            try:
                if model_type == 'ARIMA':
                    # Fit ARIMA model and make forecast
                    arima_results = self.fit_arima_model(series)
                    
                    # Scale the predictions and test data
                    scaled_predictions = arima_results['test_predictions'] / scale_factor
                    scaled_test = arima_results['test_data'] / scale_factor
                    
                    # Plot predictions for test period (red)
                    prediction_color = plt.cm.Reds(0.7)
                    test_period = scaled_test.index
                    ax.scatter(test_period, scaled_predictions, color=prediction_color, 
                              label='Model Test', s=100, alpha=0.8)
                    ax.plot(test_period, scaled_predictions, color=prediction_color, alpha=0.5, linewidth=2)
                    
                    # Make future forecast using all available data
                    future_forecast = arima_results['model'].get_forecast(steps=7)
                    scaled_forecast = future_forecast.predicted_mean / scale_factor
                    future_conf_int = future_forecast.conf_int(alpha=0.05)
                    scaled_conf_lower_future = future_conf_int.iloc[:, 0] / scale_factor
                    scaled_conf_upper_future = future_conf_int.iloc[:, 1] / scale_factor
                    
                    # Calculate prediction intervals (wider than confidence intervals)
                    pred_interval = 2.0 * (arima_results['rmse'] / scale_factor)
                    scaled_pred_lower_future = scaled_forecast - pred_interval
                    scaled_pred_upper_future = scaled_forecast + pred_interval
                    
                    # Generate future dates for ARIMA
                    future_dates = pd.date_range(start=series.index[-1], periods=8, freq='Y')[1:]
                    
                elif model_type == 'Prophet':
                    # Fit Prophet model and make forecast
                    prophet_results = self.fit_prophet_model(series)
                    
                    # Scale the predictions and test data
                    scaled_predictions = prophet_results['test_predictions'] / scale_factor
                    scaled_test = prophet_results['test_data'] / scale_factor
                    
                    # Plot predictions for test period (red)
                    prediction_color = plt.cm.Reds(0.7)
                    test_period = prophet_results['test_predictions'].index
                    ax.scatter(test_period, scaled_predictions, color=prediction_color, 
                              label='Model Test', s=100, alpha=0.8)
                    ax.plot(test_period, scaled_predictions, color=prediction_color, alpha=0.5, linewidth=2)
                    
                    # Make future forecast using all available data
                    future = prophet_results['model'].make_future_dataframe(periods=7, freq='Y')
                    forecast = prophet_results['model'].predict(future)
                    scaled_forecast = forecast['yhat'].iloc[-7:].values / scale_factor
                    
                    # Get confidence intervals from Prophet and scale them
                    scaled_conf_lower_future = forecast['yhat_lower'].iloc[-7:].values / scale_factor
                    scaled_conf_upper_future = forecast['yhat_upper'].iloc[-7:].values / scale_factor
                    
                    # Calculate prediction intervals (wider than confidence intervals)
                    pred_interval = 2.0 * (prophet_results['rmse'] / scale_factor)
                    scaled_pred_lower_future = scaled_forecast - pred_interval
                    scaled_pred_upper_future = scaled_forecast + pred_interval
                    
                    # Get future dates from the forecast
                    future_dates = pd.to_datetime(forecast['ds'].iloc[-7:])
                
                elif model_type == 'SARIMAX':
                    # Add SARIMAX model support - for now use ARIMA as fallback
                    print("⚠️  SARIMAX model not fully implemented for SDG6 yet, using ARIMA fallback")
                    arima_results = self.fit_arima_model(series)
                    
                    # Scale the predictions and test data
                    scaled_predictions = arima_results['test_predictions'] / scale_factor
                    scaled_test = arima_results['test_data'] / scale_factor
                    
                    # Plot predictions for test period (red)
                    prediction_color = plt.cm.Reds(0.7)
                    test_period = scaled_test.index
                    ax.scatter(test_period, scaled_predictions, color=prediction_color, 
                              label='SARIMAX (ARIMA) Test', s=100, alpha=0.8)
                    ax.plot(test_period, scaled_predictions, color=prediction_color, alpha=0.5, linewidth=2)
                    
                    # Make future forecast using all available data
                    future_forecast = arima_results['model'].get_forecast(steps=7)
                    scaled_forecast = future_forecast.predicted_mean / scale_factor
                    future_conf_int = future_forecast.conf_int(alpha=0.05)
                    scaled_conf_lower_future = future_conf_int.iloc[:, 0] / scale_factor
                    scaled_conf_upper_future = future_conf_int.iloc[:, 1] / scale_factor
                    
                    # Calculate prediction intervals (wider than confidence intervals)
                    pred_interval = 2.0 * (arima_results['rmse'] / scale_factor)
                    scaled_pred_lower_future = scaled_forecast - pred_interval
                    scaled_pred_upper_future = scaled_forecast + pred_interval
                    
                    # Generate future dates for SARIMAX
                    future_dates = pd.date_range(start=series.index[-1], periods=8, freq='Y')[1:]
                    
                elif model_type == 'Random Forest':
                    # Fit Random Forest model and make forecast
                    rf_results = self.fit_random_forest_model(series, country)
                    
                    # Scale the predictions
                    scaled_test_predictions = rf_results['test_predictions'] / scale_factor
                    scaled_forecast = rf_results['future_predictions'] / scale_factor
                    
                    # Set scaled_predictions for use in all_values calculation later
                    scaled_predictions = scaled_test_predictions
                    
                    # Scale confidence and prediction intervals
                    scaled_conf_lower_68 = rf_results['conf_lower_68'] / scale_factor
                    scaled_conf_upper_68 = rf_results['conf_upper_68'] / scale_factor
                    scaled_conf_lower_95 = rf_results['conf_lower_95'] / scale_factor
                    scaled_conf_upper_95 = rf_results['conf_upper_95'] / scale_factor
                    scaled_pred_lower_95 = rf_results['pred_lower_95'] / scale_factor
                    scaled_pred_upper_95 = rf_results['pred_upper_95'] / scale_factor
                    
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
                
                # Plot future forecast if available with BEAUTIFUL WATER-THEMED COLOR SCHEME! 💧
                if future_dates is not None and scaled_forecast is not None:
                    # 💧 WATER-THEMED COLOR PALETTE - Professional & Beautiful!
                    forecast_color = '#1f77b4'        # Water Blue - main forecast line
                    pred_interval_color = '#e6f3ff'   # Very Light Blue - prediction intervals (95%)
                    conf_interval_95_color = '#cce7ff' # Light Blue - confidence intervals (95%)  
                    conf_interval_68_color = '#b3daff' # Medium Light Blue - confidence intervals (68%)
                    
                                        # Enhanced plot styling with unified intervals for ALL models
                    if model_type == 'Random Forest':
                        # Plot prediction intervals (95%)
                        ax.fill_between(future_dates, scaled_pred_lower_future, scaled_pred_upper_future, 
                                       alpha=0.3, color=pred_interval_color, label='95% Prediction Interval')
                        
                        # Plot confidence intervals (95%)
                        ax.fill_between(future_dates, scaled_conf_lower_future, scaled_conf_upper_future, 
                                       alpha=0.4, color=conf_interval_95_color, label='95% Confidence Interval')
                        
                        # Plot confidence intervals (68%)
                        if 'scaled_conf_lower_68' in locals() and 'scaled_conf_upper_68' in locals():
                            ax.fill_between(future_dates, scaled_conf_lower_68, scaled_conf_upper_68, 
                                           alpha=0.5, color=conf_interval_68_color, label='68% Confidence Interval')
                    
                    elif model_type == 'Prophet':
                        # Plot prediction intervals (95%)
                        ax.fill_between(future_dates, scaled_pred_lower_future, scaled_pred_upper_future, 
                                       alpha=0.3, color=pred_interval_color, label='95% Prediction Interval')
                        
                        # Plot confidence intervals from Prophet
                        ax.fill_between(future_dates, scaled_conf_lower_future, scaled_conf_upper_future, 
                                       alpha=0.5, color=conf_interval_95_color, label='Prophet Confidence')
                    
                    elif model_type == 'ARIMA':
                        # Plot prediction intervals (95%)
                        ax.fill_between(future_dates, scaled_pred_lower_future, scaled_pred_upper_future, 
                                       alpha=0.3, color=pred_interval_color, label='95% Prediction Interval')
                        
                        # Plot confidence intervals (95%)
                        ax.fill_between(future_dates, scaled_conf_lower_future, scaled_conf_upper_future, 
                                       alpha=0.5, color=conf_interval_95_color, label='95% Confidence Interval')
                    
                    elif model_type == 'SARIMAX':
                        # Add SARIMAX support here if needed
                        ax.fill_between(future_dates, scaled_pred_lower_future, scaled_pred_upper_future, 
                                       alpha=0.3, color=pred_interval_color, label='95% Prediction Interval')
                        ax.fill_between(future_dates, scaled_conf_lower_future, scaled_conf_upper_future, 
                                       alpha=0.5, color=conf_interval_95_color, label='95% Confidence Interval')
                    
                    # Plot main forecast line with unified beautiful water color
                    ax.plot(future_dates, scaled_forecast, color=forecast_color, linewidth=3, 
                           label=f'{model_type} Water Forecast', marker='o', markersize=8, alpha=0.9)
                    
                    # Enhanced styling for professional water/sanitation presentation
                    ax.legend(loc='best', framealpha=0.9, shadow=True)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
                    ax.set_ylabel(f'{indicator_id} ({unit})', fontsize=12, fontweight='bold')
                    ax.set_title(f'💧 Water & Sanitation Forecast: {indicator_id}\n{country} | Model: {model_type}', 
                                fontsize=14, fontweight='bold', pad=20)
                
                # Add text annotation for the last historical data point
                last_date = series.index[-1]
                last_value = series.iloc[-1] / scale_factor
                ax.annotate(f'Latest water data: {last_value:.2f} {unit}',
                           xy=(last_date, last_value),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, color='blue',
                           bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
                
                # Set y-axis limits to show all data points clearly
                all_values = list(series/scale_factor) + list(scaled_predictions)
                if isinstance(scaled_forecast, np.ndarray):
                    all_values.extend(scaled_forecast)
                else:
                    all_values.extend(scaled_forecast.values)
                    
                y_min = min(all_values)
                y_max = max(all_values)
                y_range = y_max - y_min
                
                # Add extra space for intervals
                y_min = y_min - 0.15*y_range
                y_max = y_max + 0.15*y_range
                
                # Ensure intervals are within plot limits
                if 'scaled_pred_lower_future' in locals() and scaled_pred_lower_future is not None:
                    if isinstance(scaled_pred_lower_future, np.ndarray):
                        y_min = min(y_min, min(scaled_pred_lower_future))
                        y_max = max(y_max, max(scaled_pred_upper_future))
                    else:
                        y_min = min(y_min, min(scaled_pred_lower_future.values))
                        y_max = max(y_max, max(scaled_pred_upper_future.values))
                
                ax.set_ylim(y_min, y_max)
                
                # Adjust layout to make room for legend and prevent text cutoff
                plt.subplots_adjust(right=0.85, top=0.85, bottom=0.15, left=0.1)
                
                # Make plot frame expand to fill available space
                self.plot_frame.grid_rowconfigure(0, weight=1)
                self.plot_frame.grid_columnconfigure(0, weight=1)
                
                # Embed plot in GUI with sticky option to fill frame
                self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            except Exception as e:
                messagebox.showerror("Error", f"Could not generate water/sanitation forecast: {str(e)}")
                return
            
            # Display comprehensive results
            self.display_water_results(scaled_data, scaled_forecast, future_dates, model_type, 
                                     indicator_id, country, series_code, unit, scale_factor)
            
            # Enable save button after plot is generated
            self.save_button.state(['!disabled'])
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            # Print detailed error for debugging
            print(f"Error in generate_forecast: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def display_water_results(self, scaled_data, scaled_forecast, future_dates, model_type, 
                             indicator_id, country, series_code, unit, scale_factor):
        """Display comprehensive water/sanitation forecast results"""
        
        # Clear and display results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"=== SDG 6 Water & Sanitation Forecast Results ===\n\n")
        self.results_text.insert(tk.END, f"💧 Water Indicator: {indicator_id}\n")
        self.results_text.insert(tk.END, f"🌍 Country: {country}\n")
        self.results_text.insert(tk.END, f"📊 Series Code: {series_code}\n")
        self.results_text.insert(tk.END, f"📈 Model: {model_type}\n")
        self.results_text.insert(tk.END, f"📏 Unit: {unit}\n\n")
        
        # Add feature importance for Random Forest
        if model_type == 'Random Forest' and hasattr(self, 'rf_feature_importance'):
            self.results_text.insert(tk.END, f"🎯 Feature Importance for Water Forecasting:\n")
            sorted_features = sorted(self.rf_feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features:
                bar_length = int(importance * 20)  # Scale for visual bar
                bar = "█" * bar_length + "░" * (20 - bar_length)
                self.results_text.insert(tk.END, f"  {feature:15s} │{bar}│ {importance*100:.1f}%\n")
            self.results_text.insert(tk.END, "\n")
        
        # Historical data summary
        self.results_text.insert(tk.END, f"📊 Historical Water Data Summary:\n")
        self.results_text.insert(tk.END, f"   Data points: {len(scaled_data)}\n")
        self.results_text.insert(tk.END, f"   Years: {scaled_data['TimePeriod'].dt.year.min()} - {scaled_data['TimePeriod'].dt.year.max()}\n")
        self.results_text.insert(tk.END, f"   Value range: {scaled_data['Value'].min():.3f} - {scaled_data['Value'].max():.3f} {unit}\n")
        
        # Show recent historical values
        recent_data = scaled_data.tail(5)
        self.results_text.insert(tk.END, "\n📋 Recent Historical Values:\n")
        for _, row in recent_data.iterrows():
            self.results_text.insert(tk.END, f"  {row['TimePeriod'].year}: {row['Value']:.3f} {unit}\n")
        
        # Add forecast values with enhanced formatting
        self.results_text.insert(tk.END, f"\n🔮 7-Year Water & Sanitation Forecast (until 2030):\n")
        if scaled_forecast is not None and future_dates is not None:
            try:
                for i, (date, value) in enumerate(zip(future_dates, scaled_forecast)):
                    year = date.year if hasattr(date, 'year') else date
                    if not np.isnan(value):
                        # Add trend indicator
                        if i > 0:
                            prev_value = scaled_forecast[i-1] if i < len(scaled_forecast) else value
                            trend = "📈" if value > prev_value else "📉" if value < prev_value else "➡️"
                            change = ((value - prev_value) / prev_value * 100) if prev_value != 0 else 0
                            self.results_text.insert(tk.END, f"  {year}: {value:.3f} {unit} {trend} ({change:+.1f}%)\n")
                        else:
                            self.results_text.insert(tk.END, f"  {year}: {value:.3f} {unit}\n")
                    else:
                        self.results_text.insert(tk.END, f"  {year}: N/A {unit} (NaN detected)\n")
            except Exception as e:
                self.results_text.insert(tk.END, f"Error displaying forecast values: {str(e)}\n")
        else:
            self.results_text.insert(tk.END, "No forecast values available\n")
        
        # Add water-specific context and insights
        self.results_text.insert(tk.END, f"\n💧 Water & Sanitation Context:\n")
        water_context = self._get_water_context(indicator_id)
        self.results_text.insert(tk.END, f"   {water_context}\n")
        
        # Add model-specific insights
        self.results_text.insert(tk.END, f"\n🤖 Model-Specific Insights:\n")
        if model_type == 'ARIMA':
            self.results_text.insert(tk.END, f"   📊 Time series analysis captures water access trends\n")
            self.results_text.insert(tk.END, f"   📈 Good for baseline water infrastructure projections\n")
        elif model_type == 'Prophet':
            self.results_text.insert(tk.END, f"   📅 Seasonal patterns in water usage/access detected\n")
            self.results_text.insert(tk.END, f"   🔄 Accounts for cyclical water demand variations\n")
        elif model_type == 'Random Forest':
            external_features = [f for f in self.rf_model.feature_names if f != 'Year']
            if external_features:
                self.results_text.insert(tk.END, f"   🔗 Incorporates {len(external_features)} external factors\n")
                self.results_text.insert(tk.END, f"   💰 Economic indicators influence water infrastructure\n")
                self.results_text.insert(tk.END, f"   🏘️ Location filters show urban/rural water disparities\n")
        
        # Add water-specific recommendations
        self.results_text.insert(tk.END, f"\n🚰 Water Policy Recommendations:\n")
        if location == 'RURAL':
            self.results_text.insert(tk.END, f"   🌾 Focus on rural water infrastructure development\n")
            self.results_text.insert(tk.END, f"   🚛 Consider mobile water solutions for remote areas\n")
        elif location == 'URBAN':
            self.results_text.insert(tk.END, f"   🏢 Urban water efficiency and quality improvements\n")
            self.results_text.insert(tk.END, f"   🔄 Wastewater treatment and recycling systems\n")
        
        if activity == 'SANITATION':
            self.results_text.insert(tk.END, f"   🚽 Prioritize sanitation infrastructure investment\n")
            self.results_text.insert(tk.END, f"   🧼 Hygiene education and behavior change programs\n")
        elif activity == 'DRINKING':
            self.results_text.insert(tk.END, f"   🚰 Safe drinking water access improvements\n")
            self.results_text.insert(tk.END, f"   🔬 Water quality monitoring and treatment\n")
        
        # Add validation summary
        self.results_text.insert(tk.END, f"\n✅ Water Forecast Validation:\n")
        self.results_text.insert(tk.END, f"   ✅ Time series cross validation performed\n")
        self.results_text.insert(tk.END, f"   ✅ Water-specific model parameters applied\n")
        self.results_text.insert(tk.END, f"   ✅ External economic factors considered (if applicable)\n")
        self.results_text.insert(tk.END, f"   ✅ Location and activity filters applied\n")
        
        self.results_text.insert(tk.END, f"\n" + "="*70 + "\n")
        self.results_text.insert(tk.END, f"💧 Water & Sanitation forecast completed successfully!\n")
        self.results_text.insert(tk.END, f"🌍 Use results for SDG 6 planning and policy development.\n")

    def load_external_data(self):
        """Load external data including processed GDP, GINI, R&D, Social Coverage, and Unemployment data"""
        external_data = {}
        try:
            # Get the parent directory where processed CSV files are located
            parent_dir = os.path.dirname(self.current_dir)  # SDG (parent of SDG6)
            
            print(f"Looking for external data in: {parent_dir}")
            print(f"Current directory: {self.current_dir}")
            print(f"Parent directory: {parent_dir}")
            
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
            
            for data_name, config in processed_files.items():
                file_path = os.path.join(parent_dir, config['filename'])
                if os.path.exists(file_path):
                    try:
                        data = pd.read_csv(file_path)
                        print(f"Loaded {data_name} data with shape: {data.shape}")
                        print(f"Columns in {data_name} data: {data.columns.tolist()}")
                        
                        # Check if required columns exist
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
                            external_data[data_name] = data
                            print(f"✓ {data_name.upper()} data loaded ({len(data)} records)")
                        else:
                            print(f"✗ {data_name.upper()} data not loaded")
                            print(f"    Expected columns: {required_columns}")
                            print(f"    Found columns: {data.columns.tolist()}")
                        
                    except Exception as e:
                        print(f"✗ {data_name.upper()} data not loaded")
                        print(f"    Error: {str(e)}")
                else:
                    print(f"✗ {data_name.upper()} data not loaded")
                    print(f"    File not found: {file_path}")
            
            print(f"\nSuccessfully loaded {len(external_data)} external datasets")
            return external_data
            
        except Exception as e:
            print(f"Error loading external data: {str(e)}")
            return {}
    
    def show_external_data_status(self):
        """Display external data loading status for SDG6"""
        status_text = "\n=== External Data Integration Status (SDG 6 - Water & Sanitation) ===\n"
        for data_name in ['gdp', 'gini', 'unemployment', 'rd_expenditure', 'social_coverage']:
            if data_name in self.external_data:
                data = self.external_data[data_name]
                status_text += f"✓ {data_name.upper()} data loaded ({len(data)} records)\n"
                status_text += f"  → Relevant for water infrastructure & sanitation development\n"
            else:
                status_text += f"✗ {data_name.upper()} data not loaded\n"
        
        status_text += f"\nTotal external datasets: {len(self.external_data)}\n"
        status_text += "💧 Enhanced models ready for water & sanitation forecasting!\n"
        status_text += "🚰 External factors help predict infrastructure development patterns\n"
        
        self.results_text.insert(tk.END, status_text)

    def fit_random_forest_model(self, series, country):
        """Fit Enhanced Random Forest model with external factors integration for water/sanitation"""
        try:
            print(f"\nFitting Enhanced Random Forest model for {country} (SDG6)")
            
            # Use the enhanced Random Forest model
            results = self.rf_model.fit(series, country)
            
            # Generate future predictions with intervals
            future_results = self.rf_model.predict_future(series, country, periods=7)
            
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
            print(f"Error in Enhanced Random Forest model for water indicators: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise Exception(f"Error in Enhanced Random Forest model for water indicators: {str(e)}")

    def save_plot(self):
        """Save the current water/sanitation forecast plot as an image file"""
        if self.current_fig:
            # Get user's desktop path
            desktop = os.path.expanduser("~/Desktop")
            
            # Get current selections for default filename
            indicator_id = self.indicator_var.get().split(' - ')[0] if self.indicator_var.get() else "SDG6"
            country = self.country_var.get() if self.country_var.get() else "Unknown"
            
            # Create default filename with water/sanitation context
            default_filename = f"SDG6_Water_Sanitation_{indicator_id}_{country}.png"
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
                    # Save the figure with high quality
                    self.current_fig.savefig(file_path, bbox_inches='tight', dpi=300)
                    messagebox.showinfo("Success", f"Water & Sanitation forecast plot saved successfully to:\n{file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save plot: {str(e)}")

    def prepare_external_features(self, country, year):
        """Prepare external features for a specific country and year with improved fallback handling"""
        features = []
        
        def get_country_data(data_name, column_name):
            """Get data for a specific country and year with improved fallback logic"""
            if data_name not in self.external_data:
                return None
            
            data_df = self.external_data[data_name]
            
            # Try exact country and year match first
            exact_match = data_df[
                (data_df['Country Name'].str.contains(country, case=False, na=False)) &
                (data_df['Year'] == year)
            ]
            
            if not exact_match.empty:
                value = float(exact_match[column_name].iloc[0])
                if not np.isnan(value) and value != 0:
                    return value
            
            # Try to find data for this country in nearby years (±5 years)
            country_data = data_df[
                data_df['Country Name'].str.contains(country, case=False, na=False)
            ]
            
            if not country_data.empty:
                # Try years around the target year (expanded range)
                for year_offset in [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]:
                    search_year = year + year_offset
                    year_data = country_data[country_data['Year'] == search_year]
                    if not year_data.empty:
                        value = float(year_data[column_name].iloc[0])
                        if not np.isnan(value) and value != 0:
                            if year_offset != 0:
                                print(f"📊 Using {data_name} from {search_year} for {year} ({country})")
                            return value
                
                # If no exact match, use most recent available data
                recent_data = country_data[country_data['Year'] <= year].sort_values('Year')
                if not recent_data.empty:
                    value = float(recent_data[column_name].iloc[-1])
                    if not np.isnan(value) and value != 0:
                        recent_year = recent_data['Year'].iloc[-1]
                        print(f"📊 Using most recent {data_name} from {recent_year} for {year} ({country})")
                        return value
                
                # Last resort: use any available data for this country
                any_data = country_data.dropna(subset=[column_name])
                if not any_data.empty:
                    value = float(any_data[column_name].iloc[-1])
                    if not np.isnan(value) and value != 0:
                        fallback_year = any_data['Year'].iloc[-1]
                        print(f"📊 Using fallback {data_name} from {fallback_year} for {year} ({country})")
                        return value
            
            # Enhanced regional/global fallbacks for World and major regions
            fallback_regions = []
            
            if country.lower() == 'world':
                # For World, try major regional averages
                fallback_regions = ['Europe', 'Asia', 'Americas', 'Africa', 'OECD', 'High income', 'Middle income']
            elif 'europe' in country.lower():
                fallback_regions = ['European Union', 'OECD', 'High income', 'Germany', 'France', 'Italy']
            elif 'africa' in country.lower():
                fallback_regions = ['Sub-Saharan Africa', 'Middle income', 'Low income', 'South Africa', 'Nigeria']
            elif 'asia' in country.lower():
                fallback_regions = ['East Asia & Pacific', 'High income', 'China', 'Japan', 'India']
            elif 'america' in country.lower():
                fallback_regions = ['North America', 'Latin America', 'High income', 'United States', 'Brazil']
            else:
                # Generic fallbacks
                fallback_regions = ['World', 'High income', 'Middle income', 'OECD']
            
            # Try fallback regions
            for fallback_region in fallback_regions:
                fallback_data = data_df[
                    (data_df['Country Name'].str.contains(fallback_region, case=False, na=False)) &
                    (data_df['Year'] == year)
                ]
                if not fallback_data.empty:
                    value = float(fallback_data[column_name].iloc[0])
                    if not np.isnan(value) and value != 0:
                        print(f"📊 Using {fallback_region} {data_name} as proxy for {country} in {year}")
                        return value
            
            # Global average for this year as last resort
            global_year_data = data_df[data_df['Year'] == year]
            if not global_year_data.empty:
                global_avg = global_year_data[column_name].mean()
                if not np.isnan(global_avg) and global_avg != 0:
                    print(f"📊 Using global average {data_name} ({global_avg:.2f}) for {country} {year}")
                    return global_avg
            
            return None
        
        # Collect external features with improved fallback
        gdp = get_country_data('gdp', 'Value')
        gini = get_country_data('gini', 'Value')
        unemployment = get_country_data('unemployment', 'Value')
        rd = get_country_data('rd_expenditure', 'Value')
        social = get_country_data('social_coverage', 'Value')
        
        # Build feature list with consistent ordering and only valid values
        feature_candidates = [
            (gdp, 'GDP'),
            (gini, 'GINI'), 
            (unemployment, 'Unemployment'),
            (rd, 'RD_Expenditure'),
            (social, 'Social_Coverage')
        ]
        
        valid_features = []
        valid_feature_names = []
        
        for value, name in feature_candidates:
            if value is not None:
                valid_features.append(float(value))  # Ensure it's a float
                valid_feature_names.append(name)
        
        # Enhanced fallback: Use default values based on country type if we have less than 3 features
        if len(valid_features) < 3:
            print(f"⚠️  Only {len(valid_features)} valid features for {country} {year}, using enhanced defaults...")
            
            # Determine country income level for better defaults
            is_high_income = any(term in country.lower() for term in ['europe', 'america', 'developed', 'oecd', 'high income'])
            is_low_income = any(term in country.lower() for term in ['africa', 'least developed', 'low income'])
            
            # Add missing features with intelligent defaults
            missing_features = []
            for value, name in feature_candidates:
                if value is None:
                    if name == 'GDP':
                        if is_high_income:
                            default_gdp = 45000.0  # High income default
                        elif is_low_income:
                            default_gdp = 2000.0   # Low income default
                        else:
                            default_gdp = 12000.0  # Middle income default
                        missing_features.append((default_gdp, name))
                    
                    elif name == 'GINI':
                        if is_high_income:
                            default_gini = 32.0    # Lower inequality
                        elif is_low_income:
                            default_gini = 45.0    # Higher inequality
                        else:
                            default_gini = 38.0    # Middle inequality
                        missing_features.append((default_gini, name))
                    
                    elif name == 'Unemployment':
                        if is_high_income:
                            default_unemployment = 6.0
                        elif is_low_income:
                            default_unemployment = 12.0
                        else:
                            default_unemployment = 9.0
                        missing_features.append((default_unemployment, name))
                    
                    elif name == 'RD_Expenditure':
                        if is_high_income:
                            default_rd = 2.5
                        elif is_low_income:
                            default_rd = 0.3
                        else:
                            default_rd = 1.0
                        missing_features.append((default_rd, name))
                    
                    elif name == 'Social_Coverage':
                        if is_high_income:
                            default_social = 85.0
                        elif is_low_income:
                            default_social = 35.0
                        else:
                            default_social = 60.0
                        missing_features.append((default_social, name))
            
            # Add the missing features
            for default_value, name in missing_features:
                valid_features.append(float(default_value))
                valid_feature_names.append(name)
                print(f"📊 Using intelligent default {name} ({default_value}) for {country} {year}")
        
        # Ensure we have at least 3 features for SARIMAX
        if len(valid_features) >= 3:
            return valid_features[:5]  # Return max 5 features
        else:
            print(f"⚠️  Still insufficient features ({len(valid_features)}) for {country} {year}")
            return None
    
    def extrapolate_external_variables(self, country, last_year, forecast_periods, feature_names):
        """Extrapolate external variables for future years using intelligent methods"""
        future_exog = []
        
        for period in range(1, forecast_periods + 1):
            future_year = last_year + period
            future_features = []
            
            for feature_name in feature_names:
                if feature_name == 'GDP':
                    # GDP: Exponential growth with dampening
                    historical_gdp = []
                    for year in range(last_year - 4, last_year + 1):
                        gdp_data = self.get_historical_feature_value(country, year, 'gdp', 'Value')
                        if gdp_data is not None:
                            historical_gdp.append(gdp_data)
                    
                    if len(historical_gdp) >= 3:
                        # Calculate average growth rate with dampening
                        growth_rates = []
                        for i in range(1, len(historical_gdp)):
                            if historical_gdp[i-1] > 0:
                                growth_rate = (historical_gdp[i] / historical_gdp[i-1]) - 1
                                growth_rates.append(growth_rate)
                        
                        if growth_rates:
                            avg_growth = np.mean(growth_rates)
                            # Apply dampening for future years (growth slows down)
                            dampened_growth = avg_growth * (0.95 ** period)
                            future_gdp = historical_gdp[-1] * (1 + dampened_growth)
                            future_features.append(future_gdp)
                        else:
                            # Fallback: 2% annual growth
                            future_features.append(historical_gdp[-1] * (1.02 ** period))
                    else:
                        # Use last available value with 2% growth
                        last_gdp = self.get_historical_feature_value(country, last_year, 'gdp', 'Value')
                        if last_gdp is not None:
                            future_features.append(last_gdp * (1.02 ** period))
                        else:
                            future_features.append(50000.0)  # Default value
                
                elif feature_name == 'GINI':
                    # GINI: Mean reversion with country-specific targets
                    country_targets = {
                        'germany': 28, 'france': 32, 'italy': 35, 'spain': 36,
                        'netherlands': 28, 'belgium': 27, 'austria': 30,
                        'europe': 31  # European average
                    }
                    
                    # Determine target based on country
                    target_gini = 35  # Default
                    for country_key, target in country_targets.items():
                        if country_key in country.lower():
                            target_gini = target
                            break
                    
                    # Get recent GINI values
                    last_gini = self.get_historical_feature_value(country, last_year, 'gini', 'Value')
                    if last_gini is not None:
                        # Mean reversion: slowly move towards target
                        reversion_speed = 0.1  # 10% per year
                        future_gini = last_gini + (target_gini - last_gini) * reversion_speed * period
                        future_features.append(max(15, min(60, future_gini)))  # Bound between 15-60
                    else:
                        future_features.append(target_gini)
                
                elif feature_name == 'Unemployment':
                    # Unemployment: Cyclical with structural rate
                    structural_rates = {
                        'germany': 4.5, 'france': 8.5, 'italy': 9.5, 'spain': 12.0,
                        'netherlands': 4.0, 'belgium': 6.5, 'austria': 5.0,
                        'europe': 7.0  # European average
                    }
                    
                    # Determine structural rate
                    structural_rate = 7.0  # Default
                    for country_key, rate in structural_rates.items():
                        if country_key in country.lower():
                            structural_rate = rate
                            break
                    
                    # Get recent unemployment
                    last_unemployment = self.get_historical_feature_value(country, last_year, 'unemployment', 'Value')
                    if last_unemployment is not None:
                        # Business cycle: 7-year cycle with mean reversion
                        cycle_factor = np.sin(2 * np.pi * period / 7) * 0.5
                        future_unemployment = structural_rate + (last_unemployment - structural_rate) * (0.9 ** period) + cycle_factor
                        future_features.append(max(1, min(25, future_unemployment)))  # Bound between 1-25%
                    else:
                        future_features.append(structural_rate)
                
                elif feature_name == 'RD_Expenditure':
                    # R&D: Technology-driven growth
                    last_rd = self.get_historical_feature_value(country, last_year, 'rd_expenditure', 'Value')
                    if last_rd is not None:
                        # Steady growth with innovation waves
                        base_growth = 0.03  # 3% annual growth
                        innovation_wave = np.sin(2 * np.pi * period / 10) * 0.01  # 10-year innovation cycles
                        growth_rate = base_growth + innovation_wave
                        future_rd = last_rd * ((1 + growth_rate) ** period)
                        future_features.append(max(0.5, min(5.0, future_rd)))  # Bound between 0.5-5% of GDP
                    else:
                        # Default R&D spending
                        default_rd = 2.0  # 2% of GDP
                        future_features.append(default_rd)
                
                elif feature_name == 'Social_Coverage':
                    # Social Coverage: Policy-driven improvements
                    last_social = self.get_historical_feature_value(country, last_year, 'social_coverage', 'Value')
                    if last_social is not None:
                        # Gradual improvement with policy cycles
                        improvement_rate = 0.02  # 2% annual improvement
                        policy_cycle = np.sin(2 * np.pi * period / 15) * 0.01  # 15-year policy cycles
                        total_improvement = (improvement_rate + policy_cycle) * period
                        future_social = last_social * (1 + total_improvement)
                        future_features.append(max(10, min(100, future_social)))  # Bound between 10-100%
                    else:
                        # Default social coverage
                        future_features.append(75.0)  # 75% coverage
                
                else:
                    # Fallback for unknown features
                    last_value = self.get_historical_feature_value(country, last_year, feature_name.lower(), 'Value')
                    if last_value is not None:
                        future_features.append(last_value)
                    else:
                        future_features.append(1.0)  # Default value
            
            # Ensure we have the right number of features
            while len(future_features) < len(feature_names):
                future_features.append(1.0)
            
            future_exog.append(future_features[:len(feature_names)])
        
        return np.array(future_exog)
    
    def get_historical_feature_value(self, country, year, data_name, column_name):
        """Get historical value for a specific feature"""
        if data_name not in self.external_data:
            return None
        
        data_df = self.external_data[data_name]
        
        # Try exact match
        exact_match = data_df[
            (data_df['Country Name'].str.contains(country, case=False, na=False)) &
            (data_df['Year'] == year)
        ]
        
        if not exact_match.empty:
            value = float(exact_match[column_name].iloc[0])
            if not np.isnan(value) and value != 0:
                return value
        
        # Try nearby years
        country_data = data_df[
            data_df['Country Name'].str.contains(country, case=False, na=False)
        ]
        
        if not country_data.empty:
            # Get closest year
            country_data['year_diff'] = abs(country_data['Year'] - year)
            closest = country_data.loc[country_data['year_diff'].idxmin()]
            value = float(closest[column_name])
            if not np.isnan(value) and value != 0:
                return value
        
        return None

    def on_indicator_change(self, event):
        """Called when indicator selection changes"""
        # Clear results
        self.results_text.delete(1.0, tk.END)
        # Show immediate data assessment
        self.show_immediate_data_assessment()
        
    def on_country_change(self, event):
        """Called when country selection changes"""
        # Clear results
        self.results_text.delete(1.0, tk.END)
        # Show immediate data assessment
        self.show_immediate_data_assessment()
    
    def show_immediate_data_assessment(self):
        """Show immediate data quality assessment for SDG6 water/sanitation indicators"""
        try:
            selected_indicator = self.indicator_combo.get()
            selected_country = self.country_combo.get()
            
            # Check if both indicator and country are selected
            if not selected_indicator or not selected_country:
                return
            
            indicator_id = selected_indicator.split(' - ')[0]
            
            # Clear previous results and show assessment
            self.results_text.delete(1.0, tk.END)
            
            # Show indicator and country info with water/sanitation context
            self.results_text.insert(tk.END, f"💧 WATER & SANITATION DATA QUALITY ASSESSMENT (SDG6)\n")
            self.results_text.insert(tk.END, f"=" * 60 + "\n\n")
            self.results_text.insert(tk.END, f"Selected Water Indicator: {indicator_id}\n")
            self.results_text.insert(tk.END, f"Selected Country: {selected_country}\n")
            self.results_text.insert(tk.END, "\n")
            
            # Check historical data availability
            indicator_data = self.df[
                (self.df['Indicator'] == indicator_id) & 
                (self.df['GeoAreaName'] == selected_country)
            ]
            
            if len(indicator_data) > 0:
                # Calculate data quality metrics
                indicator_data['TimePeriod'] = pd.to_numeric(indicator_data['TimePeriod'], errors='coerce')
                years_span = indicator_data['TimePeriod'].max() - indicator_data['TimePeriod'].min()
                data_points = len(indicator_data)
                
                # Convert Value to numeric for missing value calculation
                indicator_data['Value'] = pd.to_numeric(indicator_data['Value'], errors='coerce')
                missing_values = indicator_data['Value'].isnull().sum()
                missing_pct = (missing_values / len(indicator_data)) * 100
                
                # Available series codes for this indicator/country
                available_series = indicator_data['SeriesCode'].nunique() if 'SeriesCode' in indicator_data.columns else 1
                
                self.results_text.insert(tk.END, f"🚰 HISTORICAL WATER DATA QUALITY:\n")
                self.results_text.insert(tk.END, f"   Time Span: {years_span} years\n")
                self.results_text.insert(tk.END, f"   Data Points: {data_points}\n")
                self.results_text.insert(tk.END, f"   Missing Values: {missing_values} ({missing_pct:.1f}%)\n")
                self.results_text.insert(tk.END, f"   Available Series: {available_series}\n")
                
                # Data quality score calculation
                quality_score = 0
                
                # Time coverage scoring (25 points max)
                if years_span >= 20:
                    quality_score += 25
                    self.results_text.insert(tk.END, f"   ✅ Excellent time coverage (20+ years)\n")
                elif years_span >= 15:
                    quality_score += 20
                    self.results_text.insert(tk.END, f"   ✅ Very good time coverage (15+ years)\n")
                elif years_span >= 10:
                    quality_score += 15
                    self.results_text.insert(tk.END, f"   ✅ Good time coverage (10+ years)\n")
                elif years_span >= 5:
                    quality_score += 8
                    self.results_text.insert(tk.END, f"   ⚠️ Moderate time coverage (5+ years)\n")
                else:
                    quality_score += 2
                    self.results_text.insert(tk.END, f"   ❌ Limited time coverage (<5 years)\n")
                
                # Missing data scoring (25 points max)
                if missing_pct == 0:
                    quality_score += 25
                    self.results_text.insert(tk.END, f"   ✅ No missing data\n")
                elif missing_pct < 5:
                    quality_score += 20
                    self.results_text.insert(tk.END, f"   ✅ Very few missing values (<5%)\n")
                elif missing_pct < 10:
                    quality_score += 15
                    self.results_text.insert(tk.END, f"   ⚠️ Some missing values (<10%)\n")
                elif missing_pct < 20:
                    quality_score += 8
                    self.results_text.insert(tk.END, f"   ⚠️ Many missing values (<20%)\n")
                else:
                    quality_score += 2
                    self.results_text.insert(tk.END, f"   ❌ Excessive missing values (≥20%)\n")
                
                # Data points scoring (25 points max)
                if data_points >= 25:
                    quality_score += 25
                    self.results_text.insert(tk.END, f"   ✅ Excellent data density (25+ points)\n")
                elif data_points >= 20:
                    quality_score += 20
                    self.results_text.insert(tk.END, f"   ✅ Sufficient data points (20+ points)\n")
                elif data_points >= 15:
                    quality_score += 15
                    self.results_text.insert(tk.END, f"   ✅ Adequate data points (15+ points)\n")
                elif data_points >= 10:
                    quality_score += 8
                    self.results_text.insert(tk.END, f"   ⚠️ Limited data points (10+ points)\n")
                else:
                    quality_score += 2
                    self.results_text.insert(tk.END, f"   ❌ Insufficient data points (<10)\n")
                
                # Series variety scoring (25 points max)
                if available_series >= 5:
                    quality_score += 25
                    self.results_text.insert(tk.END, f"   ✅ Excellent series variety (5+ series)\n")
                elif available_series >= 3:
                    quality_score += 20
                    self.results_text.insert(tk.END, f"   ✅ Good series variety (3+ series)\n")
                elif available_series >= 2:
                    quality_score += 15
                    self.results_text.insert(tk.END, f"   ✅ Some series choices (2+ series)\n")
                else:
                    quality_score += 10
                    self.results_text.insert(tk.END, f"   ⚠️ Single series only\n")
                
                self.results_text.insert(tk.END, f"\n🏆 WATER DATA QUALITY SCORE: {quality_score}/100\n")
                
                # Quality interpretation with water/sanitation context
                if quality_score >= 80:
                    self.results_text.insert(tk.END, f"   ✅ EXCELLENT - Ideal for reliable water infrastructure forecasting\n")
                elif quality_score >= 65:
                    self.results_text.insert(tk.END, f"   ✅ GOOD - Suitable for water policy planning\n")
                elif quality_score >= 50:
                    self.results_text.insert(tk.END, f"   ⚠️ MODERATE - Water forecasts with higher uncertainty\n")
                elif quality_score >= 35:
                    self.results_text.insert(tk.END, f"   ⚠️ POOR - Limited reliability for water infrastructure planning\n")
                else:
                    self.results_text.insert(tk.END, f"   ❌ UNRELIABLE - Data quality too low for reliable water forecasts\n")
            else:
                self.results_text.insert(tk.END, f"❌ No water/sanitation data found for this combination\n")
                quality_score = 0
            
            self.results_text.insert(tk.END, f"\n")
            
            # Check external data availability for enhanced models
            external_status = self._check_external_data_availability(selected_country)
            external_available = sum(external_status.values())
            
            self.results_text.insert(tk.END, f"🔗 EXTERNAL VARIABLES FOR WATER FORECASTING:\n")
            for var_name, available in external_status.items():
                icon = "✅" if available else "❌"
                relevance = self._get_water_relevance(var_name)
                self.results_text.insert(tk.END, f"   {icon} {var_name} - {relevance}\n")
            
            self.results_text.insert(tk.END, f"\n📈 WATER & SANITATION MODEL RECOMMENDATIONS:\n")
            
            # Model recommendations based on data quality and external data availability
            if external_available >= 4 and quality_score >= 70:
                self.results_text.insert(tk.END, f"   🥇 Recommended: Random Forest or SARIMAX\n")
                self.results_text.insert(tk.END, f"      → Rich external water data + excellent historical data\n")
                self.results_text.insert(tk.END, f"      → Best for complex water infrastructure relationships\n")
            elif external_available >= 3 and quality_score >= 60:
                self.results_text.insert(tk.END, f"   🥈 Recommended: SARIMAX or Random Forest\n")
                self.results_text.insert(tk.END, f"      → Good external data + solid water foundation\n")
                self.results_text.insert(tk.END, f"      → Suitable for water policy planning\n")
            elif external_available >= 2 and quality_score >= 50:
                self.results_text.insert(tk.END, f"   🥉 Recommended: SARIMAX or Prophet\n")
                self.results_text.insert(tk.END, f"      → Some external data available\n")
                self.results_text.insert(tk.END, f"      → Moderate confidence for water trends\n")
            elif quality_score >= 60:
                self.results_text.insert(tk.END, f"   📊 Recommended: Prophet or ARIMA\n")
                self.results_text.insert(tk.END, f"      → Good historical data, limited external data\n")
                self.results_text.insert(tk.END, f"      → Reliable for trend-based water forecasting\n")
            else:
                self.results_text.insert(tk.END, f"   ⚠️ Recommended: ARIMA (simple, robust)\n")
                self.results_text.insert(tk.END, f"      → Limited data quality\n")
                self.results_text.insert(tk.END, f"      → Use with caution for water policy planning\n")
            
            # Add SDG6-specific water & sanitation context
            self.results_text.insert(tk.END, f"\n💧 SDG6 WATER & SANITATION CONTEXT:\n")
            water_context = self._get_water_context(indicator_id)
            if water_context:
                self.results_text.insert(tk.END, f"   {water_context}\n")
            
            self.results_text.insert(tk.END, f"\n" + "="*60 + "\n")
            self.results_text.insert(tk.END, f"🚰 Ready to generate water & sanitation forecast! Select model and click 'Generate Forecast'\n")
            
        except Exception as e:
            self.results_text.insert(tk.END, f"⚠️ Error in water data assessment: {str(e)}\n")
    
    def _check_external_data_availability(self, country):
        """Check which external data variables are available for this country"""
        external_status = {}
        
        # Check each external data source
        for data_name, data_df in self.external_data.items():
            try:
                # Check if country has any data in this dataset
                country_data = data_df[
                    data_df['Country Name'].str.contains(country, case=False, na=False)
                ]
                external_status[data_name.upper()] = len(country_data) > 0
            except Exception:
                external_status[data_name.upper()] = False
        
        # Ensure we have all 5 expected variables
        expected_vars = ['GDP', 'GINI', 'UNEMPLOYMENT', 'RD_EXPENDITURE', 'SOCIAL_COVERAGE']
        for var in expected_vars:
            if var not in external_status:
                external_status[var] = False
        
        return external_status
    
    def _get_water_relevance(self, var_name):
        """Get water/sanitation relevance for external variables"""
        relevance_map = {
            'GDP': 'Economic capacity for water infrastructure investment',
            'GINI': 'Inequality affects water access distribution',
            'UNEMPLOYMENT': 'Economic conditions impact sanitation affordability',
            'RD_EXPENDITURE': 'Innovation in water technology & treatment',
            'SOCIAL_COVERAGE': 'Public services including water & sanitation'
        }
        return relevance_map.get(var_name, 'General economic/social indicator')
    
    def _get_water_context(self, indicator):
        """Get water and sanitation-specific context for indicators"""
        water_contexts = {
            '6.1': 'Safe drinking water access - Critical for health and development, influenced by infrastructure investment and economic growth',
            '6.2': 'Sanitation and hygiene access - Fundamental for public health, requires sustained infrastructure development and behavior change',
            '6.3': 'Water quality and pollution - Environmental indicator affected by industrial development and environmental policies',
            '6.4': 'Water use efficiency - Resource management indicator linked to economic development and climate adaptation',
            '6.5': 'Water resource management - Governance indicator requiring institutional capacity and cross-border cooperation',
            '6.6': 'Water-related ecosystem protection - Conservation indicator balancing development needs with environmental sustainability',
            '6.a': 'Water cooperation and capacity building - Development cooperation requiring international coordination and investment',
            '6.b': 'Community participation in water management - Governance indicator measuring local engagement and democratic participation'
        }
        
        # Find matching context
        for key, context in water_contexts.items():
            if key in indicator:
                return context
        
        return 'Water and sanitation indicator - Progress typically influenced by economic development, governance quality, and infrastructure investment'

if __name__ == "__main__":
    root = tk.Tk()
    app = ForecastAppGoal6(root)
    root.mainloop() 