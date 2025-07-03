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
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

class SDGRandomForestModel:
    """
    Enhanced Random Forest model specifically designed for SDG indicators
    that incorporates GDP and other external factors
    """
    
    def __init__(self, external_data):
        self.external_data = external_data
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def prepare_features_for_country_year(self, country, year):
        """Prepare feature vector for a specific country and year"""
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
        
        # Add GDP data if available
        if 'gdp' in self.external_data:
            gdp_value = get_country_year_value(self.external_data['gdp'], country, year, 'Value')
            features.append(gdp_value)
            feature_names.append('GDP')
            gdp_available = (gdp_value != 0.0)
        
        # Add GINI data if available
        if 'gini' in self.external_data:
            gini_value = get_country_year_value(self.external_data['gini'], country, year, 'Value')
            features.append(gini_value)
            feature_names.append('GINI')
            gini_available = (gini_value != 0.0)
        
        # Add Unemployment data if available
        if 'unemployment' in self.external_data:
            unemployment_value = get_country_year_value(self.external_data['unemployment'], country, year, 'Value')
            features.append(unemployment_value)
            feature_names.append('Unemployment')
            unemployment_available = (unemployment_value != 0.0)
        
        # Add R&D Expenditure data if available
        if 'rd_expenditure' in self.external_data:
            rd_value = get_country_year_value(self.external_data['rd_expenditure'], country, year, 'Value')
            features.append(rd_value)
            feature_names.append('R&D Expenditure')
            rd_available = (rd_value != 0.0)
        
        # Add Social Coverage data if available
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
        """Fit the Random Forest model"""
        print(f"\nFitting Enhanced Random Forest model for {country}")
        
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
                # Überprüfen, ob series.loc[year] eine Series oder ein einzelner Wert ist
                value = series.loc[year]
                
                # Wenn value eine Series ist, nehmen wir den Mittelwert oder den ersten Wert
                if isinstance(value, pd.Series):
                    value = value.mean()  # oder value.iloc[0] für den ersten Wert
                
                if pd.notna(value):
                    features, feature_names = self.prepare_features_for_country_year(country, year)
                    features_list.append(features)
                    targets.append(value)
                    years_list.append(year)
            except Exception as e:
                print(f"Error processing year {year}: {e}")
                continue
        
        # Debug-Ausgabe einfügen
        print(f"Processed {len(series)} years, created {len(features_list)} feature vectors")
        if len(features_list) == 0:
            print(f"Series has {len(series)} values: {series.values}")
            # Versuche mit leeren Features fortzufahren
            features = [year]  # Mindestens das Jahr als Feature
            feature_names = ['Year']
            for year in sorted(series.index):
                if pd.notna(series.loc[year]):
                    features_list.append(features)
                    targets.append(series.loc[year])
                    years_list.append(year)
        
        if len(features_list) == 0:
            raise ValueError("No valid training data available. Please check if there's enough historical data for this series.")
        
        self.feature_names = feature_names
        X = np.array(features_list)
        y = np.array(targets)
        years_array = np.array(years_list)
        
        print(f"Training data shape: {X.shape}")
        print(f"Feature names: {self.feature_names}")
        print(f"Years range: {years_array.min()} to {years_array.max()}")
        
        # If we have very few samples, adjust the number of estimators to avoid overfitting
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
        print(f"Trend calculation: slope={slope:.4f}, intercept={intercept:.4f}")
        
        return {
            'train_predictions': pd.Series(train_predictions, index=train_datetime_indices),
            'test_predictions': pd.Series(test_predictions, index=test_datetime_indices),
            'rmse': rmse,
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
        }
    
    def predict_future(self, series, country, periods=5):
        """Make future predictions with confidence and prediction intervals using intelligent extrapolation"""
        # Get the last year from the series
        if not all(isinstance(x, (int, np.integer)) for x in series.index):
            last_year = pd.to_datetime(series.index).year.max()
        else:
            last_year = max(series.index)
            
        # Only forecast future years - this is the key fix!
        future_years = range(last_year + 1, last_year + periods + 1)
        
        print(f"🔮 Random Forest forecasting from {last_year + 1} to {last_year + periods} ({periods} periods)")
        
        # STEP 1: Enhanced trend analysis with multiple methods
        slope = self.trend_params['slope']
        intercept = self.trend_params['intercept']
        last_year_trend = self.trend_params['last_year']
        last_value = self.trend_params['last_value']
        
        # Get historical data for advanced analysis
        years_hist = pd.to_datetime(series.index).year.values if not all(isinstance(x, (int, np.integer)) for x in series.index) else series.index.values
        values_hist = series.values
        
        # STEP 2: Intelligent feature-based extrapolation for external variables
        future_features_enhanced = []
        for i, year in enumerate(future_years):
            period = i + 1  # Period relative to last historical year
            enhanced_features = []
            
            # Get base features for this FUTURE year only
            base_features, feature_names = self.prepare_features_for_country_year(country, year)
            
            if len(base_features) >= 3:  # We have external data
                # GDP: Exponential growth with dampening and business cycles
                if 'GDP' in str(feature_names) or len(base_features) > 0:
                    last_gdp = base_features[0] if len(base_features) > 0 else 50000
                    # 7-year business cycle + dampening growth
                    business_cycle = np.sin(2 * np.pi * period / 7) * 0.03  # ±3% cycle
                    base_growth = 0.025  # 2.5% base growth
                    dampening = (0.98 ** period)  # Slight dampening over time
                    growth_rate = (base_growth * dampening) + business_cycle
                    
                    # Anti-bubble mechanism
                    if growth_rate > 0.06:  # Cap at 6%
                        growth_rate = 0.06 - (growth_rate - 0.06) * 0.5
                    
                    future_gdp = last_gdp * ((1 + growth_rate) ** period)
                    enhanced_features.append(future_gdp)
                
                # GINI: Country-specific mean reversion with policy cycles
                if len(base_features) > 1:
                    last_gini = base_features[1]
                    # Country-specific targets
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
                    
                    # 12-year policy cycle + mean reversion
                    policy_cycle = np.sin(2 * np.pi * period / 12) * 2.0  # ±2 GINI points
                    reversion_speed = 0.08  # 8% per year towards target
                    future_gini = last_gini + (target_gini - last_gini) * reversion_speed * period + policy_cycle
                    future_gini = max(15, min(60, future_gini))  # Bound between 15-60
                    enhanced_features.append(future_gini)
                
                # Unemployment: Structural rate + anti-cyclical business cycle
                if len(base_features) > 2:
                    last_unemployment = base_features[2]
                    # Country-specific structural rates
                    structural_rates = {
                        'germany': 4.5, 'france': 8.5, 'italy': 9.5, 'spain': 12.0,
                        'united states': 5.5, 'brazil': 11.0, 'china': 4.0,
                        'world': 7.0, 'europe': 7.5, 'africa': 12.0
                    }
                    
                    structural_rate = 7.0  # Default
                    for country_key, rate in structural_rates.items():
                        if country_key in country.lower():
                            structural_rate = rate
                            break
                    
                    # Anti-cyclical: when GDP grows, unemployment falls
                    gdp_effect = -business_cycle * 0.8  # Opposite of GDP cycle
                    mean_reversion = (structural_rate - last_unemployment) * 0.15 * period
                    future_unemployment = last_unemployment + mean_reversion + gdp_effect
                    future_unemployment = max(1, min(25, future_unemployment))  # Bound 1-25%
                    enhanced_features.append(future_unemployment)
                
                # R&D: Tech-driven growth with innovation waves
                if len(base_features) > 3:
                    last_rd = base_features[3]
                    # 10-year innovation waves + steady growth
                    innovation_wave = np.sin(2 * np.pi * period / 10) * 0.15  # ±0.15% of GDP
                    base_rd_growth = 0.03  # 3% annual growth
                    future_rd = last_rd * ((1 + base_rd_growth) ** period) + innovation_wave
                    future_rd = max(0.1, min(5.0, future_rd))  # Bound 0.1-5% of GDP
                    enhanced_features.append(future_rd)
                
                # Social Coverage: Policy-driven improvements
                if len(base_features) > 4:
                    last_social = base_features[4]
                    # 15-year policy cycles + gradual improvement
                    policy_cycle = np.sin(2 * np.pi * period / 15) * 3.0  # ±3% coverage
                    improvement_rate = 0.015  # 1.5% annual improvement
                    future_social = last_social * (1 + improvement_rate * period) + policy_cycle
                    future_social = max(10, min(100, future_social))  # Bound 10-100%
                    enhanced_features.append(future_social)
                
                # Pad with remaining features if needed
                while len(enhanced_features) < len(base_features):
                    enhanced_features.append(base_features[len(enhanced_features)])
                    
                future_features_enhanced.append(enhanced_features[:len(base_features)])
            else:
                # Fallback to original features if no external data
                future_features_enhanced.append(base_features)
        
        # STEP 3: Model predictions with enhanced features
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
            
            print(f"Enhanced external features used for {len(future_features)} future periods")
            print(f"Model prediction std: {np.mean(prediction_std):.4f}")
        else:
            # Fallback to trend if no features available
            model_predictions = np.array([last_value + slope * period for period in range(1, periods + 1)])
            prediction_std = np.ones_like(model_predictions) * abs(last_value) * 0.1
            print("Using trend-based predictions (no external features)")
        
        # STEP 4: Enhanced trend predictions with economic realism
        trend_predictions = []
        for i, year in enumerate(future_years):
            period = i + 1
            years_since_last = period
            
            # Base trend
            base_trend = last_value + (slope * years_since_last)
            
            # Add economic cycles and volatility
            # Long-term cycle (8-year economic cycle)
            long_cycle = np.sin(2 * np.pi * period / 8) * abs(last_value) * 0.05
            
            # Short-term volatility (2-3 year cycle)
            short_cycle = np.sin(2 * np.pi * period / 2.5) * abs(last_value) * 0.02
            
            # Trend dampening for extreme slopes
            if abs(slope) > abs(last_value) * 0.1:  # If slope > 10% of current value
                dampening_factor = 0.95 ** period
                dampened_slope = slope * dampening_factor
                trend_prediction = last_value + (dampened_slope * years_since_last)
            else:
                trend_prediction = base_trend
            
            # Add cycles
            trend_prediction += long_cycle + short_cycle
            
            trend_predictions.append(trend_prediction)
            print(f"Year {year}: Enhanced trend = {trend_prediction:.2f} (base: {base_trend:.2f})")
        
        trend_predictions = np.array(trend_predictions)
        
        # STEP 5: Intelligent combination of model and trend
        if np.std(model_predictions) < 0.005 * np.mean(np.abs(model_predictions)):
            print("⚠️  Model predictions too similar - using enhanced trend predictions")
            future_predictions = trend_predictions
            combination_weights = "100% Enhanced Trend"
        else:
            # Dynamic weighting based on prediction consistency
            model_consistency = 1.0 / (1.0 + np.std(model_predictions) / np.mean(np.abs(model_predictions)))
            trend_weight = 0.3 + (1.0 - model_consistency) * 0.4  # 30-70% trend weight
            model_weight = 1.0 - trend_weight
            
            future_predictions = trend_weight * trend_predictions + model_weight * model_predictions
            combination_weights = f"{model_weight*100:.0f}% Enhanced RF, {trend_weight*100:.0f}% Enhanced Trend"
            print(f"Combined predictions: {combination_weights}")
        
        # STEP 6: Enhanced uncertainty estimation
        # Base uncertainty from model
        base_std = np.maximum(prediction_std, np.abs(future_predictions) * 0.03)  # At least 3%
        
        # Add uncertainty from economic cycles and external shocks
        cycle_uncertainty = np.abs(future_predictions) * 0.02 * np.sqrt(np.arange(1, periods + 1))  # Increases with time
        external_shock_uncertainty = np.abs(future_predictions) * 0.04  # 4% for external shocks
        
        total_uncertainty = np.sqrt(base_std**2 + cycle_uncertainty**2 + external_shock_uncertainty**2)
        
        # Calculate realistic confidence and prediction intervals
        confidence_interval_68 = 1.0 * total_uncertainty  # ±1σ
        confidence_interval_95 = 2.0 * total_uncertainty  # ±2σ  
        prediction_interval_95 = 2.8 * total_uncertainty  # Wider for individual predictions
        
        print(f"Enhanced uncertainty: base={np.mean(base_std):.3f}, cycle={np.mean(cycle_uncertainty):.3f}, external={np.mean(external_shock_uncertainty):.3f}")
        print(f"Future predictions range: {np.min(future_predictions):.2f} to {np.max(future_predictions):.2f}")
        print(f"Prediction intervals: ±{np.mean(prediction_interval_95):.2f}")
        print(f"✅ Generated {len(future_predictions)} predictions for years {min(future_years)}-{max(future_years)}")
        
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

class ForecastAppGoal3:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SDG Goal 3 Indicator Forecast with Multiple External Factors (GDP, GINI, Unemployment, R&D, Social Coverage)")
        self.root.geometry("1400x900")
        
        # Get current directory
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load data
        self.df = self.load_data()
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
        self.selection_frame = ttk.LabelFrame(self.main_frame, text="Model Selection & Parameters", padding="10")
        self.selection_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Create a PanedWindow for resizable plot and results areas
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.VERTICAL)
        self.paned_window.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.plot_frame = ttk.LabelFrame(self.paned_window, text="Forecast Plot", padding="10")
        self.results_frame = ttk.LabelFrame(self.paned_window, text="Results & Feature Analysis", padding="10")
        
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
        """Load the processed SDG data"""
        try:
            file_path = os.path.join(self.current_dir, 'Goal3_processed.csv')
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load SDG data: {str(e)}")
            return None
    
    def load_external_data(self):
        """Load external data including processed GDP, GINI, R&D, Social Coverage, and Unemployment data"""
        external_data = {}
        try:
            # Get the parent directory where processed CSV files are located
            sdg3_dir = os.path.dirname(self.current_dir)  # SDG3
            parent_dir = os.path.dirname(sdg3_dir)  # SDG
            
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
        """Display external data loading status"""
        status_text = "\n=== External Data Integration Status (SDG 3) ===\n"
        for data_name in ['gdp', 'gini', 'unemployment', 'rd_expenditure', 'social_coverage']:
            if data_name in self.external_data:
                data = self.external_data[data_name]
                status_text += f"✓ {data_name.upper()} data loaded ({len(data)} records)\n"
            else:
                status_text += f"✗ {data_name.upper()} data not loaded\n"
        
        status_text += f"\nTotal external datasets: {len(self.external_data)}\n"
        status_text += "Random Forest model ready for enhanced predictions!\n"
        
        self.results_text.insert(tk.END, status_text)
    
    def get_available_indicators(self):
        """Get list of available indicators with their descriptions"""
        indicators = self.df[['Indicator', 'SeriesDescription']].drop_duplicates()
        return indicators.sort_values('Indicator')
    
    def get_available_countries(self, indicator_id):
        """Get list of available countries for a specific indicator"""
        countries = self.df[self.df['Indicator'] == indicator_id]['GeoAreaName'].unique()
        return sorted(countries)
    
    def get_available_series_codes(self, indicator_id, country):
        """Get list of available series codes for a specific indicator and country"""
        series_codes = self.df[
            (self.df['Indicator'] == indicator_id) & 
            (self.df['GeoAreaName'] == country)
        ]['SeriesCode'].unique()
        return sorted(series_codes)
    
    def create_selection_widgets(self):
        # Model selection
        ttk.Label(self.selection_frame, text="Select Model:").grid(row=0, column=0, padx=5, pady=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(self.selection_frame, textvariable=self.model_var)
        self.model_combo['values'] = ['ARIMA', 'Prophet', 'Random Forest', 'SARIMAX']
        self.model_combo.set('ARIMA')
        self.model_combo.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Indicator selection
        ttk.Label(self.selection_frame, text="Select Indicator:").grid(row=1, column=0, padx=5, pady=5)
        self.indicator_var = tk.StringVar()
        self.indicator_combo = ttk.Combobox(self.selection_frame, textvariable=self.indicator_var, width=70)
        self.indicator_combo['values'] = [f"{ind} - {desc}" for ind, desc in zip(self.indicators['Indicator'], self.indicators['SeriesDescription'])]
        self.indicator_combo.grid(row=1, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.indicator_combo.bind('<<ComboboxSelected>>', self.update_countries)
        
        # Country selection
        ttk.Label(self.selection_frame, text="Select Country:").grid(row=2, column=0, padx=5, pady=5)
        self.country_var = tk.StringVar()
        self.country_combo = ttk.Combobox(self.selection_frame, textvariable=self.country_var)
        self.country_combo.grid(row=2, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.country_combo.bind('<<ComboboxSelected>>', self.update_series_codes)
        
        # Series Code selection
        ttk.Label(self.selection_frame, text="Select Series Code:").grid(row=3, column=0, padx=5, pady=5)
        self.series_code_var = tk.StringVar()
        self.series_code_combo = ttk.Combobox(self.selection_frame, textvariable=self.series_code_var)
        self.series_code_combo.grid(row=3, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Gender selection (SDG3 specific)
        ttk.Label(self.selection_frame, text="Select Gender:").grid(row=4, column=0, padx=5, pady=5)
        self.gender_var = tk.StringVar()
        self.gender_combo = ttk.Combobox(self.selection_frame, textvariable=self.gender_var)
        self.gender_combo['values'] = ['BOTHSEX', 'MALE', 'FEMALE']
        self.gender_combo.set('BOTHSEX')
        self.gender_combo.grid(row=4, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Age selection (SDG3 specific)
        ttk.Label(self.selection_frame, text="Select Age Group:").grid(row=5, column=0, padx=5, pady=5)
        self.age_var = tk.StringVar()
        self.age_combo = ttk.Combobox(self.selection_frame, textvariable=self.age_var)
        self.age_combo['values'] = ['ALLAGE', '<15Y', '15-24', '15+', '25+', '65+']
        self.age_combo.set('ALLAGE')
        self.age_combo.grid(row=5, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Forecast button
        self.forecast_button = ttk.Button(self.selection_frame, text="Generate Forecast", command=self.generate_forecast)
        self.forecast_button.grid(row=6, column=0, columnspan=2, padx=5, pady=5)
    
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
                # Trigger immediate data quality assessment
                self.show_immediate_data_assessment()
    
    def update_series_codes(self, event=None):
        """Update series code combobox when country is selected"""
        selected = self.indicator_var.get()
        country = self.country_var.get()
        if selected and country:
            indicator_id = selected.split(' - ')[0]
            series_codes = self.get_available_series_codes(indicator_id, country)
            self.series_code_combo['values'] = series_codes
            if series_codes:
                self.series_code_combo.set(series_codes[0])
                # Trigger immediate data quality assessment
                self.show_immediate_data_assessment()
    
    def show_immediate_data_assessment(self):
        """Show immediate data quality assessment when selections change"""
        try:
            selected_indicator = self.indicator_var.get()
            selected_country = self.country_var.get()
            selected_series = self.series_code_var.get()
            
            if not selected_indicator or not selected_country:
                return
            
            indicator_id = selected_indicator.split(' - ')[0]
            
            # Clear previous results and show assessment
            self.results_text.delete(1.0, tk.END)
            
            # Show indicator and country info
            self.results_text.insert(tk.END, f"🎯 DATA QUALITY ASSESSMENT (SDG3)\n")
            self.results_text.insert(tk.END, f"=" * 50 + "\n\n")
            self.results_text.insert(tk.END, f"Selected Indicator: {indicator_id}\n")
            self.results_text.insert(tk.END, f"Selected Country: {selected_country}\n")
            if selected_series:
                self.results_text.insert(tk.END, f"Selected Series: {selected_series}\n")
            self.results_text.insert(tk.END, "\n")
            
            # Check historical data availability using hierarchical search
            indicator_data, data_source = self.get_hierarchical_data(
                indicator_id, selected_country, selected_series or '', 
                self.gender_var.get(), self.age_var.get()
            )
            
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
                available_series = indicator_data['SeriesCode'].nunique()
                
                self.results_text.insert(tk.END, f"📊 HISTORICAL DATA QUALITY:\n")
                self.results_text.insert(tk.END, f"   Time Span: {years_span} years\n")
                self.results_text.insert(tk.END, f"   Data Points: {data_points}\n")
                self.results_text.insert(tk.END, f"   Missing Values: {missing_values} ({missing_pct:.1f}%)\n")
                self.results_text.insert(tk.END, f"   Available Series: {available_series}\n")
                
                # Add data source information
                if "Aggregated from" in data_source:
                    self.results_text.insert(tk.END, f"   📋 Data Source: Hierarchical aggregation\n")
                    self.results_text.insert(tk.END, f"   ℹ️  {data_source}\n")
                elif "Similar region" in data_source:
                    self.results_text.insert(tk.END, f"   📋 Data Source: Similar region proxy\n")
                    self.results_text.insert(tk.END, f"   ℹ️  {data_source}\n")
                else:
                    self.results_text.insert(tk.END, f"   📋 Data Source: Direct country data\n")
                
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
                
                self.results_text.insert(tk.END, f"\n🏆 DATA QUALITY SCORE: {quality_score}/100\n")
                
                # Quality interpretation
                if quality_score >= 80:
                    self.results_text.insert(tk.END, f"   ✅ EXCELLENT - Ideal for reliable health forecasting\n")
                elif quality_score >= 65:
                    self.results_text.insert(tk.END, f"   ✅ GOOD - Suitable for health policy forecasting\n")
                elif quality_score >= 50:
                    self.results_text.insert(tk.END, f"   ⚠️ MODERATE - Forecasts with higher uncertainty\n")
                elif quality_score >= 35:
                    self.results_text.insert(tk.END, f"   ⚠️ POOR - Limited forecasting reliability\n")
                else:
                    self.results_text.insert(tk.END, f"   ❌ UNRELIABLE - Data quality too low for reliable forecasts\n")
            else:
                self.results_text.insert(tk.END, f"❌ No data found for this combination\n")
                quality_score = 0
            
            self.results_text.insert(tk.END, f"\n")
            
            # Check external data availability for enhanced models
            external_status = self._check_external_data_availability(selected_country)
            external_available = sum(external_status.values())
            
            self.results_text.insert(tk.END, f"🔗 EXTERNAL VARIABLES AVAILABILITY (SDG3):\n")
            for var_name, available in external_status.items():
                icon = "✅" if available else "❌"
                self.results_text.insert(tk.END, f"   {icon} {var_name}\n")
            
            self.results_text.insert(tk.END, f"\n📈 ENHANCED MODELS AVAILABLE:\n")
            
            # Model recommendations based on data quality and external data availability
            if external_available >= 4 and quality_score >= 70:
                self.results_text.insert(tk.END, f"   🥇 Recommended: Random Forest or SARIMAX\n")
                self.results_text.insert(tk.END, f"      → Rich external health data + excellent historical data\n")
                self.results_text.insert(tk.END, f"      → Best for complex health indicator relationships\n")
            elif external_available >= 3 and quality_score >= 60:
                self.results_text.insert(tk.END, f"   🥈 Recommended: SARIMAX or Random Forest\n")
                self.results_text.insert(tk.END, f"      → Good external data + solid historical foundation\n")
                self.results_text.insert(tk.END, f"      → Suitable for health policy planning\n")
            elif external_available >= 2 and quality_score >= 50:
                self.results_text.insert(tk.END, f"   🥉 Recommended: SARIMAX or Prophet\n")
                self.results_text.insert(tk.END, f"      → Some external data available\n")
                self.results_text.insert(tk.END, f"      → Moderate confidence for health trends\n")
            elif quality_score >= 60:
                self.results_text.insert(tk.END, f"   📊 Recommended: Prophet or ARIMA\n")
                self.results_text.insert(tk.END, f"      → Good historical data, limited external data\n")
                self.results_text.insert(tk.END, f"      → Reliable for trend-based health forecasting\n")
            else:
                self.results_text.insert(tk.END, f"   ⚠️ Recommended: ARIMA (simple, robust)\n")
                self.results_text.insert(tk.END, f"      → Limited data quality\n")
                self.results_text.insert(tk.END, f"      → Use with caution for health planning\n")
            
            # Add SDG3-specific health indicators context
            self.results_text.insert(tk.END, f"\n🏥 SDG3 HEALTH CONTEXT:\n")
            if "3.1" in indicator_id:
                self.results_text.insert(tk.END, f"   📋 Maternal Health: Critical for MDG monitoring\n")
            elif "3.2" in indicator_id:
                self.results_text.insert(tk.END, f"   👶 Child Health: Essential for development tracking\n")
            elif "3.3" in indicator_id:
                self.results_text.insert(tk.END, f"   🦠 Infectious Disease: Epidemic preparedness focus\n")
            elif "3.4" in indicator_id:
                self.results_text.insert(tk.END, f"   💊 Non-communicable Disease: Lifestyle & aging trends\n")
            elif "3.5" in indicator_id:
                self.results_text.insert(tk.END, f"   🚭 Substance Abuse: Social determinants important\n")
            elif "3.6" in indicator_id:
                self.results_text.insert(tk.END, f"   🚗 Road Safety: Infrastructure & policy dependent\n")
            elif "3.7" in indicator_id:
                self.results_text.insert(tk.END, f"   🤱 Reproductive Health: Gender equality linked\n")
            elif "3.8" in indicator_id:
                self.results_text.insert(tk.END, f"   🏥 Universal Health Coverage: System capacity key\n")
            elif "3.9" in indicator_id:
                self.results_text.insert(tk.END, f"   ☠️ Environmental Health: Pollution & climate sensitive\n")
            
            self.results_text.insert(tk.END, f"\n" + "="*50 + "\n")
            self.results_text.insert(tk.END, f"💡 Ready to generate health forecast! Select model and click 'Generate Forecast'\n")
            
        except Exception as e:
            self.results_text.insert(tk.END, f"⚠️ Error in data assessment: {str(e)}\n")
    
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
    
    def prepare_time_series(self, data):
        """Prepare time series data for modeling - aggregate to one value per year"""
        data['TimePeriod'] = pd.to_datetime(data['TimePeriod'], format='%Y')
        
        # Group by year and take the mean if there are multiple values per year
        data_grouped = data.groupby(data['TimePeriod'].dt.year).agg({
            'Value': 'mean'  # Take average if multiple values per year
        }).reset_index()
        
        # Convert back to datetime with year only
        data_grouped['TimePeriod'] = pd.to_datetime(data_grouped['TimePeriod'], format='%Y')
        data_grouped = data_grouped.set_index('TimePeriod')
        data_grouped = data_grouped.sort_index()
        
        print(f"📊 Time series aggregated: {len(data)} original points → {len(data_grouped)} annual points")
        
        return data_grouped['Value']
    
    def fit_arima_model(self, series):
        """Fit ARIMA model to the time series with proper time series cross validation"""
        print(f"\n🔄 ARIMA Model with Time Series Cross Validation")
        print(f"Data points: {len(series)}")
        
        # Time Series Cross Validation for ARIMA order selection
        best_order = None
        best_cv_score = float('inf')
        cv_results = {}
        
        # Test different ARIMA orders
        orders_to_test = [
            (1, 1, 1), (1, 1, 0), (0, 1, 1), 
            (2, 1, 1), (1, 1, 2), (1, 0, 1), 
            (2, 0, 2), (0, 1, 2), (2, 1, 0)
        ]
        
        print("📊 Testing ARIMA orders with time series cross validation...")
        
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
            print(f"✅ Best ARIMA order: {best_order} (CV RMSE: {best_cv_score:.4f})")
        
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
        """Fit Prophet model to the time series with time series cross validation"""
        print(f"\n🔄 Prophet Model with Time Series Cross Validation")
        print(f"Data points: {len(series)}")
        
        # Convert series to DataFrame with proper datetime index
        df = pd.DataFrame({'ds': pd.to_datetime(series.index), 'y': series.values})
        df = df.drop_duplicates(subset='ds').sort_values('ds')
        
        # Time Series Cross Validation for Prophet
        cv_scores = []
        n_splits = min(5, len(df) // 4)
        
        if n_splits >= 3:
            print("📊 Performing Prophet cross validation...")
            
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
                    # Fit Prophet model on training data
                    model = Prophet(
                        yearly_seasonality=True,
                        daily_seasonality=False,
                        weekly_seasonality=False
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
                print(f"✅ Prophet CV: {mean_cv_score:.4f} ± {std_cv_score:.4f} RMSE ({len(cv_scores)} folds)")
            else:
                print("⚠️  Prophet cross validation failed")
        else:
            print(f"⚠️  Not enough data for Prophet cross validation ({len(df)} points)")
        
        # Final model training with train/test split
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        print(f"📈 Final training: {len(train_df)} train, {len(test_df)} test points")
        
        # Fit model on training data
        model = Prophet(
            yearly_seasonality=True,
            daily_seasonality=False,
            weekly_seasonality=False
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
        
        print(f"✅ Test RMSE: {test_rmse:.4f}")
        
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

    def fit_random_forest_model(self, series, country):
        """Fit Enhanced Random Forest model with external factors integration"""
        try:
            print(f"\nFitting Enhanced Random Forest model for {country}")
            
            # Use the enhanced Random Forest model
            results = self.rf_model.fit(series, country)
            
            # Calculate periods needed to reach 2030 (same as other models)
            last_year = pd.to_datetime(series.index[-1]).year
            target_year = 2030
            periods_to_2030 = max(5, min(10, target_year - last_year))  # Limit to max 10 years
            
            print(f"📈 Random Forest forecasting {periods_to_2030} periods to reach {target_year} (last data: {last_year})")
            
            # Generate future predictions with intervals
            future_results = self.rf_model.predict_future(series, country, periods=periods_to_2030)
            
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
        
        # Check if we have features for all years
        if len(all_features_by_year) < 8:  # Use same absolute threshold as SDG2
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
        
        if min_features < 3:  # Reduced from 2 to ensure SARIMAX quality
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
        
        if len(external_data_matrix) < 6:  # Reduced from 8 to be more flexible
            print(f"⚠️  After filtering, insufficient external data points ({len(external_data_matrix)} < 6). Falling back to ARIMA.")
            return self.fit_arima_model(series)
        
        # Align series with available external data (like SDG2)
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
                        
                        # Validate shapes before fitting - this should now be correct
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
            
            # Try to find data for this country in nearby years (±5 years instead of ±3)
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

    def save_plot(self):
        """Save the current plot as an image file"""
        if self.current_fig:
            # Get user's desktop path
            desktop = os.path.expanduser("~/Desktop")
            
            # Get current selections for default filename
            indicator_id = self.indicator_var.get().split(' - ')[0]
            country = self.country_var.get()
            series_code = self.series_code_var.get()
            gender = self.gender_var.get()
            age = self.age_var.get()
            
            # Create default filename
            default_filename = f"SDG3_{indicator_id}_{country}_{series_code}_{gender}_{age}.png"
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
            country = self.country_var.get()
            series_code = self.series_code_var.get()
            gender = self.gender_var.get()
            age = self.age_var.get()
            model_type = self.model_var.get()
            
            if not country:
                messagebox.showerror("Error", "Please select a country")
                return
            
            if not series_code:
                messagebox.showerror("Error", "Please select a series code")
                return
            
            # Get data using hierarchical search
            indicator_data, data_source = self.get_hierarchical_data(indicator_id, country, series_code, gender, age)
            
            if len(indicator_data) == 0:
                messagebox.showerror("Error", f"No data found for {indicator_id} in {country} with series code {series_code}")
                return
            
            # Check if we have enough data points (with more flexible threshold)
            MIN_DATA_POINTS = 15  # Reduced from 20 to be more flexible with hierarchical data
            if len(indicator_data) < MIN_DATA_POINTS:
                # More informative error message with suggestions
                available_regions = self.df[
                    (self.df['Indicator'] == indicator_id) & 
                    (self.df['SeriesCode'] == series_code)
                ]['GeoAreaName'].value_counts()
                
                suggestions = []
                for region, count in available_regions.head(5).items():
                    if count >= MIN_DATA_POINTS:
                        suggestions.append(f"{region} ({count} points)")
                
                suggestion_text = ""
                if suggestions:
                    suggestion_text = f"\n\nRegions with sufficient data:\n• " + "\n• ".join(suggestions)
                
                messagebox.showerror("Error", 
                    f"Not enough data points for {gender}, age {age} in Series {series_code}.\n"
                    f"Found {len(indicator_data)} points, but need at least {MIN_DATA_POINTS} points "
                    f"for a reliable forecast.\n"
                    f"Data source: {data_source}"
                    f"{suggestion_text}")
                return
            
            # Convert TimePeriod to datetime and sort
            indicator_data['TimePeriod'] = pd.to_datetime(indicator_data['TimePeriod'], format='%Y')
            indicator_data = indicator_data.sort_values('TimePeriod')
            
            # Create plot
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            self.current_fig = fig
            
            # Set smaller font sizes
            plt.rcParams.update({'font.size': 8})
            
            # Ensure Value column is numeric
            indicator_data['Value'] = pd.to_numeric(indicator_data['Value'], errors='coerce')
            indicator_data = indicator_data.dropna(subset=['Value'])
            
            if len(indicator_data) == 0:
                messagebox.showerror("Error", "No valid numeric data found for the selected series")
                return
            
            print(f"Valid data points after cleaning: {len(indicator_data)}")
            print(f"Data range: {indicator_data['Value'].min()} to {indicator_data['Value'].max()}")
            
            # Determine the unit and scale factor
            series_description = indicator_data['SeriesDescription'].iloc[0].lower()
            unit = ""
            scale_factor = 1.0
            
            if any(word in series_description for word in ['percentage', 'percent', '%']):
                unit = "%"
            elif any(word in series_description for word in ['thousand', '1000']):
                unit = "thousands"
            elif any(word in series_description for word in ['million']):
                unit = "millions"
            else:
                max_value = indicator_data['Value'].max()
                if max_value > 1000000:
                    unit = "millions"
                    scale_factor = 1000000.0
                elif max_value > 1000:
                    unit = "thousands"
                    scale_factor = 1000.0
            
            # Scale the data
            scaled_data = indicator_data.copy()
            scaled_data['Value'] = scaled_data['Value'] / scale_factor
            
            # Plot historical data
            ax.scatter(scaled_data['TimePeriod'], scaled_data['Value'], 
                      color='blue', label=f'Series {series_code} (Historical)', s=100, alpha=0.8)
            
            # Prepare data for modeling
            series = self.prepare_time_series(scaled_data)
            
            print(f"Time series for {series_code}: {len(series)} entries")
            print(f"First 5 values: {series.head()}")
            print(f"NaN values: {series.isna().sum()}")
            
            try:
                if model_type == 'ARIMA':
                    # Fit ARIMA model
                    arima_results = self.fit_arima_model(series)
                    model_fit = arima_results['model']
                    test_predictions = arima_results['test_predictions']
                    test = arima_results['test_data']
                    rmse = arima_results['rmse']
                    
                    # Scale the predictions and test data
                    scaled_test_predictions = test_predictions / scale_factor
                    scaled_test = test / scale_factor
                    
                    # Plot test predictions
                    prediction_color = plt.cm.Reds(0.7)
                    ax.scatter(test.index, scaled_test_predictions, color=prediction_color, 
                              label=f'Series {series_code} (Model Test)', s=100, alpha=0.8)
                    ax.plot(test.index, scaled_test_predictions, color=prediction_color, alpha=0.5, linewidth=2)
                    
                    # Calculate periods needed to reach 2030
                    last_year = pd.to_datetime(series.index[-1]).year
                    target_year = 2030
                    periods_to_2030 = max(5, min(10, target_year - last_year))  # Limit to max 10 years
                    
                    # Make future forecast
                    future_forecast = model_fit.get_forecast(steps=periods_to_2030)
                    scaled_forecast = future_forecast.predicted_mean / scale_factor
                    future_dates = pd.date_range(start=series.index[-1], periods=periods_to_2030+1, freq='Y')[1:]
                    
                    # Calculate enhanced confidence and prediction intervals based on RMSE
                    pred_interval_95 = 1.96 * rmse / scale_factor
                    conf_interval_95 = 1.5 * rmse / scale_factor
                    conf_interval_68 = 1.0 * rmse / scale_factor
                    
                    # Create interval bounds
                    scaled_pred_lower_95 = scaled_forecast - pred_interval_95
                    scaled_pred_upper_95 = scaled_forecast + pred_interval_95
                    scaled_conf_lower_95 = scaled_forecast - conf_interval_95
                    scaled_conf_upper_95 = scaled_forecast + conf_interval_95
                    scaled_conf_lower_68 = scaled_forecast - conf_interval_68
                    scaled_conf_upper_68 = scaled_forecast + conf_interval_68
                
                elif model_type == 'Prophet':
                    # Fit Prophet model
                    prophet_results = self.fit_prophet_model(series)
                    model_fit = prophet_results['model']
                    test_predictions = prophet_results['test_predictions']
                    test = prophet_results['test_data']
                    rmse = prophet_results['rmse']
                    
                    # Scale the predictions and test data
                    scaled_test_predictions = test_predictions / scale_factor
                    scaled_test = test / scale_factor
                    
                    # Plot test predictions
                    prediction_color = plt.cm.Reds(0.7)
                    ax.scatter(test_predictions.index, scaled_test_predictions, color=prediction_color, 
                              label=f'Series {series_code} (Model Test)', s=100, alpha=0.8)
                    ax.plot(test_predictions.index, scaled_test_predictions, color=prediction_color, alpha=0.5, linewidth=2)
                    
                    # Calculate periods needed to reach 2030
                    last_year = pd.to_datetime(series.index[-1]).year
                    target_year = 2030
                    periods_to_2030 = max(5, min(10, target_year - last_year))  # Limit to max 10 years
                    
                    # Make future forecast
                    future = model_fit.make_future_dataframe(periods=periods_to_2030, freq='Y')
                    # Ensure we only get future dates
                    last_date = series.index[-1]
                    future = future[future['ds'] > last_date]
                    forecast = model_fit.predict(future)
                    
                    scaled_forecast = forecast['yhat'].values / scale_factor
                    future_dates = pd.to_datetime(forecast['ds'])
                    
                    # Calculate enhanced confidence and prediction intervals based on RMSE
                    pred_interval_95 = 1.96 * rmse / scale_factor
                    conf_interval_95 = 1.5 * rmse / scale_factor
                    conf_interval_68 = 1.0 * rmse / scale_factor
                    
                    # Create interval bounds
                    scaled_pred_lower_95 = scaled_forecast - pred_interval_95
                    scaled_pred_upper_95 = scaled_forecast + pred_interval_95
                    scaled_conf_lower_95 = scaled_forecast - conf_interval_95
                    scaled_conf_upper_95 = scaled_forecast + conf_interval_95
                    scaled_conf_lower_68 = scaled_forecast - conf_interval_68
                    scaled_conf_upper_68 = scaled_forecast + conf_interval_68
                
                elif model_type == 'SARIMAX':
                    # Fit SARIMAX model
                    sarimax_results = self.fit_sarimax_model(series, country)
                    
                    # Check if SARIMAX actually worked or fell back to ARIMA
                    if 'feature_names' in sarimax_results:
                        print("✅ True SARIMAX with external variables")
                        model_fit = sarimax_results['model']
                        test_predictions = sarimax_results['test_predictions']
                        test = sarimax_results['test_data']
                        rmse = sarimax_results['rmse']
                        
                        # Scale the predictions and test data
                        scaled_test_predictions = test_predictions / scale_factor
                        scaled_test = test / scale_factor
                        
                        # Plot test predictions
                        prediction_color = plt.cm.Reds(0.7)
                        ax.scatter(test.index, scaled_test_predictions, color=prediction_color, 
                                  label=f'Series {series_code} (SARIMAX Test)', s=100, alpha=0.8)
                        ax.plot(test.index, scaled_test_predictions, color=prediction_color, alpha=0.5, linewidth=2)
                        
                        # Calculate periods needed to reach 2030
                        last_year = pd.to_datetime(series.index[-1]).year
                        target_year = 2030
                        periods_to_2030 = max(5, min(10, target_year - last_year))  # Limit to max 10 years
                        
                        # For SARIMAX, we need to generate future external variables
                        # Use intelligent extrapolation instead of just repeating last values
                        future_exog = self.extrapolate_external_variables(
                            country, last_year, periods_to_2030, sarimax_results['feature_names']
                        )
                        
                        # Scale the future external variables using the same scaler
                        future_exog_scaled = sarimax_results['scaler'].transform(future_exog)
                        
                        print(f"📊 Future external variables generated: {future_exog_scaled.shape}")
                        print(f"Sample future features (year {last_year + 1}): {[f'{f:.2f}' for f in future_exog_scaled[0]]}")
                        
                        future_forecast = model_fit.forecast(steps=periods_to_2030, exog=future_exog_scaled)
                        scaled_forecast = future_forecast / scale_factor
                        future_dates = pd.date_range(start=series.index[-1], periods=periods_to_2030+1, freq='Y')[1:]
                        
                        # Store for results display
                        self.sarimax_features_used = sarimax_results['feature_names']
                        self.sarimax_order = sarimax_results['best_order']
                        self.sarimax_seasonal_order = sarimax_results['best_seasonal_order']
                    else:
                        print("⚠️  SARIMAX fell back to ARIMA")
                        # Fell back to ARIMA, use ARIMA results
                        model_fit = sarimax_results['model']
                        test_predictions = sarimax_results['test_predictions']
                        test = sarimax_results['test_data']
                        rmse = sarimax_results['rmse']
                        
                        # Scale the predictions and test data
                        scaled_test_predictions = test_predictions / scale_factor
                        scaled_test = test / scale_factor
                        
                        # Plot test predictions
                        prediction_color = plt.cm.Reds(0.7)
                        ax.scatter(test.index, scaled_test_predictions, color=prediction_color, 
                                  label=f'Series {series_code} (ARIMA Fallback)', s=100, alpha=0.8)
                        ax.plot(test.index, scaled_test_predictions, color=prediction_color, alpha=0.5, linewidth=2)
                        
                        # Calculate periods needed to reach 2030
                        last_year = pd.to_datetime(series.index[-1]).year
                        target_year = 2030
                        periods_to_2030 = max(5, min(10, target_year - last_year))  # Limit to max 10 years
                        
                        # Make future forecast using ARIMA approach
                        future_forecast = model_fit.get_forecast(steps=periods_to_2030)
                        scaled_forecast = future_forecast.predicted_mean / scale_factor
                        future_dates = pd.date_range(start=series.index[-1], periods=periods_to_2030+1, freq='Y')[1:]
                    
                    # Calculate enhanced confidence and prediction intervals based on RMSE
                    pred_interval_95 = 1.96 * rmse / scale_factor
                    conf_interval_95 = 1.5 * rmse / scale_factor
                    conf_interval_68 = 1.0 * rmse / scale_factor
                    
                    # Create interval bounds
                    scaled_pred_lower_95 = scaled_forecast - pred_interval_95
                    scaled_pred_upper_95 = scaled_forecast + pred_interval_95
                    scaled_conf_lower_95 = scaled_forecast - conf_interval_95
                    scaled_conf_upper_95 = scaled_forecast + conf_interval_95
                    scaled_conf_lower_68 = scaled_forecast - conf_interval_68
                    scaled_conf_upper_68 = scaled_forecast + conf_interval_68
                
                # Plot future forecast
                forecast_color = plt.cm.Greens(0.7)
                
                if model_type == 'Random Forest':
                    # Fit Random Forest model
                    rf_results = self.fit_random_forest_model(series, country)
                    
                    # Scale the predictions
                    scaled_test_predictions = rf_results['test_predictions'] / scale_factor
                    scaled_forecast = rf_results['future_predictions'] / scale_factor
                    
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
                              label=f'Series {series_code} (Model Test)', s=100, alpha=0.8)
                    ax.plot(rf_results['test_predictions'].index, scaled_test_predictions, color=prediction_color, alpha=0.5, linewidth=2)
                    
                    # Get future dates
                    future_dates = rf_results['future_predictions'].index
                    
                    # Store for results display
                    self.rf_features_used = self.rf_model.feature_names
                    self.rf_feature_importance = rf_results['feature_importance']
                    rmse = rf_results['rmse']
                
                # Plot future forecast with unified interval visualization
                forecast_color = plt.cm.Greens(0.7)
                
                # Apply unified interval visualization for all models
                if model_type == 'Random Forest':
                    # Plot prediction intervals first (widest, darkest shade)
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
                              label=f'Series {series_code} (Future Forecast)', s=100, alpha=1.0, zorder=4)
                    ax.plot(future_dates, scaled_forecast, color=forecast_color, alpha=0.8, linewidth=3, zorder=4)
                else:
                    # For ALL other models (ARIMA, Prophet, SARIMAX) - same beautiful intervals!
                    # Plot prediction intervals first (widest, darkest shade)
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
                              label=f'Series {series_code} (Future Forecast)', s=100, alpha=1.0, zorder=4)
                    ax.plot(future_dates, scaled_forecast, color=forecast_color, alpha=0.8, linewidth=3, zorder=4)
                
                # Add annotation for last data point
                last_date = series.index[-1]
                last_value = series.iloc[-1] / scale_factor
                ax.annotate(f'Latest data: {last_value:.2f} {unit}',
                           xy=(last_date, last_value),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, color='blue',
                           bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
                
                # Set y-axis limits
                all_values = list(series/scale_factor) + list(scaled_test_predictions) + list(scaled_forecast)
                y_min = min(all_values)
                y_max = max(all_values)
                y_range = y_max - y_min
                ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not generate forecast for Series {series_code}: {str(e)}")
                return
            
            # Customize plot
            source = indicator_data['Source'].iloc[0]
            series_description = indicator_data['SeriesDescription'].iloc[0]
            title = f'Forecast for {indicator_id}\n({series_description})\nin {country}\nSeries Code: {series_code}'
            title += f'\nGender: {gender}, Age: {age}'
            title += f'\nSource: {source}'
            
            # Add data source information if using hierarchical aggregation
            if "Aggregated from" in data_source or "Similar region" in data_source:
                title += f'\nData: {data_source}'
            
            title += f'\nModel: {model_type}'
            
            ax.set_title(title, fontsize=9, pad=10)
            ax.set_xlabel('Year', fontsize=8)
            ax.set_ylabel(f'Value ({unit})', fontsize=8)
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
            plt.xticks(rotation=45)
            plt.subplots_adjust(right=0.85, top=0.85, bottom=0.15)
            
            # Embed plot in GUI
            self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # Enable save button after plot is generated
            self.save_button.state(['!disabled'])
            
            # Update results text
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"=== SDG Goal 3 Forecast Results ===\n\n")
            self.results_text.insert(tk.END, f"Indicator: {indicator_id} ({series_description})\n")
            self.results_text.insert(tk.END, f"Country: {country}\n")
            
            # Add data source information
            if "Aggregated from" in data_source:
                self.results_text.insert(tk.END, f"Data Source: {data_source}\n")
                self.results_text.insert(tk.END, f"Note: Data was aggregated from sub-regions due to insufficient direct data\n")
            elif "Similar region" in data_source:
                self.results_text.insert(tk.END, f"Data Source: {data_source}\n")
                self.results_text.insert(tk.END, f"Note: Using similar region data due to insufficient direct data\n")
            else:
                self.results_text.insert(tk.END, f"Data Source: Direct data for {country}\n")
            
            self.results_text.insert(tk.END, f"Series Code: {series_code}\n")
            self.results_text.insert(tk.END, f"Gender: {gender}\n")
            self.results_text.insert(tk.END, f"Age: {age}\n")
            self.results_text.insert(tk.END, f"Source: {source}\n")
            self.results_text.insert(tk.END, f"Unit: {unit}\n")
            self.results_text.insert(tk.END, f"Model: {model_type}\n\n")
            
            # Store model results for validation
            model_results = {}
            if model_type == 'ARIMA' and 'arima_results' in locals():
                model_results = arima_results
            elif model_type == 'Prophet' and 'prophet_results' in locals():
                model_results = prophet_results
            elif model_type == 'SARIMAX' and 'sarimax_results' in locals():
                model_results = sarimax_results
            elif model_type == 'Random Forest' and 'rf_results' in locals():
                model_results = rf_results
            
            # Add future predictions to model results for validation
            if 'scaled_forecast' in locals() and 'future_dates' in locals():
                model_results['future_predictions'] = pd.Series(
                    scaled_forecast * scale_factor,  # Convert back to original scale
                    index=future_dates
                )
            
            # Run integrated validation system
            validation_text, validation_results = self.integrated_validation_system(
                model_results, model_type, country, indicator_id, scaled_data, scale_factor
            )
            self.results_text.insert(tk.END, validation_text)
            
            # Add cross validation results
            if model_type == 'ARIMA' and 'arima_results' in locals() and arima_results.get('cv_results'):
                self.results_text.insert(tk.END, "=== ARIMA Cross Validation Results ===\n")
                cv_results = arima_results['cv_results']
                for order, results in cv_results.items():
                    self.results_text.insert(tk.END, f"ARIMA{order}: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f} RMSE ({results['n_folds']} folds)\n")
                self.results_text.insert(tk.END, f"Best order: {arima_results['best_order']}\n\n")
            
            elif model_type == 'Prophet' and 'prophet_results' in locals() and prophet_results.get('cv_scores'):
                self.results_text.insert(tk.END, "=== Prophet Cross Validation Results ===\n")
                cv_scores = prophet_results['cv_scores']
                mean_cv = np.mean(cv_scores)
                std_cv = np.std(cv_scores)
                self.results_text.insert(tk.END, f"Prophet CV: {mean_cv:.4f} ± {std_cv:.4f} RMSE ({len(cv_scores)} folds)\n\n")
            
            elif model_type == 'SARIMAX' and 'sarimax_results' in locals():
                self.results_text.insert(tk.END, "=== SARIMAX Cross Validation Results ===\n")
                
                if 'feature_names' in sarimax_results:
                    # True SARIMAX results
                    cv_results = sarimax_results.get('cv_results', {})
                    if cv_results:
                        self.results_text.insert(tk.END, "SARIMAX Parameter Optimization:\n")
                        # Show top 3 best models
                        sorted_results = sorted(cv_results.items(), key=lambda x: x[1]['mean_rmse'])[:3]
                        for (order, seasonal_order), results in sorted_results:
                            self.results_text.insert(tk.END, f"  SARIMAX{order}x{seasonal_order}: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f} RMSE ({results['n_folds']} folds)\n")
                    
                    self.results_text.insert(tk.END, f"\nBest Model: SARIMAX{sarimax_results['best_order']}x{sarimax_results['best_seasonal_order']}\n")
                    
                    # External features information
                    self.results_text.insert(tk.END, f"\nExternal Variables Used:\n")
                    for i, feature in enumerate(sarimax_results['feature_names']):
                        self.results_text.insert(tk.END, f"  • {feature}\n")
                    
                    self.results_text.insert(tk.END, f"\nExternal Data Matrix Shape: {sarimax_results['exog_data'].shape}\n")
                    
                else:
                    # Fell back to ARIMA
                    self.results_text.insert(tk.END, "⚠️  SARIMAX fell back to ARIMA (insufficient external data)\n")
                    if 'cv_results' in sarimax_results:
                        cv_results = sarimax_results['cv_results']
                        for order, results in cv_results.items():
                            self.results_text.insert(tk.END, f"ARIMA{order}: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f} RMSE ({results['n_folds']} folds)\n")
                
                self.results_text.insert(tk.END, "\n")
            
            elif model_type == 'Random Forest' and 'rf_results' in locals() and rf_results.get('feature_importance'):
                self.results_text.insert(tk.END, "=== Random Forest Results ===\n")
                
                # Add feature importance for Random Forest
                if 'feature_importance' in rf_results:
                    self.results_text.insert(tk.END, "\nFeature Importance:\n")
                    for feature, importance in sorted(rf_results['feature_importance'].items(), 
                                                    key=lambda x: x[1], reverse=True):
                        self.results_text.insert(tk.END, f"  {feature}: {importance*100:.1f}%\n")
                self.results_text.insert(tk.END, "\n")
            
            # Add model performance
            self.results_text.insert(tk.END, f"=== Model Performance ===\n")
            if model_type == 'ARIMA' and 'arima_results' in locals():
                self.results_text.insert(tk.END, f"Test RMSE: {arima_results['rmse']/scale_factor:.4f} {unit}\n")
            elif model_type == 'Prophet' and 'prophet_results' in locals():
                self.results_text.insert(tk.END, f"Test RMSE: {prophet_results['rmse']/scale_factor:.4f} {unit}\n")
            elif model_type == 'SARIMAX' and 'sarimax_results' in locals():
                self.results_text.insert(tk.END, f"Test RMSE: {sarimax_results['rmse']/scale_factor:.4f} {unit}\n")
                if 'feature_names' in sarimax_results:
                    self.results_text.insert(tk.END, f"Model Type: SARIMAX with {len(sarimax_results['feature_names'])} external variables\n")
                else:
                    self.results_text.insert(tk.END, f"Model Type: ARIMA (SARIMAX fallback)\n")
            elif model_type == 'Random Forest' and 'rf_results' in locals():
                self.results_text.insert(tk.END, f"Test RMSE: {rf_results['rmse']/scale_factor:.4f} {unit}\n")
            
            self.results_text.insert(tk.END, f"\n=== Historical Data ===\n")
            self.results_text.insert(tk.END, f"Data points: {len(scaled_data)}\n")
            self.results_text.insert(tk.END, f"Years: {scaled_data['TimePeriod'].dt.year.min()} - {scaled_data['TimePeriod'].dt.year.max()}\n")
            
            # Show recent historical values
            recent_data = scaled_data.tail(5)
            self.results_text.insert(tk.END, "\nRecent Historical Values:\n")
            for _, row in recent_data.iterrows():
                self.results_text.insert(tk.END, f"  {row['TimePeriod'].year}: {row['Value']:.3f} {unit}\n")
            
            # Add forecast values
            self.results_text.insert(tk.END, f"\n=== Future Forecast ===\n")
            if 'scaled_forecast' in locals() and 'future_dates' in locals() and scaled_forecast is not None and future_dates is not None:
                try:
                    for i, (date, value) in enumerate(zip(future_dates, scaled_forecast)):
                        year = date.year if hasattr(date, 'year') else date
                        if not np.isnan(value):
                            self.results_text.insert(tk.END, f"  {year}: {value:.3f} {unit}\n")
                        else:
                            self.results_text.insert(tk.END, f"  {year}: N/A {unit} (NaN detected)\n")
            except Exception as e:
                    self.results_text.insert(tk.END, f"Error displaying forecast values: {str(e)}\n")
            else:
                self.results_text.insert(tk.END, "No forecast values available\n")
            
            # Add external features for SARIMAX
            if model_type == 'SARIMAX' and hasattr(self, 'sarimax_features_used'):
                if self.sarimax_features_used:
                    features_str = ', '.join(self.sarimax_features_used)
                    title += f'\nExternal Variables: {features_str}'
                    title += f'\nSARIMAX Order: {self.sarimax_order}x{self.sarimax_seasonal_order}'
            
            self.results_text.insert(tk.END, f"\n=== Model Validation Summary ===\n")
            self.results_text.insert(tk.END, f"✅ Time series cross validation performed\n")
            self.results_text.insert(tk.END, f"✅ Proper temporal train/test split used\n")
            self.results_text.insert(tk.END, f"✅ Out-of-sample testing completed\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def integrated_validation_system(self, model_results, model_name, country, indicator, historical_data, scale_factor):
        """Integrated validation system that runs automatically after each forecast"""
        try:
            import scipy.stats as stats
            
            validation_results = {}
            validation_text = f"\n🔍 AUTOMATIC VALIDATION for {model_name} (SDG3)\n" + "="*60 + "\n"
            
            # 1. Statistical Validation
            if 'test_predictions' in model_results and 'test_data' in model_results:
                predictions = model_results['test_predictions']
                true_values = model_results['test_data']
                
                # Ensure both are numeric arrays
                if hasattr(predictions, 'values'):
                    predictions = predictions.values
                if hasattr(true_values, 'values'):
                    true_values = true_values.values
                
                # Calculate metrics
                mse = np.mean((predictions - true_values) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(predictions - true_values))
                
                # Calculate MAPE safely
                non_zero_mask = true_values != 0
                if np.any(non_zero_mask):
                    mape = np.mean(np.abs((true_values[non_zero_mask] - predictions[non_zero_mask]) / true_values[non_zero_mask])) * 100
                else:
                    mape = np.inf
                
                # Statistical tests
                residuals = predictions - true_values
                if len(residuals) >= 3:
                    try:
                        shapiro_stat, shapiro_p = stats.shapiro(residuals)
                        dw_stat = self._durbin_watson_stat(residuals)
                    except:
                        shapiro_p, dw_stat = np.nan, np.nan
                else:
                    shapiro_p, dw_stat = np.nan, np.nan
                
                # Score statistical performance (0-40 points)
                stat_score = 0
                rmse_scaled = rmse / scale_factor
                if rmse_scaled < 1.0:
                    stat_score = 40
                elif rmse_scaled < 2.0:
                    stat_score = 30
                elif rmse_scaled < 3.0:
                    stat_score = 20
                elif rmse_scaled < 5.0:
                    stat_score = 10
                else:
                    stat_score = 5
                
                validation_results['statistical'] = {
                    'rmse': rmse_scaled, 'mae': mae/scale_factor, 'mape': mape,
                    'shapiro_p': shapiro_p, 'dw_stat': dw_stat,
                    'score': stat_score
                }
                
                validation_text += f"📊 Statistical Performance:\n"
                validation_text += f"   RMSE: {rmse_scaled:.4f} | MAE: {mae/scale_factor:.4f} | MAPE: {mape:.2f}%\n"
                if not np.isnan(shapiro_p):
                    validation_text += f"   Residuals Normal: {'✅' if shapiro_p > 0.05 else '⚠️'} (p={shapiro_p:.3f})\n"
                if not np.isnan(dw_stat):
                    validation_text += f"   Autocorrelation: {'✅' if 1.5 <= dw_stat <= 2.5 else '⚠️'} (DW={dw_stat:.3f})\n"
                validation_text += f"   Score: {stat_score}/40 {'✅' if stat_score >= 30 else '⚠️'}\n\n"
            
            # 2. Data Quality Assessment
            if hasattr(historical_data, 'index'):
                # For DataFrame with datetime index
                if hasattr(historical_data.index, 'year'):
                    years_span = historical_data.index.year.max() - historical_data.index.year.min()
                else:
                    years_span = (historical_data.index.max() - historical_data.index.min()).days / 365.25
            else:
                years_span = len(historical_data) / 2  # Rough estimate
            
            # Calculate missing percentage
            if hasattr(historical_data, 'isnull'):
                missing_pct = historical_data['Value'].isnull().sum() / len(historical_data) * 100
            else:
                missing_pct = 0
            
            # Check external data availability
            external_status = self._check_external_data_availability(country)
            external_available = sum(external_status.values())
            
            # Data quality score (0-30 points)
            quality_score = self._calculate_data_quality_score(years_span, len(historical_data), missing_pct, external_status)
            
            validation_results['data_quality'] = {
                'years_span': years_span,
                'missing_pct': missing_pct,
                'external_available': external_available,
                'score': quality_score
            }
            
            validation_text += f"📋 Data Quality Assessment:\n"
            validation_text += f"   Time Span: {years_span:.1f} years | Data Points: {len(historical_data)}\n"
            validation_text += f"   Missing: {missing_pct:.1f}% | External Variables: {external_available}/5\n"
            validation_text += f"   Score: {quality_score:.0f}/30 {'✅' if quality_score >= 20 else '⚠️'}\n\n"
            
            # 3. Health-Specific Economic Realism Check
            if 'future_predictions' in model_results:
                forecasts = model_results['future_predictions']
                
                # Growth rate analysis
                growth_rates = []
                forecast_values = forecasts.values if hasattr(forecasts, 'values') else forecasts
                
                # Handle the case where forecast_values might be a scalar
                if np.isscalar(forecast_values):
                    forecast_values = [forecast_values]
                
                for i in range(1, len(forecast_values)):
                    if forecast_values[i-1] != 0:
                        growth_rate = (forecast_values[i] - forecast_values[i-1]) / forecast_values[i-1]
                        growth_rates.append(growth_rate)
                
                if growth_rates:
                    avg_growth = np.mean(growth_rates) * 100
                    growth_volatility = np.std(growth_rates) * 100
                    
                    # Health-specific realism scoring
                    realism_score = self._score_health_realism(avg_growth, growth_volatility, country, indicator)
                    
                    validation_results['realism'] = {
                        'avg_growth': avg_growth,
                        'growth_volatility': growth_volatility,
                        'score': realism_score
                    }
                    
                    validation_text += f"🏥 Health Indicator Realism:\n"
                    validation_text += f"   Avg Growth: {avg_growth:+.2f}%/year | Volatility: {growth_volatility:.2f}%\n"
                    validation_text += f"   Health Context: {self._get_health_context(indicator)}\n"
                    validation_text += f"   Score: {realism_score}/30 {'✅' if realism_score >= 20 else '⚠️'}\n\n"
            
            # 4. Overall Validation Score
            total_score = 0
            max_score = 0
            
            if 'statistical' in validation_results:
                total_score += validation_results['statistical']['score']
                max_score += 40
            if 'data_quality' in validation_results:
                total_score += validation_results['data_quality']['score']
                max_score += 30
            if 'realism' in validation_results:
                total_score += validation_results['realism']['score']
                max_score += 30
            
            final_score = (total_score / max_score * 100) if max_score > 0 else 0
            
            validation_text += f"🏆 OVERALL VALIDATION SCORE: {final_score:.1f}/100\n"
            validation_text += self._get_health_validation_recommendation(final_score) + "\n"
            validation_text += "="*60 + "\n"
            
            return validation_text, validation_results
            
        except Exception as e:
            return f"\n⚠️ Validation Error: {str(e)}\n", {}
    
    def _durbin_watson_stat(self, residuals):
        """Calculate Durbin-Watson statistic for autocorrelation"""
        if len(residuals) < 2:
            return np.nan
        diff = np.diff(residuals)
        return np.sum(diff**2) / np.sum(residuals**2)
    
    def _calculate_data_quality_score(self, years_span, n_points, missing_pct, external_status):
        """Calculate data quality score (0-30 points)"""
        score = 0
        
        # Temporal coverage (10 points max)
        if years_span >= 20:
            score += 10
        elif years_span >= 15:
            score += 8
        elif years_span >= 10:
            score += 6
        elif years_span >= 5:
            score += 3
        
        # Data completeness (10 points max)
        if missing_pct == 0:
            score += 10
        elif missing_pct < 5:
            score += 8
        elif missing_pct < 10:
            score += 6
        elif missing_pct < 20:
            score += 3
        
        # External data availability (10 points max)
        available_external = sum(external_status.values())
        total_external = len(external_status)
        if total_external > 0:
            external_score = (available_external / total_external) * 10
            score += external_score
        
        return score
    
    def _score_health_realism(self, avg_growth, growth_volatility, country, indicator):
        """Score health indicator realism (0-30 points)"""
        score = 30
        
        # Health-specific growth rate plausibility
        if "3.1" in indicator:  # Maternal mortality
            if abs(avg_growth) > 8:  # Maternal mortality can change rapidly with interventions
                score -= 15
            elif abs(avg_growth) > 4:
                score -= 8
        elif "3.2" in indicator:  # Child mortality
            if abs(avg_growth) > 10:  # Child mortality can improve quickly
                score -= 15
            elif abs(avg_growth) > 5:
                score -= 8
        elif "3.3" in indicator:  # Infectious diseases
            if abs(avg_growth) > 15:  # Can change rapidly during epidemics
                score -= 10
            elif abs(avg_growth) > 8:
                score -= 5
        elif "3.4" in indicator:  # Non-communicable diseases
            if abs(avg_growth) > 5:  # Usually gradual changes
                score -= 15
            elif abs(avg_growth) > 3:
                score -= 8
        elif "3.6" in indicator:  # Road traffic deaths
            if abs(avg_growth) > 8:  # Can improve with safety measures
                score -= 12
            elif abs(avg_growth) > 4:
                score -= 6
        else:  # Other health indicators
            if abs(avg_growth) > 10:
                score -= 15
            elif abs(avg_growth) > 5:
                score -= 8
        
        # Volatility check
        if growth_volatility > 25:
            score -= 10
        elif growth_volatility > 15:
            score -= 5
        
        # Country-specific adjustments
        country_lower = country.lower()
        developed_countries = ['germany', 'united states', 'france', 'japan', 'australia', 'canada', 'united kingdom']
        if any(dc in country_lower for dc in developed_countries):
            if abs(avg_growth) > 3:  # Lower tolerance for developed countries
                score -= 5
        
        return max(0, score)
    
    def _get_health_context(self, indicator):
        """Get health-specific context for the indicator"""
        if "3.1" in indicator:
            return "Maternal health - interventions can show rapid improvements"
        elif "3.2" in indicator:
            return "Child health - vaccines and nutrition programs effective"
        elif "3.3" in indicator:
            return "Infectious diseases - epidemic patterns, treatment advances"
        elif "3.4" in indicator:
            return "Non-communicable diseases - lifestyle and aging trends"
        elif "3.5" in indicator:
            return "Substance abuse - policy and social interventions"
        elif "3.6" in indicator:
            return "Road safety - infrastructure and enforcement policies"
        elif "3.7" in indicator:
            return "Reproductive health - education and access programs"
        elif "3.8" in indicator:
            return "Universal health coverage - system strengthening"
        elif "3.9" in indicator:
            return "Environmental health - pollution control and climate"
        else:
            return "General health indicator - multiple determinants"
    
    def _get_health_validation_recommendation(self, score):
        """Get health-specific recommendation based on validation score"""
        if score >= 80:
            return "✅ EXCELLENT: Highly reliable for health policy decisions and planning"
        elif score >= 65:
            return "✅ GOOD: Suitable for health system planning with normal uncertainty"
        elif score >= 50:
            return "⚠️ MODERATE: Use cautiously for health planning, consider confidence intervals"
        elif score >= 35:
            return "⚠️ POOR: Significant limitations, only for rough health trend estimates"
        else:
            return "❌ UNRELIABLE: Data quality too poor for reliable health forecasting"

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

    def get_hierarchical_data(self, indicator_id, country, series_code, gender, age):
        """
        Intelligent hierarchical data aggregation system
        Automatically combines data from sub-regions when main region lacks sufficient data
        """
        print(f"\n🔍 Hierarchical Data Search for {country}")
        
        # Define hierarchical relationships
        hierarchical_mapping = {
            'Europe': ['Eastern Europe', 'Western Europe', 'Northern Europe', 'Southern Europe'],
            'Africa': ['Eastern Africa', 'Western Africa', 'Northern Africa', 'Southern Africa', 'Middle Africa'],
            'Asia': ['Eastern Asia', 'Western Asia', 'Southern Asia', 'South-eastern Asia', 'Central Asia'],
            'Americas': ['Northern America', 'South America', 'Central America', 'Caribbean'],
            'Oceania': ['Australia and New Zealand', 'Melanesia', 'Micronesia', 'Polynesia'],
            'World': ['Europe', 'Africa', 'Asia', 'Americas', 'Oceania'],
            
            # Additional common patterns
            'Sub-Saharan Africa': ['Eastern Africa', 'Western Africa', 'Southern Africa', 'Middle Africa'],
            'Latin America and the Caribbean': ['South America', 'Central America', 'Caribbean'],
            'Least Developed Countries': [],  # Special case - keep as is
            'Landlocked Developing Countries': [],  # Special case - keep as is
            'Small Island Developing States': [],  # Special case - keep as is
        }
        
        # Try to get direct data first
        direct_data = self.df[
            (self.df['Indicator'] == indicator_id) & 
            (self.df['GeoAreaName'] == country) &
            (self.df['SeriesCode'] == series_code)
        ]
        
        # Apply filters
        if len(direct_data) > 0:
            direct_data['Sex'] = direct_data['Sex'].fillna('BOTHSEX')
            direct_data['Age'] = direct_data['Age'].fillna('ALLAGE')
            
            if gender != 'BOTHSEX':
                direct_data = direct_data[direct_data['Sex'] == gender]
            if age != 'ALLAGE':
                direct_data = direct_data[direct_data['Age'] == age]
        
        print(f"📊 Direct data for {country}: {len(direct_data)} points")
        
        # If we have enough direct data, return it
        if len(direct_data) >= 20:
            print(f"✅ Sufficient direct data ({len(direct_data)} ≥ 20)")
            return direct_data, f"Direct data for {country}"
        
        # Check if this country has sub-regions we can aggregate
        if country in hierarchical_mapping:
            sub_regions = hierarchical_mapping[country]
            print(f"🔍 Searching sub-regions for {country}: {sub_regions}")
            
            aggregated_data = []
            sources_used = []
            
            for sub_region in sub_regions:
                sub_data = self.df[
                    (self.df['Indicator'] == indicator_id) & 
                    (self.df['GeoAreaName'] == sub_region) &
                    (self.df['SeriesCode'] == series_code)
                ]
                
                # Apply same filters
                if len(sub_data) > 0:
                    sub_data['Sex'] = sub_data['Sex'].fillna('BOTHSEX')
                    sub_data['Age'] = sub_data['Age'].fillna('ALLAGE')
                    
                    if gender != 'BOTHSEX':
                        sub_data = sub_data[sub_data['Sex'] == gender]
                    if age != 'ALLAGE':
                        sub_data = sub_data[sub_data['Age'] == age]
                
                if len(sub_data) > 0:
                    print(f"  📊 {sub_region}: {len(sub_data)} points")
                    aggregated_data.append(sub_data)
                    sources_used.append(sub_region)
            
            # Combine all sub-region data
            if aggregated_data:
                combined_data = pd.concat(aggregated_data, ignore_index=True)
                
                # For overlapping years, take the mean
                if len(combined_data) > 0:
                    # Group by year and take mean of values
                    combined_data['TimePeriod'] = pd.to_numeric(combined_data['TimePeriod'], errors='coerce')
                    combined_data['Value'] = pd.to_numeric(combined_data['Value'], errors='coerce')
                    
                    # Group by year and calculate weighted average (by data availability)
                    yearly_data = combined_data.groupby('TimePeriod').agg({
                        'Value': 'mean',  # Take mean across sub-regions
                        'GeoAreaName': 'first',  # Keep first region name for reference
                        'SeriesCode': 'first',
                        'SeriesDescription': 'first',
                        'Indicator': 'first',
                        'Source': 'first',
                        'Sex': 'first',
                        'Age': 'first'
                    }).reset_index()
                    
                    # Update the GeoAreaName to reflect aggregation
                    yearly_data['GeoAreaName'] = country
                    
                    print(f"✅ Aggregated data: {len(yearly_data)} points from {len(sources_used)} sub-regions")
                    print(f"   Sources: {', '.join(sources_used)}")
                    
                    # Check if aggregated data is sufficient
                    if len(yearly_data) >= 20:
                        source_description = f"Aggregated from {len(sources_used)} sub-regions: {', '.join(sources_used[:3])}"
                        if len(sources_used) > 3:
                            source_description += f" and {len(sources_used)-3} others"
                        return yearly_data, source_description
                    else:
                        print(f"⚠️  Aggregated data still insufficient ({len(yearly_data)} < 20)")
        
        # Try fuzzy matching for similar region names
        print(f"🔍 Trying fuzzy matching for similar regions...")
        all_countries = self.df['GeoAreaName'].unique()
        
        # Look for countries that contain the search term or vice versa
        similar_regions = []
        country_lower = country.lower()
        
        for region in all_countries:
            region_lower = region.lower()
            
            # Skip exact matches (already tried)
            if region_lower == country_lower:
                continue
            
            # Look for partial matches
            if (country_lower in region_lower or region_lower in country_lower or
                any(word in region_lower for word in country_lower.split() if len(word) > 3) or
                any(word in country_lower for word in region_lower.split() if len(word) > 3)):
                
                similar_data = self.df[
                    (self.df['Indicator'] == indicator_id) & 
                    (self.df['GeoAreaName'] == region) &
                    (self.df['SeriesCode'] == series_code)
                ]
                
                # Apply filters
                if len(similar_data) > 0:
                    similar_data['Sex'] = similar_data['Sex'].fillna('BOTHSEX')
                    similar_data['Age'] = similar_data['Age'].fillna('ALLAGE')
                    
                    if gender != 'BOTHSEX':
                        similar_data = similar_data[similar_data['Sex'] == gender]
                    if age != 'ALLAGE':
                        similar_data = similar_data[similar_data['Age'] == age]
                
                if len(similar_data) > 5:  # At least some data
                    similar_regions.append((region, len(similar_data), similar_data))
                    print(f"  📊 Similar region '{region}': {len(similar_data)} points")
        
        # Sort by data availability and take the best match
        if similar_regions:
            similar_regions.sort(key=lambda x: x[1], reverse=True)
            best_match = similar_regions[0]
            
            print(f"✅ Best similar region: '{best_match[0]}' with {best_match[1]} points")
            
            # If best match has enough data, use it
            if best_match[1] >= 20:
                return best_match[2], f"Similar region: {best_match[0]} (closest match to {country})"
            
            # If best match is insufficient, try combining multiple similar regions
            elif len(similar_regions) > 1:
                print(f"🔄 Attempting to combine multiple similar regions...")
                
                combined_similar_data = []
                combined_sources = []
                total_points = 0
                
                # Take up to 5 best similar regions
                for region_name, point_count, region_data in similar_regions[:5]:
                    if point_count >= 10:  # Only regions with reasonable data
                        combined_similar_data.append(region_data)
                        combined_sources.append(f"{region_name} ({point_count}pts)")
                        total_points += point_count
                        print(f"  ➕ Adding {region_name}: {point_count} points")
                        
                        # Stop if we have enough combined data
                        if total_points >= 60:  # Generous threshold for combined data
                            break
                
                if len(combined_similar_data) >= 2 and total_points >= 30:
                    print(f"🎯 Combining {len(combined_similar_data)} similar regions with {total_points} total points")
                    
                    # Combine all similar region data
                    combined_data = pd.concat(combined_similar_data, ignore_index=True)
                    
                    # Group by year and take mean of values
                    combined_data['TimePeriod'] = pd.to_numeric(combined_data['TimePeriod'], errors='coerce')
                    combined_data['Value'] = pd.to_numeric(combined_data['Value'], errors='coerce')
                    
                    # Group by year and calculate weighted average
                    yearly_combined = combined_data.groupby('TimePeriod').agg({
                        'Value': 'mean',  # Take mean across similar regions
                        'GeoAreaName': 'first',
                        'SeriesCode': 'first',
                        'SeriesDescription': 'first',
                        'Indicator': 'first',
                        'Source': 'first',
                        'Sex': 'first',
                        'Age': 'first'
                    }).reset_index()
                    
                    # Update the GeoAreaName to reflect combination
                    yearly_combined['GeoAreaName'] = country
                    
                    print(f"✅ Combined similar regions: {len(yearly_combined)} unique years")
                    
                    if len(yearly_combined) >= 15:  # More flexible threshold for combined similar data
                        source_description = f"Combined {len(combined_similar_data)} similar regions: {', '.join(combined_sources[:3])}"
                        if len(combined_sources) > 3:
                            source_description += f" and {len(combined_sources)-3} others"
                        return yearly_combined, source_description
                    else:
                        print(f"⚠️  Combined similar data still insufficient ({len(yearly_combined)} < 15)")
                
            # Fallback to best single match if combination doesn't work
            if best_match[1] >= 12:  # More flexible threshold for single region
                print(f"⚠️  Using best single match with {best_match[1]} points")
                return best_match[2], f"Similar region: {best_match[0]} (partial match, {best_match[1]} points)"
        
        # If all else fails, return the direct data (even if insufficient)
        print(f"❌ No sufficient alternative data found. Returning direct data ({len(direct_data)} points)")
        return direct_data, f"Direct data for {country} (insufficient: {len(direct_data)} points)"

if __name__ == "__main__":
    app = ForecastAppGoal3()
    app.root.mainloop()
