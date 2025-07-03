import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Load SDG data
try:
    data = pd.read_csv('Goal1_processed.csv')
    print(f"SDG data loaded: {data.shape}")
except Exception as e:
    print(f"Error loading SDG data: {e}")
    exit(1)

# Load external data
external_data = {}
try:
    # Get the parent directory where processed CSV files are located
    sdg1_dir = os.path.dirname(os.getcwd())  # SDG1
    parent_dir = os.path.dirname(sdg1_dir)  # SDG
    
    processed_files = {
        'GDP': 'GDP_processed.csv',
        'GINI': 'GINI_processed.csv',
        'UNEMPLOYMENT': 'Unemployment_processed.csv',
        'RD': 'R&D Expenditures_processed.csv',
        'SOCIAL': 'social_coverage_processed.csv'
    }
    
    for data_name, filename in processed_files.items():
        file_path = os.path.join(parent_dir, filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"Loaded {data_name} data with shape: {df.shape}")
                
                # Process each dataset according to its format
                if data_name == 'GDP':
                    if 'Country Name' in df.columns and 'Year' in df.columns and 'GDP' in df.columns:
                        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                        df['GDP'] = pd.to_numeric(df['GDP'], errors='coerce')
                        df = df.dropna(subset=['Year', 'GDP'])
                        df = df[df['Year'] > 0]
                        external_data['GDP'] = df
                        print(f"GDP data processed successfully: {len(df)} records")
                    else:
                        print(f"GDP data has unexpected columns: {df.columns.tolist()}")
                
                # Add other datasets similarly (simplified for testing)
                elif data_name == 'UNEMPLOYMENT':
                    if 'Country Name' in df.columns and 'Year' in df.columns and 'Unemployment' in df.columns:
                        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                        df['Unemployment'] = pd.to_numeric(df['Unemployment'], errors='coerce')
                        df = df.dropna(subset=['Year', 'Unemployment'])
                        df = df[df['Year'] > 0]
                        external_data['UNEMPLOYMENT'] = df
                        print(f"Unemployment data processed successfully: {len(df)} records")
                        
            except Exception as e:
                print(f"Error loading {data_name}: {str(e)}")
        else:
            print(f"File not found: {file_path}")
            
except Exception as e:
    print(f"Error loading external data: {str(e)}")

print(f"Successfully loaded {len(external_data)} external datasets")

# Test with a specific indicator and country
indicator_id = '1.1.1'  # Common poverty indicator
country = 'Albania'  # Test country
series_code = 'SI_POV_DAY1'  # Common series code

print(f"\nTesting ARIMAX with: {indicator_id}, {country}, {series_code}")

# Get specific data
indicator_data = data[
    (data['Indicator'] == indicator_id) & 
    (data['GeoAreaName'] == country) &
    (data['SeriesCode'] == series_code)
]

print(f"Found {len(indicator_data)} data points for this combination")

if len(indicator_data) > 0:
    # Prepare time series
    indicator_data['TimePeriod'] = pd.to_datetime(indicator_data['TimePeriod'], format='%Y')
    indicator_data = indicator_data.sort_values('TimePeriod')
    indicator_data['Value'] = pd.to_numeric(indicator_data['Value'], errors='coerce')
    indicator_data = indicator_data.dropna(subset=['Value'])
    
    print(f"After cleaning: {len(indicator_data)} valid data points")
    print(f"Years range: {indicator_data['TimePeriod'].dt.year.min()} to {indicator_data['TimePeriod'].dt.year.max()}")
    
    if len(indicator_data) >= 10:
        # Create series
        series = indicator_data.set_index('TimePeriod')['Value']
        print(f"Series created with {len(series)} points")
        
        # Test ARIMAX data preparation
        print(f"\n=== Testing ARIMAX Data Preparation ===")
        
        def prepare_arimax_features_for_year(country, year):
            """Test version of the feature preparation"""
            features = []
            feature_names = []
            
            def get_country_year_value(data_df, country_name, year, value_column):
                print(f"    Looking for {value_column} data: country='{country_name}', year={year}")
                
                # Try exact match first
                country_data = data_df[
                    (data_df['Country Name'].str.strip().str.lower() == country_name.strip().lower()) &
                    (data_df['Year'] == year)
                ]
                
                if not country_data.empty:
                    value = float(country_data[value_column].iloc[0])
                    print(f"      ✓ Exact match found: {value}")
                    return value
                
                # Try contains match
                country_data = data_df[
                    data_df['Country Name'].str.contains(country_name, case=False, na=False) &
                    (data_df['Year'] == year)
                ]
                
                if not country_data.empty:
                    value = float(country_data[value_column].iloc[0])
                    print(f"      ✓ Contains match found: {value}")
                    return value
                
                # Try to find the most recent value for this country
                country_data = data_df[
                    data_df['Country Name'].str.contains(country_name, case=False, na=False)
                ]
                
                if not country_data.empty:
                    # Get the most recent year with data that's <= current year
                    recent_data = country_data[country_data['Year'] <= year].sort_values('Year')
                    if not recent_data.empty:
                        value = float(recent_data[value_column].iloc[-1])
                        recent_year = recent_data['Year'].iloc[-1]
                        print(f"      ✓ Recent match found: {value} (from year {recent_year})")
                        return value
                
                print(f"      ✗ No data found, returning 0")
                return 0.0
            
            # Add external variables (no Year feature for ARIMAX)
            if 'GDP' in external_data:
                gdp_value = get_country_year_value(external_data['GDP'], country, year, 'GDP')
                features.append(gdp_value)
                feature_names.append('GDP')
            
            if 'UNEMPLOYMENT' in external_data:
                unemployment_value = get_country_year_value(external_data['UNEMPLOYMENT'], country, year, 'Unemployment')
                features.append(unemployment_value)
                feature_names.append('Unemployment')
            
            return features, feature_names
        
        # Test feature extraction for each year
        exog_data = []
        valid_indices = []
        
        for date in series.index:
            year = date.year
            features, feature_names = prepare_arimax_features_for_year(country, year)
            
            print(f"ARIMAX DEBUG - Year {year}: Country '{country}'")
            print(f"  Features extracted: {features}")
            print(f"  Feature names: {feature_names}")
            print(f"  All features non-zero: {all(f != 0 for f in features) if features else False}")
            print(f"  Number of features: {len(features)}")
            
            if len(features) > 0 and all(f != 0 for f in features):
                exog_data.append(features)
                valid_indices.append(date)
                print(f"  ✓ Added to exog_data (total now: {len(exog_data)})")
            else:
                print(f"  ✗ Rejected - features empty or contains zeros")
        
        print(f"\nARIMAX SUMMARY:")
        print(f"Total valid external data points: {len(exog_data)}")
        print(f"Required minimum: 8")
        print(f"Will proceed with ARIMAX: {len(exog_data) >= 8}")
        
    else:
        print("Not enough cleaned data points for ARIMAX testing")
else:
    print("No data found for this combination") 