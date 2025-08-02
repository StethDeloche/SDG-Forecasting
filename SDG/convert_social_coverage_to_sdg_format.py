#!/usr/bin/env python3
"""
Convert social coverage.csv from wide format to SDG long format
Using GDP_processed.csv as template
"""

import pandas as pd
import numpy as np

def convert_social_coverage_to_sdg_format():
    """Convert social coverage.csv to SDG format like GDP_processed.csv"""
    print("🔧 Converting social coverage.csv to SDG format...")
    
    try:
        # Read the wide-format social coverage file
        df = pd.read_csv('social coverage.csv', sep=';')
        print(f"✅ Loaded social coverage data: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)[:10]}...") # Show first 10 columns
        
        # Identify year columns (should be numeric)
        year_columns = []
        for col in df.columns:
            try:
                if col.strip().isdigit() and 1960 <= int(col.strip()) <= 2030:
                    year_columns.append(col.strip())
            except:
                continue
        
        print(f"📅 Found {len(year_columns)} year columns: {year_columns[:5]}...")
        
        # Prepare the data for melting
        id_columns = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
        
        # Check if all required columns exist
        missing_cols = [col for col in id_columns if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return
        
        # Melt the dataframe to long format
        df_melted = pd.melt(df, 
                           id_vars=id_columns,
                           value_vars=year_columns,
                           var_name='Year',
                           value_name='Social_Coverage')
        
        print(f"📊 After melting: {len(df_melted)} rows")
        
        # Clean the data
        # Convert comma decimal separators to dots
        df_melted['Social_Coverage'] = df_melted['Social_Coverage'].astype(str).str.replace(',', '.')
        
        # Convert to numeric, coercing errors to NaN
        df_melted['Social_Coverage'] = pd.to_numeric(df_melted['Social_Coverage'], errors='coerce')
        df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce')
        
        # Remove rows with missing values
        df_clean = df_melted.dropna(subset=['Year', 'Social_Coverage'])
        
        print(f"📊 After cleaning: {len(df_clean)} rows")
        print(f"📊 Year range: {df_clean['Year'].min():.0f}-{df_clean['Year'].max():.0f}")
        print(f"📊 Social Coverage range: {df_clean['Social_Coverage'].min():.2f}-{df_clean['Social_Coverage'].max():.2f}%")
        
        # Rename columns to SDG format
        df_renamed = df_clean.rename(columns={
            'Country Name': 'GeoAreaName',
            'Country Code': 'GeoAreaCode', 
            'Indicator Name': 'SeriesDescription',
            'Indicator Code': 'SeriesCode',
            'Year': 'TimePeriod',
            'Social_Coverage': 'Value'
        })
        
        # Add standard SDG columns
        df_renamed['Source'] = 'World Bank'
        df_renamed['Location'] = ''
        df_renamed['Sex'] = ''
        df_renamed['TypeofProduct'] = ''
        df_renamed['GroundsOfDiscrimination'] = ''
        df_renamed['Units'] = 'PERCENT'
        
        # Reorder columns to match SDG format
        sdg_columns = ['GeoAreaName', 'GeoAreaCode', 'SeriesDescription', 'SeriesCode', 
                      'TimePeriod', 'Value', 'Source', 'Location', 'Sex', 
                      'TypeofProduct', 'GroundsOfDiscrimination', 'Units']
        df_final = df_renamed[sdg_columns]
        
        # Save as semicolon-separated SDG format
        df_final.to_csv('social_coverage_processed.csv', sep=';', index=False)
        
        print(f"✅ Converted social_coverage_processed.csv: {len(df_final)} rows, {len(df_final.columns)} columns, semicolon-separated")
        
        # Test the output
        test_df = pd.read_csv('social_coverage_processed.csv', sep=';')
        print(f"✅ Verification: {len(test_df)} rows, {len(test_df.columns)} columns")
        print(f"Sample countries: {test_df['GeoAreaName'].unique()[:3]}")
        print(f"Recent years: {sorted(test_df['TimePeriod'].unique())[-3:]}")
        print(f"Value range: {test_df['Value'].min():.2f}-{test_df['Value'].max():.2f}%")
        
    except Exception as e:
        print(f"❌ Error converting social coverage: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    convert_social_coverage_to_sdg_format() 