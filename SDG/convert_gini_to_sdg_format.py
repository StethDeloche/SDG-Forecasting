#!/usr/bin/env python3
"""
Convert GINI.csv from wide format to SDG long format
Using GDP_processed.csv as template
"""

import pandas as pd
import numpy as np

def convert_gini_to_sdg_format():
    """Convert GINI.csv to SDG format like GDP_processed.csv"""
    print("🔧 Converting GINI.csv to SDG format...")
    
    try:
        # Read the wide-format GINI file
        df = pd.read_csv('GINI.csv', sep=';')
        print(f"✅ Loaded GINI data: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)[:10]}...") # Show first 10 columns
        
        # Identify year columns (should be numeric)
        year_columns = []
        for col in df.columns:
            try:
                if col.strip().isdigit() and 1960 <= int(col.strip()) <= 2030:
                    year_columns.append(col)
            except:
                continue
        
        print(f"📊 Found {len(year_columns)} year columns: {year_columns[:5]}...")
        
        # Keep only the basic info columns and year columns
        info_columns = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
        columns_to_keep = info_columns + year_columns
        
        # Filter to existing columns
        available_columns = [col for col in columns_to_keep if col in df.columns]
        df_filtered = df[available_columns]
        
        print(f"📊 Filtered to {len(available_columns)} columns")
        
        # Melt from wide to long format
        df_long = df_filtered.melt(
            id_vars=info_columns,
            value_vars=year_columns,
            var_name='Year',
            value_name='Gini_index'
        )
        
        print(f"📊 Melted to long format: {len(df_long)} rows")
        
        # Clean the data - HANDLE COMMA DECIMAL SEPARATOR!
        df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')
        
        # Convert comma decimal separator to dot
        print("🔧 Converting comma decimal separators to dots...")
        df_long['Gini_index'] = df_long['Gini_index'].astype(str).str.replace(',', '.', regex=False)
        df_long['Gini_index'] = pd.to_numeric(df_long['Gini_index'], errors='coerce')
        
        # Count before and after cleaning
        before_clean = len(df_long)
        print(f"📊 Before cleaning: {before_clean} rows")
        print(f"📊 Non-null GINI values: {df_long['Gini_index'].notna().sum()}")
        
        # Remove rows with missing values (but be less aggressive)
        df_clean = df_long.dropna(subset=['Year', 'Gini_index'])
        print(f"📊 After cleaning: {len(df_clean)} rows")
        
        if len(df_clean) < 100:
            print("⚠️  Very few rows! Let's check what's happening...")
            print("Sample GINI values:")
            print(df_long['Gini_index'].head(20))
            print("Value counts:")
            print(df_long['Gini_index'].value_counts().head(10))
        
        # Convert to SDG format (exactly like GDP_processed.csv)
        df_sdg = pd.DataFrame()
        df_sdg['GeoAreaName'] = df_clean['Country Name']
        df_sdg['GeoAreaCode'] = df_clean['Country Code'] 
        df_sdg['SeriesDescription'] = df_clean['Indicator Name']
        df_sdg['SeriesCode'] = df_clean['Indicator Code']
        df_sdg['TimePeriod'] = df_clean['Year'].astype(int)
        df_sdg['Value'] = df_clean['Gini_index']
        df_sdg['Source'] = 'World Bank'
        df_sdg['Location'] = ''
        df_sdg['Sex'] = ''
        df_sdg['TypeofProduct'] = ''
        df_sdg['GroundsOfDiscrimination'] = ''
        df_sdg['Units'] = 'INDEX'
        
        # Sort by country and year
        df_sdg = df_sdg.sort_values(['GeoAreaName', 'TimePeriod'])
        
        print(f"✅ Created SDG format: {len(df_sdg)} rows, {len(df_sdg.columns)} columns")
        print(f"Sample countries: {df_sdg['GeoAreaName'].unique()[:5]}")
        print(f"Year range: {df_sdg['TimePeriod'].min()}-{df_sdg['TimePeriod'].max()}")
        print(f"Value range: {df_sdg['Value'].min():.2f}-{df_sdg['Value'].max():.2f}")
        
        # Save as semicolon-separated (like GDP_processed.csv)
        df_sdg.to_csv('GINI_processed.csv', sep=';', index=False)
        
        print(f"✅ Saved as GINI_processed.csv (SDG format, semicolon-separated)")
        
        # Verify by reading it back
        test_df = pd.read_csv('GINI_processed.csv', sep=';')
        print(f"✅ Verification: {len(test_df)} rows, {len(test_df.columns)} columns")
        print(f"Columns: {list(test_df.columns)}")
        
    except Exception as e:
        print(f"❌ Error converting GINI data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    convert_gini_to_sdg_format() 