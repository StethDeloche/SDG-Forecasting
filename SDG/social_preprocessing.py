import pandas as pd
import numpy as np

def process_social_data():
    # Read the Social Coverage data, skipping the first 4 metadata rows
    # The file uses double quotes and commas in a specific format
    df = pd.read_csv('Coverage of social safety net programs.csv', skiprows=4, quotechar='"', skipinitialspace=True)
    
    print("Original Social Coverage data shape:", df.shape)
    print("Columns:", df.columns.tolist()[:5])  # Show first 5 columns
    
    # Check if we have the expected columns
    if 'Country Name' not in df.columns:
        print("Column names found:", df.columns.tolist()[:5])
        # Try to parse manually if automatic parsing failed
        print("Attempting manual parsing...")
        
        # Read the file line by line and parse manually
        with open('Coverage of social safety net programs.csv', 'r') as f:
            lines = f.readlines()
        
        # Skip metadata lines and get header
        header_line = lines[4].strip()
        print("Header line:", header_line[:200] + "...")
        
        # Parse header manually - this format is very specific
        # The format is: "Country Name,""Country Code"",""Indicator Name"",...
        
        # First, let's handle this step by step
        # Remove the outer quotes
        header_line = header_line.strip('"')
        if header_line.endswith(','):
            header_line = header_line[:-1]
        
        # Now split by '","' but we need to handle the first element specially
        # The first element is "Country Name,""Country Code" instead of just "Country Name"
        parts = header_line.split('","')
        
        headers = []
        for i, part in enumerate(parts):
            if i == 0:
                # First part: "Country Name,""Country Code" -> split by ','
                first_parts = part.split(',')
                if len(first_parts) >= 2:
                    headers.append(first_parts[0].strip('"'))  # Country Name
                    headers.append(first_parts[1].strip('"'))  # Country Code
                else:
                    headers.append(part.strip('"'))
            else:
                headers.append(part.strip('"'))
        
        print(f"Parsed {len(headers)} headers:", headers[:10])
        
        # Parse data lines
        data_lines = []
        for line in lines[5:]:  # Skip header and metadata
            if line.strip():
                # Similar parsing for data lines
                line = line.strip().strip('"')
                if line.endswith(','):
                    line = line[:-1]
                    
                parts = line.split('","')
                row = []
                
                for i, part in enumerate(parts):
                    if i == 0:
                        # First part might have the same issue
                        first_parts = part.split(',')
                        if len(first_parts) >= 2:
                            row.append(first_parts[0].strip('"'))
                            row.append(first_parts[1].strip('"'))
                        else:
                            row.append(part.strip('"'))
                    else:
                        row.append(part.strip('"'))
                
                if len(row) == len(headers):
                    data_lines.append(row)
                elif len(row) > 0:  # Debug problematic lines
                    print(f"Row length mismatch: expected {len(headers)}, got {len(row)}")
                    if len(data_lines) < 3:  # Only show first few mismatches
                        print(f"First few elements: {row[:5]}")
        
        # Create DataFrame from parsed data
        df = pd.DataFrame(data_lines, columns=headers)
        print(f"Manually parsed data shape: {df.shape}")
    
    print("First few countries:", df['Country Name'].head().tolist())
    
    # Remove rows where Country Name is NaN or empty
    df = df.dropna(subset=['Country Name'])
    df = df[df['Country Name'].str.strip() != '']
    
    print(f"After removing empty countries: {df.shape}")
    
    # Get year columns (should be from 1960 to 2024)
    year_columns = [col for col in df.columns if col.isdigit() and 1960 <= int(col) <= 2024]
    print(f"Found {len(year_columns)} year columns: {year_columns[:5]}...{year_columns[-5:]}")
    
    # Keep only relevant columns
    keep_columns = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'] + year_columns
    df = df[keep_columns]
    
    # Melt the dataframe to have Year and Social Coverage columns
    df_melted = df.melt(
        id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
        value_vars=year_columns,
        var_name='Year',
        value_name='Social_Coverage'
    )
    
    # Convert Year to integer and Social Coverage to numeric
    df_melted['Year'] = df_melted['Year'].astype(int)
    df_melted['Social_Coverage'] = pd.to_numeric(df_melted['Social_Coverage'], errors='coerce')
    
    # Remove rows with missing Social Coverage values
    df_melted = df_melted.dropna(subset=['Social_Coverage'])
    
    print(f"Final processed data shape: {df_melted.shape}")
    print("Sample of processed data:")
    print(df_melted.head(10))
    
    # Save the processed data
    df_melted.to_csv('social_coverage_processed.csv', index=False)
    print("Processed Social Coverage data saved to social_coverage_processed.csv")
    
    # Show some statistics
    print(f"\nData spans from {df_melted['Year'].min()} to {df_melted['Year'].max()}")
    print(f"Number of unique countries: {df_melted['Country Name'].nunique()}")
    print(f"Countries with data: {df_melted['Country Name'].unique()[:10]}")

if __name__ == "__main__":
    process_social_data() 