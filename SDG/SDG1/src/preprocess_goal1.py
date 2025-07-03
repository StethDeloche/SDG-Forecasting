import pandas as pd
import os

def preprocess_data():
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        root_dir = os.path.dirname(parent_dir)
        
        # Read the raw data
        input_file = os.path.join(root_dir, 'Goal1.csv')
        output_file = os.path.join(root_dir, 'Goal1_processed.csv')
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file {input_file} not found")
            return False
            
        # Read the CSV file with error handling
        print("Reading Goal1.csv...")
        df = pd.read_csv(input_file, sep=';', on_bad_lines='skip')
        
        # Check required columns
        required_columns = ['Indicator', 'SeriesCode', 'SeriesDescription', 'GeoAreaName', 'TimePeriod', 'Value', 'Source', 'Sex','Age']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
            
        # Clean the data
        print("Cleaning data...")
        df = df[required_columns].copy()
        df = df.dropna(subset=['Value'])
        df['TimePeriod'] = df['TimePeriod'].astype(str)
        
        # Sort the data
        print("Sorting data...")
        df = df.sort_values(['Indicator', 'SeriesCode', 'GeoAreaName', 'TimePeriod'])
        
        # Save processed data
        print("Saving processed data...")
        df.to_csv(output_file, index=False)
        print(f"Successfully processed data and saved to {output_file}")
        print(f"Processed {len(df)} rows of data")
        return True
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return False

if __name__ == "__main__":
    preprocess_data() 