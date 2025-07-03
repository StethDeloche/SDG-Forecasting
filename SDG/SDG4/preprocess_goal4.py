import pandas as pd
import os

def preprocess_data():
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        # Read the raw data from the SDG directory
        input_file = os.path.join(parent_dir, 'Goal4.csv')
        output_file = os.path.join(current_dir, 'Goal4_processed.csv')
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file {input_file} not found")
            return False
            
        # Read the CSV file with error handling
        print("Reading Goal4.csv...")
        df = pd.read_csv(input_file, sep=';', on_bad_lines='skip')
        
        # Check required columns
        required_columns = ['Indicator', 'SeriesDescription', 'GeoAreaName', 'TimePeriod', 'Value', 'Source', 'Sex', 'Education level', 'Type of skill', 'Age', 'Location']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
            
        # Clean the data
        print("Cleaning data...")
        df = df[required_columns].copy()
        df = df.dropna(subset=['Value'])
        df['TimePeriod'] = df['TimePeriod'].astype(str)
        
        # Rename Education level column to EducationLevel
        df = df.rename(columns={'Education level': 'EducationLevel', 'Type of skill': 'TypeofSkill'})
        
        # Sort the data
        print("Sorting data...")
        df = df.sort_values(['Indicator', 'GeoAreaName', 'TimePeriod'])
        
        # Save processed data with semicolon separator
        print("Saving processed data...")
        df.to_csv(output_file, index=False, sep=';')
        print(f"Successfully processed data and saved to {output_file}")
        print(f"Processed {len(df)} rows of data")
        return True
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return False

if __name__ == "__main__":
    preprocess_data() 
    