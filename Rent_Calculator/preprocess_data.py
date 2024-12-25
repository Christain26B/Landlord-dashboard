import pandas as pd
from functions.estimator_functions import process_dataset
import os

def preprocess_and_save():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print("Loading original dataset...")
    df = pd.read_csv('/Users/cezarscerbina/Downloads/rent_ads_rightmove_extended.csv')
    
    print("Processing addresses...")
    location_features = process_dataset(df)
    
    # Combine with original features we want to keep
    processed_df = pd.concat([
        location_features[['latitude', 'longitude']],
        df[['PROPERTY TYPE', 'Furnish Type', 'BEDROOMS', 'BATHROOMS', 'rent']]
    ], axis=1)
    
    print("Saving processed dataset...")
    processed_df.to_csv('data/processed_dataset.csv', index=False)
    print("Done! Processed dataset saved to 'data/processed_dataset.csv'")

if __name__ == "__main__":
    preprocess_and_save() 