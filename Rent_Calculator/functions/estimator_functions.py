import pandas as pd
import numpy as np
from googlemaps import Client
import time
import os
#from dotenv import load_dotenv
import requests
# Load environment variables
#load_dotenv()

# Get API key from environment variable
OPENCAGE_API_KEY = 'd42a3bc5b523494ab4c1406fe324de3b'

def geocode_address(address):
    """Convert a single address to latitude and longitude"""
    url = f'https://api.opencagedata.com/geocode/v1/json?q={address}&key={OPENCAGE_API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            location = data['results'][0]['geometry']
            return {'lat': location['lat'], 'lng': location['lng']}
    print(f"Error geocoding address: {response.status_code}")
    return None


def process_dataset(df):
    """Process multiple addresses for training data"""

    # Initialize lists for coordinates
    lats = []
    lngs = []

    # Process each address
    for address in df['address']:
        try:
            result = geocode_address(address)
            if result:
                lats.append(result['lat'])
                lngs.append(result['lng'])
            else:
                lats.append(None)
                lngs.append(None)
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            print(f"Error geocoding address {address}: {e}")
            lats.append(None)
            lngs.append(None)

    # Create DataFrame with results
    location_df = pd.DataFrame({
        'latitude': lats,
        'longitude': lngs
    })

    # Fill any missing values with median
    location_df['latitude'].fillna(location_df['latitude'].median(), inplace=True)
    location_df['longitude'].fillna(location_df['longitude'].median(), inplace=True)

    return location_df