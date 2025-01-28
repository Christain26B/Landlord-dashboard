import pandas as pd
import numpy as np
from googlemaps import Client
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variable
GOOGLE_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError("Google Maps API key not found in environment variables")

def geocode_address(address):
    """Convert a single address to latitude and longitude"""
    gmaps = Client(key=GOOGLE_API_KEY)
    try:
        result = gmaps.geocode(address)
        if result:
            location = result[0]['geometry']['location']
            return {'lat': location['lat'], 'lng': location['lng']}
    except Exception as e:
        print(f"Error geocoding address: {e}")
    return None

def process_dataset(df):
    """Process multiple addresses for training data"""
    gmaps = Client(key=GOOGLE_API_KEY)
    
    # Initialize lists for coordinates
    lats = []
    lngs = []
    
    # Process each address
    for address in df['address']:
        try:
            result = gmaps.geocode(address)
            if result:
                location = result[0]['geometry']['location']
                lats.append(location['lat'])
                lngs.append(location['lng'])
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