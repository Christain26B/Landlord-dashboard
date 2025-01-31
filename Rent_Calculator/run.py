import joblib
import pandas as pd
import numpy as np
from functions.estimator_functions import geocode_address
import os

def load_model_files():
    """Load all necessary model files"""
    model = joblib.load('model/rent_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    encoders = joblib.load('model/label_encoders.pkl')
    features = joblib.load('model/feature_columns.pkl')
    return model, scaler, encoders, features

def get_user_input():
    """Get property information from user"""
    print("\nPlease enter the property details:")
    
    # Get address
    address = input("\nEnter the property address: ")

    # Get property type
    print("\nProperty Types available:")
    property_types = [
        "Bungalow Property", "Converted Flat", "Detached Bungalow", "Detached House",
        "Detached Property", "End Terrace Bungalow", "End Terrace House",
        "End Terrace Property", "Flat/Maisonette", "Mid Terrace Bungalow",
        "Mid Terrace House", "Mid Terrace Property", "Purpose Built Flat",
        "Semi-Detached Bungalow", "Semi-Detached House", "Semi-Detached Property",
        "Terrace Property", "Terraced", "Terraced Bungalow"
    ]
    for i, ptype in enumerate(property_types, 1):
        print(f"{i}. {ptype}")
    property_type = input("Enter the property type (1-19): ")
    property_type_map = {str(i): ptype for i, ptype in enumerate(property_types, 1)}
    property_type = property_type_map.get(property_type, 'Flat')

    # Get bedrooms and bathrooms
    bedrooms = int(input("\nEnter number of bedrooms: "))
    bathrooms = int(input("Enter number of bathrooms: "))
    livingrooms = int(input("Enter number of living rooms: "))
    
    return {
        'bathrooms': bathrooms,
        'bedrooms': bedrooms,
        'address': address,
        'livingRooms': livingrooms,
        'propertyType': property_type,
    }


def predict_rent(property_info):
    """Process address and make prediction"""
    try:
        # Load model files
        model, scaler, encoders, features = load_model_files()
        
        # Process address to get coordinates
        print("\nProcessing address...")
        location = geocode_address(property_info['address'])
        if location is None:
            raise ValueError("Could not process the address. Please check if it's correct.")
        
        # Combine location with other property info
        property_data = {
            'latitude': location['lat'],
            'longitude': location['lng'],
            'propertyType': property_info['propertyType'],
            'livingRooms': property_info['livingRooms'],
            'bedrooms': property_info['bedrooms'],
            'bathrooms': property_info['bathrooms']
        }
        
        # Create DataFrame with correct feature order
        X_new = pd.DataFrame(columns=features)
        X_new.loc[0] = 0  # Initialize with zeros

        # Fill in the basic features
        for col in ['latitude', 'longitude', 'bedrooms', 'bathrooms', 'livingRooms']:
            X_new[col] = property_data[col]

        # Encode categorical features
        X_new['propertyType'] = encoders['propertyType'].transform([property_data['propertyType']])[0]

        # Add feature interactions and polynomial features
        X_new['BEDS_BATHS'] = X_new['bedrooms'] * X_new['bathrooms']
        X_new['LOCATION'] = X_new['latitude'] * X_new['longitude']
        X_new['BEDROOMS_SQ'] = X_new['bedrooms'] ** 2
        X_new['BATHROOMS_SQ'] = X_new['bathrooms'] ** 2

        # Scale numeric features
        numeric_features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'livingRooms']
        X_new[numeric_features] = scaler.transform(X_new[numeric_features])
        
        # Make prediction (log scale)
        pred_log = model.predict(X_new)[0]
        
        # Transform back from log scale
        pred = np.expm1(pred_log)
        
        # Calculate confidence intervals
        confidence = 0.10  # 10% confidence interval
        lower_bound = pred * (1 - confidence)
        upper_bound = pred * (1 + confidence)
        
        return {
            'predicted_rent': round(pred, 2),
            'lower_bound': round(lower_bound, 2),
            'upper_bound': round(upper_bound, 2),
            'location': location
        }
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        return None

def main():
    print("Welcome to the Rent Calculator!")
    
    try:
        # Get property information from user
        property_info = get_user_input()
        
        # Make prediction
        result = predict_rent(property_info)
        
        if result:
            print("\nRent Prediction Results:")
            print(f"Estimated Rent: £{result['predicted_rent']:,.2f}")
            print(f"Likely Range: £{result['lower_bound']:,.2f} - £{result['upper_bound']:,.2f}")
            
            # Show location details for verification
            print(f"\nLocation Used:")
            print(f"Latitude: {result['location']['lat']:.6f}")
            print(f"Longitude: {result['location']['lng']:.6f}")
            
            # Additional information about the prediction
            print("\nNote: This is an estimate based on similar properties in the area.")
            print("The actual rent may vary based on specific property features and market conditions.")
            
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please try again with valid inputs.")

if __name__ == "__main__":
    main()
