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
    print("1. Flat")
    print("2. House")
    print("3. Studio")
    property_type = input("Enter the property type (1-3): ")
    property_type_map = {'1': 'Flat', '2': 'House', '3': 'Studio'}
    property_type = property_type_map.get(property_type, 'Flat')
    
    # Get furnishing type
    print("\nFurnishing Types available:")
    print("1. Furnished")
    print("2. Unfurnished")
    print("3. Part Furnished")
    furnish_type = input("Enter the furnishing type (1-3): ")
    furnish_map = {'1': 'Furnished', '2': 'Unfurnished', '3': 'Part Furnished'}
    furnish_type = furnish_map.get(furnish_type, 'Furnished')
    
    # Get bedrooms and bathrooms
    bedrooms = int(input("\nEnter number of bedrooms: "))
    bathrooms = int(input("Enter number of bathrooms: "))
    
    return {
        'address': address,
        'PROPERTY TYPE': property_type,
        'Furnish Type': furnish_type,
        'BEDROOMS': bedrooms,
        'BATHROOMS': bathrooms
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
            'PROPERTY TYPE': property_info['PROPERTY TYPE'],
            'Furnish Type': property_info['Furnish Type'],
            'BEDROOMS': property_info['BEDROOMS'],
            'BATHROOMS': property_info['BATHROOMS']
        }
        
        # Create DataFrame with correct feature order
        X_new = pd.DataFrame(columns=features)
        X_new.loc[0] = 0  # Initialize with zeros
        
        # Fill in the basic features
        for col in ['latitude', 'longitude', 'BEDROOMS', 'BATHROOMS']:
            X_new[col] = property_data[col]
        
        # Encode categorical features
        X_new['PROPERTY TYPE'] = encoders['PROPERTY TYPE'].transform([property_data['PROPERTY TYPE']])[0]
        X_new['Furnish Type'] = encoders['Furnish Type'].transform([property_data['Furnish Type']])[0]
        
        # Add feature interactions and polynomial features
        X_new['BEDS_BATHS'] = X_new['BEDROOMS'] * X_new['BATHROOMS']
        X_new['LOCATION'] = X_new['latitude'] * X_new['longitude']
        X_new['BEDROOMS_SQ'] = X_new['BEDROOMS'] ** 2
        X_new['BATHROOMS_SQ'] = X_new['BATHROOMS'] ** 2
        
        # Scale numeric features
        numeric_features = ['BEDROOMS', 'BATHROOMS', 'latitude', 'longitude']
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
