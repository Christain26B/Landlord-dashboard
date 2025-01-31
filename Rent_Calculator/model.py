import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
import numpy as np
import joblib
import os

def train_rent_model(data_path='data/kaggle_london_house_price_data.csv'):
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Load preprocessed dataset
    print("Loading preprocessed dataset...")
    df = pd.read_csv(data_path)
    
    # # Remove outliers using IQR method
    # print("Removing outliers...")
    # Q1 = df['rent'].quantile(0.05)
    # Q3 = df['rent'].quantile(0.95)
    # IQR = Q3 - Q1
    # df = df[
    #     (df['rent'] >= Q1 - 1.5 * IQR) &
    #     (df['rent'] <= Q3 + 1.5 * IQR)
    # ]
    
    # Create label encoders dictionary
    label_encoders = {}
    
    # Encode categorical columns
    categorical_columns = ['propertyType']
    for col in categorical_columns:
        if col in df.columns:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col].astype(str))

        # Prepare initial features
        feature_columns = ['latitude', 'longitude', 'bathrooms', 'bedrooms', 'livingRooms', 'propertyType']
        X = df[feature_columns].copy()
    

    # Use RobustScaler instead of StandardScaler for better handling of outliers
    numeric_features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'livingRooms']
    scaler = RobustScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])

    # Add feature interactions
    print("Adding feature interactions...")
    X['BEDS_BATHS'] = X['bedrooms'] * X['bathrooms']
    X['LOCATION'] = X['latitude'] * X['longitude']

    # Add polynomial features for bedrooms and bathrooms
    X['BEDROOMS_SQ'] = X['bedrooms'] ** 2
    X['BATHROOMS_SQ'] = X['bathrooms'] ** 2

    # Target variable
    y = df['rentEstimate_currentPrice']

    # Check for NaN, infinity, or too large values in the target variable
    y = y.replace([np.inf, -np.inf, np.nan], 0)

    # Log transform the target variable
    y_log = np.log1p(y)

    # Split into train, validation, and test sets
    print("Splitting data into train, validation, and test sets...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    # Store training data globally for prediction intervals
    global X_train_global, y_train_global
    X_train_global = X_train
    y_train_global = y_train

    # Train model with tuned parameters
    print("Training model...")
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42
    )
    
    # Train and evaluate
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Make predictions and transform back from log scale
    y_pred = np.expm1(model.predict(X_test))
    y_test = np.expm1(y_test)
    
    # Calculate error metrics
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mse = np.mean((y_test - y_pred) ** 2)
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
    
    print("\nModel Performance:")
    print(f"R-squared: {r2:.4f}")
    print(f"MSE: £{mse:.2f}")
    print(f"RMSE: £{rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save model and associated objects
    print("\nSaving model files...")
    joblib.dump(model, 'model/rent_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(label_encoders, 'model/label_encoders.pkl')
    joblib.dump(list(X.columns), 'model/feature_columns.pkl')
    
    return model, scaler, label_encoders, list(X.columns)

def predict_with_intervals(model, X, percentile=0.10):
    """
    Predict rent with lower and upper bounds.
    """
    # Get the base prediction in log scale
    pred_log = model.predict(X)
    
    # Transform prediction back to original scale
    pred = np.expm1(pred_log)
    
    # Calculate prediction intervals in log scale
    train_pred_log = model.predict(X_train_global)
    residuals_log = y_train_global - train_pred_log
    
    lower_bound_log = pred_log + np.percentile(residuals_log, percentile * 100)
    upper_bound_log = pred_log + np.percentile(residuals_log, (1 - percentile) * 100)
    
    # Transform bounds back to original scale
    lower_bound = np.expm1(lower_bound_log)
    upper_bound = np.expm1(upper_bound_log)
    
    return pred, lower_bound, upper_bound

def predict_rent_with_range(model, scaler, encoders, features, property_data):
    """
    Make a prediction with confidence intervals for new data.
    """
    # Create DataFrame with correct feature order
    X_new = pd.DataFrame(columns=features)
    X_new.loc[0] = 0  # Initialize with zeros
    
    # Fill in the basic features
    for col in ['latitude', 'longitude', 'bathrooms', 'bedrooms', 'livingRooms']:
        X_new[col] = property_data[col]

    # Encode categorical features
    X_new['propertyType'] = encoders['propertyType'].transform([property_data['propertyType']])[0]

    # Add feature interactions
    X_new['BEDS_BATHS'] = X_new['bedrooms'] * X_new['bathrooms']
    X_new['LOCATION'] = X_new['latitude'] * X_new['longitude']

    # Scale numeric features
    numeric_features = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'livingRooms']
    X_new[numeric_features] = scaler.transform(X_new[numeric_features])


    # Get predictions with intervals
    pred, lower, upper = predict_with_intervals(model, X_new)

    return {
        'predicted_rent': round(pred[0], 2),
        'lower_bound': round(lower[0], 2),
        'upper_bound': round(upper[0], 2)
    }

if __name__ == "__main__":
    model, scaler, encoders, features = train_rent_model()
    print("\nModel trained successfully!")
    
    # Example prediction with range
    sample_property = {
        'bathrooms': 1,
        'bedrooms': 2,
        'latitude': 51.5074,
        'longitude': -0.1278,
        'livingRooms': 1,
        'propertyType': 'Purpose Built Flat',
    }
    
    result = predict_rent_with_range(model, scaler, encoders, features, sample_property)
    print("\nSample Prediction:")
    print(f"Predicted Rent: £{result['predicted_rent']}")
    print(f"Range: £{result['lower_bound']} - £{result['upper_bound']}")
