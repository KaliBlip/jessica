#!/usr/bin/env python3
"""
Script to train the Random Forest model for Ghana Food Prices prediction.
This recreates the model that was trained in the Jupyter notebook.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the data exactly as done in the notebook"""
    print("Loading and preprocessing data...")
    
    # Load the dataset
    df = pd.read_csv('cleaned_selected_columns.csv')
    print(f"Loaded {len(df)} rows")
    
    # Remove the header row that was loaded as data
    if df.iloc[0]['date'] == 'date':
        df = df.iloc[1:].copy()
        print("Removed header row")
    
    # Convert numeric columns
    numeric_cols = ['price', 'usdprice', 'commodity_id', 'latitude', 'longitude']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print("Converted numeric columns")
    
    # Separate target and features
    y = df["price"]
    X = df.drop(columns=["price"])
    
    # Drop non-numeric columns from X
    non_numeric_cols_to_drop = ['date', 'admin1', 'admin2', 'market', 'market_id', 'unit', 'priceflag', 'pricetype', 'currency']
    cols_to_drop = [col for col in non_numeric_cols_to_drop if col in X.columns]
    X = X.drop(columns=cols_to_drop)
    
    print("Dropped non-numeric columns")
    
    # Handle missing values in y
    y = y.fillna(y.mean())
    
    # Identify categorical columns for one-hot encoding
    categorical_cols = ['category', 'commodity']
    
    # Convert categorical columns to numeric using one-hot encoding
    X = pd.get_dummies(X, columns=categorical_cols, dummy_na=False)
    
    print("Applied one-hot encoding")
    
    # Convert booleans to integers
    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(int)
    
    # Handle missing values
    X = X.fillna(0)
    
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X, y

def train_model(X, y):
    """Train the Random Forest model"""
    print("\nTraining Random Forest model...")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate performance
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    return model, X.columns.tolist()

def save_model_and_columns(model, columns):
    """Save the trained model and column information"""
    print("\nSaving model and columns...")
    
    # Save the model
    model_filename = 'random_forest_regressor_model.joblib'
    joblib.dump(model, model_filename)
    print(f"Model saved to '{model_filename}'")
    
    # Save the columns
    columns_filename = 'model_columns.joblib'
    joblib.dump(columns, columns_filename)
    print(f"Columns saved to '{columns_filename}'")
    
    return model_filename, columns_filename

def test_prediction(model, columns):
    """Test the model with a sample prediction"""
    print("\nTesting model with sample prediction...")
    
    # Create sample data (Maize in Kumasi)
    sample_data = {
        'latitude': 6.68,
        'longitude': -1.62,
        'commodity_id': 1,
        'usdprice': 0.0,
        'category_cereals and tubers': 1,
        'category_meat, fish and eggs': 0,
        'category_pulses and nuts': 0,
        'category_vegetables and fruits': 0,
        'commodity_Cassava': 0,
        'commodity_Cowpeas': 0,
        'commodity_Cowpeas (white)': 0,
        'commodity_Eggplants': 0,
        'commodity_Eggs': 0,
        'commodity_Fish (mackerel, fresh)': 0,
        'commodity_Gari': 0,
        'commodity_Maize': 1,
        'commodity_Maize (yellow)': 0,
        'commodity_Meat (chicken)': 0,
        'commodity_Meat (chicken, local)': 0,
        'commodity_Millet': 0,
        'commodity_Onions': 0,
        'commodity_Peppers (dried)': 0,
        'commodity_Peppers (fresh)': 0,
        'commodity_Plantains (apem)': 0,
        'commodity_Plantains (apentu)': 0,
        'commodity_Rice (imported)': 0,
        'commodity_Rice (local)': 0,
        'commodity_Rice (paddy)': 0,
        'commodity_Sorghum': 0,
        'commodity_Soybeans': 0,
        'commodity_Tomatoes (local)': 0,
        'commodity_Tomatoes (navrongo)': 0,
        'commodity_Yam': 0,
        'commodity_Yam (puna)': 0
    }
    
    # Create DataFrame with proper column order
    df_input = pd.DataFrame([sample_data])
    
    # Ensure all columns are present
    for col in columns:
        if col not in df_input.columns:
            df_input[col] = 0
    
    # Reorder columns to match training data
    df_input = df_input[columns]
    
    # Make prediction
    prediction = model.predict(df_input)[0]
    print(f"Sample prediction (Maize in Kumasi): {prediction:.2f} GHS")
    
    return prediction

def main():
    """Main function to train and save the model"""
    print("🌾 Ghana Food Prices - Model Training")
    print("=" * 50)
    
    try:
        # Load and preprocess data
        X, y = load_and_preprocess_data()
        
        # Train model
        model, columns = train_model(X, y)
        
        # Save model and columns
        model_file, columns_file = save_model_and_columns(model, columns)
        
        # Test prediction
        test_prediction(model, columns)
        
        print("\n" + "=" * 50)
        print("✅ Model training completed successfully!")
        print(f"Model file: {model_file}")
        print(f"Columns file: {columns_file}")
        print("\nYou can now run the Streamlit app:")
        print("streamlit run ghana_food_prices_app.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during model training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
