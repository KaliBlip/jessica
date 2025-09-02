#!/usr/bin/env python3
"""
Test script to verify that all components of the Ghana Food Prices app are working correctly.
"""

import sys
import traceback

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✓ Streamlit imported successfully")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import joblib
        print("✓ Joblib imported successfully")
    except ImportError as e:
        print(f"✗ Joblib import failed: {e}")
        return False
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        print("✓ Plotly imported successfully")
    except ImportError as e:
        print(f"✗ Plotly import failed: {e}")
        return False
    
    try:
        from datetime import datetime, date
        print("✓ Datetime imported successfully")
    except ImportError as e:
        print(f"✗ Datetime import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if the model files can be loaded"""
    print("\nTesting model loading...")
    
    try:
        import joblib
        
        # Test model columns loading
        columns = joblib.load('model_columns.joblib')
        print(f"✓ Model columns loaded successfully ({len(columns)} columns)")
        
        # Test model loading
        model = joblib.load('random_forest_regressor_model.joblib')
        print("✓ Random Forest model loaded successfully")
        
        return True, model, columns
        
    except FileNotFoundError as e:
        print(f"✗ Model file not found: {e}")
        return False, None, None
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False, None, None

def test_data_loading():
    """Test if the data file can be loaded"""
    print("\nTesting data loading...")
    
    try:
        import pandas as pd
        
        df = pd.read_csv('cleaned_selected_columns.csv')
        print(f"✓ Data loaded successfully ({len(df)} rows, {len(df.columns)} columns)")
        
        # Check if header row needs to be removed
        if df.iloc[0]['date'] == 'date':
            df = df.iloc[1:].copy()
            print("✓ Header row removed")
        
        return True, df
        
    except FileNotFoundError as e:
        print(f"✗ Data file not found: {e}")
        return False, None
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False, None

def test_prediction():
    """Test if a sample prediction can be made"""
    print("\nTesting prediction...")
    
    try:
        import pandas as pd
        import joblib
        
        # Load model and columns
        model = joblib.load('random_forest_regressor_model.joblib')
        columns = joblib.load('model_columns.joblib')
        
        # Create sample input data
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
        df_input = df_input[columns]  # Ensure correct column order
        
        # Make prediction
        prediction = model.predict(df_input)[0]
        print(f"✓ Sample prediction successful: {prediction:.2f} GHS")
        
        return True
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🌾 Ghana Food Prices App - Component Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install missing packages.")
        return False
    
    # Test model loading
    model_ok, model, columns = test_model_loading()
    if not model_ok:
        print("\n❌ Model loading test failed. Please check model files.")
        return False
    
    # Test data loading
    data_ok, df = test_data_loading()
    if not data_ok:
        print("\n❌ Data loading test failed. Please check data file.")
        return False
    
    # Test prediction
    if not test_prediction():
        print("\n❌ Prediction test failed.")
        return False
    
    print("\n" + "=" * 50)
    print("✅ All tests passed! The app should work correctly.")
    print("\nTo run the app, use:")
    print("streamlit run ghana_food_prices_app.py")
    print("\nThe app will be available at: http://localhost:8501")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
