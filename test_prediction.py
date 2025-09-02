import joblib
import pandas as pd

# Load model and columns
model = joblib.load('random_forest_regressor_model.joblib')
columns = joblib.load('model_columns.joblib')

print("Model expects these columns:")
for i, col in enumerate(columns):
    print(f"{i}: {col}")

print("\n" + "="*50)

# Test data
test_data = {
    'category': 'cereals and tubers',
    'commodity': 'Maize',
    'latitude': 6.68,
    'longitude': -1.62,
    'commodity_id': 0,
    'usdprice': 0.0
}

print("Input data:")
for key, value in test_data.items():
    print(f"{key}: {value}")

print("\n" + "="*50)

# Create DataFrame and preprocess
df_input = pd.DataFrame([test_data])
print("Before one-hot encoding:")
print("Columns:", list(df_input.columns))

# One-hot encode categorical columns
df_input = pd.get_dummies(df_input, columns=['category', 'commodity'], dummy_na=False)
print("\nAfter one-hot encoding:")
print("Columns:", list(df_input.columns))

# Ensure all model columns are present
for col in columns:
    if col not in df_input.columns:
        df_input[col] = 0

print("\nAfter adding missing columns:")
print("Columns:", list(df_input.columns))

# Reorder columns
df_input = df_input[columns]

print("\nFinal columns order:")
for i, col in enumerate(df_input.columns):
    print(f"{i}: {col}")

print(f"\nShape: {df_input.shape}")
print(f"Columns match: {list(df_input.columns) == list(columns)}")

# Make prediction
try:
    prediction = model.predict(df_input)[0]
    print(f"\nPrediction successful: {prediction}")
except Exception as e:
    print(f"\nPrediction error: {e}")
