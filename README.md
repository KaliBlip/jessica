# 🌾 Ghana Food Prices Prediction App

A comprehensive Streamlit web application for predicting food prices in Ghana using machine learning. This app uses a Random Forest model trained on World Food Programme data to forecast food prices based on various market and commodity factors.

## 📊 Dataset Information

This application uses the **Ghana Food Prices** dataset from the World Food Programme Price Database, which contains:

- **Source**: World Food Programme Price Database
- **Coverage**: Ghana food markets
- **Time Period**: Historical data from 2006 onwards
- **Updates**: Weekly updates with monthly data
- **Currency**: Ghanaian Cedi (GHS)
- **Food Items**: Maize, rice, beans, fish, sugar, and other commodities
- **Markets**: 98 countries and 3000+ markets globally

## 🚀 Features

### 🔮 Price Prediction
- Interactive form to input market and commodity details
- Real-time price predictions using trained Random Forest model
- Support for various food categories and commodities
- Geographic location-based predictions

### 📊 Data Analysis
- Comprehensive data visualization and analysis
- Price distribution analysis by category and location
- Geographic mapping of price data
- Time series analysis of price trends

### 📈 Market Trends
- Market comparison and benchmarking
- Seasonal price pattern analysis
- Price volatility analysis
- Feature correlation insights

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download
```bash
# If you have the files locally, navigate to the directory
cd /path/to/your/project
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Required Files
Make sure you have the following files in your project directory:
- `ghana_food_prices_app.py` (main Streamlit app)
- `random_forest_regressor_model.joblib` (trained model)
- `model_columns.joblib` (model column information)
- `cleaned_selected_columns.csv` (sample data)
- `wfp_food_prices_gha_qc.xlsx` (original dataset)

### Step 4: Run the Application
```bash
streamlit run ghana_food_prices_app.py
```

The app will open in your default web browser at `http://localhost:8501`

## 📱 App Navigation

The application has four main sections:

### 🏠 Home
- Overview of the dataset and model performance
- Key insights and statistics
- Sample data preview

### 🔮 Price Prediction
- Interactive form for price prediction
- Input fields for location, commodity, and market details
- Real-time prediction results with confidence metrics

### 📊 Data Analysis
- Price distribution visualizations
- Geographic analysis with interactive maps
- Time series analysis
- Top commodities analysis

### 📈 Market Trends
- Market comparison tables
- Seasonal price patterns
- Price volatility analysis
- Feature correlation matrix

## 🤖 Model Information

### Algorithm
- **Model Type**: Random Forest Regressor
- **Performance**: R² Score = 0.80, MAE = 44.17 GHS
- **Features**: 34 engineered features including:
  - Geographic coordinates (latitude, longitude)
  - Commodity information (category, type, ID)
  - Market details (location, administrative regions)
  - Price metadata (type, currency, flags)

### Preprocessing
- One-hot encoding for categorical variables
- Numeric conversion with error handling
- Missing value imputation
- Feature scaling and normalization

## 📊 Data Schema

The dataset includes the following key columns:

| Column | Type | Description |
|--------|------|-------------|
| `date` | Date | Record date |
| `market` | String | Market name |
| `commodity` | String | Food item name |
| `category` | String | Food category |
| `price` | Float | Price in GHS |
| `usdprice` | Float | Price in USD |
| `unit` | String | Unit of measurement |
| `currency` | String | Currency code (GHS) |
| `admin1` | String | Administrative region |
| `admin2` | String | Administrative district |
| `latitude` | Float | Geographic latitude |
| `longitude` | Float | Geographic longitude |
| `pricetype` | String | Wholesale/Retail |
| `priceflag` | String | Price flag type |

## 🔧 Customization

### Adding New Features
1. Modify the `preprocess_input_data()` function to handle new input fields
2. Update the prediction form in `show_prediction_page()`
3. Ensure the model columns are updated accordingly

### Styling
- Custom CSS is included in the app for better visual appeal
- Modify the `st.markdown()` sections to change styling
- Update colors and layout as needed

## 🐛 Troubleshooting

### Common Issues

1. **Model file not found**
   - Ensure `random_forest_regressor_model.joblib` is in the project directory
   - Check file permissions

2. **Data loading errors**
   - Verify `cleaned_selected_columns.csv` exists and is readable
   - Check file format and encoding

3. **Import errors**
   - Run `pip install -r requirements.txt` to install all dependencies
   - Check Python version compatibility

4. **Port already in use**
   - Use `streamlit run ghana_food_prices_app.py --server.port 8502` to use a different port

## 📈 Performance

- **Model Accuracy**: 80% (R² Score)
- **Prediction Speed**: < 1 second for real-time predictions
- **Data Processing**: Optimized with caching for better performance
- **Memory Usage**: Efficient data handling for large datasets

## 🤝 Contributing

Feel free to contribute to this project by:
- Adding new visualization features
- Improving the prediction accuracy
- Enhancing the user interface
- Adding new data analysis capabilities

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For questions or issues, please refer to the troubleshooting section or create an issue in the project repository.

---

**Note**: This application is for educational and research purposes. Predictions should be used as guidance and not as definitive price forecasts for commercial purposes.
