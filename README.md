# Adult Income Prediction Streamlit App

A comprehensive web application for predicting adult income levels using three different machine learning models.

## Features

### 🔮 Prediction Mode
- Interactive feature selection interface
- Real-time predictions using your trained models:
  - Logistic Regression
  - XGBoost (custom implementation)
  - Random Forest
- Prediction confidence visualization
- Support for both categorical and continuous features

### 📊 Model Comparison Mode
- Adjustable train/validation split ratios
- F1 score comparison across all three models
- Performance trend analysis across different data splits
- Interactive visualizations with Plotly

## Setup and Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Files:**
   - Ensure `adult.csv` is in the same directory
   - Keep your model files (`LogisticRegression.py`, `XGBoost.py`, `RandomForest.py`) in the same directory

3. **Run the Application:**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Navigate the App:**
   - Use the sidebar to switch between Prediction and Comparison modes
   - In Prediction Mode: Select a model, adjust features, and get predictions
   - In Comparison Mode: Adjust data splits and compare model performance

## Model Integration

The app integrates your existing implementations:
- **Logistic Regression**: Uses your preprocessing pipeline with StandardScaler
- **XGBoost**: Custom implementation with early stopping and histogram-based splits
- **Random Forest**: Sklearn-based implementation with balanced class weights

## Data Preprocessing

Follows your existing preprocessing approach:
- Handles missing values by replacing '?' with mode
- Drops 'education' column
- One-hot encodes categorical variables
- Fills remaining missing values with mean

## Performance Metrics

- Primary metric: F1 Score (weighted average)
- Includes confidence intervals and trend analysis
- Interactive visualizations for easy comparison
