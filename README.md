# Wine Quality Prediction

A machine learning project that implements multiple models (Random Forest, SVM, Logistic Regression) to predict wine quality based on physicochemical properties. The models classify wines into two categories (good/poor quality) through a user-friendly web interface.

## Project Overview

This project uses multiple machine learning models to predict wine quality based on chemical properties. The models have been trained on the Red Wine Quality dataset and implemented as a web application using Streamlit. Users can choose between Random Forest and SVM models for predictions.

## Features

The models take into account the following wine characteristics:
- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

## Model Performance

### Random Forest Model
- Accuracy: 90%
- Precision: 
  - Poor Quality (0): 96%
  - Good Quality (1): 47%
- Recall:
  - Poor Quality (0): 93%
  - Good Quality (1): 63%
- F1-Score:
  - Poor Quality (0): 95%
  - Good Quality (1): 54%

### SVM Model
- Accuracy: 86%

## Project Structure

```
├── app.py                     # Streamlit web application
├── random_forest_model.pkl    # Trained Random Forest model
├── svm_model.pkl             # Trained SVM model
├── logistic_model.pkl        # Trained Logistic Regression model
├── RandomForest.ipynb        # Random Forest development notebook
├── SVM-final.ipynb          # SVM development notebook
├── Logistic.ipynb           # Logistic Regression development notebook
└── winequality-red.csv      # Dataset
```

## Installation

1. Clone this repository
2. Install required packages:
```bash
pip install streamlit numpy pandas scikit-learn joblib
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

## Data Processing

The data processing pipeline includes:
- Data cleaning and outlier removal using Isolation Forest
- Feature importance analysis using Random Forest
- Binary classification (0: quality < 7, 1: quality >= 7)
- Train-test split (80-20) with model evaluation
- Multiple model implementations (Random Forest, SVM, Logistic Regression)

## Technologies Used

- Python 3.8+
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib
- Streamlit
- Joblib

## Contributing

Feel free to fork the project and submit pull requests.

## License

This project is open source and available