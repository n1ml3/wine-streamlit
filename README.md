# Wine Quality Prediction

A machine learning project that predicts wine quality using Random Forest Classification. The model classifies wines into two categories (good/poor quality) based on their physicochemical properties.

## Project Overview

This project uses a Random Forest Classifier to predict wine quality based on various chemical properties. The model has been trained on the Red Wine Quality dataset and implemented as a web application using Streamlit.

## Features

The model takes into account the following wine characteristics:
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

## Project Structure

```
├── app.py                     # Streamlit web application
├── random_forest_model.pkl    # Trained model
├── RandomForest.ipynb        # Model development notebook  
└── winequality-red.csv       # Dataset
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

## Technologies Used

- Python 3.8+
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib
- Streamlit

## Contributing

Feel free to fork the project and submit pull requests.

## License

This project is licensed under