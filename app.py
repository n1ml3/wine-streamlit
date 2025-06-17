import streamlit as st
import joblib
import numpy as np

# Load mô hình
model = joblib.load("random_forest_model.pkl")

st.title("Dự đoán chất lượng rượu vang")

# Nhập dữ liệu
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0)
citric_acid = st.number_input("Citric Acid", min_value=0.0)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0)
chlorides = st.number_input("Chlorides", min_value=0.0)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0)
density = st.number_input("Density", min_value=0.0)
pH = st.number_input("pH", min_value=0.0)
sulphates = st.number_input("Sulphates", min_value=0.0)
alcohol = st.number_input("Alcohol", min_value=0.0)

if st.button("Dự đoán"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
    prediction = model.predict(input_data)
    st.success(f"Chất lượng dự đoán: {prediction[0]}")
