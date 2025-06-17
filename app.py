import streamlit as st
import numpy as np
import joblib

# Load mô hình
rf_model = joblib.load("random_forest_model.pkl")
lr_model = joblib.load("logistic_model.pkl")

# Giả định accuracy của từng mô hình (có thể load từ file .txt hoặc pickle)
accuracy_rf = 0.91
accuracy_lr = 0.83

# Giao diện
st.set_page_config(page_title="So sánh Mô hình Dự đoán Rượu", layout="centered")
st.title("🍷 Dự đoán chất lượng rượu vang với 2 mô hình học máy")

# Chọn mô hình
model_choice = st.selectbox("Chọn mô hình dự đoán", ["Random Forest", "Logistic Regression"])

# Nhập dữ liệu
st.subheader("🔢 Nhập dữ liệu mẫu rượu")

fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", value=0.3)
citric_acid = st.number_input("Citric Acid", value=0.34)
residual_sugar = st.number_input("Residual Sugar", value=2.0)
chlorides = st.number_input("Chlorides", value=0.045)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=30.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=97.0)
density = st.number_input("Density", value=0.9942)
pH = st.number_input("pH", value=3.3)
sulphates = st.number_input("Sulphates", value=0.7)
alcohol = st.number_input("Alcohol", value=12.8)

if st.button("🔍 Dự đoán"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                            residual_sugar, chlorides, free_sulfur_dioxide,
                            total_sulfur_dioxide, density, pH, sulphates, alcohol]])

    # Chọn mô hình
    if model_choice == "Random Forest":
        prediction = rf_model.predict(input_data)[0]
        accuracy = accuracy_rf
    else:
        prediction = lr_model.predict(input_data)[0]
        accuracy = accuracy_lr

    # Hiển thị kết quả
    st.success(f"🎯 Kết quả dự đoán ({model_choice}): {prediction}")
    st.info(f"📈 Độ chính xác mô hình: {accuracy*100:.2f}%")
