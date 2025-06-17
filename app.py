import streamlit as st
import numpy as np
import joblib

# Tải các mô hình
rf_model = joblib.load("random_forest_model.pkl")
svm_model = joblib.load("svm_model.pkl")

# Độ chính xác mô hình
accuracy_rf = 0.91
accuracy_svm = 0.86

# Giao diện
st.set_page_config(page_title="Wine Quality Prediction", layout="centered")
st.title("🍷 Dự đoán chất lượng rượu vang")

# Chọn mô hình
model_choice = st.selectbox(
    "🔍 Chọn mô hình dự đoán",
    ["Random Forest", "Support Vector Machine"]
)

st.markdown("Những ô dữ liệu có màu đỏ là của mô hình SVM")

st.markdown("---")

st.subheader("📥 Nhập dữ liệu mẫu rượu")

highlight = model_choice == "Support Vector Machine"

# Hàm tạo label có highlight
def input_label(label, is_highlight=False):
    if is_highlight:
        st.markdown(f"<span style='color:#d63384;font-weight:bold'>{label}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='font-weight:normal'>{label}</span>", unsafe_allow_html=True)

# Nhập dữ liệu
input_label("Fixed Acidity")
fixed_acidity = st.number_input("", key="fixed", value=7.4)

input_label("Volatile Acidity", highlight)
volatile_acidity = st.number_input("", key="volatile", value=0.3)

input_label("Citric Acid")
citric_acid = st.number_input("", key="citric", value=0.34)

input_label("Residual Sugar")
residual_sugar = st.number_input("", key="residual", value=2.0)

input_label("Chlorides")
chlorides = st.number_input("", key="chlorides", value=0.045)

input_label("Free Sulfur Dioxide")
free_sulfur_dioxide = st.number_input("", key="free_so2", value=30.0)

input_label("Total Sulfur Dioxide")
total_sulfur_dioxide = st.number_input("", key="total_so2", value=97.0)

input_label("Density")
density = st.number_input("", key="density", value=0.9942)

input_label("pH")
pH = st.number_input("", key="pH", value=3.3)

input_label("Sulphates", highlight)
sulphates = st.number_input("", key="sulphates", value=0.7)

input_label("Alcohol", highlight)
alcohol = st.number_input("", key="alcohol", value=12.8)

# Nút dự đoán
if st.button("🚀 Dự đoán chất lượng"):
    # Chuẩn bị dữ liệu đầy đủ
    full_input = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                            residual_sugar, chlorides, free_sulfur_dioxide,
                            total_sulfur_dioxide, density, pH, sulphates, alcohol]])

    # Dữ liệu cho SVM
    svm_input = np.array([[alcohol, sulphates, volatile_acidity]])

    # Dự đoán theo mô hình
    if model_choice == "Random Forest":
        prediction = rf_model.predict(full_input)[0]
        accuracy = accuracy_rf
    else:
        prediction = svm_model.predict(svm_input)[0]
        accuracy = accuracy_svm

    st.success(f"🎯 Kết quả dự đoán ({model_choice}): {prediction}")
    st.info(f"📈 Độ chính xác mô hình: {accuracy*100:.2f}%")
