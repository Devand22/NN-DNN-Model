import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# =====================
# Load model dan scaler
# =====================
st.set_page_config(page_title="Prediksi Diabetes", layout="centered")

st.title("Prediksi Risiko Diabetes (ANN & DNN)")
st.markdown("Masukkan data pasien untuk memprediksi risiko diabetes menggunakan 2 model neural network.")

try:
    model_ann = load_model("model_ann.h5")
    model_dnn = load_model("model_dnn.h5")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Gagal memuat model atau scaler: {e}")
    st.stop()

# =====================
# Input Form
# =====================
with st.form("input_form"):
    gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-Laki"])
    age = st.number_input("Usia", min_value=1)
    hypertension = st.selectbox("Hipertensi", ["Tidak", "Ya"])
    heart_disease = st.selectbox("Penyakit Jantung", ["Tidak", "Ya"])
    smoking_history = st.selectbox("Riwayat Merokok", ["Tidak Pernah", "Dulu Pernah", "Sekarang Merokok"])
    bmi = st.number_input("BMI", min_value=10.0)
    hba1c_level = st.number_input("HbA1c Level", min_value=3.0)
    glucose = st.number_input("Blood Glucose Level", min_value=50.0)
    submitted = st.form_submit_button("Prediksi")

# =====================
# Interpretasi
# =====================
def interpret(prob):
    if prob >= 0.8:
        return "Risiko Tinggi"
    elif prob >= 0.5:
        return "Risiko Sedang"
    else:
        return "Risiko Rendah"

# =====================
# Prediksi
# =====================
if submitted:
    input_data = np.array([
        1 if gender == "Laki-Laki" else 0,
        age,
        1 if hypertension == "Ya" else 0,
        1 if heart_disease == "Ya" else 0,
        {"Tidak Pernah": 0, "Dulu Pernah": 1, "Sekarang Merokok": 2}[smoking_history],
        bmi,
        hba1c_level,
        glucose
    ]).reshape(1, -1)

    try:
        scaled_input = scaler.transform(input_data)
        pred_ann = model_ann.predict(scaled_input)[0][0]
        pred_dnn = model_dnn.predict(scaled_input)[0][0]

        st.subheader("Hasil Prediksi")
        st.write(f"📘 **ANN**: {interpret(pred_ann)} ({pred_ann:.2f})")
        st.write(f"📗 **DNN**: {interpret(pred_dnn)} ({pred_dnn:.2f})")

        # Bar chart perbandingan
        fig, ax = plt.subplots()
        ax.bar(["ANN", "DNN"], [pred_ann, pred_dnn], color=["skyblue", "lightgreen"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probabilitas")
        ax.set_title("Perbandingan Prediksi ANN vs DNN")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
