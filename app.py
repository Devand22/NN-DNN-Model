import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime

# =============================
# Load Model dan Scaler
# =============================
st.set_page_config(page_title="Prediksi Diabetes", layout="centered")

st.title("Prediksi Risiko Diabetes")
st.markdown("Aplikasi ini membandingkan hasil prediksi dari dua model: **ANN** dan **DNN**.")

try:
    model_ann = load_model("model_ann.h5")
    model_dnn = load_model("model_dnn.h5")
    scaler = pickle.load(open("scaler.pkl", "rb"))
except Exception as e:
    st.error(f"Gagal memuat model atau scaler: {e}")
    st.stop()

# =============================
# Input Data User
# =============================
st.subheader("Form Input Data Pasien")
with st.form("input_form"):
    gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-Laki"])
    age = st.number_input("Usia", min_value=1, max_value=120)
    hypertension = st.selectbox("Hipertensi", ["Tidak", "Ya"])
    heart_disease = st.selectbox("Penyakit Jantung", ["Tidak", "Ya"])
    smoking_history = st.selectbox("Riwayat Merokok", ["Tidak Pernah", "Dulu Pernah", "Sekarang Merokok"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0)
    hba1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0)
    glucose = st.number_input("Blood Glucose Level", min_value=50.0, max_value=300.0)
    
    submitted = st.form_submit_button("Prediksi")

# =============================
# Proses Prediksi
# =============================
def interpret(prob):
    if prob >= 0.8:
        return "Risiko Tinggi"
    elif prob >= 0.5:
        return "Risiko Sedang"
    else:
        return "Risiko Rendah"

if submitted:
    input_array = np.array([
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
        scaled_input = scaler.transform(input_array)
        prob_ann = model_ann.predict(scaled_input)[0][0]
        prob_dnn = model_dnn.predict(scaled_input)[0][0]

        # =============================
        # Tabel Input
        # =============================
        st.subheader("Data yang Dimasukkan")
        input_df = pd.DataFrame([{
            "Jenis Kelamin": gender,
            "Usia": age,
            "Hipertensi": hypertension,
            "Penyakit Jantung": heart_disease,
            "Riwayat Merokok": smoking_history,
            "BMI": bmi,
            "HbA1c": hba1c_level,
            "Glukosa": glucose
        }])
        st.table(input_df)

        # =============================
        # Hasil Prediksi
        # =============================
        st.subheader("Hasil Prediksi")
        st.write(f"📘 **ANN**: {'Berisiko Diabetes' if prob_ann >= 0.5 else 'Tidak Berisiko'} ({prob_ann:.2f}) → *{interpret(prob_ann)}*")
        st.write(f"📗 **DNN**: {'Berisiko Diabetes' if prob_dnn >= 0.5 else 'Tidak Berisiko'} ({prob_dnn:.2f}) → *{interpret(prob_dnn)}*")

        # =============================
        # Visualisasi Bar Chart
        # =============================
        fig, ax = plt.subplots()
        ax.bar(["ANN", "DNN"], [prob_ann, prob_dnn], color=["skyblue", "lightgreen"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probabilitas")
        ax.set_title("Perbandingan Prediksi ANN vs DNN")
        st.pyplot(fig)

        # =============================
        # Tips Gaya Hidup Sehat
        # =============================
        st.markdown("### 🩺 Tips Mencegah Diabetes")
        st.markdown("- Rajin olahraga 30 menit setiap hari")
        st.markdown("- Kurangi konsumsi makanan dan minuman manis")
        st.markdown("- Jaga berat badan ideal dan BMI normal")
        st.markdown("- Cek kadar gula darah dan HbA1c secara berkala")
        st.markdown("- Hindari merokok dan minuman beralkohol")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
