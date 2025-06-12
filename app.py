
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime

# Set up the page
st.set_page_config(page_title="Cek Risiko Diabetes Yuk!", layout="centered")

# Fun header
st.title("🩺 Cek Risiko Diabetes")
st.markdown("""
Ayo cek seberapa besar risiko diabetes kamu!
Aplikasi ini punya 3 asisten canggih yang akan menganalisis data kamu.
""")

# Load models with error handling
try:
    # These are our AI doctors
    model_ann = load_model("model_ann.h5")
    model_dnn = load_model("model_dnn.h5")
    model_tab = TabNetClassifier()
    model_tab.load_model("tabnet_model.zip")
    scaler = joblib.load("scaler.pkl")
except:
    st.error("Waduh, asisten AI-nya belum siap nih...")
    st.stop()

# Input form with fun explanations
with st.form("cek_diabetes"):
    st.subheader("📋 Data Diri Kamu")

    gender = st.radio("Jenis Kelamin", ["Perempuan", "Laki-Laki"],
                     help="Pria biasanya lebih berisiko, tapi jangan sedih dulu!")

    age = st.slider("Usia", 1, 120, 25,
                   help="Semakin tua, risikonya biasanya meningkat")

    st.write("💊 Riwayat Kesehatan:")
    col1, col2 = st.columns(2)
    with col1:
        hypertension = st.selectbox("Punya hipertensi?", ["Tidak", "Ya"],
                                  help="Tekanan darah tinggi bisa pengaruhi risiko")
    with col2:
        heart_disease = st.selectbox("Punya penyakit jantung?", ["Tidak", "Ya"])

    smoking = st.selectbox("Kebiasaan merokok",
                         ["Tidak Pernah", "Dulu Pernah", "Sekarang Merokok"],
                         help="Rokok itu temennya diabetes lho!")

    st.write("📊 Data Kesehatan:")
    bmi = st.slider("BMI (Indeks Massa Tubuh)", 10.0, 60.0, 22.0,
                   help="Normalnya 18.5-25. Kurang atau lebih bisa berisiko")

    col1, col2 = st.columns(2)
    with col1:
        hba1c = st.number_input("Nilai HbA1c", 3.0, 15.0, 5.0,
                              help="Normal <5.7, waspada 5.7-6.4, diabetes ≥6.5")
    with col2:
        glucose = st.number_input("Gula Darah (mg/dL)", 50.0, 300.0, 100.0,
                                help="Normal <140, waspada 140-200, diabetes ≥200")

    submitted = st.form_submit_button("🚀 Cek Sekarang!")

# Prediction function with fun responses
def interpret_risk(prob):
    if prob >= 0.8:
        return "🔥 Risiko Tinggi! Yuk lebih perhatikan kesehatan!"
    elif prob >= 0.5:
        return "⚠️ Risiko Sedang. Awas ya, perlu lebih hati-hati!"
    else:
        return "✅ Risiko Rendah. Pertahankan gaya hidup sehat!"

# When submitted
if submitted:
    st.balloons()  # Fun celebration!

    # Prepare data
    input_array = np.array([[
        1 if gender == "Laki-Laki" else 0,
        age,
        1 if hypertension == "Ya" else 0,
        1 if heart_disease == "Ya" else 0,
        {"Tidak Pernah": 0, "Dulu Pernah": 1, "Sekarang Merokok": 2}[smoking],
        bmi,
        hba1c,
        glucose
    ]])

    try:
        scaled_input = scaler.transform(input_array)
        prob_ann = model_ann.predict(scaled_input)[0][0]
        prob_dnn = model_dnn.predict(scaled_input)[0][0]
        prob_tab = model_tab.predict_proba(scaled_input)[0][1]
        
        # Show user's data
        st.subheader("📋 Data yang Kamu Masukkan")
        user_df = pd.DataFrame({
            "Info": ["Jenis Kelamin", "Usia", "Hipertensi", "Penyakit Jantung",
                    "Merokok", "BMI", "HbA1c", "Gula Darah"],
            "Nilai": [gender, age, hypertension, heart_disease,
                     smoking, f"{bmi:.1f}", f"{hba1c:.1f}", f"{glucose:.1f}"]
        })
        st.table(user_df)

        # Show results in a fun way
        st.subheader("🎯 Hasil Analisis 3 Dokter AI")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Dokter ANN",
                 value=f"{prob_ann*100:.1f}%",  # Changed from dr_ann_pred
                 help=interpret_risk(prob_ann))  # Changed from dr_ann_pred
        with col2:
            st.metric(label="Dokter DNN",
                 value=f"{prob_dnn*100:.1f}%",  # Changed from dr_dnn_pred
                 help=interpret_risk(prob_dnn))  # Changed from dr_dnn_pred
        with col3:
            st.metric(label="Dokter Tab",
                 value=f"{prob_tab*100:.1f}%",  # Changed from dr_tab_pred
                 help=interpret_risk(prob_tab))  # Changed from dr_tab_pred

        # Visual comparison
        st.subheader("📊 Perbandingan Hasil")
        fig, ax = plt.subplots()
        models = ["Dokter ANN", "Dokter DNN", "Dokter Tab"]
        probs = [prob_ann, prob_dnn, prob_tab]
        colors = ["#FF9AA2", "#FFB7B2", "#FFDAC1"]

        bars = ax.bar(models, probs, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Tingkat Risiko")

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f"{height*100:.1f}%",
                    ha='center', va='bottom')

        st.pyplot(fig)

        # Fun health tips
        st.subheader("💡 Tips Sehat ala Dokter AI")

        if any(p >= 0.5 for p in [prob_ann, prob_dnn, prob_tab]):
            st.warning("""
            **Waduh, ada tanda-tanda risiko nih!**
            Yuk lakukan ini:
            - Kurangi gula dan makanan manis
            - Olahraga 30 menit/hari
            - Cek rutin ke dokter
            - Stop merokok kalau masih merokok
            """)
        else:
            st.success("""
            **Mantap! Hasil kamu bagus!**
            Tetap pertahankan dengan:
            - Makan sayur dan buah tiap hari
            - Jangan lupa bergerak aktif
            - Jaga berat badan ideal
            - Cek kesehatan rutin
            """)


    except Exception as e:
        st.error(f"Waduh error nih: {e}")
        st.image("https://media.giphy.com/media/3o7abKhOpu0NwenH3O/giphy.gif",
                caption="Dokter AI-nya lagi error, coba lagi ya!")
