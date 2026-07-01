# Import Library
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# MUAT MODEL & SCALER DARI FOLDER
@st.cache_resource
def load_assets():
    # Membaca file .joblib dari folder models/
    model = joblib.load('models/model_ml.joblib')
    scaler = joblib.load('models/scaler.joblib')
    data_latih = joblib.load('models/data_latih.joblib') # Diperlukan untuk background data SHAP
    return model, scaler, data_latih

model_ml, scaler, data_latih = load_assets()

# ==========================================
# FUNGSI SPK (SAW)
def jalankan_saw(matriks_x, bobot):
    norm_x = np.zeros(matriks_x.shape)
    norm_x[:, 0] = np.min(matriks_x[:, 0]) / matriks_x[:, 0]
    norm_x[:, 1] = np.min(matriks_x[:, 1]) / matriks_x[:, 1]
    norm_x[:, 2] = matriks_x[:, 2] / np.max(matriks_x[:, 2])
    skor_akhir = np.sum(norm_x * bobot, axis=1)
    return skor_akhir

# ==========================================
# ANTARMUKA STREAMLIT UTAMA

# -- A. Titik Masuk dan Konteks Utama --
st.title("⚙️ Simulator Prioritas Pemeliharaan Sistem")
st.write("Sistem Pendukung Keputusan Cerdas untuk Pemeliharaan Infrastruktur IT. Sesuaikan parameter Suhu dan Beban Jaringan di bawah ini untuk mensimulasikan prediksi risiko kegagalan (ML) dan melihat perubahan prioritas perbaikan server secara real-time (SAW).")
st.markdown("---")

# -- B. Area Kendali Intervensi --
st.subheader("🎛️ Panel Intervensi Skenario")
col1, col2 = st.columns(2)
with col1:
    suhu_input = st.slider("Suhu Server (°C)", min_value=30.0, max_value=100.0, value=60.0)
with col2:
    beban_input = st.slider("Beban Jaringan (%)", min_value=0.0, max_value=100.0, value=50.0)

input_dari_slider = pd.DataFrame({'Suhu_Server': [suhu_input], 'Beban_Jaringan': [beban_input]})

st.markdown("---")

# ==========================================
# INFERENSI ML & UI PERINGATAN

# Scaling input baru
input_scaled = scaler.transform(input_dari_slider)

# Prediksi model
prediksi_risiko = model_ml.predict(input_scaled)[0]

# -- C. Hierarki Umpan Balik Visual --
st.subheader("📊 Status Sistem Saat Ini")
st.metric(label="Prediksi Risiko Kegagalan (Skala 1-100)", value=round(prediksi_risiko, 2))

if prediksi_risiko >= 75.0:
    st.error("🚨 PERINGATAN KRITIS: Risiko kegagalan server sangat tinggi!")
elif 50.0 <= prediksi_risiko < 75.0:
    st.warning("⚠️ Waspada: Risiko berada di ambang batas menengah.")
else:
    st.success("✅ Aman: Kondisi server stabil.")

st.markdown("---")

# ==========================================
# MATRIKS SAW & TABEL REKOMENDASI
matriks_x = np.array([
    [prediksi_risiko, 15, 80],
    [55.0, 10, 75],
    [40.0, 20, 90]
])

bobot_ahp = np.array([0.5, 0.3, 0.2])
skor_akhir = jalankan_saw(matriks_x, bobot_ahp)

df_ranking = pd.DataFrame({
    'Alternatif': ['Server A (Simulasi)', 'Server B (Statis)', 'Server C (Statis)'],
    'Skor SAW': skor_akhir
}).sort_values(by='Skor SAW', ascending=False).reset_index(drop=True)

# -- D. Penyajian Data Siap Eksekusi --
st.subheader("📋 Rekomendasi Tindakan (Ranking SAW)")
st.dataframe(df_ranking, use_container_width=True)

with st.expander("💡 Penjelasan: Tabel Keputusan"):
    st.write("Tabel ini mengonversi perhitungan matematis matriks yang rumit menjadi daftar urutan prioritas yang siap dieksekusi (*Actionable Table*). Pengguna tidak perlu membandingkan angka secara manual, cukup melihat baris teratas sebagai prioritas utama perbaikan.")

st.markdown("---")

# ==========================================
# XAI - SHAP
# -- E. Blok Transparansi (Explainability) --
st.subheader("🧠 Mengapa Hasilnya Demikian? (Analisis SHAP)")
st.write("Grafik *waterfall* di bawah menunjukkan kekuatan dorongan masing-masing variabel terhadap risiko prediksi.")

# 1. Ambil nama fitur asli dari DataFrame
nama_fitur = input_dari_slider.columns

# 2. Kembalikan data latih dan input yang sudah di-scale menjadi DataFrame
data_latih_scaled = pd.DataFrame(scaler.transform(data_latih), columns=nama_fitur)
input_scaled_df = pd.DataFrame(input_scaled, columns=nama_fitur)

# 3. Masukkan DataFrame ke dalam Explainer
explainer = shap.Explainer(model_ml, data_latih_scaled)
shap_values = explainer(input_scaled_df)

# 4. Visualisasikan grafik Waterfall
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)

with st.expander("💡 Penjelasan: Transparansi Model"):
    st.write("Bagian ini memecahkan stigma *black-box* (kotak hitam) pada Artificial Intelligence. Grafik memvisualisasikan ukuran balok warna untuk membeberkan argumen teknis—apakah Suhu_Server atau Beban_Jaringan yang menjadi pemicu dominan sehingga sistem memberikan rekomendasi tindakan spesifik.")