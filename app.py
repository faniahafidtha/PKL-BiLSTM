import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense


# --- Mengatur konfigurasi halaman Streamlit (judul dan ikon) ---
st.set_page_config(
    page_title="Prediksi Konsumsi Listrik",  # Judul halaman di tab browser
    page_icon="⚡",  # Ikon halaman (opsional, bisa menggunakan emoji atau path ke gambar)
    layout="centered",  # Opsi layout (default: "centered" atau "wide")
)

# --- Styling CSS untuk warna background dan sidebar ---
st.markdown("""
    <style>
    body {
        background-color: #ffffff;  /* Warna putih untuk halaman utama */
    }
    .sidebar .sidebar-content {
        background-color: #003366;  /* Biru dongker untuk sidebar */
    }
    .sidebar .sidebar-header {
        font-size: 1.5em;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


# --- Membaca dan memproses data ---
file_path = 'PLN 15-24.xlsx'  # Ganti dengan path file yang sesuai
years = range(2014, 2025)

# Container untuk semua data
all_data = []

for year in years:
    # Membaca sheet
    df = pd.read_excel(file_path, sheet_name=str(year))

    # Bersihkan nama kolom
    df.columns = df.columns.str.strip()

    # Pastikan semua kolom selain 'Satuan PLN/Provinsi' adalah numerik
    for col in df.columns:
        if col != 'Satuan PLN/Provinsi':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Isi NaN dengan 0
    df = df.fillna(0)

    # Tambahkan kolom 'Tahun'
    df['Tahun'] = year
    all_data.append(df)

# Gabungkan semua tahun
combined_df = pd.concat(all_data, ignore_index=True)

# Grupkan data berdasarkan 'Tahun'
df_national = combined_df.groupby('Tahun')[[ 
    'Rumah Tangga', 'Industri', 'Bisnis', 'Sosial', 'GKP', 'PJU'
]].sum().reset_index()

# Menampilkan data sektor
sectors = ['Rumah Tangga', 'Industri', 'Bisnis', 'Sosial', 'GKP', 'PJU']

# Pemetaan sektor (singkatan ke nama lengkap)
sector_mapping = {
    'Rumah Tangga': 'Rumah Tangga',
    'Industri': 'Industri',
    'Bisnis': 'Bisnis',
    'Sosial': 'Sosial',
    'GKP': 'Gedung Kantor Pemerintah',
    'PJU': 'Penerangan Jalan Umum'
}

# --- Fungsi untuk membuat urutan data ---
def create_sequences(data, n_steps=3):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

# --- Fungsi untuk format ribuan dan desimal (dengan koma sebagai pemisah desimal) ---
def format_thousands_and_decimal_vectorized(arr):
    return np.array([f"{x:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".") for x in arr])

# --- Tampilan Streamlit ---
st.title("Prediksi Konsumsi Listrik per Sektor (BiLSTM)")

# Pilih sektor dari sidebar
sector = st.sidebar.selectbox('Pilih Sektor', [sector_mapping[s] for s in sectors])

# Menampilkan nama sektor yang lengkap menggunakan pemetaan
full_sector_name = sector

# Pilih tampilan antara Tabel atau Grafik
display_option = st.radio("Pilih Tampilan", ("Tabel", "Grafik"))

# Tampilkan data sektor yang dipilih, termasuk kolom 'Satuan PLN/Provinsi'
# Mengambil sektor yang sesuai dengan nama lengkap
sector_key = list(sector_mapping.keys())[list(sector_mapping.values()).index(full_sector_name)]
sector_data = combined_df[['Tahun', 'Satuan PLN/Provinsi', sector_key]]

# --- Jika memilih Tabel ---
if display_option == "Tabel":
    st.write(f"Data Sektor: {full_sector_name}")
    
    # Format nilai untuk ribuan dan desimal
    sector_data[sector_key] = sector_data[sector_key].apply(lambda x: format_thousands_and_decimal_vectorized(np.array([x]))[0])
    
    # Tampilkan tabel dengan keterangan "GWh"
    sector_data = sector_data.rename(columns={sector_key: f'{full_sector_name} (GWh)'})
    st.dataframe(sector_data)

# --- Jika memilih Grafik ---
else:
    # --- Model BiLSTM dan Prediksi ---
    n_steps = 3
    n_future = 6  # Prediksi 2025–2030
    future_years = list(range(2025, 2025 + n_future))

    predictions_bilstm_2030 = {}
    evaluations_bilstm_2030 = {}

    # --- Proses prediksi untuk sektor yang dipilih ---
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_national[[sector_key]])

    # Persiapkan urutan data
    X, y = create_sequences(scaled_data, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Definisikan dan latih model BiLSTM
    model = Sequential([
        Bidirectional(LSTM(64, activation='relu'), input_shape=(n_steps, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)

    # Prediksi untuk data historis
    predicted_scaled = model.predict(X, verbose=0)
    predicted = scaler.inverse_transform(predicted_scaled).flatten()

    # Data aktual (ground truth)
    actual = scaler.inverse_transform(y.reshape(-1, 1)).flatten()

    # Prediksi untuk 2025–2030 (nilai masa depan)
    future_preds = []
    last_input = scaled_data[-n_steps:].reshape((1, n_steps, 1))

    for _ in range(n_future):
        future_scaled = model.predict(last_input, verbose=0)
        future_actual = scaler.inverse_transform(future_scaled)[0, 0]
        future_preds.append(future_actual)

        future_scaled_reshaped = future_scaled.reshape(1, 1, 1)
        last_input = np.concatenate((last_input[:, 1:, :], future_scaled_reshaped), axis=1)

    # Simpan prediksi
    predictions_bilstm_2030[sector_key] = {
        'actual': actual,
        'predicted': predicted,
        'future_years': future_years,
        'future_preds': future_preds
    }

    # Evaluasi dengan kesalahan terukur
    min_val, max_val = df_national[sector_key].min(), df_national[sector_key].max()
    range_val = max_val - min_val if max_val != min_val else 1

    mae = mean_absolute_error(actual, predicted) / range_val
    rmse = np.sqrt(mean_squared_error(actual, predicted)) / range_val
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    evaluations_bilstm_2030[sector_key] = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape
    }

    # --- Tampilan Evaluasi (MAE, RMSE, MAPE) ---
    eval_df = pd.DataFrame(evaluations_bilstm_2030).T
    eval_df = eval_df.rename(index={sector_key: full_sector_name})  # Menambah nama sektor lengkap
    st.subheader("Evaluasi Model (BiLSTM)")

    # Menengahkan teks di tabel evaluasi
    st.write(eval_df.style.set_properties(**{'text-align': 'center'}))  # Menengahkan teks di kolom tabel

    # --- Visualisasi Grafik ---
    st.subheader(f"Grafik Prediksi Konsumsi Listrik Sektor: {full_sector_name}")

    plt.figure(figsize=(10, 6))

    actual_years = df_national['Tahun'].values[n_steps:]
    future_years = predictions_bilstm_2030[sector_key]['future_years']

    actual = predictions_bilstm_2030[sector_key]['actual']
    predicted = predictions_bilstm_2030[sector_key]['predicted']
    future_preds = predictions_bilstm_2030[sector_key]['future_preds']

    # Plot data aktual
    plt.plot(actual_years, actual, label='Aktual', color='blue', marker='o')

    # Plot prediksi model pada data historis
    plt.plot(actual_years, predicted, label='Prediksi (hist)', color='orange', marker='x')

    # Plot prediksi masa depan (2025–2030)
    plt.plot(future_years, future_preds, label='Prediksi 2025–2030', color='red', marker='s')

    # Garis vertikal untuk menandai 2024
    plt.axvline(2024, linestyle='--', color='gray', label='Tahun 2024')

    plt.title(f"{full_sector_name}")
    plt.xlabel("Tahun")
    plt.ylabel("Nilai Konsumsi")
    plt.grid(True)
    plt.legend()

    # Tampilkan grafik
    st.pyplot(plt)

    # Tampilkan tabel untuk data aktual dan prediksi
    result_df = pd.DataFrame({
        'Tahun': actual_years.astype(int),
        'Aktual (Gwh)': format_thousands_and_decimal_vectorized(actual),
        'Prediksi (Gwh)': format_thousands_and_decimal_vectorized(predicted)
    })

    st.subheader(f"Tabel Prediksi Konsumsi Listrik Sektor: {full_sector_name}")
    st.write(result_df)

    # Tampilkan tabel untuk data aktual dan prediksi masa depan
    future_result_df = pd.DataFrame({
        'Tahun': np.array(future_years).astype(int),
        'Prediksi Masa Depan (Gwh)': format_thousands_and_decimal_vectorized(future_preds)
    })


    st.subheader(f"Tabel Prediksi Masa Depan (2025-2030) untuk Sektor: {full_sector_name} (Gwh)")
    st.write(future_result_df)
