import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import random
import math
import io

# --- Konfigurasi dan Fungsi-Fungsi dari Notebook ---

def set_seeds(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)

def clean_data(excel_file):
    xls = pd.ExcelFile(excel_file)
    all_dataframes = []
    sheet_names = [name for name in xls.sheet_names if str(name).isdigit()]
    for sheet_name in sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        df['Tahun'] = int(sheet_name)
        all_dataframes.append(df)
    df_kotor = pd.concat(all_dataframes, ignore_index=True)

    df_cleaned = df_kotor.copy()
    df_cleaned.columns = df_cleaned.columns.str.strip()
    nama_mapping = {'Jakarta Raya & Tangerang': 'Jakarta Raya'}
    if 'Satuan PLN/Provinsi' in df_cleaned.columns:
        df_cleaned['Satuan PLN/Provinsi'] = df_cleaned['Satuan PLN/Provinsi'].replace(nama_mapping)
    
    sektor_cols = ['Rumah Tangga', 'Industri', 'Bisnis', 'Sosial', 'GKP', 'PJU']
    for col in sektor_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
    
    df_cleaned = df_cleaned.fillna(0)
    df_final = df_cleaned.groupby(['Tahun', 'Satuan PLN/Provinsi'])[sektor_cols].sum().reset_index()
    return df_final, sektor_cols

def create_stacked_bilstm_model(n_steps):
    model = Sequential([
        Input(shape=(n_steps, 1)),
        Bidirectional(LSTM(50, activation='relu', return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(50, activation='relu')),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

@st.cache_resource(show_spinner="Melatih model untuk semua sektor...")
def get_all_models_and_scalers(_df_cleaned, _sektor_cols):
    set_seeds()
    data_nasional = _df_cleaned.groupby('Tahun')[_sektor_cols].sum().reset_index()
    
    best_n_steps = 7
    best_split_ratio = 0.8
    train_size_final = int(len(data_nasional) * best_split_ratio)
    
    all_models, all_scalers = {}, {}
    early_stopping_val = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    for sektor in _sektor_cols:
        scaler_final = MinMaxScaler()
        scaler_final.fit(data_nasional.head(train_size_final)[[sektor]])
        full_s = scaler_final.transform(data_nasional[[sektor]])
        
        X_final, y_final = [], []
        for i in range(len(full_s) - best_n_steps):
            X_final.append(full_s[i:i+best_n_steps])
            y_final.append(full_s[i+best_n_steps])
        
        if not X_final: continue
        X_final, y_final = np.array(X_final), np.array(y_final)
        
        train_idx_final = len(data_nasional.head(train_size_final)) - best_n_steps
        if train_idx_final <= 0: continue
        
        X_train, X_test = X_final[:train_idx_final], X_final[train_idx_final:]
        y_train, y_test = y_final[:train_idx_final], y_final[train_idx_final:]
        if len(X_test) == 0: continue
        
        model_final = create_stacked_bilstm_model(best_n_steps)
        model_final.fit(X_train, y_train, epochs=500, verbose=0, validation_data=(X_test, y_test), callbacks=[early_stopping_val])
        
        all_models[sektor] = model_final
        all_scalers[sektor] = scaler_final
        
    return data_nasional, all_models, all_scalers, best_n_steps

def make_future_predictions(data_nasional, all_models, all_scalers, best_n_steps, sektor_terpilih, target_year):
    future_data = data_nasional.copy()
    latest_year = future_data['Tahun'].max()
    
    for year_to_predict in range(latest_year + 1, target_year + 1):
        new_row = {'Tahun': year_to_predict}
        for sektor in sektor_terpilih:
            if sektor in all_models:
                model, scaler = all_models[sektor], all_scalers[sektor]
                
                input_data = future_data[sektor].values[-best_n_steps:].reshape(-1, 1)
                scaled_input = scaler.transform(input_data).reshape(1, best_n_steps, 1)
                
                pred_s = model.predict(scaled_input, verbose=0)
                pred_gwh = scaler.inverse_transform(pred_s)[0, 0]
                new_row[sektor] = pred_gwh
            
        future_data = pd.concat([future_data, pd.DataFrame([new_row])], ignore_index=True)
        
    return future_data

@st.cache_data
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Prediksi')
    processed_data = output.getvalue()
    return processed_data

# --- Antarmuka Aplikasi Streamlit ---

st.set_page_config(layout="wide")
st.title("📊 Aplikasi Prediksi Konsumsi Listrik PLN")
st.write("Aplikasi ini menggunakan model Bi-LSTM untuk memprediksi konsumsi listrik per sektor berdasarkan data historis.")

# ===== BAGIAN YANG DIPERBARUI (KAMUS LABEL) =====
SEKTOR_MAPPING = {
    'Rumah Tangga': 'Rumah Tangga',
    'Industri': 'Industri',
    'Bisnis': 'Bisnis',
    'Sosial': 'Sosial',
    'GKP': 'Gedung Kantor Pemerintah',
    'PJU': 'Penerangan Jalan Umum'
}
# ===============================================

uploaded_file = st.file_uploader("Unggah file Excel data historis PLN (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        df_cleaned, sektor_cols = clean_data(uploaded_file)
        data_nasional_ori, all_models, all_scalers, best_n_steps = get_all_models_and_scalers(df_cleaned, sektor_cols)
        
        st.sidebar.header("⚙️ Opsi Prediksi")
        
        latest_year = data_nasional_ori['Tahun'].max()
        target_year = st.sidebar.number_input(
            "Masukkan Tahun Prediksi:", 
            min_value=latest_year + 1, 
            max_value=latest_year + 20, 
            value=latest_year + 1,
            step=1
        )
        
        predict_all = st.sidebar.checkbox("Prediksi Semua Sektor", value=True)
        
        # ===== BAGIAN YANG DIPERBARUI (PILIHAN MENU) =====
        sektor_terpilih_display = []
        if not predict_all:
            display_options = [SEKTOR_MAPPING[sektor] for sektor in sektor_cols]
            sektor_terpilih_display = st.sidebar.multiselect(
                "Pilih Sektor untuk Prediksi:",
                options=display_options,
                default=display_options[0]
            )
        # ===============================================
            
        if st.sidebar.button("🚀 Buat Prediksi", use_container_width=True):
            
            # ===== BAGIAN YANG DIPERBARUI (TERJEMAHKAN PILIHAN) =====
            if predict_all:
                sektor_terpilih = sektor_cols
            else:
                # Buat kamus terbalik untuk menerjemahkan pilihan pengguna
                REVERSE_SEKTOR_MAPPING = {v: k for k, v in SEKTOR_MAPPING.items()}
                sektor_terpilih = [REVERSE_SEKTOR_MAPPING[display] for display in sektor_terpilih_display]
            # =======================================================

            if not sektor_terpilih:
                st.warning("Silakan pilih minimal satu sektor.")
            else:
                with st.spinner(f"Membuat prediksi hingga tahun {target_year}..."):
                    future_predictions_df = make_future_predictions(data_nasional_ori, all_models, all_scalers, best_n_steps, sektor_terpilih, target_year)
                
                st.header(f"📈 Hasil Prediksi Tahun {target_year}")

                n_sektor = len(sektor_terpilih)
                n_cols = 2 if n_sektor > 1 else 1
                n_rows = math.ceil(n_sektor / n_cols)
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6 * n_rows), squeeze=False)
                axes = axes.flatten()
                
                for i, sektor in enumerate(sektor_terpilih):
                    ax = axes[i]
                    display_name = SEKTOR_MAPPING.get(sektor, sektor) # Dapatkan nama tampilan
                    ax.plot(data_nasional_ori['Tahun'], data_nasional_ori[sektor], marker='o', linestyle='-', label='Data Aktual')
                    pred_years = future_predictions_df[future_predictions_df['Tahun'] > latest_year]
                    ax.plot(pred_years['Tahun'], pred_years[sektor], 'o--', color='orange', label='Prediksi Masa Depan')
                    target_pred_val = future_predictions_df[future_predictions_df['Tahun'] == target_year][sektor].values[0]
                    ax.plot(target_year, target_pred_val, 'ro', markersize=10, label=f'Prediksi {target_year}')
                    ax.set_title(f"Tren dan Prediksi Sektor {display_name}") # Gunakan nama tampilan di judul
                    ax.legend()
                    ax.ticklabel_format(style='plain', axis='y')
                    ax.set_ylabel("Konsumsi (GWh)")
                    ax.set_xlabel("Tahun")
                
                for i in range(n_sektor, len(axes)):
                    fig.delaxes(axes[i])
                
                plt.tight_layout()
                st.pyplot(fig)

                st.subheader("Tabel Prediksi Nasional")
                
                target_year_data = future_predictions_df.loc[future_predictions_df['Tahun'] == target_year]
                final_predictions = target_year_data[sektor_terpilih].T.reset_index()
                final_predictions.columns = ['Sektor', f'Prediksi {target_year} (GWh)']
                # Ganti nama sektor di tabel hasil
                final_predictions['Sektor'] = final_predictions['Sektor'].map(SEKTOR_MAPPING)

                total_prediksi = final_predictions[f'Prediksi {target_year} (GWh)'].sum()
                
                st.dataframe(final_predictions.style.format({f'Prediksi {target_year} (GWh)': '{:,.2f}'}), use_container_width=True)
                st.metric(label=f"Total Prediksi untuk Sektor Terpilih ({target_year})", value=f"{total_prediksi:,.2f} GWh")

                with st.expander("Lihat Prediksi Detail per Provinsi dan Unduh Hasil"):
                    all_province_results = []
                    provinsi_terakhir = df_cleaned[df_cleaned['Tahun'] == latest_year]['Satuan PLN/Provinsi'].unique()
                    
                    for provinsi in provinsi_terakhir:
                        prov_result = {'Provinsi': provinsi}
                        for sektor in sektor_terpilih:
                            input_data_prov = df_cleaned[df_cleaned['Satuan PLN/Provinsi'] == provinsi].sort_values('Tahun')
                            model, scaler = all_models[sektor], all_scalers[sektor]
                            if len(input_data_prov) < best_n_steps:
                                pred_val = np.nan
                            else:
                                future_prov_data = input_data_prov.copy()
                                for year_to_predict_prov in range(latest_year + 1, target_year + 1):
                                    input_sequence = future_prov_data[sektor].values[-best_n_steps:].reshape(-1, 1)
                                    scaled_input = scaler.transform(input_sequence).reshape(1, best_n_steps, 1)
                                    pred_s = model.predict(scaled_input, verbose=0)
                                    pred_gwh = scaler.inverse_transform(pred_s)[0, 0]
                                    new_prov_row = {'Tahun': year_to_predict_prov, sektor: pred_gwh, 'Satuan PLN/Provinsi': provinsi}
                                    future_prov_data = pd.concat([future_prov_data, pd.DataFrame([new_prov_row])], ignore_index=True)
                                pred_val = future_prov_data[future_prov_data['Tahun'] == target_year][sektor].values[0]

                            prov_result[sektor] = pred_val
                        all_province_results.append(prov_result)
                    
                    df_summary_provinsi = pd.DataFrame(all_province_results).dropna()
                    
                    # Ubah nama kolom sebelum ditampilkan dan diunduh
                    df_summary_provinsi_display = df_summary_provinsi.rename(columns=SEKTOR_MAPPING)
                    
                    st.dataframe(df_summary_provinsi_display.style.format(formatter="{:,.2f}", subset=list(SEKTOR_MAPPING.values())), use_container_width=True)

                    df_xlsx = to_excel(df_summary_provinsi_display)
                    st.download_button(
                        label="📥 Unduh Tabel Prediksi per Provinsi (.xlsx)",
                        data=df_xlsx,
                        file_name=f'prediksi_provinsi_{target_year}.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
        st.exception(e) # Menampilkan detail error untuk debugging