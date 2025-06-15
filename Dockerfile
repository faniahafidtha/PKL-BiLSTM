# Gunakan image Python 3.10 yang ringan
FROM python:3.10-slim

# Atur direktori kerja
WORKDIR /app

# Salin semua file ke dalam container
COPY . .

# Install pip dan semua dependensi
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Buka port 8000 (port default Streamlit)
EXPOSE 8000

# Jalankan Streamlit saat container aktif
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
