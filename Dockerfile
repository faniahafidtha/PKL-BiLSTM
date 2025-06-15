# Gunakan Python 3.10 image resmi
FROM python:3.10-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Salin semua file proyek ke dalam container
COPY . .
# Gunakan Python 3.10 image resmi
FROM python:3.10-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Salin semua file proyek ke dalam container
COPY . .

# Install pip dan semua dependensi
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Buka port 8000 (yang akan digunakan Streamlit)
EXPOSE 8000

# Perintah untuk menjalankan Streamlit saat container aktif
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]

# Install pip dan semua dependensi
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Buka port 8000 (yang akan digunakan Streamlit)
EXPOSE 8000

# Perintah untuk menjalankan Streamlit saat container aktif
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
