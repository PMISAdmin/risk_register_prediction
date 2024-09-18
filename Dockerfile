# Gunakan image dasar Python
FROM python:3.10-slim

# Install dependencies untuk psycopg2 dan spaCy
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory di dalam container
WORKDIR /app

# Salin file requirements.txt ke dalam container
COPY requirements.txt .

# Install dependencies dari requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Instal spaCy dan model bahasa Inggris
RUN pip install spacy \
    && python -m spacy download en_core_web_sm

# Salin semua file dari direktori lokal ke dalam container
COPY . .

# Tentukan perintah untuk menjalankan aplikasi
CMD ["python", "main.py"]
