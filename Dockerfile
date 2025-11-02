# Python base
FROM python:3.11-slim

# System packages for OCR/PDF & Streamlit
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Faster pip & no cache
ENV PIP_NO_CACHE_DIR=1

# Workdir
WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy app
COPY . .

# Streamlit runtime env
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONUNBUFFERED=1

# Expose (Render will set $PORT at runtime; EXPOSE is optional/for docs)
EXPOSE 10000

# IMPORTANT: bind to Render's PORT (fallback 10000 for local)
CMD ["bash", "-lc", "streamlit run streamlit_app.py --server.port=${PORT:-10000} --server.address=0.0.0.0"]