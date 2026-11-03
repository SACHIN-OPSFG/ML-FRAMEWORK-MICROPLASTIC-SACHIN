FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements-tested.txt .
RUN pip install --no-cache-dir -r requirements-tested.txt

# Copy application files
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY sample_data/ ./sample_data/
COPY frontend/dist/ ./frontend/dist/
COPY yolov8n.pt .

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
