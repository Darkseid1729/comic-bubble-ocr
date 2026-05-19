# ==========================================
# STAGE 1: Build the React Frontend
# ==========================================
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend

# Copy frontend packaging info and install dependencies
COPY frontend/package*.json ./
RUN npm ci

# Copy frontend source and compile production build
COPY frontend/ ./
RUN npm run build

# ==========================================
# STAGE 2: Run the Unified FastAPI Backend
# ==========================================
FROM python:3.11-slim AS backend-runner
WORKDIR /app

# Install system dependencies (Tesseract OCR, OpenCV prerequisites)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-jpn \
    tesseract-ocr-kor \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements first for caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend codebase
COPY api.py main.py preprocessing.py segmentation.py detection.py ocr.py utils.py evaluate.py ./

# Copy built frontend assets from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Create output folder structures
RUN mkdir -p outputs/uploads outputs/api

# Expose FastAPI server port
EXPOSE 8000

# Environment variables
ENV APP_PORT=8000
ENV APP_HOST=0.0.0.0
ENV PORT=8000

# Start Uvicorn production server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
