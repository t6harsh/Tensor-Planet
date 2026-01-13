FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
# libgomp1 is needed for XGBoost on Linux/Debian
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY data/ data/
# In production, we assume models are mounted or pulled from S3/MLflow
# For this demo, we copy locally generated models if they exist, or rely on volume mounts
# We'll create the directory just in case
RUN mkdir -p models

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
