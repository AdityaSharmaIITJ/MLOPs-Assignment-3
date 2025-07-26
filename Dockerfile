# Use Python 3.9 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "src/predict.py"]
