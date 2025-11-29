# Python Flask Backend
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY web/backend-flask/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY web/backend-flask/app.py .
COPY main ./main
COPY data_process ./data_process

# Create data directory
RUN mkdir -p /app/data/processed

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]

