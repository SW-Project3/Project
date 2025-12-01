FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY web/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY web/api/chart_service.py .
COPY web/api/chart_api.py .

# Copy project modules (프로젝트 루트가 build context이므로)
COPY main /project/main
COPY data_process /project/data_process

# Set Python path
ENV PYTHONPATH=/project

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "chart_api.py"]

