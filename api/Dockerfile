FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt .
COPY ../model/requirements.txt model_requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r model_requirements.txt

# Copy model and API code
COPY . .
COPY ../model /app/model

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Download model if not present
RUN python -c "import os; os.makedirs('/app/model/output', exist_ok=True)"

# Expose the port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]