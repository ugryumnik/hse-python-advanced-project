# Use official Python image as base
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install system dependencies for WeasyPrint
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libpangoft2-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libcairo2 \
    libffi-dev \
    libglib2.0-0 \
    libgirepository-1.0-1 \
    gir1.2-pango-1.0 \
    fonts-dejavu-core \
    fonts-liberation \
    libgl1 \
    libglx-mesa0 \
    libgl1-mesa-dri \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY src/ ./src/

# Run the bot app
CMD ["python", "src/main.py"]