# Use official Python image as base
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY src/ ./src/

COPY .env ./

# Set environment variables (if needed)
# ENV ...

# Run the bot app
CMD ["python", "src/main.py"]
