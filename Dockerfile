# Use official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files to container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command to run your app
CMD ["python", "main.py"]
