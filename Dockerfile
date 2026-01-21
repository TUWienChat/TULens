# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY app.py .
COPY config.py .
COPY rag_system.py .
COPY .env_docker .env

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the application
# --server.address=0.0.0.0 is crucial for Docker to expose the app externally
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]