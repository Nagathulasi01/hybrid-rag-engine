FROM python:3.10-slim

WORKDIR /app

# Install python dependencies
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend application
COPY frontend/app.py .

# Set environment variables
ENV API_URL=http://backend:8000

# Expose port
EXPOSE 8501

# Start Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
