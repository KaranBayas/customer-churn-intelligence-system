# Use a small official Python runtime image
FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer outputs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first to leverage Docker caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project into the container
COPY . ./

# Expose common ports for Streamlit and FastAPI
EXPOSE 8501
EXPOSE 8000

# Default to the Streamlit dashboard.
# Override the command at runtime to launch the API instead if needed.
CMD ["streamlit", "run", "dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
