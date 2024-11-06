# Use Python as the base image
FROM python:3.8-slim

# Set the working directory to /app/src
WORKDIR /app/src

# Install dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Copy the requirements file to /app, then install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the entire project directory to /app
COPY . /app

# Expose port 8501 for the Streamlit app
EXPOSE 8501

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]


