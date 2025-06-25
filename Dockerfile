# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .
# Install the dependencies

RUN apt-get update \
    && apt-get -y install libpq-dev gcc

RUN python3 -m pip install -r requirements.txt --no-cache-dir



# Copy the rest of the application code into the container
COPY . .

# Specify the command to run on container start
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]