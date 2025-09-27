# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local code to the container
COPY ./src /app/src
COPY ./configs /app/configs
COPY ./data/processed /app/data/processed

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run predict.py when the container launches
CMD exec uvicorn src.predict:app --host 0.0.0.0 --port ${PORT}