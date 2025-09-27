# Dockerfile

FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# We only need the code and configs now
COPY ./src /app/src
COPY ./configs /app/configs

# --- REMOVED THIS LINE: COPY ./data/processed /app/data/processed ---

EXPOSE 8000
CMD exec uvicorn src.predict:app --host 0.0.0.0 --port ${PORT}
