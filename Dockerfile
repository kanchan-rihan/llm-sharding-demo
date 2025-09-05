# Dockerfile
FROM python:3.11-slim

# small apt deps required by some HF tokenizers / torch
RUN apt-get update && apt-get install -y git build-essential curl && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# small apt deps required by some HF tokenizers / torch
RUN apt-get update && apt-get install -y git build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt

COPY server.py .

EXPOSE 5000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5000", "--log-level", "info"]