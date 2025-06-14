FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    git \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY nomic ./nomic
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
