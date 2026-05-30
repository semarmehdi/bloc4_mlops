FROM python:3.11-slim

WORKDIR /app

# Logs en temps réel + pas de .pyc
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY etl.py .
COPY utils/ ./utils/

CMD ["python", "etl.py"]