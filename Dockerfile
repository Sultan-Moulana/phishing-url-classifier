FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Patch OS-level Debian packages
RUN apt-get update && apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create non-root execution user
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

COPY requirements.txt .

# Forcefully patch the core build tools (setuptools/wheel) to clear CVEs
RUN pip install --no-cache-dir --upgrade pip setuptools>=78.1.1 wheel>=0.46.2 && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY train_model.py .

# Train and serialize model during build, clear cache, enforce ownership
RUN python train_model.py && \
    rm -rf data/ && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]