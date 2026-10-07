FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Run as a non-root user, with a writable /data for the three state paths
# (CHROMA_DIRECTORY, UPLOAD_DIRECTORY, REGISTRY_PATH) mounted in compose.
RUN useradd -m appuser && mkdir -p /data && chown -R appuser /data

USER appuser

EXPOSE 8000

# --workers 1 is mandatory, not a performance choice: embedded ChromaDB plus a
# process-wide ingestion lock cannot take more than one worker. The app also
# refuses to start if WEB_CONCURRENCY > 1 without MULTI_WORKER_ACK.
# ${PORT:-8000} honours the port a managed platform injects, while compose
# (which sets no PORT) gets 8000.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
