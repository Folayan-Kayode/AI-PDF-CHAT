FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# All runtime state lives under /data, which is a volume. Create the three
# subdirectories *before* chowning: Docker initialises a new named volume from
# the image's directory, preserving ownership. Without this, the mountpoints are
# root-owned and the app cannot write its index ("unable to open database
# file"). Setting the defaults here also makes a bare `docker run` work, not
# only compose.
ENV CHROMA_DIRECTORY=/data/chroma_db \
    UPLOAD_DIRECTORY=/data/uploads \
    REGISTRY_PATH=/data/registry/registry.sqlite3

RUN useradd -m appuser \
    && mkdir -p /data/chroma_db /data/uploads /data/registry \
    && chown -R appuser:appuser /data

USER appuser

EXPOSE 8000

# --workers 1 is mandatory, not a performance choice: embedded ChromaDB plus a
# process-wide ingestion lock cannot take more than one worker. The app also
# refuses to start if WEB_CONCURRENCY > 1 without MULTI_WORKER_ACK.
# ${PORT:-8000} honours the port a managed platform injects, while compose
# (which sets no PORT) gets 8000.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
