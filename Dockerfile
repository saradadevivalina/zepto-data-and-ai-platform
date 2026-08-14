# 1. Base Image: Lightweight Python 3.10
FROM python:3.10-slim

# 2. Set Working Directory inside the container
WORKDIR /app

# 3. Set Environment Variables
# Ensures Python output is sent directly to terminal without buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MOCK_LLM=1

# 4. Install essential OS-level build dependencies (needed for packages like ChromaDB/pydantic)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy and install Python dependencies FIRST (caches Docker layer for faster builds)
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy application code and policy documents into container
COPY docs /app/docs
COPY app /app/app

# 7. Expose the port FastAPI runs on
EXPOSE 7860

# 8. Entrypoint Command to launch Uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]