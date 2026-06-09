# =====================================================================
# STAGE 1: BUILD ENVIRONMENT & DEPENDENCY PIPELINE
# =====================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install compilation essentials needed for binary extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Generate isolated local Python dependency wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# =====================================================================
# STAGE 2: HARDENED RUNTIME ENVIRONMENT
# =====================================================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime database client shared libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy python packages compiled in the builder stage
COPY --from=builder /root/.local /root/.local
COPY . /app

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Create a limited privilege system account to run the application
RUN useradd -u 1001 -m appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Fire up Gunicorn managing Uvicorn workers for asynchronous multi-core scaling
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "main:app"]