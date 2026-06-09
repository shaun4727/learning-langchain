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

# Create an isolated virtual environment structure
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =====================================================================
# STAGE 2: HARDENED RUNTIME ENVIRONMENT
# =====================================================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime database client shared libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire standalone virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
COPY . /app

# Route system execution paths straight through the virtual environment
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Create a limited privilege account and grant explicit ownership of app + venv
RUN useradd -u 1001 -m appuser && \
    chown -R appuser:appuser /app /opt/venv
USER appuser

EXPOSE 8000

# Fire up Gunicorn managing asynchronous multi-core scaling workers
# Fire up Gunicorn with 1 worker optimized for 1-CPU environments, and push the timeout to 2 minutes
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120", "main:app"]