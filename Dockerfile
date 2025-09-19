# Multi-stage production Dockerfile
FROM python:3.9-slim as base

# Security and optimization
RUN groupadd -r aptuser && useradd -r -g aptuser aptuser
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Optimized dependency installation
COPY requirements_interface.txt /tmp/
    # Removed separate torch install; all dependencies installed via requirements_interface.txt
RUN pip install --no-cache-dir --default-timeout=600 -r /tmp/requirements_interface.txt

# Production stage
FROM base as production
WORKDIR /app

# Copy application code with proper ownership
COPY --chown=aptuser:aptuser *.py /app/
COPY --chown=aptuser:aptuser best_cysecbert_max_performance.pt /app/

# Create logs directory
RUN mkdir -p /app/logs /app/.cache/huggingface && chown -R aptuser:aptuser /app/logs /app/.cache

# Switch to non-root user
USER aptuser

# Set environment variables
ENV MODEL_PATH=/app/best_cysecbert_max_performance.pt
ENV PYTHONPATH=/app

# Set HuggingFace and Transformers cache directories to a writable location
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Integrated health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Use uvicorn as specified in thesis
CMD ["uvicorn", "apt_classification_api:app", "--host", "0.0.0.0", "--port", "8000"]