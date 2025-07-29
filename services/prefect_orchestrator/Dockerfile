# services/prefect_orchestrator/Dockerfile
FROM python:3.11-bookworm
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Optional: Prefect config
ENV PREFECT_LOGGING_LEVEL=INFO

CMD ["python", "services/prefect_orchestrator/run_flows.py"]


