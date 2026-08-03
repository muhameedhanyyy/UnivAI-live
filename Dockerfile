FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml requirements-ci.txt ./
RUN pip install --no-cache-dir '.[integrated]'
COPY . .
ENV PYTHONUNBUFFERED=1 UNIVAI_MODE=standalone
HEALTHCHECK --interval=30s --timeout=5s CMD ["python", "-c", "from health import health_payload; assert health_payload()['live']"]
CMD ["python", "simulate.py", "run"]
