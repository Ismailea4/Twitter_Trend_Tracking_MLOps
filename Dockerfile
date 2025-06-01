# filepath: Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements-ci.txt .
RUN pip install --no-cache-dir -r requirements-ci.txt
COPY . .
CMD ["python", "pipeline_all.py"]