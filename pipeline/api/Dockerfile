FROM python:3.10-slim

WORKDIR /app

COPY requirements-ci.txt .
RUN pip install --no-cache-dir -r requirements-ci.txt
RUN pip install fastapi uvicorn

COPY ../../src ./src
COPY ../../pipeline/prediction ./pipeline/prediction
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]