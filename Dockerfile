FROM python:3.11-slim

WORKDIR /app

COPY deliops_fastapi_rag/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY deliops_fastapi_rag/ .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
