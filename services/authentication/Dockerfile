FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY authenticate_api.py .

CMD ["uvicorn", "authenticate_api:app", "--host", "0.0.0.0", "--port", "8000"]