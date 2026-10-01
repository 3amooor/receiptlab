FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY receiptlab ./receiptlab
COPY models ./models
COPY reports ./reports
COPY frontend/public/samples ./frontend/public/samples
COPY frontend/dist ./frontend/dist
RUN useradd --create-home receiptlab && mkdir -p /app/data && chown -R receiptlab:receiptlab /app
USER receiptlab
EXPOSE 8010
CMD ["uvicorn", "receiptlab.api:app", "--host", "0.0.0.0", "--port", "8010"]
