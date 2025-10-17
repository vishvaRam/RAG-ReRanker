FROM python:3.11.14-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y gunicorn && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

ENV RERANKER_PORT=5656
ENV BATCH_SIZE=32
ENV GUNICORN_WORKERS=2

CMD ["gunicorn", "api:app", "-b", "0.0.0.0:${RERANKER_PORT}", "-w", "${GUNICORN_WORKERS}","-k","uvicorn.workers.UvicornWorker","--timeout","120"]