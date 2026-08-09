FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py inference_pb2.py inference_pb2_grpc.py context.py repl.py ./
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "repl:app", "--host", "0.0.0.0", "--port", "8000"]
