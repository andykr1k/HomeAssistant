FROM python:3.10-slim

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "src/main.py"]
