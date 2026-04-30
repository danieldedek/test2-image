FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_PREFER_BINARY=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg gosu \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -u 1000 -m appuser

RUN pip install --upgrade pip setuptools wheel
RUN pip install numpy==1.26.4 pyarrow==14.0.2

COPY app/requirements.txt .

RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY . .

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "app/main.py"]
