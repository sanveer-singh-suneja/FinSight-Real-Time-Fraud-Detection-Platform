FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir setuptools==69.5.1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN groupadd -r finsight && useradd -r -g finsight finsight \
    && mkdir -p /app/data /app/models /app/artifacts /app/reports \
    && chown -R finsight:finsight /app

COPY --chown=finsight:finsight ml/ ./ml/
COPY --chown=finsight:finsight database/ ./database/
COPY --chown=finsight:finsight configs/ ./configs/

USER finsight

CMD ["python", "-m", "ml.pipeline"]
