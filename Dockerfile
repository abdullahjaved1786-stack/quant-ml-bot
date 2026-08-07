FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 TZ=UTC
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends tzdata ca-certificates && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && pip install --no-cache-dir joblib
COPY core/ /app/core/
COPY main.py /app/main.py
COPY quickstart.py /app/quickstart.py
RUN mkdir -p /app/data /app/data/models
VOLUME ["/app/data"]
ENV GOOGLE_SHEET_NAME="" GOOGLE_SHEET_CREDS=""
HEALTHCHECK --interval=5m --timeout=10s --start-period=60s --retries=3 CMD test -d /app/data || exit 1
CMD ["python","-u","main.py","--source","ccxt","--symbol","BTC/USDT","--timeframe","5m","--poll","300"]
