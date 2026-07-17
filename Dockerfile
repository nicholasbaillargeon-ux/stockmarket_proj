FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so code edits don't bust the layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY analysis.py data.py app.py ./
COPY assets/ ./assets/
COPY landing/ ./landing/

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8050

# yfinance calls can be slow, hence the generous timeout.
CMD ["gunicorn", "app:server", \
     "--bind", "0.0.0.0:8050", \
     "--workers", "2", \
     "--threads", "4", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
