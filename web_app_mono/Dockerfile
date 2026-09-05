# PFMS web app container.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and the trained model artifact.
COPY . .

# If the model artifact isn't committed, train it at build time.
# (Requires the training data to be present in the build context.)
RUN test -f model/pfms_pipeline.joblib || python train.py || true

EXPOSE 8000

# gunicorn: 2 workers is a sane default for small instances.
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT} --workers 2 --timeout 120"]
