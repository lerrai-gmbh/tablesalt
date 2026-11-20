# Submission contract: the container must run
#   <entrypoint> /data/images /data/predictions
# offline (no network). Bundle all model weights in the image.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pytesseract pillow numpy
COPY . /app
WORKDIR /app
ENTRYPOINT ["python", "submission_template/predict.py", "/data/images", "/data/predictions"]
