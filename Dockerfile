FROM python:3.11-slim

WORKDIR /app

# System deps: OpenCV runtime libs, curl (healthcheck), ffmpeg (yt-dlp trims/merges)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render injects PORT; Gradio reads it via config.py
ENV GRADIO_PORT=${PORT:-7860}
ENV GRADIO_SHARE=false

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-7860}/ || exit 1

EXPOSE 7860

CMD ["python", "app.py"]
