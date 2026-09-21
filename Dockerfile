FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models/huggingface

WORKDIR /app

# Kokoro uses espeak-ng as a fallback for English pronunciation.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        espeak-ng \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Explicit CPU-only PyTorch.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src

EXPOSE 8092

# One worker deliberately:
# otherwise every worker would load its own Kokoro model.
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8092", "--workers", "1"]