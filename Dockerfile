FROM --platform=linux/amd64 nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# -----------------------------
# Hugging Face cache locations
# -----------------------------
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf
ENV HF_HUB_CACHE=/models/hf
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1

# -----------------------------
# System deps
# -----------------------------
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip

# -----------------------------
# Python deps
# -----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# ✅ PRE-DOWNLOAD RolmOCR MODEL (CRITICAL FIX)
# -----------------------------
RUN python - <<'EOF'
from transformers import AutoProcessor, AutoModelForVision2Seq

model_id = "reducto/RolmOCR"

print("Downloading RolmOCR processor...")
AutoProcessor.from_pretrained(model_id)

print("Downloading RolmOCR model...")
AutoModelForVision2Seq.from_pretrained(model_id)

print("RolmOCR download complete")
EOF

# -----------------------------
# App
# -----------------------------
COPY handler.py .

CMD ["python", "-u", "handler.py"]
