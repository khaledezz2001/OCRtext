# -------------------------------------------------
# Base image
# -------------------------------------------------
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# -------------------------------------------------
# Hugging Face OFFLINE & CACHE SAFETY
# -------------------------------------------------
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf
ENV HF_HUB_CACHE=/models/hf
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
ENV TOKENIZERS_PARALLELISM=false

# -------------------------------------------------
# System dependencies
# -------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip

# -------------------------------------------------
# Python dependencies
# -------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------
# ✅ FULL SNAPSHOT DOWNLOAD (CRITICAL)
# -------------------------------------------------
RUN python - <<'EOF'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="reducto/RolmOCR",
    local_dir="/models/hf/reducto/RolmOCR",
    local_dir_use_symlinks=False,
    allow_patterns=["*"]
)

print("RolmOCR fully downloaded")
EOF

# -------------------------------------------------
# (DEBUG – SAFE TO REMOVE LATER)
# -------------------------------------------------
RUN ls -lah /models/hf/reducto/RolmOCR

# -------------------------------------------------
# App code
# -------------------------------------------------
COPY handler.py .

# -------------------------------------------------
# RunPod entrypoint
# -------------------------------------------------
CMD ["python", "-u", "handler.py"]
