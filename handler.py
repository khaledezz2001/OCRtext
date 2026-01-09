import os
import base64
import io
import torch
import runpod
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ===============================
# HARD OFFLINE MODE (RUNTIME)
# ===============================
os.environ["HF_HOME"] = "/models/hf"
os.environ["TRANSFORMERS_CACHE"] = "/models/hf"
os.environ["HF_HUB_CACHE"] = "/models/hf"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ===============================
# Configuration
# ===============================
MODEL_PATH = "/models/hf/reducto/RolmOCR"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = None
model = None


def log(msg):
    print(f"[BOOT] {msg}", flush=True)


def decode_image(b64: str) -> Image.Image:
    image_bytes = base64.b64decode(b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((2048, 2048), Image.BICUBIC)
    return image


# ===============================
# Load model ONCE
# ===============================
def load_model():
    global processor, model

    if model is not None:
        return

    log("Loading RolmOCR processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    log("Loading RolmOCR model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True
    )

    model.eval()
    log("RolmOCR model loaded successfully")


# ===============================
# RunPod Handler
# ===============================
def handler(event):
    log("Handler called")
    load_model()

    if "image" not in event["input"]:
        return {"error": "Missing image in input"}

    image = decode_image(event["input"]["image"])
    prompt = "<image>\nRead all text in this image."

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512
        )

    text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]

    return {
        "text": text.strip(),
        "format": "markdown"
    }


# ===============================
# Preload at container start
# ===============================
log("Preloading model at startup...")
load_model()

# ===============================
# Start RunPod worker
# ===============================
runpod.serverless.start({
    "handler": handler
})
