import base64
import io
import torch
import runpod
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# ===============================
# Configuration
# ===============================
MODEL_ID = "reducto/RolmOCR"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = None
model = None


# ===============================
# Helpers
# ===============================
def log(msg):
    print(f"[BOOT] {msg}", flush=True)


def decode_image(b64_string: str) -> Image.Image:
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


# ===============================
# Model Loader (runs once)
# ===============================
def load_model():
    global processor, model

    if model is not None and processor is not None:
        return

    log("Loading RolmOCR processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        use_fast=True
    )

    log("Loading RolmOCR model...")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto"
    )

    model.eval()
    log("RolmOCR model loaded")


# ===============================
# RunPod Handler
# ===============================
def handler(event):
    log("Handler called")
    load_model()

    if "image" not in event["input"]:
        return {"error": "Missing 'image' field in input"}

    image = decode_image(event["input"]["image"])

    log("Running OCR")

    prompt = "Read all text in this image."

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048
        )

    text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]

    log("OCR finished")

    return {
        "text": text.strip(),
        "format": "markdown"
    }


# ===============================
# Start Serverless Worker
# ===============================
runpod.serverless.start({
    "handler": handler
})
