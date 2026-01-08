import base64
import io
import torch
import runpod
from PIL import Image
from transformers import AutoModel

MODEL_ID = "stepfun-ai/GOT-OCR2_0"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None


def log(msg):
    print(f"[BOOT] {msg}", flush=True)


def load_model():
    global model
    if model is not None:
        return

    log("Loading GOT-OCR2_0 model (custom code)...")

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    model.eval()
    log("Model loaded successfully")


def decode_image(b64: str) -> Image.Image:
    image_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def handler(event):
    log("Handler called")
    load_model()

    image_b64 = event["input"]["image"]
    image = decode_image(image_b64)

    log("Running OCR")

    # ✅ CORRECT CALL (positional argument)
    text = model.chat(image)

    log("OCR finished")

    return {
        "text": text,
        "format": "markdown"
    }


runpod.serverless.start({
    "handler": handler
})
