import base64
import io
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_ID = "stepfun-ai/GOT-OCR2_0"

# -------------------------------------------------
# Device
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# Load model once (cold start only)
# -------------------------------------------------
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

model.eval()

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def decode_base64_image(b64: str) -> Image.Image:
    image_bytes = base64.b64decode(b64)
    image = Image.open(io.BytesIO(image_bytes))
    return image.convert("RGB")


def clean_text(text: str) -> str:
    """
    Light cleanup:
    - remove empty lines
    - normalize spacing
    """
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


# -------------------------------------------------
# RunPod handler
# -------------------------------------------------
def handler(event):
    """
    Expected input:
    {
        "input": {
        "image": "<base64>",
        "language": "ru"
        }
    }
    """
    try:
        payload = event.get("input", {})

        image_b64 = payload.get("image")
        if not image_b64:
            return {"error": "Missing 'image' (base64 string)"}

        image = decode_base64_image(image_b64)

        inputs = processor(
            images=image,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=4096
            )

        text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        text = clean_text(text)

        return {
            "text": text,
            "format": "markdown"
        }

    except Exception as e:
        return {
            "error": str(e)
        }

