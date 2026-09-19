"""Encode emulator frames for the browser."""

import base64
import io

from PIL import Image


def encode_frame(frame) -> str:
    buffer = io.BytesIO()
    Image.fromarray(frame).convert("RGB").save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def encode_thumb(frame, width: int = 192, quality: int = 52) -> str:
    image = Image.fromarray(frame).convert("RGB")
    if image.width > width:
        height = max(1, round(image.height * width / image.width))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
