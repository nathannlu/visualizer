import base64
from pathlib import Path
import numpy as np
import math
import os
import io
import json
from torchvision import transforms
from torch import autocast

from PIL import Image


def cv2_to_b64(cv2_img):
    """
    Convert a cv2 image to a base64 string
    """
    img = Image.fromarray(cv2_img)

    # Save the image to a bytes buffer
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")  # PNG format for Base64 encoding

    buffer.seek(0)

    # Encode the bytes buffer into a Base64 string
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_base64



def encode_pil_to_base64(pil_image: Image.Image, format="PNG") -> str:
    buffered = io.BytesIO()
    pil_image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def decode_base64_to_pil(base64_str: str):
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_bytes))
    return image
