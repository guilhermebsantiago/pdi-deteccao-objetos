import base64

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile


async def read_image_upload(file: UploadFile) -> np.ndarray:
    payload = await file.read()
    data = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Arquivo de imagem inválido.")
    return image


def image_to_data_url(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise HTTPException(status_code=500, detail="Falha ao codificar imagem anotada.")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def file_to_data_url(path: str, content_type: str) -> str:
    with open(path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("ascii")
    return f"data:{content_type};base64,{encoded}"
