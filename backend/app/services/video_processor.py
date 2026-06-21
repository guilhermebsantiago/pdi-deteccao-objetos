import tempfile
from pathlib import Path

import cv2
from fastapi import HTTPException, UploadFile

from app.services.detectors import Detector


async def process_video_upload(
    file: UploadFile,
    detector: Detector,
    confidence: float,
    iou: float,
    max_frames: int,
) -> tuple[str, int, int]:
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as input_file:
        input_path = input_file.name
        input_file.write(await file.read())

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise HTTPException(status_code=400, detail="Arquivo de vídeo inválido.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 24
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frames_processed = 0
    detections_total = 0

    while frames_processed < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        result = detector.predict(frame, confidence, iou)
        writer.write(result.annotated_image)
        frames_processed += 1
        detections_total += len(result.detections)

    capture.release()
    writer.release()

    return output_path, frames_processed, detections_total
