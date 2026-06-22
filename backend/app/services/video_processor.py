import shutil
import subprocess
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

    raw_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise HTTPException(status_code=400, detail="Arquivo de vídeo inválido.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 24
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(raw_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

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

    if frames_processed == 0:
        raise HTTPException(status_code=400, detail="Nenhum frame válido encontrado no vídeo.")

    return _to_browser_mp4(raw_output_path), frames_processed, detections_total


def _to_browser_mp4(input_path: str) -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return input_path

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_path,
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        output_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not Path(output_path).exists():
        return input_path

    return output_path
