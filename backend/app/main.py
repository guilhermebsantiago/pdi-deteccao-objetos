from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import HealthResponse, ImageDetectionResponse, ModelInfo, ModelSyncResponse, VideoDetectionResponse
from app.services.image_codec import file_to_data_url, image_to_data_url, read_image_upload
from app.services.model_registry import ModelRegistry
from app.services.video_processor import process_video_upload

settings = get_settings()
registry = ModelRegistry(settings)

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", app=settings.app_name)


@app.get("/api/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    return registry.list_models()


@app.post("/api/models/{model_id}/sync", response_model=ModelSyncResponse)
def sync_model(model_id: str) -> ModelSyncResponse:
    try:
        return registry.sync_model(model_id)
    except (KeyError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/detect/image", response_model=ImageDetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    model_id: str = Form("yolo"),
    confidence: float = Form(0.35),
    iou: float = Form(0.45),
) -> ImageDetectionResponse:
    try:
        detector = registry.get_detector(model_id)
    except (KeyError, FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    image = await read_image_upload(file)
    result = detector.predict(image, confidence, iou)
    height, width = image.shape[:2]

    return ImageDetectionResponse(
        model_id=model_id,
        filename=file.filename or "imagem",
        width=width,
        height=height,
        detections=result.detections,
        annotated_image=image_to_data_url(result.annotated_image),
    )


@app.post("/api/detect/frame", response_model=ImageDetectionResponse)
async def detect_frame(
    file: UploadFile = File(...),
    model_id: str = Form("yolo"),
    confidence: float = Form(0.35),
    iou: float = Form(0.45),
) -> ImageDetectionResponse:
    return await detect_image(file=file, model_id=model_id, confidence=confidence, iou=iou)


@app.post("/api/detect/video", response_model=VideoDetectionResponse)
async def detect_video(
    file: UploadFile = File(...),
    model_id: str = Form("yolo"),
    confidence: float = Form(0.35),
    iou: float = Form(0.45),
) -> VideoDetectionResponse:
    try:
        detector = registry.get_detector(model_id)
    except (KeyError, FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    output_path, frames_processed, detections_total = await process_video_upload(
        file=file,
        detector=detector,
        confidence=confidence,
        iou=iou,
        max_frames=settings.max_video_frames,
    )

    return VideoDetectionResponse(
        model_id=model_id,
        filename=file.filename or "video",
        frames_processed=frames_processed,
        detections_total=detections_total,
        annotated_video=file_to_data_url(output_path, "video/mp4"),
    )
