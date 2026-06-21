from pydantic import BaseModel


class ModelInfo(BaseModel):
    id: str
    name: str
    family: str
    weights_path: str
    available: bool
    status: str


class Detection(BaseModel):
    label: str
    confidence: float
    box: list[float]


class ImageDetectionResponse(BaseModel):
    model_id: str
    filename: str
    width: int
    height: int
    detections: list[Detection]
    annotated_image: str


class VideoDetectionResponse(BaseModel):
    model_id: str
    filename: str
    frames_processed: int
    detections_total: int
    annotated_video: str
    content_type: str = "video/mp4"


class HealthResponse(BaseModel):
    status: str
    app: str
