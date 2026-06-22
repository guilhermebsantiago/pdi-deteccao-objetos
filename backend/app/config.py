from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    app_name: str = "Detector de Objetos PDI"
    app_env: str = "development"
    cors_origins: str = "http://localhost:5173"

    yolo_weights: str = "storage/models/yolo/best.pt"
    yolo_source_url: str = ""
    yolo_source_file: str = ""
    yolo_labels: str = "caneta,maca"

    yolo_caneta_weights: str = "storage/models/yolo/caneta.pt"
    yolo_caneta_source_url: str = ""
    yolo_caneta_source_file: str = "yolo_caneta_best.pt"
    yolo_caneta_labels: str = "caneta"

    yolo_maca_weights: str = "storage/models/yolo/maca.pt"
    yolo_maca_source_url: str = ""
    yolo_maca_source_file: str = "yolo_maca_best.pt"
    yolo_maca_labels: str = "maca"

    ssdlite_weights: str = "storage/models/ssdlite/model.pth"
    ssdlite_source_url: str = ""
    ssdlite_source_file: str = ""
    ssdlite_format: str = "ssdlite_torchvision"
    ssdlite_labels: str = "__background__,caneta,maca"
    ssdlite_num_classes: int = 3

    ssdlite_caneta_weights: str = "storage/models/ssdlite/caneta.pth"
    ssdlite_caneta_source_url: str = ""
    ssdlite_caneta_source_file: str = "ssdlite_caneta_best.pth"
    ssdlite_caneta_labels: str = "__background__,caneta"

    ssdlite_maca_weights: str = "storage/models/ssdlite/maca.pth"
    ssdlite_maca_source_url: str = ""
    ssdlite_maca_source_file: str = "ssdlite_maca_best.pth"
    ssdlite_maca_labels: str = "__background__,maca"

    max_video_frames: int = 450
    demo_mode: bool = True

    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8")

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return BACKEND_DIR / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
