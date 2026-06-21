from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Detector de Objetos PDI"
    app_env: str = "development"
    cors_origins: str = "http://localhost:5173"

    yolo_weights: str = "storage/models/yolo/best.pt"
    yolo_source_url: str = ""
    yolo_labels: str = "caneta,maca"

    ssdlite_weights: str = "storage/models/ssdlite/model.pth"
    ssdlite_source_url: str = ""
    ssdlite_format: str = "ssdlite_torchvision"
    ssdlite_labels: str = "__background__,caneta,maca"
    ssdlite_num_classes: int = 3

    max_video_frames: int = 450
    demo_mode: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return Path.cwd() / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
