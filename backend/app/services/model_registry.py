from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.schemas import ModelInfo, ModelSyncResponse
from app.services.detectors import DemoDetector, Detector, OnnxDetector, TorchVisionSSDLiteDetector, YoloDetector
from app.services.model_downloader import download_model


@dataclass(frozen=True)
class ModelDefinition:
    id: str
    name: str
    family: str
    weights_path: Path
    source_url: str
    labels: list[str]
    loader: str
    num_classes: int | None = None


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.definitions = {
            "yolo": ModelDefinition(
                id="yolo",
                name="YOLOv8n treinado no Colab",
                family="YOLO",
                weights_path=settings.resolve_path(settings.yolo_weights),
                source_url=settings.yolo_source_url,
                labels=self._split_labels(settings.yolo_labels),
                loader="yolo",
            ),
            "ssdlite": ModelDefinition(
                id="ssdlite",
                name="SSDLite320 treinado no Colab",
                family="SSDLite",
                weights_path=settings.resolve_path(settings.ssdlite_weights),
                source_url=settings.ssdlite_source_url,
                labels=self._split_labels(settings.ssdlite_labels),
                loader=settings.ssdlite_format,
                num_classes=settings.ssdlite_num_classes,
            ),
        }
        self._cache: dict[str, Detector] = {}

    def list_models(self) -> list[ModelInfo]:
        return [self._to_info(definition) for definition in self.definitions.values()]

    def get_detector(self, model_id: str) -> Detector:
        if model_id not in self.definitions:
            raise KeyError(f"Modelo '{model_id}' não cadastrado.")

        if model_id not in self._cache:
            definition = self.definitions[model_id]
            self._cache[model_id] = self._build_detector(definition)

        return self._cache[model_id]

    def sync_model(self, model_id: str) -> ModelSyncResponse:
        if model_id not in self.definitions:
            raise KeyError(f"Modelo '{model_id}' não cadastrado.")

        definition = self.definitions[model_id]
        download_model(definition.source_url, definition.weights_path)
        self._cache.pop(model_id, None)

        return ModelSyncResponse(
            id=definition.id,
            weights_path=str(definition.weights_path),
            downloaded=definition.weights_path.exists(),
            status="Modelo sincronizado com sucesso.",
        )

    def _build_detector(self, definition: ModelDefinition) -> Detector:
        if not definition.weights_path.exists():
            if self.settings.demo_mode:
                return DemoDetector(definition.labels)
            raise FileNotFoundError(f"Pesos não encontrados: {definition.weights_path}")

        if definition.loader == "yolo":
            return YoloDetector(definition.weights_path, definition.labels)
        if definition.loader in ("ssdlite_torchvision", "torchvision"):
            return TorchVisionSSDLiteDetector(
                definition.weights_path,
                definition.labels,
                definition.num_classes or len(definition.labels),
            )
        if definition.loader == "onnx":
            return OnnxDetector(definition.weights_path, definition.labels)
        if definition.loader == "torch_module":
            return TorchVisionSSDLiteDetector(
                definition.weights_path,
                definition.labels,
                definition.num_classes or len(definition.labels),
            )

        raise ValueError(f"Loader não suportado: {definition.loader}")

    def _to_info(self, definition: ModelDefinition) -> ModelInfo:
        available = definition.weights_path.exists()
        if available:
            status = "Pesos encontrados."
        elif self.settings.demo_mode:
            status = "Pesos ausentes; usando modo demonstração."
        else:
            status = "Pesos ausentes."

        return ModelInfo(
            id=definition.id,
            name=definition.name,
            family=definition.family,
            weights_path=str(definition.weights_path),
            available=available,
            source_configured=bool(definition.source_url.strip()),
            status=status,
        )

    def _split_labels(self, raw: str) -> list[str]:
        return [item.strip() for item in raw.split(",") if item.strip()]
