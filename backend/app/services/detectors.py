from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.schemas import Detection


@dataclass
class PredictionResult:
    detections: list[Detection]
    annotated_image: np.ndarray


class Detector(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray, confidence: float, iou: float) -> PredictionResult:
        raise NotImplementedError


class DemoDetector(Detector):
    def __init__(self, labels: list[str]) -> None:
        self.labels = [label for label in labels if label != "__background__"] or ["objeto"]

    def predict(self, image: np.ndarray, confidence: float, iou: float) -> PredictionResult:
        height, width = image.shape[:2]
        x1, y1 = int(width * 0.25), int(height * 0.25)
        x2, y2 = int(width * 0.75), int(height * 0.75)
        annotated = image.copy()
        label = self.labels[0]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (34, 197, 94), 2)
        cv2.putText(
            annotated,
            f"{label} {max(confidence, 0.50):.2f}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (34, 197, 94),
            2,
            cv2.LINE_AA,
        )
        return PredictionResult(
            detections=[
                Detection(label=label, confidence=max(confidence, 0.50), box=[x1, y1, x2, y2])
            ],
            annotated_image=annotated,
        )


class YoloDetector(Detector):
    def __init__(self, weights_path: Path, labels: list[str]) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights_path))
        self.labels = labels

    def predict(self, image: np.ndarray, confidence: float, iou: float) -> PredictionResult:
        results = self.model.predict(image, conf=confidence, iou=iou, verbose=False)
        result = results[0]
        annotated = result.plot()
        detections: list[Detection] = []

        for box in result.boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(float).tolist()
            cls_id = int(box.cls[0].detach().cpu().item())
            conf = float(box.conf[0].detach().cpu().item())
            label = result.names.get(cls_id, self.labels[cls_id] if cls_id < len(self.labels) else str(cls_id))
            detections.append(Detection(label=label, confidence=conf, box=xyxy))

        return PredictionResult(detections=detections, annotated_image=annotated)


class TorchVisionSSDLiteDetector(Detector):
    def __init__(self, weights_path: Path, labels: list[str], num_classes: int) -> None:
        import torch
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large

        self.torch = torch
        self.labels = labels
        checkpoint = torch.load(weights_path, map_location="cpu")

        model_from_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else None

        if hasattr(checkpoint, "eval") and hasattr(checkpoint, "__call__"):
            self.model = checkpoint
        elif hasattr(model_from_dict, "eval") and hasattr(model_from_dict, "__call__"):
            self.model = model_from_dict
        else:
            self.model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=num_classes)
            state_dict = self._extract_state_dict(checkpoint)
            self.model.load_state_dict(state_dict)

        self.model.eval()

    def _extract_state_dict(self, checkpoint: Any) -> dict[str, Any]:
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value
            if all(isinstance(key, str) for key in checkpoint.keys()):
                return checkpoint
        raise ValueError("Checkpoint SSDLite não contém state_dict compatível.")

    def predict(self, image: np.ndarray, confidence: float, iou: float) -> PredictionResult:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        with self.torch.no_grad():
            output = self.model([tensor])[0]

        annotated = image.copy()
        detections: list[Detection] = []
        boxes = output.get("boxes", [])
        scores = output.get("scores", [])
        labels = output.get("labels", [])

        for box, score, label_id in zip(boxes, scores, labels):
            conf = float(score.detach().cpu().item())
            if conf < confidence:
                continue
            xyxy = box.detach().cpu().numpy().astype(float).tolist()
            cls_id = int(label_id.detach().cpu().item())
            label = self.labels[cls_id] if cls_id < len(self.labels) else str(cls_id)
            x1, y1, x2, y2 = [int(value) for value in xyxy]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (59, 130, 246), 2)
            cv2.putText(
                annotated,
                f"{label} {conf:.2f}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (59, 130, 246),
                2,
                cv2.LINE_AA,
            )
            detections.append(Detection(label=label, confidence=conf, box=xyxy))

        return PredictionResult(detections=detections, annotated_image=annotated)


class OnnxDetector(Detector):
    def __init__(self, weights_path: Path, labels: list[str]) -> None:
        self.net = cv2.dnn.readNetFromONNX(str(weights_path))
        self.labels = labels

    def predict(self, image: np.ndarray, confidence: float, iou: float) -> PredictionResult:
        raise NotImplementedError(
            "O loader ONNX foi reservado para o formato exportado do Colab. Ajuste o parser de saída em detectors.py."
        )
