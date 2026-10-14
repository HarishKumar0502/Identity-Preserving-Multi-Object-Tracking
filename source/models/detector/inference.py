"""YOLO11n person detector (``yolo11n.pt`` in this folder, COCO class 0 = person).

Detection is independent of identity assignment."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from source.config.settings import DETECTOR, DetectorConfig
from source.utils.device_utils import resolve_device

log = logging.getLogger("ps2.detector")


class DetectorError(RuntimeError):
    pass


@dataclass
class Detections:
    """Per-frame detections. ``boxes`` are (N, 4) x1, y1, x2, y2 in pixels."""

    boxes: np.ndarray
    scores: np.ndarray
    frame_index: int

    def __len__(self) -> int:
        return len(self.boxes)

    @classmethod
    def empty(cls, frame_index: int) -> "Detections":
        return cls(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), frame_index)


class PersonDetector(ABC):
    name = "base"

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_index: int) -> Detections:
        ...


class YoloPersonDetector(PersonDetector):
    """Ultralytics YOLO (.pt, .onnx or .engine weights) restricted to class 'person'."""

    name = "ultralytics-yolo"

    def __init__(self, cfg: DetectorConfig = DETECTOR):
        self.cfg = cfg
        weights = Path(cfg.weights)
        if not weights.is_file():
            raise DetectorError(
                f"Detector weights not found ({weights.name}). Place a COCO person-capable YOLO "
                f"model at source/models/detector/yolo11n.pt or set PS2_DETECTOR_WEIGHTS.")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorError("ultralytics is not installed.") from exc
        self.model = YOLO(str(weights), task="detect")
        names = getattr(self.model, "names", None) or {}
        if names and names.get(cfg.person_class_id) != "person":
            raise DetectorError(f"Detector class {cfg.person_class_id} is "
                                f"'{names.get(cfg.person_class_id)}', not 'person'.")
        self.device = resolve_device(cfg.device)
        self.weights_name = weights.name

    def detect(self, frame: np.ndarray, frame_index: int) -> Detections:
        kwargs = {"half": True} if self.cfg.half else {}
        res = self.model.predict(frame, conf=self.cfg.min_confidence, iou=self.cfg.iou_nms,
                                 classes=[self.cfg.person_class_id], imgsz=self.cfg.imgsz,
                                 device=self.device, verbose=False, **kwargs)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return Detections.empty(frame_index)
        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = res.boxes.conf.cpu().numpy().astype(np.float32)
        h, w = frame.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
        keep = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
        return Detections(boxes[keep], scores[keep], frame_index)


def create_detector(cfg: Optional[DetectorConfig] = None) -> PersonDetector:
    return YoloPersonDetector(cfg or DETECTOR)
