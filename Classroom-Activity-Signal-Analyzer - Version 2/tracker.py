# Person detections and optional spatial tracking.

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / "Ultralytics"))

from ultralytics import YOLO


@dataclass
class Detection:
    bbox: tuple
    confidence: float
    track_id: int | None = None


class PersonTracker:
    def __init__(self, model_path="yolov8n.pt", conf=0.25, imgsz=1280, max_detections=300, iou_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.max_detections = max_detections
        self.iou_threshold = iou_threshold
        self._next_id = 0
        self._objects = {}

    def reset(self):
        self._objects.clear()
        self._next_id = 0

    def update(self, frame, tracking=False):
        detections = self._detect(frame)
        if not tracking:
            self.reset()
            return detections
        return self._match(detections)

    def _detect(self, frame):
        results = self.model(
            frame, classes=[0], conf=self.conf, imgsz=self.imgsz,
            max_det=self.max_detections, iou=self.iou_threshold, verbose=False,
        )
        height, width = frame.shape[:2]
        detections = []
        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                if not np.isfinite(coords).all() or not np.isfinite(confidence):
                    continue
                x1, y1, x2, y2 = map(int, coords)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 - x1 >= 8 and y2 - y1 >= 8:
                    detections.append(Detection((x1, y1, x2, y2), confidence))
        return sorted(detections, key=lambda d: (d.bbox[1], d.bbox[0]))

    def _match(self, detections):
        pairs = sorted(
            ((_iou(old, detection.bbox), track_id, i)
             for track_id, old in self._objects.items()
             for i, detection in enumerate(detections)),
            reverse=True,
        )
        used_tracks, used_detections = set(), set()
        for overlap, track_id, i in pairs:
            if overlap < 0.30:
                break
            if track_id in used_tracks or i in used_detections:
                continue
            detections[i].track_id = track_id
            used_tracks.add(track_id)
            used_detections.add(i)
        for detection in detections:
            if detection.track_id is None:
                detection.track_id = self._next_id
                self._next_id += 1
        self._objects = {d.track_id: d.bbox for d in detections}
        return detections


def _iou(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / max(union, 1e-6)
