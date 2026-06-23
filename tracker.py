"""YOLO person detection with simple IoU-based tracking."""

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd()))

from ultralytics import YOLO


class PersonTracker:
    def __init__(self, model_path: str = "yolov8n.pt",
                 conf: float = 0.4, max_disappeared: int = 20):
        self.model = YOLO(model_path)
        self.conf = conf
        self.max_disappeared = max_disappeared

        self._next_id: int = 0
        self._objects: dict = {}
        self._disappeared: dict = {}

    def update(self, frame: np.ndarray) -> list:
        bboxes = self._detect(frame)
        return self._match(bboxes)

    def reset(self):
        self._next_id = 0
        self._objects.clear()
        self._disappeared.clear()

    @property
    def active_count(self) -> int:
        return len(self._objects)

    def _detect(self, frame: np.ndarray) -> list:
        results = self.model(frame, classes=[0], conf=self.conf, verbose=False)
        bboxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if (x2 - x1) > 20 and (y2 - y1) > 30:
                    bboxes.append((x1, y1, x2, y2))
        return bboxes

    def _match(self, bboxes: list) -> list:
        if not bboxes:
            for object_id in list(self._disappeared):
                self._disappeared[object_id] += 1
                if self._disappeared[object_id] > self.max_disappeared:
                    del self._objects[object_id]
                    del self._disappeared[object_id]
            return []

        if not self._objects:
            results = []
            for bbox in bboxes:
                track_id = self._register(bbox)
                results.append((track_id, bbox))
            return results

        object_ids = list(self._objects.keys())
        object_bboxes = list(self._objects.values())
        iou_mat = np.zeros((len(object_bboxes), len(bboxes)), dtype=np.float32)
        for i, old_bbox in enumerate(object_bboxes):
            for j, new_bbox in enumerate(bboxes):
                iou_mat[i, j] = _iou(old_bbox, new_bbox)

        matched_objects = set()
        matched_detections = set()
        results = []

        # Greedily match the highest-overlap pairs first.
        tmp = iou_mat.copy()
        while True:
            if tmp.size == 0 or tmp.max() < 0.15:
                break
            i, j = np.unravel_index(tmp.argmax(), tmp.shape)
            object_id = object_ids[i]
            self._objects[object_id] = bboxes[j]
            self._disappeared[object_id] = 0
            results.append((object_id, bboxes[j]))
            matched_objects.add(i)
            matched_detections.add(j)
            tmp[i, :] = -1
            tmp[:, j] = -1

        for i, object_id in enumerate(object_ids):
            if i not in matched_objects:
                self._disappeared[object_id] += 1
                if self._disappeared[object_id] > self.max_disappeared:
                    del self._objects[object_id]
                    del self._disappeared[object_id]

        for j, bbox in enumerate(bboxes):
            if j not in matched_detections:
                track_id = self._register(bbox)
                results.append((track_id, bbox))

        return results

    def _register(self, bbox: tuple) -> int:
        track_id = self._next_id
        self._objects[track_id] = bbox
        self._disappeared[track_id] = 0
        self._next_id += 1
        return track_id


def _iou(a: tuple, b: tuple) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)
