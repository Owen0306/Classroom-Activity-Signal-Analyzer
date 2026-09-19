# Independent pose observations with explicit missing evidence.

from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np

from behavior import UNKNOWN, SIGNAL_META, RuleConfig, SignalState, classify_signal


@dataclass
class ParticipantObservation:
    observation_id: int
    bbox: tuple
    detection_confidence: float
    track_id: int | None = None
    seat_id: str | None = None
    seat_assignment: str = "not_configured"
    observable_signal: str = UNKNOWN
    raw_signal: str = UNKNOWN
    status: str = "unknown"
    pose_detected: bool = False
    landmark_visibility: float | None = None
    supporting_visibility: float | None = None
    observed_span_seconds: float = 0.0
    reasons: list = field(default_factory=list)
    features: dict = field(default_factory=dict)
    anchor: tuple | None = None
    anchor_source: str = "bbox_upper_center"

    def to_dict(self):
        return {
            "observationId": self.observation_id, "trackId": self.track_id,
            "seatId": self.seat_id, "seatAssignment": self.seat_assignment,
            "bbox": list(self.bbox), "anchor": list(self.anchor) if self.anchor else None,
            "anchorSource": self.anchor_source,
            "observableSignal": self.observable_signal,
            "signalLabel": SIGNAL_META[self.observable_signal]["label"],
            "rawSignal": self.raw_signal, "status": self.status,
            "detectionConfidence": round(self.detection_confidence, 4),
            "poseDetected": self.pose_detected,
            "landmarkVisibility": self.landmark_visibility,
            "supportingLandmarkVisibility": self.supporting_visibility,
            "observedSpanSeconds": round(self.observed_span_seconds, 3),
            "reasons": self.reasons, "features": self.features,
        }


class ParticipantAnalyzer:
    def __init__(self, rules=None, confirmation_delay=False, max_gap=2.0):
        self.rules = rules or RuleConfig()
        self.confirmation_delay = confirmation_delay
        self.max_gap = max_gap
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True, model_complexity=1,
            min_detection_confidence=0.5,
        )
        self._signal_states = {}

    def reset(self):
        self._signal_states.clear()

    def retain_tracks(self, visible_ids):
        # A missed sampled observation breaks continuity immediately.
        for track_id in set(self._signal_states) - set(visible_ids):
            del self._signal_states[track_id]

    def analyze(self, frame, detection, observation_id, timestamp=None):
        box = detection.bbox
        observation = ParticipantObservation(
            observation_id, box, detection.confidence, detection.track_id,
            anchor=((box[0] + box[2]) / 2, box[1] + (box[3] - box[1]) * 0.20),
        )
        crop, origin = self._crop(frame, box)
        if crop is None:
            observation.reasons = ["invalid_person_crop"]
            self._forget(detection.track_id)
            return observation
        try:
            result = self._pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        except (ValueError, RuntimeError) as exc:
            observation.reasons = ["pose_processing_failed", str(exc)]
            self._forget(detection.track_id)
            return observation
        landmarks = result.pose_landmarks.landmark if result.pose_landmarks else None
        decision = classify_signal(landmarks, (crop.shape[1], crop.shape[0]), self.rules)
        observation.pose_detected = landmarks is not None
        observation.raw_signal = decision.signal
        observation.reasons = decision.reasons
        observation.features = decision.features
        observation.supporting_visibility = decision.supporting_visibility
        if landmarks:
            values = [landmarks[i].visibility for i in (0, 2, 5, 11, 12, 13, 14, 15, 16)
                      if np.isfinite(landmarks[i].visibility)]
            observation.landmark_visibility = round(float(np.mean(values)), 4) if values else None
            nose = landmarks[0]
            if (nose.visibility >= self.rules.min_visibility
                    and 0 <= nose.x <= 1 and 0 <= nose.y <= 1):
                observation.anchor = (origin[0] + nose.x * crop.shape[1],
                                      origin[1] + nose.y * crop.shape[0])
                observation.anchor_source = "pose_nose"
        if detection.track_id is None:
            state = SignalState()
        else:
            state = self._signal_states.setdefault(detection.track_id, SignalState())
        confirmation = state.update(
            decision.signal, timestamp, enabled=self.confirmation_delay,
            thresholds=self.rules.delays(), max_gap=self.max_gap,
        )
        observation.observable_signal = confirmation.signal
        observation.status = confirmation.status
        observation.observed_span_seconds = confirmation.observed_span_seconds
        return observation

    def _forget(self, track_id):
        self._signal_states.pop(track_id, None)

    @staticmethod
    def _crop(frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        px, py = int((x2 - x1) * 0.04), int((y2 - y1) * 0.04)
        x1, y1 = max(0, x1 - px), max(0, y1 - py)
        x2, y2 = min(w, x2 + px), min(h, y2 + py)
        if x1 >= x2 or y1 >= y2:
            return None, (x1, y1)
        return frame[y1:y2, x1:x2], (x1, y1)

    def close(self):
        self._pose.close()
        self.reset()
