"""Pose, gaze, observable signal, and activity-index analysis."""

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from behavior import (
    FORWARD_FACING,
    HEAD_DOWN,
    PEER_FACING,
    RAISED_HAND,
    SIDE_FACING,
    SIGNAL_META,
    SUSTAINED_HEAD_DOWN,
    WRITING_POSTURE,
    SignalState,
    classify_signal,
    is_review_flag_signal,
)


_PL = mp.solutions.pose.PoseLandmark


@dataclass
class ParticipantObservation:
    track_id: int
    observable_signal: str
    signal_label: str
    activity_index: float
    head_pose: str
    posture: str
    gaze_direction: str
    review_flag: bool
    confidence: float
    bbox: tuple


class ParticipantAnalyzer:
    def __init__(self):
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.45,
        )
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.45,
        )
        self._signal_states: dict = {}

    def analyze(self, frame: np.ndarray, bbox: tuple, track_id: int,
                timestamp: Optional[float] = None) -> Optional[ParticipantObservation]:
        crop = self._crop(frame, bbox)
        if crop is None:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        head_pose, posture, pose_landmarks, pose_conf = self._run_pose(crop_rgb)
        gaze, face_conf = self._run_face(crop_rgb)

        state = self._get_state(track_id)
        if pose_landmarks:
            signal = classify_signal(pose_landmarks, head_pose, state, timestamp=timestamp)
        else:
            signal = state.update(FORWARD_FACING, timestamp=timestamp)

        activity_index = self._activity_index(head_pose, posture, signal, gaze)
        meta = SIGNAL_META.get(signal, {})

        return ParticipantObservation(
            track_id=track_id,
            observable_signal=signal,
            signal_label=meta.get("label", signal),
            activity_index=round(activity_index, 1),
            head_pose=head_pose,
            posture=posture,
            gaze_direction=gaze,
            review_flag=is_review_flag_signal(signal),
            confidence=round(max(pose_conf, face_conf), 2),
            bbox=bbox,
        )

    def remove_participant(self, track_id: int):
        self._signal_states.pop(track_id, None)

    def close(self):
        self._pose.close()
        self._face_mesh.close()

    def _crop(self, frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        pad_x = int((x2 - x1) * 0.08)
        pad_y = int((y2 - y1) * 0.08)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _run_pose(self, crop_rgb: np.ndarray) -> tuple:
        result = self._pose.process(crop_rgb)
        if not result.pose_landmarks:
            return "facing_front", "upright", None, 0.3

        landmarks = result.pose_landmarks.landmark
        head_pose = _estimate_head_pose_from_pose(landmarks)
        posture = _estimate_posture(landmarks)
        return head_pose, posture, landmarks, 0.8

    def _run_face(self, crop_rgb: np.ndarray) -> tuple:
        result = self._face_mesh.process(crop_rgb)
        if not result.multi_face_landmarks:
            return "center", 0.3

        landmarks = result.multi_face_landmarks[0].landmark
        gaze = _estimate_gaze(landmarks)
        return gaze, 0.85

    def _get_state(self, track_id: int) -> SignalState:
        if track_id not in self._signal_states:
            self._signal_states[track_id] = SignalState()
        return self._signal_states[track_id]

    def _activity_index(self, head_pose: str, posture: str,
                        signal: str, gaze: str) -> float:
        score = 50.0

        score += {
            "facing_front": 20, "looking_up": 8,
            "looking_down": -8, "looking_left": -12, "looking_right": -12,
        }.get(head_pose, 0)

        score += {
            "upright": 15, "leaning_forward": 10,
            "leaning_back": -5, "slouching": -18, "lying_down": -35,
        }.get(posture, 0)

        score += {
            FORWARD_FACING: 15,
            WRITING_POSTURE: 12,
            RAISED_HAND: 18,
            HEAD_DOWN: -12,
            SIDE_FACING: -18,
            PEER_FACING: -10,
            SUSTAINED_HEAD_DOWN: -40,
        }.get(signal, 0)

        score += {"center": 5, "down": -5, "left": -8, "right": -8, "up": 3}.get(gaze, 0)
        return max(0.0, min(100.0, score))


def _estimate_head_pose_from_pose(landmarks) -> str:
    nose = landmarks[_PL.NOSE.value]
    left_ear = landmarks[_PL.LEFT_EAR.value]
    right_ear = landmarks[_PL.RIGHT_EAR.value]
    left_shoulder = landmarks[_PL.LEFT_SHOULDER.value]
    right_shoulder = landmarks[_PL.RIGHT_SHOULDER.value]

    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    nose_rel_y = shoulder_y - nose.y
    if nose_rel_y < 0.05:
        return "looking_down"

    left_vis = left_ear.visibility
    right_vis = right_ear.visibility
    if left_vis > 0.5 and right_vis < 0.2:
        return "looking_right"
    if right_vis > 0.5 and left_vis < 0.2:
        return "looking_left"

    return "facing_front"


def _estimate_posture(landmarks) -> str:
    nose = landmarks[_PL.NOSE.value]
    left_shoulder = landmarks[_PL.LEFT_SHOULDER.value]
    right_shoulder = landmarks[_PL.RIGHT_SHOULDER.value]

    if left_shoulder.visibility < 0.3 or right_shoulder.visibility < 0.3:
        return "upright"

    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    nose_to_shoulder = shoulder_y - nose.y

    if nose_to_shoulder < 0.02:
        return "lying_down"
    if nose_to_shoulder < 0.08:
        return "slouching"
    if nose_to_shoulder > 0.25:
        return "leaning_back"
    return "upright"


def _estimate_gaze(face_lm) -> str:
    nose = face_lm[1]
    left_eye_outer = face_lm[33]
    right_eye_outer = face_lm[263]
    forehead = face_lm[10]
    chin = face_lm[152]

    eye_center_x = (left_eye_outer.x + right_eye_outer.x) / 2
    nose_offset_x = nose.x - eye_center_x

    face_h = abs(forehead.y - chin.y)
    nose_rel_y = (nose.y - forehead.y) / max(face_h, 0.01)

    if nose_rel_y > 0.72:
        return "down"
    if nose_rel_y < 0.28:
        return "up"
    if nose_offset_x < -0.04:
        return "right"
    if nose_offset_x > 0.04:
        return "left"
    return "center"
