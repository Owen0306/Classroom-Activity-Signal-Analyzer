"""Rule-based observable classroom activity signals."""

import time
from dataclasses import dataclass, field
from typing import Optional

import mediapipe as mp


_PL = mp.solutions.pose.PoseLandmark

FORWARD_FACING = "forward_facing_posture"
WRITING_POSTURE = "writing_posture"
RAISED_HAND = "raised_hand"
HEAD_DOWN = "head_down_posture"
SIDE_FACING = "side_facing_posture"
PEER_FACING = "possible_peer_facing_interaction"
SUSTAINED_HEAD_DOWN = "sustained_head_down_posture"


THRESHOLDS = {
    SUSTAINED_HEAD_DOWN: 20,
    SIDE_FACING: 10,
    PEER_FACING: 8,
    HEAD_DOWN: 10,
    RAISED_HAND: 0,
    WRITING_POSTURE: 0,
    FORWARD_FACING: 0,
}


@dataclass
class SignalState:
    onset_times: dict = field(default_factory=dict)
    last_confirmed: str = FORWARD_FACING

    def update(self, raw_signal: str, timestamp: Optional[float] = None) -> str:
        now = time.time() if timestamp is None else timestamp
        threshold = THRESHOLDS.get(raw_signal, 0)

        if threshold == 0:
            self.onset_times.clear()
            self.last_confirmed = raw_signal
            return raw_signal

        if raw_signal not in self.onset_times:
            self.onset_times = {raw_signal: now}

        elapsed = now - self.onset_times[raw_signal]
        if elapsed >= threshold:
            self.last_confirmed = raw_signal
            return raw_signal

        return self.last_confirmed

    def reset(self):
        self.onset_times.clear()
        self.last_confirmed = FORWARD_FACING


def detect_sustained_head_down_candidate(landmarks) -> bool:
    nose = landmarks[_PL.NOSE.value]
    left_shoulder = landmarks[_PL.LEFT_SHOULDER.value]
    right_shoulder = landmarks[_PL.RIGHT_SHOULDER.value]

    if left_shoulder.visibility < 0.3 and right_shoulder.visibility < 0.3:
        return False

    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    return (nose.y - shoulder_y) > -0.02


def detect_side_facing(landmarks, head_pose: str) -> bool:
    if head_pose in ("looking_left", "looking_right"):
        left_eye = landmarks[_PL.LEFT_EYE.value]
        right_eye = landmarks[_PL.RIGHT_EYE.value]
        if left_eye.visibility > 0.3 and right_eye.visibility > 0.3:
            return abs(left_eye.y - right_eye.y) < 0.015
    return False


def detect_raised_hand(landmarks) -> bool:
    left_wrist = landmarks[_PL.LEFT_WRIST.value]
    right_wrist = landmarks[_PL.RIGHT_WRIST.value]
    left_shoulder = landmarks[_PL.LEFT_SHOULDER.value]
    right_shoulder = landmarks[_PL.RIGHT_SHOULDER.value]

    left_raised = (
        left_wrist.visibility > 0.4 and
        left_wrist.y < left_shoulder.y - 0.12
    )
    right_raised = (
        right_wrist.visibility > 0.4 and
        right_wrist.y < right_shoulder.y - 0.12
    )
    return left_raised or right_raised


def detect_writing_posture(landmarks) -> bool:
    left_wrist = landmarks[_PL.LEFT_WRIST.value]
    right_wrist = landmarks[_PL.RIGHT_WRIST.value]

    left_low = left_wrist.visibility > 0.3 and left_wrist.y > 0.62
    right_low = right_wrist.visibility > 0.3 and right_wrist.y > 0.62
    return left_low and right_low


def detect_peer_facing_interaction(landmarks, head_pose: str = "") -> bool:
    left_shoulder = landmarks[_PL.LEFT_SHOULDER.value]
    right_shoulder = landmarks[_PL.RIGHT_SHOULDER.value]

    if head_pose not in ("looking_left", "looking_right"):
        return False
    if left_shoulder.visibility < 0.3 or right_shoulder.visibility < 0.3:
        return False

    shoulder_tilt = abs(left_shoulder.y - right_shoulder.y)
    return shoulder_tilt > 0.06


def classify_signal(landmarks, head_pose: str, state: SignalState,
                    timestamp: Optional[float] = None) -> str:
    # Priority: raised hand, deep head-down, writing, side-facing, peer-facing, head-down.
    if detect_raised_hand(landmarks):
        raw = RAISED_HAND
    elif detect_sustained_head_down_candidate(landmarks):
        raw = SUSTAINED_HEAD_DOWN
    elif detect_writing_posture(landmarks):
        raw = WRITING_POSTURE
    elif detect_side_facing(landmarks, head_pose):
        raw = SIDE_FACING
    elif detect_peer_facing_interaction(landmarks, head_pose):
        raw = PEER_FACING
    elif head_pose == "looking_down":
        raw = HEAD_DOWN
    else:
        raw = FORWARD_FACING

    return state.update(raw, timestamp=timestamp)


SIGNAL_META = {
    FORWARD_FACING: {
        "label": "forward-facing posture",
        "review_flag": False,
        "color": "#22c55e",
    },
    WRITING_POSTURE: {
        "label": "writing posture",
        "review_flag": False,
        "color": "#3b82f6",
    },
    RAISED_HAND: {
        "label": "raised hand",
        "review_flag": False,
        "color": "#a855f7",
    },
    HEAD_DOWN: {
        "label": "head-down posture",
        "review_flag": True,
        "color": "#f59e0b",
    },
    SIDE_FACING: {
        "label": "side-facing posture",
        "review_flag": True,
        "color": "#f97316",
    },
    PEER_FACING: {
        "label": "possible peer-facing interaction",
        "review_flag": True,
        "color": "#eab308",
    },
    SUSTAINED_HEAD_DOWN: {
        "label": "sustained head-down posture",
        "review_flag": True,
        "color": "#dc2626",
    },
}


def is_review_flag_signal(signal: str) -> bool:
    return SIGNAL_META.get(signal, {}).get("review_flag", False)
