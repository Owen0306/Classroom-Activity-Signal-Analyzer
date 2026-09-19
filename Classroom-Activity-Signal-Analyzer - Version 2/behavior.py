# Five observable posture categories and optional temporal confirmation.

from dataclasses import asdict, dataclass, field
import math
import time

FORWARD_FACING = "forward_facing_posture"
WRITING_POSTURE = "writing_posture"
RAISED_HAND = "raised_hand"
HEAD_DOWN = "head_down_posture"
SIDE_FACING = "side_facing_posture"
UNKNOWN = "unknown"
NOT_DETECTED = "not_detected"
CATEGORIES = (FORWARD_FACING, WRITING_POSTURE, RAISED_HAND, HEAD_DOWN, SIDE_FACING)
RULE_VERSION = "2.0.0"
LEGACY_SIGNALS = {
    "possible_peer_facing_interaction": SIDE_FACING,
    "sustained_head_down_posture": HEAD_DOWN,
    "deep_head_down": HEAD_DOWN,
}
SIGNAL_META = {
    FORWARD_FACING: {"label": "Forward facing", "color": "#22c55e"},
    WRITING_POSTURE: {"label": "Writing posture", "color": "#3b82f6"},
    RAISED_HAND: {"label": "Raised hand", "color": "#a855f7"},
    HEAD_DOWN: {"label": "Head down", "color": "#f59e0b"},
    SIDE_FACING: {"label": "Side / peer facing", "color": "#f97316"},
    UNKNOWN: {"label": "Unable to determine", "color": "#94a3b8"},
}


@dataclass(frozen=True)
class RuleConfig:
    # Geometry uses pixel distances divided by visible shoulder width.
    min_visibility: float = 0.5
    min_shoulder_pixels: float = 10.0
    raised_wrist_above_shoulder: float = 0.35
    head_down_nose_gap: float = 0.20
    side_nose_offset: float = 0.28
    front_nose_offset: float = 0.18
    writing_wrist_min_drop: float = 0.25
    writing_wrist_max_drop: float = 1.65
    writing_forearm_max_slope: float = 0.65
    writing_wrist_max_distance: float = 1.20
    head_down_delay_seconds: float = 10.0
    side_facing_delay_seconds: float = 10.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Rule {key} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Rule {key} must be finite and nonnegative")
        if not 0 < self.min_visibility <= 1:
            raise ValueError("min_visibility must be in (0, 1]")
        if self.min_shoulder_pixels <= 0:
            raise ValueError("min_shoulder_pixels must be positive")
        if self.front_nose_offset >= self.side_nose_offset:
            raise ValueError("front_nose_offset must be less than side_nose_offset")
        if self.writing_wrist_min_drop >= self.writing_wrist_max_drop:
            raise ValueError("Invalid writing wrist range")

    def delays(self):
        return {HEAD_DOWN: self.head_down_delay_seconds,
                SIDE_FACING: self.side_facing_delay_seconds}


@dataclass
class Decision:
    signal: str = UNKNOWN
    reasons: list[str] = field(default_factory=list)
    features: dict = field(default_factory=dict)
    supporting_visibility: float | None = None


@dataclass
class Confirmation:
    signal: str
    status: str
    observed_span_seconds: float


@dataclass
class SignalState:
    # Confirm consecutive observations without substituting a stale label.
    candidate: str | None = None
    onset: float | None = None
    last_seen: float | None = None

    def reset(self):
        self.candidate = self.onset = self.last_seen = None

    def update(self, raw_signal, timestamp=None, *, enabled=True,
               thresholds=None, max_gap=2.0) -> Confirmation:
        now = time.monotonic() if timestamp is None else timestamp
        if not math.isfinite(now):
            self.reset()
            return Confirmation(UNKNOWN, "unknown", 0.0)
        if raw_signal == UNKNOWN:
            self.reset()
            return Confirmation(UNKNOWN, "unknown", 0.0)
        if not enabled:
            self.reset()
            return Confirmation(raw_signal, "instant", 0.0)
        if self.last_seen is not None and (now <= self.last_seen or now - self.last_seen > max_gap):
            self.reset()
        if self.candidate != raw_signal:
            self.candidate, self.onset = raw_signal, now
        self.last_seen = now
        span = now - self.onset
        threshold = (RuleConfig().delays() if thresholds is None else thresholds).get(raw_signal, 0.0)
        if span >= threshold:
            return Confirmation(raw_signal, "confirmed", span)
        return Confirmation(UNKNOWN, "pending", span)


# Stable MediaPipe Pose indices; geometry can be checked without inference.
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 2, 5, 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST = 13, 14, 15, 16


def classify_signal(landmarks, image_size, config=None) -> Decision:
    # Priority: raised hand, side facing, writing posture, head down, front.
    # Writing posture is a geometric candidate; ambiguous evidence stays unknown.
    cfg = config or RuleConfig()
    if landmarks is None or len(landmarks) < 17:
        return Decision(reasons=["pose_not_detected"])
    width, height = image_size

    def visible(*indices):
        return all(
            math.isfinite(landmarks[i].x) and math.isfinite(landmarks[i].y)
            and 0 <= landmarks[i].x <= 1 and 0 <= landmarks[i].y <= 1
            and math.isfinite(landmarks[i].visibility)
            and landmarks[i].visibility >= cfg.min_visibility
            for i in indices
        )

    def point(i):
        return landmarks[i].x * width, landmarks[i].y * height

    if not visible(LEFT_SHOULDER, RIGHT_SHOULDER):
        return Decision(reasons=["shoulders_not_reliable"])
    left, right = point(LEFT_SHOULDER), point(RIGHT_SHOULDER)
    scale = math.dist(left, right)
    if scale < cfg.min_shoulder_pixels:
        return Decision(reasons=["body_too_small"], features={"shoulderWidthPixels": round(scale, 3)})
    center = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
    features = {"shoulderWidthPixels": round(scale, 3)}
    support = [LEFT_SHOULDER, RIGHT_SHOULDER]

    def result(signal, reason, indices=()):
        return Decision(signal, [reason], features,
                        round(min(landmarks[i].visibility for i in support + list(indices)), 4))

    raised = []
    for name, wrist, shoulder in (("left", LEFT_WRIST, LEFT_SHOULDER),
                                  ("right", RIGHT_WRIST, RIGHT_SHOULDER)):
        if visible(wrist):
            elevation = (point(shoulder)[1] - point(wrist)[1]) / scale
            features[f"{name}WristElevation"] = round(elevation, 4)
            if elevation >= cfg.raised_wrist_above_shoulder:
                raised.append(wrist)
    if raised:
        return result(RAISED_HAND, "wrist_above_visible_shoulder", raised)
    if not visible(NOSE):
        return Decision(reasons=["nose_not_reliable"], features=features)
    nose = point(NOSE)
    gap = (center[1] - nose[1]) / scale
    offset = (nose[0] - center[0]) / scale
    features.update(noseShoulderGap=round(gap, 4), noseHorizontalOffset=round(offset, 4))
    down = gap < cfg.head_down_nose_gap

    # An offset alone can be body lean; require a second facial cue.
    face_asymmetry = False
    if visible(LEFT_EYE, RIGHT_EYE):
        le, re = point(LEFT_EYE), point(RIGHT_EYE)
        eye_span = math.dist(le, re)
        if eye_span >= 3:
            asymmetry = abs(nose[0] - (le[0] + re[0]) / 2) / eye_span
            features["noseEyeAsymmetry"] = round(asymmetry, 4)
            face_asymmetry = asymmetry > 0.45
    ear_asymmetry = (
        visible(LEFT_EAR) and landmarks[RIGHT_EAR].visibility < 0.25
    ) or (visible(RIGHT_EAR) and landmarks[LEFT_EAR].visibility < 0.25)
    if not down and abs(offset) >= cfg.side_nose_offset and (face_asymmetry or ear_asymmetry):
        face_support = [LEFT_EYE, RIGHT_EYE] if face_asymmetry else [
            LEFT_EAR if visible(LEFT_EAR) else RIGHT_EAR]
        return result(SIDE_FACING, "lateral_head_offset_and_face_asymmetry", [NOSE, *face_support])

    wrists = (LEFT_WRIST, RIGHT_WRIST)
    if down and visible(*wrists):
        drops = [(point(i)[1] - center[1]) / scale for i in wrists]
        distance = math.dist(point(LEFT_WRIST), point(RIGHT_WRIST)) / scale
        bent_arm = False
        elbow_support = []
        for elbow, wrist in ((LEFT_ELBOW, LEFT_WRIST), (RIGHT_ELBOW, RIGHT_WRIST)):
            if visible(elbow):
                dx = abs(point(wrist)[0] - point(elbow)[0]) / scale
                dy = abs(point(wrist)[1] - point(elbow)[1]) / scale
                if dx > 0.15 and dy / dx <= cfg.writing_forearm_max_slope:
                    bent_arm = True
                    elbow_support = [elbow]
                    break
        features.update(wristDistance=round(distance, 4), writingArmGeometry=bent_arm)
        if (all(cfg.writing_wrist_min_drop <= d <= cfg.writing_wrist_max_drop for d in drops)
                and distance <= cfg.writing_wrist_max_distance and bent_arm):
            return result(WRITING_POSTURE, "head_down_with_visible_desk_level_arm_geometry",
                          [NOSE, *wrists, *elbow_support])
    if down:
        return result(HEAD_DOWN, "nose_low_relative_to_shoulders", [NOSE])
    if abs(offset) <= cfg.front_nose_offset and visible(LEFT_EYE, RIGHT_EYE):
        if not face_asymmetry and not ear_asymmetry:
            return result(FORWARD_FACING, "centered_head_with_visible_eyes", [NOSE, LEFT_EYE, RIGHT_EYE])
    return Decision(reasons=["ambiguous_head_orientation"], features=features)
