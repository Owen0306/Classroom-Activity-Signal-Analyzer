"""
result_writer.py - Local aggregate result writer.

Stores session-level and window-level summaries without persistent participant
profiles.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("result_writer")


class LocalResultWriter:
    """Collect aggregate results, then write them to JSON."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.windows: list[dict[str, Any]] = []
        self.signal_counts: dict[str, int] = defaultdict(int)
        self.review_flag_counts: dict[str, int] = defaultdict(int)
        self.participant_counts: list[int] = []
        self.activity_indexes: list[float] = []

    def record_window(self, window_index: int, timestamp: float,
                      observed_participant_count: int,
                      observable_activity_index: float,
                      signal_counts: dict,
                      signal_ratios: dict,
                      review_flag_counts: dict) -> bool:
        dominant_signal = _dominant_key(signal_counts)
        review_flag_total = sum(review_flag_counts.values())

        self.windows.append({
            "windowIndex": window_index,
            "timestamp": round(timestamp, 2),
            "observedParticipantCount": observed_participant_count,
            "observableActivityIndex": round(observable_activity_index, 1),
            "dominantObservableSignal": dominant_signal,
            "signalCounts": signal_counts,
            "signalRatios": signal_ratios,
            "reviewFlagCounts": review_flag_counts,
            "reviewFlagTotal": review_flag_total,
        })

        self.participant_counts.append(observed_participant_count)
        self.activity_indexes.append(observable_activity_index)

        for signal, count in signal_counts.items():
            self.signal_counts[signal] += count
        for signal, count in review_flag_counts.items():
            self.review_flag_counts[signal] += count

        return True

    def complete(self) -> bool:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self._summary(),
            "session": self._session(),
            "windows": self.windows,
        }
        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"Results written to: {self.output_path}")
        return True

    @property
    def stats(self) -> dict:
        return {
            "windows": len(self.windows),
            "output_path": str(self.output_path),
        }

    def _summary(self) -> dict:
        frame_count = len(self.windows)
        participant_counts = self.participant_counts or [0]
        activity_indexes = self.activity_indexes or [0.0]
        duration = self.windows[-1]["timestamp"] if self.windows else 0.0

        return {
            "windowCount": frame_count,
            "durationSeconds": round(duration, 2),
            "maxObservedParticipants": max(participant_counts),
            "avgObservedParticipants": round(
                sum(participant_counts) / len(participant_counts),
                1,
            ),
            "meanObservableActivityIndex": round(
                sum(activity_indexes) / len(activity_indexes),
                1,
            ),
            "reviewFlagTotal": sum(self.review_flag_counts.values()),
        }

    def _session(self) -> dict:
        total = sum(self.signal_counts.values())
        signal_distribution = dict(self.signal_counts)
        signal_ratios = {
            signal: round(count / total, 4)
            for signal, count in signal_distribution.items()
        } if total else {}

        return {
            "signalDistribution": signal_distribution,
            "signalRatios": signal_ratios,
            "dominantObservableSignal": _dominant_key(signal_distribution),
            "reviewFlagDistribution": dict(self.review_flag_counts),
        }


def _dominant_key(counts: dict) -> str | None:
    if not counts:
        return None
    return max(counts, key=counts.get)
