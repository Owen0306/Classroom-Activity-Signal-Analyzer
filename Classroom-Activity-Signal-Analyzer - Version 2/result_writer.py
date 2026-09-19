# Versioned result files with per-detection detail and explicit coverage.

from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from behavior import CATEGORIES, UNKNOWN, NOT_DETECTED
from version import __version__, SCHEMA_VERSION


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def aggregate(observations, expected=None, missing_seats=None, processed=True):
    counts = {signal: 0 for signal in (*CATEGORIES, UNKNOWN)}
    for observation in observations:
        counts[observation.observable_signal] += 1
    detected = len(observations)
    known = detected - counts[UNKNOWN]
    assigned = [o for o in observations if o.seat_id is not None]
    missing = None
    if processed and expected is not None:
        missing = len(missing_seats) if missing_seats is not None else max(expected - detected, 0)
    result = {
        "detectedPersonCount": detected, "classifiedPersonCount": known,
        "unknownPersonCount": counts[UNKNOWN],
        "pendingPersonCount": sum(o.status == "pending" for o in observations),
        "expectedParticipantCount": expected, "missingParticipantCount": missing,
        "extraDetectionCount": max(detected - expected, 0) if expected is not None and processed else None,
        "classificationCoverage": round(known / detected, 4) if detected else None,
        "signalCounts": counts,
        "signalRatiosAmongDetections": {s: round(n / detected, 4) if detected else None for s, n in counts.items()},
    }
    if missing_seats is not None:
        seat_counts = {signal: 0 for signal in (*CATEGORIES, UNKNOWN, NOT_DETECTED)}
        for observation in assigned:
            seat_counts[observation.observable_signal] += 1
        seat_counts[NOT_DETECTED] = len(missing_seats)
        result.update(
            assignedSeatCount=len(assigned),
            unassignedDetectionCount=detected - len(assigned),
            missingSeatIds=missing_seats if processed else None,
            unassignedSeatIds=missing_seats,
            seatSignalCounts=seat_counts if processed else None,
            seatSignalRatios={s: round(n / expected, 4) for s, n in seat_counts.items()}
            if processed and expected else None,
        )
    return result


class LocalResultWriter:
    def __init__(self, output_path, metadata):
        self.output_path = Path(output_path)
        self.metadata = metadata
        self.samples = []
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.run_status = "running"
        self.stop_reason = None
        self.error = None

    def record(self, sample, observations, expected=None, missing_seats=None, status="ok", error=None, warnings=None):
        processed = status in ("ok", "no_detections")
        record = {
            "sampleIndex": len(self.samples), "sampleId": sample.sample_id,
            "sourcePath": sample.source_path, "sourceTimestampSeconds": sample.timestamp,
            "processedAt": datetime.now(timezone.utc).isoformat(),
            "status": status, "error": error, "warnings": warnings or [],
            "imageWidth": sample.frame.shape[1] if sample.frame is not None else None,
            "imageHeight": sample.frame.shape[0] if sample.frame is not None else None,
            "observations": [observation.to_dict() for observation in observations],
            **aggregate(observations, expected, missing_seats, processed),
        }
        self.samples.append(record)
        self.checkpoint()
        return record

    def payload(self):
        processed = [s for s in self.samples if s["status"] in ("ok", "no_detections")]
        distribution = Counter({s: 0 for s in (*CATEGORIES, UNKNOWN)})
        for sample in processed:
            distribution.update(sample["signalCounts"])
        total = sum(distribution.values())
        known = total - distribution[UNKNOWN]
        timestamps = [s["sourceTimestampSeconds"] for s in self.samples
                      if s["sourceTimestampSeconds"] is not None]
        missing = [s["missingParticipantCount"] for s in processed
                   if s["missingParticipantCount"] is not None]
        return {
            "schemaVersion": SCHEMA_VERSION, "applicationVersion": __version__,
            "runStatus": self.run_status, "stopReason": self.stop_reason, "error": self.error,
            "startedAt": self.started_at,
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "metadata": self.metadata,
            "summary": {
                "sampleCount": len(self.samples), "processedSampleCount": len(processed),
                "failedSampleCount": len(self.samples) - len(processed),
                "emptySampleCount": sum(s["detectedPersonCount"] == 0 for s in processed),
                "totalPersonObservations": total, "classifiedPersonObservations": known,
                "unknownPersonObservations": distribution[UNKNOWN],
                "classificationCoverage": round(known / total, 4) if total else None,
                "maxDetectedPersons": max((s["detectedPersonCount"] for s in processed), default=0),
                "meanDetectedPersons": round(total / len(processed), 2) if processed else None,
                "missingParticipantObservations": sum(missing) if missing else None,
                "sourceObservedSpanSeconds": round(max(timestamps) - min(timestamps), 3) if timestamps else None,
            },
            "session": {
                "signalDistribution": dict(distribution),
                "signalRatiosAmongDetections": {
                    s: round(n / total, 4) if total else None for s, n in distribution.items()
                },
                "countUnit": "person observations, not unique people",
            },
            "samples": self.samples,
        }

    def checkpoint(self):
        atomic_json(self.output_path, self.payload())

    def complete(self, status="completed", stop_reason="end_of_input", error=None):
        self.run_status, self.stop_reason, self.error = status, stop_reason, error
        self.checkpoint()
        self._write_csv()
        return self.payload()

    def _write_csv(self):
        path = self.output_path.with_suffix(".csv")
        temporary = path.with_name(path.name + ".tmp")
        fields = ["sample_id", "source_path", "source_timestamp_seconds", "sample_status",
                  "observation_id", "track_id", "seat_id", "seat_assignment", "signal", "raw_signal",
                  "status", "detection_confidence", "landmark_visibility", "bbox", "reasons"]
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for sample in self.samples:
                base = {"sample_id": sample["sampleId"], "source_path": sample["sourcePath"],
                        "source_timestamp_seconds": sample["sourceTimestampSeconds"],
                        "sample_status": sample["status"]}
                for o in sample["observations"]:
                    writer.writerow({
                        **base, "observation_id": o["observationId"], "track_id": o["trackId"],
                        "seat_id": o["seatId"], "seat_assignment": o["seatAssignment"],
                        "signal": o["observableSignal"], "raw_signal": o["rawSignal"],
                        "status": o["status"], "detection_confidence": o["detectionConfidence"],
                        "landmark_visibility": o["landmarkVisibility"],
                        "bbox": json.dumps(o["bbox"]), "reasons": "; ".join(o["reasons"]),
                    })
                for seat_id in sample.get("unassignedSeatIds", []):
                    writer.writerow({**base, "seat_id": seat_id,
                                     "signal": NOT_DETECTED if sample["status"] in ("ok", "no_detections") else "not_evaluated",
                                     "status": "unassigned"})
                if not sample["observations"] and not sample.get("unassignedSeatIds"):
                    writer.writerow({**base, "status": sample["status"], "reasons": sample["error"] or ""})
        temporary.replace(path)
