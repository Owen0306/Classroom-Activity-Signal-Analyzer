# Command-line application for independent photos and optional temporal analysis.

import argparse
from dataclasses import asdict
import json
import logging
import math
from importlib.metadata import version
from pathlib import Path
import sys

import cv2

from analyzer import ParticipantAnalyzer
from behavior import RuleConfig, RULE_VERSION, SIGNAL_META
from result_writer import LocalResultWriter
from seating import SeatMap
from sources import photo_samples, capture_samples, image_paths, write_image
from tracker import PersonTracker
from version import __version__

logger = logging.getLogger("classroom")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Classroom Activity Signal Analyzer 2.0")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--source", choices=["image", "images", "video", "camera", "rtsp"])
    parser.add_argument("--image-path")
    parser.add_argument("--images-dir")
    parser.add_argument("--video-path")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--rtsp-url")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--watch", action="store_true", help="Process incoming or updated photos after they stabilize")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--analysis-mode", choices=["snapshot", "temporal"], default="snapshot")
    parser.add_argument("--confirmation-delay", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-observation-gap", type=float)
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between capture samples")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-read-failures", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.25, help="Person detection threshold, not posture accuracy")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="Overlap threshold for duplicate suppression")
    parser.add_argument("--expected-participants", type=int, help="Optional expected count; no default class size")
    parser.add_argument("--seat-map", help="Optional normalized seat regions for a fixed camera")
    parser.add_argument("--rules-config", help="JSON overrides for posture rules")
    parser.add_argument("--output-json", default="output/results.json")
    parser.add_argument("--annotated-dir")
    parser.add_argument("--no-charts", action="store_true")
    parser.add_argument("--display", dest="display", action="store_true", default=False)
    parser.add_argument("--no-display", dest="display", action="store_false")
    args = parser.parse_args(argv)

    inferred = [name for name, value in (("image", args.image_path), ("images", args.images_dir),
                                        ("video", args.video_path), ("rtsp", args.rtsp_url)) if value]
    if len(inferred) > 1:
        parser.error("Supply only one input path or URL")
    args.source = args.source or (inferred[0] if inferred else None)
    if args.source is None:
        parser.error("Choose --image-path, --images-dir, --video-path, or --source camera/rtsp")
    if inferred and args.source != inferred[0]:
        parser.error("--source does not match the supplied input")
    required = {"image": args.image_path, "images": args.images_dir,
                "video": args.video_path, "rtsp": args.rtsp_url}
    if args.source in required and not required[args.source]:
        parser.error(f"--source {args.source} requires its input path or URL")
    for name in ("image_path", "video_path", "seat_map", "rules_config"):
        path = getattr(args, name)
        if path and not Path(path).is_file():
            parser.error(f"File not found: {path}")
    if args.source == "images":
        if not Path(args.images_dir).is_dir():
            parser.error("Photo directory does not exist")
        if not args.watch and not image_paths(args.images_dir, args.recursive):
            parser.error("Photo directory contains no supported images")
    elif args.watch or args.recursive:
        parser.error("--watch and --recursive require --source images")

    for name in ("interval", "poll_interval"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("max_samples", "max_read_failures", "imgsz", "max_detections", "expected_participants"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.confidence) or not 0 < args.confidence <= 1:
        parser.error("--confidence must be in (0, 1]")
    if not math.isfinite(args.iou_threshold) or not 0 < args.iou_threshold <= 1:
        parser.error("--iou-threshold must be in (0, 1]")
    if args.analysis_mode == "temporal" and args.source in ("image", "images"):
        parser.error("Independent photos use snapshot mode; temporal mode requires a continuous capture source")
    if args.analysis_mode == "snapshot" and args.confirmation_delay:
        parser.error("Snapshot mode cannot confirm duration; use --no-confirmation-delay or temporal mode")
    if args.confirmation_delay is None:
        args.confirmation_delay = args.analysis_mode == "temporal"
    if args.max_observation_gap is None:
        args.max_observation_gap = args.interval * 1.5
    if not math.isfinite(args.max_observation_gap) or args.max_observation_gap <= 0:
        parser.error("--max-observation-gap must be finite and positive")

    try:
        settings = json.loads(Path(args.rules_config).read_text(encoding="utf-8-sig")) if args.rules_config else {}
        if not isinstance(settings, dict):
            raise ValueError("Rules configuration must be a JSON object")
        args.rules = RuleConfig(**settings)
        args.seats = SeatMap(args.seat_map) if args.seat_map else None
        if args.seats:
            count = len(args.seats.seats)
            if args.expected_participants is not None and args.expected_participants != count:
                raise ValueError("Expected count must equal the number of configured seat regions")
            args.expected_participants = count
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    if Path(args.output_json).suffix.lower() != ".json":
        parser.error("--output-json must use a .json extension")
    if args.source == "images":
        root = Path(args.images_dir).resolve()
        outputs = [Path(args.output_json).resolve().parent]
        if args.annotated_dir:
            outputs.append(Path(args.annotated_dir).resolve())
        if any(p == root or (args.recursive and p.is_relative_to(root)) for p in outputs):
            parser.error("Store results and annotated images outside the scanned photo directory")
    return args


def draw_annotations(frame, observations, record, seat_map=None):
    out = frame.copy()
    height, width = out.shape[:2]
    if seat_map:
        for seat in seat_map.seats:
            x1, y1, x2, y2 = seat["region"]
            cv2.rectangle(out, (int(x1 * width), int(y1 * height)),
                          (int(x2 * width), int(y2 * height)), (160, 160, 160), 1)
    for observation in observations:
        x1, y1, x2, y2 = observation.bbox
        meta = SIGNAL_META[observation.observable_signal]
        value = meta["color"].lstrip("#")
        color = tuple(int(value[i:i + 2], 16) for i in (4, 2, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        identity = f"Seat {observation.seat_id}" if observation.seat_id else f"P{observation.observation_id:03d}"
        short_labels = {"forward_facing_posture": "Forward", "writing_posture": "Writing",
                        "raised_hand": "Hand up", "head_down_posture": "Head down",
                        "side_facing_posture": "Side/peer", "unknown": "Unknown"}
        label = f"{identity}: {short_labels[observation.observable_signal]}"
        if observation.status == "pending":
            label = f"{identity}: Pending"
        scale = 0.42
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        x = max(0, min(x1, width - text_width - 6))
        y = max(text_height + 5, min(y1, height - baseline - 2))
        cv2.rectangle(out, (x, y - text_height - 5), (min(width - 1, x + text_width + 5), y + baseline),
                      (25, 25, 25), -1)
        cv2.putText(out, label, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    info = f"Detected: {len(observations)}  |  Classified: {record['classifiedPersonCount']}  |  Unknown: {record['unknownPersonCount']}"
    if record["missingParticipantCount"] is not None:
        info += f"  |  Missing: {record['missingParticipantCount']}"
    cv2.rectangle(out, (0, 0), (min(width - 1, 1050), 32), (20, 20, 20), -1)
    cv2.putText(out, info, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
    return out


class ClassroomSession:
    def __init__(self, args):
        self.args = args
        metadata = {
            "sourceType": args.source, "analysisMode": args.analysis_mode,
            "confirmationDelayEnabled": args.confirmation_delay,
            "captureIntervalSeconds": args.interval if args.source not in ("image", "images") else None,
            "maxObservationGapSeconds": args.max_observation_gap if args.analysis_mode == "temporal" else None,
            "ruleVersion": RULE_VERSION, "rules": asdict(args.rules),
            "detector": {"model": str(args.yolo_model), "confidenceThreshold": args.confidence,
                         "imageSize": args.imgsz, "maxDetections": args.max_detections,
                         "iouThreshold": args.iou_threshold},
            "runtime": {"python": sys.version.split()[0], **{name: version(name) for name in
                        ("opencv-python", "mediapipe", "ultralytics", "numpy")}},
            "expectedParticipantCount": args.expected_participants,
            "seatMap": args.seats.seats if args.seats else None,
            "observationIdScope": "one sample; not a seat number or personal identity",
            "ratioDenominator": "all detected people, including unknown and pending",
            "confidenceMeaning": "person detector score; landmark visibility is evidence quality, not posture accuracy",
        }
        self.writer = LocalResultWriter(args.output_json, metadata)
        self.tracker = self.analyzer = None

    def run(self):
        args = self.args
        stream = None
        run_status, stop_reason, run_error = "completed", "end_of_input", None
        exit_code = 0
        try:
            self.writer.checkpoint()
            self.tracker = PersonTracker(args.yolo_model, args.confidence, args.imgsz,
                                         args.max_detections, args.iou_threshold)
            self.analyzer = ParticipantAnalyzer(args.rules, args.confirmation_delay, args.max_observation_gap)
            stream = photo_samples(args) if args.source in ("image", "images") else capture_samples(args)
            for sample in stream:
                observations, missing_seats, warnings = [], None, []
                status, error = "ok", sample.error
                if args.seats:
                    missing_seats = [s["id"] for s in args.seats.seats]
                if sample.frame is None:
                    status = "read_error"
                    self.analyzer.reset()
                    self.tracker.reset()
                else:
                    try:
                        temporal = args.analysis_mode == "temporal"
                        if not temporal:
                            self.analyzer.reset()
                        detections = self.tracker.update(sample.frame, tracking=temporal)
                        self.analyzer.retain_tracks([d.track_id for d in detections])
                        observations = [self.analyzer.analyze(sample.frame, d, i + 1, sample.timestamp)
                                        for i, d in enumerate(detections)]
                        if args.seats:
                            missing_seats = args.seats.assign(observations, (sample.frame.shape[1], sample.frame.shape[0]))
                        if not observations:
                            status = "no_detections"
                        if len(detections) >= args.max_detections:
                            warnings.append("detector_limit_reached")
                            logger.warning("Detection limit reached in %s; consider --max-detections", sample.sample_id)
                    except Exception as exc:
                        logger.exception("Analysis failed for %s", sample.sample_id)
                        observations = []
                        status, error = "analysis_error", str(exc)
                        self.analyzer.reset()
                        self.tracker.reset()
                record = self.writer.record(sample, observations, args.expected_participants,
                                            missing_seats, status, error, warnings)
                logger.info("Sample %d | %s | detected %d | classified %d | unknown %d | %s",
                            record["sampleIndex"] + 1, sample.sample_id, len(observations),
                            record["classifiedPersonCount"], record["unknownPersonCount"], status)
                if sample.frame is not None and (args.annotated_dir or args.display):
                    annotated = draw_annotations(sample.frame, observations, record, args.seats)
                    if args.annotated_dir:
                        path = Path(args.annotated_dir) / f"sample_{record['sampleIndex'] + 1:06d}.jpg"
                        write_image(path, annotated)
                    if args.display:
                        cv2.imshow("Classroom Analyzer", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            run_status, stop_reason = "interrupted", "preview_closed"
                            break
                if args.max_samples and len(self.writer.samples) >= args.max_samples:
                    stop_reason = "max_samples"
                    break
            if not self.writer.samples:
                raise ValueError("The input produced no samples")
            if any(s["status"] not in ("ok", "no_detections") for s in self.writer.samples):
                if run_status == "completed":
                    run_status = "completed_with_errors"
                exit_code = 1
        except KeyboardInterrupt:
            run_status, stop_reason = "interrupted", "keyboard_interrupt"
            logger.info("Stopping and saving available results.")
        except Exception as exc:
            run_status, stop_reason, run_error = "failed", "error", str(exc)
            exit_code = 1
            logger.exception("Session failed")
        finally:
            cleanup = []
            if stream is not None:
                cleanup.append(stream.close)
            if self.analyzer is not None:
                cleanup.append(self.analyzer.close)
            if args.display:
                cleanup.append(cv2.destroyAllWindows)
            for close in cleanup:
                try:
                    close()
                except Exception as exc:
                    logger.exception("Resource cleanup failed")
                    run_status, stop_reason, exit_code = "failed", "cleanup_error", 1
                    run_error = str(exc) if run_error is None else run_error + "; " + str(exc)
            self.writer.complete(run_status, stop_reason, run_error)

        if not args.no_charts:
            try:
                from visualization import generate_visualizations
                for path in generate_visualizations(args.output_json):
                    logger.info("Chart: %s", path)
            except Exception as exc:
                logger.exception("Results are saved, but chart generation failed")
                self.writer.metadata["artifactErrors"] = [{"stage": "charts", "error": str(exc)}]
                if self.writer.run_status == "completed":
                    self.writer.run_status = "completed_with_errors"
                self.writer.checkpoint()
                exit_code = 1
        logger.info("Results: %s", args.output_json)
        return exit_code


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    logger.info("Classroom Activity Signal Analyzer %s | %s | %s", __version__, args.source, args.analysis_mode)
    return ClassroomSession(args).run()


if __name__ == "__main__":
    sys.exit(main())
