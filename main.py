"""Command-line entry point for classroom activity analysis."""

import argparse
import logging
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

from analyzer import ParticipantAnalyzer, ParticipantObservation
from behavior import SIGNAL_META
from result_writer import LocalResultWriter
from tracker import PersonTracker


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


class ClassroomSession:
    """Runs one complete aggregate analysis session."""

    def __init__(self, args):
        self.args = args
        self.tracker = PersonTracker(
            model_path=args.yolo_model,
            conf=args.confidence,
            max_disappeared=20,
        )
        self.analyzer = ParticipantAnalyzer()
        self.writer = LocalResultWriter(args.output_json)
        self.frame_index: int = 0
        self.start_time: float = 0.0
        self.last_capture_time: float | None = None

    def run(self):
        cap = self._open_source()
        if not cap.isOpened():
            logger.error("Could not open video source.")
            sys.exit(1)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Video source opened: {width}x{height}")
        logger.info(f"Sampling interval: {self.args.interval}s")
        logger.info(f"Output JSON: {self.args.output_json}")
        logger.info("Press 'q' to exit when preview is enabled.")

        self.start_time = time.time()
        self.last_capture_time = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if self.args.source == "video":
                        logger.info("Video file finished.")
                        break
                    time.sleep(0.05)
                    continue

                timestamp = self._source_timestamp(cap, time.time())

                # Skip frames until the next sampling point.
                if (self.last_capture_time is not None and
                        timestamp - self.last_capture_time < self.args.interval):
                    if self.args.display:
                        cv2.imshow("Classroom Analyzer", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    continue

                self.last_capture_time = timestamp
                tracked = self.tracker.update(frame)

                if not tracked:
                    self.frame_index += 1
                    continue

                observations = []
                for track_id, bbox in tracked:
                    observation = self.analyzer.analyze(
                        frame,
                        bbox,
                        track_id,
                        timestamp=timestamp,
                    )
                    if observation is not None:
                        observations.append(observation)

                if not observations:
                    self.frame_index += 1
                    continue

                aggregate = _aggregate_window(observations)
                self.writer.record_window(
                    window_index=self.frame_index,
                    timestamp=timestamp,
                    observed_participant_count=len(observations),
                    observable_activity_index=aggregate["observableActivityIndex"],
                    signal_counts=aggregate["signalCounts"],
                    signal_ratios=aggregate["signalRatios"],
                    review_flag_counts=aggregate["reviewFlagCounts"],
                )

                logger.info(
                    f"Window {self.frame_index:>4d} | "
                    f"Observed participants {len(observations)} | "
                    f"Activity index {aggregate['observableActivityIndex']:.0f} | "
                    f"Review flags {aggregate['reviewFlagTotal']}"
                )

                if self.args.display:
                    annotated = _draw(frame, observations, self.frame_index,
                                      aggregate["observableActivityIndex"])
                    cv2.imshow("Classroom Analyzer", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                self.frame_index += 1

        except KeyboardInterrupt:
            logger.info("Analysis interrupted.")
        finally:
            self._finalize(cap)

    def _open_source(self) -> cv2.VideoCapture:
        if self.args.source == "camera":
            logger.info(f"Opening camera ID={self.args.camera_id}")
            return cv2.VideoCapture(self.args.camera_id)
        if self.args.source == "video":
            logger.info(f"Opening video file: {self.args.video_path}")
            return cv2.VideoCapture(self.args.video_path)
        if self.args.source == "rtsp":
            logger.info(f"Opening RTSP stream: {self.args.rtsp_url}")
            return cv2.VideoCapture(self.args.rtsp_url)
        logger.error(f"Unknown video source type: {self.args.source}")
        sys.exit(1)

    def _source_timestamp(self, cap: cv2.VideoCapture, current_time: float) -> float:
        if self.args.source == "video":
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_msec > 0:
                return pos_msec / 1000.0

            fps = cap.get(cv2.CAP_PROP_FPS)
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if fps and fps > 0:
                return max(0.0, (pos_frame - 1) / fps)

        return current_time - self.start_time

    def _finalize(self, cap: cv2.VideoCapture):
        self.writer.complete()
        cap.release()
        if self.args.display:
            cv2.destroyAllWindows()
        self.analyzer.close()

        elapsed = time.time() - self.start_time
        stats = self.writer.stats
        try:
            from visualization import generate_visualizations

            chart_paths = generate_visualizations(stats["output_path"])
            if chart_paths:
                logger.info(
                    "Visualization charts generated: "
                    + ", ".join(str(path) for path in chart_paths)
                )
            else:
                logger.warning("No visualization charts were generated.")
        except Exception as exc:
            logger.exception(f"Failed to generate visualization charts: {exc}")

        logger.info(
            f"Analysis complete | elapsed {elapsed:.1f}s | "
            f"windows {stats['windows']} | output {stats['output_path']}"
        )


def _aggregate_window(observations: list[ParticipantObservation]) -> dict:
    signal_counts = defaultdict(int)
    review_flag_counts = defaultdict(int)
    activity_indexes = []

    for observation in observations:
        signal_counts[observation.observable_signal] += 1
        activity_indexes.append(observation.activity_index)
        if observation.review_flag:
            review_flag_counts[observation.observable_signal] += 1

    total = max(len(observations), 1)
    signal_counts = dict(signal_counts)
    review_flag_counts = dict(review_flag_counts)
    signal_ratios = {
        signal: round(count / total, 4)
        for signal, count in signal_counts.items()
    }
    activity_index = sum(activity_indexes) / len(activity_indexes)

    return {
        "observableActivityIndex": activity_index,
        "signalCounts": signal_counts,
        "signalRatios": signal_ratios,
        "reviewFlagCounts": review_flag_counts,
        "reviewFlagTotal": sum(review_flag_counts.values()),
    }


def _draw(frame: np.ndarray, observations: list[ParticipantObservation],
          frame_idx: int, activity_index: float) -> np.ndarray:
    out = frame.copy()
    for observation in observations:
        x1, y1, x2, y2 = observation.bbox
        color = _color_for_index(observation.activity_index)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        meta = SIGNAL_META.get(observation.observable_signal, {})
        label_text = (
            f"{observation.activity_index:.0f}%  "
            f"{meta.get('label', observation.observable_signal)}"
        )
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label_text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        if observation.review_flag:
            cv2.putText(out, "REVIEW", (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)

    info = (
        f"Window {frame_idx}  |  Observed: {len(observations)}  |  "
        f"Activity Index: {activity_index:.0f}%"
    )
    cv2.rectangle(out, (0, 0), (len(info) * 9 + 10, 32), (20, 20, 20), -1)
    cv2.putText(out, info, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    return out


def _color_for_index(activity_index: float) -> tuple:
    if activity_index >= 70:
        return (0, 200, 80)
    if activity_index >= 40:
        return (0, 180, 255)
    return (0, 60, 230)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classroom activity signal analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --source video --video-path class.mp4 --no-display
  python main.py --source rtsp --rtsp-url rtsp://192.168.1.10:554/stream --interval 2
        """,
    )

    parser.add_argument("--output-json", default="classroom_results.json",
                        help="Local aggregate result JSON file")
    parser.add_argument("--source", choices=["camera", "video", "rtsp"], default="camera")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device ID")
    parser.add_argument("--video-path", type=str, default="", help="Video file path")
    parser.add_argument("--rtsp-url", type=str, default="", help="RTSP stream URL")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between analyzed windows")
    parser.add_argument("--confidence", type=float, default=0.15,
                        help="YOLO detection confidence threshold")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt",
                        help="YOLO model path")
    parser.add_argument("--display", dest="display", action="store_true", default=True,
                        help="Show the live annotated preview")
    parser.add_argument("--no-display", dest="display", action="store_false",
                        help="Disable the preview window")

    args = parser.parse_args()

    if args.source == "video" and not args.video_path:
        parser.error("--source video requires --video-path")
    if args.source == "rtsp" and not args.rtsp_url:
        parser.error("--source rtsp requires --rtsp-url")

    return args


def main():
    args = parse_args()

    print("=" * 58)
    print("  Classroom Activity Signal Analyzer")
    print("=" * 58)
    print(f"  Source      : {args.source}")
    print(f"  Output JSON : {args.output_json}")
    print(f"  YOLO model  : {args.yolo_model}")
    print(f"  Interval    : {args.interval}s")
    print(f"  Confidence  : {args.confidence}")
    print(f"  Preview     : {'on' if args.display else 'off'}")
    print("=" * 58)

    session = ClassroomSession(args)
    session.run()


if __name__ == "__main__":
    main()
