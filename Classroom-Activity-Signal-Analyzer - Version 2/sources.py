# Photo batches, incoming photo folders, and sampled capture sources.

from dataclasses import dataclass
from pathlib import Path
import re
import time

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Sample:
    sample_id: str
    source_path: str
    frame: np.ndarray | None
    timestamp: float | None = None
    error: str | None = None


def read_image(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if frame is None:
            return None, "Image could not be decoded"
        return frame, None
    except (OSError, cv2.error) as exc:
        return None, str(exc)


def write_image(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, frame)
    if not ok:
        raise OSError(f"Could not encode image: {path}")
    data.tofile(path)


def natural_key(path):
    return [int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", str(path))]


def image_paths(directory, recursive=False):
    root = Path(directory)
    paths = root.rglob("*") if recursive else root.iterdir()
    return sorted((p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                  key=lambda p: natural_key(p.relative_to(root)))


def photo_samples(args):
    if args.source == "image":
        path = Path(args.image_path)
        frame, error = read_image(path)
        yield Sample(path.name, str(path), frame, error=error)
        return
    root = Path(args.images_dir)
    seen, pending = {}, {}
    while True:
        paths = image_paths(root, args.recursive)
        for path in paths:
            signature = None
            if args.watch:
                try:
                    signature = (path.stat().st_size, path.stat().st_mtime_ns)
                except OSError:
                    continue
                if seen.get(path) == signature:
                    continue
                if pending.get(path) != signature:
                    pending[path] = signature
                    continue
            frame, error = read_image(path)
            if args.watch:
                try:
                    if (path.stat().st_size, path.stat().st_mtime_ns) != signature:
                        continue
                except OSError:
                    continue
            seen[path] = signature
            pending.pop(path, None)
            sample_id = path.relative_to(root).as_posix()
            if args.watch:
                sample_id += f"@{signature[1]}"
            yield Sample(sample_id, str(path), frame, error=error)
        if not args.watch:
            return
        time.sleep(args.poll_interval)


def capture_samples(args):
    if args.source == "video":
        target = args.video_path
    elif args.source == "camera":
        target = args.camera_id
    else:
        target = args.rtsp_url
    cap = cv2.VideoCapture(target)
    try:
        if not cap.isOpened():
            raise OSError("Could not open capture source")
        start = time.monotonic()
        next_sample, last_timestamp = 0.0, -1.0
        index, failed_reads = 0, 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.source == "video":
                    if total_frames > 0 and cap.get(cv2.CAP_PROP_POS_FRAMES) < total_frames - 1:
                        yield Sample(f"frame_{index:06d}", str(target), None,
                                     max(last_timestamp, 0), "Video decoding stopped before the end")
                    return
                failed_reads += 1
                yield Sample(f"capture_error_{index:06d}", args.source, None,
                             time.monotonic() - start, "Capture frame unavailable")
                index += 1
                if failed_reads >= args.max_read_failures:
                    raise OSError("Capture source exceeded the consecutive read failure limit")
                time.sleep(min(args.interval, 1.0))
                continue
            failed_reads = 0
            if args.source == "video":
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if not np.isfinite(timestamp) or timestamp <= last_timestamp:
                    if not np.isfinite(fps) or fps <= 0:
                        raise ValueError("Video has no usable timestamps or frame rate")
                    timestamp = max(0.0, (cap.get(cv2.CAP_PROP_POS_FRAMES) - 1) / fps)
                if timestamp <= last_timestamp:
                    raise ValueError("Video timestamps are not increasing")
            else:
                timestamp = time.monotonic() - start
            last_timestamp = timestamp
            if timestamp + 1e-6 < next_sample:
                continue
            yield Sample(f"frame_{index:06d}", str(target) if args.source == "video" else args.source,
                         frame, timestamp)
            index += 1
            next_sample = timestamp + args.interval
    finally:
        cap.release()
