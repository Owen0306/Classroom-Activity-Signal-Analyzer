# Classroom Activity Signal Analyzer

Classroom Activity Signal Analyzer is a local Python prototype for analyzing
observable activity signals in classroom video. It detects people with YOLOv8,
uses MediaPipe pose and face landmarks to infer simple posture/activity
patterns, and writes aggregate session summaries to JSON. At the end of each
run, it automatically generates charts for quick review.

The project is designed for classroom analytics experiments, teaching demos,
and research prototypes where computer vision processing should stay on the
local machine.

## Data Use Declaration

The tool makes session-level summaries without identification of specific students. 
Temporary trackers are used in the process of one analysis session in order not to 
double-count observations. Those are not personal identifiers, are not persistent 
across sessions and get erased after summary.

Personal identification is not used by this tool and therefore it should not be used 
to assess academic performance, attention level, emotions, motivation 
and risk of misbehavior.

## Responsible Use

This is a classroom analytics prototype and should not be used for any student evaluations.

Do not use the tool to calculate grades, monitor attendance, make decisions about disciplinary actions, perform psychological evaluations, detect emotions, make decisions about admissions or any other high stakes decision making about individuals.

Make sure that meaningful informed consent is received before you record and analyze videos from classroom; follow your school policies on collecting and analyzing data. Although local processing decreases some risks of data leaking, still appropriate measures are needed for data collecting, access control and data deleting.

## Features

- Webcam, video file, and RTSP stream input.
- YOLOv8 person detection.
- Temporary IoU tracking for within-session de-duplication.
- MediaPipe pose and face-landmark analysis.
- Rule-based observable signal labels:
  - forward-facing posture
  - writing posture
  - raised hand
  - head-down posture
  - side-facing posture
  - possible peer-facing interaction
  - sustained head-down posture
- Aggregate classroom activity signal trends.
- Per-window counts and ratios for observable signals.
- Session-level statistics.
- Local JSON output.
- Automatic aggregate chart generation after analysis.

## How It Works

```text
Video source
    -> YOLOv8 person detection
    -> temporary IoU tracking
    -> MediaPipe pose and face-landmark analysis
    -> observable signal classification
    -> window-level aggregation
    -> JSON summaries and aggregate charts
```

## Project Structure

```text
ClassroomAnalyzer/
├── main.py              # CLI entry point and analysis loop
├── tracker.py           # YOLO detection and simple temporary tracking
├── analyzer.py          # Pose, gaze, signal, and activity-index analysis
├── behavior.py          # Rule-based observable signal detection
├── result_writer.py     # Aggregate JSON result writer
├── visualization.py     # Aggregate chart generation
├── requirements.txt     # Python dependencies
└── README.md
```


## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

The default YOLO model path is `yolov8n.pt`. If the model file is not already
present, Ultralytics may download it on first use.

## Usage

Run with the default webcam:

```bash
python main.py
```

Analyze a video file without opening a preview window:

```bash
python main.py --source video --video-path classroom.mp4 --no-display
```

Analyze an RTSP stream:

```bash
python main.py --source rtsp --rtsp-url rtsp://192.168.1.10:554/stream
```

Customize sampling interval and output path:

```bash
python main.py --interval 2.0 --output-json results/classroom_results.json
```

## CLI Options

| Option | Default | Description |
| --- | --- | --- |
| `--output-json` | `classroom_results.json` | Aggregate result file |
| `--source` | `camera` | Input source: `camera`, `video`, or `rtsp` |
| `--camera-id` | `0` | Webcam device ID |
| `--video-path` | empty | Video file path |
| `--rtsp-url` | empty | RTSP stream URL |
| `--interval` | `1.0` | Seconds between analyzed windows |
| `--confidence` | `0.15` | YOLO detection confidence threshold |
| `--yolo-model` | `yolov8n.pt` | YOLO model path |
| `--no-display` | off | Disable the OpenCV preview window |

## Outputs

The JSON output contains aggregate data only:

- `summary`: session basics such as window count, duration, observed participant
  count statistics, mean observable activity index, and review flag total.
- `session`: aggregate observable signal distribution, ratios, dominant
  observable signal, and review flag distribution.
- `windows`: per-time-window observed participant count, observable activity
  index, signal counts, signal ratios, and review flag counts.

The output does not include persistent participant profiles, `Student A/B/C`
labels, individual timelines, per-participant activity scores, emotion fields,
or snapshot images.

After analysis, these aggregate charts are generated next to the JSON file:

- `class_activity_timeline.png`
- `observable_signal_distribution.png`
- `review_flags_over_time.png`

Charts can also be regenerated manually:

```bash
python visualization.py --input-json classroom_results.json
```

## Observable Signal Rules

| Observable signal | Confirmation delay | Review flag |
| --- | --- | --- |
| forward-facing posture | immediate | no |
| writing posture | immediate | no |
| raised hand | immediate | no |
| head-down posture | 10s | yes |
| side-facing posture | 10s | yes |
| possible peer-facing interaction | 8s | yes |
| sustained head-down posture | 20s | yes |

The current model is rule-based. It is useful for experimentation, but it should
not be treated as a reliable measure of learning, attention, intent, emotion, or
individual behavior.

## How Signal Detection Works

Signals are inferred from MediaPipe landmarks and simple timing rules:

- `forward-facing posture`: default state when no other observable pattern is detected.
- `writing posture`: both wrists are visible and positioned low in the frame.
- `raised hand`: at least one wrist is visibly above the matching shoulder.
- `head-down posture`: estimated head pose points downward for long enough.
- `side-facing posture`: head is turned left or right with a stable eye-line pattern.
- `possible peer-facing interaction`: head is turned to the side and shoulders suggest side-facing posture.
- `sustained head-down posture`: nose is close to or below shoulder height for a sustained period.

Short-lived signals are filtered with confirmation delays to reduce noise. These
rules describe observable pose patterns only; they do not determine why a person
is behaving that way.

## Privacy

- All computer vision processing runs locally.
- Raw frames are not uploaded by this application.
- Output data is aggregate and does not contain persistent participant profiles.
- Review local privacy, consent, and school policies before using classroom video.

## Limitations

- The tracker is a simple IoU matcher, so temporary IDs may change during
  occlusion or crowded scenes.
- Observable signal labels are approximate and depend on camera angle, lighting,
  and body visibility.
- The observable activity index is a heuristic, not a validated educational
  metric.
- Review flags are prompts for aggregate review, not disciplinary alerts.
- Long live streams may produce large JSON files.

## Roadmap Ideas

- Add unit tests for signal rules and aggregate result writing.
- Replace IoU tracking with a stronger tracker such as ByteTrack.
- Add a small dashboard for reviewing saved aggregate sessions.
- Add configurable signal thresholds.
- Export CSV summaries.

## Contributing

Issues and pull requests are welcome. Good first contributions include
documentation improvements, test coverage, tracking improvements, and safer
handling for long-running streams.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
