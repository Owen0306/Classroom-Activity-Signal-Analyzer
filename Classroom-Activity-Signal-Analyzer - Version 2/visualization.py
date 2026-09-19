# Charts for observed categories, missing evidence, and detection counts.

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from behavior import SIGNAL_META


def generate_visualizations(input_json="output/results.json", output_dir=None):
    input_path = Path(input_json)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    chart_dir = Path(output_dir) if output_dir else input_path.parent / (input_path.stem + "_charts")
    chart_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    distribution = data.get("session", {}).get("signalDistribution", {})
    if distribution:
        labels = [SIGNAL_META.get(key, {"label": key})["label"] for key in distribution]
        colors = [SIGNAL_META.get(key, {"color": "#94a3b8"})["color"] for key in distribution]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(labels, list(distribution.values()), color=colors)
        ax.set(xlabel="Person observations", title="Observable posture distribution")
        ax.invert_yaxis()
        fig.tight_layout()
        path = chart_dir / "posture_distribution.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    samples = data.get("samples", [])
    if samples:
        fig, ax = plt.subplots(figsize=(11, 5))
        x = [s["sampleIndex"] + 1 for s in samples]
        success = [s["status"] in ("ok", "no_detections") for s in samples]
        ax.plot(x, [s["detectedPersonCount"] if ok else float("nan") for s, ok in zip(samples, success)],
                label="Detected people", color="#2563eb")
        ax.plot(x, [s["unknownPersonCount"] if ok else float("nan") for s, ok in zip(samples, success)],
                label="Unable to determine / pending", color="#64748b")
        if any(s["missingParticipantCount"] is not None for s in samples):
            ax.plot(x, [s["missingParticipantCount"] if s["missingParticipantCount"] is not None else float("nan")
                        for s in samples], label="Missing against configured expectation", color="#d97706")
        failed_x = [v for v, ok in zip(x, success) if not ok]
        if failed_x:
            ax.scatter(failed_x, [0] * len(failed_x), marker="x", color="#dc2626", label="Failed input")
        ax.set(xlabel="Sample number", ylabel="People", title="Detection and evidence coverage")
        ax.set_ylim(bottom=0)
        ax.legend()
        fig.tight_layout()
        path = chart_dir / "detection_coverage.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Generate posture and coverage charts.")
    parser.add_argument("--input-json", default="output/results.json")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    for path in generate_visualizations(args.input_json, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
