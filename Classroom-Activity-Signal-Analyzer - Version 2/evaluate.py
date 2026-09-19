# Compare optional human seat labels against saved per-photo observations.

import argparse
import csv
import json
from pathlib import Path

from behavior import CATEGORIES, LEGACY_SIGNALS, UNKNOWN, NOT_DETECTED
from result_writer import atomic_json


def evaluate_results(results, labels):
    seats = results.get("metadata", {}).get("seatMap")
    if not seats:
        raise ValueError("Seat-based evaluation requires results produced with --seat-map")
    seat_ids = {str(s["id"]) for s in seats}
    samples = {s["sampleId"]: s for s in results["samples"]}
    if len(samples) != len(results["samples"]):
        raise ValueError("Results contain duplicate sample IDs")
    columns = (*CATEGORIES, UNKNOWN, NOT_DETECTED, "not_evaluated")
    confusion = {truth: {predicted: 0 for predicted in columns} for truth in CATEGORIES}
    seen, unmatched, rows = set(), [], []
    excluded = 0
    for label in labels:
        if not all(isinstance(label.get(key), str) and label[key].strip()
                   for key in ("sample_id", "seat_id", "signal")):
            raise ValueError("Every label requires nonempty sample_id, seat_id, and signal fields")
        sample_id, seat_id = label["sample_id"].strip(), str(label["seat_id"]).strip()
        key = (sample_id, seat_id)
        if key in seen:
            raise ValueError(f"Duplicate label: {key}")
        seen.add(key)
        if seat_id not in seat_ids:
            raise ValueError(f"Unknown seat ID: {seat_id}")
        raw_truth = label["signal"].strip()
        truth = LEGACY_SIGNALS.get(raw_truth, raw_truth)
        if truth == UNKNOWN:
            excluded += 1
            continue
        if truth not in CATEGORIES:
            raise ValueError(f"Unknown label category: {raw_truth}")
        if sample_id not in samples:
            unmatched.append({"sampleId": sample_id, "seatId": seat_id})
            continue
        sample = samples[sample_id]
        matching = [o for o in sample["observations"] if o["seatId"] == seat_id]
        if len(matching) > 1:
            raise ValueError(f"Multiple predictions assigned to {key}")
        if sample["status"] not in ("ok", "no_detections"):
            predicted = "not_evaluated"
        elif matching:
            predicted = matching[0]["observableSignal"]
        else:
            predicted = NOT_DETECTED
        confusion[truth][predicted] += 1
        rows.append({"sampleId": sample_id, "seatId": seat_id, "truth": truth,
                     "prediction": predicted, "correct": truth == predicted})
    correct = sum(r["correct"] for r in rows)
    per_class = {}
    for category in CATEGORIES:
        true_positive = confusion[category][category]
        support = sum(confusion[category].values())
        predicted_count = sum(confusion[c][category] for c in CATEGORIES)
        precision = true_positive / predicted_count if predicted_count else None
        recall = true_positive / support if support else None
        f1 = (2 * true_positive / (support + predicted_count)) if support + predicted_count else None
        per_class[category] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    total_slots = len(samples) * len(seat_ids)
    return {
        "evaluatedSeatCount": len(rows), "correctSeatCount": correct,
        "accuracy": correct / len(rows) if rows else None,
        "labelCoverage": len(rows) / total_slots if total_slots else None,
        "excludedUnknownLabelCount": excluded, "unmatchedLabels": unmatched,
        "confusionMatrix": confusion, "perClass": per_class, "comparisons": rows,
        "accuracyDenominator": "all matched human-labeled seats, including unknown, missed, and failed predictions",
    }


def main():
    parser = argparse.ArgumentParser(description="Compare saved predictions with manually checked seat labels")
    parser.add_argument("--results", required=True)
    parser.add_argument("--labels", required=True, help="CSV: sample_id,seat_id,signal")
    parser.add_argument("--output", default="output/evaluation.json")
    args = parser.parse_args()
    try:
        results = json.loads(Path(args.results).read_text(encoding="utf-8"))
        with Path(args.labels).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"sample_id", "seat_id", "signal"}.issubset(reader.fieldnames or []):
                raise ValueError("Labels require sample_id, seat_id, and signal columns")
            report = evaluate_results(results, list(reader))
        atomic_json(args.output, report)
        print(f"Evaluated seats: {report['evaluatedSeatCount']} | Accuracy: {report['accuracy']}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
