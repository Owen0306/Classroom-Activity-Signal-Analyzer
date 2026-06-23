import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_visualizations(input_json: str = "classroom_results.json",
                            output_dir: str | None = None) -> list[Path]:
    """Generate aggregate charts from a classroom analysis JSON file."""
    input_path = Path(input_json)
    chart_dir = Path(output_dir) if output_dir else input_path.parent
    chart_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    windows = data.get("windows", [])
    session = data.get("session", {})
    df_windows = pd.DataFrame(windows)
    output_paths: list[Path] = []

    if not df_windows.empty and {"timestamp", "observableActivityIndex"}.issubset(df_windows.columns):
        path = chart_dir / "class_activity_timeline.png"
        plt.figure(figsize=(12, 6))
        plt.plot(
            df_windows["timestamp"],
            df_windows["observableActivityIndex"],
            color="coral",
            linewidth=2,
        )
        plt.fill_between(
            df_windows["timestamp"],
            df_windows["observableActivityIndex"],
            alpha=0.3,
            color="coral",
        )
        plt.title("Classroom Observable Activity Index Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Observable Activity Index")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        output_paths.append(path)

    signal_distribution = session.get("signalDistribution", {})
    if signal_distribution:
        path = chart_dir / "observable_signal_distribution.png"
        df_signals = pd.DataFrame({
            "signal": list(signal_distribution.keys()),
            "count": list(signal_distribution.values()),
        }).sort_values("count", ascending=False)
        plt.figure(figsize=(11, 6))
        sns.barplot(
            data=df_signals,
            x="signal",
            y="count",
            hue="signal",
            palette="viridis",
            legend=False,
        )
        plt.title("Aggregate Observable Signal Distribution")
        plt.xlabel("Observable Signal")
        plt.ylabel("Count")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        output_paths.append(path)

    if not df_windows.empty and {"timestamp", "reviewFlagTotal"}.issubset(df_windows.columns):
        path = chart_dir / "review_flags_over_time.png"
        plt.figure(figsize=(12, 6))
        plt.plot(
            df_windows["timestamp"],
            df_windows["reviewFlagTotal"],
            color="firebrick",
            linewidth=2,
        )
        plt.title("Review Flag Events Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Review Flag Count")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        output_paths.append(path)

    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Generate aggregate classroom charts.")
    parser.add_argument("--input-json", default="classroom_results.json")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    paths = generate_visualizations(args.input_json, args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
