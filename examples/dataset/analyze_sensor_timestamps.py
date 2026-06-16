#!/usr/bin/env python

"""Analyze per-sensor timestamps recorded in a LeRobot dataset.

Example:
    python examples/dataset/analyze_sensor_timestamps.py \
        --repo-id justintiensmith/DepthTest \
        --root /vol/dissolve/justin/datasets/DepthTest \
        --episode 0 \
        --reference camera.middle.perf_counter_s \
        --csv-output /vol/dissolve/justin/outputs/timestamp_offsets_episode0.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets import LeRobotDataset

TIMESTAMP_KEY = "observation.sensor_timestamps"


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def parse_episodes(values: list[int] | None) -> list[int] | None:
    return values if values else None


def select_reference(names: list[str], requested: str | None) -> str:
    if requested is not None:
        if requested not in names:
            raise ValueError(f"Reference '{requested}' not found. Available names: {names}")
        return requested

    preferred = "camera.middle.perf_counter_s"
    if preferred in names:
        return preferred

    camera_names = [name for name in names if name.startswith("camera.")]
    if camera_names:
        return camera_names[0]

    return names[0]


def summarize_ms(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: float("nan") for key in ["mean", "std", "min", "p05", "median", "p95", "max"]}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def format_stats(stats: dict[str, float]) -> str:
    return (
        f"mean={stats['mean']:+7.2f}  std={stats['std']:6.2f}  "
        f"min={stats['min']:+7.2f}  p05={stats['p05']:+7.2f}  "
        f"median={stats['median']:+7.2f}  p95={stats['p95']:+7.2f}  max={stats['max']:+7.2f}"
    )


def load_timestamp_matrix(dataset: LeRobotDataset, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = []
    frame_indices = []
    episode_indices = []
    raw_hf_dataset = dataset.hf_dataset.with_format(None)

    for idx in range(len(raw_hf_dataset)):
        # LeRobotDataset installs hf_transform_to_torch on its HF dataset. That transform can
        # downcast large absolute perf_counter values, which quantizes millisecond offsets.
        # with_format(None) gives us the stored values directly for timestamp diagnostics.
        sample = raw_hf_dataset[idx]
        timestamps.append(to_numpy(sample[key]).astype(np.float64))
        frame_indices.append(int(to_numpy(sample["frame_index"]).item()))
        episode_indices.append(int(to_numpy(sample["episode_index"]).item()))

    return np.stack(timestamps), np.asarray(frame_indices), np.asarray(episode_indices)


def write_csv(
    path: Path,
    names: list[str],
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    episode_indices: np.ndarray,
    reference_name: str,
) -> None:
    reference_idx = names.index(reference_name)
    reference = timestamps[:, reference_idx]
    path.parent.mkdir(parents=True, exist_ok=True)

    # CSV contains both absolute perf_counter values and per-frame offsets from the chosen reference.
    # Absolute values are useful only within the same recording process; offsets are the main diagnostic.
    fieldnames = ["episode_index", "frame_index", "reference"]
    fieldnames += [f"timestamp.{name}" for name in names]
    fieldnames += [f"offset_ms.{name}" for name in names]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx in range(timestamps.shape[0]):
            row: dict[str, int | float | str] = {
                "episode_index": int(episode_indices[row_idx]),
                "frame_index": int(frame_indices[row_idx]),
                "reference": reference_name,
            }
            for sensor_idx, name in enumerate(names):
                row[f"timestamp.{name}"] = float(timestamps[row_idx, sensor_idx])
                row[f"offset_ms.{name}"] = float((timestamps[row_idx, sensor_idx] - reference[row_idx]) * 1e3)
            writer.writerow(row)


def print_offset_summary(
    names: list[str], timestamps: np.ndarray, reference_name: str, outlier_threshold_ms: float
) -> None:
    reference_idx = names.index(reference_name)
    reference = timestamps[:, reference_idx]

    print("\nOffset to reference, in milliseconds")
    print(f"Reference: {reference_name}")
    print("Sign convention: positive means sensor timestamp is later than reference.\n")

    for sensor_idx, name in enumerate(names):
        offsets_ms = (timestamps[:, sensor_idx] - reference) * 1e3
        stats = summarize_ms(offsets_ms)
        outlier_fraction = float(np.mean(np.abs(offsets_ms) > outlier_threshold_ms))
        print(
            f"{name:40s} {format_stats(stats)}  outliers>|{outlier_threshold_ms:.1f}ms|={outlier_fraction:.2%}"
        )


def print_interval_summary(names: list[str], timestamps: np.ndarray, fps: float) -> None:
    expected_interval_ms = 1e3 / fps if fps else float("nan")
    print("\nPer-sensor frame-to-frame timestamp intervals, in milliseconds")
    print(f"Dataset FPS: {fps:g}; expected interval: {expected_interval_ms:.2f} ms\n")

    for sensor_idx, name in enumerate(names):
        intervals_ms = np.diff(timestamps[:, sensor_idx]) * 1e3
        stats = summarize_ms(intervals_ms)
        nonmonotonic = float(np.mean(intervals_ms < -1e-6)) if intervals_ms.size else 0.0
        repeated = float(np.mean(np.isclose(intervals_ms, 0.0))) if intervals_ms.size else 0.0
        print(f"{name:40s} {format_stats(stats)}  nonmonotonic={nonmonotonic:.2%}  repeated={repeated:.2%}")


def print_recommendation(names: list[str], timestamps: np.ndarray, reference_name: str, fps: float) -> None:
    reference_idx = names.index(reference_name)
    reference = timestamps[:, reference_idx]
    frame_ms = 1e3 / fps if fps else 33.333

    # Use the worst p95 absolute offset as a conservative frame-level alignment summary.
    # At 30 FPS, one frame is 33.3 ms; offsets below a small fraction of that are usually acceptable.
    worst_p95_abs = 0.0
    worst_name = None
    for sensor_idx, name in enumerate(names):
        if name == reference_name:
            continue
        offsets_ms = (timestamps[:, sensor_idx] - reference) * 1e3
        p95_abs = float(np.percentile(np.abs(offsets_ms), 95))
        if p95_abs > worst_p95_abs:
            worst_p95_abs = p95_abs
            worst_name = name

    print("\nPractical recommendation")
    if worst_name is None:
        print("Only one timestamp source is present; no cross-sensor alignment can be evaluated.")
    elif worst_p95_abs < 0.25 * frame_ms:
        print(
            f"Looks well aligned for {fps:g} FPS: worst p95 absolute offset is "
            f"{worst_p95_abs:.2f} ms for {worst_name}. Post-processing is probably unnecessary."
        )
    elif worst_p95_abs < 0.75 * frame_ms:
        print(
            f"Moderate offsets: worst p95 absolute offset is {worst_p95_abs:.2f} ms for {worst_name}. "
            "For ordinary visual policy training this may be acceptable, but timestamp-based interpolation could help."
        )
    else:
        print(
            f"Large offsets: worst p95 absolute offset is {worst_p95_abs:.2f} ms for {worst_name}. "
            "Post-processing alignment or recording-loop optimization is recommended before collecting a large dataset."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-id", required=True, help="Dataset repo id")
    parser.add_argument("--root", type=Path, default=None, help="Local dataset root")
    parser.add_argument(
        "--episode", type=int, action="append", default=None, help="Episode index. Can be repeated."
    )
    parser.add_argument(
        "--timestamp-key", default=TIMESTAMP_KEY, help=f"Timestamp feature key. Default: {TIMESTAMP_KEY}"
    )
    parser.add_argument(
        "--reference", default=None, help="Reference timestamp name. Defaults to camera.middle if present."
    )
    parser.add_argument("--csv-output", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument(
        "--outlier-threshold-ms", type=float, default=16.7, help="Offset threshold for outlier counts"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = parse_episodes(args.episode)
    dataset = LeRobotDataset(args.repo_id, root=args.root, episodes=episodes)

    if args.timestamp_key not in dataset.features:
        raise KeyError(
            f"Missing '{args.timestamp_key}'. Did you record with --robot.observe_sensor_timestamps=true?"
        )

    names = list(dataset.features[args.timestamp_key].get("names", []))
    if not names:
        raise ValueError(
            f"Feature '{args.timestamp_key}' has no names metadata: {dataset.features[args.timestamp_key]}"
        )

    timestamps, frame_indices, episode_indices = load_timestamp_matrix(dataset, args.timestamp_key)
    reference_name = select_reference(names, args.reference)

    print(f"Loaded {len(dataset)} frames from repo_id={args.repo_id}")
    print(f"Episodes: {sorted(set(episode_indices.tolist()))}")
    print(f"Timestamp key: {args.timestamp_key}")
    print(f"Timestamp columns: {names}")

    print_offset_summary(names, timestamps, reference_name, args.outlier_threshold_ms)
    print_interval_summary(names, timestamps, dataset.fps)
    print_recommendation(names, timestamps, reference_name, dataset.fps)

    if args.csv_output is not None:
        write_csv(args.csv_output, names, timestamps, frame_indices, episode_indices, reference_name)
        print(f"\nWrote per-frame timestamp offsets: {args.csv_output}")


if __name__ == "__main__":
    main()
