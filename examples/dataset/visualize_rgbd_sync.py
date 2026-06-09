#!/usr/bin/env python

"""Visualize RGB/depth pairs from a LeRobot dataset and estimate temporal lag.

Example:
    python examples/dataset/visualize_rgbd_sync.py \
        --repo-id justintiensmith/SP_Relational_Placement \
        --root /vol/dissolve/justin/datasets/SP_Relational_Placement \
        --camera middle \
        --episode 0 \
        --output /vol/dissolve/justin/outputs/rgbd_sync_episode0.mp4 \
        --max-depth-mm 2000
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.datasets import LeRobotDataset


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def tensor_image_to_uint8_hwc(image: Any) -> np.ndarray:
    image = to_numpy(image)

    if image.ndim == 3 and image.shape[0] in {1, 3, 4}:
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3:
        raise ValueError(f"Expected RGB image with 3 dimensions, got shape={image.shape}")

    if image.dtype.kind == "f":
        image = np.clip(image, 0.0, 1.0) * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]

    return image


def depth_to_uint16_hw(depth: Any) -> np.ndarray:
    depth = np.squeeze(to_numpy(depth))
    if depth.ndim != 2:
        raise ValueError(f"Expected depth map with 2 dimensions, got shape={depth.shape}")
    return depth.astype(np.uint16, copy=False)


def colorize_depth(depth: np.ndarray, min_depth_mm: int, max_depth_mm: int) -> np.ndarray:
    valid = depth > 0
    clipped = np.clip(depth, min_depth_mm, max_depth_mm)
    scaled = ((clipped - min_depth_mm) / max(max_depth_mm - min_depth_mm, 1) * 255.0).astype(np.uint8)
    scaled[~valid] = 0

    colormap = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET
    colorized_bgr = cv2.applyColorMap(scaled, colormap)
    colorized_bgr[~valid] = 0
    return cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)


def mean_abs_change(current: np.ndarray, previous: np.ndarray | None) -> float:
    if previous is None:
        return 0.0
    return float(np.mean(np.abs(current.astype(np.float32) - previous.astype(np.float32))))


def prepare_rgb_motion_image(rgb: np.ndarray, size: tuple[int, int] = (160, 120)) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def prepare_depth_motion_image(
    depth: np.ndarray, max_depth_mm: int, size: tuple[int, int] = (160, 120)
) -> np.ndarray:
    valid = depth > 0
    clipped = np.clip(depth, 0, max_depth_mm).astype(np.float32)
    normalized = clipped / max(max_depth_mm, 1) * 255.0
    normalized[~valid] = 0.0
    return cv2.resize(normalized.astype(np.uint8), size, interpolation=cv2.INTER_AREA)


def estimate_lag(rgb_motion: list[float], depth_motion: list[float], max_lag: int) -> tuple[int, float]:
    rgb = np.asarray(rgb_motion, dtype=np.float32)
    depth = np.asarray(depth_motion, dtype=np.float32)

    if len(rgb) < 3 or len(depth) < 3:
        return 0, float("nan")

    best_lag = 0
    best_corr = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            rgb_slice = rgb[:-lag]
            depth_slice = depth[lag:]
        elif lag < 0:
            rgb_slice = rgb[-lag:]
            depth_slice = depth[:lag]
        else:
            rgb_slice = rgb
            depth_slice = depth

        if len(rgb_slice) < 3:
            continue

        rgb_centered = rgb_slice - rgb_slice.mean()
        depth_centered = depth_slice - depth_slice.mean()
        denom = np.linalg.norm(rgb_centered) * np.linalg.norm(depth_centered)
        if denom == 0:
            continue

        corr = float(np.dot(rgb_centered, depth_centered) / denom)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    return best_lag, best_corr


def resolve_keys(camera: str, image_key: str | None, depth_key: str | None) -> tuple[str, str]:
    if image_key is None:
        image_key = camera if camera.startswith("observation.images.") else f"observation.images.{camera}"
    if depth_key is None:
        if image_key.startswith("observation.images."):
            camera_name = image_key.removeprefix("observation.images.")
        else:
            camera_name = camera
        depth_key = f"observation.depths.{camera_name}"
    return image_key, depth_key


def draw_panel_label(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(output, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def make_montage(rgb: np.ndarray, depth_color: np.ndarray, text: str) -> np.ndarray:
    if depth_color.shape[:2] != rgb.shape[:2]:
        depth_color = cv2.resize(depth_color, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    rgb_panel = draw_panel_label(rgb, "RGB")
    depth_panel = draw_panel_label(depth_color, "Depth colorized")
    montage = np.concatenate([rgb_panel, depth_panel], axis=1)

    text_bar = np.zeros((44, montage.shape[1], 3), dtype=np.uint8)
    cv2.putText(text_bar, text, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return np.concatenate([montage, text_bar], axis=0)


def valid_depth_stats(depth: np.ndarray) -> tuple[float, float, float]:
    valid = depth[depth > 0]
    if valid.size == 0:
        return 0.0, float("nan"), float("nan")
    return float(valid.size / depth.size), float(np.median(valid)), float(np.percentile(valid, 99))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo-id", required=True, help="Dataset repo id, e.g. justintiensmith/SP_Relational_Placement"
    )
    parser.add_argument(
        "--root", type=Path, default=None, help="Local dataset root. Use this for offline/local datasets."
    )
    parser.add_argument("--camera", default="middle", help="Camera name or full image key. Default: middle")
    parser.add_argument("--image-key", default=None, help="Override RGB key, e.g. observation.images.middle")
    parser.add_argument(
        "--depth-key", default=None, help="Override depth key, e.g. observation.depths.middle"
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index to visualize. Default: 0")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path")
    parser.add_argument(
        "--csv-output", type=Path, default=None, help="Optional CSV path for per-frame diagnostics"
    )
    parser.add_argument("--fps", type=float, default=None, help="Output video FPS. Defaults to dataset FPS.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame. Default: 1")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after this many exported frames.")
    parser.add_argument(
        "--min-depth-mm", type=int, default=250, help="Depth color scale minimum in millimeters."
    )
    parser.add_argument(
        "--max-depth-mm", type=int, default=2000, help="Depth color scale maximum in millimeters."
    )
    parser.add_argument(
        "--max-lag", type=int, default=5, help="Largest lag, in sampled frames, for motion correlation."
    )
    parser.add_argument("--codec", default="mp4v", help="OpenCV output codec. Default: mp4v")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_key, depth_key = resolve_keys(args.camera, args.image_key, args.depth_key)

    dataset = LeRobotDataset(args.repo_id, root=args.root, episodes=[args.episode])
    missing = [key for key in (image_key, depth_key) if key not in dataset.features]
    if missing:
        raise KeyError(f"Missing required dataset keys: {missing}. Available keys: {list(dataset.features)}")

    output_fps = args.fps if args.fps is not None else dataset.fps / args.stride
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_output = args.csv_output or args.output.with_suffix(".csv")

    writer: cv2.VideoWriter | None = None
    previous_rgb_motion = None
    previous_depth_motion = None
    rgb_motion_values: list[float] = []
    depth_motion_values: list[float] = []
    csv_rows: list[dict[str, float | int]] = []

    exported = 0
    for local_idx in range(0, len(dataset), args.stride):
        if args.max_frames is not None and exported >= args.max_frames:
            break

        sample = dataset[local_idx]
        rgb = tensor_image_to_uint8_hwc(sample[image_key])
        depth = depth_to_uint16_hw(sample[depth_key])
        depth_color = colorize_depth(depth, args.min_depth_mm, args.max_depth_mm)

        rgb_motion_image = prepare_rgb_motion_image(rgb)
        depth_motion_image = prepare_depth_motion_image(depth, args.max_depth_mm)
        rgb_change = mean_abs_change(rgb_motion_image, previous_rgb_motion)
        depth_change = mean_abs_change(depth_motion_image, previous_depth_motion)
        previous_rgb_motion = rgb_motion_image
        previous_depth_motion = depth_motion_image
        rgb_motion_values.append(rgb_change)
        depth_motion_values.append(depth_change)

        valid_fraction, median_depth, p99_depth = valid_depth_stats(depth)
        frame_index = int(to_numpy(sample["frame_index"]).item())
        episode_index = int(to_numpy(sample["episode_index"]).item())
        text = (
            f"episode={episode_index} frame={frame_index} "
            f"valid_depth={valid_fraction:.2%} median={median_depth:.0f}mm p99={p99_depth:.0f}mm "
            f"rgb_change={rgb_change:.2f} depth_change={depth_change:.2f}"
        )
        montage = make_montage(rgb, depth_color, text)

        if writer is None:
            height, width = montage.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*args.codec)
            writer = cv2.VideoWriter(str(args.output), fourcc, output_fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {args.output}")

        writer.write(cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
        csv_rows.append(
            {
                "episode_index": episode_index,
                "frame_index": frame_index,
                "local_index": local_idx,
                "rgb_change": rgb_change,
                "depth_change": depth_change,
                "valid_depth_fraction": valid_fraction,
                "median_depth_mm": median_depth,
                "p99_depth_mm": p99_depth,
            }
        )
        exported += 1

    if writer is not None:
        writer.release()

    with csv_output.open("w", newline="") as f:
        fieldnames = [
            "episode_index",
            "frame_index",
            "local_index",
            "rgb_change",
            "depth_change",
            "valid_depth_fraction",
            "median_depth_mm",
            "p99_depth_mm",
        ]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(csv_rows)

    best_lag, best_corr = estimate_lag(rgb_motion_values, depth_motion_values, args.max_lag)
    lag_seconds = best_lag / output_fps if output_fps else float("nan")

    print(f"Wrote video: {args.output}")
    print(f"Wrote frame diagnostics: {csv_output}")
    print(f"Exported frames: {exported}")
    print(f"Estimated best lag: {best_lag:+d} sampled frames ({lag_seconds:+.3f}s), corr={best_corr:.3f}")
    print("Interpretation: positive lag means depth motion appears later than RGB; zero is what you want.")


if __name__ == "__main__":
    main()
