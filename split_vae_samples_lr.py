from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
TARGET_OUTPUT_HEIGHT = 720


@dataclass
class DividerHint:
    fixed_x: int | None = None
    ratio: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split normal VAE sample videos into left/right road videos using notebook road divider logic."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("vae_normal_samples"),
        help="Input video folder or a single video file (default: vae_normal_samples).",
    )
    parser.add_argument(
        "--output-left",
        type=Path,
        default=Path("vae_split_samples/left"),
        help="Output folder for left-road videos.",
    )
    parser.add_argument(
        "--output-right",
        type=Path,
        default=Path("vae_split_samples/right"),
        help="Output folder for right-road videos.",
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("YOLO.ipynb"),
        help="Notebook path used to read road_divider_x variable logic (default: YOLO.ipynb).",
    )
    parser.add_argument(
        "--divider-x",
        type=int,
        default=None,
        help="Explicit divider x in pixels. If set, this overrides notebook parsing.",
    )
    parser.add_argument(
        "--fallback-ratio",
        type=float,
        default=1.88,
        help="Fallback divider ratio. Divider becomes int(width / ratio).",
    )
    parser.add_argument(
        "--buffer-px",
        type=int,
        default=0,
        help="Optional dead-zone around divider to exclude center pixels from both sides.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search videos when input is a folder.",
    )
    return parser.parse_args()


def parse_divider_hint_from_notebook(notebook_path: Path) -> DividerHint:
    hint = DividerHint()
    if not notebook_path.exists():
        return hint

    try:
        nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    except Exception:
        return hint

    cells = nb.get("cells", [])
    pattern_fixed = re.compile(r"road_divider_x\s*=\s*(\d+)")
    pattern_ratio = re.compile(r"road_divider_x\s*=\s*int\(\s*w\s*/\s*([0-9]*\.?[0-9]+)\s*\)")

    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))

        ratio_match = pattern_ratio.search(src)
        if ratio_match:
            try:
                hint.ratio = float(ratio_match.group(1))
            except ValueError:
                pass

        fixed_match = pattern_fixed.search(src)
        if fixed_match:
            try:
                hint.fixed_x = int(fixed_match.group(1))
            except ValueError:
                pass

    return hint


def list_videos(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in VIDEO_SUFFIXES else []

    if not path.exists() or not path.is_dir():
        return []

    if recursive:
        videos = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
    else:
        videos = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]

    videos.sort()
    return videos


def compute_divider_x(width: int, explicit_divider: int | None, hint: DividerHint, fallback_ratio: float) -> int:
    if explicit_divider is not None:
        return clamp_divider(explicit_divider, width)

    if hint.ratio is not None and hint.ratio > 0:
        return clamp_divider(int(width / hint.ratio), width)

    if hint.fixed_x is not None:
        return clamp_divider(hint.fixed_x, width)

    if fallback_ratio <= 0:
        fallback_ratio = 1.88
    return clamp_divider(int(width / fallback_ratio), width)


def clamp_divider(divider_x: int, width: int) -> int:
    return max(1, min(width - 1, int(divider_x)))


def pick_fourcc(suffix: str) -> int:
    suffix = suffix.lower()
    if suffix == ".avi":
        return cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter_fourcc(*"avc1")


def _resize_for_output(frame: cv2.typing.MatLike, out_w: int, out_h: int) -> cv2.typing.MatLike:
    src_h, _ = frame.shape[:2]
    interp = cv2.INTER_AREA if src_h > out_h else cv2.INTER_LINEAR
    return cv2.resize(frame, (out_w, out_h), interpolation=interp)


def split_one_video(
    video_path: Path,
    rel_path: Path,
    out_left_root: Path,
    out_right_root: Path,
    divider_x: int,
    buffer_px: int,
) -> tuple[Path, Path, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = float(cap.get(cv2.CAP_PROP_FPS))
    original_fps = original_fps if original_fps > 0 else 30.0
    target_fps = 15
    skip_interval = max(1, int(original_fps / target_fps))

    divider_x = clamp_divider(divider_x, width)
    buffer_px = max(0, int(buffer_px))

    left_x2 = max(1, min(width - 1, divider_x - buffer_px))
    right_x1 = min(width - 1, max(0, divider_x + buffer_px))
    left_w = left_x2
    right_w = width - right_x1
    target_h = TARGET_OUTPUT_HEIGHT
    left_out_w = max(2, int(round(left_w * target_h / max(height, 1))))
    right_out_w = max(2, int(round(right_w * target_h / max(height, 1))))

    if left_w < 8 or right_w < 8:
        cap.release()
        raise RuntimeError(
            f"Invalid split for '{video_path.name}': width={width}, divider={divider_x}, buffer={buffer_px}."
        )

    out_left = out_left_root / rel_path.parent / f"{video_path.stem}_left{video_path.suffix}"
    out_right = out_right_root / rel_path.parent / f"{video_path.stem}_right{video_path.suffix}"
    out_left.parent.mkdir(parents=True, exist_ok=True)
    out_right.parent.mkdir(parents=True, exist_ok=True)

    fourcc = pick_fourcc(video_path.suffix)
    writer_left = cv2.VideoWriter(str(out_left), fourcc, target_fps, (left_out_w, target_h))
    writer_right = cv2.VideoWriter(str(out_right), fourcc, target_fps, (right_out_w, target_h))

    if (not writer_left.isOpened() or not writer_right.isOpened()) and video_path.suffix.lower() != ".avi":
        # Fallback codec for wider Windows compatibility when avc1 is unavailable.
        writer_left.release()
        writer_right.release()
        fallback = cv2.VideoWriter_fourcc(*"mp4v")
        writer_left = cv2.VideoWriter(str(out_left), fallback, target_fps, (left_out_w, target_h))
        writer_right = cv2.VideoWriter(str(out_right), fallback, target_fps, (right_out_w, target_h))

    if not writer_left.isOpened() or not writer_right.isOpened():
        cap.release()
        writer_left.release()
        writer_right.release()
        raise RuntimeError(f"Cannot open writer for outputs: {out_left} / {out_right}")

    try:
        frame_counter = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_counter % skip_interval == 0:
                left_frame = _resize_for_output(frame[:, :left_x2], left_out_w, target_h)
                right_frame = _resize_for_output(frame[:, right_x1:], right_out_w, target_h)
                writer_left.write(left_frame)
                writer_right.write(right_frame)
            frame_counter += 1
    finally:
        cap.release()
        writer_left.release()
        writer_right.release()

    return out_left, out_right, divider_x


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    out_left_root = args.output_left.resolve()
    out_right_root = args.output_right.resolve()
    notebook_path = args.notebook.resolve()

    videos = list_videos(input_path, recursive=args.recursive)
    if not videos:
        print(f"No video files found at: {input_path}")
        return 1

    hint = parse_divider_hint_from_notebook(notebook_path)
    source_note = "explicit --divider-x"
    if args.divider_x is None:
        if hint.ratio is not None:
            source_note = f"notebook road_divider_x ratio (w / {hint.ratio})"
        elif hint.fixed_x is not None:
            source_note = f"notebook road_divider_x fixed value ({hint.fixed_x})"
        else:
            source_note = f"fallback ratio (w / {args.fallback_ratio})"

    print(f"Input: {input_path}")
    print(f"Videos found: {len(videos)}")
    print(f"Divider source: {source_note}")
    print(f"Buffer px: {args.buffer_px}")
    print(f"Left output dir: {out_left_root}")
    print(f"Right output dir: {out_right_root}")

    failures = 0
    base_root = input_path if input_path.is_dir() else input_path.parent

    for i, video in enumerate(videos, start=1):
        rel_path = video.relative_to(base_root)

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"[{i}/{len(videos)}] ERROR open failed: {video}")
            failures += 1
            continue
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        divider_x = compute_divider_x(width, args.divider_x, hint, args.fallback_ratio)

        try:
            out_left, out_right, used_divider = split_one_video(
                video_path=video,
                rel_path=rel_path,
                out_left_root=out_left_root,
                out_right_root=out_right_root,
                divider_x=divider_x,
                buffer_px=args.buffer_px,
            )
            print(
                f"[{i}/{len(videos)}] OK {video.name} | divider_x={used_divider} -> "
                f"{out_left.name}, {out_right.name}"
            )
        except Exception as exc:
            print(f"[{i}/{len(videos)}] ERROR {video.name}: {exc}")
            failures += 1

    done = len(videos) - failures
    print(f"Completed. Success: {done}, Failed: {failures}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
