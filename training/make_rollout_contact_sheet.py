#!/usr/bin/env python3
"""Create an evenly sampled, labeled contact sheet from a rollout video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 1.0
    if total < 2:
        raise SystemExit("Video has fewer than two frames.")

    start = max(0, min(args.start_frame, total - 1))
    end = total - 1 if args.end_frame is None else min(args.end_frame, total - 1)
    if end < start:
        raise SystemExit("end-frame must not precede start-frame.")
    indices = np.linspace(start, end, args.frames, dtype=int)
    sampled: list[np.ndarray] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            raise SystemExit(f"Could not decode frame {index}.")
        frame = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
        cv2.rectangle(frame, (0, 0), (180, 30), (0, 0, 0), thickness=-1)
        cv2.putText(
            frame,
            f"{index / fps:5.2f} s  frame {index}",
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        sampled.append(frame)
    capture.release()

    columns = max(1, args.columns)
    rows = (len(sampled) + columns - 1) // columns
    blank = np.zeros_like(sampled[0])
    sampled.extend([blank] * (rows * columns - len(sampled)))
    sheet_rows = [
        np.hstack(sampled[row * columns : (row + 1) * columns])
        for row in range(rows)
    ]
    sheet = np.vstack(sheet_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise SystemExit(f"Could not write {args.output}.")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
