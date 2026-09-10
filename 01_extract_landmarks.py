"""
Stage 1: Landmark extraction.

For every video in an input directory, detects the largest face in each
frame and regresses 68 facial landmarks with dlib. Saves one .npy file per
video containing an array of shape (T, 68, 2) in raw pixel coordinates,
plus a companion .json with per-frame detection success flags so that
downstream stages can identify dropped/undetected frames.

Usage:
    python 01_extract_landmarks.py \
        --video_dir /path/to/videos \
        --out_dir /path/to/output/landmarks \
        --predictor_path /path/to/shape_predictor_68_face_landmarks.dat \
        --label real   # or 'fake' -- stored in metadata, not used for tracking
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import dlib
import numpy as np
from tqdm import tqdm


def shape_to_np(shape, dtype="float32"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def largest_face(dets):
    if len(dets) == 0:
        return None
    return max(dets, key=lambda d: d.width() * d.height())


def process_video(video_path, detector, predictor, upsample=1):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    landmarks = []
    detected_flags = []
    prev_rect = None  # reuse last known face box to stabilize detection

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, upsample)
        rect = largest_face(dets)

        if rect is None and prev_rect is not None:
            # fall back to previous frame's box, slightly padded, to avoid
            # a hard drop on a single bad frame (e.g. transient motion blur)
            pad = 20
            x1 = max(prev_rect.left() - pad, 0)
            y1 = max(prev_rect.top() - pad, 0)
            x2 = min(prev_rect.right() + pad, gray.shape[1])
            y2 = min(prev_rect.bottom() + pad, gray.shape[0])
            sub_dets = detector(gray[y1:y2, x1:x2], upsample)
            rect = largest_face(sub_dets)
            if rect is not None:
                rect = dlib.rectangle(
                    rect.left() + x1, rect.top() + y1,
                    rect.right() + x1, rect.bottom() + y1,
                )

        if rect is None:
            landmarks.append(np.full((68, 2), np.nan, dtype="float32"))
            detected_flags.append(False)
            continue

        shape = predictor(gray, rect)
        coords = shape_to_np(shape)
        landmarks.append(coords)
        detected_flags.append(True)
        prev_rect = rect

    cap.release()
    return np.stack(landmarks, axis=0), detected_flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--predictor_path", required=True,
                     help="Path to shape_predictor_68_face_landmarks.dat")
    ap.add_argument("--label", default="unknown",
                     help="Metadata label, e.g. 'real', 'deepfakes', 'face2face'")
    ap.add_argument("--extensions", nargs="+", default=[".mp4", ".avi", ".mov"])
    ap.add_argument("--upsample", type=int, default=1,
                     help="Dlib detector upsample factor; raise for small/far faces")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.predictor_path)

    video_paths = [
        p for p in Path(args.video_dir).rglob("*")
        if p.suffix.lower() in args.extensions
    ]
    print(f"Found {len(video_paths)} videos in {args.video_dir}")

    for vp in tqdm(video_paths, desc="Extracting landmarks"):
        out_npy = out_dir / f"{vp.stem}.npy"
        out_json = out_dir / f"{vp.stem}.meta.json"
        if out_npy.exists():
            continue  # resumable: skip already-processed videos

        try:
            landmarks, flags = process_video(vp, detector, predictor, args.upsample)
        except Exception as e:
            print(f"[WARN] Failed on {vp}: {e}")
            continue

        detection_rate = float(np.mean(flags))
        if detection_rate < 0.5:
            print(f"[WARN] Low detection rate ({detection_rate:.2f}) for {vp.name}")

        np.save(out_npy, landmarks)
        with open(out_json, "w") as f:
            json.dump({
                "source_video": str(vp),
                "label": args.label,
                "num_frames": int(landmarks.shape[0]),
                "detection_rate": detection_rate,
                "detected_flags": flags,
            }, f, indent=2)


if __name__ == "__main__":
    main()
