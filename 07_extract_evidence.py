#!/usr/bin/env python3
"""
Stage 7: Human-interpretable kinematic evidence visualization.

For one video, this script:

1. Loads Stage-3 kinematics.
2. Loads the trained Stage-4 region-specific authentic-motion GMM.
3. Recomputes per-landmark and per-frame kinematic inconsistency scores
   exactly as in Stage 5.
4. Selects the strongest suspicious centered observations.
5. Maps each kinematic observation back to source center frame t.
6. Loads the Stage-1 68-point landmarks.
7. Extracts source frames (t-1, t, t+1).
8. Identifies the facial region contributing the strongest inconsistency.
9. Produces:
      - evidence.csv
      - annotated center-frame PNGs
      - three-frame triptych PNGs
      - per-frame score archive (.npz)

IMPORTANT:
Each score S_t represents a centered three-frame observation
(t-1, t, t+1). It should not be interpreted as independent
single-frame fake classification.

Example
-------
python 07_extract_evidence.py \
    --video_path FFPP/manipulated_sequences/Deepfakes/c23/videos/000_003.mp4 \
    --landmarks_path landmarks/deepfakes/000_003.npy \
    --kinematics_path kinematics/deepfakes/000_003.npz \
    --model_path models/gmm_region_combined.pkl \
    --out_dir evidence/000_003 \
    --representation combined \
    --top_n 6 \
    --min_separation 15 \
    --threshold 2.35
"""

import argparse
import csv
import pickle
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Colors (OpenCV BGR)
# ============================================================

COLOR_LANDMARK = (255, 255, 255)       # white
COLOR_REGION = (0, 165, 255)           # orange
COLOR_CENTER = (0, 255, 255)           # yellow
COLOR_BOX = (0, 165, 255)              # orange
COLOR_TEXT = (255, 255, 255)           # white
COLOR_BG = (20, 20, 20)

REGION_DISPLAY_NAMES = {
    "upper_face_periorbital": "Periorbital",
    "nasal": "Nasal",
    "perioral": "Perioral",
    "contour": "Facial Contour",
}


# ============================================================
# Stage-5-compatible scoring
# ============================================================

def build_feature(v, a, representation):
    """
    Construct the requested kinematic representation.

    v, a: (T-2, 68, 2)
    """
    if representation == "first":
        return v

    if representation == "second":
        return a

    if representation == "combined":
        return np.concatenate([v, a], axis=-1)

    raise ValueError(f"Unknown representation: {representation}")


def region_landmark_scores(model_bundle, feat):
    """
    Reproduce Stage-5 landmark-level scoring.

    Returns
    -------
    scores : ndarray, shape (T_kin, 68)
        Negative log-likelihood for each landmark under the
        authentic-motion GMM assigned to its facial region.
    """
    T, K, D = feat.shape

    scores = np.full((T, K), np.nan, dtype=np.float32)

    for region_name, region_info in model_bundle["regions"].items():

        indices = np.asarray(region_info["indices"], dtype=int)

        gmm = region_info["model"]
        clip_value = region_info["clip_value"]

        region_feat = feat[:, indices, :]

        flat = region_feat.reshape(-1, D)

        valid = ~np.isnan(flat).any(axis=1)

        region_scores = np.full(
            flat.shape[0],
            np.nan,
            dtype=np.float32
        )

        if valid.any():

            s = -gmm.score_samples(flat[valid])

            # Same clipping rule used by Stage 5.
            s = np.clip(
                s,
                a_min=None,
                a_max=clip_value
            )

            region_scores[valid] = s.astype(np.float32)

        scores[:, indices] = region_scores.reshape(
            T,
            len(indices)
        )

    return scores


def compute_frame_scores(s_kt, p_top=25.0):
    """
    Pool landmark-level scores into frame/center-observation scores.

    Same top-p pooling logic as Stage 5.
    """
    T, K = s_kt.shape

    frame_scores = np.full(T, np.nan, dtype=np.float32)

    for t in range(T):

        row = s_kt[t]
        row = row[~np.isnan(row)]

        if row.size == 0:
            continue

        n_top = max(
            1,
            int(np.ceil(row.size * p_top / 100.0))
        )

        frame_scores[t] = np.mean(
            np.sort(row)[-n_top:]
        )

    return frame_scores


# ============================================================
# Region evidence
# ============================================================

def compute_region_scores_for_frame(
    landmark_scores,
    region_map
):
    """
    Mean inconsistency score within each facial region
    for one centered observation.
    """
    output = {}

    for region_name, indices in region_map.items():

        indices = np.asarray(indices, dtype=int)

        vals = landmark_scores[indices]
        vals = vals[~np.isnan(vals)]

        if vals.size == 0:
            output[region_name] = np.nan
        else:
            output[region_name] = float(np.mean(vals))

    return output


def strongest_region(landmark_scores, region_map):
    """
    Return highest-scoring facial region.
    """
    scores = compute_region_scores_for_frame(
        landmark_scores,
        region_map
    )

    valid = {
        k: v for k, v in scores.items()
        if np.isfinite(v)
    }

    if not valid:
        return None, np.nan, scores

    region = max(valid, key=valid.get)

    return region, valid[region], scores


# ============================================================
# Suspicious-frame selection
# ============================================================

def select_top_centers(
    frame_scores,
    top_n=6,
    threshold=None,
    min_separation=15
):
    """
    Select highest-scoring centered observations while preventing
    nearly identical neighboring frames from dominating the result.

    Parameters
    ----------
    frame_scores:
        Score for each Stage-3 kinematic center.

    threshold:
        Optional S_t threshold selected from authentic validation data.

    min_separation:
        Minimum distance, in source frames, between selected centers.
    """

    valid_indices = np.flatnonzero(
        np.isfinite(frame_scores)
    )

    if threshold is not None:
        valid_indices = valid_indices[
            frame_scores[valid_indices] > threshold
        ]

    if len(valid_indices) == 0:
        return []

    # Highest scores first.
    order = valid_indices[
        np.argsort(frame_scores[valid_indices])[::-1]
    ]

    selected = []

    for kin_idx in order:

        # Stage-3 index i corresponds to source center frame i+1.
        center_frame = int(kin_idx + 1)

        too_close = any(
            abs(center_frame - previous_center)
            < min_separation
            for previous_center in [
                int(i + 1) for i in selected
            ]
        )

        if too_close:
            continue

        selected.append(int(kin_idx))

        if len(selected) >= top_n:
            break

    return selected


# ============================================================
# Video frame access
# ============================================================

def get_video_info(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise IOError(
            f"Could not open video: {video_path}"
        )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    cap.release()

    return fps, count, width, height


def read_frame(video_path, frame_index):
    """
    Read a 0-based source frame.
    """

    cap = cv2.VideoCapture(str(video_path))

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_index)
    )

    ok, frame = cap.read()

    cap.release()

    if not ok:
        return None

    return frame


# ============================================================
# Landmark visualization
# ============================================================

def valid_landmark(pt):

    if pt is None:
        return False

    return np.all(np.isfinite(pt))


def draw_landmarks(
    frame,
    landmarks,
    region_indices=None
):
    """
    Draw all landmarks in white.

    Landmarks belonging to the strongest facial region are
    enlarged and highlighted in orange.
    """

    out = frame.copy()

    region_set = (
        set(region_indices)
        if region_indices is not None
        else set()
    )

    for i, point in enumerate(landmarks):

        if not valid_landmark(point):
            continue

        x = int(round(point[0]))
        y = int(round(point[1]))

        if i in region_set:

            cv2.circle(
                out,
                (x, y),
                4,
                COLOR_REGION,
                -1,
                lineType=cv2.LINE_AA
            )

            cv2.circle(
                out,
                (x, y),
                6,
                COLOR_REGION,
                1,
                lineType=cv2.LINE_AA
            )

        else:

            cv2.circle(
                out,
                (x, y),
                2,
                COLOR_LANDMARK,
                -1,
                lineType=cv2.LINE_AA
            )

    return out


def region_bounding_box(
    landmarks,
    indices,
    padding=18
):
    """
    Bounding box around highlighted region.
    """

    pts = landmarks[np.asarray(indices, dtype=int)]

    valid = np.isfinite(pts).all(axis=1)

    pts = pts[valid]

    if len(pts) == 0:
        return None

    x1 = int(np.min(pts[:, 0])) - padding
    y1 = int(np.min(pts[:, 1])) - padding

    x2 = int(np.max(pts[:, 0])) + padding
    y2 = int(np.max(pts[:, 1])) + padding

    return x1, y1, x2, y2


def draw_region_box(
    frame,
    landmarks,
    region_indices
):

    box = region_bounding_box(
        landmarks,
        region_indices
    )

    if box is None:
        return frame

    out = frame.copy()

    x1, y1, x2, y2 = box

    h, w = out.shape[:2]

    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)

    x2 = np.clip(x2, 0, w - 1)
    y2 = np.clip(y2, 0, h - 1)

    cv2.rectangle(
        out,
        (x1, y1),
        (x2, y2),
        COLOR_BOX,
        2,
        lineType=cv2.LINE_AA
    )

    return out


# ============================================================
# Publication annotation
# ============================================================

def add_header(
    frame,
    center_frame,
    timestamp,
    score,
    region_name,
    region_score,
    threshold=None
):
    """
    Add compact publication-style header above frame.
    """

    h, w = frame.shape[:2]

    header_h = 92

    canvas = np.full(
        (h + header_h, w, 3),
        COLOR_BG,
        dtype=np.uint8
    )

    canvas[header_h:] = frame

    display_region = REGION_DISPLAY_NAMES.get(
        region_name,
        str(region_name)
    )

    line1 = (
        f"Center frame {center_frame}   "
        f"Time {timestamp:.2f} s"
    )

    line2 = (
        f"Kinematic inconsistency "
        f"S_t = {score:.3f}"
    )

    line3 = (
        f"Strongest region: {display_region} "
        f"({region_score:.3f})"
    )

    if threshold is not None:
        line2 += f"   threshold = {threshold:.3f}"

    cv2.putText(
        canvas,
        line1,
        (18, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        COLOR_TEXT,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        line2,
        (18, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        COLOR_TEXT,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        line3,
        (18, 81),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        COLOR_REGION,
        1,
        cv2.LINE_AA
    )

    return canvas


# ============================================================
# Triptych
# ============================================================

def add_frame_label(
    frame,
    label,
    active=False
):

    out = frame.copy()

    overlay = out.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (out.shape[1], 45),
        COLOR_BG,
        -1
    )

    out = cv2.addWeighted(
        overlay,
        0.72,
        out,
        0.28,
        0
    )

    color = (
        COLOR_CENTER if active
        else COLOR_TEXT
    )

    cv2.putText(
        out,
        label,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA
    )

    return out


def make_triptych(
    previous,
    center,
    following,
    landmarks_prev,
    landmarks_center,
    landmarks_next,
    region_indices,
    center_frame,
    timestamp,
    score,
    region_name,
    threshold=None
):
    """
    Publication-ready visualization of the centered observation:

               t-1        t        t+1
    """

    frames = [
        previous.copy(),
        center.copy(),
        following.copy()
    ]

    lms = [
        landmarks_prev,
        landmarks_center,
        landmarks_next
    ]

    labels = [
        "t - 1",
        "t (center)",
        "t + 1"
    ]

    processed = []

    target_h = 520

    for i, (frame, lm, label) in enumerate(
        zip(frames, lms, labels)
    ):

        scale = target_h / frame.shape[0]

        new_w = int(
            frame.shape[1] * scale
        )

        frame = cv2.resize(
            frame,
            (new_w, target_h),
            interpolation=cv2.INTER_AREA
        )

        lm_scaled = lm.copy()

        lm_scaled[:, 0] *= scale
        lm_scaled[:, 1] *= scale

        # All three frames show trajectories/landmarks,
        # but strongest-region emphasis is most important
        # at the center observation.
        if i == 1:

            frame = draw_landmarks(
                frame,
                lm_scaled,
                region_indices
            )

            frame = draw_region_box(
                frame,
                lm_scaled,
                region_indices
            )

        else:

            frame = draw_landmarks(
                frame,
                lm_scaled,
                None
            )

        frame = add_frame_label(
            frame,
            label,
            active=(i == 1)
        )

        processed.append(frame)

    # Standardize width if video frame dimensions vary.
    min_h = min(
        img.shape[0] for img in processed
    )

    processed = [
        img[:min_h]
        for img in processed
    ]

    body = cv2.hconcat(processed)

    header_h = 84

    canvas = np.full(
        (
            body.shape[0] + header_h,
            body.shape[1],
            3
        ),
        COLOR_BG,
        dtype=np.uint8
    )

    canvas[header_h:] = body

    region_display = REGION_DISPLAY_NAMES.get(
        region_name,
        region_name
    )

    title = (
        f"Centered kinematic observation | "
        f"frame {center_frame} | "
        f"{timestamp:.2f} s | "
        f"S_t={score:.3f}"
    )

    if threshold is not None:
        title += f" | threshold={threshold:.3f}"

    subtitle = (
        f"Highest-scoring facial region: "
        f"{region_display}"
    )

    cv2.putText(
        canvas,
        title,
        (20, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        COLOR_TEXT,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        subtitle,
        (20, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        COLOR_REGION,
        1,
        cv2.LINE_AA
    )

    return canvas


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract publication-ready WHEN/WHERE "
            "kinematic evidence from a video."
        )
    )

    parser.add_argument(
        "--video_path",
        required=True,
        help="Path to original source video."
    )

    parser.add_argument(
        "--landmarks_path",
        required=True,
        help="Stage-1 .npy landmark file."
    )

    parser.add_argument(
        "--kinematics_path",
        required=True,
        help="Stage-3 .npz kinematics file."
    )

    parser.add_argument(
        "--model_path",
        required=True,
        help="Stage-4 region-specific GMM .pkl file."
    )

    parser.add_argument(
        "--out_dir",
        required=True
    )

    parser.add_argument(
        "--representation",
        choices=[
            "first",
            "second",
            "combined"
        ],
        default="combined"
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=6,
        help=(
            "Maximum number of suspicious "
            "centered observations to save."
        )
    )

    parser.add_argument(
        "--p_top",
        type=float,
        default=25.0,
        help=(
            "Top percentage of landmark scores "
            "used to compute S_t."
        )
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional validation-derived tau_val. "
            "If supplied, only S_t > tau_val are eligible."
        )
    )

    parser.add_argument(
        "--min_separation",
        type=int,
        default=15,
        help=(
            "Minimum source-frame distance "
            "between selected evidence centers."
        )
    )

    parser.add_argument(
        "--save_triptych",
        action="store_true",
        help="Save t-1, t, t+1 publication panels."
    )

    args = parser.parse_args()

    video_path = Path(args.video_path)
    landmark_path = Path(args.landmarks_path)
    kinematics_path = Path(args.kinematics_path)
    model_path = Path(args.model_path)

    out_dir = Path(args.out_dir)

    center_dir = out_dir / "center_frames"
    triptych_dir = out_dir / "triptychs"

    center_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if args.save_triptych:
        triptych_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    with open(model_path, "rb") as f:
        model_bundle = pickle.load(f)

    if model_bundle.get("model_type") != "region_gmm":
        raise ValueError(
            "Expected a Stage-4 region-specific GMM bundle."
        )

    model_representation = model_bundle.get(
        "representation"
    )

    if model_representation != args.representation:
        raise ValueError(
            f"Model representation is "
            f"{model_representation}, but requested "
            f"{args.representation}."
        )

    region_map = {
        name: info["indices"]
        for name, info
        in model_bundle["regions"].items()
    }

    # --------------------------------------------------------
    # Load Stage-1 landmarks
    # --------------------------------------------------------

    landmarks = np.load(
        landmark_path
    ).astype(np.float32)

    if (
        landmarks.ndim != 3
        or landmarks.shape[1:] != (68, 2)
    ):
        raise ValueError(
            f"Expected landmarks shape (T,68,2), "
            f"got {landmarks.shape}"
        )

    # --------------------------------------------------------
    # Load Stage-3 kinematics
    # --------------------------------------------------------

    kin = np.load(
        kinematics_path
    )

    v = kin["v"]
    a = kin["a"]

    if "fps" in kin:
        kin_fps = float(
            kin["fps"]
        )
    else:
        kin_fps = None

    feat = build_feature(
        v,
        a,
        args.representation
    )

    # --------------------------------------------------------
    # Score observations
    # --------------------------------------------------------

    print("Computing landmark-level inconsistency scores...")

    s_kt = region_landmark_scores(
        model_bundle,
        feat
    )

    frame_scores = compute_frame_scores(
        s_kt,
        p_top=args.p_top
    )

    # --------------------------------------------------------
    # Video info
    # --------------------------------------------------------

    video_fps, video_frames, width, height = get_video_info(
        video_path
    )

    fps = (
        video_fps
        if video_fps > 0
        else kin_fps
    )

    if fps is None or fps <= 0:
        raise RuntimeError(
            "Could not determine video FPS."
        )

    print()
    print("=" * 70)
    print("STAGE 7 — KINEMATIC EVIDENCE EXTRACTION")
    print("=" * 70)
    print(f"Video              : {video_path}")
    print(f"Source frames      : {video_frames}")
    print(f"Landmark frames    : {len(landmarks)}")
    print(f"Kinematic centers  : {len(frame_scores)}")
    print(f"FPS                : {fps:.3f}")
    print(f"Representation     : {args.representation}")
    print(f"Top-p landmarks    : {args.p_top:.1f}%")
    print(f"Threshold          : {args.threshold}")
    print("=" * 70)

    # --------------------------------------------------------
    # Save all scores
    # --------------------------------------------------------

    source_center_indices = (
        np.arange(len(frame_scores)) + 1
    )

    timestamps = (
        source_center_indices / fps
    )

    np.savez_compressed(
        out_dir / "per_frame_scores.npz",
        frame_scores=frame_scores,
        landmark_scores=s_kt,
        source_center_indices=source_center_indices,
        timestamps=timestamps,
        fps=np.float32(fps),
        representation=args.representation
    )

    # --------------------------------------------------------
    # Select strongest observations
    # --------------------------------------------------------

    selected = select_top_centers(
        frame_scores,
        top_n=args.top_n,
        threshold=args.threshold,
        min_separation=args.min_separation
    )

    if len(selected) == 0:

        print(
            "\nNo centered observations satisfied "
            "the selection criterion."
        )

        print(
            "Try omitting --threshold or verify tau_val."
        )

        return

    print(
        f"\nSelected {len(selected)} evidence observations."
    )

    # --------------------------------------------------------
    # Extract evidence
    # --------------------------------------------------------

    evidence_rows = []

    for rank, kin_idx in enumerate(
        selected,
        start=1
    ):

        # Stage-3:
        # kinematic index 0 -> source frame 1
        # in 0-based frame indexing.
        center_idx = kin_idx + 1

        previous_idx = center_idx - 1
        next_idx = center_idx + 1

        timestamp = center_idx / fps

        score = float(
            frame_scores[kin_idx]
        )

        landmark_score_row = s_kt[
            kin_idx
        ]

        (
            region_name,
            region_score,
            all_region_scores
        ) = strongest_region(
            landmark_score_row,
            region_map
        )

        if region_name is None:
            print(
                f"[WARN] Cannot determine region "
                f"for center {center_idx}"
            )
            continue

        region_indices = region_map[
            region_name
        ]

        # ---------------------------------------------
        # Validate corresponding Stage-1 frames
        # ---------------------------------------------

        if next_idx >= len(landmarks):

            print(
                f"[WARN] Center {center_idx}: "
                f"landmark triplet out of range."
            )

            continue

        lm_previous = landmarks[
            previous_idx
        ]

        lm_center = landmarks[
            center_idx
        ]

        lm_next = landmarks[
            next_idx
        ]

        # ---------------------------------------------
        # Read source frames
        # ---------------------------------------------

        frame_previous = read_frame(
            video_path,
            previous_idx
        )

        frame_center = read_frame(
            video_path,
            center_idx
        )

        frame_next = read_frame(
            video_path,
            next_idx
        )

        if (
            frame_previous is None
            or frame_center is None
            or frame_next is None
        ):

            print(
                f"[WARN] Could not read triplet "
                f"centered at {center_idx}."
            )

            continue

        # ---------------------------------------------
        # Annotated center frame
        # ---------------------------------------------

        center_visual = draw_landmarks(
            frame_center,
            lm_center,
            region_indices
        )

        center_visual = draw_region_box(
            center_visual,
            lm_center,
            region_indices
        )

        center_visual = add_header(
            center_visual,
            center_frame=center_idx,
            timestamp=timestamp,
            score=score,
            region_name=region_name,
            region_score=region_score,
            threshold=args.threshold
        )

        center_filename = (
            f"{rank:02d}_"
            f"center_{center_idx:06d}_"
            f"score_{score:.3f}.png"
        )

        cv2.imwrite(
            str(
                center_dir /
                center_filename
            ),
            center_visual,
            [
                cv2.IMWRITE_PNG_COMPRESSION,
                3
            ]
        )

        # ---------------------------------------------
        # Triptych
        # ---------------------------------------------

        triptych_filename = ""

        if args.save_triptych:

            triptych = make_triptych(
                frame_previous,
                frame_center,
                frame_next,
                lm_previous,
                lm_center,
                lm_next,
                region_indices,
                center_frame=center_idx,
                timestamp=timestamp,
                score=score,
                region_name=region_name,
                threshold=args.threshold
            )

            triptych_filename = (
                f"{rank:02d}_"
                f"triptych_center_"
                f"{center_idx:06d}.png"
            )

            cv2.imwrite(
                str(
                    triptych_dir /
                    triptych_filename
                ),
                triptych,
                [
                    cv2.IMWRITE_PNG_COMPRESSION,
                    3
                ]
            )

        # ---------------------------------------------
        # CSV record
        # ---------------------------------------------

        evidence_rows.append({

            "rank":
                rank,

            "kinematic_index_0based":
                kin_idx,

            "source_center_frame_0based":
                center_idx,

            "source_center_frame_1based":
                center_idx + 1,

            "previous_frame_0based":
                previous_idx,

            "next_frame_0based":
                next_idx,

            "timestamp_sec":
                timestamp,

            "S_t":
                score,

            "threshold":
                (
                    args.threshold
                    if args.threshold is not None
                    else ""
                ),

            "threshold_exceeded":
                (
                    score > args.threshold
                    if args.threshold is not None
                    else ""
                ),

            "highest_scoring_region":
                REGION_DISPLAY_NAMES.get(
                    region_name,
                    region_name
                ),

            "region_key":
                region_name,

            "highest_region_score":
                region_score,

            "periorbital_score":
                all_region_scores.get(
                    "upper_face_periorbital",
                    np.nan
                ),

            "nasal_score":
                all_region_scores.get(
                    "nasal",
                    np.nan
                ),

            "perioral_score":
                all_region_scores.get(
                    "perioral",
                    np.nan
                ),

            "contour_score":
                all_region_scores.get(
                    "contour",
                    np.nan
                ),

            "center_png":
                str(
                    Path("center_frames")
                    / center_filename
                ),

            "triptych_png":
                (
                    str(
                        Path("triptychs")
                        / triptych_filename
                    )
                    if triptych_filename
                    else ""
                )
        })

        print(
            f"#{rank:02d} | "
            f"center={center_idx:5d} | "
            f"time={timestamp:7.3f}s | "
            f"S_t={score:8.4f} | "
            f"region="
            f"{REGION_DISPLAY_NAMES.get(region_name, region_name)}"
        )

    # --------------------------------------------------------
    # evidence.csv
    # --------------------------------------------------------

    csv_path = (
        out_dir /
        "evidence.csv"
    )

    if evidence_rows:

        fieldnames = list(
            evidence_rows[0].keys()
        )

        with open(
            csv_path,
            "w",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                evidence_rows
            )

    print()
    print("=" * 70)
    print("STAGE 7 COMPLETE")
    print("=" * 70)
    print(f"Evidence CSV  : {csv_path}")
    print(f"Center PNGs   : {center_dir}")

    if args.save_triptych:
        print(
            f"Triptych PNGs : {triptych_dir}"
        )

    print(
        f"Score archive : "
        f"{out_dir / 'per_frame_scores.npz'}"
    )

    print("=" * 70)


if __name__ == "__main__":
        main()