"""
Stage 2: Normalization and temporal smoothing.

Implements the similarity-transform normalization (Section 3.1 / 4.1 of the
paper): for each frame, estimate a similarity transform from the outer
ocular canthi (landmarks 37, 46 in 1-indexed dlib scheme -> indices 36, 45
in 0-indexed numpy) and the nasal root (landmark 28 -> index 27), removing
global translation, scale, and in-plane rotation while preserving local
non-rigid facial motion. Then applies a symmetric 3-frame moving-average
filter.

Usage:
    python 02_normalize_smooth.py \
        --landmark_dir /path/to/landmarks \
        --out_dir /path/to/normalized
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

# 0-indexed dlib landmark indices
LEFT_CANTHUS = 36
RIGHT_CANTHUS = 45
NASAL_ROOT = 27


def estimate_similarity_transform(src_pts, dst_pts):
    """
    Estimate scale, rotation, translation mapping src_pts -> dst_pts using
    a Procrustes-style closed-form least squares solution (Umeyama, 1991).
    src_pts, dst_pts: (N, 2) arrays of corresponding reference points.
    Returns 2x2 rotation*scale matrix R and 2-vector translation t, such
    that dst ~= R @ src + t.
    """
    src_mean = src_pts.mean(axis=0)
    dst_mean = dst_pts.mean(axis=0)
    src_c = src_pts - src_mean
    dst_c = dst_pts - dst_mean

    cov = dst_c.T @ src_c / src_pts.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, d])
    R_rot = U @ D @ Vt

    var_src = (src_c ** 2).sum() / src_pts.shape[0]
    scale = np.trace(np.diag(S) @ D) / var_src if var_src > 1e-8 else 1.0

    R = scale * R_rot
    t = dst_mean - R @ src_mean
    return R, t


def normalize_trajectory(landmarks, canonical_ref=None):
    """
    landmarks: (T, 68, 2) raw pixel coordinates, may contain NaN rows for
    frames with failed detection.
    canonical_ref: (3, 2) canonical target positions for
    [left_canthus, right_canthus, nasal_root]. If None, uses the mean
    reference configuration across all valid frames of this video as the
    canonical target, so normalized coordinates are video-centric.
    Returns: (T, 68, 2) normalized coordinates (NaN preserved where input was NaN).
    """
    T = landmarks.shape[0]
    ref_idx = [LEFT_CANTHUS, RIGHT_CANTHUS, NASAL_ROOT]
    valid = ~np.isnan(landmarks[:, 0, 0])

    if canonical_ref is None:
        ref_pts_all = landmarks[valid][:, ref_idx, :]
        canonical_ref = ref_pts_all.mean(axis=0)

    normalized = np.full_like(landmarks, np.nan)
    for t in range(T):
        if not valid[t]:
            continue
        src_ref = landmarks[t, ref_idx, :]
        R, tr = estimate_similarity_transform(src_ref, canonical_ref)
        normalized[t] = (R @ landmarks[t].T).T + tr

    return normalized


def moving_average_smooth(x, window=3):
    """
    Symmetric moving-average filter along the time axis, NaN-aware.
    x: (T, 68, 2). window must be odd.
    """
    assert window % 2 == 1
    half = window // 2
    T = x.shape[0]
    smoothed = np.full_like(x, np.nan)
    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        segment = x[lo:hi]
        with np.errstate(invalid="ignore"):
            smoothed[t] = np.nanmean(segment, axis=0)
    return smoothed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmark_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--smoothing_window", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(Path(args.landmark_dir).glob("*.npy"))
    print(f"Found {len(npy_files)} landmark files")

    for f in tqdm(npy_files, desc="Normalizing + smoothing"):
        out_path = out_dir / f.name
        if out_path.exists():
            continue
        raw = np.load(f)
        norm = normalize_trajectory(raw)
        smooth = moving_average_smooth(norm, window=args.smoothing_window)
        np.save(out_path, smooth)


if __name__ == "__main__":
    main()
