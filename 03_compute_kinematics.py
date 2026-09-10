"""
Stage 3: Kinematic feature computation.

Computes temporally aligned first-order and second-order facial kinematics from
normalized/smoothed trajectories. Both representations are centered on the same
interior frame t, so first-vs-second-order comparisons are not confounded by a
one-frame temporal offset.

Usage:
    python 03_compute_kinematics.py \
        --normalized_dir /path/to/normalized \
        --out_dir /path/to/kinematics \
        --fps 30
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

# 1-indexed dlib landmark groupings converted to 0-indexed below.
REGIONS_1INDEXED = {
    "upper_face_periorbital": list(range(18, 28)) + list(range(37, 49)),
    "nasal": list(range(28, 37)),
    "perioral": list(range(49, 69)),
    "contour": list(range(1, 18)),
}
REGIONS = {name: [i - 1 for i in idxs] for name, idxs in REGIONS_1INDEXED.items()}


def compute_centered_first_order(x, dt):
    """Centered first derivative at interior frames: (x[t+1]-x[t-1])/(2dt)."""
    return (x[2:] - x[:-2]) / (2.0 * dt)


def compute_second_order(x, dt):
    """Second derivative at the same interior frames."""
    return (x[2:] - 2.0 * x[1:-1] + x[:-2]) / (dt ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalized_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fps", type=float, default=30.0,
                    help="Actual source video frame rate; dt=1/fps")
    args = ap.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    dt = 1.0 / args.fps
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(Path(args.normalized_dir).glob("*.npy"))
    print(f"Found {len(npy_files)} normalized trajectory files")

    for f in tqdm(npy_files, desc="Computing kinematics"):
        out_path = out_dir / f"{f.stem}.npz"
        if out_path.exists():
            continue

        x = np.load(f)  # (T, 68, 2), may contain NaN
        if x.ndim != 3 or x.shape[1:] != (68, 2):
            print(f"[WARN] Unexpected landmark shape {x.shape} in {f.name}; skipping")
            continue
        if x.shape[0] < 3:
            continue

        # Both arrays have shape (T-2, 68, 2) and correspond to original
        # interior frames 1..T-2 (0-indexed).
        v = compute_centered_first_order(x, dt)
        a = compute_second_order(x, dt)

        np.savez(
            out_path,
            v=v.astype("float32"),
            a=a.astype("float32"),
            fps=np.float32(args.fps),
            dt=np.float32(dt),
        )

    np.save(out_dir / "_region_map.npy", REGIONS, allow_pickle=True)
    print(f"Saved region map to {out_dir / '_region_map.npy'}")


if __name__ == "__main__":
    main()
