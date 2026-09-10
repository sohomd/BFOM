"""
Stage 4: Region-specific authentic-motion density-model training.

Fits one Gaussian Mixture Model (GMM) per facial region using AUTHENTIC
training sequences only. Separate bundles are trained for first-order,
second-order, and combined kinematic representations. Region-specific models
avoid pooling naturally different mouth, eye/brow, nose, and contour dynamics
into one global density.

Usage:
    python 04_train_density_model.py \
        --kinematics_dir /path/to/kinematics \
        --split_file splits/train_authentic.txt \
        --val_split_file splits/val_authentic.txt \
        --out_dir models \
        --representation combined
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

CANDIDATE_M = [2, 4, 8, 16, 32]


def build_feature(v, a, representation):
    if representation == "first":
        return v
    if representation == "second":
        return a
    if representation == "combined":
        return np.concatenate([v, a], axis=-1)
    raise ValueError(representation)


def load_region_map(kinematics_dir):
    path = Path(kinematics_dir) / "_region_map.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Region map not found: {path}. Run Stage 3 before Stage 4."
        )
    return np.load(path, allow_pickle=True).item()


def load_region_features(kinematics_dir, stem_list, representation, indices):
    feats = []
    for stem in stem_list:
        path = Path(kinematics_dir) / f"{stem}.npz"
        if not path.exists():
            print(f"[WARN] Missing kinematics file: {path}")
            continue

        data = np.load(path)
        feat = build_feature(data["v"], data["a"], representation)
        region_feat = feat[:, indices, :].reshape(-1, feat.shape[-1])
        valid = ~np.isnan(region_feat).any(axis=1)
        if valid.any():
            feats.append(region_feat[valid])

    if not feats:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(feats, axis=0)


def select_best_gmm(train_feats, val_feats, candidate_m, seed=0):
    best_bic = np.inf
    best_model = None
    best_m = None

    # Do not attempt a mixture with more components than the data support.
    feasible = [m for m in candidate_m if m < len(train_feats)]
    if not feasible:
        feasible = [1]

    for m in feasible:
        gmm = GaussianMixture(
            n_components=m,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=seed,
            max_iter=300,
            n_init=3,
        )
        gmm.fit(train_feats)
        bic = gmm.bic(val_feats)
        print(f"    M={m}: validation BIC={bic:.2f}")
        if bic < best_bic:
            best_bic = bic
            best_model = gmm
            best_m = m

    return best_model, best_m, best_bic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinematics_dir", required=True)
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--val_split_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--representation", choices=["first", "second", "combined"],
                    required=True)
    ap.add_argument("--clip_percentile", type=float, default=99.9,
                    help="Clip region-level NLL scores at this authentic-training percentile")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_stems = [l.strip() for l in open(args.split_file) if l.strip()]
    val_stems = [l.strip() for l in open(args.val_split_file) if l.strip()]
    region_map = load_region_map(args.kinematics_dir)

    region_models = {}
    for region_name, indices in region_map.items():
        print(f"\nRegion: {region_name}")
        train_feats = load_region_features(
            args.kinematics_dir, train_stems, args.representation, indices
        )
        val_feats = load_region_features(
            args.kinematics_dir, val_stems, args.representation, indices
        )
        print(f"  train={train_feats.shape}, val={val_feats.shape}")

        if train_feats.size == 0 or val_feats.size == 0:
            raise RuntimeError(f"No valid features available for region {region_name}")

        gmm, best_m, best_bic = select_best_gmm(
            train_feats, val_feats, CANDIDATE_M, args.seed
        )

        train_scores = -gmm.score_samples(train_feats)
        clip_value = float(np.percentile(train_scores, args.clip_percentile))
        print(
            f"  selected M={best_m}, BIC={best_bic:.2f}, "
            f"clip={clip_value:.4f}"
        )

        region_models[region_name] = {
            "model": gmm,
            "indices": list(indices),
            "selected_M": best_m,
            "best_bic": float(best_bic),
            "clip_value": clip_value,
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gmm_region_{args.representation}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "model_type": "region_gmm",
            "representation": args.representation,
            "clip_percentile": args.clip_percentile,
            "regions": region_models,
        }, f)

    print(f"\nSaved region-specific model bundle to {out_path}")


if __name__ == "__main__":
    main()
