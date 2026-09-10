"""
Stage 5: Region-specific scoring, evaluation, and uncertainty estimation.

Scores each landmark with the GMM associated with its facial region, pools
landmark scores to frame scores and frame scores to a video score, then reports
AUROC/AUPRC with video-level bootstrap 95% confidence intervals.

Usage:
    python 05_score_and_evaluate.py \
        --kinematics_dir /path/to/kinematics \
        --model_path models/gmm_region_combined.pkl \
        --labels_file splits/test_labels.csv \
        --representation combined \
        --p_top 25 --q_top 25 \
        --bootstrap 2000 \
        --out_csv results/combined_test.csv
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


def build_feature(v, a, representation):
    if representation == "first":
        return v
    if representation == "second":
        return a
    if representation == "combined":
        return np.concatenate([v, a], axis=-1)
    raise ValueError(representation)


def region_landmark_scores(model_bundle, feat):
    """Return s_kt with shape (T,K), scored by the appropriate regional GMM."""
    T, K, D = feat.shape
    scores = np.full((T, K), np.nan, dtype=np.float32)

    for region_name, region_info in model_bundle["regions"].items():
        indices = np.asarray(region_info["indices"], dtype=int)
        gmm = region_info["model"]
        clip_value = region_info["clip_value"]

        region_feat = feat[:, indices, :]
        flat = region_feat.reshape(-1, D)
        valid = ~np.isnan(flat).any(axis=1)
        region_scores = np.full(flat.shape[0], np.nan, dtype=np.float32)

        if valid.any():
            s = -gmm.score_samples(flat[valid])
            # Clipping limits the influence of extreme values caused by
            # tracking failures; it is not treated as numerical stabilization.
            s = np.clip(s, a_min=None, a_max=clip_value)
            region_scores[valid] = s.astype(np.float32)

        scores[:, indices] = region_scores.reshape(T, len(indices))

    return scores


def video_level_score(s_kt, p_top=25.0, q_top=25.0):
    T, K = s_kt.shape
    frame_scores = np.full(T, np.nan, dtype=float)

    for t in range(T):
        row = s_kt[t]
        row = row[~np.isnan(row)]
        if row.size == 0:
            continue
        n_top = max(1, int(np.ceil(row.size * p_top / 100.0)))
        frame_scores[t] = np.sort(row)[-n_top:].mean()

    valid_frames = frame_scores[~np.isnan(frame_scores)]
    if valid_frames.size == 0:
        return np.nan, frame_scores

    n_top_frames = max(1, int(np.ceil(valid_frames.size * q_top / 100.0)))
    score = np.sort(valid_frames)[-n_top_frames:].mean()
    return float(score), frame_scores


def bootstrap_metric_ci(y_true, y_score, metric_fn, n_boot=2000, seed=0, alpha=0.05):
    """Stratified video-level bootstrap CI to keep both classes in each resample."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    pos = np.flatnonzero(y_true == 1)
    neg = np.flatnonzero(y_true == 0)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Both authentic and manipulated videos are required")

    values = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ])
        values.append(metric_fn(y_true[idx], y_score[idx]))

    values = np.asarray(values)
    lo, hi = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinematics_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--labels_file", required=True)
    ap.add_argument("--representation", choices=["first", "second", "combined"],
                    required=True)
    ap.add_argument("--p_top", type=float, default=25.0)
    ap.add_argument("--q_top", type=float, default=25.0)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    with open(args.model_path, "rb") as f:
        model_bundle = pickle.load(f)

    if model_bundle.get("model_type") != "region_gmm":
        raise ValueError("This scorer expects a region-specific GMM bundle from revised Stage 4")
    if model_bundle.get("representation") != args.representation:
        raise ValueError(
            f"Model representation={model_bundle.get('representation')} but "
            f"--representation={args.representation}"
        )

    labels_df = pd.read_csv(args.labels_file)
    rows = []

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Scoring videos"):
        stem, label = str(row["video_stem"]), int(row["label"])
        path = Path(args.kinematics_dir) / f"{stem}.npz"
        if not path.exists():
            print(f"[WARN] Missing kinematics for {stem}, skipping")
            continue

        data = np.load(path)
        feat = build_feature(data["v"], data["a"], args.representation)
        s_kt = region_landmark_scores(model_bundle, feat)
        s_v, _ = video_level_score(s_kt, args.p_top, args.q_top)
        if np.isnan(s_v):
            print(f"[WARN] No valid frames scored for {stem}, skipping")
            continue
        rows.append({"video_stem": stem, "label": label, "score": s_v})

    result_df = pd.DataFrame(rows)
    if result_df.empty or result_df["label"].nunique() < 2:
        raise RuntimeError("Evaluation requires at least one authentic and one manipulated video")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)

    y_true = result_df["label"].to_numpy()
    y_score = result_df["score"].to_numpy()

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    auc_ci = bootstrap_metric_ci(
        y_true, y_score, roc_auc_score, n_boot=args.bootstrap, seed=args.seed
    )
    ap_ci = bootstrap_metric_ci(
        y_true, y_score, average_precision_score,
        n_boot=args.bootstrap, seed=args.seed + 1
    )

    print(f"\nRepresentation: {args.representation}")
    print(f"N videos scored: {len(result_df)}")
    print(f"AUROC: {auroc:.4f}  (95% CI {auc_ci[0]:.4f}, {auc_ci[1]:.4f})")
    print(f"AUPRC: {auprc:.4f}  (95% CI {ap_ci[0]:.4f}, {ap_ci[1]:.4f})")
    print(f"Saved per-video scores to {out_path}")


if __name__ == "__main__":
    main()
