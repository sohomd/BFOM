"""
Stage 6: Paired statistical comparison of first-, second-, and combined-motion
representations using identical video-level bootstrap resamples.

Each input CSV must contain: video_stem,label,score

Usage:
    python 06_compare_representations.py \
        --first results/first_celebdf.csv \
        --second results/second_celebdf.csv \
        --combined results/combined_celebdf.csv \
        --bootstrap 5000 \
        --out_csv results/paired_bootstrap_celebdf.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def load_and_merge(first_path, second_path, combined_path):
    def prep(path, name):
        df = pd.read_csv(path)[["video_stem", "label", "score"]].copy()
        return df.rename(columns={"score": name})

    f = prep(first_path, "first")
    s = prep(second_path, "second")
    c = prep(combined_path, "combined")

    merged = f.merge(s, on=["video_stem", "label"], how="inner")
    merged = merged.merge(c, on=["video_stem", "label"], how="inner")
    return merged


def paired_bootstrap(df, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    y = df["label"].to_numpy()
    scores = {k: df[k].to_numpy() for k in ["first", "second", "combined"]}

    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Both classes are required")

    records = []
    for b in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ])
        row = {"bootstrap": b}
        for name in scores:
            row[f"auroc_{name}"] = roc_auc_score(y[idx], scores[name][idx])
            row[f"auprc_{name}"] = average_precision_score(y[idx], scores[name][idx])
        row["delta_auroc_second_minus_first"] = row["auroc_second"] - row["auroc_first"]
        row["delta_auroc_combined_minus_first"] = row["auroc_combined"] - row["auroc_first"]
        row["delta_auroc_combined_minus_second"] = row["auroc_combined"] - row["auroc_second"]
        row["delta_auprc_second_minus_first"] = row["auprc_second"] - row["auprc_first"]
        row["delta_auprc_combined_minus_first"] = row["auprc_combined"] - row["auprc_first"]
        row["delta_auprc_combined_minus_second"] = row["auprc_combined"] - row["auprc_second"]
        records.append(row)
    return pd.DataFrame(records)


def summarize_difference(values):
    values = np.asarray(values)
    lo, hi = np.quantile(values, [0.025, 0.975])
    # Two-sided bootstrap sign probability, reported as a descriptive p-like value.
    p = 2.0 * min(np.mean(values <= 0), np.mean(values >= 0))
    p = min(1.0, float(p))
    return float(values.mean()), float(lo), float(hi), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", required=True)
    ap.add_argument("--second", required=True)
    ap.add_argument("--combined", required=True)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = load_and_merge(args.first, args.second, args.combined)
    if len(df) == 0:
        raise RuntimeError("No common videos across the three result CSVs")

    print(f"Common videos used in paired comparison: {len(df)}")
    for metric, fn in [("AUROC", roc_auc_score), ("AUPRC", average_precision_score)]:
        print(f"\n{metric} point estimates")
        for name in ["first", "second", "combined"]:
            print(f"  {name:8s}: {fn(df['label'], df[name]):.4f}")

    boot = paired_bootstrap(df, n_boot=args.bootstrap, seed=args.seed)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    boot.to_csv(out_path, index=False)

    print("\nPaired bootstrap differences (mean, 95% CI, two-sided sign p)")
    for col in [
        "delta_auroc_second_minus_first",
        "delta_auroc_combined_minus_first",
        "delta_auroc_combined_minus_second",
        "delta_auprc_second_minus_first",
        "delta_auprc_combined_minus_first",
        "delta_auprc_combined_minus_second",
    ]:
        mean, lo, hi, p = summarize_difference(boot[col])
        print(f"  {col}: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}], p~{p:.4g}")

    print(f"\nSaved bootstrap samples to {out_path}")


if __name__ == "__main__":
    main()
