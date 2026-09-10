# Second-Order Kinematic Deepfake Detection — Revised Pipeline

This revision strengthens the computational methodology in four ways:

1. first- and second-order motion are evaluated at the same centered frame;
2. authentic-motion density estimation is region-specific rather than one GMM pooled across all landmarks;
3. AUROC/AUPRC are reported with video-level bootstrap 95% confidence intervals;
4. paired bootstrap comparisons directly test first-order vs second-order vs combined representations.

## Recommended dataset protocol

Use three datasets for the main experiment:

- **FaceForensics++ (FF++) c23**: authentic videos for training/validation of the authentic-motion model, plus the four FF++ manipulation methods for the reference test.
- **Celeb-DF v2**: independent cross-dataset/cross-generator evaluation.
- **DeepSpeak**: recent talking-head / audiovisual deepfake evaluation. Verify the exact subsets and generation methods used before describing it as diffusion-based.

Keep **VoxCeleb2 optional** for a later authentic-training-diversity ablation. The cleanest primary claim is obtained when the authentic-motion distribution is estimated from FF++ authentic training data only and then evaluated on independent synthetic datasets.

No synthetic videos are used to fit the authentic-motion GMM. Synthetic labels are used only for evaluation metrics and, if needed, threshold/calibration studies that are explicitly reported as such.

## Stage 1: Extract dlib 68-point trajectories

```bash
python 01_extract_landmarks.py \
    --video_dir data/ffpp/authentic \
    --out_dir data/landmarks/ffpp_authentic \
    --predictor_path shape_predictor_68_face_landmarks.dat \
    --label real
```

Repeat for the FF++ test manipulations, Celeb-DF v2, and DeepSpeak.

## Stage 2: Normalize and smooth

```bash
python 02_normalize_smooth.py \
    --landmark_dir data/landmarks/ffpp_authentic \
    --out_dir data/normalized/ffpp_authentic \
    --smoothing_window 3
```

For the no-smoothing ablation, use `--smoothing_window 1`.

## Stage 3: Compute aligned first- and second-order kinematics

```bash
python 03_compute_kinematics.py \
    --normalized_dir data/normalized/ffpp_authentic \
    --out_dir data/kinematics/ffpp_authentic \
    --fps 30
```

`--fps` must match the actual source frame rate. The revised script uses a centered first derivative and a centered second derivative, both defined on the same interior frames.

## Stage 4: Train region-specific authentic-motion GMMs

Create identity-disjoint authentic train/validation split files first.

```bash
python 04_train_density_model.py \
    --kinematics_dir data/kinematics/ffpp_authentic \
    --split_file splits/train_authentic.txt \
    --val_split_file splits/val_authentic.txt \
    --out_dir models \
    --representation first

python 04_train_density_model.py \
    --kinematics_dir data/kinematics/ffpp_authentic \
    --split_file splits/train_authentic.txt \
    --val_split_file splits/val_authentic.txt \
    --out_dir models \
    --representation second

python 04_train_density_model.py \
    --kinematics_dir data/kinematics/ffpp_authentic \
    --split_file splits/train_authentic.txt \
    --val_split_file splits/val_authentic.txt \
    --out_dir models \
    --representation combined
```

The revised Stage 4 fits separate GMMs for upper-face/periorbital, nasal, perioral, and contour regions. The number of mixture components is selected by BIC using authentic validation data for each region.

## Stage 5: Score each test dataset and report 95% CIs

Run all three representations on exactly the same video set.

```bash
python 05_score_and_evaluate.py \
    --kinematics_dir data/kinematics/celebdf \
    --model_path models/gmm_region_first.pkl \
    --labels_file splits/celebdf_test_labels.csv \
    --representation first \
    --p_top 25 --q_top 25 \
    --bootstrap 2000 \
    --out_csv results/first_celebdf.csv

python 05_score_and_evaluate.py \
    --kinematics_dir data/kinematics/celebdf \
    --model_path models/gmm_region_second.pkl \
    --labels_file splits/celebdf_test_labels.csv \
    --representation second \
    --p_top 25 --q_top 25 \
    --bootstrap 2000 \
    --out_csv results/second_celebdf.csv

python 05_score_and_evaluate.py \
    --kinematics_dir data/kinematics/celebdf \
    --model_path models/gmm_region_combined.pkl \
    --labels_file splits/celebdf_test_labels.csv \
    --representation combined \
    --p_top 25 --q_top 25 \
    --bootstrap 2000 \
    --out_csv results/combined_celebdf.csv
```

Repeat for FF++ and DeepSpeak.

## Stage 6: Paired representation comparison

```bash
python 06_compare_representations.py \
    --first results/first_celebdf.csv \
    --second results/second_celebdf.csv \
    --combined results/combined_celebdf.csv \
    --bootstrap 5000 \
    --out_csv results/paired_bootstrap_celebdf.csv
```

This is the key statistical analysis for the paper's central claim. It uses identical bootstrap video resamples for all representations and reports confidence intervals for differences such as:

- second-order minus first-order AUROC;
- combined minus first-order AUROC;
- combined minus second-order AUROC;
- corresponding AUPRC differences.

If the 95% CI for `second - first` stays above zero, the evidence supports the claim that second-order motion outperforms first-order motion on that evaluation set.

## Remaining reviewer-driven additions

Two experiments remain separate from this core revision:

1. **Tracker robustness**: repeat the first/second/combined comparison with a modern face landmark/mesh tracker. This tests whether the second-order advantage is specific to dlib. Do not claim tracker independence until this is run.
2. **Temporal-model robustness**: compare the region-specific GMM with one small temporal model such as a GRU, using the same first/second/combined representations. The strongest result would be second-order outperforming first-order under both model families.

These can be added after the corrected core pipeline is producing stable results.

## Important reproducibility notes

- Build identity-disjoint train/validation/test splits from official identity metadata where available.
- Use the actual frame rate of each source video or preprocess to a documented common frame rate before computing derivatives.
- Manually inspect low landmark-detection-rate videos rather than silently allowing tracker failures to dominate second-order scores.
- Select `p_top`, `q_top`, smoothing window, and any detection thresholds on validation data and then hold them fixed during testing.
- Report the number of videos successfully scored and any exclusion criteria for every dataset.
