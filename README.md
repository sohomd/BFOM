# Beyond First Order Motion: Second-Order Kinematic Deepfake Detection —

![Framework](../assets/Architecture.png)

This strengthens the computational methodology in four ways:

1. first- and second-order motion are evaluated at the same centered frame;
2. authentic-motion density estimation is region-specific rather than one GMM pooled across all landmarks;
3. AUROC/AUPRC are reported with video-level bootstrap 95% confidence intervals;
4. paired bootstrap comparisons directly test first-order vs second-order vs combined representations.

## Recommended dataset protocol

Use three datasets for the main experiment:

- **FaceForensics++ (FF++) c23**: authentic videos for training/validation of the authentic-motion model, plus the four FF++ manipulation methods for the reference test.
- **Celeb-DF v2**: independent cross-dataset/cross-generator evaluation.
- **AV-Deepfake1M**: large-scale audiovisual deepfake evaluation. Verify the exact subsets and generation methods used before analyzing or reporting results.

Keep **VoxCeleb2 optional** for a later authentic-training-diversity ablation. The cleanest primary claim is obtained when the authentic-motion distribution is estimated from FF++ authentic training data only and then evaluated on independent synthetic datasets.

No synthetic videos are used to fit the authentic-motion GMM. Synthetic labels are used only for evaluation metrics and, if needed, threshold/calibration studies that are explicitly reported as such.

### Requirements / Dependencies

You can install the required Python packages for this pipeline using `pip`:

```bash
pip install \
    "dlib>=19.24" \
    "opencv-python>=4.8" \
    "numpy>=1.24" \
    "scipy>=1.10" \
    "scikit-learn>=1.3" \
    "pandas>=2.0" \
    "tqdm>=4.65" \
    "matplotlib>=3.7"
```
```bash
python -c "import dlib, cv2, numpy, scipy, sklearn, pandas, tqdm, matplotlib; print('Environment OK')"
```

```text
FFPP/
├── original_sequences/
│   └── youtube/
│       └── c23/
│           └── videos/
│
└── manipulated_sequences/
    ├── Deepfakes/
    │   └── c23/
    │       └── videos/
    ├── Face2Face/
    │   └── c23/
    │       └── videos/
    ├── FaceSwap/
    │   └── c23/
    │       └── videos/
    └── NeuralTextures/
        └── c23/
            └── videos/
```

```bash
mkdir -p landmarks/original
mkdir -p normalized/original
mkdir -p kinematics/original
mkdir -p models
mkdir -p results
mkdir -p splits
```
```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```

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
python "04_train_region_gmm.py" \
    --kinematics_dir kinematics/original \
    --split_file splits/train_authentic.txt \
    --val_split_file splits/val_authentic.txt \
    --out_dir models \
    --representation second

python "04_train_region_gmm.py" \
    --kinematics_dir kinematics/original \
    --split_file splits/train_authentic.txt \
    --val_split_file splits/val_authentic.txt \
    --out_dir models \
    --representation first

python "04_train_region_gmm.py" \
    --kinematics_dir kinematics/original \
    --split_file splits/train_authentic.txt \
    --val_split_file splits/val_authentic.txt \
    --out_dir models \
    --representation combined
```

The Stage 4 fits separate GMMs for upper-face/periorbital, nasal, perioral, and contour regions. The number of mixture components is selected by BIC using authentic validation data for each region.

## Stage 5: Score each test dataset and CIs

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

