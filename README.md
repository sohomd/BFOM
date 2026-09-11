# Beyond First-Order Motion: Second-Order Kinematic Deepfake Detection

![Framework](Architecture.png)

This repository implements an authentic-reference approach to synthetic-video
detection using first- and second-order facial kinematics. Facial landmark
trajectories are normalized and converted into velocity and acceleration
representations. Region-specific Gaussian Mixture Models (GMMs) are fitted
**only on authentic FaceForensics++ videos** and are subsequently frozen for
evaluation on authentic and manipulated videos.

No synthetic videos are used to fit the authentic-motion GMMs.

---

## Dataset Protocol

The primary experimental protocol uses:

- **FaceForensics++ (FF++) c23** — authentic videos are used for
  training/validation of the authentic-motion model. Deepfakes, Face2Face,
  FaceSwap, and NeuralTextures are used only during evaluation.
- **Celeb-DF v2** — independent cross-dataset evaluation.
- **AV-Deepfake1M** — additional large-scale audiovisual deepfake evaluation.

VoxCeleb2 may optionally be used for an authentic-training-diversity ablation.

The primary authentic-motion models are fitted using **FF++ authentic training
videos only**. Synthetic labels are used only during evaluation.

---

## Requirements

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

Verify the environment:

```bash
python -c "import dlib, cv2, numpy, scipy, sklearn, pandas, tqdm, matplotlib; print('Environment OK')"
```

Download the dlib 68-point landmark predictor:

```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

---

# FaceForensics++ Evaluation

## Required FF++ Folder Structure

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

Create the output directories:

```bash
mkdir -p landmarks/{original,deepfakes,face2face,faceswap,neuraltextures}
mkdir -p normalized/{original,deepfakes,face2face,faceswap,neuraltextures}
mkdir -p kinematics/{original,deepfakes,face2face,faceswap,neuraltextures}
mkdir -p models results splits/ffpp_official
```

---

## Stage 1: Extract 68-Point Facial Landmark Trajectories

### Authentic FF++ videos

```bash
python 01_extract_landmarks.py \
    --video_dir "FFPP/original_sequences/youtube/c23/videos" \
    --out_dir landmarks/original \
    --predictor_path shape_predictor_68_face_landmarks.dat \
    --label 0
```

### Deepfakes

```bash
python 01_extract_landmarks.py \
    --video_dir "FFPP/manipulated_sequences/Deepfakes/c23/videos" \
    --out_dir landmarks/deepfakes \
    --predictor_path shape_predictor_68_face_landmarks.dat \
    --label 1
```

### Face2Face

```bash
python 01_extract_landmarks.py \
    --video_dir "FFPP/manipulated_sequences/Face2Face/c23/videos" \
    --out_dir landmarks/face2face \
    --predictor_path shape_predictor_68_face_landmarks.dat \
    --label 1
```

### FaceSwap

```bash
python 01_extract_landmarks.py \
    --video_dir "FFPP/manipulated_sequences/FaceSwap/c23/videos" \
    --out_dir landmarks/faceswap \
    --predictor_path shape_predictor_68_face_landmarks.dat \
    --label 1
```

### NeuralTextures

```bash
python 01_extract_landmarks.py \
    --video_dir "FFPP/manipulated_sequences/NeuralTextures/c23/videos" \
    --out_dir landmarks/neuraltextures \
    --predictor_path shape_predictor_68_face_landmarks.dat \
    --label 1
```

---

## Stage 2: Normalize and Smooth Landmark Trajectories

A similarity transformation removes translation, scale, and in-plane rotation.
A three-frame moving average is used in the primary configuration.

### Authentic

```bash
python 02_normalize_smooth.py \
    --landmark_dir landmarks/original \
    --out_dir normalized/original \
    --smoothing_window 3
```

### Manipulated

```bash
python 02_normalize_smooth.py \
    --landmark_dir landmarks/deepfakes \
    --out_dir normalized/deepfakes \
    --smoothing_window 3

python 02_normalize_smooth.py \
    --landmark_dir landmarks/face2face \
    --out_dir normalized/face2face \
    --smoothing_window 3

python 02_normalize_smooth.py \
    --landmark_dir landmarks/faceswap \
    --out_dir normalized/faceswap \
    --smoothing_window 3

python 02_normalize_smooth.py \
    --landmark_dir landmarks/neuraltextures \
    --out_dir normalized/neuraltextures \
    --smoothing_window 3
```

For the no-smoothing ablation:

```bash
--smoothing_window 1
```

---

## Stage 3: Compute First- and Second-Order Kinematics

The implementation reads the frame rate of each source video and computes
centered first- and second-order derivatives on the same temporal support.

### Authentic

```bash
python 03_compute_kinematics.py \
    --normalized_dir normalized/original \
    --video_dir "FFPP/original_sequences/youtube/c23/videos" \
    --out_dir kinematics/original \
    --overwrite
```

### Deepfakes

```bash
python 03_compute_kinematics.py \
    --normalized_dir normalized/deepfakes \
    --video_dir "FFPP/manipulated_sequences/Deepfakes/c23/videos" \
    --out_dir kinematics/deepfakes \
    --overwrite
```

### Face2Face

```bash
python 03_compute_kinematics.py \
    --normalized_dir normalized/face2face \
    --video_dir "FFPP/manipulated_sequences/Face2Face/c23/videos" \
    --out_dir kinematics/face2face \
    --overwrite
```

### FaceSwap

```bash
python 03_compute_kinematics.py \
    --normalized_dir normalized/faceswap \
    --video_dir "FFPP/manipulated_sequences/FaceSwap/c23/videos" \
    --out_dir kinematics/faceswap \
    --overwrite
```

### NeuralTextures

```bash
python 03_compute_kinematics.py \
    --normalized_dir normalized/neuraltextures \
    --video_dir "FFPP/manipulated_sequences/NeuralTextures/c23/videos" \
    --out_dir kinematics/neuraltextures \
    --overwrite
```

---

## Stage 4: Obtain the Official FF++ Splits

```bash
wget -O splits/ffpp_official/train.json \
https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/train.json

wget -O splits/ffpp_official/val.json \
https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/val.json

wget -O splits/ffpp_official/test.json \
https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/test.json
```

Convert the official pair lists into authentic source-video ID lists:

```bash
python - <<'PY'
import json
from pathlib import Path

base = Path("splits/ffpp_official")

for split in ["train", "val", "test"]:
    pairs = json.load(open(base / f"{split}.json"))
    ids = sorted({x for pair in pairs for x in pair})

    output = base / f"{split}_authentic.txt"
    output.write_text("\n".join(ids) + "\n")

    print(f"{split}: {len(ids)} authentic source videos")
PY
```

Expected split sizes:

```text
train: 720
val:   140
test:  140
```
## Authentic-Motion Density Models

The framework models facial motion using region-specific Gaussian Mixture
Models (GMMs) trained **only on authentic FaceForensics++ videos**. Three
kinematic representations are evaluated:

- **First-order (`v`)** — facial landmark velocity.
- **Second-order (`a`)** — changes in landmark velocity across consecutive frames.
- **Combined (`[v, a]`)** — joint first- and second-order representation.

GMM complexity is selected independently for each facial region using the
lowest Bayesian Information Criterion (BIC) on the held-out authentic
validation set. Candidate mixture sizes are `{2, 4, 8, 16, 32}`.
Region-level negative log-likelihood (NLL) scores are clipped at the
99.9th percentile of the authentic training distribution.

| Representation | Upper-Face / Periorbital | Nasal | Perioral | Contour |
|---|---:|---:|---:|---:|
| First-order (`v`) | M=32, clip=17.1995 | M=16, clip=17.1247 | M=32, clip=18.6625 | M=16, clip=18.5237 |
| Second-order (`a`) | M=32, clip=24.1884 | M=16, clip=24.2555 | M=32, clip=25.1709 | M=16, clip=26.0161 |
| Combined (`[v,a]`) | M=32, clip=35.6278 | M=32, clip=36.3620 | M=32, clip=38.1580 | M=32, clip=38.7831 |

Here, `M` denotes the number of Gaussian mixture components selected using
validation BIC, and `clip` denotes the 99.9th-percentile authentic-training
NLL clipping threshold used during scoring.

### Combined-Representation BIC Selection

For reproducibility, the selected validation BIC values for the combined
representation were:

| Facial Region | Selected M | Validation BIC | NLL Clip |
|---|---:|---:|---:|
| Upper-Face / Periorbital | 32 | 67,998,143.55 | 35.6278 |
| Nasal | 32 | 28,803,711.41 | 36.3620 |
| Perioral | 32 | 71,479,393.90 | 38.1580 |
| Contour | 32 | 61,886,356.05 | 38.7831 |

The resulting combined authentic-motion model is stored as:

```text
models/gmm_region_combined.pkl

## Stage 5: Fit Authentic-Motion Density Models

Three representations are evaluated:

- **First-order (`v`)** — centered facial landmark velocity.
- **Second-order (`a`)** — centered second-order facial kinematics.
- **Combined (`[v,a]`)** — joint first- and second-order representation.

Importantly, all three GMMs are fitted using **authentic FF++ videos only**.

### First-order

```bash
python 04_train_region_gmm.py \
    --kinematics_dir kinematics/original \
    --split_file splits/ffpp_official/train_authentic.txt \
    --val_split_file splits/ffpp_official/val_authentic.txt \
    --out_dir models \
    --representation first
```

### Second-order

```bash
python 04_train_region_gmm.py \
    --kinematics_dir kinematics/original \
    --split_file splits/ffpp_official/train_authentic.txt \
    --val_split_file splits/ffpp_official/val_authentic.txt \
    --out_dir models \
    --representation second
```

### Combined

```bash
python 04_train_region_gmm.py \
    --kinematics_dir kinematics/original \
    --split_file splits/ffpp_official/train_authentic.txt \
    --val_split_file splits/ffpp_official/val_authentic.txt \
    --out_dir models \
    --representation combined
```

The method fits separate GMMs for four facial regions:

1. upper-face/periorbital;
2. nasal;
3. perioral;
4. facial contour.

The number of Gaussian components is selected independently for each region
using BIC on held-out authentic validation data. Candidate mixture sizes are:

```text
{2, 4, 8, 16, 32}
```

Region-level negative log-likelihood (NLL) scores are clipped at the 99.9th
percentile of the corresponding authentic training-score distribution.

---

## Stage 6: Construct the Official FF++ Test Set

Only videos belonging to the official FF++ test split are included.

```bash
rm -rf kinematics/ffpp_test
mkdir -p kinematics/ffpp_test
```

```bash
python - <<'PY'
import json
import shutil
from pathlib import Path
import pandas as pd

pairs = json.load(open("splits/ffpp_official/test.json"))

test_ids = {x for pair in pairs for x in pair}

test_pairs = set()
for a, b in pairs:
    test_pairs.add((a, b))
    test_pairs.add((b, a))

out = Path("kinematics/ffpp_test")
rows = []

# Authentic test videos
for stem in sorted(test_ids):

    src = Path("kinematics/original") / f"{stem}.npz"

    if not src.exists():
        print("[WARN] Missing authentic:", src)
        continue

    newstem = f"real_{stem}"

    shutil.copy2(
        src,
        out / f"{newstem}.npz"
    )

    rows.append((newstem, 0))


# Manipulated test videos
methods = {
    "deepfakes": Path("kinematics/deepfakes"),
    "face2face": Path("kinematics/face2face"),
    "faceswap": Path("kinematics/faceswap"),
    "neuraltextures": Path("kinematics/neuraltextures"),
}

for method, folder in methods.items():

    for src in sorted(folder.glob("*.npz")):

        parts = src.stem.split("_")

        if len(parts) < 2:
            continue

        a, b = parts[0], parts[1]

        if (a, b) not in test_pairs:
            continue

        newstem = f"{method}_{src.stem}"

        shutil.copy2(
            src,
            out / f"{newstem}.npz"
        )

        rows.append((newstem, 1))


df = pd.DataFrame(
    rows,
    columns=["video_stem", "label"]
)

df.to_csv(
    "splits/ffpp_official/ffpp_test_labels.csv",
    index=False
)

print("\nClass counts:")
print(df["label"].value_counts().sort_index())

print("\nTotal videos:", len(df))

print("\nSource counts:")
print(
    df["video_stem"]
      .str.split("_")
      .str[0]
      .value_counts()
)
PY
```

Verify the generated test set:

```bash
find kinematics/ffpp_test -name "*.npz" | wc -l
head splits/ffpp_official/ffpp_test_labels.csv
```

---

## Stage 7: Evaluate First-Order Kinematics

```bash
python 05_score_and_evaluate.py \
    --kinematics_dir kinematics/ffpp_test \
    --model_path models/gmm_region_first.pkl \
    --labels_file splits/ffpp_official/ffpp_test_labels.csv \
    --representation first \
    --p_top 25 \
    --q_top 25 \
    --bootstrap 5000 \
    --out_csv results/first_ffpp.csv
```

---

## Stage 8: Evaluate Second-Order Kinematics

```bash
python 05_score_and_evaluate.py \
    --kinematics_dir kinematics/ffpp_test \
    --model_path models/gmm_region_second.pkl \
    --labels_file splits/ffpp_official/ffpp_test_labels.csv \
    --representation second \
    --p_top 25 \
    --q_top 25 \
    --bootstrap 5000 \
    --out_csv results/second_ffpp.csv
```

---

## Stage 9: Evaluate Combined Kinematics

```bash
python 05_score_and_evaluate.py \
    --kinematics_dir kinematics/ffpp_test \
    --model_path models/gmm_region_combined.pkl \
    --labels_file splits/ffpp_official/ffpp_test_labels.csv \
    --representation combined \
    --p_top 25 \
    --q_top 25 \
    --bootstrap 5000 \
    --out_csv results/combined_ffpp.csv
```

The evaluation reports:

- AUROC;
- AUPRC;
- stratified video-level bootstrap 95% confidence intervals;
- per-video anomaly scores.

---

## Stage 10: Paired Representation Comparison

The same test videos must be used for all three representations.

```bash
python 06_paired_comparison.py \
    --first_csv results/first_ffpp.csv \
    --second_csv results/second_ffpp.csv \
    --combined_csv results/combined_ffpp.csv \
    --bootstrap 5000
```

Paired bootstrap resampling uses identical video resamples for all
representations and evaluates differences including:

- second-order minus first-order AUROC;
- combined minus first-order AUROC;
- combined minus second-order AUROC;
- corresponding AUPRC differences.
```

Repeat for FF++ and AV-Deepfake1M.

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

