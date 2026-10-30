# Road Damage Detection and Maintenance Priority Prediction

Validation-gated fusion of YOLO11m and Faster R-CNN for road damage detection on street-view images, followed by a
damage-weighted maintenance priority for each image and each 200 m road segment.

M Menmangai (23BCE1022), Vellore Institute of Technology, Chennai

> **Dataset:** the complete SVRDD dataset is in this repository's
> [Releases (dataset-v1)](https://github.com/24f1000138/road-damage-detection/releases/tag/dataset-v1), in two parts
> because GitHub limits a release file to 2 GB:
> [part00 (1.9 GB)](https://github.com/24f1000138/road-damage-detection/releases/download/dataset-v1/SVRDD_YOLO.zip.part00) and
> [part01 (0.9 GB)](https://github.com/24f1000138/road-damage-detection/releases/download/dataset-v1/SVRDD_YOLO.zip.part01).
> Download both and join them into `SVRDD_YOLO.zip` (commands in the [Dataset](#dataset) section).

## Results (SVRDD test set, 1,000 images, 7 classes)

| Model | Precision | Recall | F1 | mAP@0.5 |
|---|---|---|---|---|
| YOLO11m, tuned | 0.791 | 0.712 | 0.749 | 0.778 |
| Faster R-CNN, tuned | 0.757 | 0.728 | 0.742 | 0.775 |
| **Fused detector (VGDF)** | **0.794** | **0.750** | **0.771** | **0.808** |

Priority stage: the predicted level matches the level derived from annotated damage for 80.1% of test images
(quadratic weighted kappa 0.802); predicted and reference road-segment scores have a Spearman correlation of 0.94.

## Repository contents

| Path | What it is |
|---|---|
| `svrdd_detection.ipynb` | Stage 1: data checks, training of YOLO11m and Faster R-CNN, tuning, validation-gated fusion, test evaluation |
| `svrdd_priority.ipynb` | Stage 2: priority index, trained priority predictors, road-segment ranking |
| `svrdd_demo.ipynb` | Demo: runs the trained system on SVRDD test images or your own photos |
| `build/` | Sources of the three notebooks (`python build/build_notebook.py` regenerates them) |
| `tests/` | Unit and integration tests (CPU, synthetic data) |
| `report/figs/` | Result figures used in the paper |
| `METHODOLOGY.md` | Detailed methodology, protocol and measured results |

## Dataset

SVRDD: 8,000 street-view images (1024 × 1024) from Beijing with seven damage classes (longitudinal crack, transverse
crack, alligator crack, pothole, manhole cover, longitudinal patch, transverse patch), official split 6,000 / 1,000 / 1,000.

- Original source: https://zenodo.org/records/10100129 (`SVRDD_YOLO.zip`, 2.9 GB, CC BY 4.0)
- Copy in this repository: the **Releases** page (tag `dataset-v1`). GitHub limits a release file to 2 GB, so the zip is
  split into two parts. Download both parts into one folder and join them:
  - Windows (Command Prompt): `copy /b SVRDD_YOLO.zip.part00 + SVRDD_YOLO.zip.part01 SVRDD_YOLO.zip`
  - Linux / macOS: `cat SVRDD_YOLO.zip.part* > SVRDD_YOLO.zip`

Dataset paper: Ren, M., Zhang, X., Zhi, X., Wei, Y., & Feng, Z. (2024). An annotated street view image dataset for
automated road damage detection. *Scientific Data, 11*, 407. https://doi.org/10.1038/s41597-024-03263-7

## How to run (free Kaggle GPU)

1. Import `svrdd_detection.ipynb` into Kaggle, set Accelerator = GPU T4 and Internet = On.
2. Run all cells. If the dataset is not attached, the notebook downloads it from Zenodo inside Kaggle (a few minutes).
   Training is split across sessions; add the previous version's output as an input to resume.
3. Import `svrdd_priority.ipynb`, attach the detection notebook's output and run all cells.
4. Import `svrdd_demo.ipynb`, attach the detection notebook's output and run all cells to see predictions next to the
   ground truth.

## Tests

```
python tests/test_lib.py
python tests/test_priority.py
```
