# Road-damage detection methodology (SVRDD, Kaggle T4, zero cost)

**Recommended final methodology:** train one strong single-stage detector (YOLO11m at native 1024 px) on the SVRDD benchmark, tune its operating point on validation, add test-time views only where validation shows a gain, then train one architecturally different detector (Faster R-CNN ResNet50-FPN) and keep it only if it finds objects the first model misses and Weighted Boxes Fusion of the two beats the first model alone. A third model (RT-DETR-L) is trained only if a measured gate opens. The official test split is evaluated once, with everything frozen.

F1 ≥ 0.75 is a target. Nothing in this document is a measured result of ours yet; every number below is either published by someone else or labelled as my estimate.

How to read the labels: **[Published]** = reported in a cited source. **[Inference]** = my reasoning from those sources. **[Proposed]** = what we do.

---

## 0. Why SVRDD and not RDD2022

- **[Published]** RDD2022's test annotations were never released; evaluation is only through the CRDDC leaderboard ([Arya et al. 2022](https://arxiv.org/abs/2209.08538), [sekilab repository](https://github.com/sekilab/RoadDamageDetector)).
- **[Inference]** The Kaggle mirror's train + val (32,628 + 5,757 = 38,385) equals the official RDD2022 training set, so its 9,035 "test" images are the official unlabeled test set. That is why the label files are empty. No legitimately labelled RDD2022 test set exists.
- **[Published]** All-country RDD2022 results for single models are low: YOLOv8s F1 ≈ 0.61 ([model card](https://huggingface.co/dronefreak/rdd2022-yolov8s), not peer-reviewed), YOLOv8x F1 0.668–0.696 ([RDD-YOLO](https://www.mdpi.com/2076-3417/14/8/3360)), YOLO11 F1 0.592 ([YOLO11-MBC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12736674)). The CRDDC'2022 winner reached 0.770 with a multi-model ensemble including a Swin-backbone two-stage detector ([Arya et al., CRDDC'2022](https://arxiv.org/abs/2211.11362)).
- **[Published]** SVRDD: 8,000 street-view images (1024×1024), 20,804 boxes, 7 classes, official 6,000/1,000/1,000 split with public test labels, CC-BY 4.0. Best of ten baseline detectors: YOLOv5, P 0.752 / R 0.671 / F1 0.709 / mAP50 0.733 ([Ren et al., Scientific Data 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11035563/)). Later papers report YOLOv8s F1 0.72 and YOLO11s F1 0.72 / mAP50 0.748 on it ([HPSM-YOLO](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.70249)).
- **[Inference]** On SVRDD the gap to the target is about 3 F1 points from a small model, against 5–15 points on RDD2022, and the result can be reported on a real held-out test set.

State this openly in the paper: the dataset was chosen for its labelled test split and comparable published baselines.

---

## 1. Exact architecture

| Role | Model | Initialisation | Input |
|---|---|---|---|
| A, primary | YOLO11m (Ultralytics) | COCO `yolo11m.pt` | 1024 px |
| B, complementary | Faster R-CNN, ResNet50-FPN (torchvision), 7 classes + background | COCO `FasterRCNN_ResNet50_FPN_Weights.DEFAULT` | short side 832–1024 (train), 1024 (test) |
| C, optional and gated | RT-DETR-L (Ultralytics) | COCO `rtdetr-l.pt` | 640 px |

Fusion: class-wise Weighted Boxes Fusion (WBF) of the models' canonical prediction tables, followed by a validation-tuned confidence threshold.

Classes: longitudinal crack, transverse crack, alligator crack, pothole, manhole cover, longitudinal patch, transverse patch. The archive contains no class-name file, so this id order (0–6) is inferred from box shapes and class frequencies and must be confirmed against `figs/class_examples.png`, which section 4 of the notebook produces. Names affect only the labels in tables and figures, not training or scores.

## 2. Why these models

- **YOLO11m.** [Published] YOLO-family models were the best single detectors in the SVRDD paper and in every CRDDC top-5 solution. [Inference] The published SVRDD baselines are n/s-size models; one size up at native resolution is the cheapest well-supported gain, and it makes our baseline directly comparable with the YOLO11s number in the literature.
- **Faster R-CNN R50-FPN.** [Published] The CRDDC'2022 winner attributed its margin to combining YOLO models (higher recall) with a two-stage detector (higher precision, better on small damage) ([Ding et al. 2022](https://ieeexplore.ieee.org/document/10021000/)). [Inference] It differs from YOLO in every component that shapes errors: anchor-based region proposals, RoI-pooled second-stage classification, a different backbone and a different loss. It is implemented in torchvision, so there is no MMDetection build risk on Kaggle. [Published caveat] Faster R-CNN was weaker than YOLOv5 on SVRDD in the dataset paper, so it is expected to be the weaker model alone; its value, if any, is in fusion.
- **RT-DETR-L (gated).** [Published] An RF-DETR model card reports higher mAP50 than YOLOv8 on RDD2022 (0.65 vs 0.61, not peer-reviewed), and RT-DETR variants are used for road damage ([improved RT-DETR](https://link.springer.com/article/10.1007/s11760-025-04165-w), China subset only). [Inference] There is no peer-reviewed evidence of a gain on SVRDD, it converges slowly, and it costs about 10 GPU-hours. It is therefore not trained unless the A+B oracle analysis shows unexploited recall.

## 3. Why not simply several YOLO models

- **[Published]** In CRDDC'2022 the heterogeneous ensemble scored 0.770; the best YOLO-only ensembles scored 0.743, 0.741, 0.727 and 0.726.
- **[Inference]** That comparison is confounded (different teams, and the winner's two-stage model had a Swin backbone we cannot afford), so it supports the hypothesis without proving it. The mechanism is sound: models of the same family trained on the same data make correlated errors, and WBF can only help where one member is right and the other is wrong.
- **[Proposed]** We test it directly instead of assuming it. Section 16 of the notebook counts the ground-truth objects found only by A, only by B, by both and by neither, and B is discarded if it is redundant. A negative outcome is a valid finding for the paper.

## 4. Experimental sequence

1. Dataset verification and analysis (CPU).
2. Train A. Report the baseline at confidence 0.25.
3. Tune A on validation: NMS IoU, one confidence threshold, per-class thresholds (kept only if they win under cross-fitting).
4. Add views to A by greedy forward selection: horizontal-flip TTA, a second scale (1280), tiling. Tiling is tried only if small-object recall is at least 10 points below medium/large recall.
5. Train B. Tune it the same way.
6. Measure complementarity and fuse (WBF vs NMS, IoU, weights).
7. Keep or discard B by the gate. Open or close the gate for C.
8. Freeze the final detector on validation.
9. Evaluate the test split once. Produce tables and figures.

## 5. What is trained

- YOLO11m, once.
- Faster R-CNN R50-FPN, once.
- RT-DETR-L, only if the gate opens and `RUN_C = True` is set by hand.

## 6. What is not trained

Other YOLO versions or sizes, RetinaNet, DETR, Cascade R-CNN or Swin backbones, per-class or per-scale specialist models, any RDD2022 model. The earlier YOLO11s runs are not used. No hyperparameter search over training settings: tuning is restricted to inference-time settings, which cost minutes instead of GPU-days.

## 7. Hyperparameters

**Model A (YOLO11m).** imgsz 1024, batch 8 (halved automatically on out-of-memory), up to 100 epochs, early stopping patience 20, SGD, lr0 0.01 with cosine decay to 1e-4, momentum 0.937, weight decay 5e-4, 3 warm-up epochs, AMP. Augmentation: mosaic 1.0 switched off for the last 10 epochs, horizontal flip 0.5, scale ±0.5, translate 0.1, default HSV; no vertical flip, rotation or mixup, because they blur the longitudinal/transverse distinction and wash out thin cracks. Checkpoint every epoch (`resume.pt`), extra snapshot every 5 epochs, best checkpoint by validation fitness.

Why 100 epochs with patience 20: [Inference] 6,000 images are few, so each epoch is short and convergence takes more epochs than on a large dataset; I have not verified the schedules used by the published SVRDD baselines. With patience 20 the run usually stops earlier, and the epoch count is reported.

**Model B (Faster R-CNN).** Multi-scale short side {832, 896, 960, 1024}, horizontal flip 0.5, batch 4, SGD lr 0.005, momentum 0.9, weight decay 1e-4, 500 warm-up iterations, 24 epochs with ×0.1 at epochs 16 and 22, AMP, gradient clipping at 10, checkpoint every epoch, best checkpoint by validation mAP50. Inference: score threshold 0.001, 300 detections per image, NMS IoU 0.5.

**Model C (RT-DETR-L, if reached).** imgsz 640, batch 8, AdamW lr 1e-4, weight decay 1e-4, 72 epochs, patience 15, no mosaic.

**Inference-time search space (validation only).** NMS IoU {0.5, 0.6, 0.7}; confidence 0.05–0.95 in steps of 0.01; fusion method {WBF-average, WBF-max, NMS}; fusion IoU {0.5, 0.55, 0.6, 0.7}; weight of B relative to A {0.25, 0.5, 0.75, 1, 1.5, 2}.

**Gates.** Post-processing step kept if F1 gain ≥ 0.003; tiling ≥ 0.005; ensemble ≥ 0.005 and B-only true positives ≥ 5% of ground truth; C considered only if B was kept, fused F1 < 0.75 and oracle recall exceeds fused recall by more than 0.05.

## 8. Runtime estimates

These are my estimates for one T4, not measurements. The notebook prints the real minutes per epoch after the first epochs.

| Step | Estimate |
|---|---|
| Verification + analysis | ~10 min CPU |
| A training | 6–8 min/epoch; 11–13 h if all 100 epochs run, less with early stopping |
| B training | 22–28 min/epoch; 9–11 h; expected GPU memory 9–12 GB at batch 4 |
| Inference, tuning, fusion, plots | ~45 min |
| C (only if triggered) | ~10 h |

Total without C: about 22–25 GPU-hours, inside Kaggle's free weekly quota (about 30 h). Sessions are limited to 12 h, so training pauses itself after 11 h and continues in the next session.

## 9. Kaggle code

`svrdd_detection.ipynb` (20 numbered sections matching the requested cell list). It is generated from `build/notebook_src.py`; edit the source and run `python build/build_notebook.py` to regenerate.

**How to run it**

1. Create a Kaggle notebook, import `svrdd_detection.ipynb`, set Accelerator = GPU T4 and Internet = On.
2. Data: either upload `SVRDD_YOLO.zip` from [Zenodo](https://zenodo.org/records/10100129) as a Kaggle dataset whose name contains "svrdd" and attach it, or do nothing and let the notebook download it (2.9 GB) each session.
3. Set `SMOKE = True` and run all cells (about 10 minutes). This is the first real test of the Ultralytics and GPU code paths. Fix or report anything that fails before spending quota.
4. Set `SMOKE = False`, then **Save Version → Save & Run All**. If training pauses for the session budget, the remaining cells skip themselves and the run still completes.
5. For the next session, add the previous version's output as an input (Add Input → your notebook). The notebook finds it automatically and resumes. Repeat until section 20 has run.
6. Section 4 prints a class-order sanity table. Check it once: transverse classes must be wider than tall.

## 10. Validation methodology

- The official split is used as published. Validation (1,000 images) is the only data used for checkpoint selection, thresholds, TTA, tiling, fusion method, fusion IoU and weights.
- **Detection F1.** Predictions above the confidence threshold are matched to ground truth per image and class, highest confidence first. A prediction is a true positive if its best unmatched ground-truth box of the same class has IoU ≥ 0.5; each ground-truth box can be matched once. Unmatched predictions are false positives, unmatched ground-truth boxes are false negatives. Precision, recall and F1 are computed from the pooled counts (micro), the CRDDC convention ([Arya et al.](https://arxiv.org/abs/2211.11362)). Per-class and macro values are reported alongside.
- **mAP50 and mAP50-95** use the same matching at IoU 0.5–0.95 with 101-point interpolation (the Ultralytics/COCO convention; a perfect detector scores 0.995).
- **Comparability with published SVRDD numbers.** Those are Ultralytics-style values (class-averaged P and R at the best-F1 confidence). The notebook records Ultralytics `val()` output for Model A next to our evaluator's numbers so both conventions can be quoted.
- **Guard against tuning optimism.** Per-class thresholds are adopted only if they beat a single threshold when fitted on one half of the validation images and scored on the other half (averaged over five random halvings).
- **Leakage check.** MD5 for exact duplicates and a 64-bit difference hash (Hamming ≤ 4) for near-duplicate candidates across splits; the rate is reported as a limitation if non-zero.

## 11. Final test methodology

- Section 18 runs only after sections 8–17 have frozen the member models, views, fusion settings and thresholds. It contains no tuning code.
- Four configurations are reported on test, all fixed beforehand: A at confidence 0.25, A with its tuned recipe, B with its tuned recipe, and the final detector.
- Report the test numbers as the result. Validation numbers are optimistic because settings were selected on them.
- Do not rerun with changed settings after seeing test results. If something must change, say so in the paper.

## 12. Ablation table (filled by the notebook into `tables/ablation_val.csv` and `ablation_test.csv`)

| Stage | P | R | F1 | macro F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| A0 YOLO11m baseline (conf 0.25) | | | | | | |
| A1 + tuned global confidence | | | | | | |
| A2 + per-class confidence (if kept) | | | | | | |
| A3 + TTA / multi-scale (if kept) | | | | | | |
| A4 + tiling (if run and kept) | | | | | | |
| A* best single-model recipe | | | | | | |
| B* Faster R-CNN best recipe | | | | | | |
| A+B fusion | | | | | | |
| A+B+C fusion (if used) | | | | | | |
| FINAL detector | | | | | | |

Published reference rows to place above it: YOLOv5 (dataset paper) F1 0.709 / mAP50 0.733; YOLOv8s F1 0.72 / mAP50 0.726; YOLO11s F1 0.72 / mAP50 0.748.

## 13. Paper-ready methodology description

> **Dataset.** Experiments use the Street View Road Damage Detection dataset (SVRDD; Ren et al., 2024), comprising 8,000 street-view images of 1024×1024 pixels with 20,804 bounding boxes in seven classes: longitudinal crack, transverse crack, alligator crack, pothole, longitudinal patch, transverse patch and manhole cover. The official partition of 6,000 training, 1,000 validation and 1,000 test images was used unchanged. Before training, all annotations were validated (class range, normalisation, degenerate and duplicate boxes), and exact and near-duplicate images across partitions were identified using MD5 digests and 64-bit difference hashes.
>
> **Baseline detector.** A YOLO11m detector initialised from COCO weights was trained at the native resolution of 1024 pixels using SGD (initial learning rate 0.01, cosine decay, momentum 0.937, weight decay 5×10⁻⁴), mosaic augmentation disabled for the final ten epochs, horizontal flipping and scale jitter. Vertical flipping, rotation and mixup were excluded because they alter the orientation cues that distinguish longitudinal from transverse damage. Training ran for at most 100 epochs with early stopping (patience 20).
>
> **Failure analysis and operating-point selection.** Baseline errors were analysed by class and by object size (COCO small/medium/large). The confidence threshold and NMS IoU were selected on the validation set. Class-specific thresholds were adopted only when they outperformed a single threshold under two-fold cross-fitting on the validation images. Additional inference views (horizontal-flip test-time augmentation, a second input scale, and overlapping tiles merged with a boundary-aware procedure) were added by greedy forward selection and retained only if validation F1 improved by a pre-specified margin.
>
> **Complementary detector and fusion.** To obtain errors that are less correlated with those of the single-stage baseline, a two-stage Faster R-CNN with a ResNet50-FPN backbone was trained on the same data (SGD, learning rate 0.005, multi-scale training, 24 epochs). Complementarity was quantified as the proportion of ground-truth objects detected by only one of the two models. Predictions were combined with Weighted Boxes Fusion (Solovyev et al., 2021); the fusion IoU, model weights and confidence threshold were selected on the validation set. The second detector was retained only if it contributed unique true positives for at least 5% of ground-truth objects and fusion improved validation F1 by at least 0.5 points.
>
> **Evaluation.** A detection was counted as a true positive when its IoU with a previously unmatched ground-truth box of the same class was at least 0.5, with predictions matched in descending order of confidence. Precision, recall and F1 were computed from pooled counts over all classes; per-class values, mAP@0.5 and mAP@0.5:0.95 are also reported. All design choices were fixed using the validation set, and the test set was evaluated once.

Adjust the last sentence of the third paragraph to the outcome (retained or discarded), and add the measured duplicate rate.

## 14. Risks and bottlenecks

- **The target may not be reached.** My estimate for the final test F1 is 0.74–0.78, roughly a 50–60% chance of ≥ 0.75. This is a judgement from published baselines, not a measurement.
- **Pothole is the weakest class** in the dataset paper (about 0.45 AP) and has few instances; it will pull macro F1 down and its threshold will be noisy.
- **Faster R-CNN may fail the gate.** It was weaker than YOLOv5 in the dataset paper. If it is discarded, the final detector is A with its tuned recipe, and the complementarity analysis is reported as a negative result.
- **Dataset layout and class order are assumptions.** I could not inspect the archive. The verification cell fails loudly on an unexpected layout and prints a class-order sanity table, but you must look at that table once.
- **Untested on Kaggle.** See the verification section below.
- **Session limits.** If a "Save & Run All" run is killed (not paused) by Kaggle, that session's progress is lost; the 11 h budget is there to prevent this.
- **Single-city data.** Claims must be limited to this domain; cross-dataset generalisation is not evaluated.
- **Validation set is small (1,000 images).** Gains under about 0.5 F1 points are within noise, which is why the gates exist.

## 15. Decision tree

```
Dataset verification + analysis
│
├─ label/layout problems? ── yes ─→ fix before training
│
Train A (YOLO11m @1024) → A0 baseline at conf 0.25
│
Tune NMS IoU and confidence on val
│   └─ per-class thresholds win under cross-fitting by ≥ 0.003? ── yes → keep │ no → single threshold
│
Flip TTA: val F1 +≥ 0.003? ── yes → keep │ no → drop
Second scale (1280): val F1 +≥ 0.003? ── yes → keep │ no → drop
Small-object recall ≥ 10 pts below medium/large?
│   ├─ no  → skip tiling
│   └─ yes → tiling: val F1 +≥ 0.005? ── yes → keep │ no → drop
│
A* = best single-model recipe
│
Train B (Faster R-CNN R50-FPN) → tune the same way
│
Complementarity + WBF(A, B)
│   B-only TPs ≥ 5% of GT  AND  F1(A+B) − F1(A*) ≥ 0.005 ?
│   ├─ no  → discard B → FINAL = A*
│   └─ yes → keep B
│            F1(A+B) ≥ 0.75 ? ── yes → FINAL = A+B
│            └─ no → oracle recall − fused recall > 0.05 ?
│                    ├─ no  → FINAL = A+B (a third model has little to add)
│                    └─ yes → train C (RT-DETR-L); F1(A+B+C) − F1(A+B) ≥ 0.005 ?
│                             ├─ yes → FINAL = A+B+C
│                             └─ no  → FINAL = A+B
│
Freeze FINAL on val → evaluate test once → tables + figures
```

---

## Measured results (Kaggle T4, single run, seed 42)

Everything in this section is measured. Detection F1 uses IoU ≥ 0.5 with class match, pooled over classes.

**Test set (1,000 images, 2,572 boxes), all settings fixed on validation**

| Stage | P | R | F1 | macro F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| YOLO11m baseline (conf 0.25) | 0.727 | 0.753 | 0.740 | 0.737 | 0.762 | 0.492 |
| YOLO11m, tuned (NMS 0.5, flip TTA, conf 0.44) | 0.791 | 0.712 | 0.749 | 0.747 | 0.778 | 0.502 |
| Faster R-CNN R50-FPN, tuned (conf 0.81) | 0.757 | 0.728 | 0.742 | 0.745 | 0.775 | 0.462 |
| **Final: YOLO11m + Faster R-CNN, WBF** | **0.794** | **0.750** | **0.771** | **0.773** | **0.808** | **0.528** |

**Validation set (1,000 images, 2,612 boxes)**

| Stage | P | R | F1 | mAP50 |
|---|---|---|---|---|
| A0 YOLO11m baseline (conf 0.25) | 0.726 | 0.752 | 0.739 | 0.776 |
| A1 + tuned confidence (0.36) and NMS IoU 0.5 | 0.804 | 0.707 | 0.752 | 0.778 |
| A3 + horizontal-flip TTA | 0.806 | 0.711 | 0.755 | 0.787 |
| B0 Faster R-CNN (conf 0.25) | 0.476 | 0.828 | 0.604 | 0.771 |
| B* Faster R-CNN, tuned | 0.762 | 0.732 | 0.747 | 0.771 |
| A+B fusion (WBF, IoU 0.5, weights 1.0 / 0.75) | 0.793 | 0.755 | 0.774 | 0.808 |

**Final detector per class (test)**

| Class | Boxes | P | R | F1 | AP50 |
|---|---|---|---|---|---|
| longitudinal crack | 616 | 0.742 | 0.700 | 0.720 | 0.772 |
| transverse crack | 436 | 0.742 | 0.658 | 0.697 | 0.703 |
| alligator crack | 182 | 0.807 | 0.830 | 0.818 | 0.854 |
| pothole | 124 | 0.780 | 0.685 | 0.730 | 0.754 |
| manhole cover | 419 | 0.868 | 0.845 | 0.856 | 0.896 |
| longitudinal patch | 500 | 0.834 | 0.806 | 0.820 | 0.870 |
| transverse patch | 295 | 0.796 | 0.739 | 0.766 | 0.807 |

**Complementarity on validation (at each model's operating point).** Of 2,612 objects, 64.2% were found by both models, 6.9% only by YOLO11m, 9.0% only by Faster R-CNN and 19.9% by neither. Oracle recall is 0.801; the fused detector reaches 0.755. 46% of YOLO11m's false positives are also produced by Faster R-CNN. Fusion improved F1 in six of seven classes on validation, most for transverse crack (0.659 → 0.701) and transverse patch (0.719 → 0.759).

**Techniques tested and rejected on validation.** Tiled inference (640 px tiles) lowered F1 by 4.6 points despite a 14-point recall gap between small and large objects. A second input scale (1280) gained 0.2 points, below the gate. Class-specific confidence thresholds did not beat a single threshold under cross-fitting for any model. Flip TTA lowered Faster R-CNN's F1 by 0.5 points. A third detector was not trained: the gate closed once the target was met.

**Training facts.** YOLO11m: early stopping at epoch 92 of 100, best epoch 72, 6.9 min/epoch. Faster R-CNN: 24 epochs, 16.8 min/epoch. Total about 18 GPU-hours.

**How to read these numbers**
- The target of F1 ≥ 0.75 is met on test (0.771) only by the fused detector; YOLO11m alone reaches 0.749.
- Comparison with published SVRDD figures (YOLOv5 F1 0.709 / mAP50 0.733; YOLO11s F1 0.72 / mAP50 0.748; HPSM-YOLO mAP50 0.785) is indicative only: those papers use the Ultralytics F1 convention and their own evaluation choices. Under that convention our YOLO11m scores F1 0.742 / mAP50 0.775 on validation.
- Results come from one training run per model. With about 2,600 test boxes, differences of roughly one F1 point are within sampling noise; the 2.2-point gain of fusion over YOLO11m on test is larger than that but has no confidence interval.
- 8.6% of test images have a near-duplicate candidate in the training set under a loose perceptual-hash criterion (no exact duplicates). Report this as a limitation of the dataset's split.
- The fusion result follows a post hoc correction, described next.

## Stage 2: maintenance priority (measured, `svrdd_priority.ipynb`)

**Design.** SVRDD has no priority labels. A priority index is defined as `score(image) = Σ weight[class] × (0.1 + extent)` over damage instances, where extent is the box diagonal over the image diagonal for linear damage (cracks, patches) and the square root of relative box area for areal damage (alligator crack, pothole). Weights: pothole 10, alligator crack 8, transverse crack 5, longitudinal crack 4, patches 2, manhole cover 0. The ordering follows pavement-condition practice; the numbers are our own. Applied to the ground-truth boxes, the index gives the reference priority; cut points at the 50th and 80th percentiles of the training scores (2.02 and 3.79) define Low / Medium / High. A predictor that sees only detector output is trained on the 1,000 validation images (5-fold cross-validation for model selection) and tested once on the 1,000 test images.

**Test results (reference = index on ground-truth boxes)**

| Method | Accuracy | Macro F1 | QWK | Recall of High | Spearman (score) |
|---|---|---|---|---|---|
| Index on YOLO11m detections | 0.783 | 0.752 | 0.770 | 0.681 | 0.798 |
| Index on Faster R-CNN detections | 0.795 | 0.769 | 0.790 | 0.750 | 0.831 |
| Index on final-detector detections (no learning) | 0.801 | 0.773 | 0.802 | 0.740 | 0.842 |
| Trained: gradient-boosting classifier (selected by CV) | 0.783 | 0.760 | 0.789 | 0.711 | n/a |
| Trained: ridge regression (not selected) | 0.794 | 0.776 | 0.813 | 0.740 | 0.877 |
| Trained: gradient-boosting regressor (not selected) | 0.791 | 0.775 | 0.811 | 0.750 | 0.882 |

QWK = quadratic weighted kappa (penalises Low/High confusion more than adjacent levels).

**Findings**
- Priority derived from automatic detection agrees substantially with priority derived from annotated damage: 80% of test images receive the same level, QWK 0.80.
- Better detection gives better priority: QWK rises from 0.770 (YOLO11m) to 0.802 (fused detector), in line with the detection results.
- The trained predictor selected by cross-validation does **not** improve on applying the index directly to the detections: QWK difference −0.012, 95% bootstrap interval −0.036 to +0.013 (no reliable difference). The two regressors score slightly higher than the baseline on the test set and rank images better (Spearman 0.88 against 0.84), but they were not the cross-validated choice and that difference was not tested, so it is descriptive only.
- The index is stable to its weights: with every class weight perturbed independently by up to ±30% (200 draws), 92% of test images keep their level on average (worst case 83%) and score ranks correlate 0.987 with the nominal index. Levels are more sensitive to the cut points (19% of images change level when the quantiles shift by 0.1), as expected for any thresholded scale.
- Road segments: file-name coordinates parsed for all images (extent 56 × 51 km). With 200 m cells and the 2,000 validation and test images, 96 segments have at least three images (326 images, 16%). For those, predicted and reference segment scores correlate 0.94, and 7 of the 10 highest-priority segments coincide.

**Limitations to state**
- The reference is the same index applied to human annotations. The evaluation measures robustness of the priority to detection errors, not agreement with engineering inspections, for which no data exist here.
- Potholes carry the highest weight but contribute only 6% of the total score, and one average pothole (1.38) contributes less than one average transverse crack (1.67), because potholes are small in the image. Extent is measured in pixels of a perspective image, so near damage counts for more than far damage.
- Segment coverage is sparse (16% of evaluated images); the segment ranking is a demonstration of the aggregation step, not a city-wide survey.
- The trained model adds no measurable accuracy over the rule on detections. It is reported because a learned component was required, and the negative result is stated.

## Deviation from the planned protocol (must be disclosed)

The first complete run evaluated the single models on the test set (YOLO11m recipe: test F1 0.7492; Faster R-CNN: 0.7424) and discarded Faster R-CNN at the fusion gate. Its validation log was inconsistent: fusion tuning reported F1 0.7735 for YOLO + Faster R-CNN, while the final fused evaluation reported 0.7513. The cause was an implementation error. Tuning fused only boxes with confidence ≥ 0.05; the final fusion included boxes down to 0.001, and because Weighted Boxes Fusion averages the scores within a cluster, near-zero duplicates lowered the scores of good detections.

The correction makes both steps use the same threshold (`FUSE_SKIP = 0.05`) and adds a check that the final fusion reproduces the tuned validation F1. The error was identified from validation results alone, the keep/discard rule was not changed, and the fused detector had not been evaluated on the test set before the correction. The single-model test numbers were already known at that point, so this is a post hoc correction and should be reported as one.

Consequence for reporting: mAP of the fused detector is computed on boxes whose pre-fusion confidence is at least 0.05.

## Verification status of the code

- **Checked locally:** every cell compiles; 10 groups of synthetic checks pass for the evaluator, threshold search, WBF/NMS, YOLO↔pixel conversion, tiling geometry and complementarity counts (`python tests/test_lib.py`). The tiling checks confirm that an object crossing a tile seam yields exactly one box, with or without a full-image detection.
- **Checked locally end to end on synthetic data** (`python tests/integration_smoke.py`): dataset discovery and label cleaning, duplicate detection, analysis, threshold tuning, view selection including tiling, real Faster R-CNN training and inference on CPU (random initialisation, one epoch), fusion, complementarity, the pause/resume logic, tables and all figures.
- **Not checked:** anything that needs Ultralytics, a GPU or the real dataset. In the local run Ultralytics was replaced by a stand-in, so YOLO/RT-DETR training, resume and prediction are verified only at the interface level. Pretrained-weight downloads, AMP on GPU, memory use and all runtime figures are also unverified. The `SMOKE = True` run on Kaggle is the first real test.

## References

- Arya et al., *RDD2022: A multi-national image dataset for automatic Road Damage Detection*, 2022. https://arxiv.org/abs/2209.08538
- Arya et al., *Crowdsensing-based Road Damage Detection Challenge (CRDDC'2022)*, IEEE BigData 2022. https://arxiv.org/abs/2211.11362
- Ding et al., *An Ensemble of One-Stage and Two-Stage Detectors Approach for Road Damage Detection*, IEEE BigData 2022. https://ieeexplore.ieee.org/document/10021000/
- Ren et al., *An annotated street view image dataset for automated road damage detection*, Scientific Data 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11035563/ (data: https://zenodo.org/records/10100129)
- Ren et al., *HPSM-YOLO: A Multi-Scale Feature Aggregation Network for Pavement Distress Detection*, IET Image Processing 2025. https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.70249
- *An efficient feature aggregation network for urban road damage detection using street-view images* (EFANet). https://www.sciencedirect.com/science/article/pii/S1093968726030963
- *RDD-YOLO: Road Damage Detection Algorithm Based on Improved YOLOv8*, Applied Sciences 2024. https://www.mdpi.com/2076-3417/14/8/3360
- *Research on a Road Crack Detection Method Based on YOLO11-MBC*, Sensors 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12736674
- *Road crack detection based on improved RT-DETR*, 2025. https://link.springer.com/article/10.1007/s11760-025-04165-w
- Akyon et al., *Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection*, 2022. https://arxiv.org/abs/2202.06934
- Solovyev et al., *Weighted boxes fusion: Ensembling boxes from different object detection models*, 2021. https://arxiv.org/abs/1910.13302
