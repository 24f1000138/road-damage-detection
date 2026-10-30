# %% [markdown]
# # SVRDD road-damage detection: YOLO11m + Faster R-CNN with validation-gated fusion
#
# Run order = cell order. Every expensive step checkpoints or caches under `WORK`, so a
# restarted session continues where it stopped (set `RESUME_FROM` to the saved output of the
# previous session). Set `SMOKE = True` first for a ~10 minute end-to-end check.
#
# Rules baked into the notebook:
# - **val** (1,000 images) is the only split used for thresholds, TTA, tiling, fusion and weights.
# - **test** (1,000 images) is touched only in section 18, with every setting frozen.
# - F1 is object-detection F1: a prediction is a TP when IoU >= 0.5 with an unmatched ground-truth
#   box of the same class (greedy, highest confidence first).

# %% [markdown]
# ## 1. Package installation

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=False)

# %% [markdown]
# ## 2. Imports

# %% tags=lib
import os, sys, json, math, time, random, hashlib, shutil, zipfile, itertools, warnings
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

# %%
# Kaggle's "T4 x2" exposes two GPUs. Only one is used, and Ultralytics divides the 4 CPU cores by the
# number of visible GPUs when it sizes its data loader, so the second GPU is hidden to keep all 4 workers.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch
import torchvision
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

print("torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| cuda", torch.cuda.is_available(), "| gpus", torch.cuda.device_count())

# %% [markdown]
# ## 3. Configuration

# %%
SMOKE = False                      # True: 200 train / 60 val / 60 test images, 1 epoch each
PAUSE_TEST = False                 # with SMOKE: rehearses pause/resume. Click "Run All" three times in one session:
                                   # run 1 pauses model A, run 2 resumes A and pauses B, run 3 resumes B and finishes
DATA_ROOT = Path(os.environ.get("SVRDD_ROOT", "/kaggle/input"))   # a folder whose name contains "svrdd" is searched for here
WORK = Path(os.environ.get("SVRDD_WORK", "/kaggle/working/svrdd_run"))          # checkpoints, caches, tables, figures (saved as notebook output)
TEMP = Path(os.environ.get("SVRDD_TEMP", "/kaggle/temp" if Path("/kaggle").exists() else str(WORK.parent / "svrdd_tmp")))  # not saved
RESUME_FROM = None                 # None = auto-detect an earlier run's output under /kaggle/input; or give its path
SESSION_BUDGET_H = 11.0            # training pauses cleanly after this many hours so the 12 h session limit never kills a run
DOWNLOAD_IF_MISSING = True         # fetch SVRDD_YOLO.zip (2.9 GB, CC-BY 4.0) from Zenodo if not attached
ZENODO_URL = "https://zenodo.org/api/records/10100129/files/SVRDD_YOLO.zip/content"
SEED = 42

# Used only if the dataset ships no class file. Section 4 prints a sanity check for this order.
# The archive ships no class file. This order is inferred from box shapes and class frequencies and
# MUST be confirmed once against the example crops that section 4 saves (figs/class_examples.png).
# Names only label tables and figures; training and scores do not depend on them.
DEFAULT_CLASSES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole",
                   "manhole_cover", "longitudinal_patch", "transverse_patch"]

# Model A: primary detector
CFG_A = dict(model="yolo11m.pt", imgsz=1024, batch=8, epochs=100, patience=20,
             optimizer="SGD", lr0=0.01, lrf=0.01, cos_lr=True, momentum=0.937, weight_decay=5e-4,
             warmup_epochs=3, amp=True, mosaic=1.0, close_mosaic=10, mixup=0.0,
             fliplr=0.5, flipud=0.0, degrees=0.0, scale=0.5, translate=0.1,
             save_period=5, workers=4, seed=SEED, plots=True)

# Model B: complementary two-stage detector
CFG_B = dict(train_sizes=(832, 896, 960, 1024), max_size=1024, batch=4, epochs=24,
             lr=0.005, momentum=0.9, weight_decay=1e-4, warmup_iters=500, milestones=(16, 22),
             workers=4, pretrained=True)

# Model C: optional, only if the gate in section 16 opens AND you set RUN_C = True
RUN_C = False
CFG_C = dict(model="rtdetr-l.pt", imgsz=640, batch=8, epochs=72, patience=15,
             optimizer="AdamW", lr0=1e-4, lrf=0.01, weight_decay=1e-4, warmup_epochs=3, amp=True,
             fliplr=0.5, flipud=0.0, mosaic=0.0, mixup=0.0, save_period=5, workers=4, seed=SEED)

# Decision gates (absolute F1 or recall differences on val)
GATE = dict(post=0.003, tiling=0.005, ensemble=0.005, b_only_tp=0.05, small_gap=0.10, c_headroom=0.05)
FORCE_TILING = False               # run the tiling experiment even if the small-object gate is closed
TARGET_F1 = 0.75
EVAL_TEST = True                   # section 18; leave False until everything above is frozen if you prefer

INFER_BATCH = 8
LOW_CONF = 0.001                   # predictions are stored down to this score (needed for mAP)
FUSE_SKIP = 0.05                   # boxes below this score are left out of model fusion. WBF averages the scores in a
                                   # cluster, so near-zero duplicates would drag down good detections. Used for tuning
                                   # AND for the final fusion, so the tuned configuration is the one that is evaluated.
MAX_DET = 300

if SMOKE:
    WORK = WORK.parent / (WORK.name + "_smoke"); TEMP = TEMP / "smoke"
    CFG_A.update(epochs=1, imgsz=640, batch=8, patience=1, close_mosaic=0, plots=False)
    CFG_B.update(epochs=1, milestones=(1,), warmup_iters=10, train_sizes=(640,), max_size=640)
    CFG_C.update(epochs=1)
    if PAUSE_TEST:
        WORK = WORK.parent / (WORK.name + "_pausetest")
        CFG_A.update(epochs=3, patience=3); CFG_B.update(epochs=2, milestones=(2,))
IMGSZ = CFG_A["imgsz"]
DATA_DIR = TEMP / "data"           # normalised dataset copy (symlinks + cleaned labels)

for d in ("runs", "preds", "figs", "tables"):
    (WORK / d).mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
seed_everything()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
T_SESSION = time.time()
PAUSED = False                     # set by a training cell that ran out of session time; later cells then skip
def budget_used():
    return (time.time() - T_SESSION) / 3600 > SESSION_BUDGET_H
def out_of_time(tag):
    """Asked after each training epoch. In the pause test it answers True exactly once per model."""
    if SMOKE and PAUSE_TEST:
        marker = WORK / f"pause_test_fired_{tag}"
        if marker.exists():
            return False
        marker.write_text("fired"); return True
    return budget_used()

# Restore checkpoints/caches saved by an earlier session (never overwrites files already in WORK)
if RESUME_FROM is None and Path("/kaggle/input").exists():
    found = sorted((p.parent for p in Path("/kaggle/input").rglob("state.json") if p.parent.name == WORK.name),
                   key=lambda p: (p / "state.json").stat().st_mtime)
    RESUME_FROM = str(found[-1]) if found else None
if RESUME_FROM and Path(RESUME_FROM).exists():
    n = 0
    for src in Path(RESUME_FROM).rglob("*"):
        if src.is_file():
            dst = WORK / src.relative_to(RESUME_FROM)
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst); n += 1
    print(f"restored {n} files from {RESUME_FROM}")

# `%%stage` cells (everything from section 6 on) are skipped once training has paused, so a
# "Save & Run All" run always finishes cleanly and its output can be attached to the next session.
try:
    from IPython import get_ipython
    from IPython.core.magic import register_cell_magic
    @register_cell_magic
    def stage(line, cell):
        if PAUSED:
            print("skipped: training paused for this session. Save the output and run the notebook again to continue.")
            return
        get_ipython().run_cell(cell).raise_error()
except Exception:
    pass

# Persistent decisions and ablation rows
STATE_PATH, ABL_PATH = WORK / "state.json", WORK / "tables" / "ablation.csv"
STATE = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
def save_state(**kw):
    STATE.update(kw); STATE_PATH.write_text(json.dumps(STATE, indent=2, default=float))
def log_row(stage, split, res, note=""):
    """Upsert one row of the ablation table (keyed by stage + split). res=None removes the row,
    so a step that is rejected on a rerun does not leave a stale entry behind."""
    df = pd.read_csv(ABL_PATH) if ABL_PATH.exists() else pd.DataFrame()
    if len(df):
        df = df[~((df.stage == stage) & (df.split == split))]
    if res is not None:
        row = dict(stage=stage, split=split, **{k: round(float(v), 4) for k, v in res["summary"].items()}, note=note)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(ABL_PATH, index=False)
    return df
print("WORK =", WORK, "| SMOKE =", SMOKE, "| device =", DEVICE)

# %% [markdown]
# ### 3b. Evaluation, fusion and tiling library
# Pure NumPy/pandas. Every detector is reduced to the same table
# `image_id, class_id, confidence, x1, y1, x2, y2` (pixels, canonical class ids) and scored by the
# same code, so numbers are comparable across architectures.

# %% tags=lib
PRED_COLS = ["image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]
BOX = ["x1", "y1", "x2", "y2"]
IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)
_trapz = getattr(np, "trapezoid", None) or np.trapz

def empty_preds():
    return pd.DataFrame({c: pd.Series(dtype=("object" if c == "image_id" else "int64" if c == "class_id" else "float64"))
                         for c in PRED_COLS})

def make_preds(image_ids, cls, conf, boxes):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return pd.DataFrame({"image_id": list(image_ids), "class_id": np.asarray(cls, dtype=np.int64),
                         "confidence": np.asarray(conf, dtype=np.float64),
                         "x1": boxes[:, 0], "y1": boxes[:, 1], "x2": boxes[:, 2], "y2": boxes[:, 3]})

def yolo_to_xyxy(cx, cy, w, h, W, H):
    return (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H

def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    return (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H

def box_area(b):
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)

def box_inter(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4); b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    return wh[..., 0] * wh[..., 1]

def box_iou(a, b):
    inter = box_inter(a, b)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return inter / np.maximum(union, 1e-9)

# ---------------------------------------------------------------- matching and metrics
def match_predictions(pred, gt, iou_thrs=IOU_THRS):
    """Greedy detection matching, per image and class, highest confidence first.

    A prediction is a TP at threshold t if its best still-unmatched same-class GT box has IoU >= t.
    Because higher-confidence predictions are processed first, the match of a prediction does not
    depend on lower-confidence ones, so one matching serves every confidence threshold.
    Returns tp [n_pred, n_thr] (bool) and pred_gt [n_pred] (row index into gt at iou_thrs[0], or -1).
    Both are aligned with pred.reset_index(drop=True) / gt.reset_index(drop=True).
    """
    pred = pred.reset_index(drop=True); gt = gt.reset_index(drop=True)
    tp = np.zeros((len(pred), len(iou_thrs)), dtype=bool)
    pred_gt = np.full(len(pred), -1, dtype=np.int64)
    if len(pred) == 0 or len(gt) == 0:
        return tp, pred_gt
    pb, gb, conf = pred[BOX].to_numpy(float), gt[BOX].to_numpy(float), pred["confidence"].to_numpy(float)
    gt_groups = gt.groupby(["image_id", "class_id"]).indices
    for key, pidx in pred.groupby(["image_id", "class_id"]).indices.items():
        gidx = gt_groups.get(key)
        if gidx is None:
            continue
        pidx = pidx[np.argsort(-conf[pidx], kind="stable")]
        iou = box_iou(pb[pidx], gb[gidx])
        for ti, thr in enumerate(iou_thrs):
            used = np.zeros(len(gidx), dtype=bool)
            for r in range(len(pidx)):
                row = np.where(used, -1.0, iou[r])
                j = int(row.argmax())
                if row[j] >= thr:
                    used[j] = True
                    tp[pidx[r], ti] = True
                    if ti == 0:
                        pred_gt[pidx[r]] = gidx[j]
    return tp, pred_gt

def average_precision(tp_sorted, n_gt):
    """AP from TP flags sorted by descending confidence (101-point interpolation, as in Ultralytics/COCO)."""
    if n_gt == 0:
        return np.nan
    if len(tp_sorted) == 0:
        return 0.0
    tpc = np.cumsum(tp_sorted); fpc = np.cumsum(~tp_sorted)
    rec = tpc / n_gt; prec = tpc / np.maximum(tpc + fpc, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]]); mpre = np.concatenate([[1.0], prec, [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(_trapz(np.interp(x, mrec, mpre), x))

def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f

def conf_vector(conf_thr, n_classes):
    return np.broadcast_to(np.asarray(conf_thr, dtype=np.float64), (n_classes,)).copy()

def evaluate(pred, gt, n_classes, conf_thr=0.25, class_names=None, match=None):
    """Detection P/R/F1 at an operating point (scalar or per-class confidence) plus mAP50 / mAP50-95."""
    pred = pred.reset_index(drop=True); gt = gt.reset_index(drop=True)
    tp, pred_gt = match if match is not None else match_predictions(pred, gt)
    cls = pred["class_id"].to_numpy(np.int64); conf = pred["confidence"].to_numpy(float)
    thr = conf_vector(conf_thr, n_classes)
    keep = conf >= thr[cls] if len(pred) else np.zeros(0, dtype=bool)
    gcls = gt["class_id"].to_numpy(np.int64)
    rows, TP, FP, FN = [], 0, 0, 0
    for c in range(n_classes):
        m = cls == c
        n_gt = int((gcls == c).sum())
        tpc = int((tp[m, 0] & keep[m]).sum()); fpc = int((~tp[m, 0] & keep[m]).sum()); fnc = n_gt - tpc
        p, r, f = _prf(tpc, fpc, fnc)
        order = np.argsort(-conf[m], kind="stable")
        aps = [average_precision(tp[m][order, ti], n_gt) for ti in range(tp.shape[1])]
        rows.append(dict(class_id=c, name=class_names[c] if class_names else str(c), n_gt=n_gt,
                         TP=tpc, FP=fpc, FN=fnc, P=p, R=r, F1=f, AP50=aps[0], AP50_95=float(np.mean(aps)),
                         conf_thr=float(thr[c])))
        TP += tpc; FP += fpc; FN += fnc
    per_class = pd.DataFrame(rows)
    p, r, f = _prf(TP, FP, FN)
    present = per_class[per_class.n_gt > 0]
    summary = dict(P=p, R=r, F1=f, macro_P=present.P.mean(), macro_R=present.R.mean(), macro_F1=present.F1.mean(),
                   mAP50=present.AP50.mean(), mAP50_95=present.AP50_95.mean(), TP=TP, FP=FP, FN=FN)
    summary = {k: (v if isinstance(v, int) else float(v)) for k, v in summary.items()}
    return dict(summary=summary, per_class=per_class, keep=keep, tp=tp, pred_gt=pred_gt)

def pr_curve(pred, gt, class_id, match=None):
    pred = pred.reset_index(drop=True)
    tp, _ = match if match is not None else match_predictions(pred, gt)
    m = (pred["class_id"] == class_id).to_numpy()
    order = np.argsort(-pred["confidence"].to_numpy(float)[m], kind="stable")
    t = tp[m][order, 0]
    n_gt = int((gt["class_id"] == class_id).sum())
    tpc = np.cumsum(t); fpc = np.cumsum(~t)
    return tpc / max(n_gt, 1), tpc / np.maximum(tpc + fpc, 1e-9), pred["confidence"].to_numpy(float)[m][order]

# ---------------------------------------------------------------- threshold optimisation
CONF_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)

def _counts_over_grid(conf, tp0, n_gt, grid=CONF_GRID):
    tps = np.array([(tp0 & (conf >= t)).sum() for t in grid], dtype=float)
    fps = np.array([(~tp0 & (conf >= t)).sum() for t in grid], dtype=float)
    return tps, fps, n_gt - tps

def _f1(tps, fps, fns):
    return 2 * tps / np.maximum(2 * tps + fps + fns, 1e-9)

def best_global_threshold(pred, gt, match=None, grid=CONF_GRID):
    """Single confidence threshold maximising micro F1."""
    pred = pred.reset_index(drop=True)
    tp, _ = match if match is not None else match_predictions(pred, gt, IOU_THRS[:1])
    f = _f1(*_counts_over_grid(pred["confidence"].to_numpy(float), tp[:, 0], len(gt), grid))
    i = int(f.argmax())
    return float(grid[i]), float(f[i]), f

def best_class_thresholds(pred, gt, n_classes, match=None, grid=CONF_GRID, fallback=None):
    """Per-class confidence thresholds, each maximising that class's F1 (classes without GT use fallback)."""
    pred = pred.reset_index(drop=True)
    tp, _ = match if match is not None else match_predictions(pred, gt, IOU_THRS[:1])
    if fallback is None:
        fallback = best_global_threshold(pred, gt, (tp, None), grid)[0]
    cls = pred["class_id"].to_numpy(np.int64); conf = pred["confidence"].to_numpy(float)
    gcls = gt["class_id"].to_numpy(np.int64)
    out = np.full(n_classes, fallback, dtype=float)
    for c in range(n_classes):
        n_gt = int((gcls == c).sum()); m = cls == c
        if n_gt == 0 or m.sum() == 0:
            continue
        out[c] = grid[int(_f1(*_counts_over_grid(conf[m], tp[m, 0], n_gt, grid)).argmax())]
    return out

def micro_f1(pred, gt, n_classes, conf_thr, match=None):
    pred = pred.reset_index(drop=True)
    tp, _ = match if match is not None else match_predictions(pred, gt, IOU_THRS[:1])
    thr = conf_vector(conf_thr, n_classes)
    keep = pred["confidence"].to_numpy(float) >= thr[pred["class_id"].to_numpy(np.int64)]
    TP = int((tp[:, 0] & keep).sum()); FP = int((~tp[:, 0] & keep).sum())
    return _prf(TP, FP, len(gt) - TP)[2]

def crossfit_f1(pred, gt, n_classes, mode, seed=0, folds=2):
    """Honest estimate of tuned F1: thresholds are fitted on one half of the images and scored on the other.

    mode: 'global' or 'class'. Used to decide whether per-class thresholds beat a single threshold
    without rewarding the extra degrees of freedom."""
    ids = np.array(sorted(set(gt["image_id"]) | set(pred["image_id"])))
    rng = np.random.RandomState(seed); rng.shuffle(ids)
    parts = np.array_split(ids, folds)
    scores = []
    for k in range(folds):
        test_ids = set(parts[k]); fit_ids = set(ids) - test_ids
        pf, gf = pred[pred.image_id.isin(fit_ids)], gt[gt.image_id.isin(fit_ids)]
        pt, gtt = pred[pred.image_id.isin(test_ids)], gt[gt.image_id.isin(test_ids)]
        if mode == "global":
            thr = best_global_threshold(pf, gf)[0]
        else:
            thr = best_class_thresholds(pf, gf, n_classes)
        scores.append(micro_f1(pt, gtt, n_classes, thr))
    return float(np.mean(scores))

# ---------------------------------------------------------------- fusion: WBF and NMS
def nms_numpy(boxes, scores, iou_thr):
    order = np.argsort(-scores, kind="stable"); keep = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1:
            break
        iou = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=np.int64)

def wbf_single(boxes, scores, src, weights, iou_thr=0.55, conf_type="avg"):
    """Weighted Boxes Fusion (Solovyev et al., 2021) for one image and one class.

    boxes [n,4], scores [n], src [n] = index of the model/view that produced each box.
    conf_type 'avg': fused score = mean weighted score * min(n_boxes, sum_w) / sum_w, which penalises
                     boxes that only some models found (use for model ensembles).
              'max': fused score = highest member score (use when merging views/tiles of one model,
                     where a view legitimately cannot see every object).
    """
    w = np.asarray(weights, dtype=float)
    s = scores * w[src]
    order = np.argsort(-s, kind="stable")
    clusters, fused = [], np.zeros((0, 4))
    for i in order:
        j = -1
        if len(fused):
            iou = box_iou(boxes[i:i + 1], fused)[0]
            j = int(iou.argmax())
            if iou[j] <= iou_thr:
                j = -1
        if j < 0:
            clusters.append([i]); fused = np.vstack([fused, boxes[i]])
        else:
            clusters[j].append(i)
            idx = np.array(clusters[j])
            fused[j] = (boxes[idx] * s[idx, None]).sum(0) / max(s[idx].sum(), 1e-9)
    out_s = np.zeros(len(clusters))
    for k, idx in enumerate(clusters):
        idx = np.array(idx)
        if conf_type == "max":
            out_s[k] = scores[idx].max()
        else:
            out_s[k] = s[idx].mean() * min(len(idx), w.sum()) / w.sum()
    return fused, out_s

def fuse_predictions(dfs, weights=None, method="wbf", iou_thr=0.55, conf_type="avg", skip_thr=0.0):
    """Fuse several canonical prediction tables (models or views) image by image, class by class."""
    weights = np.ones(len(dfs)) if weights is None else np.asarray(weights, dtype=float)
    parts = []
    for i, d in enumerate(dfs):
        d = d[d["confidence"] >= skip_thr][PRED_COLS].copy(); d["src"] = i; parts.append(d)
    allp = pd.concat(parts, ignore_index=True)
    if len(allp) == 0:
        return empty_preds()
    B = allp[BOX].to_numpy(float); S = allp["confidence"].to_numpy(float); SRC = allp["src"].to_numpy(np.int64)
    ids, cls, conf, boxes = [], [], [], []
    for (img, c), idx in allp.groupby(["image_id", "class_id"]).indices.items():
        if method == "wbf":
            fb, fs = wbf_single(B[idx], S[idx], SRC[idx], weights, iou_thr, conf_type)
        else:  # plain NMS on weight-scaled scores
            s = S[idx] * weights[SRC[idx]] / weights.max()
            k = nms_numpy(B[idx], s, iou_thr); fb, fs = B[idx][k], s[k]
        ids += [img] * len(fs); cls += [c] * len(fs); conf += list(fs); boxes.append(fb)
    out = make_preds(ids, cls, conf, np.vstack(boxes) if boxes else np.zeros((0, 4)))
    return out.sort_values(["image_id", "confidence"], ascending=[True, False]).reset_index(drop=True)

# ---------------------------------------------------------------- tiling
def tile_starts(length, tile, min_overlap=0.2):
    """Start offsets of tiles covering [0, length) with at least min_overlap between neighbours; last tile is flush."""
    if tile >= length:
        return [0]
    n = max(2, math.ceil((length - tile * min_overlap) / (tile * (1 - min_overlap))))
    return [int(round(i * (length - tile) / (n - 1))) for i in range(n)]

def make_tiles(W, H, tile=640, min_overlap=0.2):
    return [(x, y, min(x + tile, W), min(y + tile, H))
            for y in tile_starts(H, tile, min_overlap) for x in tile_starts(W, tile, min_overlap)]

def truncated_flags(boxes, tile, W, H, tol=2.0):
    """True where a box (already in full-image coordinates) touches an edge of its tile that is NOT an image edge.
    Such a box is probably a fragment of an object that continues in the neighbouring tile."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    tx1, ty1, tx2, ty2 = tile
    f = np.zeros(len(boxes), dtype=bool)
    if tx1 > 0: f |= boxes[:, 0] <= tx1 + tol
    if ty1 > 0: f |= boxes[:, 1] <= ty1 + tol
    if tx2 < W: f |= boxes[:, 2] >= tx2 - tol
    if ty2 < H: f |= boxes[:, 3] >= ty2 - tol
    return f

def merge_tile_boxes(boxes, scores, tile_id, trunc, stitch_ios=0.3, contain_ios=0.6):
    """Resolve tile-boundary fragments for one image and one class (tile_id == -1 marks full-image boxes).

    1. Stitch: truncated boxes from different tiles that overlap (intersection over the smaller box
       >= stitch_ios) are fragments of one object -> replace them by their union (score = max).
    2. Contain: a still-truncated box lying mostly inside another box (>= contain_ios of its own area)
       duplicates part of that object -> drop it.
    The remaining boxes go to WBF, which merges full-view and tile detections of the same object.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4).copy(); scores = np.asarray(scores, dtype=float).copy()
    tile_id = np.asarray(tile_id).copy(); trunc = np.asarray(trunc, dtype=bool).copy()
    n = len(boxes)
    if n == 0:
        return boxes, scores, tile_id
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    t = np.where(trunc)[0]
    if len(t) > 1:
        inter = box_inter(boxes[t], boxes[t]); area = box_area(boxes[t])
        ios = inter / np.maximum(np.minimum(area[:, None], area[None, :]), 1e-9)
        for a in range(len(t)):
            for b in range(a + 1, len(t)):
                if tile_id[t[a]] != tile_id[t[b]] and ios[a, b] >= stitch_ios:
                    parent[find(t[a])] = find(t[b])
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    nb, ns, nt, ntr = [], [], [], []
    for members in groups.values():
        m = np.array(members)
        if len(m) == 1:
            nb.append(boxes[m[0]]); ns.append(scores[m[0]]); nt.append(tile_id[m[0]]); ntr.append(trunc[m[0]])
        else:  # union of fragments; no longer counts as truncated
            nb.append(np.concatenate([boxes[m, :2].min(0), boxes[m, 2:].max(0)]))
            ns.append(scores[m].max()); nt.append(-2); ntr.append(False)
    boxes, scores, tile_id, trunc = np.array(nb), np.array(ns), np.array(nt), np.array(ntr, dtype=bool)
    if trunc.any() and len(boxes) > 1:
        inter = box_inter(boxes, boxes); np.fill_diagonal(inter, 0)
        frac = inter / np.maximum(box_area(boxes)[:, None], 1e-9)      # share of box i inside box j
        bigger = box_area(boxes)[None, :] > box_area(boxes)[:, None]
        drop = trunc & ((frac >= contain_ios) & bigger).any(1)
        boxes, scores, tile_id = boxes[~drop], scores[~drop], tile_id[~drop]
    return boxes, scores, tile_id

def merge_full_and_tiles(full_df, tile_df, iou_thr=0.55):
    """full_df: canonical predictions on whole images. tile_df: canonical + columns tile_id, trunc.
    Returns one canonical table without double-counted boundary objects."""
    f = full_df[PRED_COLS].copy(); f["tile_id"] = -1; f["trunc"] = False
    allp = pd.concat([f, tile_df[PRED_COLS + ["tile_id", "trunc"]]], ignore_index=True)
    if len(allp) == 0:
        return empty_preds()
    B = allp[BOX].to_numpy(float); S = allp["confidence"].to_numpy(float)
    T = allp["tile_id"].to_numpy(np.int64); TR = allp["trunc"].to_numpy(bool)
    ids, cls, conf, boxes = [], [], [], []
    for (img, c), idx in allp.groupby(["image_id", "class_id"]).indices.items():
        b, s, t = merge_tile_boxes(B[idx], S[idx], T[idx], TR[idx])
        if len(b) == 0:
            continue
        src = (t != -1).astype(np.int64)                      # 0 = full view, 1 = tiles
        fb, fs = wbf_single(b, s, src, [1.0, 1.0], iou_thr, conf_type="max")
        ids += [img] * len(fs); cls += [c] * len(fs); conf += list(fs); boxes.append(fb)
    out = make_preds(ids, cls, conf, np.vstack(boxes) if boxes else np.zeros((0, 4)))
    return out.sort_values(["image_id", "confidence"], ascending=[True, False]).reset_index(drop=True)

# ---------------------------------------------------------------- error analysis
SIZE_BINS = [("small", 0, 32 ** 2), ("medium", 32 ** 2, 96 ** 2), ("large", 96 ** 2, float("inf"))]

def size_bucket(gt):
    a = box_area(gt[BOX].to_numpy(float))
    return np.select([a < 32 ** 2, a < 96 ** 2], ["small", "medium"], default="large")

def gt_hits(res, n_gt):
    """Boolean [n_gt]: ground-truth boxes found by the kept (above-threshold) true positives of an evaluate() result."""
    hit = np.zeros(n_gt, dtype=bool)
    idx = res["pred_gt"][res["keep"] & res["tp"][:, 0]]
    hit[idx[idx >= 0]] = True
    return hit

def recall_by_size(gt, hit):
    b = size_bucket(gt)
    return pd.DataFrame([dict(size=name, n_gt=int((b == name).sum()),
                              recall=float(hit[b == name].mean()) if (b == name).any() else np.nan)
                         for name, _, _ in SIZE_BINS])

def complementarity(gt, hit_a, hit_b, class_names):
    """Which ground-truth objects does each model find? Counts sum to n_gt in every row."""
    gt = gt.reset_index(drop=True); b = size_bucket(gt); cls = gt["class_id"].to_numpy()
    def row(name, m):
        n = int(m.sum())
        both, a_only, b_only = int((hit_a & hit_b & m).sum()), int((hit_a & ~hit_b & m).sum()), int((~hit_a & hit_b & m).sum())
        neither = n - both - a_only - b_only
        return dict(group=name, n_gt=n, both=both, A_only=a_only, B_only=b_only, neither=neither,
                    recall_A=(both + a_only) / max(n, 1), recall_B=(both + b_only) / max(n, 1),
                    oracle_recall=(n - neither) / max(n, 1), B_only_share=b_only / max(n, 1))
    rows = [row("all", np.ones(len(gt), dtype=bool))]
    rows += [row(class_names[c], cls == c) for c in range(len(class_names))]
    rows += [row(name, b == name) for name, _, _ in SIZE_BINS]
    return pd.DataFrame(rows)

def shared_false_positives(pred_a, res_a, pred_b, res_b, iou_thr=0.5):
    """Share of each model's false positives that the other model also produces (same class, IoU >= thr)."""
    fa = pred_a.reset_index(drop=True)[res_a["keep"] & ~res_a["tp"][:, 0]]
    fb = pred_b.reset_index(drop=True)[res_b["keep"] & ~res_b["tp"][:, 0]]
    gb = fb.groupby(["image_id", "class_id"]).indices
    Bb = fb[BOX].to_numpy(float); Ba = fa[BOX].to_numpy(float)
    shared_a = 0; hit_b = np.zeros(len(fb), dtype=bool)
    for key, ia in fa.groupby(["image_id", "class_id"]).indices.items():
        ib = gb.get(key)
        if ib is None:
            continue
        iou = box_iou(Ba[ia], Bb[ib])
        shared_a += int((iou.max(1) >= iou_thr).sum()); hit_b[ib[iou.max(0) >= iou_thr]] = True
    return dict(fp_A=len(fa), fp_B=len(fb), fp_A_shared=shared_a, fp_B_shared=int(hit_b.sum()),
                A_shared_frac=shared_a / max(len(fa), 1), B_shared_frac=float(hit_b.sum()) / max(len(fb), 1))

def detection_confusion(pred, gt, n_classes, conf_thr, iou_thr=0.5):
    """Confusion matrix with a background row/column: rows = true class, columns = predicted class.
    Matching here ignores class (best IoU >= thr), so off-diagonal cells are localised-but-misclassified boxes."""
    thr = conf_vector(conf_thr, n_classes)
    p = pred[pred["confidence"].to_numpy(float) >= thr[pred["class_id"].to_numpy(np.int64)]]
    M = np.zeros((n_classes + 1, n_classes + 1), dtype=np.int64)
    gg = gt.groupby("image_id").indices; pg = p.groupby("image_id").indices
    GB, GC = gt[BOX].to_numpy(float), gt["class_id"].to_numpy(np.int64)
    PB, PC, PS = p[BOX].to_numpy(float), p["class_id"].to_numpy(np.int64), p["confidence"].to_numpy(float)
    for img in set(gg) | set(pg):
        gi = gg.get(img, np.array([], dtype=np.int64)); pi = pg.get(img, np.array([], dtype=np.int64))
        used = np.zeros(len(gi), dtype=bool)
        if len(gi) and len(pi):
            pi = pi[np.argsort(-PS[pi], kind="stable")]
            iou = box_iou(PB[pi], GB[gi])
            # prefer a same-class partner, otherwise the best-overlapping box of any class
            for r in range(len(pi)):
                row = np.where(used, -1.0, iou[r])
                same = np.where(GC[gi] == PC[pi[r]], row, -1.0)
                j = int(same.argmax()) if same.max() >= iou_thr else int(row.argmax())
                if row[j] >= iou_thr:
                    used[j] = True; M[GC[gi[j]], PC[pi[r]]] += 1
                else:
                    M[n_classes, PC[pi[r]]] += 1
        else:
            for r in pi:
                M[n_classes, PC[r]] += 1
        for j in np.where(~used)[0]:
            M[GC[gi[j]], n_classes] += 1
    return M

# %% [markdown]
# ## 4. Dataset verification
# Locates SVRDD under `DATA_ROOT` (or downloads it), validates every label line, and builds a
# normalised copy under `WORK/data` (`images/<split>` symlinks + cleaned `labels/<split>`), which is what
# all detectors train and evaluate on.

# %% tags=datalib
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_ALIASES = {"train": {"train", "training"}, "val": {"val", "valid", "validation"}, "test": {"test", "testing"}}

def list_images(root):
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT]

def split_of(path, root):
    parts = [s.lower() for s in Path(path).relative_to(root).parts[:-1]]
    for split, names in SPLIT_ALIASES.items():
        if any(s in names for s in parts):
            return split
    return None

def list_entry_stem(line):
    """File stem of one line of a split list, whatever its style: bare name, relative or absolute path,
    forward or backward slashes, with or without extension, optionally followed by other columns."""
    import re
    s = line.replace("\\", "/").split(",")[0].split("\t")[0].strip().strip("\"'")
    m = re.search(r"([^/]+?)\.(?:jpe?g|png|bmp|txt|xml)(?=$|[\s\"'])", s, flags=re.IGNORECASE)
    return m.group(1) if m else s.rsplit("/", 1)[-1].strip("\"'")

def split_lists(root):
    """For datasets that define splits with train.txt / val.txt / test.txt files (SVRDD does)."""
    out, files = {}, []
    for split, names in SPLIT_ALIASES.items():
        for n in names:
            for f in Path(root).rglob(f"{n}.txt"):
                files.append(f)
                for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.strip():
                        out[list_entry_stem(line)] = split
    return out, files

def find_class_names(root):
    for f in list(Path(root).rglob("*.yaml")) + list(Path(root).rglob("*.yml")):
        try:
            import yaml
            names = yaml.safe_load(f.read_text()).get("names")
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names)]
            if names:
                return [str(n) for n in names], str(f)
        except Exception:
            pass
    for pat in ("classes.txt", "*.names", "labels.txt"):
        for f in Path(root).rglob(pat):
            names = [l.strip() for l in f.read_text().splitlines() if l.strip()]
            if 1 < len(names) < 50:
                return names, str(f)
    return None, None

def locate_dataset():
    """A folder counts as the dataset if it holds >= 100 images and >= 10 label files. Only folders named
    like the dataset are considered, so other attached datasets (e.g. an old RDD2022 mirror) are never mixed in."""
    def ok(root):
        return Path(root).exists() and len(list(itertools.islice(Path(root).rglob("*.txt"), 10))) >= 10 and \
               len(list(itertools.islice((p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT), 100))) >= 100
    raw = TEMP.parent / "svrdd_raw" if SMOKE else TEMP / "svrdd_raw"
    if DATA_ROOT.exists():
        hits = [DATA_ROOT] if "svrdd" in DATA_ROOT.name.lower() else []
        hits += sorted((p for p in DATA_ROOT.rglob("*") if p.is_dir() and "svrdd" in p.name.lower()), key=lambda p: len(p.parts))
        if str(DATA_ROOT) != "/kaggle/input":
            hits.append(DATA_ROOT)                          # an explicitly chosen folder is trusted
        for h in hits:
            if ok(h):
                return h
    if ok(raw):
        return raw
    if not DOWNLOAD_IF_MISSING:
        raise FileNotFoundError(f"No SVRDD folder under {DATA_ROOT}. Attach it, set DATA_ROOT, or set DOWNLOAD_IF_MISSING = True.")
    raw.mkdir(parents=True, exist_ok=True); z = raw / "SVRDD_YOLO.zip"
    print("downloading SVRDD_YOLO.zip from Zenodo (2.9 GB; notebook internet must be ON) ...")
    subprocess.run(["wget", "-q", "-c", "-O", str(z), ZENODO_URL], check=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(raw)
    z.unlink()
    return raw

def parse_label_file(path, n_classes_hint=None):
    """Returns (rows [k,5] = class, cx, cy, w, h in [0,1]), issue counter). Polygons are reduced to their bounding box."""
    rows, issues = [], defaultdict(int)
    if path is None or not Path(path).exists():
        issues["missing_file"] += 1
        return np.zeros((0, 5)), issues
    for line in Path(path).read_text().splitlines():
        v = line.split()
        if not v:
            continue
        try:
            vals = [float(x) for x in v]
        except ValueError:
            issues["non_numeric"] += 1; continue
        if len(vals) < 5:
            issues["too_few_values"] += 1; continue
        c = vals[0]
        if len(vals) > 5:                                   # polygon -> bbox
            xs, ys = vals[1::2], vals[2::2]; n = min(len(xs), len(ys))
            x1, x2, y1, y2 = min(xs[:n]), max(xs[:n]), min(ys[:n]), max(ys[:n])
            issues["polygon_converted"] += 1
        else:
            x1, y1, x2, y2 = vals[1] - vals[3] / 2, vals[2] - vals[4] / 2, vals[1] + vals[3] / 2, vals[2] + vals[4] / 2
        if c != int(c) or c < 0 or (n_classes_hint and c >= n_classes_hint):
            issues["bad_class_id"] += 1; continue
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 2:
            issues["not_normalised"] += 1; continue
        cx1, cy1, cx2, cy2 = (min(max(t, 0.0), 1.0) for t in (x1, y1, x2, y2))
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            issues["clipped_to_image"] += 1
        if cx2 - cx1 <= 1e-4 or cy2 - cy1 <= 1e-4:
            issues["degenerate_box"] += 1; continue
        rows.append([int(c), (cx1 + cx2) / 2, (cy1 + cy2) / 2, cx2 - cx1, cy2 - cy1])
    rows = np.array(rows).reshape(-1, 5)
    if len(rows) and len(np.unique(rows, axis=0)) < len(rows):
        issues["duplicate_lines"] += len(rows) - len(np.unique(rows, axis=0))
        rows = np.unique(rows, axis=0)
    return rows, issues

# %%
SRC_ROOT = locate_dataset()
CLASS_NAMES, class_src = find_class_names(SRC_ROOT)
if CLASS_NAMES is None:
    CLASS_NAMES = list(DEFAULT_CLASSES)
    print("WARNING: no class file found in the dataset. Using DEFAULT_CLASSES; check the sanity table below.")
else:
    print("class names from", class_src)
NC = len(CLASS_NAMES)
print("classes:", dict(enumerate(CLASS_NAMES)))

imgs = list_images(SRC_ROOT)
label_index = {}
for t in Path(SRC_ROOT).rglob("*.txt"):
    label_index.setdefault(t.stem, []).append(t)
lists, list_files = split_lists(SRC_ROOT)
for f in list_files:                                     # split lists are not label files
    label_index.pop(f.stem, None)
records, issue_total = [], defaultdict(int)
for p in imgs:
    split = split_of(p, SRC_ROOT) or lists.get(p.stem)
    if split is None:
        issue_total["image_without_split"] += 1; continue
    cands = label_index.get(p.stem, [])
    same = [c for c in cands if split_of(c, SRC_ROOT) == split]
    records.append(dict(image_id=p.stem, split=split, path=str(p), label=str((same or cands or [None])[0]) if (same or cands) else None))
images = pd.DataFrame(records)
def _layout_report():
    head = "\n".join(f"{f}: " + repr(f.read_text(encoding='utf-8', errors='ignore').splitlines()[:3]) for f in list_files)
    return (f"split lists found: {[str(f) for f in list_files]} ({len(lists)} entries)\n{head}\n"
            f"example image stems: {[p.stem for p in imgs[:3]]}\nfolders:\n" +
            "\n".join(str(p) for p in itertools.islice((q for q in Path(SRC_ROOT).rglob('*') if q.is_dir()), 30)))
assert len(images), f"No image could be assigned to a split under {SRC_ROOT}.\n" + _layout_report()
if issue_total["image_without_split"]:
    print(f"note: {issue_total['image_without_split']} images are in no split list and are ignored")
dup_ids = images.image_id[images.image_id.duplicated()].unique()
assert len(dup_ids) == 0, f"{len(dup_ids)} image stems occur more than once, e.g. {dup_ids[:5]}"
assert set(images.split) >= {"train", "val", "test"}, f"splits found: {set(images.split)}"

if SMOKE:
    images = pd.concat([images[images.split == s].sample(n=min(n, (images.split == s).sum()), random_state=SEED)
                        for s, n in (("train", 200), ("val", 60), ("test", 60))], ignore_index=True)

def _size(p):
    with Image.open(p) as im:
        return im.size
with ThreadPoolExecutor(8) as ex:
    sizes = list(ex.map(_size, images.path))
images["W"], images["H"] = [s[0] for s in sizes], [s[1] for s in sizes]

gt_rows = []
for split in ("train", "val", "test"):
    (DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
new_paths = []
for r in images.itertuples():
    rows, issues = parse_label_file(r.label, NC)
    for k, v in issues.items():
        issue_total[k] += v
    dst = DATA_DIR / "images" / r.split / Path(r.path).name
    if not dst.exists():
        try:
            os.symlink(r.path, dst)
        except OSError:
            shutil.copy2(r.path, dst)
    new_paths.append(str(dst))
    with open(DATA_DIR / "labels" / r.split / f"{r.image_id}.txt", "w") as f:
        for c, cx, cy, w, h in rows:
            f.write(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, r.W, r.H)
            gt_rows.append((r.image_id, r.split, int(c), x1, y1, x2, y2, r.W, r.H))
images["src_path"] = images["path"]; images["path"] = new_paths
GT_ALL = pd.DataFrame(gt_rows, columns=["image_id", "split", "class_id"] + BOX + ["W", "H"])
GT = {s: GT_ALL[GT_ALL.split == s].drop(columns="split").reset_index(drop=True) for s in ("train", "val", "test")}
IMAGES = {s: images[images.split == s].reset_index(drop=True) for s in ("train", "val", "test")}

DATA_YAML = DATA_DIR / "data.yaml"
DATA_YAML.write_text("path: {}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{}\n".format(
    DATA_DIR.as_posix(), "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))))

print("\nimages per split:", images.split.value_counts().to_dict())
print("label issues:", {k: v for k, v in issue_total.items() if v} or "none")
assert GT_ALL.class_id.between(0, NC - 1).all()
assert len(GT["train"]) > 0 and len(GT["val"]) > 0, "no boxes parsed; check label format"
if len(GT["test"]) == 0:
    print("WARNING: test split has no labelled boxes -> section 18 will be skipped.")

# Class-order sanity check: longitudinal classes should be taller than wide, transverse ones wider than tall.
_g = GT_ALL.assign(w=GT_ALL.x2 - GT_ALL.x1, h=GT_ALL.y2 - GT_ALL.y1)
_g["wh"] = _g.w / _g.h
sanity = _g.groupby("class_id").agg(n=("w", "size"), median_w=("w", "median"), median_h=("h", "median"),
                                    median_w_over_h=("wh", "median"))
sanity.insert(0, "name", [CLASS_NAMES[i] for i in sanity.index])
print("\nclass-order sanity: transverse classes should be much wider than tall, longitudinal ones not, "
      "alligator cracks large, potholes and manhole covers small.")
print(sanity.round(2).to_string())

# Visual confirmation of the class order: six training examples per class id, cropped with some context.
_n_ex = 6
fig, axes = plt.subplots(NC, _n_ex, figsize=(2.1 * _n_ex, 2.1 * NC), squeeze=False)
_paths = dict(zip(images.image_id, images.path))
for c in range(NC):
    d = GT["train"][GT["train"].class_id == c]
    d = d.sample(min(_n_ex, len(d)), random_state=SEED)
    for k in range(_n_ex):
        ax = axes[c, k]; ax.axis("off")
        if k < len(d):
            r = d.iloc[k]; im = Image.open(_paths[r.image_id]).convert("RGB")
            pad = 0.35 * max(r.x2 - r.x1, r.y2 - r.y1) + 12
            box = (max(r.x1 - pad, 0), max(r.y1 - pad, 0), min(r.x2 + pad, r.W), min(r.y2 + pad, r.H))
            ax.imshow(im.crop(tuple(int(v) for v in box)))
            ax.add_patch(plt.Rectangle((r.x1 - box[0], r.y1 - box[1]), r.x2 - r.x1, r.y2 - r.y1, fill=False, edgecolor="#eb6834", linewidth=1.5))
        if k == 0:
            ax.set_title(f"id {c}: {CLASS_NAMES[c]}", fontsize=9, loc="left")
plt.tight_layout(); plt.savefig(WORK / "figs" / "class_examples.png", dpi=120, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 5. Dataset analysis
# The outputs of this section decide what gets prioritised: class imbalance -> per-class thresholds,
# small-object share -> multi-scale / tiling, leakage -> how much to trust validation numbers.

# %%
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 10,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 11, "axes.titlecolor": INK,
                     "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False})
def class_color(i):
    return SERIES[i % len(SERIES)]
def savefig(name):
    plt.savefig(WORK / "figs" / f"{name}.png"); plt.show()

g = GT_ALL.assign(w=GT_ALL.x2 - GT_ALL.x1, h=GT_ALL.y2 - GT_ALL.y1)
g["area"] = g.w * g.h; g["rel_area"] = g.area / (g.W * g.H); g["aspect"] = g.w / g.h
g["size"] = size_bucket(g); g["name"] = g.class_id.map(dict(enumerate(CLASS_NAMES)))

t_split = images.groupby("split").agg(images=("image_id", "count")).join(
    g.groupby("split").agg(boxes=("class_id", "count"), images_with_boxes=("image_id", "nunique")))
t_split["empty_images"] = t_split.images - t_split.images_with_boxes
t_split["boxes_per_image"] = (t_split.boxes / t_split.images).round(2)
t_class = g.pivot_table(index="name", columns="split", values="area", aggfunc="count", fill_value=0).reindex(CLASS_NAMES)
t_class["share_train"] = (t_class["train"] / t_class["train"].sum()).round(3)
t_class["imbalance_vs_max"] = (t_class["train"].max() / t_class["train"].clip(lower=1)).round(1)
t_box = g.groupby("name")[["w", "h", "area", "rel_area", "aspect"]].median().reindex(CLASS_NAMES).round(4)
t_size = (g.pivot_table(index="name", columns="size", values="area", aggfunc="count", fill_value=0)
          .reindex(CLASS_NAMES).reindex(columns=["small", "medium", "large"], fill_value=0))
t_size_share = t_size.div(t_size.sum(1), axis=0).round(3)
t_res = images.groupby(["split", "W", "H"]).size().rename("images").reset_index()
for name, t in (("split_summary", t_split), ("class_counts", t_class), ("box_medians", t_box),
                ("size_share", t_size_share), ("resolutions", t_res)):
    t.to_csv(WORK / "tables" / f"analysis_{name}.csv"); print(f"\n== {name} =="); print(t.to_string())
print("\noverall size share (COCO definition, pixels of the original image):",
      g["size"].value_counts(normalize=True).round(3).to_dict())

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].barh(CLASS_NAMES[::-1], t_class["train"].values[::-1], color=SERIES[0], height=0.6)
ax[0].set_title("Training boxes per class"); ax[0].grid(axis="y", visible=False)
for i, v in enumerate(t_class["train"].values[::-1]):
    ax[0].text(v, i, f" {int(v):,}", va="center", color="#52514e", fontsize=9)
left = np.zeros(NC)
for k, (sz, col) in enumerate(zip(["small", "medium", "large"], ["#9ec5f4", "#3987e5", "#104281"])):
    vals = t_size_share[sz].values[::-1]
    ax[1].barh(CLASS_NAMES[::-1], vals, left=left, color=col, height=0.6, label=sz, edgecolor="white", linewidth=1.5)
    left += vals
ax[1].set_title("Object size mix per class (small < 32², large > 96² px)"); ax[1].set_yticklabels([])
ax[1].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.25)); ax[1].grid(axis="y", visible=False)
for c in range(NC):
    d = g[g.class_id == c]
    d = d.sample(min(len(d), 1500), random_state=0)
    ax[2].scatter(d.w, d.h, s=4, alpha=0.35, color=class_color(c), label=CLASS_NAMES[c], linewidths=0)
ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("box width (px)"); ax[2].set_ylabel("box height (px)")
ax[2].set_title("Box shape by class"); ax[2].legend(markerscale=3, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
savefig("analysis_overview")

# %%
# Duplicates and cross-split leakage. Exact duplicates: MD5 of the file bytes.
# Near-duplicates: 64-bit difference hash, Hamming distance <= 4 (candidates; inspect the saved figure).
def _md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()
def _dhash(p):
    with Image.open(p) as im:
        a = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)

hash_path = WORK / "tables" / "hashes.parquet"
if hash_path.exists() and len(pd.read_parquet(hash_path)) == len(images):
    H = pd.read_parquet(hash_path)
else:
    with ThreadPoolExecutor(8) as ex:
        md5 = list(ex.map(_md5, images.path)); dh = list(ex.map(_dhash, images.path))
    H = images[["image_id", "split", "path"]].assign(md5=md5, dhash=[str(x) for x in dh]); H.to_parquet(hash_path)

exact = H[H.duplicated("md5", keep=False)].sort_values("md5")
exact_cross = exact.groupby("md5").filter(lambda d: d.split.nunique() > 1)
print(f"exact duplicate files: {exact.md5.nunique()} groups ({len(exact)} images); "
      f"groups spanning more than one split: {exact_cross.md5.nunique()}")

_pop = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
def hamming_matrix(a, b):
    x = np.bitwise_xor(a[:, None], b[None, :])
    return _pop[x.view(np.uint8).reshape(len(a), len(b), 8)].sum(-1)
hv = {s: np.array([int(x) for x in H[H.split == s].dhash], dtype=np.uint64) for s in ("train", "val", "test")}
hid = {s: H[H.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
leak_rows = []
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    for start in range(0, len(hv[b]), 256):
        D = hamming_matrix(hv[b][start:start + 256], hv[a])
        for i, j in zip(*np.where(D <= 4)):
            leak_rows.append(dict(pair=f"{a}-{b}", id_a=hid[a].image_id[j], id_b=hid[b].image_id[start + i],
                                  path_a=hid[a].path[j], path_b=hid[b].path[start + i], hamming=int(D[i, j])))
LEAK = pd.DataFrame(leak_rows, columns=["pair", "id_a", "id_b", "path_a", "path_b", "hamming"])
LEAK.to_csv(WORK / "tables" / "analysis_near_duplicates.csv", index=False)
for pair in ("train-val", "train-test", "val-test"):
    d = LEAK[LEAK.pair == pair]; b = pair.split("-")[1]
    print(f"{pair}: {d.id_b.nunique()} of {len(hid[b])} {b} images have a near-duplicate candidate "
          f"({d.id_b.nunique() / max(len(hid[b]), 1):.1%})")
if len(LEAK):
    show = LEAK.sort_values("hamming").head(4)
    fig, ax = plt.subplots(2, len(show), figsize=(3.2 * len(show), 6.4), squeeze=False)
    for k, r in enumerate(show.itertuples()):
        for row, (p, lab) in enumerate(((r.path_a, r.pair.split("-")[0]), (r.path_b, r.pair.split("-")[1]))):
            ax[row, k].imshow(Image.open(p)); ax[row, k].set_title(f"{lab} (d={r.hamming})", fontsize=9); ax[row, k].axis("off")
    savefig("analysis_near_duplicate_examples")
save_state(n_train=len(IMAGES["train"]), n_val=len(IMAGES["val"]), n_test=len(IMAGES["test"]),
           small_share=float((g["size"] == "small").mean()),
           leak_train_val=float(LEAK[LEAK.pair == "train-val"].id_b.nunique() / max(len(hid["val"]), 1)))

# %% [markdown]
# ## 6. Baseline training (Model A: YOLO11m)
# Resumes automatically from `last.pt`. A `DONE` marker prevents retraining a finished run.

# %%
from ultralytics import YOLO, RTDETR

def train_ultralytics(name, cfg, cls=None):
    """Train (or continue) an Ultralytics model. Returns best.pt, or None if the session budget ran out.

    Ultralytics strips the optimiser from last.pt when a run stops, which makes it non-resumable, so a
    callback keeps a full copy (resume.pt) after every epoch and stops the run once the budget is used."""
    global PAUSED
    cls = cls or YOLO
    run = WORK / "runs" / name
    resume, best, done = run / "weights" / "resume.pt", run / "weights" / "best.pt", run / "DONE"
    if done.exists() and best.exists():
        print(f"{name}: already trained -> {best}"); return best
    paused = {"flag": False}
    def on_model_save(trainer):
        shutil.copy2(trainer.last, Path(trainer.wdir) / "resume.pt")
        # never pause on the last epoch: the run is finishing anyway and a finished run cannot be resumed
        if trainer.epoch + 1 < trainer.epochs and out_of_time(name):
            paused["flag"] = True; trainer.stop = True
    device = 0 if torch.cuda.is_available() else "cpu"
    if resume.exists():
        print(f"{name}: resuming from {resume}")
        model = cls(str(resume)); model.add_callback("on_model_save", on_model_save)
        model.train(resume=True)
    else:
        args = {k: v for k, v in cfg.items() if k != "model"}
        batch = args.pop("batch")
        while True:
            try:
                model = cls(cfg["model"]); model.add_callback("on_model_save", on_model_save)
                model.train(data=str(DATA_YAML), project=str(WORK / "runs"), name=name, exist_ok=True,
                            device=device, batch=batch, **args)
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch > 2:
                    batch //= 2; torch.cuda.empty_cache(); print(f"OOM -> retrying with batch={batch}")
                else:
                    raise
    if paused["flag"]:
        PAUSED = True
        print(f"{name}: session budget of {SESSION_BUDGET_H} h used. Progress is saved in {resume}.")
        return None
    assert best.exists(), f"{name}: training produced no best.pt"
    done.write_text(f"finished {time.strftime('%Y-%m-%d %H:%M')}")
    return best

A_WEIGHTS = train_ultralytics("A_yolo11m", CFG_A)
_r = pd.read_csv(WORK / "runs" / "A_yolo11m" / "results.csv"); _r.columns = [c.strip() for c in _r.columns]
print(f"epochs run so far: {len(_r)} | minutes per epoch ~",
      round(float(_r['time'].diff().median() / 60), 1) if "time" in _r and len(_r) > 1 else "n/a")

# %% [markdown]
# ## 7. Baseline evaluation
# Predictions are extracted once into the canonical table and scored with the notebook evaluator.
# Ultralytics' own `val()` numbers are recorded next to them because published SVRDD baselines use that convention.

# %%
def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))

class UltraPredictor:
    """YOLO / RT-DETR through Ultralytics. Takes RGB arrays, returns boxes in the pixel frame of each array."""
    def __init__(self, weights, nms_iou=0.6):
        self.model = (RTDETR if "rtdetr" in str(weights).lower() else YOLO)(str(weights)); self.nms_iou = nms_iou
    def __call__(self, arrays, imgsz):
        res = self.model.predict([a[:, :, ::-1] for a in arrays], imgsz=imgsz, conf=LOW_CONF, iou=self.nms_iou,
                                 max_det=MAX_DET, verbose=False,
                                 device=0 if torch.cuda.is_available() else "cpu")
        return [(r.boxes.xyxy.float().cpu().numpy(), r.boxes.conf.float().cpu().numpy(),
                 r.boxes.cls.cpu().numpy().astype(np.int64)) for r in res]

def run_view(predictor, img_table, imgsz, flip=False, tile=None, batch=INFER_BATCH):
    """One inference 'view' over a split. flip: horizontal-flip TTA (boxes are un-flipped).
    tile: tile side in pixels -> each tile is inferred at `imgsz`; output gets tile_id and trunc columns."""
    out = []
    buf, meta = [], []                              # meta: (image_id, W, H, x_off, y_off, tile_idx, tile_box)
    def flush():
        if not buf:
            return
        for (iid, W, H, xo, yo, tid, tb), (b, s, c) in zip(meta, predictor(buf, imgsz)):
            b = b.astype(np.float64)
            if flip:
                w = (tb[2] - tb[0]) if tb else W
                b[:, [0, 2]] = w - b[:, [2, 0]]
            b[:, [0, 2]] += xo; b[:, [1, 3]] += yo
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, W); b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
            d = make_preds([iid] * len(s), c, s, b)
            if tile:
                d["tile_id"] = tid; d["trunc"] = truncated_flags(b, tb, W, H)
            out.append(d)
        buf.clear(); meta.clear()
    for r in img_table.itertuples():
        img = load_rgb(r.path)
        H_, W_ = img.shape[:2]
        crops = [(None, img)] if not tile else [(tb, img[tb[1]:tb[3], tb[0]:tb[2]]) for tb in make_tiles(W_, H_, tile)]
        for tid, (tb, crop) in enumerate(crops):
            buf.append(np.ascontiguousarray(crop[:, ::-1] if flip else crop))
            meta.append((r.image_id, W_, H_, tb[0] if tb else 0, tb[1] if tb else 0, tid, tb))
            if len(buf) >= batch:
                flush()
    flush()
    d = pd.concat(out, ignore_index=True) if out else empty_preds()
    if tile and "tile_id" not in d:
        d["tile_id"] = pd.Series(dtype="int64"); d["trunc"] = pd.Series(dtype="bool")
    keep = ((d.x2 - d.x1) > 1) & ((d.y2 - d.y1) > 1)
    return d[keep].reset_index(drop=True)

_PREDICTORS = {}
def get_predictor(model_key, nms_iou=0.6):
    k = (model_key, nms_iou)
    if k not in _PREDICTORS:
        _PREDICTORS.clear(); torch.cuda.empty_cache()          # one model on the GPU at a time
        if model_key == "A":
            _PREDICTORS[k] = UltraPredictor(A_WEIGHTS, nms_iou)
        elif model_key == "C":
            _PREDICTORS[k] = UltraPredictor(C_WEIGHTS, nms_iou)
        else:
            _PREDICTORS[k] = FRCNNPredictor(B_WEIGHTS, nms_iou)
    return _PREDICTORS[k]

def view_name(v):
    return f"s{v['imgsz']}" + ("_flip" if v.get("flip") else "") + (f"_tile{v['tile']}" if v.get("tile") else "") + \
           f"_nms{int(round(v.get('nms_iou', 0.6) * 100))}"

def view_preds(model_key, split, view):
    """Cached canonical predictions of one model for one view on one split."""
    f = WORK / "preds" / f"{model_key}_{split}_{view_name(view)}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    t0 = time.time()
    d = run_view(get_predictor(model_key, view.get("nms_iou", 0.6)), IMAGES[split], view["imgsz"],
                 flip=view.get("flip", False), tile=view.get("tile"))
    d.to_parquet(f); print(f"  {f.name}: {len(d):,} boxes in {time.time() - t0:.0f}s")
    return d

def recipe_preds(model_key, split, views, merge_iou=0.55):
    """Fuse the views of one model. Plain views -> WBF (score = max); a tile view is merged boundary-aware."""
    plain = [v for v in views if not v.get("tile")]; tiles = [v for v in views if v.get("tile")]
    dfs = [view_preds(model_key, split, v) for v in plain]
    d = dfs[0] if len(dfs) == 1 else fuse_predictions(dfs, method="wbf", iou_thr=merge_iou, conf_type="max")
    for v in tiles:
        d = merge_full_and_tiles(d, view_preds(model_key, split, v), iou_thr=merge_iou)
    return d

BASE_VIEW = dict(imgsz=IMGSZ, flip=False, nms_iou=0.6)
A_val = recipe_preds("A", "val", [BASE_VIEW])
res = evaluate(A_val, GT["val"], NC, conf_thr=0.25, class_names=CLASS_NAMES)
log_row("A0 YOLO11m baseline (conf 0.25)", "val", res)
print(pd.Series(res["summary"]).round(4).to_string()); print(res["per_class"].round(3).to_string(index=False))

um = YOLO(str(A_WEIGHTS)).val(data=str(DATA_YAML), split="val", imgsz=IMGSZ, batch=INFER_BATCH, verbose=False, plots=False)
ultra = dict(P=float(um.box.mp), R=float(um.box.mr), mAP50=float(um.box.map50), mAP50_95=float(um.box.map))
ultra["F1"] = 2 * ultra["P"] * ultra["R"] / max(ultra["P"] + ultra["R"], 1e-9)
pd.Series(ultra).to_csv(WORK / "tables" / "A_ultralytics_val.csv")
print("\nUltralytics val() convention (class-mean P/R at the best-F1 confidence), comparable with published baselines:")
print(pd.Series(ultra).round(4).to_string())
print(f"mAP50 cross-check: notebook {res['summary']['mAP50']:.4f} vs Ultralytics {ultra['mAP50']:.4f}")

hitA0 = gt_hits(res, len(GT["val"])); size_rec = recall_by_size(GT["val"], hitA0)
print("\nrecall by object size at conf 0.25:"); print(size_rec.round(3).to_string(index=False))
_s = size_rec.set_index("size").recall
small_gap = float(np.nanmax([_s.get("medium", np.nan), _s.get("large", np.nan)]) - _s.get("small", np.nan)) \
    if not np.isnan(_s.get("small", np.nan)) else 0.0
save_state(A_small_gap=small_gap)

# %% [markdown]
# ## 8. Threshold optimisation (Model A)
# NMS IoU -> single confidence threshold -> per-class thresholds. Per-class thresholds are adopted only if
# they win under cross-fitting (fit on half of val, score on the other half), so extra parameters are not rewarded.

# %%
def tune_thresholds(pred, gt, label):
    """Returns (conf_thr, mode, table). Chooses per-class thresholds only if the cross-fitted gain clears the gate."""
    m = match_predictions(pred, gt)
    g_thr, g_f1, _ = best_global_threshold(pred, gt, m)
    c_thr = best_class_thresholds(pred, gt, NC, m, fallback=g_thr)
    c_f1 = micro_f1(pred, gt, NC, c_thr, m)
    cv_g = np.mean([crossfit_f1(pred, gt, NC, "global", seed=s) for s in range(5)])
    cv_c = np.mean([crossfit_f1(pred, gt, NC, "class", seed=s) for s in range(5)])
    use_class = (cv_c - cv_g) >= GATE["post"]
    tab = pd.DataFrame([dict(setting="conf 0.25", F1=micro_f1(pred, gt, NC, 0.25, m), crossfit_F1=np.nan),
                        dict(setting=f"global conf {g_thr:.2f}", F1=g_f1, crossfit_F1=cv_g),
                        dict(setting="per-class conf", F1=c_f1, crossfit_F1=cv_c)])
    print(f"[{label}] per-class thresholds {'KEPT' if use_class else 'rejected'} "
          f"(cross-fitted gain {cv_c - cv_g:+.4f}, gate {GATE['post']})"); print(tab.round(4).to_string(index=False))
    return (c_thr.tolist() if use_class else g_thr), ("class" if use_class else "global"), tab, m

# NMS IoU of the detector itself (each candidate is a separate cached inference pass)
nms_scores = {}
for iou in (0.5, 0.6, 0.7):
    d = recipe_preds("A", "val", [dict(BASE_VIEW, nms_iou=iou)])
    nms_scores[iou] = best_global_threshold(d, GT["val"])[1]
best_nms = max(nms_scores, key=nms_scores.get)
if nms_scores[best_nms] - nms_scores[0.6] < GATE["post"]:
    best_nms = 0.6
print("best-threshold val F1 by NMS IoU:", {k: round(v, 4) for k, v in nms_scores.items()}, "-> using", best_nms)
BASE_VIEW = dict(BASE_VIEW, nms_iou=best_nms)
A_val = recipe_preds("A", "val", [BASE_VIEW])

A_thr, A_thr_mode, tabA, mA = tune_thresholds(A_val, GT["val"], "A")
g_thr = best_global_threshold(A_val, GT["val"], mA)[0]
log_row("A1 + tuned global confidence", "val", evaluate(A_val, GT["val"], NC, g_thr, CLASS_NAMES, mA), note=f"conf={g_thr:.2f}")
log_row("A2 + per-class confidence", "val", evaluate(A_val, GT["val"], NC, A_thr, CLASS_NAMES, mA) if A_thr_mode == "class" else None)
save_state(A_views=[BASE_VIEW], A_thr=A_thr, A_thr_mode=A_thr_mode)

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(CONF_GRID, best_global_threshold(A_val, GT["val"], mA)[2], color=SERIES[0], lw=2)
ax.axvline(g_thr, color=MUTED, lw=1, ls="--"); ax.text(g_thr, ax.get_ylim()[0], f" best {g_thr:.2f}", color="#52514e", va="bottom")
ax.set_xlabel("confidence threshold"); ax.set_ylabel("micro F1 (val)"); ax.set_title("Model A: F1 against confidence threshold")
savefig("A_f1_vs_confidence")

# %% [markdown]
# ## 9. Test-time augmentation, multi-scale inference and tiling
# Greedy forward selection: a view is added only if the best-threshold val F1 rises by the gate.
# Tiling runs only if small objects are recalled clearly worse than larger ones (or `FORCE_TILING`).

# %%
def select_views(model_key, base_views, candidates, gate_of):
    views = list(base_views)
    best = best_global_threshold(recipe_preds(model_key, "val", views), GT["val"])[1]
    log = [dict(model=model_key, views=" + ".join(view_name(v) for v in views), F1=best, decision="base")]
    for cand in candidates:
        f = best_global_threshold(recipe_preds(model_key, "val", views + [cand]), GT["val"])[1]
        keep = (f - best) >= gate_of(cand)
        log.append(dict(model=model_key, views="+ " + view_name(cand), F1=f, decision=f"{'KEEP' if keep else 'drop'} ({f - best:+.4f})"))
        if keep:
            views.append(cand); best = f
    return views, pd.DataFrame(log)

cands = [dict(BASE_VIEW, flip=True)]
if not SMOKE:
    cands.append(dict(BASE_VIEW, imgsz=int(IMGSZ * 1.25 // 32 * 32)))
run_tiling = FORCE_TILING or STATE.get("A_small_gap", 0) >= GATE["small_gap"]
print(f"small-object recall gap = {STATE.get('A_small_gap', 0):.3f} (gate {GATE['small_gap']}) -> tiling experiment "
      f"{'RUN' if run_tiling else 'skipped'}")
if run_tiling:
    cands.append(dict(BASE_VIEW, tile=int(IMGSZ * 0.625 // 32 * 32)))
A_VIEWS, view_log = select_views("A", [BASE_VIEW], cands, lambda v: GATE["tiling"] if v.get("tile") else GATE["post"])
print(view_log.round(4).to_string(index=False)); view_log.to_csv(WORK / "tables" / "A_view_selection.csv", index=False)

A_val = recipe_preds("A", "val", A_VIEWS)
A_thr, A_thr_mode, tabA, mA = tune_thresholds(A_val, GT["val"], "A final recipe")
resA = evaluate(A_val, GT["val"], NC, A_thr, CLASS_NAMES, mA)
plain = [v for v in A_VIEWS if not v.get("tile")]
d = recipe_preds("A", "val", plain)
log_row("A3 + TTA / multi-scale", "val",
        evaluate(d, GT["val"], NC, best_global_threshold(d, GT["val"])[0], CLASS_NAMES) if len(plain) > 1 else None,
        note=" + ".join(view_name(v) for v in plain))
log_row("A4 + tiling", "val", resA if any(v.get("tile") for v in A_VIEWS) else None, note="boundary-aware tile merge")
log_row("A* best single-model recipe", "val", resA, note=f"{len(A_VIEWS)} view(s), {A_thr_mode} threshold")
save_state(A_views=A_VIEWS, A_thr=A_thr, A_thr_mode=A_thr_mode, A_val_F1=resA["summary"]["F1"])
print(pd.Series(resA["summary"]).round(4).to_string())

# %% [markdown]
# ## 10. Detector B: Faster R-CNN ResNet50-FPN (dataset + training)
# Same images and cleaned labels as Model A. Class ids are shifted by +1 for training (0 = background)
# and shifted back at prediction time. Checkpoints every epoch; resumes automatically.

# %%
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class DetDataset(torch.utils.data.Dataset):
    def __init__(self, img_table, gt, train):
        self.t = img_table.reset_index(drop=True); self.train = train
        self.boxes = {k: (d[BOX].to_numpy(np.float32), d["class_id"].to_numpy(np.int64)) for k, d in gt.groupby("image_id")}
    def __len__(self):
        return len(self.t)
    def __getitem__(self, i):
        r = self.t.iloc[i]
        img = torch.from_numpy(load_rgb(r.path).copy()).permute(2, 0, 1).float().div(255)
        b, c = self.boxes.get(r.image_id, (np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)))
        b = torch.from_numpy(b.copy()).reshape(-1, 4); c = torch.from_numpy(c.copy()) + 1      # 0 is background
        if self.train and random.random() < 0.5:
            img = img.flip(-1); W = img.shape[-1]
            b = torch.stack([W - b[:, 2], b[:, 1], W - b[:, 0], b[:, 3]], 1) if len(b) else b
        return img, dict(boxes=b, labels=c)

def collate(batch):
    return tuple(zip(*batch))

def build_frcnn(pretrained=True, min_size=(1024,), max_size=1024, nms_iou=0.5):
    m = fasterrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None, weights_backbone=None,
                                min_size=min_size, max_size=max_size, box_score_thresh=LOW_CONF,
                                box_detections_per_img=MAX_DET, box_nms_thresh=nms_iou)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, NC + 1)
    return m

class FRCNNPredictor:
    def __init__(self, weights, nms_iou=0.5):
        self.model = build_frcnn(pretrained=False, nms_iou=nms_iou)
        sd = torch.load(weights, map_location="cpu"); self.model.load_state_dict(sd.get("model", sd))
        self.model.to(DEVICE).eval()
    @torch.no_grad()
    def __call__(self, arrays, imgsz):
        self.model.transform.min_size = (imgsz,); self.model.transform.max_size = imgsz
        x = [torch.from_numpy(a.copy()).permute(2, 0, 1).float().div(255).to(DEVICE) for a in arrays]
        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            out = self.model(x)
        return [(o["boxes"].float().cpu().numpy(), o["scores"].float().cpu().numpy(),
                 o["labels"].cpu().numpy().astype(np.int64) - 1) for o in out]

def frcnn_val_map(model):
    model.eval()
    class _P:                                              # reuse run_view with the in-training model
        def __call__(_, arrays, imgsz):
            model.transform.min_size = (imgsz,); model.transform.max_size = imgsz
            with torch.no_grad(), torch.autocast("cuda", enabled=torch.cuda.is_available()):
                out = model([torch.from_numpy(a.copy()).permute(2, 0, 1).float().div(255).to(DEVICE) for a in arrays])
            return [(o["boxes"].float().cpu().numpy(), o["scores"].float().cpu().numpy(), o["labels"].cpu().numpy() - 1) for o in out]
    d = run_view(_P(), IMAGES["val"], CFG_B["max_size"])
    return evaluate(d, GT["val"], NC, 0.25, CLASS_NAMES)["summary"]

def train_frcnn(name="B_frcnn_r50"):
    global PAUSED
    run = WORK / "runs" / name; run.mkdir(parents=True, exist_ok=True)
    last, best, done, logf = run / "last.pt", run / "best.pt", run / "DONE", run / "log.csv"
    if done.exists() and best.exists():
        print(f"{name}: already trained -> {best}"); return best
    cfg = CFG_B
    model = build_frcnn(cfg["pretrained"], cfg["train_sizes"], cfg["max_size"]).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(cfg["milestones"]), gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    start, best_map, step = 0, -1.0, 0
    if last.exists():
        ck = torch.load(last, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); start, best_map, step = ck["epoch"] + 1, ck["best_map"], ck["step"]
        print(f"{name}: resuming at epoch {start + 1}/{cfg['epochs']} (best mAP50 {best_map:.4f})")
    g = torch.Generator(); g.manual_seed(SEED + start)
    dl = torch.utils.data.DataLoader(DetDataset(IMAGES["train"], GT["train"], True), batch_size=cfg["batch"], shuffle=True,
                                     num_workers=cfg["workers"], collate_fn=collate, drop_last=True,
                                     generator=g, persistent_workers=cfg["workers"] > 0)
    if budget_used():
        PAUSED = True
        print(f"{name}: session budget already used; not starting. Progress so far is in {run}."); return None
    for epoch in range(start, cfg["epochs"]):
        model.train(); model.transform.min_size = tuple(cfg["train_sizes"]); model.transform.max_size = cfg["max_size"]
        t0, run_loss, n = time.time(), 0.0, 0
        for imgs_, tgts in dl:
            if step < cfg["warmup_iters"]:
                for gparam in opt.param_groups:
                    gparam["lr"] = cfg["lr"] * (0.001 + 0.999 * step / cfg["warmup_iters"])
            elif step == cfg["warmup_iters"]:
                for gparam, lr in zip(opt.param_groups, sched.get_last_lr()):
                    gparam["lr"] = lr
            imgs_ = [i.to(DEVICE, non_blocking=True) for i in imgs_]
            tgts = [{k: v.to(DEVICE) for k, v in t.items()} for t in tgts]
            with torch.autocast("cuda", enabled=torch.cuda.is_available()):
                loss = sum(model(imgs_, tgts).values())
            if not torch.isfinite(loss):
                print("non-finite loss, batch skipped"); opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt); scaler.update()
            run_loss += loss.item(); n += 1; step += 1
            if step % 200 == 0:
                print(f"  ep {epoch + 1} it {n}/{len(dl)} loss {run_loss / n:.4f} lr {opt.param_groups[0]['lr']:.5f}")
        sched.step()
        s = frcnn_val_map(model)
        row = dict(epoch=epoch + 1, loss=run_loss / max(n, 1), val_mAP50=s["mAP50"], val_mAP50_95=s["mAP50_95"],
                   val_F1_at_025=s["F1"], minutes=(time.time() - t0) / 60)
        pd.DataFrame([row]).to_csv(logf, mode="a", header=not logf.exists(), index=False)
        print({k: round(v, 4) for k, v in row.items()})
        if s["mAP50"] > best_map:
            best_map = float(s["mAP50"]); torch.save({"model": model.state_dict(), "epoch": epoch, "mAP50": best_map}, best)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
                        epoch=epoch, best_map=best_map, step=step), last)
        if epoch + 1 < cfg["epochs"] and out_of_time(name):
            PAUSED = True
            print(f"{name}: session budget used after epoch {epoch + 1}. Progress is saved in {last}."); return None
    done.write_text("finished"); return best

B_WEIGHTS = train_frcnn()

# %% [markdown]
# ## 11. Detector B evaluation

# %%
B_BASE = dict(imgsz=CFG_B["max_size"], flip=False, nms_iou=0.5)
B_VIEWS, view_log_B = select_views("B", [B_BASE], [dict(B_BASE, flip=True)], lambda v: GATE["post"])
print(view_log_B.round(4).to_string(index=False))
B_val = recipe_preds("B", "val", B_VIEWS)
log_row("B0 Faster R-CNN (conf 0.25)", "val", evaluate(recipe_preds("B", "val", [B_BASE]), GT["val"], NC, 0.25, CLASS_NAMES))
B_thr, B_thr_mode, tabB, mB = tune_thresholds(B_val, GT["val"], "B")
resB = evaluate(B_val, GT["val"], NC, B_thr, CLASS_NAMES, mB)
log_row("B* Faster R-CNN best recipe", "val", resB, note=f"{len(B_VIEWS)} view(s), {B_thr_mode} threshold")
save_state(B_views=B_VIEWS, B_thr=B_thr, B_thr_mode=B_thr_mode, B_val_F1=resB["summary"]["F1"])
print(pd.Series(resB["summary"]).round(4).to_string()); print(resB["per_class"].round(3).to_string(index=False))

# %% [markdown]
# ## 12. Prediction extraction
# One canonical table per model and split: `image_id, class_id, confidence, x1, y1, x2, y2`.

# %%
MODEL_VIEWS = {"A": A_VIEWS, "B": B_VIEWS}
def model_preds(model_key, split):
    return recipe_preds(model_key, split, MODEL_VIEWS[model_key])
VAL_PREDS = {k: model_preds(k, "val") for k in MODEL_VIEWS}
for k, d in VAL_PREDS.items():
    d.to_csv(WORK / "preds" / f"canonical_{k}_val.csv", index=False)
    print(k, f"{len(d):,} boxes |", d.groupby("image_id").size().describe()[["mean", "max"]].round(1).to_dict())
print(VAL_PREDS["A"].head().to_string())

# %% [markdown]
# ## 13. Class harmonisation
# Every detector must use the canonical ids. Faster R-CNN's background offset is removed inside its predictor;
# this cell proves it: each model's predictions must agree with the ground truth on the diagonal of the
# class-agnostic confusion matrix, and with each other.

# %%
def check_harmonised(name, pred, thr):
    assert list(pred.columns[:7]) == PRED_COLS and pred.class_id.between(0, NC - 1).all(), f"{name}: schema or class ids wrong"
    assert (pred.x2 > pred.x1).all() and (pred.y2 > pred.y1).all(), f"{name}: degenerate boxes"
    M = detection_confusion(pred, GT["val"], NC, thr)[:NC, :NC]
    agree = np.trace(M) / max(M.sum(), 1)
    print(f"{name}: of {M.sum()} localised objects, {agree:.1%} carry the ground-truth class")
    # only meaningful once the model localises a reasonable number of objects
    assert agree > 0.6 or M.sum() < 50, f"{name}: class ids look permuted relative to the labels (agreement {agree:.1%})"
check_harmonised("A", VAL_PREDS["A"], A_thr); check_harmonised("B", VAL_PREDS["B"], B_thr)
_a = VAL_PREDS["A"][VAL_PREDS["A"].confidence >= 0.3]; _b = VAL_PREDS["B"][VAL_PREDS["B"].confidence >= 0.3]
_m = detection_confusion(_b, _a.drop(columns="confidence"), NC, 0.0)[:NC, :NC]
print(f"A vs B: {np.trace(_m) / max(_m.sum(), 1):.1%} of co-located boxes have the same class id")

# %% [markdown]
# ## 14. Fusion: WBF versus NMS

# %%
def fused_score(dfs, weights, method, iou_thr, conf_type="avg"):
    d = fuse_predictions(dfs, weights, method, iou_thr, conf_type, skip_thr=FUSE_SKIP)
    thr, f1, _ = best_global_threshold(d, GT["val"])
    return f1, thr

rows = []
for method, conf_type in (("wbf", "avg"), ("wbf", "max"), ("nms", "-")):
    for iou in (0.5, 0.55, 0.6, 0.7):
        f1, thr = fused_score([VAL_PREDS["A"], VAL_PREDS["B"]], [1, 1], method, iou, conf_type)
        rows.append(dict(method=method, conf_type=conf_type, iou=iou, F1=f1, conf=thr))
fusion_grid = pd.DataFrame(rows).sort_values("F1", ascending=False)
fusion_grid.to_csv(WORK / "tables" / "fusion_method_grid.csv", index=False)
print(fusion_grid.round(4).to_string(index=False))
FUSE = fusion_grid.iloc[0][["method", "conf_type", "iou"]].to_dict()

# %% [markdown]
# ## 15. Ensemble weight optimisation

# %%
rows = []
for wb in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
    f1, thr = fused_score([VAL_PREDS["A"], VAL_PREDS["B"]], [1.0, wb], FUSE["method"], FUSE["iou"], FUSE["conf_type"])
    rows.append(dict(w_A=1.0, w_B=wb, F1=f1, conf=thr))
weight_grid = pd.DataFrame(rows); weight_grid.to_csv(WORK / "tables" / "fusion_weight_grid.csv", index=False)
print(weight_grid.round(4).to_string(index=False))
FUSE["weights"] = [1.0, float(weight_grid.sort_values("F1", ascending=False).iloc[0].w_B)]
FUSE["skip"] = FUSE_SKIP

def fuse_models(preds, keys, fuse=None):
    """The final fusion. Same operation as fused_score(), so what was tuned is what gets evaluated."""
    fuse = fuse or FUSE
    w = [fuse["weights"][["A", "B", "C"].index(k)] for k in keys]
    return fuse_predictions([preds[k] for k in keys], w, fuse["method"], fuse["iou"], fuse["conf_type"], skip_thr=FUSE_SKIP)

def assert_matches_tuning(fused, tuned_f1, label):
    got = best_global_threshold(fused, GT["val"])[1]
    assert abs(got - tuned_f1) <= 0.002, (f"{label}: the final fusion scores F1 {got:.4f} but tuning found {tuned_f1:.4f}; "
                                          "the fusion being evaluated is not the one that was tuned")
    print(f"{label}: final fusion reproduces the tuned F1 ({got:.4f})")

AB_val = fuse_models(VAL_PREDS, ["A", "B"])
assert_matches_tuning(AB_val, float(weight_grid.F1.max()), "A+B")
AB_thr, AB_thr_mode, tabAB, mAB = tune_thresholds(AB_val, GT["val"], "A+B")
resAB = evaluate(AB_val, GT["val"], NC, AB_thr, CLASS_NAMES, mAB)
log_row("A+B fusion", "val", resAB, note=f"{FUSE['method']} iou={FUSE['iou']} w={FUSE['weights']} skip={FUSE_SKIP}")
print(pd.Series(resAB["summary"]).round(4).to_string())

# %% [markdown]
# ## 16. Error complementarity and the keep/discard decision for B
# B stays only if it finds objects A misses **and** the fused detector beats A alone by the gate.

# %%
gtv = GT["val"]
hitA, hitB = gt_hits(resA, len(gtv)), gt_hits(resB, len(gtv))
comp = complementarity(gtv, hitA, hitB, CLASS_NAMES); comp.to_csv(WORK / "tables" / "complementarity.csv", index=False)
print(comp.round(3).to_string(index=False))
fp = shared_false_positives(VAL_PREDS["A"], resA, VAL_PREDS["B"], resB)
pd.Series(fp).to_csv(WORK / "tables" / "shared_false_positives.csv"); print("\nfalse positives:", {k: round(v, 3) for k, v in fp.items()})
per_class_cmp = resA["per_class"][["name", "n_gt", "P", "R", "F1"]].merge(
    resB["per_class"][["name", "P", "R", "F1"]], on="name", suffixes=("_A", "_B")).merge(
    resAB["per_class"][["name", "P", "R", "F1"]].rename(columns=dict(P="P_AB", R="R_AB", F1="F1_AB")), on="name")
per_class_cmp.to_csv(WORK / "tables" / "per_class_A_B_AB.csv", index=False); print("\n", per_class_cmp.round(3).to_string(index=False))

allrow = comp.iloc[0]
gain = resAB["summary"]["F1"] - resA["summary"]["F1"]
KEEP_B = bool(allrow.B_only_share >= GATE["b_only_tp"] and gain >= GATE["ensemble"])
print(f"\nB-only true positives: {allrow.B_only_share:.1%} of GT (gate {GATE['b_only_tp']:.0%}); "
      f"fusion gain over A: {gain:+.4f} F1 (gate {GATE['ensemble']}) -> Model B is {'KEPT' if KEEP_B else 'DISCARDED'}")
headroom = allrow.oracle_recall - resAB["summary"]["R"]
C_GATE_OPEN = bool(KEEP_B and resAB["summary"]["F1"] < TARGET_F1 and headroom > GATE["c_headroom"])
print(f"third-model gate: fused F1 {resAB['summary']['F1']:.4f}, oracle-recall headroom {headroom:+.3f} -> "
      f"{'OPEN (set RUN_C = True to train RT-DETR-L)' if C_GATE_OPEN else 'closed: do not train a third model'}")
save_state(KEEP_B=KEEP_B, C_GATE_OPEN=C_GATE_OPEN, FUSE=FUSE, AB_thr=AB_thr, AB_thr_mode=AB_thr_mode)

# %%
# Optional Model C (RT-DETR-L). Runs only if the gate opened AND RUN_C was set by hand.
KEEP_C = False
if RUN_C and C_GATE_OPEN:
    C_WEIGHTS = train_ultralytics("C_rtdetr_l", CFG_C, RTDETR)
    MODEL_VIEWS["C"] = [dict(imgsz=CFG_C["imgsz"], flip=False, nms_iou=0.6)]
    VAL_PREDS["C"] = model_preds("C", "val")
    check_harmonised("C", VAL_PREDS["C"], 0.25)
    resC = evaluate(VAL_PREDS["C"], GT["val"], NC, best_global_threshold(VAL_PREDS["C"], GT["val"])[0], CLASS_NAMES)
    log_row("C* RT-DETR-L", "val", resC)
    rows = []
    for wc in (0.5, 1.0, 1.5):
        f1, thr = fused_score([VAL_PREDS[k] for k in "ABC"], FUSE["weights"][:2] + [wc], FUSE["method"], FUSE["iou"], FUSE["conf_type"])
        rows.append(dict(w_C=wc, F1=f1))
    wc = float(pd.DataFrame(rows).sort_values("F1", ascending=False).iloc[0].w_C)
    FUSE_C = dict(FUSE, weights=FUSE["weights"][:2] + [wc])
    ABC_val = fuse_models(VAL_PREDS, ["A", "B", "C"], FUSE_C)
    assert_matches_tuning(ABC_val, float(pd.DataFrame(rows).F1.max()), "A+B+C")
    ABC_thr, ABC_mode, _, mABC = tune_thresholds(ABC_val, GT["val"], "A+B+C")
    resABC = evaluate(ABC_val, GT["val"], NC, ABC_thr, CLASS_NAMES, mABC)
    log_row("A+B+C fusion", "val", resABC, note=f"w={FUSE_C['weights']}")
    KEEP_C = bool(resABC["summary"]["F1"] - resAB["summary"]["F1"] >= GATE["ensemble"])
    print(f"A+B+C gain over A+B: {resABC['summary']['F1'] - resAB['summary']['F1']:+.4f} -> C {'KEPT' if KEEP_C else 'DISCARDED'}")
    if KEEP_C:
        FUSE = FUSE_C; save_state(FUSE=FUSE, ABC_thr=ABC_thr)
else:
    log_row("C* RT-DETR-L", "val", None); log_row("A+B+C fusion", "val", None)
save_state(KEEP_C=KEEP_C)

# %% [markdown]
# ## 17. Final validation evaluation
# The final detector is frozen here: member models, views, fusion settings and thresholds.

# %%
FINAL_KEYS = ["A"] + (["B"] if KEEP_B else []) + (["C"] if KEEP_C else [])
def final_predictions(split):
    preds = {k: model_preds(k, split) for k in FINAL_KEYS}
    return preds, (preds["A"] if FINAL_KEYS == ["A"] else fuse_models(preds, FINAL_KEYS))
if FINAL_KEYS == ["A"]:
    FINAL_val, FINAL_thr, resF = A_val, A_thr, resA
elif FINAL_KEYS == ["A", "B"]:
    FINAL_val, FINAL_thr, resF = AB_val, AB_thr, resAB
else:
    FINAL_val, FINAL_thr, resF = ABC_val, ABC_thr, resABC
log_row("FINAL detector", "val", resF, note="+".join(FINAL_KEYS))
save_state(FINAL_KEYS=FINAL_KEYS, FINAL_thr=FINAL_thr, FINAL_val_F1=resF["summary"]["F1"])
print("final detector:", "+".join(FINAL_KEYS), "| thresholds:", np.round(conf_vector(FINAL_thr, NC), 2).tolist())
print(pd.Series(resF["summary"]).round(4).to_string()); print(resF["per_class"].round(3).to_string(index=False))
print("\nrecall by size:"); print(recall_by_size(GT["val"], gt_hits(resF, len(GT["val"]))).round(3).to_string(index=False))
print(f"\nTarget F1 >= {TARGET_F1}: {'reached' if resF['summary']['F1'] >= TARGET_F1 else 'NOT reached'} on validation "
      f"({resF['summary']['F1']:.4f}). Validation numbers are tuned-on; the test result below is the one to report.")

# %% [markdown]
# ## 18. Final test evaluation
# Nothing is tuned here. Every row uses settings fixed on validation.

# %%
RES_TEST = {}
if EVAL_TEST and len(GT["test"]):
    gtt = GT["test"]
    base = recipe_preds("A", "test", [dict(imgsz=IMGSZ, flip=False, nms_iou=0.6)])
    RES_TEST["A0 YOLO11m baseline (conf 0.25)"] = evaluate(base, gtt, NC, 0.25, CLASS_NAMES)
    test_preds, FINAL_test = final_predictions("test")
    RES_TEST["A* best single-model recipe"] = evaluate(test_preds["A"], gtt, NC, A_thr, CLASS_NAMES)
    # B is reported on test even if it was discarded, so the paper's comparison table is complete
    RES_TEST["B* Faster R-CNN best recipe"] = evaluate(model_preds("B", "test"), gtt, NC, B_thr, CLASS_NAMES)
    RES_TEST["FINAL detector"] = evaluate(FINAL_test, gtt, NC, FINAL_thr, CLASS_NAMES)
    for k, r in RES_TEST.items():
        log_row(k, "test", r)
    FINAL_test.to_csv(WORK / "preds" / "canonical_FINAL_test.csv", index=False)
    rf = RES_TEST["FINAL detector"]
    print(pd.Series(rf["summary"]).round(4).to_string()); print(rf["per_class"].round(3).to_string(index=False))
    save_state(FINAL_test_F1=rf["summary"]["F1"])
    print(f"\nTEST F1 = {rf['summary']['F1']:.4f} (target {TARGET_F1}: {'reached' if rf['summary']['F1'] >= TARGET_F1 else 'not reached'})")
else:
    print("test evaluation skipped (EVAL_TEST is False or the test split has no labels)")

# %% [markdown]
# ## 19. Result tables

# %%
STAGE_ORDER = ["A0 YOLO11m baseline (conf 0.25)", "A1 + tuned global confidence", "A2 + per-class confidence",
               "A3 + TTA / multi-scale", "A4 + tiling", "A* best single-model recipe", "B0 Faster R-CNN (conf 0.25)",
               "B* Faster R-CNN best recipe", "A+B fusion", "C* RT-DETR-L", "A+B+C fusion", "FINAL detector"]
abl = pd.read_csv(ABL_PATH)
abl = abl.assign(_o=abl.stage.map({s: i for i, s in enumerate(STAGE_ORDER)})).sort_values(["split", "_o"]).drop(columns="_o").reset_index(drop=True)
cols =["stage", "P", "R", "F1", "macro_F1", "mAP50", "mAP50_95", "note"]
for split in ("val", "test"):
    t = abl[abl.split == split][cols]
    if len(t):
        t.to_csv(WORK / "tables" / f"ablation_{split}.csv", index=False)
        print(f"\n=== Ablation ({split}) ==="); print(t.to_string(index=False))
        (WORK / "tables" / f"ablation_{split}.tex").write_text(t.drop(columns="note").to_latex(index=False, float_format="%.3f"))
final_pc = (RES_TEST.get("FINAL detector") or resF)["per_class"][["name", "n_gt", "TP", "FP", "FN", "P", "R", "F1", "AP50", "AP50_95", "conf_thr"]]
final_pc.to_csv(WORK / "tables" / "final_per_class.csv", index=False)
print("\n=== Final detector, per class ({}) ===".format("test" if RES_TEST else "val")); print(final_pc.round(3).to_string(index=False))
print("\n=== Decisions ==="); print(json.dumps({k: STATE[k] for k in STATE if k.startswith(("KEEP", "FINAL", "FUSE", "A_views", "B_views", "C_GATE"))}, indent=1, default=str))

# %% [markdown]
# ## 20. Plots for the paper
# Saved to `WORK/figs` at 200 dpi. Figures use the final evaluation split (test if available, otherwise val).

# %%
split_f = "test" if RES_TEST else "val"
gt_f = GT[split_f]
pred_f = FINAL_test if RES_TEST else FINAL_val
res_f = RES_TEST["FINAL detector"] if RES_TEST else resF
short = [n.replace("_", " ") for n in CLASS_NAMES]

# (a) precision-recall curves, final detector
m_f = match_predictions(pred_f, gt_f)
fig, ax = plt.subplots(figsize=(6.4, 5))
for c in range(NC):
    r, p, _ = pr_curve(pred_f, gt_f, c, m_f)
    ax.plot(r, p, color=class_color(c), lw=2, label=f"{short[c]} (AP50 {res_f['per_class'].AP50[c]:.2f})")
ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_title(f"Precision-recall by class, final detector ({split_f}), mAP50 {res_f['summary']['mAP50']:.3f}")
ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5)); savefig("pr_curves_final")

# (b) confusion matrix with background, row-normalised
M = detection_confusion(pred_f, gt_f, NC, FINAL_thr)
Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
labels = short + ["background"]
fig, ax = plt.subplots(figsize=(7.2, 6.2))
im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
ax.set_xticks(range(NC + 1)); ax.set_xticklabels(labels, rotation=40, ha="right"); ax.set_yticks(range(NC + 1)); ax.set_yticklabels(labels)
for i in range(NC + 1):
    for j in range(NC + 1):
        if M[i, j]:
            ax.text(j, i, f"{Mn[i, j]:.2f}\n({M[i, j]})", ha="center", va="center", fontsize=7, color="white" if Mn[i, j] > 0.55 else INK)
ax.set_xlabel("predicted"); ax.set_ylabel("ground truth")
ax.set_title(f"Detection confusion matrix, final detector ({split_f})\nlast column = missed objects, last row = false alarms")
fig.colorbar(im, fraction=0.046, pad=0.04, label="share of row"); savefig("confusion_matrix_final")

# (c) per-class F1: A vs B vs final (validation, where all three were scored under the same protocol)
fig, ax = plt.subplots(figsize=(9, 4)); x = np.arange(NC); wbar = 0.26
series = [("Model A (YOLO11m)", resA["per_class"].F1.values), ("Model B (Faster R-CNN)", resB["per_class"].F1.values),
          ("Final detector", resF["per_class"].F1.values)]
for k, (lab, vals) in enumerate(series):
    ax.bar(x + (k - 1) * wbar, vals, width=wbar - 0.03, color=SERIES[k], label=lab)
ax.set_xticks(x); ax.set_xticklabels(short, rotation=20, ha="right"); ax.set_ylabel("F1 (val)"); ax.set_ylim(0, 1)
ax.grid(axis="x", visible=False); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16)); savefig("per_class_f1_comparison")

# (d) ablation: F1 and mAP50 per stage (two panels, one measure each)
for split in ("val", "test"):
    t = abl[abl.split == split]
    if not len(t):
        continue
    fig, ax = plt.subplots(1, 2, figsize=(12, 0.45 * len(t) + 1.6), sharey=True)
    for a, col, title in ((ax[0], "F1", "F1 (IoU >= 0.5, class match)"), (ax[1], "mAP50", "mAP50")):
        a.barh(t.stage[::-1], t[col][::-1], color=SERIES[0], height=0.6); a.set_title(f"{title}, {split}")
        a.grid(axis="y", visible=False); a.set_xlim(0, 1)
        for i, v in enumerate(t[col][::-1]):
            a.text(v, i, f" {v:.3f}", va="center", color="#52514e", fontsize=9)
    ax[0].axvline(TARGET_F1, color=MUTED, lw=1, ls="--")
    ax[0].text(TARGET_F1, 0.01, " target", color="#52514e", fontsize=8, va="bottom", transform=ax[0].get_xaxis_transform())
    savefig(f"ablation_{split}")

# (e) recall by object size: A vs B vs final (validation)
fig, ax = plt.subplots(figsize=(6, 3.8)); x = np.arange(3)
for k, (lab, r) in enumerate((("Model A", resA), ("Model B", resB), ("Final detector", resF))):
    rs = recall_by_size(GT["val"], gt_hits(r, len(GT["val"])))
    ax.bar(x + (k - 1) * wbar, rs.recall.fillna(0).values, width=wbar - 0.03, color=SERIES[k], label=lab)
ax.set_xticks(x); ax.set_xticklabels([f"{n}\n(n={c})" for n, c in zip(rs["size"], rs.n_gt)]); ax.set_ylabel("recall (val)"); ax.set_ylim(0, 1)
ax.grid(axis="x", visible=False); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18)); savefig("recall_by_size")

# (f) who finds what: complementarity of A and B per class
c = comp[comp.group.isin(CLASS_NAMES)].set_index("group").reindex(CLASS_NAMES)
share = c[["both", "A_only", "B_only", "neither"]].div(c.n_gt.clip(lower=1), axis=0)
fig, ax = plt.subplots(figsize=(8.5, 3.8)); left = np.zeros(NC)
for col, colr, lab in (("both", "#104281", "found by both"), ("A_only", SERIES[0], "only A"), ("B_only", SERIES[1], "only B"), ("neither", "#e1e0d9", "missed by both")):
    ax.barh(short[::-1], share[col].values[::-1], left=left, color=colr, label=lab, height=0.6, edgecolor="white", linewidth=1.5)
    left += share[col].values[::-1]
ax.set_xlim(0, 1); ax.set_xlabel("share of ground-truth objects (val)"); ax.grid(axis="y", visible=False)
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2)); savefig("complementarity_by_class")

# (g) training curves of Model A
rc = pd.read_csv(WORK / "runs" / "A_yolo11m" / "results.csv"); rc.columns = [x.strip() for x in rc.columns]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for k, (col, lab) in enumerate((("train/box_loss", "train box loss"), ("val/box_loss", "val box loss"))):
    if col in rc: ax[0].plot(rc.epoch, rc[col], color=SERIES[k], lw=2, label=lab)
ax[0].set_xlabel("epoch"); ax[0].set_title("Model A: box loss"); ax[0].legend()
for k, (col, lab) in enumerate((("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95"))):
    if col in rc: ax[1].plot(rc.epoch, rc[col], color=SERIES[k], lw=2, label=lab)
ax[1].set_xlabel("epoch"); ax[1].set_title("Model A: validation mAP"); ax[1].legend(); savefig("A_training_curves")

print("figures:", sorted(p.name for p in (WORK / "figs").glob("*.png")))
print("tables :", sorted(p.name for p in (WORK / "tables").glob("*.csv")))
