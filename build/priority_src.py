# %% [markdown]
# # SVRDD maintenance priority: index, trained predictor and road-segment ranking
#
# Stage 2. Runs on CPU in a few minutes. It needs the output of the detection notebook (`svrdd_run`)
# attached as an input, and the SVRDD dataset (attached or downloaded) for the ground-truth boxes.
#
# The dataset has no maintenance-priority labels, so the design is:
# 1. **Priority index.** A transparent score per image from damage type, count and extent. Applied to the
#    ground-truth boxes it defines the *reference* priority (Low / Medium / High).
# 2. **Trained predictor.** A model that sees only the detector's output and predicts the reference priority.
#    It is trained on the validation images and tested on the test images. Its job is real: detections are
#    noisy, and it must beat the simple baseline of applying the index directly to the detections.
# 3. **Road segments.** Images are grouped by the map coordinates in their file names and segments are ranked.
#
# What this does and does not show: it measures how well priority derived from *automatic* detection agrees
# with priority derived from *human-annotated* damage under the same index. It does not validate the index
# against engineering inspections, because no such data exists for this dataset.

# %% [markdown]
# ## 1. Imports and shared library

# %% include=lib

# %%
import re, subprocess
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance

# %% [markdown]
# ## 2. Configuration

# %%
DATA_ROOT = Path(os.environ.get("SVRDD_ROOT", "/kaggle/input"))
STAGE1 = os.environ.get("SVRDD_STAGE1")          # None = find the detection notebook's svrdd_run folder under /kaggle/input
WORK = Path(os.environ.get("PRIORITY_WORK", "/kaggle/working/priority_run"))
TEMP = Path(os.environ.get("SVRDD_TEMP", "/kaggle/temp" if Path("/kaggle").exists() else str(WORK.parent / "svrdd_tmp")))
DOWNLOAD_IF_MISSING = True
ZENODO_URL = "https://zenodo.org/api/records/10100129/files/SVRDD_YOLO.zip/content"
SMOKE = False                                    # read by the shared dataset code; keep False here
SEED = 42
DEFAULT_CLASSES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole",
                   "manhole_cover", "longitudinal_patch", "transverse_patch"]

# --- Priority index (design choices; section 8 measures how much the result depends on them) ---
# Relative weight of one instance of each class. The ORDER follows pavement-condition practice (potholes and
# alligator cracking are the most severe distresses, linear cracks intermediate, patches are repaired damage);
# the NUMBERS are our own and are not taken from any standard. A manhole cover is not damage.
CLASS_WEIGHT = {"pothole": 10, "alligator_crack": 8, "transverse_crack": 5, "longitudinal_crack": 4,
                "longitudinal_patch": 2, "transverse_patch": 2, "manhole_cover": 0}
LINEAR_CLASSES = {"longitudinal_crack", "transverse_crack", "longitudinal_patch", "transverse_patch"}   # extent = length
BASE = 0.1                         # contribution of an instance regardless of its size (presence term)
PRIORITY_QUANTILES = (0.50, 0.80)  # cut points on the TRAIN reference scores: Low < q50 <= Medium < q80 <= High
LEVELS = ["Low", "Medium", "High"]

# --- Road segments ---
SEGMENT_M = 200                    # side of a grid cell in metres
MIN_IMAGES_PER_SEGMENT = 3         # segments with fewer evaluated images are not ranked

for d in ("tables", "figs"):
    (WORK / d).mkdir(parents=True, exist_ok=True)
rng = np.random.RandomState(SEED)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 10,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 11, "axes.titlecolor": INK,
                     "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False})
def savefig(name):
    plt.savefig(WORK / "figs" / f"{name}.png"); plt.show()
def save_table(df, name, index=False):
    df.to_csv(WORK / "tables" / f"{name}.csv", index=index)

# %% [markdown]
# ### 2b. Priority library
# Pure functions, so they can be unit-tested without the dataset.

# %% tags=prilib
def box_contributions(df, wh, weight_vec, linear_mask, base):
    """Contribution of every box to its image's priority score: weight[class] * (base + extent).

    extent is scale-free and in (0, 1]: for linear damage the box diagonal over the image diagonal (a length),
    for areal damage the square root of the box area over the image area. wh maps image_id -> (W, H)."""
    if len(df) == 0:
        return np.zeros(0)
    W = df["image_id"].map(lambda i: wh[i][0]).to_numpy(float); H = df["image_id"].map(lambda i: wh[i][1]).to_numpy(float)
    w = (df["x2"] - df["x1"]).to_numpy(float).clip(0); h = (df["y2"] - df["y1"]).to_numpy(float).clip(0)
    cls = df["class_id"].to_numpy(np.int64)
    length = np.hypot(w, h) / np.hypot(W, H)
    areal = np.sqrt(w * h / (W * H))
    extent = np.where(np.asarray(linear_mask)[cls], length, areal)
    return np.asarray(weight_vec, dtype=float)[cls] * (base + extent)

def image_scores(df, image_ids, wh, weight_vec, linear_mask, base):
    """Priority score per image (sum of box contributions); images without boxes score 0."""
    s = pd.Series(box_contributions(df, wh, weight_vec, linear_mask, base), index=df["image_id"].to_numpy())
    return s.groupby(level=0).sum().reindex(list(image_ids), fill_value=0.0).to_numpy(float)

def to_levels(scores, cuts):
    """0 = Low, 1 = Medium, 2 = High. A score equal to a cut point belongs to the higher level."""
    return np.digitize(np.asarray(scores, dtype=float), np.asarray(cuts, dtype=float), right=False)

def level_metrics(y_true, y_pred):
    return dict(accuracy=accuracy_score(y_true, y_pred),
                macro_F1=f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0),
                QWK=cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=[0, 1, 2]),
                high_recall=float(np.mean(np.asarray(y_pred)[np.asarray(y_true) == 2] == 2)) if (np.asarray(y_true) == 2).any() else np.nan)

def score_metrics(s_true, s_pred):
    rho = spearmanr(s_true, s_pred).statistic if len(s_true) > 2 and np.std(s_pred) > 0 and np.std(s_true) > 0 else np.nan
    return dict(spearman=float(rho), MAE=float(np.mean(np.abs(np.asarray(s_true) - np.asarray(s_pred)))))

_XY = re.compile(r"_(\d{8,11})_(\d{8,11})$")
def parse_xy(image_id):
    """Map coordinates in metres from an SVRDD file name '<panorama id>_<x*100>_<y*100>', or (nan, nan)."""
    m = _XY.search(str(image_id))
    return (int(m.group(1)) / 100.0, int(m.group(2)) / 100.0) if m else (np.nan, np.nan)

def segment_ids(x, y, cell):
    """Grid-cell id per point; '' where the coordinates are missing."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    gx = np.floor(np.where(ok, x, 0) / cell).astype(np.int64); gy = np.floor(np.where(ok, y, 0) / cell).astype(np.int64)
    return np.where(ok, np.char.add(np.char.add(gx.astype(str), "_"), gy.astype(str)), "")

def top_k_overlap(a, b, k):
    """Share of the k highest-ranked items under a that are also among the k highest under b."""
    k = int(max(1, min(k, len(a))))
    ia = set(np.argsort(-np.asarray(a), kind="stable")[:k]); ib = set(np.argsort(-np.asarray(b), kind="stable")[:k])
    return len(ia & ib) / k

# %% [markdown]
# ## 3. Ground truth
# Same dataset discovery and label cleaning as the detection notebook.

# %% include=datalib

# %%
SRC_ROOT = locate_dataset()
CLASS_NAMES = find_class_names(SRC_ROOT)[0] or list(DEFAULT_CLASSES)
NC = len(CLASS_NAMES)
assert set(CLASS_NAMES) == set(CLASS_WEIGHT), f"CLASS_WEIGHT must name exactly the dataset classes: {CLASS_NAMES}"
W_VEC = np.array([CLASS_WEIGHT[n] for n in CLASS_NAMES], dtype=float)
LIN = np.array([n in LINEAR_CLASSES for n in CLASS_NAMES])

label_index = {}
for t in Path(SRC_ROOT).rglob("*.txt"):
    label_index.setdefault(t.stem, []).append(t)
lists, list_files = split_lists(SRC_ROOT)
for f in list_files:
    label_index.pop(f.stem, None)
records = []
for p in list_images(SRC_ROOT):
    split = split_of(p, SRC_ROOT) or lists.get(p.stem)
    if split is None:
        continue
    cands = label_index.get(p.stem, [])
    same = [c for c in cands if split_of(c, SRC_ROOT) == split]
    rel = Path(p).relative_to(SRC_ROOT).parts
    records.append(dict(image_id=p.stem, split=split, path=str(p), label=(same or cands or [None])[0],
                        district=next((s for s in rel[:-1] if s.lower() not in ("images", "labels", "svrdd_yolo")), "")))
images = pd.DataFrame(records)
assert len(images) and set(images.split) >= {"train", "val", "test"}, f"dataset not understood under {SRC_ROOT}"
def _size(p):
    with Image.open(p) as im:
        return im.size
with ThreadPoolExecutor(8) as ex:
    sizes = list(ex.map(_size, images.path))
images["W"], images["H"] = [s[0] for s in sizes], [s[1] for s in sizes]
WH = {r.image_id: (r.W, r.H) for r in images.itertuples()}

gt_rows = []
for r in images.itertuples():
    rows, _ = parse_label_file(r.label, NC)
    for c, cx, cy, w, h in rows:
        gt_rows.append((r.image_id, r.split, int(c)) + tuple(yolo_to_xyxy(cx, cy, w, h, r.W, r.H)))
GT_ALL = pd.DataFrame(gt_rows, columns=["image_id", "split", "class_id"] + BOX)
GT = {s: GT_ALL[GT_ALL.split == s].drop(columns="split").reset_index(drop=True) for s in ("train", "val", "test")}
IDS = {s: images[images.split == s].image_id.tolist() for s in ("train", "val", "test")}
print("images per split:", {s: len(IDS[s]) for s in IDS}, "| boxes:", {s: len(GT[s]) for s in GT})

# %% [markdown]
# ## 4. Detector outputs from stage 1
# The final predictions are rebuilt from the cached per-model predictions with the frozen stage-1 settings,
# and checked against the test F1 that stage 1 reported.

# %%
if STAGE1 is None:
    found = sorted((p.parent for p in Path("/kaggle/input").rglob("state.json") if (p.parent / "preds").is_dir()),
                   key=lambda p: (p / "state.json").stat().st_mtime) if Path("/kaggle/input").exists() else []
    assert found, "Attach the detection notebook's output (Add Input -> Your Work -> Notebooks) or set STAGE1."
    STAGE1 = found[-1]
STAGE1 = Path(STAGE1)
S1 = json.loads((STAGE1 / "state.json").read_text())
FINAL_KEYS = S1["FINAL_KEYS"]
print("stage 1:", STAGE1, "| final detector:", "+".join(FINAL_KEYS), "| reported test F1:", S1.get("FINAL_test_F1"))

def view_name(v):
    return f"s{v['imgsz']}" + ("_flip" if v.get("flip") else "") + (f"_tile{v['tile']}" if v.get("tile") else "") + \
           f"_nms{int(round(v.get('nms_iou', 0.6) * 100))}"

def model_preds(key, split):
    views = S1[f"{key}_views"]
    assert not any(v.get("tile") for v in views), "tiled views are not supported here (stage 1 rejected tiling)"
    dfs = [pd.read_parquet(STAGE1 / "preds" / f"{key}_{split}_{view_name(v)}.parquet")[PRED_COLS] for v in views]
    return dfs[0] if len(dfs) == 1 else fuse_predictions(dfs, method="wbf", iou_thr=0.55, conf_type="max")

def final_preds(split):
    preds = {k: model_preds(k, split) for k in FINAL_KEYS}
    if FINAL_KEYS == ["A"]:
        return preds["A"]
    F = S1["FUSE"]
    w = [F["weights"][["A", "B", "C"].index(k)] for k in FINAL_KEYS]
    return fuse_predictions([preds[k] for k in FINAL_KEYS], w, F["method"], F["iou"], F["conf_type"], skip_thr=F.get("skip", 0.05))

THR = {"A": S1["A_thr"], "B": S1["B_thr"], "FINAL": S1["FINAL_thr"]}
PRED = {m: {s: (final_preds(s) if m == "FINAL" else model_preds(m, s)) for s in ("val", "test")} for m in ("A", "B", "FINAL")}
for m in PRED:
    for s in PRED[m]:
        PRED[m][s] = PRED[m][s][PRED[m][s].image_id.isin(set(IDS[s]))].reset_index(drop=True)

f1_here = evaluate(PRED["FINAL"]["test"], GT["test"], NC, THR["FINAL"], CLASS_NAMES)["summary"]["F1"]
print(f"final detector, test F1 recomputed here: {f1_here:.4f}")
if S1.get("FINAL_test_F1") is not None:
    assert abs(f1_here - S1["FINAL_test_F1"]) < 2e-3, "the rebuilt predictions do not reproduce stage 1's test F1"

# %% [markdown]
# ## 5. Priority index and reference labels
# `score(image) = sum over damage instances of weight[class] * (BASE + extent)`.
# Cut points come from the training images' reference scores only, then stay fixed.

# %%
def gt_scores(split, w=None, base=None):
    return image_scores(GT[split], IDS[split], WH, W_VEC if w is None else w, LIN, BASE if base is None else base)

S_REF = {s: gt_scores(s) for s in ("train", "val", "test")}
CUTS = np.quantile(S_REF["train"], PRIORITY_QUANTILES)
L_REF = {s: to_levels(S_REF[s], CUTS) for s in S_REF}
print("cut points on the training reference scores:", np.round(CUTS, 3))
dist = pd.DataFrame({s: pd.Series(L_REF[s]).value_counts(normalize=True).reindex(range(3), fill_value=0).values for s in L_REF},
                    index=LEVELS).round(3)
print("\nshare of images per reference priority level:"); print(dist.to_string()); save_table(dist, "reference_level_shares", index=True)

# what drives the reference score
_c = GT_ALL.assign(contrib=box_contributions(GT_ALL, WH, W_VEC, LIN, BASE))
drv = _c.groupby("class_id").agg(instances=("contrib", "size"), mean_contribution=("contrib", "mean"), total=("contrib", "sum"))
drv.insert(0, "class", [CLASS_NAMES[i] for i in drv.index]); drv.insert(1, "weight", [W_VEC[i] for i in drv.index])
drv["share_of_total_score"] = drv.total / drv.total.sum()
print("\ncontribution of each class to the reference score (all splits):"); print(drv.round(3).to_string(index=False))
save_table(drv, "index_class_contributions")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), layout="constrained")
ax[0].hist(S_REF["train"], bins=60, color=SERIES[0])
for c, lab, frac in zip(CUTS, ("Low | Medium", "Medium | High"), (0.95, 0.82)):
    ax[0].axvline(c, color=INK, lw=1, ls="--"); ax[0].text(c, ax[0].get_ylim()[1] * frac, f" {lab}", fontsize=8, color="#52514e", va="top")
ax[0].set_xlabel("reference priority score"); ax[0].set_ylabel("training images"); ax[0].set_title("Reference score distribution and cut points")
ax[1].barh([n.replace("_", " ") for n in drv["class"]][::-1], drv.share_of_total_score.values[::-1], color=SERIES[0], height=0.6)
ax[1].set_title("Share of total reference score by class"); ax[1].grid(axis="y", visible=False)
for i, v in enumerate(drv.share_of_total_score.values[::-1]):
    ax[1].text(v, i, f" {v:.0%}", va="center", fontsize=9, color="#52514e")
savefig("index_definition")

# %% [markdown]
# ## 6. Baseline: the index applied directly to detections
# No learning. Shows how much detection errors alone distort the priority, per detector.

# %%
def rule_scores(model, split, thr=None):
    p = PRED[model][split]
    t = conf_vector(THR[model] if thr is None else thr, NC)
    return image_scores(p[p.confidence.to_numpy() >= t[p.class_id.to_numpy()]], IDS[split], WH, W_VEC, LIN, BASE)

def report(name, split, s_pred=None, l_pred=None):
    l_pred = to_levels(s_pred, CUTS) if l_pred is None else l_pred
    row = dict(method=name, split=split, **level_metrics(L_REF[split], l_pred))
    if s_pred is not None:
        row.update(score_metrics(S_REF[split], s_pred))
    return row

rows = []
for m, lab in (("A", "Index on YOLO11m detections"), ("B", "Index on Faster R-CNN detections"), ("FINAL", "Index on final-detector detections")):
    for s in ("val", "test"):
        rows.append(report(lab, s, rule_scores(m, s)))
# one free parameter: the confidence threshold used for the index, chosen on validation
grid = np.round(np.arange(0.10, 0.91, 0.05), 2)
qwk = [cohen_kappa_score(L_REF["val"], to_levels(rule_scores("FINAL", "val", t), CUTS), weights="quadratic", labels=[0, 1, 2]) for t in grid]
RULE_T = float(grid[int(np.argmax(qwk))])
for s in ("val", "test"):
    rows.append(report(f"Index on final detector, threshold tuned for priority ({RULE_T:.2f})", s, rule_scores("FINAL", s, RULE_T)))
BASELINES = pd.DataFrame(rows)
print(BASELINES.round(4).to_string(index=False)); save_table(BASELINES, "baseline_index_on_detections")

# %% [markdown]
# ## 7. Trained priority predictor
# Inputs: features of the final detector's output for one image. Target: the reference priority.
# Trained on the 1,000 validation images (the detector never trained on them, so its errors there are
# realistic), hyperparameters chosen by 5-fold cross-validation, tested once on the 1,000 test images.

# %%
_thr_final = float(np.median(conf_vector(THR["FINAL"], NC)))
FEATURE_T = sorted({0.10, 0.25, round(_thr_final, 2), 0.60})

def build_features(pred, ids):
    p = pred.assign(contrib=box_contributions(pred, WH, W_VEC, LIN, BASE))
    zero = pd.DataFrame(0.0, index=ids, columns=range(NC))
    def table(q, col, how):
        if len(q) == 0:
            return zero.copy()
        g = q.groupby(["image_id", "class_id"])[col]
        t = (g.size() if how == "size" else g.sum() if how == "sum" else g.max()).unstack(fill_value=0)
        return t.reindex(index=ids, columns=range(NC), fill_value=0).astype(float)
    feats = {}
    for t in FEATURE_T:
        q = p[p.confidence >= t]
        cnt, con = table(q, "contrib", "size"), table(q, "contrib", "sum")
        for c in range(NC):
            feats[f"n_{CLASS_NAMES[c]}_t{t:.2f}"] = cnt[c].to_numpy(); feats[f"s_{CLASS_NAMES[c]}_t{t:.2f}"] = con[c].to_numpy()
        feats[f"index_score_t{t:.2f}"] = con.to_numpy().sum(1); feats[f"n_boxes_t{t:.2f}"] = cnt.to_numpy().sum(1)
    mx = table(p, "confidence", "max")
    for c in range(NC):
        feats[f"maxconf_{CLASS_NAMES[c]}"] = mx[c].to_numpy()
    cw = table(p.assign(cw=p.contrib * p.confidence), "cw", "sum")
    feats["index_score_confweighted"] = cw.to_numpy().sum(1)
    return pd.DataFrame(feats, index=ids)

X = {s: build_features(PRED["FINAL"][s], IDS[s]) for s in ("val", "test")}
print("features:", X["val"].shape[1], "| training images:", len(X["val"]), "| test images:", len(X["test"]))

folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X["val"], L_REF["val"]))
def qwk_of(l_pred, split="val"):
    return cohen_kappa_score(L_REF[split], l_pred, weights="quadratic", labels=[0, 1, 2])

CANDIDATES = []
for a in (0.1, 1.0, 10.0, 100.0):
    CANDIDATES.append(("Ridge regression", "score", dict(alpha=a), lambda a=a: make_pipeline(StandardScaler(), Ridge(alpha=a))))
for depth in (2, 3):
    for lr in (0.05, 0.1):
        for it in (100, 300):
            kw = dict(max_depth=depth, learning_rate=lr, max_iter=it, min_samples_leaf=10, random_state=SEED)
            CANDIDATES.append(("Gradient boosting regressor", "score", kw, lambda kw=kw: HistGradientBoostingRegressor(**kw)))
            CANDIDATES.append(("Gradient boosting classifier", "level", kw, lambda kw=kw: HistGradientBoostingClassifier(**kw)))

cv_rows, OOF = [], {}
for i, (family, target, params, make) in enumerate(CANDIDATES):
    y = S_REF["val"] if target == "score" else L_REF["val"]
    oof = cross_val_predict(make(), X["val"], y, cv=folds)
    lv = to_levels(oof, CUTS) if target == "score" else oof.astype(int)
    cv_rows.append(dict(i=i, family=family, target=target, params=json.dumps({k: v for k, v in params.items() if k not in ("random_state", "min_samples_leaf")}),
                        cv_QWK=qwk_of(lv), cv_accuracy=accuracy_score(L_REF["val"], lv)))
    OOF[i] = oof
CV = pd.DataFrame(cv_rows)
save_table(CV.drop(columns="i"), "priority_model_cv")
best_per_family = CV.sort_values("cv_QWK", ascending=False).groupby("family", sort=False).head(1)
print("best configuration per model family (5-fold CV on validation):"); print(best_per_family.drop(columns="i").round(4).to_string(index=False))

MODELS, rows = {}, []
for r in best_per_family.itertuples():
    family, target, params, make = CANDIDATES[r.i]
    est = make().fit(X["val"], S_REF["val"] if target == "score" else L_REF["val"])
    out = est.predict(X["test"])
    MODELS[family] = dict(est=est, target=target, test_out=out, oof=OOF[r.i], cv_QWK=r.cv_QWK)
    rows.append(report(f"Trained: {family}", "test", s_pred=out if target == "score" else None,
                       l_pred=None if target == "score" else out.astype(int)))
BEST = best_per_family.iloc[0].family
BEST_REG = best_per_family[best_per_family.target == "score"].iloc[0].family
print(f"\nselected by cross-validation: {BEST}  (score model used for road segments: {BEST_REG})")

RESULTS = pd.concat([BASELINES[BASELINES.split == "test"], pd.DataFrame(rows)], ignore_index=True)
RESULTS["selected"] = RESULTS.method.eq(f"Trained: {BEST}")
print("\n=== Priority prediction on the test set (reference = index on ground-truth boxes) ===")
print(RESULTS.drop(columns="split").round(4).to_string(index=False)); save_table(RESULTS, "priority_results_test")

# %%
# Does the trained model beat the no-learning baseline? Paired bootstrap over test images.
def levels_of(family, split="test"):
    m = MODELS[family]; out = m["test_out"] if split == "test" else m["oof"]
    return to_levels(out, CUTS) if m["target"] == "score" else out.astype(int)

L_ML = levels_of(BEST)
L_RULE = to_levels(rule_scores("FINAL", "test"), CUTS)
L_RULE_T = to_levels(rule_scores("FINAL", "test", RULE_T), CUTS)
def paired_bootstrap(l_a, l_b, n=2000):
    y = L_REF["test"]; d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        d.append(cohen_kappa_score(y[i], l_a[i], weights="quadratic", labels=[0, 1, 2]) -
                 cohen_kappa_score(y[i], l_b[i], weights="quadratic", labels=[0, 1, 2]))
    return np.percentile(d, [2.5, 50, 97.5])
boot = []
for lab, lb in (("index on final detector (stage-1 threshold)", L_RULE), (f"index on final detector (threshold {RULE_T:.2f})", L_RULE_T)):
    lo, mid, hi = paired_bootstrap(L_ML, lb)
    boot.append(dict(comparison=f"{BEST} minus {lab}", QWK_difference=mid, ci_low=lo, ci_high=hi,
                     verdict="trained model better" if lo > 0 else "baseline better" if hi < 0 else "no reliable difference"))
BOOT = pd.DataFrame(boot); print(BOOT.round(4).to_string(index=False)); save_table(BOOT, "priority_model_vs_baseline_bootstrap")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
for a, (lab, lv) in zip(ax[:2], ((f"Trained model ({BEST})", L_ML), ("Index applied to detections", L_RULE))):
    M = confusion_matrix(L_REF["test"], lv, labels=[0, 1, 2]); Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
    a.imshow(Mn, cmap="Blues", vmin=0, vmax=1); a.grid(False)
    a.set_xticks(range(3)); a.set_xticklabels(LEVELS); a.set_yticks(range(3)); a.set_yticklabels(LEVELS)
    for i in range(3):
        for j in range(3):
            a.text(j, i, f"{Mn[i, j]:.2f}\n({M[i, j]})", ha="center", va="center", fontsize=9, color="white" if Mn[i, j] > 0.55 else INK)
    a.set_xlabel("predicted priority"); a.set_ylabel("reference priority"); a.set_title(f"{lab}\ntest set, share of row")
s_ml = MODELS[BEST_REG]["test_out"]
ax[2].scatter(S_REF["test"], s_ml, s=8, alpha=0.5, color=SERIES[0], linewidths=0)
lim = max(S_REF["test"].max(), s_ml.max()) * 1.03
ax[2].plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--")
for c in CUTS:
    ax[2].axvline(c, color=GRID, lw=1); ax[2].axhline(c, color=GRID, lw=1)
ax[2].set_xlim(0, lim); ax[2].set_ylim(0, lim); ax[2].set_xlabel("reference score (ground-truth boxes)")
ax[2].set_ylabel(f"predicted score ({BEST_REG})"); ax[2].set_title("Predicted against reference score, test set")
savefig("priority_prediction_test")

imp = permutation_importance(MODELS[BEST]["est"], X["test"], S_REF["test"] if MODELS[BEST]["target"] == "score" else L_REF["test"],
                             n_repeats=10, random_state=SEED)
IMP = pd.DataFrame(dict(feature=X["test"].columns, importance=imp.importances_mean, sd=imp.importances_std)).sort_values("importance", ascending=False)
save_table(IMP, "priority_feature_importance")
top = IMP.head(12)[::-1]
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.barh(top.feature, top.importance, xerr=top.sd, color=SERIES[0], height=0.6, ecolor=MUTED)
ax.set_title(f"Permutation importance on the test set ({BEST})"); ax.grid(axis="y", visible=False); savefig("priority_feature_importance")

# %% [markdown]
# ## 8. Sensitivity of the index to its design choices
# The weights are our own. If modest changes to them reshuffled the priorities, the index would be arbitrary.

# %%
sens = []
ref_rank = S_REF["test"]
for k in range(200):
    w = W_VEC * rng.uniform(0.7, 1.3, NC)
    cuts = np.quantile(gt_scores("train", w), PRIORITY_QUANTILES)
    s_t = gt_scores("test", w); l_t = to_levels(s_t, cuts)
    p = PRED["FINAL"]["test"]; t = conf_vector(THR["FINAL"], NC)
    s_p = image_scores(p[p.confidence.to_numpy() >= t[p.class_id.to_numpy()]], IDS["test"], WH, w, LIN, BASE)
    sens.append(dict(same_level_as_nominal=float(np.mean(l_t == L_REF["test"])), rank_corr_with_nominal=spearmanr(s_t, ref_rank).statistic,
                     QWK_detections_vs_reference=cohen_kappa_score(l_t, to_levels(s_p, cuts), weights="quadratic", labels=[0, 1, 2])))
SENS = pd.DataFrame(sens)
summ = SENS.describe(percentiles=[0.05, 0.5, 0.95]).loc[["mean", "5%", "50%", "95%", "min"]].round(4)
print("class weights perturbed independently by up to +/-30% (200 draws), test set:"); print(summ.to_string()); save_table(summ, "sensitivity_weights", index=True)

rows = []
for b in (0.0, 0.05, 0.1, 0.2, 0.4):
    cuts = np.quantile(gt_scores("train", base=b), PRIORITY_QUANTILES); s_t = gt_scores("test", base=b)
    rows.append(dict(BASE=b, same_level_as_nominal=float(np.mean(to_levels(s_t, cuts) == L_REF["test"])), rank_corr_with_nominal=spearmanr(s_t, ref_rank).statistic))
for q in ((0.4, 0.7), (0.5, 0.8), (0.6, 0.9)):
    cuts = np.quantile(S_REF["train"], q)
    rows.append(dict(quantiles=str(q), same_level_as_nominal=float(np.mean(to_levels(S_REF["test"], cuts) == L_REF["test"])), rank_corr_with_nominal=1.0))
SENS2 = pd.DataFrame(rows); print("\npresence term and cut points:"); print(SENS2.round(4).to_string(index=False)); save_table(SENS2, "sensitivity_base_and_cuts")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].hist(SENS.same_level_as_nominal, bins=25, color=SERIES[0]); ax[0].set_xlabel("share of test images keeping their priority level")
ax[0].set_ylabel("weight draws"); ax[0].set_title("Stability of reference levels under +/-30% weight changes")
ax[1].hist(SENS.QWK_detections_vs_reference, bins=25, color=SERIES[0]); ax[1].set_xlabel("QWK, index on detections vs reference")
ax[1].set_title("Agreement is not specific to the chosen weights"); savefig("sensitivity")

# %% [markdown]
# ## 9. Road segments
# File names carry map coordinates. Images are binned into square cells and a segment's priority is the mean
# score of its images. Only validation and test images are used: validation images get out-of-fold
# predictions, test images get test predictions, so no image is scored by a model that was trained on it.

# %%
ev = images[images.split.isin(["val", "test"])].set_index("image_id")
order = IDS["val"] + IDS["test"]
ev = ev.loc[order]
xy = np.array([parse_xy(i) for i in order])
ev["x"], ev["y"] = xy[:, 0], xy[:, 1]
ev["ref"] = np.concatenate([S_REF["val"], S_REF["test"]])
ev["pred_model"] = np.concatenate([MODELS[BEST_REG]["oof"], MODELS[BEST_REG]["test_out"]])
ev["pred_index"] = np.concatenate([rule_scores("FINAL", "val"), rule_scores("FINAL", "test")])

parsed = float(np.isfinite(ev.x).mean())
spread_km = (float(np.nanmax(ev.x) - np.nanmin(ev.x)) / 1000, float(np.nanmax(ev.y) - np.nanmin(ev.y)) / 1000) if parsed > 0 else (0, 0)
COORDS_OK = parsed >= 0.95 and 1 <= spread_km[0] <= 300 and 1 <= spread_km[1] <= 300
print(f"coordinates parsed for {parsed:.1%} of images; extent {spread_km[0]:.1f} km x {spread_km[1]:.1f} km -> "
      f"{'using grid segments' if COORDS_OK else 'NOT plausible: falling back to districts as segments'}")
if COORDS_OK:
    print("x range (m):", round(float(ev.x.min())), "-", round(float(ev.x.max())), "| y range (m):", round(float(ev.y.min())), "-", round(float(ev.y.max())))
    occ = []
    for cell in (100, 200, 500, 1000):
        n = pd.Series(segment_ids(ev.x, ev.y, cell)).value_counts()
        occ.append(dict(cell_m=cell, segments=len(n), median_images=float(n.median()), segments_with_min_images=int((n >= MIN_IMAGES_PER_SEGMENT).sum()),
                        images_covered=int(n[n >= MIN_IMAGES_PER_SEGMENT].sum())))
    OCC = pd.DataFrame(occ); print(OCC.to_string(index=False)); save_table(OCC, "segment_cell_size_occupancy")
    ev["segment"] = segment_ids(ev.x, ev.y, SEGMENT_M)
else:
    ev["segment"] = ev["district"].replace("", "unknown")

seg = ev.groupby("segment").agg(n_images=("ref", "size"), x=("x", "mean"), y=("y", "mean"), district=("district", "first"),
                                ref=("ref", "mean"), pred_model=("pred_model", "mean"), pred_index=("pred_index", "mean")).reset_index()
SEG = seg[seg.n_images >= (MIN_IMAGES_PER_SEGMENT if COORDS_OK else 1)].copy()
SEG["ref_rank"] = SEG.ref.rank(ascending=False, method="min").astype(int)
SEG["pred_rank"] = SEG.pred_model.rank(ascending=False, method="min").astype(int)
SEG["pred_level"] = [LEVELS[i] for i in to_levels(SEG.pred_model, CUTS)]
SEG = SEG.sort_values("pred_rank")
save_table(SEG, "segments_ranked")
print(f"\nsegments ranked: {len(SEG)} (of {len(seg)}), covering {int(SEG.n_images.sum())} of {len(ev)} evaluated images")

if len(SEG) >= 5:
    k = max(3, int(round(0.1 * len(SEG))))
    rows = [dict(source=lab, spearman=spearmanr(SEG.ref, SEG[col]).statistic, **{f"top_{k}_overlap": top_k_overlap(SEG[col].to_numpy(), SEG.ref.to_numpy(), k)},
                 MAE=float(np.mean(np.abs(SEG.ref - SEG[col]))))
            for lab, col in ((f"Trained model ({BEST_REG})", "pred_model"), ("Index on detections", "pred_index"))]
    SEGM = pd.DataFrame(rows); print(SEGM.round(4).to_string(index=False)); save_table(SEGM, "segment_agreement")
    print("\nten highest-priority segments (predicted):")
    print(SEG.head(10)[["segment", "district", "n_images", "pred_model", "pred_rank", "ref", "ref_rank", "pred_level"]].round(3).to_string(index=False))
else:
    print("too few segments to rank; lower MIN_IMAGES_PER_SEGMENT or raise SEGMENT_M")

if COORDS_OK and len(SEG) >= 5:
    x0, y0 = SEG.x.min(), SEG.y.min()
    vmax = float(max(SEG.ref.max(), SEG.pred_model.max()))
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 0.8]), layout="constrained")
    for a, col, title in ((ax[0], "pred_model", "Predicted segment priority"), (ax[1], "ref", "Reference segment priority")):
        sc = a.scatter((SEG.x - x0) / 1000, (SEG.y - y0) / 1000, c=SEG[col], cmap="Blues", vmin=0, vmax=vmax,
                       s=14 + 6 * SEG.n_images.clip(upper=12), edgecolors="#52514e", linewidths=0.4)
        a.set_aspect("equal"); a.set_xlabel("east (km)"); a.set_ylabel("north (km)"); a.set_title(f"{title} ({SEGMENT_M} m cells, n = {len(SEG)})")
        fig.colorbar(sc, ax=a, fraction=0.046, pad=0.04, label="mean priority score")
    ax[2].scatter(SEG.ref, SEG.pred_model, s=14, color=SERIES[0], alpha=0.7, linewidths=0)
    ax[2].plot([0, vmax], [0, vmax], color=MUTED, lw=1, ls="--"); ax[2].set_xlabel("reference segment score"); ax[2].set_ylabel("predicted segment score")
    ax[2].set_title("Segment scores: predicted against reference"); savefig("segment_map")

# %% [markdown]
# ## 10. Summary

# %%
summary = dict(final_detector="+".join(FINAL_KEYS), detector_test_F1=float(f1_here), cut_points=[float(c) for c in CUTS],
               selected_model=BEST, selected_model_cv_QWK=float(MODELS[BEST]["cv_QWK"]),
               test=RESULTS.drop(columns="split").round(4).to_dict(orient="records"),
               model_vs_baseline=BOOT.round(4).to_dict(orient="records"),
               weight_sensitivity_mean=SENS.mean().round(4).to_dict(), coordinates_usable=bool(COORDS_OK), segments_ranked=int(len(SEG)))
(WORK / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=2, default=str))
print("\nfigures:", sorted(p.name for p in (WORK / "figs").glob("*.png")))
print("tables :", sorted(p.name for p in (WORK / "tables").glob("*.csv")))
