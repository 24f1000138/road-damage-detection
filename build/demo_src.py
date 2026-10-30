# %% [markdown]
# # SVRDD demo: run the trained system on road images
#
# Loads the two trained detectors from the detection notebook's output (`svrdd_run`), fuses their
# predictions with the frozen stage-1 settings (VGDF), computes the maintenance priority (DWPS) and draws
# the result. Works on SVRDD test images (with ground truth shown side by side) or on your own photos.
#
# Inputs to attach (Add Input): the detection notebook (its output contains `svrdd_run`), optionally the
# priority notebook (`priority_run`, for the cut points). The SVRDD dataset is downloaded from Zenodo
# automatically if it is not attached (Settings → Internet on). A GPU is optional.

# %% [markdown]
# ## 1. Setup

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=False)

# %% include=lib

# %%
import re, glob
import torch
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# %% include=prilib

# %% [markdown]
# ## 2. Configuration

# %%
INPUT = Path(os.environ.get("DEMO_INPUT", "/kaggle/input"))
MY_IMAGES = os.environ.get("DEMO_MY_IMAGES")     # optional folder with your own road photos
N_TEST_IMAGES = 6                                # random SVRDD test images to show
SEED = 7
DOWNLOAD_IF_MISSING = True                       # fetch SVRDD_YOLO.zip (2.9 GB) from Zenodo if the dataset is not attached
ZENODO_URL = "https://zenodo.org/api/records/10100129/files/SVRDD_YOLO.zip/content"
DATA_DIR = Path(os.environ.get("DEMO_DATA", "/kaggle/working/svrdd_data"))
CLASS_NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole",
               "manhole_cover", "longitudinal_patch", "transverse_patch"]
CLASS_WEIGHT = {"pothole": 10, "alligator_crack": 8, "transverse_crack": 5, "longitudinal_crack": 4,
                "longitudinal_patch": 2, "transverse_patch": 2, "manhole_cover": 0}
LINEAR_CLASSES = {"longitudinal_crack", "transverse_crack", "longitudinal_patch", "transverse_patch"}
BASE = 0.1
DEFAULT_CUTS = [2.0228, 3.7917]                 # cut points measured in the priority notebook
LEVELS = ["Low", "Medium", "High"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NC = len(CLASS_NAMES)
W_VEC = np.array([CLASS_WEIGHT[n] for n in CLASS_NAMES], dtype=float)
LIN = np.array([n in LINEAR_CLASSES for n in CLASS_NAMES])

def find_dir(name, must_have):
    hits = sorted((p for p in INPUT.rglob(name) if p.is_dir() and (p / must_have).exists()), key=lambda p: len(p.parts))
    return hits[0] if hits else None

STAGE1 = Path(os.environ["DEMO_STAGE1"]) if os.environ.get("DEMO_STAGE1") else find_dir("svrdd_run", "state.json")
assert STAGE1 is not None, "Attach the detection notebook's output (it contains svrdd_run)."
S1 = json.loads((STAGE1 / "state.json").read_text())
A_W = STAGE1 / "runs" / "A_yolo11m" / "weights" / "best.pt"
B_W = STAGE1 / "runs" / "B_frcnn_r50" / "best.pt"
pr = find_dir("priority_run", "summary.json")
CUTS = json.loads((pr / "summary.json").read_text())["cut_points"] if pr else DEFAULT_CUTS
print("stage 1:", STAGE1, "| final detector:", "+".join(S1["FINAL_KEYS"]), "| threshold:", S1["FINAL_thr"], "| cut points:", np.round(CUTS, 3))
print("YOLO11m weights:", A_W.exists(), "| Faster R-CNN weights:", B_W.exists(), "| device:", DEVICE)

# %% [markdown]
# ## 3. Load the trained detectors

# %%
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

yolo = YOLO(str(A_W))
frcnn = None
if "B" in S1["FINAL_KEYS"]:
    frcnn = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=1024, max_size=1024,
                                    box_score_thresh=0.001, box_detections_per_img=300, box_nms_thresh=0.5)
    frcnn.roi_heads.box_predictor = FastRCNNPredictor(frcnn.roi_heads.box_predictor.cls_score.in_features, NC + 1)
    sd = torch.load(B_W, map_location="cpu"); frcnn.load_state_dict(sd.get("model", sd)); frcnn.to(DEVICE).eval()
print("loaded:", "YOLO11m" + (" + Faster R-CNN" if frcnn is not None else ""))

def predict_yolo(img_rgb, view):
    arr = img_rgb[:, ::-1] if view.get("flip") else img_rgb
    res = yolo.predict([np.ascontiguousarray(arr[:, :, ::-1])], imgsz=view["imgsz"], conf=0.001, iou=view.get("nms_iou", 0.6),
                       max_det=300, verbose=False, device=0 if torch.cuda.is_available() else "cpu")[0]
    b = res.boxes.xyxy.float().cpu().numpy().astype(float); s = res.boxes.conf.float().cpu().numpy(); c = res.boxes.cls.cpu().numpy().astype(int)
    if view.get("flip"):
        b[:, [0, 2]] = img_rgb.shape[1] - b[:, [2, 0]]
    return b, s, c

@torch.no_grad()
def predict_frcnn(img_rgb, view):
    arr = img_rgb[:, ::-1] if view.get("flip") else img_rgb
    frcnn.transform.min_size = (view["imgsz"],); frcnn.transform.max_size = view["imgsz"]
    x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float().div(255).to(DEVICE)
    o = frcnn([x])[0]
    b = o["boxes"].float().cpu().numpy().astype(float); s = o["scores"].float().cpu().numpy(); c = o["labels"].cpu().numpy().astype(int) - 1
    if view.get("flip"):
        b[:, [0, 2]] = img_rgb.shape[1] - b[:, [2, 0]]
    return b, s, c

def model_preds(key, image_id, img):
    fn = predict_yolo if key == "A" else predict_frcnn
    dfs = [make_preds([image_id] * len(s), c, s, b) for b, s, c in (fn(img, v) for v in S1[f"{key}_views"])]
    return dfs[0] if len(dfs) == 1 else fuse_predictions(dfs, method="wbf", iou_thr=0.55, conf_type="max")

def analyse(path):
    """Fused detections above the frozen threshold, priority score and level for one image."""
    image_id = Path(path).stem
    with Image.open(path) as im:
        img = np.asarray(im.convert("RGB"))
    t0 = time.time()
    preds = {k: model_preds(k, image_id, img) for k in S1["FINAL_KEYS"]}
    if S1["FINAL_KEYS"] == ["A"]:
        d = preds["A"]
    else:
        F = S1["FUSE"]; w = [F["weights"][["A", "B", "C"].index(k)] for k in S1["FINAL_KEYS"]]
        d = fuse_predictions([preds[k] for k in S1["FINAL_KEYS"]], w, F["method"], F["iou"], F["conf_type"], skip_thr=F.get("skip", 0.05))
    thr = conf_vector(S1["FINAL_thr"], NC)
    d = d[d.confidence.to_numpy() >= thr[d.class_id.to_numpy()]].reset_index(drop=True)
    wh = {image_id: (img.shape[1], img.shape[0])}
    score = float(image_scores(d, [image_id], wh, W_VEC, LIN, BASE)[0])
    level = LEVELS[int(to_levels([score], CUTS)[0])]
    return img, d, score, level, time.time() - t0

def draw(ax, img, boxes, title):
    ax.imshow(img); ax.set_title(title, fontsize=9); ax.axis("off")
    for r_ in boxes.itertuples():
        col = COLORS[int(r_.class_id) % len(COLORS)]
        ax.add_patch(Rectangle((r_.x1, r_.y1), r_.x2 - r_.x1, r_.y2 - r_.y1, fill=False, edgecolor=col, linewidth=1.6))
        label = CLASS_NAMES[int(r_.class_id)].replace("_", " ") + (f" {r_.confidence:.2f}" if hasattr(r_, "confidence") else "")
        ax.text(r_.x1, max(r_.y1 - 4, 8), label, fontsize=6, color="white", bbox=dict(facecolor=col, edgecolor="none", pad=1))

# %% [markdown]
# ## 4. SVRDD test images: prediction next to ground truth

# %%
import zipfile
def find_split_file(base):
    return next((p for p in base.rglob("test.txt") if p.parent.name.lower().startswith("svrdd")), None) if base.exists() else None

data_root = find_split_file(INPUT) or find_split_file(DATA_DIR)
if data_root is None and DOWNLOAD_IF_MISSING:
    DATA_DIR.mkdir(parents=True, exist_ok=True); z = DATA_DIR / "SVRDD_YOLO.zip"
    print("downloading SVRDD_YOLO.zip from Zenodo (2.9 GB; notebook internet must be ON, a few minutes on Kaggle) ...")
    subprocess.run(["wget", "-q", "-c", "-O", str(z), ZENODO_URL], check=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(DATA_DIR)
    z.unlink()
    data_root = find_split_file(DATA_DIR)
if data_root is None:
    print("SVRDD not attached: skip this section, or attach the dataset to compare with ground truth.")
else:
    root = data_root.parent
    stems = [Path(l.strip().replace("\\", "/")).stem for l in data_root.read_text().splitlines() if l.strip()]
    imgs = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    labels = {p.stem: p for p in root.rglob("*.txt") if p.stem in imgs}
    chosen = [s for s in np.random.RandomState(SEED).permutation(stems) if s in imgs][:N_TEST_IMAGES]
    rows = []
    fig, axes = plt.subplots(len(chosen), 2, figsize=(10, 5 * len(chosen)), squeeze=False)
    for k, s in enumerate(chosen):
        img, d, score, level, secs = analyse(imgs[s])
        H, W = img.shape[:2]; gt = []
        for line in (labels[s].read_text().splitlines() if s in labels else []):
            v = line.split()
            if len(v) >= 5:
                c, cx, cy, w, h = int(float(v[0])), *map(float, v[1:5])
                gt.append((c,) + tuple(yolo_to_xyxy(cx, cy, w, h, W, H)))
        gt = pd.DataFrame(gt, columns=["class_id", "x1", "y1", "x2", "y2"])
        ref = float(image_scores(gt.assign(image_id=s), [s], {s: (W, H)}, W_VEC, LIN, BASE)[0])
        draw(axes[k, 0], img, d, f"Prediction: {len(d)} damages | priority {level} (score {score:.2f}) | {secs:.2f} s")
        draw(axes[k, 1], img, gt, f"Ground truth: {len(gt)} damages | priority {LEVELS[int(to_levels([ref], CUTS)[0])]} (score {ref:.2f})")
        rows.append(dict(image=s[:24], predicted=len(d), ground_truth=len(gt), pred_level=level,
                         ref_level=LEVELS[int(to_levels([ref], CUTS)[0])], seconds=round(secs, 2)))
    plt.tight_layout(); plt.show()
    print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# ## 5. Your own road photos (optional)
# Upload photos as a Kaggle dataset, attach it, and set `MY_IMAGES` in section 2 to its folder.

# %%
if MY_IMAGES and Path(MY_IMAGES).exists():
    paths = sorted(p for p in Path(MY_IMAGES).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:8]
    fig, axes = plt.subplots(len(paths), 1, figsize=(8, 6 * len(paths)), squeeze=False)
    for k, pth in enumerate(paths):
        img, d, score, level, secs = analyse(pth)
        draw(axes[k, 0], img, d, f"{pth.name}: {len(d)} damages | priority {level} (score {score:.2f})")
        print(pth.name, "->", level, f"score {score:.2f}", "|", d.class_id.map(lambda c: CLASS_NAMES[c]).value_counts().to_dict())
    plt.tight_layout(); plt.show()
else:
    print("No folder of own photos set (MY_IMAGES).")
