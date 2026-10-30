"""End-to-end dry run of every notebook cell on a tiny synthetic dataset, on CPU.

    python tests/integration_smoke.py [workdir]

What is real: dataset discovery/cleaning, analysis, the evaluator, threshold tuning, view selection,
tiling, Faster R-CNN training/inference (torchvision, random init, 1 epoch), fusion, complementarity,
tables and plots.
What is faked: Ultralytics. A stand-in `YOLO` class "detects" the synthetic rectangles by colour, so
the YOLO training/inference calls are exercised only at the interface level. The real Ultralytics path
must be validated on Kaggle with SMOKE = True.
"""
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))
from build_notebook import parse_cells

base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="svrdd_smoke_"))
if base.exists():
    shutil.rmtree(base)
data = base / "input" / "SVRDD_YOLO"
NCLS, SIZE = 7, 256
COLORS = [(40 + 30 * c, 255 - 30 * c, 100) for c in range(NCLS)]
rng = np.random.RandomState(0)

# ---------------------------------------------------------------- synthetic dataset
# Same layout as the real archive: district folders with images/ and labels/, splits given by list files.
DISTRICTS = ["Haidian", "Xicheng", "Chaoyang"]
WHERE = {}
for dname in DISTRICTS:
    (data / dname / "images").mkdir(parents=True); (data / dname / "labels").mkdir(parents=True)
for split, n in (("train", 120), ("val", 100), ("test", 100)):
    entries = []
    for i in range(n):
        dname = DISTRICTS[i % 3]; WHERE[f"{split}_{i:04d}"] = dname
        entries.append(f"D:\\SVRDD\\{dname}\\images\\{split}_{i:04d}.png")      # Windows-style absolute paths
        img_dir, lbl_dir = data / dname / "images", data / dname / "labels"
        img = np.zeros((SIZE, SIZE, 3), np.uint8); lines = []
        for c in rng.choice(NCLS, size=rng.randint(0, 4), replace=False):
            w, h = (rng.randint(60, 200), rng.randint(8, 20)) if c in (1, 5) else \
                   (rng.randint(8, 20), rng.randint(60, 200)) if c in (0, 4) else (rng.randint(20, 70), rng.randint(20, 70))
            x, y = rng.randint(0, SIZE - w), rng.randint(0, SIZE - h)
            if (img[y:y + h, x:x + w] != 0).any():
                continue                                             # keep objects disjoint
            img[y:y + h, x:x + w] = COLORS[c]
            lines.append(f"{c} {(x + w / 2) / SIZE:.6f} {(y + h / 2) / SIZE:.6f} {w / SIZE:.6f} {h / SIZE:.6f}")
        Image.fromarray(img).save(img_dir / f"{split}_{i:04d}.png")
        (lbl_dir / f"{split}_{i:04d}.txt").write_text("\n".join(lines))
    (data / f"{split}.txt").write_text("\n".join(entries) + "\n")
# deliberate defects the verification cell must cope with
(data / WHERE["train_0000"] / "labels" / "train_0000.txt").write_text("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n9 0.5 0.5 0.1 0.1\n1 0.98 0.5 0.2 0.1\nabc\n")
shutil.copy(data / WHERE["train_0001"] / "images" / "train_0001.png", data / WHERE["val_0000"] / "images" / "val_0000.png")   # leaked duplicate
Image.fromarray(np.zeros((SIZE, SIZE, 3), np.uint8)).save(data / "Haidian" / "images" / "unlisted_0001.png")       # in no split list

# ---------------------------------------------------------------- stand-in for Ultralytics
class _T:                                   # tensor-like with the attributes the notebook touches
    def __init__(self, a): self.a = torch.as_tensor(np.asarray(a))
    def float(self): return self.a.float()
    def cpu(self): return self.a

class _Res:
    def __init__(self, b, s, c):
        self.boxes = types.SimpleNamespace(xyxy=_T(np.asarray(b, np.float32).reshape(-1, 4)), conf=_T(np.asarray(s, np.float32)),
                                           cls=_T(np.asarray(c, np.float32)))

class FakeYOLO:
    calls = []
    epoch_ranges = []
    def __init__(self, weights): self.weights = str(weights); self.cb = {}
    def add_callback(self, name, fn): self.cb[name] = fn
    def train(self, **kw):
        FakeYOLO.calls.append(("train", kw))
        if kw.get("resume"):
            wdir = Path(self.weights).parent; ck = torch.load(self.weights)
            assert "epoch" in ck, "resume was given a stripped checkpoint"
            start, epochs = ck["epoch"] + 1, ck["epochs"]
            assert start < epochs, "nothing to resume: the run had already finished"
        else:
            wdir = Path(kw["project"]) / kw["name"] / "weights"; wdir.mkdir(parents=True, exist_ok=True)
            start, epochs = 0, kw["epochs"]
            (wdir.parent / "results.csv").write_text("epoch,time,train/box_loss,val/box_loss,metrics/mAP50(B),metrics/mAP50-95(B)\n")
        tr = types.SimpleNamespace(last=wdir / "last.pt", wdir=wdir, stop=False, epoch=start, epochs=epochs)
        for e in range(start, epochs):
            tr.epoch = e
            torch.save({"epoch": e, "epochs": epochs}, wdir / "last.pt")
            with open(wdir.parent / "results.csv", "a") as f:
                f.write(f"{e + 1},{60 * (e + 1)},{1.2 - 0.1 * e},{1.3 - 0.1 * e},{0.4 + 0.05 * e},{0.2 + 0.05 * e}\n")
            if "on_model_save" in self.cb:
                self.cb["on_model_save"](tr)
            if tr.stop:
                break
        FakeYOLO.epoch_ranges.append((start + 1, tr.epoch + 1))
        torch.save({"stripped": True}, wdir / "last.pt"); torch.save({"stripped": True}, wdir / "best.pt")   # as Ultralytics does
    def predict(self, arrays, imgsz, conf, iou, max_det, verbose, device):
        out, r = [], np.random.RandomState(len(arrays) + imgsz)
        for a in arrays:
            a = a[:, :, ::-1]                                          # BGR -> RGB
            b, s, c = [], [], []
            for k, col in enumerate(COLORS):
                ys, xs = np.where((a == np.array(col)).all(-1))
                if len(xs) and r.rand() > 0.15:                        # misses 15% of objects
                    j = r.randint(-2, 3, 4)
                    b.append([xs.min() + j[0], ys.min() + j[1], xs.max() + 1 + j[2], ys.max() + 1 + j[3]])
                    s.append(0.4 + 0.6 * r.rand()); c.append(k if r.rand() > 0.05 else (k + 1) % NCLS)
            for _ in range(r.randint(0, 3)):                           # low-confidence false alarms
                x, y = r.randint(0, a.shape[1] - 30), r.randint(0, a.shape[0] - 30)
                b.append([x, y, x + 25, y + 25]); s.append(0.05 + 0.5 * r.rand()); c.append(r.randint(NCLS))
            out.append(_Res(b, s, c))
        return out
    def val(self, **kw):
        return types.SimpleNamespace(box=types.SimpleNamespace(mp=0.7, mr=0.6, map50=0.65, map=0.4))

sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=FakeYOLO, RTDETR=FakeYOLO)

# ---------------------------------------------------------------- run the notebook cells
os.environ.update(SVRDD_ROOT=str(base / "input"), SVRDD_WORK=str(base / "work" / "svrdd_run"), SVRDD_TEMP=str(base / "temp"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: plt.close("all")

OVERRIDES = """
CFG_A.update(imgsz=256, epochs=2); IMGSZ = 256; INFER_BATCH = 8
CFG_B.update(epochs=1, milestones=(1,), warmup_iters=5, train_sizes=(128,), max_size=128, batch=4, workers=0, pretrained=False)
CFG_C.update(imgsz=256, epochs=1)
DOWNLOAD_IF_MISSING = False; FORCE_TILING = True
"""

PAUSE_OVERRIDES = """
CFG_A.update(imgsz=256); IMGSZ = 256; INFER_BATCH = 8
CFG_B.update(warmup_iters=5, train_sizes=(128,), max_size=128, batch=4, workers=0, pretrained=False)
DOWNLOAD_IF_MISSING = False
"""

def run_all(extra="", label="run", pause_test=False):
    ns = {"__name__": "__main__"}
    cells = [c for c in parse_cells() if c["type"] == "code"]
    for k, c in enumerate(cells):
        src = c["source"]
        if "pip" in src and "install" in src:
            continue
        if "SMOKE = False" in src:
            if pause_test:
                assert "PAUSE_TEST = False" in src
                src = src.replace("SMOKE = False", "SMOKE = True", 1).replace("PAUSE_TEST = False", "PAUSE_TEST = True", 1)
                src += PAUSE_OVERRIDES + extra
            else:
                src += OVERRIDES + extra
        if c["stage"] and ns.get("PAUSED"):
            continue                                                    # what %%stage does in the notebook
        try:
            exec(compile(src, f"<cell {k}>", "exec"), ns)
        except Exception:
            print(f"\n[{label}] FAILED in code cell {k}:\n" + "\n".join(src.splitlines()[:4]))
            raise
    return ns

print("=== pass 1: session budget of zero -> training must pause and later cells must be skipped ===")
ns = run_all("SESSION_BUDGET_H = 0.0\n", "pause")
assert ns["PAUSED"] and ns["A_WEIGHTS"] is None and "A_val" not in ns
assert (Path(ns["WORK"]) / "runs" / "A_yolo11m" / "weights" / "resume.pt").exists()

print("\n=== pass 2: resume and run everything (RT-DETR branch forced on) ===")
ns = run_all("RUN_C = True; GATE.update(c_headroom=-1.0, b_only_tp=0.0, ensemble=-1.0); TARGET_F1 = 2.0\n", "full")
assert any(k == "train" and kw.get("resume") for k, kw in FakeYOLO.calls), "model A did not resume from resume.pt"
W = Path(ns["WORK"])
abl = ns["pd"].read_csv(W / "tables" / "ablation.csv")
print("\n", abl[["stage", "split", "P", "R", "F1", "mAP50", "mAP50_95"]].to_string(index=False))
assert {"val", "test"} <= set(abl.split) and (abl.F1.between(0, 1)).all()
assert ns["issue_total"]["bad_class_id"] == 1 and ns["issue_total"]["non_numeric"] == 1 and ns["issue_total"]["duplicate_lines"] == 1
assert ns["issue_total"]["clipped_to_image"] == 1, dict(ns["issue_total"])
_grp = ns["exact_cross"].groupby("md5").image_id.apply(set)
assert any({"train_0001", "val_0000"} <= s for s in _grp), "the leaked train/val duplicate was not detected"
assert (ns["LEAK"].pair == "train-val").any()
assert ns["KEEP_B"] and ns["C_GATE_OPEN"] and "C" in ns["VAL_PREDS"]
assert any(v.get("tile") for v in ns["cands"]), "tiling candidate was not evaluated"
figs = sorted(p.name for p in (W / "figs").glob("*.png")); print("\nfigures:", figs)
assert len(figs) >= 10, figs
for d in ns["VAL_PREDS"].values():
    assert list(d.columns) == ns["PRED_COLS"] and d.class_id.between(0, NCLS - 1).all()

print("\n=== pass 3: rerun with caches (nothing retrains) ===")
n_train = sum(k == "train" for k, _ in FakeYOLO.calls)
ns3 = run_all("", "cached")
assert sum(k == "train" for k, _ in FakeYOLO.calls) == n_train, "a finished model was retrained"

print("\n=== pause test: three consecutive runs, as on Kaggle with SMOKE = True and PAUSE_TEST = True ===")
FakeYOLO.epoch_ranges.clear()
p1 = run_all(label="pause-test 1", pause_test=True)
PW = Path(p1["WORK"]); assert PW.name.endswith("_smoke_pausetest"), PW
assert p1["PAUSED"] and FakeYOLO.epoch_ranges == [(1, 1)] and not (PW / "runs" / "A_yolo11m" / "DONE").exists()
p2 = run_all(label="pause-test 2", pause_test=True)
assert FakeYOLO.epoch_ranges == [(1, 1), (2, 3)], f"model A did not resume at epoch 2: {FakeYOLO.epoch_ranges}"
assert (PW / "runs" / "A_yolo11m" / "DONE").exists() and p2["PAUSED"] and p2["B_WEIGHTS"] is None
blog = p2["pd"].read_csv(PW / "runs" / "B_frcnn_r50" / "log.csv"); assert list(blog.epoch) == [1], blog
p3 = run_all(label="pause-test 3", pause_test=True)
blog = p3["pd"].read_csv(PW / "runs" / "B_frcnn_r50" / "log.csv"); assert list(blog.epoch) == [1, 2], blog
assert not p3["PAUSED"] and (PW / "runs" / "B_frcnn_r50" / "DONE").exists()
assert FakeYOLO.epoch_ranges == [(1, 1), (2, 3)], "model A was retrained"
assert len(list((PW / "figs").glob("*.png"))) >= 10
print("pause test passed: A paused after epoch 1 and resumed at 2; B paused after epoch 1 and resumed at 2")
print("\nintegration smoke passed; workdir:", base)
