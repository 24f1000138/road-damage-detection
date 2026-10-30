"""Dry run of the demo notebook on the synthetic data left by integration_smoke.py (CPU, Ultralytics stubbed).

    python tests/integration_smoke.py <dir>
    python tests/integration_demo.py <dir>

Faster R-CNN runs for real (the randomly initialised smoke-test weights); YOLO is a colour-matching stand-in.
The stage-1 state is copied and switched to the fused A+B configuration so that the fusion path is exercised.
"""
import json, os, shutil, sys, types
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))
from build_notebook import parse_priority_cells, SRC3

base = Path(sys.argv[1])
s1_src = base / "work" / "svrdd_run"
assert (s1_src / "runs" / "B_frcnn_r50" / "best.pt").exists(), "run tests/integration_smoke.py on this directory first"
case = base / "demo_case"
if case.exists():
    shutil.rmtree(case)
s1 = case / "svrdd_run"; s1.mkdir(parents=True)
shutil.copytree(s1_src / "runs", s1 / "runs")
state = json.loads((s1_src / "state.json").read_text())
state.update(FINAL_KEYS=["A", "B"], FUSE=dict(method="wbf", conf_type="avg", iou=0.5, weights=[1.0, 0.75], skip=0.05), FINAL_thr=0.3)
(s1 / "state.json").write_text(json.dumps(state))

COLORS = [(40 + 30 * c, 255 - 30 * c, 100) for c in range(7)]
class _T:
    def __init__(self, a): self.a = torch.as_tensor(np.asarray(a))
    def float(self): return self.a.float()
    def cpu(self): return self.a
class FakeYOLO:
    def __init__(self, w): pass
    def predict(self, arrays, imgsz, conf, iou, max_det, verbose, device):
        out = []
        for a in arrays:
            a = a[:, :, ::-1]; b, s, c = [], [], []
            for k, col in enumerate(COLORS):
                ys, xs = np.where((a == np.array(col)).all(-1))
                if len(xs):
                    b.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]); s.append(0.9); c.append(k)
            out.append(types.SimpleNamespace(boxes=types.SimpleNamespace(xyxy=_T(np.asarray(b, np.float32).reshape(-1, 4)),
                                                                     conf=_T(np.asarray(s, np.float32)), cls=_T(np.asarray(c, np.float32)))))
        return out
sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=FakeYOLO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
shown = []
plt.show = lambda *a, **k: (shown.append(len(plt.gcf().axes)), plt.close("all"))

os.environ.update(DEMO_INPUT=str(base / "input"), DEMO_STAGE1=str(s1), DEMO_MY_IMAGES=str(base / "input" / "SVRDD_YOLO" / "Haidian" / "images"))
ns = {"__name__": "__main__"}
for k, c in enumerate(c for c in parse_priority_cells(SRC3) if c["type"] == "code"):
    if "pip" in c["source"] and "install" in c["source"]:
        continue
    exec(compile(c["source"], f"<demo cell {k}>", "exec"), ns)

assert ns["frcnn"] is not None, "Faster R-CNN was not loaded"
img, d, score, level, secs = ns["analyse"](next((base / "input" / "SVRDD_YOLO").rglob("test_0003.png")))
assert set(d.columns) >= {"class_id", "confidence", "x1"} and level in ("Low", "Medium", "High")
assert len(shown) == 2, shown
print(f"demo integration passed: example -> {len(d)} detections, priority {level} (score {score:.2f}); figures shown: {shown}")
