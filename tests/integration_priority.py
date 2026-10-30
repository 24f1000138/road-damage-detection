"""End-to-end dry run of the stage-2 (priority) notebook on the synthetic data left by integration_smoke.py.

    python tests/integration_smoke.py <dir>        # first: produces <dir>/input and <dir>/work/svrdd_run
    python tests/integration_priority.py <dir>

The synthetic file names carry no coordinates, so this script makes a copy in which every image is renamed
to the real SVRDD pattern '<name>_<x*100>_<y*100>' (in the dataset, the split lists and the cached
predictions). That exercises the road-segment code path used on the real data.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))
from build_notebook import parse_priority_cells

base = Path(sys.argv[1])
src_data, src_s1 = base / "input" / "SVRDD_YOLO", base / "work" / "svrdd_run"
assert src_data.exists() and (src_s1 / "state.json").exists(), "run tests/integration_smoke.py on this directory first"
out = base / "priority_case"
if out.exists():
    shutil.rmtree(out)

# ---------------------------------------------------------------- renamed copy with coordinates
stems = sorted(p.stem for p in src_data.rglob("*.png"))
rng = np.random.RandomState(1)
centres = rng.uniform(0, 6000, (30, 2))                       # 30 places within a 6 km square
new = {}
for i, s in enumerate(stems):
    cx, cy = centres[i % len(centres)]
    x, y = 12_950_000 + cx - cx % 200 + rng.uniform(5, 190), 4_825_000 + cy - cy % 200 + rng.uniform(5, 190)
    new[s] = f"{s.replace('_', '')}XX_{int(round(x * 100))}_{int(round(y * 100))}"
for p in src_data.rglob("*"):
    if p.is_dir():
        continue
    rel = p.relative_to(src_data)
    dst = out / "input" / "SVRDD_YOLO" / rel.parent / ((new[p.stem] + p.suffix) if p.stem in new else rel.name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".txt" and p.stem in ("train", "val", "test"):
        lines = p.read_text().splitlines()
        dst.write_text("\n".join(l.replace(Path(l.replace("\\", "/")).stem, new[Path(l.replace("\\", "/")).stem]) for l in lines if l.strip()) + "\n")
    else:
        shutil.copy2(p, dst)
(out / "stage1" / "preds").mkdir(parents=True)
shutil.copy2(src_s1 / "state.json", out / "stage1" / "state.json")
for f in (src_s1 / "preds").glob("*.parquet"):
    d = pd.read_parquet(f); d["image_id"] = d["image_id"].map(new); assert d.image_id.notna().all(); d.to_parquet(out / "stage1" / "preds" / f.name)

# ---------------------------------------------------------------- run the notebook cells
os.environ.update(SVRDD_ROOT=str(out / "input"), SVRDD_STAGE1=str(out / "stage1"), PRIORITY_WORK=str(out / "work"), SVRDD_TEMP=str(out / "temp"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: plt.close("all")

ns = {"__name__": "__main__"}
for k, c in enumerate(c for c in parse_priority_cells() if c["type"] == "code"):
    src = c["source"]
    if "DOWNLOAD_IF_MISSING = True" in src:
        src += "\nDOWNLOAD_IF_MISSING = False\n"
    try:
        exec(compile(src, f"<cell {k}>", "exec"), ns)
    except Exception:
        print(f"\nFAILED in code cell {k}:\n" + "\n".join(src.splitlines()[:4]))
        raise

W = Path(ns["WORK"])
summary = json.loads((W / "summary.json").read_text())
assert ns["COORDS_OK"], "coordinates were not recognised"
assert len(ns["SEG"]) >= 5 and (W / "tables" / "segments_ranked.csv").exists()
assert set(ns["RESULTS"].columns) >= {"method", "accuracy", "macro_F1", "QWK"} and len(ns["RESULTS"]) >= 6
assert ns["RESULTS"].selected.sum() == 1
assert abs(ns["f1_here"] - ns["S1"]["FINAL_test_F1"]) < 2e-3
# the reference labels follow the quantile design on the training split
shares = np.bincount(ns["L_REF"]["train"], minlength=3) / len(ns["L_REF"]["train"])
assert abs(shares[0] - 0.5) < 0.08 and abs(shares[2] - 0.2) < 0.08, shares
figs = sorted(p.name for p in (W / "figs").glob("*.png"))
assert len(figs) == 5, figs
print("\nintegration (priority) passed; workdir:", out)
