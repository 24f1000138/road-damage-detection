"""Synthetic checks for the priority library (stage 2).

    python tests/test_priority.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
from build_notebook import parse_cells, parse_priority_cells

ns = {}
for c in parse_cells():
    if "lib" in c["tags"]:
        exec(compile(c["source"], "<lib>", "exec"), ns)
exec("import re\nfrom scipy.stats import spearmanr\n"
     "from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score", ns)
for c in parse_priority_cells():
    if "prilib" in c["tags"]:
        exec(compile(c["source"], "<prilib>", "exec"), ns)
globals().update(ns)

WH = {"i": (1000, 1000), "j": (1000, 1000), "k": (500, 500)}
W = np.array([4.0, 10.0, 0.0])          # linear crack, pothole (areal), manhole (ignored)
LIN = np.array([True, False, False])


def boxes(rows):
    return pd.DataFrame(rows, columns=["image_id", "class_id", "x1", "y1", "x2", "y2"])


def test_contributions():
    df = boxes([("i", 0, 0, 0, 300, 400),        # linear: diagonal 500 / 1414.2
                ("i", 1, 0, 0, 100, 100),        # areal: sqrt(1e4 / 1e6) = 0.1
                ("i", 2, 0, 0, 900, 900)])       # weight 0
    c = box_contributions(df, WH, W, LIN, base=0.1)
    assert np.allclose(c, [4 * (0.1 + 500 / np.hypot(1000, 1000)), 10 * (0.1 + 0.1), 0.0])
    # scale-free: the same scene at half resolution scores the same
    half = boxes([("k", 0, 0, 0, 150, 200), ("k", 1, 0, 0, 50, 50)])
    assert np.allclose(box_contributions(half, WH, W, LIN, 0.1), c[:2])
    assert len(box_contributions(df.iloc[:0], WH, W, LIN, 0.1)) == 0


def test_image_scores_and_levels():
    df = boxes([("i", 1, 0, 0, 100, 100), ("i", 1, 200, 200, 300, 300), ("j", 2, 0, 0, 50, 50)])
    s = image_scores(df, ["i", "j", "k"], WH, W, LIN, 0.1)
    assert np.allclose(s, [4.0, 0.0, 0.0]), s            # two potholes; manhole only; no boxes
    assert s.shape == (3,)
    # more damage never lowers the score
    more = pd.concat([df, boxes([("j", 0, 0, 0, 10, 500)])])
    assert (image_scores(more, ["i", "j", "k"], WH, W, LIN, 0.1) >= s).all()
    assert list(to_levels([0.0, 0.99, 1.0, 2.5, 3.0, 9.0], [1.0, 3.0])) == [0, 0, 1, 1, 2, 2]


def test_metrics():
    y = np.array([0, 0, 1, 1, 2, 2])
    m = level_metrics(y, y)
    assert m["accuracy"] == 1 and m["macro_F1"] == 1 and m["QWK"] == 1 and m["high_recall"] == 1
    worse = level_metrics(y, np.array([2, 2, 1, 1, 0, 0])); near = level_metrics(y, np.array([1, 0, 1, 2, 2, 1]))
    assert worse["QWK"] < near["QWK"] < 1, "confusing Low with High must cost more than adjacent levels"
    assert worse["high_recall"] == 0
    s = score_metrics([1, 2, 3, 4], [2, 4, 6, 8]); assert abs(s["spearman"] - 1) < 1e-9 and abs(s["MAE"] - 2.5) < 1e-9
    assert np.isnan(score_metrics([1, 2, 3], [5, 5, 5])["spearman"])


def test_coordinates_and_segments():
    assert parse_xy("0900220012200316144604377HJ_1294207500_482955500") == (12942075.0, 4829555.0)
    assert all(np.isnan(v) for v in parse_xy("train_0001"))
    ids = segment_ids([12942075.0, 12942190.0, 12942210.0, np.nan], [4829555.0, 4829590.0, 4829555.0, 1.0], 200)
    assert ids[0] == ids[1] and ids[1] != ids[2] and ids[3] == ""
    assert top_k_overlap([5, 4, 3, 2, 1], [5, 4, 3, 2, 1], 2) == 1.0
    assert top_k_overlap([5, 4, 3, 2, 1], [1, 2, 3, 4, 5], 2) == 0.0
    assert top_k_overlap([5, 4, 3, 2, 1], [5, 1, 4, 2, 3], 2) == 0.5


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print("ok  ", t.__name__)
    print(f"{len(tests)} test groups passed")
