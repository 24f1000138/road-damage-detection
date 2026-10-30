"""Synthetic checks for the notebook's pure functions (evaluator, WBF, tiling, complementarity).

    python tests/test_lib.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
from build_notebook import parse_cells

ns = {}
for c in parse_cells():
    if "lib" in c["tags"]:
        exec(compile(c["source"], "<lib>", "exec"), ns)
globals().update(ns)

NAMES = ["a", "b", "c"]


def gt_table(rows):
    return pd.DataFrame(rows, columns=["image_id", "class_id"] + BOX)


def test_perfect_and_swapped():
    gt = gt_table([("i1", 0, 10, 10, 50, 50), ("i1", 1, 60, 60, 90, 90), ("i2", 2, 0, 0, 30, 30)])
    pred = make_preds(gt.image_id, gt.class_id, [0.9, 0.8, 0.7], gt[BOX].values)
    r = evaluate(pred, gt, 3, 0.25, NAMES)["summary"]
    # 0.995, not 1.0: the 101-point interpolation used by Ultralytics caps a perfect detector there
    assert r["F1"] == 1.0 and r["mAP50"] > 0.99 and r["mAP50_95"] > 0.99, r
    swapped = make_preds(gt.image_id, (gt.class_id + 1) % 3, [0.9, 0.8, 0.7], gt[BOX].values)
    r = evaluate(swapped, gt, 3, 0.25, NAMES)["summary"]
    assert r["F1"] == 0.0 and r["TP"] == 0 and r["FP"] == 3 and r["FN"] == 3, r


def test_duplicates_and_iou_boundary():
    gt = gt_table([("i1", 0, 0, 0, 100, 100)])
    # two detections of the same object: the second is a false positive
    pred = make_preds(["i1", "i1"], [0, 0], [0.9, 0.8], [[0, 0, 100, 100], [2, 2, 100, 100]])
    r = evaluate(pred, gt, 1, 0.25)["summary"]
    assert (r["TP"], r["FP"], r["FN"]) == (1, 1, 0), r
    # IoU exactly 0.5 counts, just below does not
    assert evaluate(make_preds(["i1"], [0], [0.9], [[0, 0, 100, 50]]), gt, 1, 0.25)["summary"]["TP"] == 1
    assert evaluate(make_preds(["i1"], [0], [0.9], [[0, 0, 100, 49]]), gt, 1, 0.25)["summary"]["TP"] == 0
    # confidence threshold removes the low-score detection; per-class thresholds are honoured
    r = evaluate(pred, gt, 1, 0.85)["summary"]
    assert (r["TP"], r["FP"]) == (1, 0), r
    # a missed image and a prediction on an image without ground truth
    pred2 = make_preds(["i9"], [0], [0.9], [[0, 0, 10, 10]])
    r = evaluate(pred2, gt, 1, 0.25)["summary"]
    assert (r["TP"], r["FP"], r["FN"]) == (0, 1, 1), r
    r = evaluate(empty_preds(), gt, 1, 0.25)["summary"]
    assert (r["TP"], r["FP"], r["FN"], r["F1"]) == (0, 0, 1, 0.0), r


def test_matching_independent_of_lower_scores():
    rng = np.random.RandomState(0)
    gtb = np.array([[x, y, x + 40, y + 40] for x in range(0, 200, 50) for y in range(0, 200, 50)], float)
    gt = gt_table([("i", 0, *b) for b in gtb])
    pb = np.repeat(gtb, 3, axis=0) + rng.randint(-12, 12, (48, 4))
    pred = make_preds(["i"] * 48, [0] * 48, rng.rand(48), pb)
    tp_all, _ = match_predictions(pred, gt)
    for t in (0.2, 0.5, 0.8):
        sub = pred[pred.confidence >= t]
        tp_sub, _ = match_predictions(sub, gt)
        assert (tp_all[(pred.confidence >= t).values] == tp_sub).all()
    # threshold search agrees with direct evaluation
    thr, f1, _ = best_global_threshold(pred, gt)
    assert abs(evaluate(pred, gt, 1, thr)["summary"]["F1"] - f1) < 1e-9
    assert abs(micro_f1(pred, gt, 1, thr) - f1) < 1e-9
    ct = best_class_thresholds(pred, gt, 1)
    assert micro_f1(pred, gt, 1, ct) >= f1 - 1e-9
    assert 0 <= crossfit_f1(pd.concat([pred, pred.assign(image_id="j")]), pd.concat([gt, gt.assign(image_id="j")]), 1, "class") <= 1


def test_ap_values():
    # one TP at the top, one FP, one TP: AP = area under the precision envelope
    gt = gt_table([("i", 0, 0, 0, 10, 10), ("i", 0, 50, 50, 60, 60)])
    pred = make_preds(["i"] * 3, [0] * 3, [0.9, 0.8, 0.7], [[0, 0, 10, 10], [100, 100, 110, 110], [50, 50, 60, 60]])
    ap = evaluate(pred, gt, 1, 0.25)["per_class"].AP50[0]
    assert abs(ap - (0.5 * 1.0 + 0.5 * 2 / 3)) < 0.01, ap


def test_yolo_roundtrip():
    x = yolo_to_xyxy(0.5, 0.25, 0.2, 0.1, 1024, 512)
    assert np.allclose(x, (409.6, 102.4, 614.4, 153.6))
    assert np.allclose(xyxy_to_yolo(*x, 1024, 512), (0.5, 0.25, 0.2, 0.1))


def test_wbf_and_nms():
    a = make_preds(["i", "i"], [0, 1], [0.9, 0.6], [[10, 10, 50, 50], [100, 100, 150, 150]])
    b = make_preds(["i"], [0], [0.7], [[12, 12, 52, 52]])
    f = fuse_predictions([a, b], [1, 1], "wbf", 0.55, "avg")
    assert len(f) == 2, f
    top = f[f.class_id == 0].iloc[0]
    assert abs(top.confidence - 0.8) < 1e-9 and 10 < top.x1 < 12 and 50 < top.x2 < 52, top
    lone = f[f.class_id == 1].iloc[0]
    assert abs(lone.confidence - 0.3) < 1e-9, "a box found by one of two models is down-weighted under 'avg'"
    f = fuse_predictions([a, b], [1, 1], "wbf", 0.55, "max")
    assert abs(f[f.class_id == 1].iloc[0].confidence - 0.6) < 1e-9 and abs(f[f.class_id == 0].iloc[0].confidence - 0.9) < 1e-9
    f = fuse_predictions([a, b], [1, 1], "nms", 0.5)
    assert len(f) == 2 and abs(f.confidence.max() - 0.9) < 1e-9
    # different classes never merge, weights change the ranking, skip_thr filters
    c = make_preds(["i"], [1], [0.7], [[10, 10, 50, 50]])
    assert len(fuse_predictions([a.iloc[:1], c], None, "wbf", 0.55)) == 2
    f = fuse_predictions([a.iloc[:1], b], [1, 3], "wbf", 0.55)
    assert f.iloc[0].x1 > 11, "the heavier model should dominate the fused coordinates"
    assert len(fuse_predictions([a, b], None, "wbf", 0.55, skip_thr=0.65)) == 1
    assert len(fuse_predictions([empty_preds(), empty_preds()])) == 0


def test_wbf_avg_is_diluted_by_low_score_duplicates():
    """Why model fusion must skip near-zero boxes: WBF-avg averages the scores in a cluster."""
    good_a = make_preds(["i"], [0], [0.9], [[10, 10, 110, 110]])
    good_b = make_preds(["i"], [0], [0.8], [[12, 12, 112, 112]])
    junk = make_preds(["i"] * 6, [0] * 6, [0.004] * 6, [[10 + k, 10 + k, 110 + k, 110 + k] for k in range(6)])
    clean = fuse_predictions([good_a, good_b], [1, 1], "wbf", 0.5, "avg")
    assert len(clean) == 1 and abs(clean.confidence[0] - 0.85) < 1e-9
    diluted = fuse_predictions([good_a, pd.concat([good_b, junk])], [1, 1], "wbf", 0.5, "avg")
    assert len(diluted) == 1 and diluted.confidence[0] < 0.3, diluted
    restored = fuse_predictions([good_a, pd.concat([good_b, junk])], [1, 1], "wbf", 0.5, "avg", skip_thr=0.05)
    assert abs(restored.confidence[0] - 0.85) < 1e-9


def test_tiles_cover_image():
    for L, t in ((1024, 640), (1000, 640), (640, 640), (2000, 640), (256, 160)):
        s = tile_starts(L, t)
        assert s[0] == 0 and s[-1] + min(t, L) == L, (L, t, s)
        for a, b in zip(s, s[1:]):
            assert a + t - b >= 0.2 * t - 1, f"overlap below 20%: {L}, {t}, {s}"
    assert len(make_tiles(1024, 1024, 640)) == 4
    assert make_tiles(600, 400, 640) == [(0, 0, 600, 400)]


def _tile_detections(obj, tiles, W, H, score=0.8):
    """What an ideal detector returns for one object when run on each tile: the visible part of the box."""
    rows = []
    for tid, (tx1, ty1, tx2, ty2) in enumerate(tiles):
        x1, y1, x2, y2 = max(obj[0], tx1), max(obj[1], ty1), min(obj[2], tx2), min(obj[3], ty2)
        if x2 - x1 > 4 and y2 - y1 > 4:
            rows.append((tid, [x1, y1, x2, y2]))
    d = make_preds(["i"] * len(rows), [0] * len(rows), [score - 0.01 * k for k in range(len(rows))], [r[1] for r in rows])
    d["tile_id"] = [r[0] for r in rows]
    d["trunc"] = [bool(truncated_flags([r[1]], tiles[r[0]], W, H)[0]) for r in rows]
    return d


def test_tile_boundary_objects_counted_once():
    W = H = 1024
    tiles = make_tiles(W, H, 640)
    objects = {
        "crosses vertical seam": [200, 100, 800, 160],
        "crosses both seams": [300, 300, 760, 760],
        "full-width crack": [0, 500, 1024, 540],
        "inside the overlap strip": [450, 100, 560, 200],
        "inside one tile only": [50, 50, 200, 120],
        "touches the image border": [0, 0, 120, 90],
    }
    for name, obj in objects.items():
        td = _tile_detections(obj, tiles, W, H)
        # (a) the full-image pass missed the object: tiles alone must still yield exactly one box
        m = merge_full_and_tiles(empty_preds(), td)
        assert len(m) == 1, f"{name}: {len(m)} boxes without a full-view detection\n{m}"
        assert box_iou(m[BOX].values, [obj])[0, 0] > 0.9, f"{name}: stitched box is off: {m[BOX].values}"
        # (b) the full-image pass found it too
        full = make_preds(["i"], [0], [0.7], [obj])
        m = merge_full_and_tiles(full, td)
        assert len(m) == 1, f"{name}: {len(m)} boxes with a full-view detection\n{m}"
        assert box_iou(m[BOX].values, [obj])[0, 0] > 0.9
        gt = gt_table([("i", 0, *obj)])
        r = evaluate(m, gt, 1, 0.25)["summary"]
        assert (r["TP"], r["FP"], r["FN"]) == (1, 0, 0), (name, r)
    # two genuinely different neighbouring objects in different tiles stay separate
    td = pd.concat([_tile_detections([100, 100, 180, 180], tiles, W, H), _tile_detections([850, 100, 930, 180], tiles, W, H)])
    assert len(merge_full_and_tiles(empty_preds(), td)) == 2
    # truncation flags: only interior tile edges count
    t = tiles[0]
    assert list(truncated_flags([[0, 0, 100, 100], [500, 10, 640, 60], [10, 500, 60, 640]], t, W, H)) == [False, True, True]


def test_complementarity_and_sizes():
    gt = gt_table([("i", 0, 0, 0, 10, 10), ("i", 0, 0, 0, 50, 50), ("i", 1, 0, 0, 200, 200), ("j", 1, 0, 0, 20, 20), ("j", 2, 5, 5, 300, 300)])
    hit_a = np.array([True, True, False, False, True]); hit_b = np.array([True, False, True, False, False])
    c = complementarity(gt, hit_a, hit_b, NAMES)
    assert (c.both + c.A_only + c.B_only + c.neither == c.n_gt).all()
    allr = c.iloc[0]
    assert (allr.both, allr.A_only, allr.B_only, allr.neither) == (1, 2, 1, 1) and abs(allr.oracle_recall - 0.8) < 1e-9
    assert c[c.group.isin(NAMES)].n_gt.sum() == len(gt) and c[c.group.isin(["small", "medium", "large"])].n_gt.sum() == len(gt)
    assert list(size_bucket(gt)) == ["small", "medium", "large", "small", "large"]
    rs = recall_by_size(gt, hit_a).set_index("size").recall
    assert abs(rs["small"] - 0.5) < 1e-9 and rs["medium"] == 1.0 and abs(rs["large"] - 0.5) < 1e-9
    # gt_hits: derived from an evaluate() result at its operating point
    pred = make_preds(["i", "i", "j"], [0, 1, 2], [0.9, 0.2, 0.9], [[0, 0, 50, 50], [0, 0, 200, 200], [5, 5, 300, 300]])
    res = evaluate(pred, gt, 3, 0.5, NAMES)
    assert list(gt_hits(res, len(gt))) == [False, True, False, False, True]
    fp = shared_false_positives(pred, evaluate(pred, gt.iloc[:0], 3, 0.1), pred, evaluate(pred, gt.iloc[:0], 3, 0.1))
    assert fp["fp_A"] == 3 and fp["A_shared_frac"] == 1.0


def test_confusion_matrix():
    gt = gt_table([("i", 0, 0, 0, 100, 100), ("i", 1, 200, 200, 300, 300), ("i", 2, 400, 400, 500, 500)])
    pred = make_preds(["i"] * 3, [0, 2, 1], [0.9, 0.9, 0.9], [[0, 0, 100, 100], [200, 200, 300, 300], [700, 700, 800, 800]])
    M = detection_confusion(pred, gt, 3, 0.25)
    assert M[0, 0] == 1 and M[1, 2] == 1 and M[3, 1] == 1 and M[2, 3] == 1 and M.sum() == 4, M


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print("ok  ", t.__name__)
    print(f"{len(tests)} test groups passed")
