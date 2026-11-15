"""Scan the held-out rubbings and all crate rubbings once and store every detection (score >= 0.3).
Thresholds and count corrections are then chosen from the stored detections without rescanning."""
import json, sys, time, torch
from pathlib import Path
from window_clf import Net, scan, gray, load
D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../task_china"); torch.set_num_threads(8)
labels, crates, classes, names, by = load(D)
ck = torch.load("window_clf.pt"); net = Net(); net.load_state_dict(ck["state"]); net.eval()
t0 = time.time(); out = {"val": {}, "crates": {}}
with torch.no_grad():
    for crate in crates.crate:
        for sd in sorted((D / "crates" / crate).iterdir()):
            for f in sorted(sd.glob("*.jpg")):
                out["crates"][f"{crate}/{sd.name}/{f.name}"] = [[round(s, 4), c] for s, c, cx, cy, side in scan(net, gray(f), thr=0.3)]
        json.dump(out, open("detections.json", "w")); print(crate, "done", round((time.time() - t0) / 60, 1), "min", flush=True)
    for n in names[:300]:
        dets = scan(net, gray(D / "train" / "images" / n), thr=0.3); gt = by[n]; rec = []
        used = set()
        for s, c, cx, cy, side in dets:
            g = [k for k, v in classes.items() if v == c][0]
            hit = [i for i, (x, y, w, h, gg) in enumerate(gt) if gg == g and i not in used and abs(cx - x - w / 2) < 0.5 * max(w, h) and abs(cy - y - h / 2) < 0.5 * max(w, h)]
            if hit: used.add(hit[0])
            rec.append([round(s, 4), c, 1 if hit else 0])
        out["val"][n] = {"dets": rec, "gt": [classes[b[4]] for b in gt if b[4] in classes]}
json.dump(out, open("detections.json", "w")); print("all done", round((time.time() - t0) / 60, 1), "min", flush=True)
