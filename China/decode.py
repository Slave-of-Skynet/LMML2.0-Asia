"""Turn the stored detections into counts.csv and the flag.

Raw detections above THR are corrected with numbers measured on 300 held-out labelled rubbings:
count = (found - 30 * false_hits_per_rubbing) / recall. A set is a 1 bit when its glyph is frequent,
a 0 bit when it is (almost) absent; five bits, first set = highest bit, give the letter (A = 1).
"""
import json
import pandas as pd

THR, BIT = 0.9, 8          # score threshold; a set with more than BIT glyphs is a 1 bit
d = json.load(open("detections.json"))
GLYPHS = ["g015", "g028", "g005", "g018", "g003", "g011", "g001", "g010"]   # crates.csv order, crate_01..08

val = d["val"].values()
rows, word = [], ""
for c, g in enumerate(GLYPHS):
    tp = sum(1 for v in val for s, k, hit in v["dets"] if k == c and s >= THR and hit)
    fp = sum(1 for v in val for s, k, hit in v["dets"] if k == c and s >= THR and not hit)
    recall, fp_rate = tp / sum(v["gt"].count(c) for v in val), fp / len(val)
    bits = ""
    for s in range(1, 6):
        key = f"crate_{c + 1:02d}/set_{s}/"
        found = sum(1 for name, dets in d["crates"].items() if name.startswith(key) for sc, k in dets if k == c and sc >= THR)
        count = max(0, round((found - 30 * fp_rate) / recall))
        rows.append([f"crate_{c + 1:02d}", f"set_{s}", g, count]); bits += "1" if count > BIT else "0"
    letter = chr(64 + int(bits, 2)); word += letter
    print(f"crate_{c + 1:02d} {g} recall {recall:.2f} fp/rubbing {fp_rate:.3f} counts {[r[3] for r in rows[-5:]]} bits {bits} -> {letter}")
pd.DataFrame(rows, columns=["crate", "set", "glyph_id", "count"]).to_csv("counts.csv", index=False)
assert len(word) == len(GLYPHS)
print("Decoded word:", word)
