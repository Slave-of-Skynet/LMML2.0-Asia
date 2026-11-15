# China — Carved in bones, engraved in the heart

Count target glyphs using a sliding-window classifier, calibrate detection counts and decode the five-bit crate patterns.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python window_clf.py train --data /path/to/task_china --ckpt window_clf.pt
# Scan using the same data directory and newly trained checkpoint.
python scan_all.py /path/to/task_china
# Replay the bundled detection evidence without a checkpoint:
python decode.py
```

## Input and output

The complete train/crates directories and CSV labels. detections.json is retained for a lightweight decoder replay.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

## Files

- [`decode.py`](decode.py)
- [`detections.json`](detections.json)
- [`requirements.txt`](requirements.txt)
- [`scan_all.py`](scan_all.py)
- [`window_clf.py`](window_clf.py)

Keep submission values on the organizer platform. No final submission value is supplied in these files.
