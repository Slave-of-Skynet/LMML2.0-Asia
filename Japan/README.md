# Japan — City of Subtitles

Estimate orientation-aware sensor fingerprints, attribute anonymous photos and recover caption-cell patterns from the target contributor.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python solve_japan.py --dataset /path/to/materials --output outputs/attribution
python secret/decode_secret.py --dataset /path/to/materials --attribution outputs/attribution/japan_attribution.csv --out outputs/decoding
```

## Input and output

The 253 organizer JPEGs, preserving the credited/anonymous directory structure.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

Organizer material: [source 1](https://drive.google.com/drive/folders/1dKvKxO5BTcpjR9QSumq5pAfeRYXYYS92).

## Files

- [`requirements.txt`](requirements.txt)
- [`secret/cellclf.py`](secret/cellclf.py)
- [`secret/decode_secret.py`](secret/decode_secret.py)
- [`secret/feat.py`](secret/feat.py)
- [`secret/font_decode.py`](secret/font_decode.py)
- [`secret/mask2.py`](secret/mask2.py)
- [`secret/subcells.py`](secret/subcells.py)
- [`solve_japan.py`](solve_japan.py)

Keep submission values on the organizer platform. No final submission value is supplied in these files.
