# Mongolia — The Baa-rilliant task!

Predict sheep density with a VGG/CSRNet counter, reconstruct the rotated survey grid and correlate 5x7 glyph templates.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python predict_v2.py --data /path/to/materials --ckpt fold0.pt --out outputs/counts.csv
python grid_decode.py --counts outputs/counts.csv --plots /path/to/materials/survey_plots.csv --out outputs/grid
python glyph_match.py --matrix outputs/grid/grid_matrix.csv
```

## Input and output

train/, train_counts.csv, train_points.csv, survey/, survey_plots.csv, and the fold0.pt release asset.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

## Files

- [`counts.csv`](counts.csv)
- [`glyph_match.py`](glyph_match.py)
- [`grid_decode.py`](grid_decode.py)
- [`predict_v2.py`](predict_v2.py)
- [`requirements.txt`](requirements.txt)
- [`train_v2.py`](train_v2.py)

Keep submission values on the organizer platform. No final submission value is supplied in these files.

## Checkpoint and lightweight replay

Download [fold0.pt](https://github.com/Slave-of-Skynet/LMML2.0-Asia/releases/download/jury-models-20261115/fold0.pt)
beside this README. SHA-256: `c5d84c331d4f3b55febe3ffa29d065c88dd7c378bb753e86c2e86a9742485e23`.
The packaged `counts.csv` is the verified v2 prediction table, not the earlier baseline.
If the checkpoint or full dataset is unavailable, replay the geometry/decoder using:

```bash
python grid_decode.py --counts counts.csv --plots survey_plots.csv --out outputs/grid
python glyph_match.py --matrix outputs/grid/grid_matrix.csv
```

Retrain with `python train_v2.py --data /path/to/materials --out outputs/training --fold 0 --epochs 30`.
The pretrained VGG initialization is downloaded through torchvision on the first training run.

Bundled geometry: [`survey_plots.csv`](survey_plots.csv).
