# Vietnam — That isn't my shoe!

Fine-tune a compact ConvNeXt candidate and calibrate an explicit unknown-class confidence threshold on held-out known-brand examples.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python train_convnext.py --data /path/to/materials --out outputs --mode holdout --device cpu
# For a final candidate, use --mode final; inspect its size and metrics before upload.
```

## Input and output

train-input/ and train-output.csv. No unknown-brand images are required for training.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

This is a candidate, not an accepted checkpoint. Earlier submitted candidates were rejected; this packaging does not establish that accuracy, unknown rejection, loader compatibility or upload size pass the official gates.

## Files

- [`requirements.txt`](requirements.txt)
- [`train_convnext.py`](train_convnext.py)

Keep submission values on the organizer platform. No final submission value is supplied in these files.
