# Kazakhstan — Stamp of Disapproval

Use JPEG-history features to localize suspect patches and fit the pooled album coordinates against glyph templates.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python jpeg_forensics.py exhibits --data /path/to/materials --output outputs/jpeg
python jpeg_forensics.py albums --data /path/to/materials --output outputs/jpeg
python jpeg_candidates.py --data /path/to/materials --jpeg outputs/jpeg --output outputs/candidates
python read_letters.py --rescored outputs/candidates/candidates.csv --output outputs/letters --controls 20 --keep 12
```

## Input and output

exhibits/ and albums/album_1/ through album_5/. A fonts directory can be supplied with --fonts.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

Organizer material: [source 1](https://drive.google.com/drive/folders/1VKPTZUZf9sUxQrBkpr4kuDpcAXRfKZoC?usp=sharing).

## Files

- [`geometry.py`](geometry.py)
- [`jpeg_candidates.py`](jpeg_candidates.py)
- [`jpeg_forensics.py`](jpeg_forensics.py)
- [`noise_forensics.py`](noise_forensics.py)
- [`read_letters.py`](read_letters.py)
- [`requirements.txt`](requirements.txt)

Keep submission values on the organizer platform. No final submission value is supplied in these files.
