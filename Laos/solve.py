"""Unsupervised shell inspection. No previous submission is used as training labels.

Run: python tasks/laos/solve.py --images DIR --output DIR --reference FILE
Optional --history DIR produces set comparisons without influencing selection.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

VERSION = 'laos-normalized-multisignal-v2'
H, W = 192, 80


def normalize(path):
    original = Image.open(path)
    meta = {'format': original.format, 'size': list(original.size), 'mode': original.mode,
            'render_date': original.info.get('Date', '')}
    rgb = np.asarray(original.convert('RGB').resize((256, 256)), dtype=np.float32) / 255
    r, g, b = rgb.transpose(2, 0, 1)
    green = (g-r > 8/255) & (g-b > 12/255)
    green[:, :72] = False
    green[:, 184:] = False
    labels, n = ndi.label(green)
    components = []
    for k, sl in enumerate(ndi.find_objects(labels), 1):
        if sl is None:
            continue
        yy, xx = np.nonzero(labels[sl] == k)
        xx = xx + sl[1].start
        yy = yy + sl[0].start
        if len(xx) >= 100 and 96 < xx.mean() < 164:
            components.append(k)
    mask = np.isin(labels, components)
    if mask.sum() < 800:
        raise ValueError(f'{path.name}: insufficient foreground ({mask.sum()})')
    mask = ndi.binary_fill_holes(ndi.binary_closing(mask, iterations=2))
    yy, xx = np.nonzero(mask)
    xy = np.column_stack((xx, yy))
    center = np.median(xy, axis=0)
    cov = np.cov(xy.T)
    axis = np.linalg.eigh(cov)[1][:, -1]
    if axis[1] < 0:
        axis = -axis
    side = np.array([axis[1], -axis[0]])
    proj = (xy-center) @ axis
    lo, hi = np.percentile(proj, [0.2, 99.8])
    center += axis * (lo+hi)/2
    height = hi-lo
    # Direct inverse mapping avoids ambiguous image-rotation signs.
    v, u = np.mgrid[:H, :W].astype(float)
    v = (v-(H-1)/2) * height/(H-12)
    u = (u-(W-1)/2) * height/(H-12)
    source_x = center[0] + v*axis[0] + u*side[0]
    source_y = center[1] + v*axis[1] + u*side[1]
    coords = np.array([source_y, source_x])
    aligned = np.stack([ndi.map_coordinates(rgb[:, :, c], coords, order=1, mode='nearest') for c in range(3)], -1)
    aligned_mask = ndi.map_coordinates(mask.astype(float), coords, order=0) > .5
    # Fill the tape gap for interior inspection; preserve silhouette separately.
    hull_rows = aligned_mask.copy()
    for y in range(H):
        xs = np.flatnonzero(aligned_mask[y])
        if len(xs):
            hull_rows[y, xs.min():xs.max()+1] = True
    hull_rows = ndi.binary_fill_holes(ndi.binary_closing(hull_rows, iterations=5))
    interior = ndi.binary_erosion(hull_rows, iterations=3)
    meta.update(angle_deg=float(np.degrees(np.arctan2(axis[0], axis[1]))),
                foreground_pixels=int(mask.sum()), source_height=float(height))
    return aligned, aligned_mask, interior, meta


def component_score(mask):
    lab, n = ndi.label(mask)
    sizes = np.bincount(lab.ravel())[1:]
    return float(sizes.max()) if n else 0.0


def features(rgb, silhouette, interior):
    r, g, b = rgb.transpose(2, 0, 1)
    lum = .299*r + .587*g + .114*b
    smooth = ndi.gaussian_filter(lum, 2)
    high = lum-smooth
    # Tape is found from colour, with its search range relative to the object.
    tape = (abs(r-g) < .04) & (g-b < .09) & (r > .28) & interior
    tape[:int(H*.66)] = False
    tape[int(H*.9):] = False
    tape_y = np.flatnonzero(tape.sum(axis=1) > 8)
    band_y = float(np.median(tape_y)) if len(tape_y) else H*.79
    body = interior.copy()
    body[max(0, int(band_y)-4):min(H, int(band_y)+8)] = False
    body[:12] = False
    body[-12:] = False
    rust_map = np.maximum((r-g) + .045, 0) * np.maximum(g-b, 0)
    rust = component_score((r > .96*g) & (g-b > .045) & body)
    dark = component_score((high < -.035) & body)
    texture = float(np.mean(np.sort(abs(high[body]))[-max(5, int(body.sum()*.01)):] ))
    band_points = []
    for x in range(W//2-15, W//2+16):
        ys = np.flatnonzero(tape[:, x])
        if len(ys) > 2:
            band_points.append((x, np.median(ys)))
    band = 0.0
    if len(band_points) >= 20:
        x, y = np.array(band_points).T
        residual = y-np.polyval(np.polyfit(x, y, 2), x)
        band = float(np.percentile(abs(residual), 95))
    else:
        band = 1.0
    left, right = [], []
    for y in range(30, max(50, int(band_y)-8)):
        xs = np.flatnonzero(silhouette[y])
        if len(xs) > 4:
            left.append(xs.min()); right.append(xs.max())
        else:
            left.append(W/2); right.append(W/2)
    left, right = ndi.median_filter(left, 5), ndi.median_filter(right, 5)
    widths = np.asarray(right)-left
    contour = float(np.percentile(abs(widths-ndi.gaussian_filter1d(widths.astype(float), 3)), 95))
    # Background is replaced before constructing structural descriptors.
    surface = np.stack((lum, r-g, g-b), -1)
    surface[~interior] = 0
    small = surface[::4, ::2].copy()
    # High-pass luminance reduces nuisance illumination changes.
    small[:, :, 0] -= ndi.gaussian_filter(small[:, :, 0], 2)
    vector = small.ravel()
    # Local descriptors are compared only against the same normalized location.
    patch = np.stack((high, r-g, g-b, abs(ndi.laplace(lum))), -1)
    patch = np.stack([ndi.uniform_filter(patch[:, :, c], 5) for c in range(4)], -1)
    patch[~body] = 0
    patch = patch[::6, ::4]
    return dict(rust=rust, dark=dark, texture=texture, band=band, contour=contour), vector, patch, rust_map, high


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def percentile(values):
    # Equal values receive equal ranks, independent of filenames.
    values = np.asarray(values)
    return np.array([(np.count_nonzero(values < v)+.5*np.count_nonzero(values == v))/len(values) for v in values])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--history', type=Path)
    parser.add_argument('--no-render-prior', action='store_true',
                        help='ablation: rank exclusively from visual signals')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    diagnostics = args.output/'diagnostics'; diagnostics.mkdir(exist_ok=True)
    files = sorted(args.images.glob('*.png'))
    rows, vectors, patches, hashes, image_meta = [], [], [], {}, []
    for i, f in enumerate(files):
        aligned, mask, interior, meta = normalize(f)
        feat, vector, patch, rust_map, high = features(aligned, mask, interior)
        rows.append(dict(name=f.name, **feat)); vectors.append(vector); patches.append(patch)
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        hashes.setdefault(digest, []).append(f.name)
        image_meta.append(dict(name=f.name, sha256=digest, **meta))
        diagnostic = np.concatenate((aligned, np.repeat(interior[:, :, None], 3, axis=2),
                                     np.stack((np.clip(rust_map*70, 0, 1), np.clip(-high*8, 0, 1), np.zeros_like(high)), -1)), axis=1)
        Image.fromarray(np.uint8(np.clip(diagnostic, 0, 1)*255)).save(diagnostics/(f.stem+'.png'))
        if i % 200 == 0: print(f'normalized {i}/{len(files)}', flush=True)
    X = np.asarray(vectors, dtype=np.float32)
    pca = PCA(n_components=min(24, len(X)-1), svd_solver='randomized', random_state=17)
    z = pca.fit_transform(X)
    reconstruction = pca.inverse_transform(z)
    structural = np.mean(np.sort(abs(X-reconstruction), axis=1)[:, -120:], axis=1)
    # Per-location leave-one-out kNN; no candidate pseudo-labels are used.
    P = np.asarray(patches, dtype=np.float32).reshape(len(files), -1, 4)
    patch_scores = np.zeros(P.shape[:2])
    for j in range(P.shape[1]):
        q = P[:, j]
        scale = np.maximum(np.std(q, axis=0), [.006, .012, .012, .006])
        distances, _ = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(q/scale).kneighbors(q/scale)
        patch_scores[:, j] = distances[:, 1:4].mean(axis=1)
    local = np.mean(np.sort(patch_scores, axis=1)[:, -5:], axis=1)
    signals = ['rust', 'dark', 'texture', 'band', 'contour', 'structural', 'local']
    for i, row in enumerate(rows):
        row.update(structural=float(structural[i]), local=float(local[i]))
    ranks = np.column_stack([percentile([r[s] for r in rows]) for s in signals])
    # Average the strongest three signals. They can be correlated; this is
    # an unsupervised ranking heuristic, not a calibrated probability.
    combined = np.mean(np.sort(ranks, axis=1)[:, -3:], axis=1)
    # A small rendering session is an auxiliary provenance signal, never a label.
    # Sessions are inferred from all PNG dates; no previous candidate IDs enter it.
    sessions = []
    for i in sorted(range(len(image_meta)), key=lambda k: image_meta[k]['render_date']):
        try:
            date = datetime.strptime(image_meta[i]['render_date'], '%Y/%m/%d %H:%M:%S')
        except ValueError:
            continue
        if not sessions or (date-sessions[-1][-1][0]).total_seconds() > 45:
            sessions.append([])
        sessions[-1].append((date, i))
    batch_sizes = np.full(len(files), len(files), dtype=int)
    for session in sessions:
        for date, i in session:
            batch_sizes[i] = len(session)
    for i, row in enumerate(rows):
        row['visual_score'] = float(combined[i])
        row['render_batch_size'] = int(batch_sizes[i])
        sparse = batch_sizes[i] <= max(40, .05*len(files))
        row['provenance_bonus'] = .1 if sparse and not args.no_render_prior else 0.0
        row['score'] = float(combined[i]) + row['provenance_bonus']
        row['reason'] = '; '.join(f'{signals[k]} percentile={ranks[i,k]:.3f}' for k in np.argsort(ranks[i])[::-1][:3])
    for row in rows:
        if row['provenance_bonus']:
            row['reason'] += f"; sparse render session n={row['render_batch_size']} (auxiliary)"
    ranking = sorted(rows, key=lambda r: (-r['score'], r['name']))
    selected = ranking[:28]
    entries = ['images/'+r['name'] for r in selected]
    assert len(set(entries)) == 28 and all((args.images/Path(e).name).is_file() for e in entries)
    reference = args.reference.read_text(encoding='utf-8-sig').splitlines()
    assert reference and all(e.startswith('images/') and e.endswith('.png') for e in reference)
    content = ''.join(e+'\n' for e in sorted(entries))
    (args.output/'test-output.txt').write_text(content, encoding='utf-8')
    write_csv(args.output/'ranking.csv', ranking); write_csv(args.output/'candidates.csv', selected)
    write_csv(args.output/'image_metadata.csv', image_meta)
    audit = dict(version=VERSION, render_prior_enabled=not args.no_render_prior,
                 render_session_sizes=[len(x) for x in sessions], file_count=len(files), unique_sha256=len(hashes),
                 duplicate_content_groups=[v for v in hashes.values() if len(v)>1],
                 sizes=sorted({str(m['size']) for m in image_meta}),
                 formats=sorted({m['format'] for m in image_meta}),
                 late_count=sum(int(f.stem.split('_')[-1])>1125 for f in files),
                 late_selected=[r['name'] for r in selected if int(r['name'][6:10])>1125],
                 submission_sha256=hashlib.sha256((args.output/'test-output.txt').read_bytes()).hexdigest())
    if args.history:
        comparison = {}
        for old in sorted(args.history.glob('attempt-*-19-of-28.txt')):
            prior = set(old.read_text(encoding='utf-8-sig').splitlines())
            comparison[old.name] = dict(overlap=len(set(entries)&prior), removed=sorted(prior-set(entries)), added=sorted(set(entries)-prior))
        audit['comparison'] = comparison
    (args.output/'audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    sheet = Image.new('RGB', (1200, 8*220), 'white'); draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(ranking[:48]):
        im = Image.open(diagnostics/(Path(row['name']).stem+'.png')).crop((0, 0, W, H)).resize((80, 192))
        x=(i%6)*200; y=(i//6)*220; sheet.paste(im, (x+45,y))
        draw.text((x,y+193), f"{i+1} {row['name']} {row['score']:.3f}", fill='black')
    sheet.save(args.output/'contact_sheet.png')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
