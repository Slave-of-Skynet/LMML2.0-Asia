"""Platform-compatible Vietnam model: torchvision convnext_tiny state_dict + tau.

Platform evidence (task-card tracebacks, 2026-11-15):
  main.py:71  ckpt = torch.load(args.model, map_location=device)     # weights_only=True
  main.py:73  tau = float(ckpt["tau"])
  later       strict load_state_dict into a torchvision-style ConvNeXt
              (features.0.0.weight [96,3,4,4], features.0.1 LayerNorm2d(96)).
Hints: explicit rejection by a threshold on max softmax probability (or prototype
similarity); calibrate it on a held-out part of the known TRAINING images only;
fine-tune the network.

A full convnext_tiny (27.8M params) cannot fit 10 MB. This keeps the stem, stages
1-2, all downsampling layers and the first --keep3 blocks of stage 3 trainable.
All other CNBlocks are exact identities (layer_scale = 0, weights = 0) and their
zeros compress to almost nothing in a DEFLATE-compressed checkpoint zip, which
torch.load reads natively.

The classifier LayerNorm2d is fixed (weight 1, bias 0), so its output norm is
sqrt(768); Linear weight = s * normalize(P) / sqrt(768) gives logits = s * cosine
to learned class prototypes. Training uses an additive cosine margin.
Only official known-brand images are used.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
import zipfile
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms as T

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SIZE, SCALE, MARGIN = 224, 16.0, 0.2
train_tf = T.Compose([T.RandomResizedCrop(SIZE, scale=(0.6, 1.0)), T.RandomHorizontalFlip(),
                      T.RandomRotation(10), T.ColorJitter(0.3, 0.3, 0.2, 0.02),
                      T.ToTensor(), T.Normalize(MEAN, STD)])
eval_tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def split(labels, classes, seed=0):
    # Identical to finish.py: per class 49 test, 25 calibration, rest train.
    rng = np.random.RandomState(seed); tr, ca, te = [], [], []
    for c in classes:
        ix = np.where(labels == c)[0]; rng.shuffle(ix)
        tr.extend(ix[74:]); ca.extend(ix[49:74]); te.extend(ix[:49])
    return np.array(tr), np.array(ca), np.array(te)


def identity_blocks(net, keep3):
    """CNBlocks turned into exact identities: stage 3 beyond keep3, all of stage 4."""
    return list(net.features[5])[keep3:] + list(net.features[7])


class Trainer(nn.Module):
    def __init__(self, n_classes, keep3, pretrained=True):
        super().__init__()
        w = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = models.convnext_tiny(weights=w)
        self.net.classifier[2] = nn.Linear(768, n_classes)
        with torch.no_grad():
            for blk in identity_blocks(self.net, keep3):
                for p in blk.parameters():
                    p.zero_(); p.requires_grad_(False)
            ln = self.net.classifier[0]
            ln.weight.fill_(1.0); ln.bias.zero_()
            ln.weight.requires_grad_(False); ln.bias.requires_grad_(False)
            self.net.classifier[2].weight.requires_grad_(False)
            self.net.classifier[2].bias.requires_grad_(False)
        self.protos = nn.Parameter(torch.randn(n_classes, 768) * 0.01)

    def cos(self, x):
        n = self.net
        z = n.classifier[1](n.classifier[0](n.avgpool(n.features(x))))
        return F.normalize(z, dim=1) @ F.normalize(self.protos, dim=1).t()

    def export_state_dict(self):
        with torch.no_grad():
            self.net.classifier[2].weight.copy_(SCALE * F.normalize(self.protos, dim=1) / math.sqrt(768))
            self.net.classifier[2].bias.zero_()
        return {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}


def fit(imgs, ys, n_classes, keep3, epochs, device, seed, batch):
    seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    model = Trainer(n_classes, keep3).to(device)
    back = [p for p in model.net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{'params': back, 'lr': 1e-4}, {'params': [model.protos], 'lr': 3e-3}],
                            weight_decay=0.05)
    bs = batch + 1 if len(imgs) % batch == 1 else batch
    steps = epochs * ((len(imgs) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 3e-3], total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
    yt = torch.tensor(ys, device=device)
    for ep in range(epochs):
        tick = time.perf_counter(); model.train(); total = 0.0
        order = torch.randperm(len(imgs))
        for s in range(0, len(imgs), bs):
            ix = order[s:s + bs]
            x = torch.stack([train_tf(imgs[i]) for i in ix.tolist()]).to(device)
            y = yt[ix.to(device)]
            with torch.autocast(device_type='cuda', enabled=device == 'cuda'):
                cos = model.cos(x)
            logits = SCALE * (cos.float() - MARGIN * F.one_hot(y, n_classes))
            loss = F.cross_entropy(logits, y, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            old = scaler.get_scale(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if scaler.get_scale() >= old: sched.step()
            total += float(loss.detach()) * len(ix)
        print(json.dumps({'epoch': ep + 1, 'loss': round(total / len(imgs), 4),
                          'seconds': round(time.perf_counter() - tick, 1)}), flush=True)
    model.eval().cpu()
    return model


def plain_net(sd, n_classes):
    net = models.convnext_tiny(weights=None, num_classes=n_classes)
    net.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    return net.eval()


@torch.inference_mode()
def softmax_probs(net, imgs, device, size=SIZE):
    tf = eval_tf if size == SIZE else T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])
    net.to(device); out = []
    for i in range(0, len(imgs), 32):
        out.append(F.softmax(net(torch.stack([tf(im) for im in imgs[i:i + 32]]).to(device)).float(), 1).cpu())
    net.cpu()
    return torch.cat(out).numpy()


def report(p, y, tau, n):
    pred = np.where(p.max(1) >= tau, p.argmax(1), n)
    m = np.zeros((n, n + 1), int)
    for a, b in zip(y, pred): m[a, b] += 1
    return {'n': int(len(y)), 'correct': int((pred == y).sum()), 'known_accuracy': float((pred == y).mean()),
            'false_unknown': int((pred == n).sum()), 'closed_accuracy': float((p.argmax(1) == y).mean()),
            'confusion': m.tolist()}


def pick_tau(p, y, target):
    conf, ok = p.max(1), p.argmax(1) == y
    feasible = [t for t in np.unique(conf) if (ok & (conf >= t)).mean() >= target]
    return float(max(feasible)) if feasible else float(conf.min())


def truncate_fp16(t, keep_bits):
    """Zero the low (10 - keep_bits) mantissa bits of an fp16 tensor (round to nearest)."""
    if keep_bits >= 10: return t
    drop = 10 - keep_bits
    i = t.view(torch.int16).to(torch.int32)
    i = ((i + (1 << (drop - 1))) >> drop) << drop
    return i.to(torch.int16).view(torch.float16)


def save_deflated(obj, path):
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    with zipfile.ZipFile(tmp) as zi, zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zo:
        for info in zi.infolist():
            zo.writestr(info.filename, zi.read(info.filename))
    tmp.unlink()


def build_submission(sd, classes, tau, keep_bits):
    half = {k: truncate_fp16(v.half(), keep_bits) if v.is_floating_point() else v for k, v in sd.items()}
    return {'tau': float(tau), 'classes': list(classes), 'num_classes': len(classes),
            'arch': 'convnext_tiny', 'state_dict': half, 'model_state_dict': half,
            'img_size': SIZE, 'input_size': SIZE, 'mean': MEAN, 'std': STD,
            'rule': 'softmax over classes; unknown if max softmax prob < tau'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--mode', choices=['holdout', 'lobo', 'final'], default='holdout')
    ap.add_argument('--keep3', type=int, default=2); ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=24); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--target', type=float, default=0.88); ap.add_argument('--keep_bits', type=int, default=7)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--reuse', action='store_true')
    a = ap.parse_args(); torch.set_num_threads(4); a.out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((a.data / 'train-output.csv').open()))
    labels = np.array([r['label'] for r in rows]); classes = sorted(set(labels.tolist()))
    imgs = [Image.open(a.data / 'train-input' / r['filename']).convert('RGB') for r in rows]
    print(json.dumps({'images': len(imgs), 'torch': torch.__version__, 'args': {k: str(v) for k, v in vars(a).items()}}), flush=True)
    for unknown in (classes if a.mode == 'lobo' else [None]):
        kc = [c for c in classes if c != unknown]; tag = unknown or a.mode
        tr, ca, te = split(labels, kc, a.seed)
        if a.mode == 'final':
            tr = np.concatenate([tr, te])  # calibration split stays held out for tau
        y = np.array([kc.index(c) if c in kc else -1 for c in labels])
        ckpt_path = a.out / f'{tag}_sd.pt'
        if a.reuse and ckpt_path.exists():
            sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)['state_dict']
        else:
            model = fit([imgs[i] for i in tr], [int(y[i]) for i in tr], len(kc), a.keep3, a.epochs, a.device, a.seed, a.batch)
            sd = model.export_state_dict()
            torch.save({'state_dict': sd, 'classes': kc, 'keep3': a.keep3, 'train_indices': tr.tolist(),
                        'calibration_indices': ca.tolist(), 'test_indices': te.tolist()}, ckpt_path)
        # Evaluate exactly what is shipped: fp16 + mantissa-truncated weights in plain convnext_tiny.
        sub = build_submission(sd, kc, 0.0, a.keep_bits)
        net = plain_net(sub['state_dict'], len(kc))
        p = softmax_probs(net, imgs, a.device)
        tau = pick_tau(p[ca], y[ca], a.target)
        res = {'tag': tag, 'tau': tau, 'calibration': report(p[ca], y[ca], tau, len(kc))}
        if a.mode != 'final':
            res['test'] = report(p[te], y[te], tau, len(kc))
        if unknown is not None:
            u = np.where(y == -1)[0]
            res['unknown_n'] = int(len(u)); res['unknown_rejection'] = float((p[u].max(1) < tau).mean())
        p256 = softmax_probs(net, imgs, a.device, 256)
        res['test_256'] = report(p256[te], y[te], tau, len(kc)) if a.mode != 'final' else None
        np.savez_compressed(a.out / f'{tag}_probs.npz', p224=p, p256=p256, labels=labels)
        print('RESULT', json.dumps(res), flush=True)
        (a.out / f'{tag}_metrics.json').write_text(json.dumps(res, indent=2))
        if unknown is None:
            sub['tau'] = float(tau)
            out = a.out / f'vietnam-convnext-{tag}.pt'
            save_deflated(sub, out)
            back = torch.load(out, map_location='cpu', weights_only=True)
            chk = models.convnext_tiny(weights=None, num_classes=len(kc))
            chk.load_state_dict(back['state_dict'], strict=True)
            size = out.stat().st_size
            print(json.dumps({'file': str(out.resolve()), 'bytes': size, 'under_10e6': size <= 10_000_000,
                              'sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'tau': back['tau'],
                              'keys': sorted(back), 'strict_load_convnext_tiny': 'passed'}), flush=True)


if __name__ == '__main__':
    main()
