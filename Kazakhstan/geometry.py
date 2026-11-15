"""Generic stamp segmentation; no page-specific coordinates or album labels."""
import cv2, numpy as np

def exhibit_box(im):
    d = np.abs(im.astype('int16') - np.array([38, 36, 34])).max(2)
    mask = cv2.morphologyEx((d > 30).astype('uint8'), cv2.MORPH_CLOSE, np.ones((9, 9), 'uint8'))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = [tuple(map(int, s[:4])) for s in stats[1:] if s[2] > 120 and s[3] > 120]
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    return (x + 6, y + 6, w - 12, h - 12)

def stamps(im):
    d = np.abs(im.astype('int16') - np.array([38, 36, 34])).max(2)
    m = cv2.morphologyEx((d > 30).astype('uint8'), cv2.MORPH_CLOSE, np.ones((9, 9), 'uint8'))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(m.shape, 'uint8')
    cv2.drawContours(filled, contours, -1, 1, -1)
    filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, np.ones((11, 11), 'uint8'))
    _, _, stats, _ = cv2.connectedComponentsWithStats(filled)
    boxes = [tuple(map(int, s[:4])) for s in stats[1:] if s[4] > 4000 and min(s[2], s[3]) > 40]
    changed = True
    while changed and len(boxes) > 3:
        changed = False
        for i in range(len(boxes)):
            if changed:
                break
            for j in range(i + 1, len(boxes)):
                a, b = (boxes[i], boxes[j])
                ax, ay, aw, ah = a
                bx, by, bw, bh = b
                inter = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
                small = min(aw * ah, bw * bh)
                large = max(aw * ah, bw * bh)
                gx = max(ax - bx - bw, bx - ax - aw, 0)
                gy = max(ay - by - bh, by - ay - ah, 0)
                overlap = inter > small * 0.1
                fragment = small / large < 0.3 and (gx == 0 and gy <= 30 or (gy == 0 and gx <= 30))
                if overlap or fragment:
                    x, y = (min(ax, bx), min(ay, by))
                    merged = (x, y, max(ax + aw, bx + bw) - x, max(ay + ah, by + bh) - y)
                    boxes = [v for k, v in enumerate(boxes) if k not in [i, j]] + [merged]
                    changed = True
                    break
    return sorted(boxes, key=lambda b: (b[1], b[0]))
