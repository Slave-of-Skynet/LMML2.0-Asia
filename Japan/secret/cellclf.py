import os
"""Kanji / kana / punctuation classifier for caption character cells, trained on hand-read lines."""
import numpy as np, json, os, sys, csv
from PIL import Image
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subcells

LABELLED = {
 "w08_002": "秋の雨、歩き続けた|夜がまだ続いていた、街灯のない帰り道で|遠い花火、夜はいつも少し遠い|河川敷の風の中で明日のことは考えなかった",
 "w08_006": "最終バス、冷めた缶コーヒーの味|答えのない問いのまま、水たまりの縁に|眠らない街の灯|日曜の朝の商店街で誰の足音でもなかった",
 "w08_010": "立ち尽くした|雨上がり、鳥が飛んだ|名前のない光、街灯のない帰り道で|おかえり、と言ってくれる人はいない|鍵を落とした、昼下がりの立体駐車場で",
 "w08_011": "雨音だけが聞こえた、古い公衆電話の前で",
 "w08_013": "開けてほしいと、少しだけ思っている|光がゆっくり滲んだ、夜勤明けの駅前で",
 "w08_017": "冬の気配、最終電車の音|街灯、傘を閉じた|長い夜、誰かが笑っている",
 "w08_019": "雨上がり、まだ名前を知らない",
 "w08_023": "踏切の音、時間が止まって見えた|遠い街、雨はやみそうにない|空き地のフェンス越しに夜はいつも少し遠い|河川敷の風の中で昨日の雨がまだ残る|それだけのこと、明日のことは考えなかった",
 "w09_001": "深夜のコンビニの前で、春はまだ遠かった|冷たい手すりに触れて、何も変わらなかった",
 "w09_004": "誰かの帽子と傘、雨戸の閉まった家の前で|三月の冷たい雨、冷たい手すりに触れて|電線の絡まる空の下、空を見上げた|眠らない街、誰かが笑っている",
 "w09_008": "夜の駅、何も始まっていない|工事中、目を閉じた|路地裏の猫、帰る理由を探していた|工事中、誰もいない夜の駅",
 "w09_011": "夜の駅、一人ではなかった|誰もいなかった、開かずの踏切の前で|秋の雨|傘を忘れた帰り道に遠くで電車の音がした|午前三時、息を吐いた",
 "w09_013": "誰も読まない掲示板に、街の音が遠くなった|誰かの忘れ物の前で街灯が一つ消えた|月が雲に隠れた、歩道橋の真ん中で",
 "w09_014": "息を吐いた、閉店後の商店街を|遠い街、忘れかけた夏の匂い|鍵を落とした",
 "w09_015": "錆びた階段、明日のことは考えなかった|始発、息を吐いた|曇りガラスの向こうに雨が止んだ|影が長くなった|排気口の温い風の中で、影が長くなった",
 "w09_021": "遠い街、もう一度だけ振り返った|遠い街、片方だけの手袋|草の匂いと、遠くで錆びる鉄の匂い。|誰もいない踏切でまだ誰も知らない朝",
 "w09_022": "答えはまだ出ていない、夕暮れの団地の隅で|いつもの道|雨の匂いの残る路地に、片方だけの手袋|ガード下",
 "w12_001": "雨宿り、傘を開く音がした|雨の日の洗濯物の下で片方だけの手袋",
 "w12_005": "誰も降りないバス停に月が雲に隠れた|名前のない路地の奥で、三月の冷たい雨|秋の雨|排気口の温い風の中で雨が止んだ",
 "w12_008": "排気口の温い風の中で、夜がまだ続いていた|誰かの帽子と傘、消えかけた横断歩道に|白い息の向こうに、誰もいない夜の駅|閉店後、少しだけ立ち止まった",
 "w12_010": "帰り道、ただ静かだった|郵便受けの並ぶ玄関で雨音だけが聞こえた|曇りガラスの向こうに立ち尽くした",
 "w12_020": "遠い花火|湿った段ボールの匂いに、冬の気配がした|遠回りをした|冷たい手すりに触れて、傘を閉じた",
 "w12_025": "三月の冷たい雨|まだ濡れている階段で、言葉にならなかった|街の音が遠くなった、まだ濡れている階段で|遠い花火、傘を閉じた|空き缶、時間が止まって見えた",
 "w12_027": "光の向こう、ポケットの中の小銭を数えた",
 "w12_029": "雨傘、傘を閉じた|雨上がり、ここで降りるつもりだ|遠い街、心だけが先に帰った",
}
S = 20


def cls_of(c):
    o = ord(c)
    if 0x4E00 <= o <= 0x9FFF or c == "々":
        return 2
    if c in "、。・":
        return 0
    return 1


def cell_images(path):
    """Re-run the cell geometry and return per line a list of normalised SxS glyph crops."""
    x = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    info = subcells.process(path)
    if not info:
        return None
    y0, y1, x0, x1 = info["box"]; y0 += 3; x0 += 3; y1 -= 3; x1 -= 3
    g = x.mean(-1)[y0:y1, x0:x1]
    bg = ndimage.grey_opening(g, size=(15, 15))
    gm = subcells.glyph_mask(x)[y0:y1, x0:x1]
    t = np.percentile(g[gm], 90) if gm.any() else 255
    ink = np.clip((g - bg) / max(t - np.median(bg), 1), 0, 1)
    sub = gm
    lines = [l for l in subcells.lines_of(sub) if 12 <= l[1] - l[0] <= 80]
    p = info["pitch"]; left = info["left"]
    # keep lines aligned to the block (same rule as subcells.process)
    keep = []
    for a, b in lines:
        cols = np.where(sub[a:b].any(0))[0]
        if len(cols) and abs(cols[0] - left) < 0.6 * p:
            keep.append((a, b, cols[-1]))
    out = []
    for a, b, c1 in keep:
        n = int(round((c1 - left + 1 + 0.08 * p) / p))
        mid = (a + b) / 2
        row = []
        for k in range(n):
            xa = int(round(left + k * p)); xb = int(round(left + (k + 1) * p))
            ya = int(round(mid - p / 2)); yb = int(round(mid + p / 2))
            crop = ink[max(ya, 0):yb, xa:xb]
            if crop.size == 0:
                row.append(np.zeros(S * S)); continue
            im = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize((S, S), Image.BILINEAR), np.float32) / 255
            row.append(im.ravel())
        out.append(row)
    return out


def feats(v):
    im = v.reshape(S, S)
    gy, gx = np.gradient(im)
    return np.concatenate([v, [im.mean(), np.abs(gx).mean(), np.abs(gy).mean(), (im > 0.5).mean()],
                           im.mean(0), im.mean(1)])


