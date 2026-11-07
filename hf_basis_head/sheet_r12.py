"""RESEARCH ONLY (R12): VAL contact sheet Target | A stem | B const | C hf for one quartet (saved exports only).
usage: python -I sheet_r12.py <analysis.json> <which:rule|final> <quartet_key> <out.png> <label=run_dir> ..."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
an = json.loads(Path(sys.argv[1]).read_text()); which, key, out = sys.argv[2], sys.argv[3], Path(sys.argv[4])
arms = [a.split("=", 1) for a in sys.argv[5:]]
T, GAP = 208, 10
F = "/System/Library/Fonts/Supplemental/Arial.ttf"; FB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
f = lambda s, b=False: ImageFont.truetype(FB if b else F, s)
INK, GREY, RED = (31, 41, 51), (91, 103, 112), (200, 30, 30)
Z = ["24,1", "25,3", "26,5", "27,7"]; num = lambda v: f"{v:.2f}".replace(".", ",").replace("-", "−")
def tile(p):
    im = Image.open(p); assert im.mode == "L" and im.size == (128, 128), p
    return im.resize((T, T), Image.NEAREST).convert("RGB")
def qdir(label, run):
    c = an[label]["checkpoints"][which]; d = Path(run) / "eval_val" / "export" / c["name"] / "val"
    for q in sorted(d.iterdir()):
        if json.loads((q / "manifest.json").read_text())["metadata"]["sample_key"] == key:
            return q, c, next(r for r in c["rows"] if r["key"] == key)
    raise SystemExit(f"{key} not found")
LX, LABW = 30, 250; DX = LX + LABW; GX = DX + T + 30; W = GX + 4 * T + 3 * GAP + 30; ROWH = T + 38
H = 110 + (1 + len(arms)) * ROWH + 50
can = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(can)
d.text((LX, 20), f"VAL {key} · R12 STEM + basis head ({which} checkpoints)", fill=INK, font=f(22, True))
d.text((LX, 52), "Saved uint8 DOE.png → reload → official Fresnel |U| → PNG. ×(208/128) nearest, PNG values unchanged. Под тайлом: Pearson · SSIM · argmax цели.", fill=GREY, font=f(14))
y0 = 82
for i in range(4):
    d.text((GX + i * (T + GAP) + T / 2, y0 + 8), f"z{i + 1} = {Z[i]} мм", fill=INK, font=f(15, True), anchor="mm")
q0 = qdir(*arms[0])[0]
ry = y0 + 24
d.text((LX, ry + T / 2 - 10), "TARGET", fill=INK, font=f(16, True))
for i in range(4):
    can.paste(tile(q0 / "targets" / f"target_{i + 1:02d}.png"), (GX + i * (T + GAP), ry))
for r, (label, run) in enumerate(arms):
    q, c, row = qdir(label, run); ry = y0 + 24 + (r + 1) * ROWH
    d.text((LX, ry + T / 2 - 34), f"{label}", fill=INK, font=f(16, True))
    d.text((LX, ry + T / 2 - 12), f"step {c['step']} · own {row['own']}/4", fill=GREY, font=f(13))
    d.text((LX, ry + T / 2 + 6), f"margin {num(row['margin'])} · inter-z {num(row['interz']['full'])}", fill=GREY, font=f(13))
    d.text((LX, ry + T / 2 + 24), f"HF {num(row['hf'])} · R90 {num(row['r90'])}", fill=GREY, font=f(13))
    can.paste(tile(q / "DOE.png"), (DX, ry))
    worst = min(range(4), key=lambda i: row["P"][i])
    for i in range(4):
        x = GX + i * (T + GAP)
        can.paste(tile(q / f"reconstructed_{i + 1:02d}.png"), (x, ry))
        if i == worst:
            d.rectangle((x - 3, ry - 3, x + T + 2, ry + T + 2), outline=RED, width=3)
        am = max(range(4), key=lambda j: row["C"][i][j])
        d.text((x + T / 2, ry + T + 12), f"P {num(row['P'][i])} · S {num(row['S'][i])} · z{am + 1}", fill=RED if i == worst else INK, font=f(13), anchor="mm")
d.text((LX, H - 34), "Research only, VAL. A = STEM scalar head; B = +16 coef maps 32×32, R_j = 1; C = same, R_j = HF uniform(seed 2026). Same seed/batches/init.", fill=GREY, font=f(13))
can.save(out, optimize=True); print(out, can.size)
