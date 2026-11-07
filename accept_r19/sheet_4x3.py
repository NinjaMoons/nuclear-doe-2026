"""RESEARCH ONLY (#45/#47): honest 4x3 sheet (rows TARGET / arm1 / arm2, columns z1..z4) + 4x4 own/foreign Pearson
matrices, from official saved PNG exports (no contrast/denoise; uniform nearest resize).
usage: python -I sheet_4x3.py <analysis.json> <which> <quartet_key> <out.png> <label=run_dir> <label=run_dir>"""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
an = json.loads(Path(sys.argv[1]).read_text()); which, key, out = sys.argv[2], sys.argv[3], Path(sys.argv[4])
arms = [a.split("=", 1) for a in sys.argv[5:]]
T, GAP = 224, 10
F = "/System/Library/Fonts/Supplemental/Arial.ttf"; FB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
f = lambda s, b=False: ImageFont.truetype(FB if b else F, s)
INK, GREY, RED = (31, 41, 51), (91, 103, 112), (200, 30, 30)
Z = ["24,1", "25,3", "26,5", "27,7"]; num = lambda v: f"{v:+.2f}".replace(".", ",").replace("-", "−") if v < 0 else f"{v:.2f}".replace(".", ",")
def tile(p):
    im = Image.open(p); assert im.mode == "L" and im.size == (128, 128), p
    return im.resize((T, T), Image.NEAREST).convert("RGB")
def qdir(label, run):
    c = an[label]["checkpoints"][which]; d = Path(run) / "eval_val" / "export" / c["name"] / "val"
    for q in sorted(d.iterdir()):
        if json.loads((q / "manifest.json").read_text())["metadata"]["sample_key"] == key:
            return q, c, next(r for r in c["rows"] if r["key"] == key)
    raise SystemExit(key)
LX, LABW = 30, 190; GX = LX + LABW; W = GX + 4 * T + 3 * GAP + 30; ROWH = T + 36
MAT_H = 150
H = 96 + 3 * ROWH + MAT_H + 40
can = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(can)
d.text((LX, 18), f"VAL {key} · {' vs '.join(a for a, _ in arms)} · {which} checkpoints", fill=INK, font=f(21, True))
d.text((LX, 48), "Official DOE.png uint8 → reload → official Fresnel |U| → PNG exactly as evaluated; uniform nearest ×1.75, no contrast/denoise. Native per-z pitch.", fill=GREY, font=f(13))
y0 = 72
for i in range(4):
    d.text((GX + i * (T + GAP) + T / 2, y0 + 8), f"z{i + 1} = {Z[i]} мм", fill=INK, font=f(15, True), anchor="mm")
q0 = qdir(*arms[0])[0]; ry = y0 + 22
d.text((LX, ry + T / 2 - 10), "TARGET", fill=INK, font=f(16, True))
for i in range(4):
    can.paste(tile(q0 / "targets" / f"target_{i + 1:02d}.png"), (GX + i * (T + GAP), ry))
rows = []
for r, (label, run) in enumerate(arms):
    q, c, row = qdir(label, run); rows.append((label, row)); ry = y0 + 22 + (r + 1) * ROWH
    d.text((LX, ry + T / 2 - 22), label, fill=INK, font=f(16, True))
    d.text((LX, ry + T / 2), f"step {c['step']} · own {row['own']}/4", fill=GREY, font=f(13))
    d.text((LX, ry + T / 2 + 18), f"margin {num(row['margin'])}", fill=GREY, font=f(13))
    worst = min(range(4), key=lambda i: row["P"][i])
    for i in range(4):
        x = GX + i * (T + GAP)
        can.paste(tile(q / f"reconstructed_{i + 1:02d}.png"), (x, ry))
        if i == worst:
            d.rectangle((x - 3, ry - 3, x + T + 2, ry + T + 2), outline=RED, width=3)
        am = max(range(4), key=lambda j: row["C"][i][j])
        d.text((x + T / 2, ry + T + 12), f"P {num(row['P'][i])} · S {num(row['S'][i])} · argmax z{am + 1}", fill=RED if am != i else INK, font=f(13), anchor="mm")
my = y0 + 22 + 3 * ROWH + 4
for j, (label, row) in enumerate(rows):
    mx = LX + j * 520
    d.text((mx, my), f"{label}: Pearson(recon z_k, target z_j)", fill=INK, font=f(14, True))
    for k in range(4):
        cells = "   ".join(("[" if jj == k else " ") + f"{row['C'][k][jj]:+.2f}" + ("]" if jj == k else " ") for jj in range(4))
        d.text((mx, my + 22 + 20 * k), f"z{k + 1}: {cells}", fill=INK, font=f(13))
d.text((LX, H - 28), "Red caption = recon matches a foreign target best (ghost/identity error); red frame = worst plane. Research only, VAL.", fill=GREY, font=f(13))
can.save(out, optimize=True); print(out, can.size)
