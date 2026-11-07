"""Visual sheet from saved VAL exports (no inference): per quartet one row block TARGET / arm1 / arm2 x z1..z4.
argv: <out_dir> <label1=export_val_dir1> <label2=export_val_dir2>   -> writes sheet_A.png, sheet_B.png"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
arms = [a.split("=", 1) for a in sys.argv[2:]]
dirs = {lab: sorted(Path(d).iterdir()) for lab, d in arms}
first = dirs[arms[0][0]]
T, PAD, LAB = 128, 4, 14
for group in ("A", "B"):
    rows = []
    for i, q in enumerate(first):
        key = json.loads((q / "manifest.json").read_text(encoding="utf-8"))["metadata"]["sample_key"]
        if key[0] != group:
            continue
        lines = [("TARGET " + key, [q / "targets" / f"target_{k:02d}.png" for k in range(1, 5)])]
        for lab, _ in arms:
            qq = dirs[lab][i]
            assert json.loads((qq / "manifest.json").read_text(encoding="utf-8"))["metadata"]["sample_key"] == key
            met = json.loads((qq / "metrics.json").read_text(encoding="utf-8"))["planes"]
            ps = " ".join(f"{p['pearson']:.2f}" for p in met)
            lines.append((f"{lab} P {ps}", [qq / f"reconstructed_{k:02d}.png" for k in range(1, 5)]))
        rows.extend(lines)
    W = 4 * (T + PAD) + PAD
    H = len(rows) * (T + LAB + PAD) + PAD
    sheet = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(sheet)
    for r, (label, paths) in enumerate(rows):
        y = PAD + r * (T + LAB + PAD)
        draw.text((PAD, y), label, fill=255)
        for k, p in enumerate(paths):
            sheet.paste(Image.open(p).convert("L"), (PAD + k * (T + PAD), y + LAB))
    sheet.save(out / f"sheet_{group}.png")
    print("wrote", out / f"sheet_{group}.png", sheet.size)
