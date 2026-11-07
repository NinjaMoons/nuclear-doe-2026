"""Deterministic synthetic P0 quartet: 4 distinct L8 128x128 amplitude PNGs (not contest data)."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=False)
def canvas():
    return Image.new("L", (128, 128), 0)
imgs = []
a = canvas(); ImageDraw.Draw(a).ellipse((34, 34, 94, 94), fill=255); imgs.append(a)
b = canvas(); ImageDraw.Draw(b).rectangle((20, 50, 108, 78), fill=255); imgs.append(b)
c = canvas(); d = ImageDraw.Draw(c); d.line((24, 24, 104, 104), fill=255, width=9); d.line((24, 104, 104, 24), fill=255, width=9); imgs.append(c)
y, x = np.mgrid[0:128, 0:128]
g = np.where((x - 64) ** 2 + (y - 64) ** 2 < 50 ** 2, np.floor(255 * x / 127 + 0.5), 0).astype(np.uint8)
imgs.append(Image.fromarray(g, mode="L"))
for i, im in enumerate(imgs, 1):
    im.save(out / f"z{i}.png")
print("ok", [str(out / f"z{i}.png") for i in range(1, 5)])
