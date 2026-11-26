import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

src_dir = sys.argv[1]
frame = int(sys.argv[2])
out = sys.argv[3]
names = ["source"] + sys.argv[4].split(",")
tiles = []
for n in names:
    exr = os.path.join(src_dir, "%s_%04d.exr" % (n, frame))
    png = os.path.join(tempfile.gettempdir(), "tp_cs_%s.png" % n)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", exr, "-vf", "scale=640:360", png], check=True)
    im = Image.open(png).convert("RGB")
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        f = ImageFont.load_default()
    d.rectangle([0, 0, 640, 30], fill=(0, 0, 0))
    d.text((8, 3), n, fill=(218, 163, 108), font=f)
    tiles.append(im)
cols = 3
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (cols * 640, rows * 360), (20, 23, 25))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * 640, (i // cols) * 360))
sheet.save(out)
print(out, sheet.size)
