import os
import sys
from PIL import Image, ImageDraw

PETROL = (0x3F, 0x98, 0xA6)
SLICES = [(0.0, 0.35), (3.5, 0.6), (7.0, 1.0)]
W, H, R, SPAN, PAD = 9.0, 14.0, 2.0, 16.0, 1.0


SNAPPED = {
    16: (9, 14, 2, [0, 4, 7]),
    20: (12, 18, 2, [0, 4, 8]),
    24: (14, 21, 3, [0, 5, 10]),
    32: (18, 28, 4, [0, 7, 14]),
    48: (27, 42, 6, [0, 11, 21]),
}


def draw(size, w, h, r, offsets, ox, oy):
    ss = 8
    out = Image.new("RGBA", (size * ss, size * ss), (0, 0, 0, 0))
    for (_, a), x in zip(SLICES, offsets):
        layer = Image.new("RGBA", out.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).rounded_rectangle(
            [(ox + x) * ss, oy * ss, (ox + x + w) * ss - 1, (oy + h) * ss - 1], radius=r * ss, fill=PETROL + (round(255 * a),))
        out = Image.alpha_composite(out, layer)
    return out.resize((size, size), Image.BOX)


def mark(size):
    if size in SNAPPED:
        w, h, r, offsets = SNAPPED[size]
        return draw(size, w, h, r, offsets, (size - offsets[-1] - w) // 2, (size - h) // 2)
    unit = size / (SPAN + 2 * PAD)
    return draw(size, W * unit, H * unit, R * unit, [x * unit for x, _ in SLICES], PAD * unit, (size - H * unit) / 2)


def svg_mark():
    rects = "".join(
        f'<rect x="{x}" y="1" width="{W:g}" height="{H:g}" rx="{R:g}" fill="#3f98a6" fill-opacity="{a}"/>' for x, a in SLICES)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="16" viewBox="-1 0 18 16" role="img">'
            f'<title>TEMPORAL</title>{rects}</svg>\n')


def main(out):
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "temporal-mark.svg"), "w", encoding="utf-8").write(svg_mark())
    for s in (16, 32, 64, 128, 256, 512):
        mark(s).save(os.path.join(out, f"temporal-mark-{s}.png"))
    sizes = [16, 20, 24, 32, 48, 64, 256]
    mark(256).save(os.path.join(out, "temporal.ico"), sizes=[(s, s) for s in sizes], append_images=[mark(s) for s in sizes[:-1]])
    mark(128).save(os.path.join(out, "ofx-icon.png"))
    print("ok", out)


if __name__ == "__main__":
    main(sys.argv[1])
