"""Cell materials and temporal ink; independent implementations, no site code."""

from __future__ import annotations

import math
from collections import deque
from functools import wraps
from typing import Any

import numpy as np

if __package__:
    from cutwright.modules.fx.runners import fx_cells as C
    from cutwright.modules.fx.runners.fx_base import KERNELS, Frame, kernel
    from cutwright.modules.fx.runners.fx_glyphs import FONT, atlas, is_cjk, label_font
else:
    import fx_cells as C
    from fx_base import KERNELS, Frame, kernel
    from fx_glyphs import FONT, atlas, is_cjk, label_font


def cell_kernel(name: str) -> Any:
    def decorate(fn: Any) -> Any:
        @wraps(fn)
        def draw(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
            if getattr(x, "is_cuda", False):
                if __package__:
                    from cutwright.modules.fx.runners.fx_cells_gpu import draw as gpu_draw
                else:
                    from fx_cells_gpu import draw as gpu_draw
                result = gpu_draw(o, name, x, p, f)
                if result is not None:
                    return result
            image = np.nan_to_num(o.numpy(x)).astype(np.float32)
            chosen = fn
            if p.get("render_style", "reference") == "reference":
                if __package__:
                    from cutwright.modules.fx.runners.fx_cells_reference import DRAWERS
                else:
                    from fx_cells_reference import DRAWERS
                chosen = DRAWERS.get(name, fn)
            result = chosen(image, p, f)
            amount = C.zone(image, p)[None]
            return o.asarray(np.clip(image + (result - image) * amount, 0, 1))

        kernel(name)(draw)
        return draw

    return decorate


def glyphs(
    g: C.Grid, indices: Any, chars: str, *, sorted_ink: bool = False, regular: bool = False
) -> Any:
    table = atlas(chars, g.ch, g.cw, regular=regular)
    if sorted_ink:
        table = table[np.argsort(table.mean((1, 2)), kind="stable")]
    idx = np.clip(g.expand(indices).astype(np.int32), 0, len(chars) - 1)
    yy = np.clip((g.py * table.shape[1]).astype(np.int32), 0, table.shape[1] - 1)
    xx = np.clip((g.px * table.shape[2]).astype(np.int32), 0, table.shape[2] - 1)
    return table[idx, yy, xx]


def paper(p: dict[str, Any]) -> Any:
    return C.rgb(p.get("paper", "#0c0c0c"))


def covered(g: C.Grid, p: dict[str, Any]) -> Any:
    gap = float(p.get("gap", 0)) / 2
    return (g.px >= gap) & (g.px <= 1 - gap) & (g.py >= gap) & (g.py <= 1 - gap)


@cell_kernel("ascii_cells")
def ascii_cells(x: Any, p: dict[str, Any], f: Frame) -> Any:
    g = C.grid(x, p, f, 0.6)
    chars = p.get("charset", " .,:;irsXA253hMHGS9B&@")
    a = glyphs(
        g,
        np.floor(g.tone * (len(chars) - 0.001)),
        chars,
        sorted_ink=True,
        regular=not p.get("bold", True),
    )
    color = g.expand(C.ink(g, p))
    out = paper(p) + (color - paper(p)) * (a * covered(g, p))[None]
    return out + float(p.get("photo", 0)) * x


@cell_kernel("glyph_haze")
def glyph_haze(x: Any, p: dict[str, Any], f: Frame) -> Any:
    g = C.grid(x, p, f, 0.65)
    chars = p.get("charset", " .:-=+*#%@")
    a = glyphs(
        g,
        np.floor(g.tone * (len(chars) - 0.001)),
        chars,
        sorted_ink=True,
        regular=not p.get("bold", True),
    )
    color = g.expand(C.ink(g, p))
    y = C.lum(x)[None]
    photograph = np.clip(y + (x - y) * float(p.get("saturation", 1)), 0, 1)
    out = paper(p) + float(p.get("photo", 0.25)) * photograph
    out += color * a[None] * covered(g, p)[None]
    glow = float(p.get("glow", 0))
    return out + glow * C.gaussian(out, max(0.5, x.shape[2] / 400))


_BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16


@cell_kernel("phrase_dither")
def phrase_dither(x: Any, p: dict[str, Any], f: Frame) -> Any:
    g = C.grid(x, p, f, 0.6)
    phrase = p.get("phrase", "NOWBOARDINGPLATFORMNINE") or " "
    shift = 0
    if p.get("animate", True):
        shift = math.floor(C.clock(p, f) * float(p.get("speed", 1.2)) * 8)
    columns = math.ceil(x.shape[2] / g.cw)
    index = (g.cx + g.cy * columns + shift).astype(np.int32) % len(phrase)
    a = glyphs(g, index, phrase)
    threshold = float(p.get("threshold", 0.57))
    text = g.expand(C.smooth(threshold - 0.08, threshold + 0.08, g.tone))
    h, w = x.shape[1:]
    yy, xx = np.mgrid[:h, :w]
    unit = max(1, round(float(p.get("dither_size", 3)) * w / float(p.get("reference_width", 1200))))
    bayer = _BAYER[(yy // unit) % 4, (xx // unit) % 4]
    fine = np.clip((C.lum(x) - 0.5) * float(p.get("contrast", 1)) + 0.5, 0, 1)
    dither = (fine > bayer + 0.02).astype(np.float32)
    coverage = (text * a + (1 - text) * dither * 0.65) * covered(g, p)
    out = paper(p) + (g.expand(C.ink(g, p)) - paper(p)) * coverage[None]
    radius = float(p.get("glow_radius", 10)) * w / float(p.get("reference_width", 1200))
    return out + C.gaussian(np.maximum(out - paper(p), 0), max(0.3, radius)) * float(
        p.get("glow", 0)
    )


@cell_kernel("code_blocks")
def code_blocks(x: Any, p: dict[str, Any], f: Frame) -> Any:
    g = C.grid(x, p, f)
    chars = p.get("charset", "0X8SG>")
    t = C.clock(p, f)
    n = C.noise(g.cy, g.cx, f.seed)
    tick = np.floor(t * float(p.get("flicker", 4.9)) + n * 11)
    index = ((n * 811 + tick) % len(chars)).astype(np.int32)
    a = glyphs(g, index, chars)
    alpha = (
        float(p.get("opacity_min", 0))
        + (float(p.get("opacity_max", 0.68)) - float(p.get("opacity_min", 0))) * g.tone
    )
    phase = 2 * math.pi * (t / float(p.get("loop_seconds", 5)))
    alpha *= 0.75 + 0.25 * np.sin(n * 7 + phase * round(float(p.get("flow", 2.2))))
    blocks = paper(p) + (g.colors * float(p.get("gain", 1)) - paper(p)) * alpha[None]
    out = g.expand(blocks)
    color = C.ink(g, p)
    pick = C.noise(g.cy + 17, g.cx, f.seed)
    color = np.where((pick < 0.18)[None], C.rgb(p.get("accent_a", "#7adfff")), color)
    color = np.where(
        ((pick >= 0.18) & (pick < 0.36))[None], C.rgb(p.get("accent_b", "#ff7a4a")), color
    )
    a *= g.expand(0.15 + 0.55 * n)
    out = out + (g.expand(color) - out) * a[None]
    return out * covered(g, p) + paper(p) * (~covered(g, p))


@cell_kernel("glyph_blocks")
def glyph_blocks(x: Any, p: dict[str, Any], f: Frame) -> Any:
    g = C.grid(x, p, f)
    n = C.noise(g.cy, g.cx, f.seed)
    threshold = float(p.get("threshold", 0.6))
    lit = C.smooth(threshold - 0.12, threshold + 0.08, g.tone)
    block = ((g.py < float(p.get("block_height", 0.85))) & covered(g, p)).astype(np.float32)
    chars = p.get("charset", "0123456789")
    digit = glyphs(g, np.floor(n * len(chars)), chars)
    choose = g.expand(n < float(p.get("digits", 0.25)))
    a = g.expand(lit) * np.where(choose, digit, block)
    return paper(p) + (g.expand(C.ink(g, p)) - paper(p)) * a[None]


@cell_kernel("tone_atlas")
def tone_atlas(x: Any, p: dict[str, Any], f: Frame) -> Any:
    if p.get("style") == "lines":
        tone = C.lum(C.gaussian(x, float(p.get("smoothing", 1)) * x.shape[2] / 1200))
        tone = np.clip((tone - 0.5) * float(p.get("contrast", 1)) + 0.5, 0, 1) ** (
            1 / float(p.get("gamma", 1))
        )
        bands = int(p.get("bands", 16))
        y, z = np.gradient(tone) if min(tone.shape) > 1 else (tone * 0, tone * 0)
        width = np.maximum(np.hypot(y, z) * bands, 0.005)
        distance = np.abs((tone * bands + 0.5) % 1 - 0.5)
        a = np.clip((width - distance) / np.maximum(width, 0.01) + 0.5, 0, 1)
        colors = C.rgb(p.get("ink", "#d8ff3e")) * 0.6
        return paper(p) + (colors - paper(p)) * a[None] * 0.6
    g = C.grid(x, p, f)
    n = C.noise(g.cy, g.cx, int(p.get("seed", 0)))
    tone = C.gaussian(g.tone[None], float(p.get("smoothing", 1)) * 0.3)[0]
    gx = np.abs(tone - np.roll(tone, 1, 1))
    gy = np.abs(tone - np.roll(tone, 1, 0))
    edge = np.maximum(gx, gy)
    level = np.floor(tone * int(p.get("bands", 10)))
    border = (level != np.roll(level, 1, 0)) | (level != np.roll(level, 1, 1))
    band = np.exp(-((np.sin(tone * int(p.get("bands", 10)) * math.pi) / 0.3) ** 2))
    energy = np.clip((band * 0.3 + edge * 2) * (tone > 0.03) * (tone < 0.98), 0, 0.6)
    py, px = g.py - 0.5, g.px - 0.5
    style = p.get("style", "dots")
    if style == "blocks":
        a = covered(g, {**p, "gap": max(0.08, p.get("gap", 0))}).astype(np.float32)
        a *= g.expand(energy * (0.35 + n * 0.65))
        out = paper(p) * 0.7 + g.expand(C.ink(g, p)) * a[None] * 0.65
    else:
        radius = g.expand(0.08 + 0.32 * energy)
        a = np.clip((radius - np.hypot(py, px)) * min(g.ch, g.cw) + 0.5, 0, 1)
        out = paper(p) * 0.05 + g.expand(C.ink(g, p) * (energy * 1.5)[None]) * a[None]
    boxes = g.expand(border & (n < float(p.get("boxes", 0.3))))
    rim = ((np.abs(px) > 0.35) | (np.abs(py) > 0.35)) & (np.abs(px) < 0.45) & (np.abs(py) < 0.45)
    out += g.expand(C.ink(g, p)) * (boxes * rim * float(p.get("line_opacity", 0.9)) * 0.5)[None]
    labels = (n < float(p.get("labels", 0.25)) * 0.15) & (energy > 0.25)
    label = p.get("label_text", "PLATFORM") or " "
    text = glyphs(g, np.floor(n * len(label)), label) * g.expand(labels)
    out += C.rgb("#ffdfba") * text[None] * 1.5
    heroes = (energy > 0.12) & (n > 1 - float(p.get("heroes", 0.55)) * 0.035)
    hot = g.expand(heroes) * np.exp(-(px * px + py * py) * 14)
    out += C.rgb(p.get("accent", "#ef282d")) * hot[None] * (1.5 + 2 * float(p.get("glow", 0.45)))
    ticks = g.expand((n < float(p.get("ticks", 0.45)) * 0.2) & border)
    cross = (np.abs(px) < 0.07) | (np.abs(py) < 0.07)
    out += C.rgb(p.get("ink", "#c9c7c0")) * (ticks * cross * 0.35)[None]
    glow = float(p.get("glow", 0.45))
    return out + glow * C.gaussian(np.maximum(out - 0.2, 0), max(0.5, x.shape[2] / 400))


@cell_kernel("tone_stripes")
def tone_stripes(x: Any, p: dict[str, Any], f: Frame) -> Any:
    g = C.grid(x, p, f, 1.4)
    tone = g.tone
    color = np.where(
        (tone > 0.7)[None],
        C.rgb(p.get("light", "#bcbcbc")),
        np.where(
            (tone > 0.3)[None], C.rgb(p.get("medium", "#787878")), C.rgb(p.get("dark", "#343434"))
        ),
    )
    a = g.py < float(p.get("stripe_height", 0.8))
    out = paper(p) + (g.expand(color) - paper(p)) * a[None]
    if p.get("axis"):
        out[:, :, out.shape[2] // 2] = C.rgb(p.get("ink", "#ffffff"))[:, 0]
    return out


@cell_kernel("glass_tags")
def glass_tags(x: Any, p: dict[str, Any], f: Frame) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    _, h, w = x.shape
    scale = w / float(p.get("reference_width", 1200))
    height = max(8, round(float(p.get("cell_size", 35)) * scale))
    gap = max(1, round(7 * scale))
    row_gap = max(1, round(float(p.get("row_gap", 6)) * scale))
    font = ImageFont.truetype(str(FONT), max(6, round(height * 0.5)))
    words = [
        word.strip()
        for word in p.get("words", "MOTH, SLEEPY, KICKS, FAST, NOW").split(",")
        if word.strip()
    ]
    if not words:
        return np.broadcast_to(paper(p), x.shape).copy()
    image = Image.new("RGB", (w, h), tuple((paper(p)[:, 0, 0] * 255).astype(np.uint8)))
    draw = ImageDraw.Draw(image)
    y, row, word_index = 0, 0, 0
    while y < h:
        left, col = -height * (row % 2), 0
        while left < w:
            n = float(C.noise(row, col, int(p.get("seed", 0))))
            index = word_index if p.get("word_order") == "sequence" else int(n * len(words))
            word = words[index % len(words)]
            face = label_font(max(6, round(height * 0.5)), True) if is_cjk(word) else font
            width = max(height, round(draw.textlength(word, font=face) + height * 1.1))
            cx, cy = min(w - 1, max(0, left + width // 2)), min(h - 1, y + height // 2)
            color = x[:, cy, cx]
            luma = float(color @ C.LUMA)
            color = np.clip(luma + (color - luma) * float(p.get("saturation", 1.75)), 0, 1)
            color = color * (0.7 + 0.2 * float(p.get("glass", 0.85))) + 0.08
            fill = tuple((np.clip(color, 0, 1) * 255).astype(np.uint8))
            radius = min(height / 2, float(p.get("rounding", 10)) * scale)
            if float(p.get("shadow", 1)) > 0:
                draw.rounded_rectangle(
                    (left + 1, y + 2, left + width + 1, y + height + 2),
                    radius=radius,
                    fill="#030305",
                )
            draw.rounded_rectangle(
                (left, y, left + width, y + height),
                radius=radius,
                fill=fill,
                outline=tuple(min(255, int(c) + 15) for c in fill),
                width=1,
            )
            text = "#141414" if luma > 0.4 else "#f5f1e9"
            draw.text((left + width / 2, y + height / 2), word, font=face, anchor="mm", fill=text)
            left += width + gap
            col += 1
            word_index += 1
        y += height + row_gap
        row += 1
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255


@cell_kernel("edge_sparks")
def edge_sparks(x: Any, p: dict[str, Any], f: Frame) -> Any:
    g = C.grid(x, {**p, "cell_size": max(2, 12 / float(p.get("density", 4)))}, f)
    gy, gx = np.gradient(g.tone) if min(g.tone.shape) > 1 else (0 * g.tone, 0 * g.tone)
    edge = np.clip(np.hypot(gx, gy) * 8, 0, 1)
    n = C.noise(g.cy, g.cx, int(p.get("seed", 10)))
    hit = n < (edge * 0.16 + g.tone * float(p.get("dust", 0.62)) * 0.008)
    phase = 2 * math.pi * C.clock(p, f) / float(p.get("loop_seconds", 5))
    energy = (
        hit * (0.5 + 0.5 * g.tone) * (1 + float(p.get("twinkle", 0.34)) * np.sin(n * 100 + phase))
    )
    radius = float(p.get("particle_size", 3.1)) / max(2, g.ch) * 0.5
    a = np.clip((radius - np.hypot(g.px - 0.5, g.py - 0.5)) * g.ch + 0.5, 0, 1)
    out = paper(p) + x * float(p.get("underlay", 0.16))
    color = g.colors / np.maximum(g.colors.max(0, keepdims=True), 0.05)
    return out + g.expand(color * energy[None]) * a[None] * float(p.get("gain", 1.15))


def temporal(value: Any, f: Frame, name: str, decay: float, count: int) -> Any:
    state = f.state.setdefault(name, {})
    previous = state.get("index", f.index - 1)
    if f.index != previous + 1 or state.get("shape") != value.shape or state.get("count") != count:
        state.clear()
    queue = state.setdefault("queue", deque())
    acc = value + decay * state.get("acc", 0)
    if len(queue) == count:
        acc -= decay**count * queue.popleft()
    queue.append(value.copy())
    state.update(acc=acc, index=f.index, shape=value.shape, count=count)
    return np.maximum(acc, 0)


def normal_flow(before: Any, after: Any) -> tuple[Any, Any, Any]:
    middle = (before + after) / 2
    gy, gx = np.gradient(middle)
    dt = after - before

    def avg(v: Any) -> Any:
        return C.gaussian(v[None], 2)[0]

    a, b, c = avg(gx * gx), avg(gx * gy), avg(gy * gy)
    d, e = -avg(gx * dt), -avg(gy * dt)
    det = a * c - b * b
    u = np.clip((c * d - b * e) / (det + 1e-6), -12, 12)
    v = np.clip((a * e - b * d) / (det + 1e-6), -12, 12)
    return u, v, np.sqrt(gx * gx + gy * gy)


@cell_kernel("motion_ink")
def motion_ink(x: Any, p: dict[str, Any], f: Frame) -> Any:
    _, h, w = x.shape
    scale = min(1, 320 / w)
    small = C.resize(x, max(4, round(h * scale)), max(4, round(w * scale)))
    tone = C.lum(small)
    previous = f.state.get("previous")
    f.state["previous"] = tone.copy()
    if (
        previous is None
        or previous.shape != tone.shape
        or f.state.get("last_index", f.index - 1) != f.index - 1
    ):
        f.state.pop("trace", None)
        f.state["last_index"] = f.index
        return x * float(p.get("source", 0))
    f.state["last_index"] = f.index
    if np.abs(tone - previous).mean() > 0.3:
        f.state.pop("trace", None)
        return x * float(p.get("source", 0))
    u, v, edge = normal_flow(previous, tone)
    if p.get("camera_compensate", True):
        u -= np.median(u)
        v -= np.median(v)
    difference = np.abs(tone - previous)
    threshold = 0.006 + 0.02 * float(p.get("noise_reject", 0.62))
    activity = C.smooth(threshold, threshold + 0.07, difference + edge * np.hypot(u, v))
    activity *= 0.1 + 0.5 * float(p.get("gain", 0.48))
    angle = np.arctan2(v, u)
    color = np.stack([0.5 + 0.5 * np.cos(angle + a) for a in (0, 2.0944, 4.1888)])
    color = (
        0.15 + 0.85 * color
        if p.get("direction_color", True)
        else np.broadcast_to(C.rgb(p.get("ink", "#d8ff3e")), small.shape)
    )
    marks = color * activity[None]
    seconds = float(p.get("trail_seconds", 1.1))
    count = max(1, round(seconds * f.fps))
    decay = math.exp(-4 / max(1, count))
    memory = temporal(marks, f, "trace", decay, count)
    out = C.resize(np.maximum(marks, memory * 0.1), h, w)
    material = p.get("material", "dither")
    if material != "smooth":
        unit = max(
            1, round(float(p.get("cell_size", 9)) * w / float(p.get("reference_width", 1200)) / 3)
        )
        yy, xx = np.mgrid[:h, :w]
        if material == "dither":
            coverage = np.clip(
                (C.lum(out) * 2 - _BAYER[(yy // unit) % 4, (xx // unit) % 4]) * 4 + 0.5, 0, 1
            )
        elif material == "halftone":
            coverage = (xx % (unit * 3) - unit) ** 2 + (yy % (unit * 3) - unit) ** 2 < unit**2
        else:
            g = C.grid(out, {**p, "cell_size": max(12, p.get("cell_size", 9))}, f, 0.6)
            coverage = glyphs(g, np.floor(g.tone * 9), " .:-=+*#%@", sorted_ink=True)
        out *= coverage[None]
    return out + x * float(p.get("source", 0))


@cell_kernel("brand_trace")
def brand_trace(x: Any, p: dict[str, Any], f: Frame) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    _, h, w = x.shape
    g = C.grid(x, p, f)
    score = g.tone + C.noise(g.cy, g.cx, int(p.get("seed", 0))) * 0.05
    order = np.argsort(score.ravel())[::-1]
    overlay = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(overlay)
    points = []
    label = p.get("label_text", "")
    font = ImageFont.truetype(str(FONT), max(6, round(h / 60))) if label else None
    radius = max(2, float(p.get("radius", 12)) * w / float(p.get("reference_width", 1200)))
    for index in order:
        iy, ix = np.unravel_index(index, score.shape)
        if g.tone[iy, ix] < float(p.get("threshold", 0.3)):
            break
        px, py = (ix + 0.5) * g.cw, (iy + 0.5) * g.ch
        if any((px - a) ** 2 + (py - b) ** 2 < (radius * 3) ** 2 for a, b in points):
            continue
        color = tuple((np.clip(g.colors[:, iy, ix], 0, 1) * 255).astype(np.uint8))
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            outline=color,
            width=max(1, round(w / 1000)),
        )
        if label:
            draw.text((px + radius + 2, py), label, font=font, fill=color, anchor="lm")
        if points and p.get("connections", True):
            near = min(points, key=lambda q: (q[0] - px) ** 2 + (q[1] - py) ** 2)
            draw.line((near, (px, py)), fill=color, width=1)
        points.append((px, py))
        if len(points) >= int(p.get("max_shapes", 50)):
            break
    return (
        x * float(p.get("source", 0.85)) + np.asarray(overlay, np.float32).transpose(2, 0, 1) / 255
    )


@cell_kernel("thermal_reveal")
def thermal_reveal(x: Any, p: dict[str, Any], f: Frame) -> Any:
    colors = [
        "#b9a2f2",
        "#cf8eea",
        "#dc8fd6",
        "#e692bc",
        "#ec9aa8",
        "#efab9c",
        "#f3bd93",
        "#fbc985",
        "#f6efe6",
        "#bbede8",
        "#bbe4f1",
        "#bfccf3",
    ]
    y = C.lum(C.gaussian(x, float(p.get("soften", 16)) * x.shape[2] / 1200))
    phase = ((1 - y) * float(p.get("cycles", 1.2)) + float(p.get("phase", 0.67))) % 1
    stops = np.array([C.rgb(c)[:, 0, 0] for c in colors])
    pos = phase * (len(stops) - 1)
    lo = np.floor(pos).astype(int)
    color = (
        stops[lo] + (stops[np.minimum(lo + 1, len(stops) - 1)] - stops[lo]) * (pos - lo)[..., None]
    ).transpose(2, 0, 1)
    progress = float(p.get("progress", 0))
    if p.get("automatic", True):
        t = C.clock(p, f) / float(p.get("duration", 3.433333))
        progress = float(np.interp(t, [0, 0.2913, 0.4951, 0.7476, 1], [0, 1.13, 1.13, 2, 2]))
    xx = np.linspace(0, 1, x.shape[2])[None]
    soft = float(p.get("soft", 0.18))
    order = 0.1 + xx * 0.7 + (1 - y) * 0.2
    enter = C.smooth(-soft, soft, progress - order)
    leave = C.smooth(-soft, soft, progress - 1 - order)
    amount = np.clip(enter - leave, 0, 1)
    if progress <= 0 or progress >= 2:
        amount *= 0
    return x + (color - x) * amount[None]


def dot_screen(x: Any, cell: float, angle: float, scale: float) -> Any:
    h, w = x.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    a = math.radians(angle)
    u = xx * math.cos(a) + yy * math.sin(a)
    v = -xx * math.sin(a) + yy * math.cos(a)
    px, py = u % cell - cell / 2, v % cell - cell / 2
    cx = np.floor(u / cell) * cell + cell / 2
    cy = np.floor(v / cell) * cell + cell / 2
    tone = C.sample(
        C.OPS, x[None], cx * math.sin(a) + cy * math.cos(a), cx * math.cos(a) - cy * math.sin(a)
    )[0]
    radius = np.sqrt(np.clip(tone, 0, 1)) * cell * 0.48 * scale
    return np.clip(radius - np.hypot(px, py) + 0.5, 0, 1)


@cell_kernel("halftone_cells")
def halftone_cells(x: Any, p: dict[str, Any], f: Frame) -> Any:
    tone = np.clip((C.lum(x) - 0.12) / 0.88, 0, 1) ** 1.8
    cell = max(
        2, float(p.get("cell_size", 22)) * x.shape[2] / float(p.get("reference_width", 1200))
    )
    cover = dot_screen(tone, cell, float(p.get("angle", 64)), float(p.get("dot_scale", 1.17)))
    out = paper(p) + (C.rgb(p.get("ink", "#ffb54d")) - paper(p)) * cover[None] * np.sqrt(tone)[None]
    hot = np.maximum(out - 0.45, 0)
    return (
        out
        + float(p.get("edge_glow", 0.53)) * C.gaussian(hot, max(0.5, cell / 5))
        + hot * float(p.get("dot_light", 0.37))
    )


@cell_kernel("riso_cells")
def riso_cells(x: Any, p: dict[str, Any], f: Frame) -> Any:
    _, h, w = x.shape
    scale = w / float(p.get("reference_width", 1200))
    cell = max(2, float(p.get("cell_size", 12)) * scale)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    shifted = C.sample(
        C.OPS,
        x,
        yy - float(p.get("drift_y", 13.5)) * scale,
        xx - float(p.get("drift_x", 8)) * scale,
    )
    first = dot_screen(1 - x[0], cell, 15, 1.7)
    second = dot_screen(1 - shifted[2], cell, 75, 1.7)
    bleed = float(p.get("bleed", 0.65))
    first = np.clip(first + C.gaussian(first[None], max(0.1, scale * bleed))[0] * bleed * 0.2, 0, 1)
    second = np.clip(
        second + C.gaussian(second[None], max(0.1, scale * bleed))[0] * bleed * 0.2, 0, 1
    )
    out = paper(p) * (1 - first[None] * (1 - C.rgb(p.get("ink", "#ff453a"))))
    out *= 1 - second[None] * (1 - C.rgb(p.get("ink2", "#1d4ed8")))
    fibre = C.noise(yy // max(1, round(scale)), xx, int(p.get("seed", 0))) - 0.5
    return out * (1 + float(p.get("fibre", 0.7)) * fibre[None] * 0.15)


@cell_kernel("cell_finish")
def cell_finish(x: Any, p: dict[str, Any], f: Frame) -> Any:
    _, h, w = x.shape
    q = min(1, 480 / max(h, w))
    small = C.resize(x, max(2, round(h * q)), max(2, round(w * q)))
    level = C.lum(small)
    threshold, knee = float(p.get("threshold", 0.35)), float(p.get("knee", 0.5))
    light = small * C.smooth(threshold - knee / 2, threshold + knee / 2, level)[None]
    bloom = 0 * light
    for radius, gain in ((1, 0.5), (3, 0.3), (9, 0.15), (24, 0.05)):
        bloom += C.gaussian(light, radius * float(p.get("radius", 1.8))) * gain
    finish = bloom * float(p.get("bloom", 0.6)) * C.rgb(p.get("bloom_tint", "#ffffff"))
    if float(p.get("streaks", 0)) > 0:
        a = math.radians(float(p.get("angle", 18)))
        length = float(p.get("length", 0.42)) * small.shape[2]
        yy, xx = np.mgrid[: small.shape[1], : small.shape[2]].astype(np.float32)
        ray = 0 * light
        for distance in np.linspace(0, length, 22):
            ray += C.sample(
                C.OPS, light, yy - math.sin(a) * distance, xx - math.cos(a) * distance, "zero"
            ) * math.exp(-3 * distance / max(1, length))
        finish += ray * C.rgb(p.get("tint", "#ff6beb")) * float(p.get("streaks", 0)) * 0.25
    out = x + C.resize(finish, h, w)
    decay = float(p.get("trails", 0))
    if decay:
        memory = temporal(small, f, "finish", decay, 180)
        out = np.maximum(out, C.resize(memory * (1 - decay), h, w))
    out *= 2 ** float(p.get("exposure", 0))
    y = C.lum(out)[None]
    return y + (out - y) * float(p.get("saturation", 1))


@cell_kernel("cell_vhs")
def cell_vhs(x: Any, p: dict[str, Any], f: Frame) -> Any:
    filtered = KERNELS["vhs"](C.OPS, x, p, f)
    return x + filtered if p.get("layer_mode") == "add" else filtered
