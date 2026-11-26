"""Frame kernels of the FX-C effects (stylized and censor). Same rules as ``fx_kernels``: imports
nothing from Cutwright (it runs in the objects environment too), numpy or torch through ``o``,
randomness from numpy so both backends draw the same."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

if __package__:  # in Cutwright's process
    from cutwright.modules.fx.runners.fx_base import (
        LUMA,
        REGION,
        Frame,
        blur,
        downscale,
        hblur,
        hex_rgb,
        kernel,
        luma,
        pixel,
        sample,
        shifted,
    )
    from cutwright.modules.fx.runners.fx_kernels_paint import (
        _table,
        pattern_grid,
        pattern_layers,
        texture,
    )
else:  # in the objects environment: the files next to the runner, by their bare names
    from fx_base import (  # type: ignore[no-redef]
        LUMA,
        REGION,
        Frame,
        blur,
        downscale,
        hblur,
        hex_rgb,
        kernel,
        luma,
        pixel,
        sample,
        shifted,
    )
    from fx_kernels_paint import _table, pattern_grid, pattern_layers, texture  # type: ignore[no-redef]

FONT = Path(__file__).resolve().parents[3] / "assets" / "fonts" / "Inter-Bold.ttf"


# --- censor ------------------------------------------------------------------------------------


@kernel("mosaic")
def mosaic(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Block means on a grid from the top-left corner (the frame mirrored out to a multiple of
    the block), each block one colour."""
    _, h, w = x.shape
    b = max(2, round(float(p["block"]) * min(h, w)))
    ph, pw = (-h) % b, (-w) % b
    grown = o.pad(x, ph, pw)[:, ph:, pw:] if ph or pw else x
    small = downscale(o, grown, b)
    ys, xs = o.grid(h, w)
    return o.gather(small, ys // b, xs // b)


@pixel("bar")
def bar(xp: Any, x: Any, p: dict[str, Any]) -> Any:
    """The colour over the picture by ``opacity``."""
    k = float(p["opacity"])
    return xp.stack([x[i] + (c - x[i]) * k for i, c in enumerate(hex_rgb(p["color"]))])


@kernel("stripes")
def stripes(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Bands of ``width`` px every ``width + gap`` px (rows or columns), hard edged."""
    _, h, w = x.shape
    short = min(h, w)
    on, off = max(1, round(float(p["width"]) * short)), max(1, round(float(p["gap"]) * short))
    ys, xs = o.grid(h, w)
    band = ((ys if p["direction"] == "horizontal" else xs) % (on + off)) < on
    color = o.asarray(hex_rgb(p["color"]))[:, None, None]
    return x + (color - x) * (band[None] * float(p["opacity"]))


@kernel("manga")
def manga(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Thin horizontal slices moved sideways by whole pixels, neighbours the other way; the
    pattern is the effect's own (seeded), so it does not flicker."""
    _, h, w = x.shape
    tall = max(1, round(float(p["slice"]) * min(h, w)))
    n = h // tall + 1
    rng = np.random.default_rng(f.seed)
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    offsets = np.round(sign * float(p["shift"]) * w * rng.uniform(0.4, 1.0, n))
    row = o.asarray(offsets[np.arange(h) // tall])[:, None]
    ys, xs = o.grid(h, w)
    return o.gather(x, ys, (xs - row).clip(0, w - 1))


@kernel("frosted")
def frosted(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Frosted glass: blurred, lifted towards white, with a fine grain (seeded; where
    ``pattern`` puts it: with the picture by default, so it does not sit on the screen)."""
    _, h, w = x.shape
    soft = blur(o, x, max(0.5, float(p["blur"]) * min(h, w)))
    out = soft + (1 - soft) * float(p["frost"])
    # ~0.6 px grain at a spread of 0.47 (white noise blurred by 0.6 px, as it was).
    grain = texture(o, pattern_layers(o, x, p, f), _table(f.seed, 512), 0.75, (h, w)) * 0.47
    return (out + 0.08 * float(p["grain"]) * grain[None]).clip(0, 1)


def _shape(name: str, color: str, side: int = 512) -> Any:
    """A built-in sticker drawn with Pillow (no third-party images): float RGBA (4, s, s)."""
    from PIL import Image, ImageDraw

    big = side * 2
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fill = (*(round(v * 255) for v in hex_rgb(color)), 255)
    dark = (40, 30, 20, 255)
    m = big * 0.04
    if name in ("circle", "smile"):
        d.ellipse((m, m, big - m, big - m), fill=fill)
    if name == "smile":
        e = big * 0.09
        for cx in (0.36, 0.64):
            d.ellipse((big * cx - e * 0.6, big * 0.36 - e, big * cx + e * 0.6, big * 0.36 + e),
                      fill=dark)  # fmt: skip
        d.arc((big * 0.27, big * 0.3, big * 0.73, big * 0.76), 20, 160, fill=dark,
              width=round(big * 0.05))  # fmt: skip
    elif name == "star":
        pts = [(big / 2 + (0.47 if i % 2 == 0 else 0.2) * big * math.sin(i * math.pi / 5),
                big * 0.53 - (0.47 if i % 2 == 0 else 0.2) * big * math.cos(i * math.pi / 5))
               for i in range(10)]  # fmt: skip
        d.polygon(pts, fill=fill)
    elif name == "heart":
        ts = np.linspace(0, 2 * math.pi, 120)
        hx, hy = 16 * np.sin(ts) ** 3, 13 * np.cos(ts) - 5 * np.cos(2 * ts) - 2 * np.cos(3 * ts) - (
            np.cos(4 * ts))  # fmt: skip
        d.polygon([(big / 2 + big * 0.028 * a, big * 0.47 - big * 0.028 * b)
                   for a, b in zip(hx, hy, strict=True)], fill=fill)  # fmt: skip
    img = img.resize((side, side), Image.Resampling.LANCZOS)
    return np.asarray(img, np.float32).transpose(2, 0, 1) / 255


def _sticker_image(p: dict[str, Any]) -> Any:
    if p.get("image"):
        from PIL import Image

        with Image.open(p["image"]) as img:
            return np.asarray(img.convert("RGBA"), np.float32).transpose(2, 0, 1) / 255
    return _shape(p["shape"], p["color"])


@kernel("sticker")
def sticker(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """A sticker over the scope mask's region (its bounding box at this frame, so it follows a
    tracked, object, face or person mask), covering it × ``scale``; without a mask at ``center``
    and ``size``. Where the mask is empty (the object is lost) nothing is drawn."""
    _, h, w = x.shape
    if f.mask is not None:
        inside = np.asarray(o.numpy(f.mask)) > 0.5
        rows, cols = np.nonzero(inside.any(1))[0], np.nonzero(inside.any(0))[0]
        if not len(rows):
            return x
        cx, cy = (cols[0] + cols[-1] + 1) / 2, (rows[0] + rows[-1] + 1) / 2
        bw, bh = cols[-1] - cols[0] + 1, rows[-1] - rows[0] + 1
    else:
        cx, cy = float(p["center"][0]) * w, float(p["center"][1]) * h
        bw = bh = float(p["size"]) * min(h, w)
    key = (p.get("image"), p["shape"], p["color"])
    if f.state.get("key") != key:
        f.state.update(key=key, image=_sticker_image(p), sized={})
    image = f.state["image"]
    k = max(bw / image.shape[2], bh / image.shape[1]) * float(p["scale"])
    sw, sh = max(1, round(image.shape[2] * k)), max(1, round(image.shape[1] * k))
    sized = f.state["sized"]
    if (sw, sh) not in sized:
        from PIL import Image

        rgba = Image.fromarray((image.transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8))
        small = rgba.resize((sw, sh), Image.Resampling.LANCZOS)
        if len(sized) > 64:
            sized.clear()
        sized[(sw, sh)] = np.asarray(small, np.float32).transpose(2, 0, 1) / 255
    tile = sized[(sw, sh)]
    left, top = round(cx - sw / 2), round(cy - sh / 2)
    x0, y0, x1, y1 = max(0, left), max(0, top), min(w, left + sw), min(h, top + sh)
    if x0 >= x1 or y0 >= y1:
        return x
    part = o.asarray(tile[:, y0 - top : y1 - top, x0 - left : x1 - left])
    alpha = part[3:4] * float(p["opacity"])
    out = x * 1
    out[:, y0:y1, x0:x1] = x[:, y0:y1, x0:x1] * (1 - alpha) + part[:3] * alpha
    return out


REGION.add("sticker")


# --- stylized ----------------------------------------------------------------------------------


def _ramp(o: Any, phase: Any, p: dict[str, Any]) -> Any:
    """The acid palette at ``phase`` (turns): a full rainbow, or the neon colours in a loop."""
    xp = o.xp
    t = phase - xp.floor(phase)
    if p["palette"] == "rainbow":
        h6 = t * 6
        return xp.stack([(xp.abs(h6 - 3) - 1).clip(0, 1), (2 - xp.abs(h6 - 2)).clip(0, 1),
                         (2 - xp.abs(h6 - 4)).clip(0, 1)])  # fmt: skip
    colors = [hex_rgb(c) for c in p["colors"]]
    n = len(colors)
    pos = t * n
    out = 0 * xp.stack([t, t, t])
    for i, (a, b) in enumerate(zip(colors, colors[1:] + colors[:1], strict=True)):
        k = (pos - i).clip(0, 1)
        k = k * k * (3 - 2 * k)  # smooth between neighbours
        inside = ((pos >= i) & (pos < i + 1)) * 1.0
        out = out + inside * xp.stack([a[c] + (b[c] - a[c]) * k for c in range(3)])
    return out


def trail_frames(p: dict[str, Any]) -> int:
    return round(8 * float(p.get("trail", 0)))


@kernel("acid")
def acid(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Acid colour: luma (solarized by ``solarize``) mapped onto a palette that repeats
    ``cycles`` times from black to white and turns ``speed`` times a second; optional swirl of
    the picture and a trail of the frames before (``trail``: up to 8 frames, weights trail^j)."""
    _, h, w = x.shape
    xp = o.xp
    t = f.index / f.fps
    swirl = float(p.get("swirl", 0))
    if swirl > 0:
        ys, xs = o.grid(h, w)
        dx, dy = xs - (w - 1) / 2, ys - (h - 1) / 2
        r = xp.sqrt(dx * dx + dy * dy) / (0.5 * math.hypot(w, h))
        breathe = 1 + 0.3 * math.sin(0.4 * math.pi * t)
        turn = swirl * 2 * math.pi * (1 - r).clip(0, 1) ** 2 * breathe
        c, s = xp.cos(turn), xp.sin(turn)
        x = sample(o, x, (h - 1) / 2 + dx * s + dy * c, (w - 1) / 2 + dx * c - dy * s)
    y = luma(x).clip(0, 1)
    sol = float(p["solarize"])
    y = y + sol * ((y > 0.5) * (1 - 2 * y))  # above mid-grey: flipped
    colour = _ramp(o, float(p["cycles"]) * y + float(p["speed"]) * t, p) * (0.55 + 0.45 * y)[None]
    grey = luma(colour)[None]
    fresh = (grey + float(p["saturation"]) * (colour - grey)).clip(0, 1)
    k = trail_frames(p)
    if k:
        past = [img for i, img in f.state.get("trail", []) if f.index - k <= i < f.index]
        weights = [float(p["trail"]) ** j for j in range(len(past) + 1)]
        mixed = fresh * weights[0]
        for wgt, img in zip(weights[1:], reversed(past), strict=True):
            mixed = mixed + img * wgt
        f.state["trail"] = [*f.state.get("trail", [])[-(k - 1) :], (f.index, fresh)] if k > 1 else [
            (f.index, fresh)]  # fmt: skip
        fresh = mixed / sum(weights)
    return x + float(p["amount"]) * (fresh - x)


def _tear_rows(rng: Any, h: int, w: int, p: dict[str, Any]) -> tuple[Any, int]:
    """Per-row sideways moves of the bottom band (head switching): growing towards the last
    row, one direction per frame; and the band's height in rows."""
    rows = np.zeros(h, np.float32)
    tracking = float(p["tracking"])
    if tracking <= 0:
        return rows, 0
    band = max(2, round(h * (0.015 + 0.045 * tracking)))
    reach = tracking * w * rng.uniform(0.025, 0.05) * rng.choice((-1.0, 1.0))
    rows[h - band :] = reach * ((np.arange(band) + 1) / band) ** 1.5
    return rows, band


def _dashes(rng: Any, o: Any, h: int, w: int, density: float, length: float) -> Any:
    """White horizontal dashes (tape dropouts): sparse points smeared along the line."""
    dots = (rng.random((h, w)) > 1 - density).astype(np.float32)
    return hblur(o, o.asarray(dots)[None], length) * (2.5 * length)


@kernel("vhs")
def vhs(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """VHS / NTSC: luma soft (about 330 lines across) with ringing at edges, chroma (B−Y, R−Y)
    blurred to a sixth of that and delayed along the line, desaturated; row noise, streaky
    grain and dropout dashes; the bottom rows torn sideways by the head switch, with noise in
    the band; vertical jitter and a slight line wobble; an OSD drawn crisp over it."""
    _, h, w = x.shape
    xp = o.xp
    rng = np.random.default_rng([f.seed, f.index])
    t = f.index / f.fps
    s = h / 480
    jitter = float(p["jitter"])
    wobble = jitter * 0.8 * s * np.sin(2 * math.pi * (np.arange(h) / h * 3 + 0.7 * t
                                                      + rng.random()))  # fmt: skip
    dy = jitter * s * rng.uniform(-1, 1)
    tear, band = _tear_rows(rng, h, w, p)
    dx = (wobble + tear).astype(np.float32)
    if dy or dx.any():
        ys, xs = o.grid(h, w)
        x = sample(o, x, ys - dy, xs - o.asarray(dx)[:, None])
    y = luma(x)[None]
    u, v = x[2:3] - y, x[0:1] - y
    soft = blur(o, hblur(o, y, max(0.3, 0.0028 * w)), 0.6 * s)  # tape luma, a touch vertical
    y = soft + 1.5 * float(p["sharpen"]) * (soft - hblur(o, soft, max(0.6, 0.006 * w)))
    y = 0.05 + 0.9 * y  # lifted blacks, dulled whites
    bleed = float(p["chroma_blur"])
    uv = hblur(o, xp.stack([u[0], v[0]]), (0.006 + 0.022 * bleed) * w)
    uv = shifted(o, uv, 0.012 * w * bleed, 0) * float(p["saturation"])
    noise = float(p["noise"])
    if noise > 0:
        rows = rng.standard_normal(h).astype(np.float32)[:, None] * 0.03 * noise
        grain = hblur(o, o.asarray(rng.standard_normal((1, h, w)).astype(np.float32)), 1.2 * s)
        y = y + o.asarray(rows)[None] + grain * 0.1 * noise
        y = y + _dashes(rng, o, h, w, 12 * noise / (480 * w), 4 * s)  # a dozen or so
        streaks = rng.standard_normal((2, h, w)).astype(np.float32) * 0.3 * noise
        uv = uv + hblur(o, o.asarray(streaks), 6 * s)  # chroma noise smeared along the lines
    color_noise = float(p.get("color_noise", 0))
    if color_noise:
        rows = np.arange(h, dtype=np.float32) / max(1, h)
        bands = np.stack([np.sin(rows * 39 + t * 4 + phase) for phase in (0, 2.1)])
        uv = uv + o.asarray(bands[:, :, None]) * color_noise * 0.18
    if band:  # the head-switch band: torn, brighter, full of dropouts
        tracking = float(p["tracking"])
        hit = _dashes(rng, o, band, w, 0.012 * tracking, 6 * s)[0] + 0.05 * tracking
        lift = 0 * y[0]
        lift[h - band :] = hit  # (torch has no concatenate before 2.1)
        y = y + lift[None]
    y = y[0]
    r, b = uv[1] + y, uv[0] + y
    g = (y - LUMA[0] * r - LUMA[2] * b) / LUMA[1]
    out = xp.stack([r, g, b]).clip(0, 1)
    if p.get("osd"):
        alpha = _osd(o, f, h, w, p)
        if p.get("osd_style") == "deck":
            out = out * (1 - shifted(o, alpha, max(1, w / 320), max(1, h / 480)) * 0.8)
        out = out * (1 - alpha) + alpha
    return out


def _osd(o: Any, f: Frame, h: int, w: int, p: dict[str, Any]) -> Any:
    """The deck's text as an alpha layer: PLAY ▶ and SP top-left, the tape counter top-right,
    the date and time bottom-left (bundled Inter Bold; cached while the text stays)."""
    seconds = int(f.index / f.fps)
    counter = f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
    key = (
        h,
        w,
        counter,
        p.get("osd_text", ""),
        p.get("osd_style", "classic"),
        p.get("osd_label", "PLAY"),
        p.get("osd_counter", True),
    )
    if f.state.get("osd_key") != key:
        from PIL import Image, ImageDraw, ImageFont

        size = max(8, round(h / 15))
        try:
            font_path = (
                FONT.parent / "JetBrainsMono-Bold.ttf" if p.get("osd_style") == "deck" else FONT
            )
            font = ImageFont.truetype(str(font_path), size)
        except OSError:
            font = ImageFont.load_default()
        deck = p.get("osd_style") == "deck"
        canvas_w = round(w / 0.8) if deck else w
        img = Image.new("L", (canvas_w, h), 0)
        d = ImageDraw.Draw(img)
        m, gap = round(0.06 * h), round(1.4 * size)
        left = round(0.05 * canvas_w)
        label = p.get("osd_label", "PLAY")
        d.text((left, m), label, fill=255, font=font)
        tri = left + d.textlength(label + " ", font=font)
        d.polygon([(tri, m + 0.15 * size), (tri, m + 0.95 * size),
                   (tri + 0.7 * size, m + 0.55 * size)], fill=255)  # fmt: skip
        if p.get("osd_style", "classic") == "classic":
            d.text((left, m + gap), "SP", fill=255, font=font)
        if p.get("osd_counter", True):
            d.text(
                (canvas_w - left - d.textlength(counter, font=font), m),
                counter,
                fill=255,
                font=font,
            )
        lines = str(p.get("osd_text", "")).count("\n") + 1
        d.multiline_text(
            (left, h - m - size * 1.2 * lines),
            str(p.get("osd_text", "")),
            fill=255,
            font=font,
            spacing=0,
        )
        if deck:
            img = img.resize((max(1, w // 2), max(1, h // 2)), Image.Resampling.NEAREST)
            img = img.resize((w, h), Image.Resampling.NEAREST)
        layer = np.asarray(img, np.float32)[None] / 255
        f.state["osd_key"], f.state["osd"] = key, hblur(o, o.asarray(layer), 0.4 * w / 640)
    return f.state["osd"]


def strong_chance(p: dict[str, Any]) -> float:
    """The share of disturbed frames that tear hard (the rest flicker subtly)."""
    return 0.1 + 0.9 * float(p["amount"]) ** 2


def _glitched(o: Any, x: Any, p: dict[str, Any], rng: Any) -> Any:
    """A disturbed frame (``frequency`` of them): most are subtle (a few thin slices nudged,
    a slight RGB split), some strong (``slices`` tall tears, the full split, blocks)."""
    _, h, w = x.shape
    if rng.random() >= float(p.get("frequency", 0.9)):
        return x
    strong = rng.random() < strong_chance(p)
    k = 1.0 if strong else 0.3
    rows = np.zeros(h, np.float32)
    tall_range = (0.01, 0.08) if strong else (0.004, 0.02)
    for _ in range(int(p["slices"]) if strong else int(rng.integers(1, 4))):
        tall = max(2, round(h * rng.uniform(*tall_range)))
        top = int(rng.integers(0, max(1, h - tall + 1)))
        move = rng.uniform(0.3, 1) * rng.choice((-1.0, 1.0)) * k * float(p["shift"]) * 0.25 * w
        rows[top : top + tall] = round(move) or (1 if move > 0 else -1)
    ys, xs = o.grid(h, w)
    out = o.gather(x, ys, (xs - o.asarray(rows)[:, None]) % w)
    split = float(p["rgb"]) * 0.02 * w * (1.0 if strong else 0.5)
    if split > 0:
        dr, db = round(rng.uniform(0.3, 1) * split), -round(rng.uniform(0.3, 1) * split)
        out = o.xp.stack([o.gather(out[0:1], ys, (xs - dr) % w)[0], out[1],
                          o.gather(out[2:3], ys, (xs - db) % w)[0]])  # fmt: skip
    n = int(rng.poisson(float(p["blocks"]) * 10)) if strong else 0
    if n:
        out = out * 1
        unit = max(4, round(min(h, w) / 45))
        for _ in range(n):
            b = min(int(rng.choice((2, 4, 8))) * unit, h // 2, w // 2)
            y0, x0 = int(rng.integers(0, h - b + 1)), int(rng.integers(0, w - b + 1))
            sy, sx = int(rng.integers(0, h - b + 1)), int(rng.integers(0, w - b + 1))
            out[:, y0 : y0 + b, x0 : x0 + b] = out[:, sy : sy + b, sx : sx + b][[2, 0, 1]] * 1
    return out


def _repeats(seed: int, index: int, chance: float) -> bool:
    """Frame ``index`` repeats the one before (never two in a row, decided by the numbers
    alone, so a still and the render agree)."""

    def hit(i: int) -> bool:
        return i >= 0 and bool(np.random.default_rng([seed, i, 1]).random() < chance)

    return chance > 0 and hit(index) and not hit(index - 1)


@kernel("glitch")
def glitch(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Glitch, seeded per frame: horizontal slices torn sideways (wrapping), red and blue
    pulled apart, blocks copied from elsewhere with their channels swapped; now and then a
    frame repeats the one before."""
    fresh = _glitched(o, x, p, np.random.default_rng([f.seed, f.index]))
    before = f.state.get("prev")
    f.state["prev"] = (f.index, fresh)
    repeat = _repeats(f.seed, f.index, float(p.get("repeat", 0)))
    return before[1] if repeat and before is not None and before[0] == f.index - 1 else fresh


@kernel("pixel_sort")
def pixel_sort(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Each run of pixels whose luma lies in [low, high] along a row (or column) is sorted by
    luma; the rest stay put. Keys are exact in float32: run number × 2048 + 10-bit luma."""
    vertical = p["direction"] == "vertical"
    lines = x.swapaxes(1, 2) if vertical else x
    y = luma(lines).clip(0, 1)
    band = ((y >= float(p["low"])) & (y <= float(p["high"]))) * 1.0
    before = shifted(o, band[None], 1, 0)[0]  # the pixel to the left is in the band too
    run = o.cumsum(1 - band * before)
    level = o.xp.floor(y * 1023)
    key = run * 2048 + (1023 - level if p.get("reverse") else level)
    out = o.take(lines, o.argsort(key))
    return out.swapaxes(1, 2) if vertical else out


_SPOT: dict[str, Any] = {}


def _thresholds() -> Any:
    """Spot-function levels so that ink covers exactly c of a cell: ``T[round(256 c)]`` (the
    Euclidean dot: round dots that meet in a checker at 50 %)."""
    if "t" not in _SPOT:
        g = (np.arange(128) + 0.5) / 128 - 0.5
        s = ((np.cos(2 * np.pi * g)[:, None] + np.cos(2 * np.pi * g)[None, :]) / 2).ravel()
        _SPOT["t"] = np.quantile(s, 1 - np.linspace(0, 1, 257)).astype(np.float32)
    return _SPOT["t"]


def _screen(o: Any, cover: Any, angle: float, cell: float, coords: Any) -> Any:
    """Ink (0–1) of one halftone screen laid at ``coords`` (``pattern_grid``): each cell's dot
    sized by the coverage at its centre."""
    xp = o.xp
    _, h, w = cover.shape
    gy, gx = o.grid(h, w)
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    table = o.asarray(_thresholds())[None, None]
    u, v = coords[1] * c + coords[0] * s, -coords[1] * s + coords[0] * c
    cu, cv = (xp.floor(u / cell) + 0.5) * cell, (xp.floor(v / cell) + 0.5) * cell
    du, dv = cu - u, cv - v  # to the dot's centre, turned back onto the frame
    k = sample(o, cover, gy + du * s + dv * c, gx + du * c - dv * s)[0].clip(0, 1)
    spot = (xp.cos(2 * math.pi * du / cell) + xp.cos(2 * math.pi * dv / cell)) / 2
    level = o.gather(table, 0 * k, xp.floor(k * 256 + 0.5))[0]
    ink = ((spot - level) * cell / math.pi + 0.5).clip(0, 1)
    return xp.maximum(ink * (k * 20).clip(0, 1), ((k - 0.95) * 20).clip(0, 1))


@kernel("halftone")
def halftone(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Halftone: CMYK screens at 15°, 75°, 0°, 45° (inks multiply on the paper) or one ink at
    ``angle``; a dot's area is the darkness (coverage) at its cell's centre; the screens lie
    where ``pattern`` puts them (with the picture by default)."""
    _, h, w = x.shape
    cell = max(3.0, float(p["size"]) * min(h, w))
    soft = blur(o, x, cell / 3)
    paper = o.asarray(hex_rgb(p["paper"]))[:, None, None]
    grid = pattern_grid(o, x, p, f)
    if p["mode"] == "mono":
        ink = _screen(o, 1 - luma(soft)[None], float(p["angle"]), cell, grid)[None]
        color = o.asarray(hex_rgb(p["ink"]))[:, None, None]
        return paper + (color - paper) * ink
    black = 1 - o.xp.maximum(o.xp.maximum(soft[0], soft[1]), soft[2])
    rest = (1 - black + 1e-6)[None]
    cmy = ((1 - soft - black[None]) / rest).clip(0, 1)
    out = paper * 1
    for i, (angle, col) in enumerate(((15, (0, 1, 1)), (75, (1, 0, 1)), (0, (1, 1, 0)))):
        a = _screen(o, cmy[i : i + 1], angle, cell, grid)[None]
        out = out * (1 - a * (1 - o.asarray(col)[:, None, None]))
    return out * (1 - _screen(o, black[None], 45, cell, grid)[None])


@kernel("posterize")
def posterize(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Each channel down to ``levels`` steps (as the 16-bit lutrgb: floor(v·n/65536)/(n−1))."""
    n = int(p["levels"])
    return o.xp.floor(x.clip(0, 1) * 65535 * n / 65536) / (n - 1)
