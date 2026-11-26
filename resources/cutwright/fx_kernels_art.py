"""Frame kernels of the FX-E effects that look at one frame: lens and filter emulation, film
processes, print, retro and geometry. Same rules as ``fx_kernels``: imports nothing from
Cutwright (it runs in the objects environment too), numpy or torch through ``o``, randomness from
numpy so both backends draw the same. Per-pixel colour processes are ``@pixel`` functions (the
ffmpeg path bakes them into a 3D LUT)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

if __package__:  # in Cutwright's process
    from cutwright.modules.fx.runners.fx_base import (
        Frame,
        _ray,
        blur,
        downscale,
        hblur,
        hex_rgb,
        hue_rgb,
        kernel,
        luma,
        peak,
        pixel,
        sample,
        screen,
        shifted,
        upscale,
    )
    from cutwright.modules.fx.runners.fx_constants import JITTER
    from cutwright.modules.fx.runners.fx_kernels_paint import (
        _table,
        pattern_grid,
        pattern_layers,
        texture,
    )
else:  # in the objects environment: the files next to the runner, by their bare names
    from fx_base import (  # type: ignore[no-redef]
        Frame,
        _ray,
        blur,
        downscale,
        hblur,
        hex_rgb,
        hue_rgb,
        kernel,
        luma,
        peak,
        pixel,
        sample,
        screen,
        shifted,
        upscale,
    )
    from fx_constants import JITTER  # type: ignore[no-redef]
    from fx_kernels_paint import _table, pattern_grid, pattern_layers, texture  # type: ignore[no-redef]

FONT = Path(__file__).resolve().parents[3] / "assets" / "fonts" / "Inter-Bold.ttf"


def _mix(a: Any, b: Any, k: float) -> Any:
    return a + (b - a) * k


def _scurve(v: Any, c: float) -> Any:
    """More contrast in the mid-tones, 0 and 1 kept, monotonic for c in 0–1 (1 = smoothstep)."""
    return v + c * 2 * (v - 0.5) * v * (1 - v)


def ramp(xp: Any, y: Any, colors: list[str]) -> Any:
    """``y`` (0–1) through evenly spaced colours, piecewise linear: (3, …)."""
    stops = [hex_rgb(c) for c in colors]
    n = len(stops) - 1
    out = [stops[0][ch] + 0 * y for ch in range(3)]
    for i in range(n):
        t = (y * n - i).clip(0, 1)
        out = [out[ch] + (stops[i + 1][ch] - stops[i][ch]) * t for ch in range(3)]
    return xp.stack(out)


# --- lens and filter ---------------------------------------------------------------------------


def diffusion_parts(p: dict[str, Any], short: int) -> dict[str, float]:
    """The numbers both the kernel and the ffmpeg graph use."""
    s = float(p["strength"])
    sigma = max(0.5, float(p["size"]) * short)
    return {"veil": 0.5 * s * float(p["veil"]), "sigma": sigma, "veil_sigma": 0.2 * short,
            "gain": 3.5 * s * float(p["halo"]), "t": float(p["threshold"])}  # fmt: skip


@kernel("diffusion")
def diffusion(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Diffusion filter (a black-mist look): part of the picture's light scattered wide (a
    veil: contrast drops, blacks lift) and the highlights over ``threshold`` spilling softly
    around themselves, tinted, screened over."""
    d = diffusion_parts(p, min(x.shape[1:]))
    out = _mix(x, blur(o, x, d["veil_sigma"]), d["veil"])
    over = ((x - d["t"]) / (1 - d["t"])).clip(0, 1)
    tint = o.asarray(hex_rgb(p["tint"]))[:, None, None]
    glow = (blur(o, over, d["sigma"]) * d["gain"] * tint).clip(0, 1)
    return screen(out, glow)


@kernel("anamorphic")
def anamorphic(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Anamorphic flare: points of light over ``threshold`` drawn out into a long, thin,
    horizontal streak both ways (falling off along it), coloured by ``tint`` with a little of
    the light's own colour, over a faint wide horizontal haze. Worked at up to ~360 px on the
    short side, then scaled back."""
    _, h, w = x.shape
    xp = o.xp
    t = float(p["threshold"])
    over = ((peak(o, x) - t) / (1 - t)).clip(0, 1)
    q = max(1, min(h, w) // 360)
    light, amount = downscale(o, x * over[None], q), downscale(o, over[None], q)
    s1 = max(0.5, float(p["size"]) * min(light.shape[1:]) / 6)
    near, wide = blur(o, amount, s1), blur(o, amount, 4 * s1)
    point = ((near / (wide + 1e-4) - 1.5) / 5).clip(0, 1)  # big bright areas streak less
    energy = amount * point  # as thin as the light's core
    length = float(p["length"]) * light.shape[2]

    def both(n: float) -> Any:
        return _ray(o, energy, 1, 0, n) + _ray(o, energy, -1, 0, n) - energy

    streak = 0.65 * both(length / 4) + 0.35 * both(length)  # bright near the light, long tail
    haze = hblur(o, blur(o, energy, s1), 0.12 * light.shape[2]) * 6
    own = light / (peak(o, light)[None] + 1e-4)
    tint = o.asarray(hex_rgb(p["tint"]))[:, None, None]
    colour = 0.8 * tint + 0.2 * own.clip(0, 1)
    e = (streak + haze * 0.15) * float(p["intensity"])
    soft = (1 - xp.exp(-2 * e)) * colour
    return screen(x, upscale(o, soft, h, w).clip(0, 1))


@kernel("tilt_shift")
def tilt_shift(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Tilt-shift miniature: a sharp band at ``focus`` (``band`` tall), the blur growing towards
    the top and bottom (four blur levels mixed per row), saturation and contrast lifted the
    way a toy model looks."""
    _, h, w = x.shape
    xp = o.xp
    rows = (np.arange(h) + 0.5) / h
    focus, band = float(p["focus"]), float(p["band"])
    reach = max(focus - band / 2, 1 - focus - band / 2, 1e-3)
    d = (np.clip(np.abs(rows - focus) - band / 2, 0, None) / reach) ** 1.3 * 3
    sigma = float(p["blur"]) * min(h, w)
    out = 0 * x
    for i, img in enumerate([x] + [blur(o, x, sigma * k / 3) for k in (1, 2, 3)]):
        wgt = np.clip(1 - np.abs(d - i), 0, 1).astype(np.float32)
        if wgt.any():
            out = out + img * o.asarray(wgt)[None, :, None]
    y = luma(out)[None]
    out = y + float(p["saturation"]) * (out - y)
    return xp.clip(0.5 + float(p["contrast"]) * (out - 0.5), 0, 1)


# --- film processes (per pixel: 3D LUT on the ffmpeg path) -------------------------------------


@pixel("bleach_bypass")
def bleach_bypass(xp: Any, x: Any, p: dict[str, Any]) -> Any:
    """Bleach bypass: the silver left in the print — the picture overlaid with its own luma
    (deeper shadows, harder highlights), then half its colour gone."""
    y = luma(x)
    hard = (y >= 0.5) * 1.0
    over = (1 - hard) * 2 * x * y + hard * (1 - 2 * (1 - x) * (1 - y))
    grey = luma(over)
    silver = grey + 0.45 * (over - grey)
    return _mix(x, silver, float(p["amount"]))


@pixel("cross_process")
def cross_process(xp: Any, x: Any, p: dict[str, Any]) -> Any:
    """Cross-processed slide film: red and green with hard contrast, blue flattened and lifted
    (blue-cyan shadows, yellow-green highlights), saturated."""
    r, g, b = x[0], x[1], x[2]
    out = xp.stack([_scurve(r, 1.0), 0.03 + 0.95 * _scurve(g, 0.9), 0.14 + 0.72 * b])
    return _mix(x, out, float(p["amount"]))


TONES = {
    "sepia": ["#1e130a", "#8a5a2e", "#f5e6c8"],
    "selenium": ["#16121a", "#6e5a63", "#ece4dc"],
    "tintype": ["#0e0d0c", "#5f5951", "#dcd7cb"],
}


@pixel("sepia")
def sepia(xp: Any, x: Any, p: dict[str, Any]) -> Any:
    """Toned black-and-white: luma through a sepia (warm brown), selenium (cool purple-brown)
    or tintype (dark, hard silver) ramp."""
    y = luma(x).clip(0, 1)
    if p["kind"] == "tintype":
        y = _scurve(y, 0.8)
    return _mix(x, ramp(xp, y, TONES[p["kind"]]), float(p["amount"]))


@pixel("cyanotype")
def cyanotype(xp: Any, x: Any, p: dict[str, Any]) -> Any:
    """Cyanotype: luma printed in prussian blue on white paper."""
    y = luma(x).clip(0, 1) ** 0.9
    return _mix(x, ramp(xp, y, ["#0b1f4b", "#2b5c96", "#9fb8cc", "#eef1ee"]), float(p["amount"]))


def _step(v: Any, lo: float, hi: float) -> Any:
    """Smoothstep: 0 up to ``lo``, 1 from ``hi``."""
    t = ((v - lo) / (hi - lo)).clip(0, 1)
    return t * t * (3 - 2 * t)


@pixel("infrared")
def infrared(xp: Any, x: Any, p: dict[str, Any]) -> Any:
    """Infrared film. Foliage is what reflects infrared strongly. A pixel's foliage and sky
    weights are smooth ramps on its hue (so noise and the 3D LUT stay calm), read from channel
    differences against the brightness: graded footage has olive leaves (red about equal to
    green, hue 60–90°) and a teal-grey sky (blue a few levels over green), so "green excess" and
    "blue excess" are near zero there. Foliage is hue ~55–170° (sand, rock, skin and wood are
    redder and get nothing), sky ~185–260°; a grey, pale or very dark pixel has no reliable hue
    and gets neither; a bright pale colour (cloud, wall) is no leaf. ``color``: false-colour film
    where it is foliage (infrared → red, red → green, green → blue: red-pink leaves), darker
    where it is sky (an already vivid blue goes deeper, a dull one just darker), every other
    colour kept (sand, rock and skin stay warm, never green); ``mono``: black-and-white infrared
    (white leaves, black sky, the rest by its brightness). Its LUT is baked 49 points wide
    (``art``)."""
    r, g, b = x[0], x[1], x[2]
    y = luma(x)
    top = xp.maximum(xp.maximum(r, g), b)
    chroma = top - xp.minimum(xp.minimum(r, g), b)
    d = top + 0.15  # the floor keeps a dark pixel's few levels from reading as a clear hue
    sure = _step(chroma / d, 0.045, 0.14) * _step(top, 0.03, 0.1)
    leaf = _step((g - r) / d, -0.06, 0.08) * _step((g - b) / d, 0.02, 0.1) * sure
    pale = 1 - _step(chroma / (top + 1e-4), 0.16, 0.3)  # a bright, pale colour is a cloud or a wall
    leaf = leaf * (1 - _step(top, 0.3, 0.5) * pale)  # (leaves of the same brightness are richer)
    leaf = (leaf * float(p["foliage"])).clip(0, 1)
    sky = _step((b - g) / d, -0.05, 0.04) * _step((b - r) / d, 0.0, 0.12) * sure
    if p["kind"] == "color":
        ir = (0.4 + 1.2 * y).clip(0, 1)
        out = _mix(x, xp.stack([ir, r, g]), leaf[None])
        # The sky goes darker, not bluer: an even darkening keeps a dull bluish colour (black
        # leather, a teal sky) as saturated as it was; only an already vivid blue is deepened.
        vivid = _step(chroma / d, 0.25, 0.45)
        out = out * (1 - sky * xp.stack([0.5 + 0.05 * vivid, 0.5 + 0 * vivid, 0.5 - 0.1 * vivid]))
    else:
        mono = (y + leaf * (1 - y) * 0.9) * (1 - 0.7 * sky)
        out = xp.stack([mono, mono, mono])
    return _mix(x, out, float(p["amount"]))


@pixel("strip_color")
def strip_color(xp: Any, x: Any, p: dict[str, Any]) -> Any:
    """Early colour film. ``2``: two records only (red and a green-blue one: skin warm,
    blues and greens turn teal, no true yellow or blue); ``3``: three dye layers with strong
    crosstalk correction (deep saturated primaries, dense blacks)."""
    r, g, b = x[0], x[1], x[2]
    if int(p["strips"]) == 2:
        c = 0.8 * g + 0.2 * b
        out = xp.stack([r, 0.97 * c + 0.02, 0.85 * c + 0.02])
    else:
        out = xp.stack([1.35 * r - 0.25 * g - 0.10 * b, -0.15 * r + 1.30 * g - 0.15 * b,
                        -0.10 * r - 0.30 * g + 1.40 * b])  # fmt: skip
        out = _scurve(out.clip(0, 1), 0.5)
    return _mix(x, out, float(p["amount"]))


def _smooth_noise(rng: Any, n: int, t: float, speed: float) -> float:
    """A smooth random value in about −1…1 at time ``t`` (a few seeded sines)."""
    freqs, phases = rng.uniform(0.3, 1.7, n), rng.random(n)
    return float(sum(np.sin(2 * math.pi * (f * speed * t + ph)) for f, ph in zip(freqs, phases,
                                                                                 strict=True))
                 / math.sqrt(n / 2))  # fmt: skip


@kernel("light_leak")
def light_leak(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Light leaking onto the film: ``leak`` — soft warm glows drifting in from an edge,
    breathing; ``burn`` — the end of the roll burning out from an edge, a hot white core with an
    orange rim. Seeded (per effect) and moving with time; screened over (only brightens)."""
    _, h, w = x.shape
    xp = o.xp
    rng = np.random.default_rng(f.seed)
    t = f.index / f.fps
    speed = float(p["speed"])
    edge = p["edge"] if p["edge"] != "auto" else ["left", "right", "top", "bottom"][
        int(rng.integers(0, 4))]  # fmt: skip
    q = max(1, min(h, w) // 90)
    sh, sw = max(2, h // q), max(2, w // q)
    ys, xs = o.grid(sh, sw)
    u, v = (xs + 0.5) / sw, (ys + 0.5) / sh  # 0–1 across, 0–1 down
    along, depth = {"left": (v, u), "right": (v, 1 - u), "top": (u, v), "bottom": (u, 1 - v)}[
        edge]  # fmt: skip
    aspect = sw / sh if edge in ("left", "right") else sh / sw
    base = hex_rgb(p["color"])
    layer = 0 * xp.stack([u, u, u])
    flicker = 1 + 0.12 * np.random.default_rng([f.seed, f.index]).standard_normal()
    if p["kind"] == "burn":
        pos = 0.5 + 0.3 * _smooth_noise(rng, 3, t, 0.4 * speed)
        grow = 0.35 + 0.25 * (0.5 + 0.5 * _smooth_noise(rng, 3, t, 0.6 * speed))
        d2 = ((along - pos) * aspect) ** 2 + depth**2
        core = xp.exp(-d2 / (2 * (0.45 * grow) ** 2))
        rim = xp.exp(-d2 / (2 * grow**2))
        for c in range(3):
            layer[c] = rim * base[c] * 1.2 + core**2 * 1.0
    else:
        for _ in range(3):
            pos = rng.uniform(0.15, 0.85) + 0.25 * _smooth_noise(rng, 3, t, 0.3 * speed)
            radius = rng.uniform(0.18, 0.38)
            pulse = 0.55 + 0.45 * _smooth_noise(rng, 2, t, 0.5 * speed)
            warm = rng.uniform(0, 1)
            colour = [base[c] + (hex_rgb("#ffd27a")[c] - base[c]) * warm for c in range(3)]
            d2 = ((along - pos) * aspect) ** 2 + (depth + 0.1) ** 2
            glow = xp.exp(-d2 / (2 * radius**2)) * max(0.0, pulse)
            for c in range(3):
                layer[c] = layer[c] + glow * colour[c]
    layer = upscale(o, layer, h, w) * float(p["intensity"]) * flicker
    return screen(x, layer.clip(0, 1))


GAUGES = {  # how each film width looks projected: weave, dust, scratches, coarse grain, softness
    "8": {"weave": 1.0, "dust": 1.0, "scratches": 0.25, "grain": 0.8, "flicker": 0.6,
          "soft": 1.0, "vignette": 0.45},
    "16": {"weave": 0.5, "dust": 0.6, "scratches": 0.15, "grain": 0.45, "flicker": 0.35,
           "soft": 0.5, "vignette": 0.25},
    "35": {"weave": 0.2, "dust": 0.3, "scratches": 0.08, "grain": 0.15, "flicker": 0.15,
           "soft": 0.1, "vignette": 0.08},
}  # fmt: skip


def gauge_value(p: dict[str, Any], name: str) -> float:
    value = p.get(name)
    return float(GAUGES[str(p["gauge"])][name] if value is None else value)


def _specks(rng: Any, h: int, w: int, n: int, short: int) -> Any:
    """``n`` dust specks and hairs (0–1 opacity), float32 (h, w), numpy."""
    layer = np.zeros((h, w), np.float32)
    for _ in range(n):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(0.0015, 0.005) * short
        hair = rng.random() < 0.3
        reach = math.ceil(r * (8 if hair else 3)) + 1
        x0, x1 = max(0, int(cx) - reach), min(w, int(cx) + reach + 1)
        y0, y1 = max(0, int(cy) - reach), min(h, int(cy) + reach + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        dy, dx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy = dx - cx, dy - cy
        if hair:  # a thin curved fibre
            bend, angle = rng.uniform(-0.15, 0.15), rng.uniform(0, math.pi)
            a, b = dx * math.cos(angle) + dy * math.sin(angle), -dx * math.sin(angle) + dy * (
                math.cos(angle))  # fmt: skip
            spot = np.exp(-((b - bend * a * a / max(r, 1)) ** 2) / (2 * (0.25 * r) ** 2)) * (
                np.abs(a) < 7 * r)  # fmt: skip
        else:
            spot = np.exp(-(dx * dx + dy * dy) / (2 * r * r))
        layer[y0:y1, x0:x1] = np.maximum(layer[y0:y1, x0:x1], spot * rng.uniform(0.5, 1.0))
    return layer


@kernel("projector")
def projector(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Projected film of a ``gauge`` (8, 16 or 35 mm; any of its parts can be set alone): gate
    weave (a slow random wander of the frame), softness, brightness flicker, coarse grain clumps
    the size of the gauge's grain (the fine grain is ``color.film``'s), dust and hairs that
    change every frame, scratches that stay for a while and drift, a darker edge. Seeded."""
    _, h, w = x.shape
    xp = o.xp
    short = min(h, w)
    t = f.index / f.fps
    rng = np.random.default_rng(f.seed)
    frame_rng = np.random.default_rng([f.seed, f.index])
    weave = gauge_value(p, "weave") * 0.006 * short
    if weave > 0.05:
        dx = weave * _smooth_noise(rng, 4, t, 1.3)
        dy = 0.6 * weave * _smooth_noise(rng, 4, t, 0.9)
        ys, xs = o.grid(h, w)
        x = sample(o, x, ys - dy, xs - dx)
    x = blur(o, x, GAUGES[str(p["gauge"])]["soft"] * 0.0015 * short)
    flicker = gauge_value(p, "flicker") * 0.05 * frame_rng.standard_normal()
    x = x * (1 + flicker)
    grain = gauge_value(p, "grain")
    if grain > 0:
        size = {"8": 0.006, "16": 0.0035, "35": 0.002}[str(p["gauge"])] * short
        gh, gw = max(2, round(h / max(size, 1))), max(2, round(w / max(size, 1)))
        clumps = upscale(o, o.asarray(frame_rng.standard_normal((1, gh, gw))), h, w)[0]
        y = luma(x)
        x = x + (clumps * 0.035 * grain * 4 * (y * (1 - y)).clip(0.05, None))[None]
    dust = gauge_value(p, "dust")
    if dust > 0:
        n = int(frame_rng.poisson(dust * 10))
        specks = _specks(frame_rng, h, w, n, short)
        x = x * (1 - 0.85 * o.asarray(specks)[None])
    scratches = gauge_value(p, "scratches")
    if scratches > 0:
        _, xs = o.grid(h, w)
        lines = 0 * xs
        for j in range(4):
            life = 12 + 9 * j  # frames a scratch lasts (each slot its own rhythm)
            window = (f.index + 5 * j) // life
            srng = np.random.default_rng([f.seed, 77, j, window])
            if srng.random() > scratches:
                continue
            cx = srng.uniform(0.05, 0.95) * w + srng.uniform(-1, 1) * 0.002 * w * (
                (f.index + 5 * j) % life)  # fmt: skip
            width = srng.uniform(0.35, 0.8) * max(1.0, short / 540)
            lines = lines + xp.exp(-((xs - cx) ** 2) / (2 * width**2)) * srng.uniform(0.1, 0.3)
        x = screen(x, lines.clip(0, 1)[None] * 0.9)
    vignette = GAUGES[str(p["gauge"])]["vignette"]
    ys, xs = o.grid(h, w)
    r2 = ((xs - (w - 1) / 2) / (w / 2)) ** 2 + ((ys - (h - 1) / 2) / (h / 2)) ** 2
    return (x * (1 - vignette * (r2 / 2) ** 1.5)[None]).clip(0, 1)


# --- retro -------------------------------------------------------------------------------------

THERMAL = {
    "iron": ["#000000", "#1c0050", "#7a0a8c", "#d4282d", "#f5871e", "#fde26a", "#ffffff"],
    "rainbow": ["#00001e", "#0000ff", "#00ffff", "#00ff00", "#ffff00", "#ff0000", "#ffffff"],
    "white_hot": ["#000000", "#ffffff"],
    "black_hot": ["#ffffff", "#000000"],
}


@pixel("thermal")
def thermal(xp: Any, x: Any, p: dict[str, Any]) -> Any:
    """Thermal / false colour: brightness read as heat and mapped onto a camera palette."""
    y = luma(x).clip(0, 1)
    return _mix(x, ramp(xp, y, THERMAL[p["palette"]]), float(p["amount"]))


def bayer(n: int) -> Any:
    """The n × n Bayer threshold matrix (n a power of two), values 0 … n² − 1."""
    m = np.zeros((1, 1), np.int64)
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return m


def _cells(o: Any, x: Any, ch: int, cw: int) -> Any:
    """Area means over ch × cw cells (the frame cropped to whole cells)."""
    c, h, w = x.shape
    hh, ww = max(1, h // ch), max(1, w // cw)
    return x[:, : hh * ch, : ww * cw].reshape(c, hh, ch, ww, cw).mean((2, 4))


def _blocky(o: Any, small: Any, ch: int, cw: int, h: int, w: int) -> Any:
    """``small`` (cells) back to h × w, each cell a flat block."""
    ys, xs = o.grid(h, w)
    iy = (ys // ch).clip(0, small.shape[1] - 1)
    ix = (xs // cw).clip(0, small.shape[2] - 1)
    return o.gather(small, iy, ix)


def _threshold(o: Any, n: int, hh: int, ww: int) -> Any:
    """The Bayer matrix tiled over hh × ww cells, centred on 0 (−0.5 … 0.5)."""
    m = (bayer(n) + 0.5) / (n * n) - 0.5
    reps = (hh // n + 1, ww // n + 1)
    return o.asarray(np.tile(m, reps)[:hh, :ww].astype(np.float32))


_BLUE: dict[int, Any] = {}


def blue_noise(n: int = 64) -> Any:
    """An n × n blue-noise threshold table (−0.5 … 0.5), our own void-and-cluster (Ulichney):
    a seeded 10 % pattern relaxed until its tightest cluster is its largest void, ranked by
    taking clusters out, then by filling the largest void until full. Toroidal Gaussian energy
    (σ 1.5) kept up to date one point at a time. Made once a process."""
    if n in _BLUE:
        return _BLUE[n]
    d = np.minimum(np.arange(n), n - np.arange(n)).astype(np.float64)
    kern = np.exp(-(d[:, None] ** 2 + d[None, :] ** 2) / (2 * 1.5**2))

    def at(i: int) -> Any:
        return np.roll(np.roll(kern, i // n, 0), i % n, 1)

    rng = np.random.default_rng(1)
    ones = np.zeros(n * n, bool)
    ones[rng.choice(n * n, n * n // 10, replace=False)] = True
    energy = sum((at(i) for i in np.flatnonzero(ones)), np.zeros((n, n))).ravel()
    for _ in range(4 * n * n):  # relax: move the tightest cluster to the largest void
        tight = int(np.argmax(np.where(ones, energy, -np.inf)))
        ones[tight], energy = False, energy - at(tight).ravel()
        void = int(np.argmin(np.where(ones, np.inf, energy)))
        ones[void], energy = True, energy + at(void).ravel()
        if void == tight:
            break
    rank = np.zeros(n * n, np.int64)
    count = int(ones.sum())
    left, e = ones.copy(), energy.copy()
    for r in range(count - 1, -1, -1):  # the initial points: tightest first gets the top rank
        tight = int(np.argmax(np.where(left, e, -np.inf)))
        left[tight], e, rank[tight] = False, e - at(tight).ravel(), r
    for r in range(count, n * n):  # then fill the largest void
        void = int(np.argmin(np.where(ones, np.inf, energy)))
        ones[void], energy, rank[void] = True, energy + at(void).ravel(), r
    _BLUE[n] = ((rank.reshape(n, n) + 0.5) / (n * n) - 0.5).astype(np.float32)
    return _BLUE[n]


DIFFUSION = {  # (dy, dx, weight): Floyd–Steinberg keeps all the error, Atkinson 6/8 of it
    "floyd": ((0, 1, 7 / 16), (1, -1, 3 / 16), (1, 0, 5 / 16), (1, 1, 1 / 16)),
    "atkinson": ((0, 1, 1 / 8), (0, 2, 1 / 8), (1, -1, 1 / 8), (1, 0, 1 / 8), (1, 1, 1 / 8),
                 (2, 0, 1 / 8)),
}  # fmt: skip


def diffuse(v: Any, n: int, kind: str, offset: float = 0.0) -> Any:
    """Error diffusion of ``v`` (c, h, w) numpy to ``n`` steps (``DIFFUSION[kind]``). A cell's
    error goes only right and down, so all cells with the same x + 2y depend only on earlier
    ones: each such wavefront is one vectorised step (w + 2h steps, not w × h)."""
    c, h, w = v.shape
    err = np.zeros((c, h + 2, w + 4), np.float64)
    flat = err.reshape(c, -1)
    out = np.zeros_like(v, dtype=np.float64)
    taps = DIFFUSION[kind]
    for k in range(w + 2 * (h - 1)):
        y = np.arange(max(0, -((w - 1 - k) // 2)), min(h - 1, k // 2) + 1)
        x = k - 2 * y
        if not len(y):
            continue
        val = v[:, y, x] + err[:, y, x + 2]
        q = np.clip(np.floor(val * n + 0.5 + offset) / n, 0, 1)
        out[:, y, x] = q
        e = val - q
        for dy, dx, wt in taps:
            idx = (y + dy) * (w + 4) + x + dx + 2
            for ch in range(c):
                np.add.at(flat[ch], idx, e[ch] * wt)
    return out.astype(np.float32)


@kernel("dither")
def dither(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Dither: each channel (``mono``: the luma, in two colours) quantised to ``levels`` on a
    grid of ``size`` (a coarse retro pixel), by ``algorithm``:

    - ``bayer`` (a 2, 4 or 8 matrix) or ``bluenoise`` (a 64 × 64 void-and-cluster table):
      ordered, read where ``pattern`` puts it (at each cell's centre) — with the picture
      (``content``: along the camera path), ``boil`` or on the ``screen``;
    - ``floyd`` (Floyd–Steinberg, keeps the mean) or ``atkinson`` (diffuses 6/8: the lightest
      and darkest tones blow out, Mac-like): error diffusion, worked out anew every frame (it
      shimmers on video by nature; ordered tables stay put).

    ``contrast`` and ``edge_preserve`` (unsharp at the cell scale) shape the input first;
    ``wobble`` sways the threshold over time."""
    _, h, w = x.shape
    s = max(1, round(float(p["size"]) * min(h, w)))
    small = _cells(o, x, s, s)
    hh, ww = small.shape[1:]
    contrast, edge = float(p.get("contrast", 1.0)), float(p.get("edge_preserve", 0.0))
    if edge > 0:
        small = small + edge * (small - blur(o, small, 1.0))
    if contrast != 1.0:
        small = (small - 0.5) * contrast + 0.5
    if edge > 0 or contrast != 1.0:
        small = small.clip(0, 1)
    wobble = float(p.get("wobble", 0.0))
    sway = wobble * 0.25 * math.sin(2 * math.pi * float(p.get("wobble_speed", 0.5)) * f.index
                                    / f.fps) if wobble > 0 else 0.0  # fmt: skip
    n = int(p["levels"]) - 1
    algo = p.get("algorithm", "bayer")
    src = luma(small)[None] if p["mono"] else small
    if algo in DIFFUSION:
        v = o.asarray(diffuse(np.asarray(o.numpy(src), np.float64), n, algo, sway))
    else:
        if algo == "bluenoise":
            k, raw = 64, blue_noise(64)
        else:
            k = int(p["matrix"])
            raw = ((bayer(k) + 0.5) / (k * k) - 0.5).astype(np.float32)
        table = o.asarray(raw)[None]
        at = pattern_grid(o, x, p, f)[:, s // 2 :: s, s // 2 :: s][:, :hh, :ww]  # cell centres
        t = o.gather(table, o.xp.floor(at[0] / s) % k, o.xp.floor(at[1] / s) % k)[0] + sway
        v = (o.xp.floor(src * n + 0.5 + t[None]) / n).clip(0, 1)
    if p["mono"]:
        ink = o.asarray(hex_rgb(p["ink"]))[:, None, None]
        paper = o.asarray(hex_rgb(p["paper"]))[:, None, None]
        v = ink + (paper - ink) * v
    return _blocky(o, v, s, s, h, w)


PALETTES = {
    "lcd": ["#0f380f", "#306230", "#8bac0f", "#9bbc0f"],
    "cga": ["#000000", "#55ffff", "#ff55ff", "#ffffff"],
    "ega": ["#000000", "#0000aa", "#00aa00", "#00aaaa", "#aa0000", "#aa00aa", "#aa5500",
            "#aaaaaa", "#555555", "#5555ff", "#55ff55", "#55ffff", "#ff5555", "#ff55ff",
            "#ffff55", "#ffffff"],
    "sepya": ["#2b1d14", "#6b4a2f", "#b08654", "#ecd9b0"],
    "gece": ["#0b0c24", "#2e1f5e", "#6b2d82", "#c2408f", "#ff8170", "#ffd8a8"],
    "pastel": ["#2b2d42", "#5c6784", "#a3b1c6", "#f2efe9", "#e07a5f", "#f2cc8f", "#81b29a",
               "#3d405b"],
}  # fmt: skip


@kernel("pixel_art")
def pixel_art(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Pixel art: the picture on a coarse grid of ``size``, every pixel snapped to the nearest
    colour of a limited palette (a preset or your own), optionally dithered between them."""
    _, h, w = x.shape
    s = max(2, round(float(p["size"]) * min(h, w)))
    small = _cells(o, x, s, s)
    colors = p.get("colors") or PALETTES[p["palette"]]
    pal = o.asarray(np.array([hex_rgb(c) for c in colors], np.float32).T)  # (3, n)
    amount = float(p["dither"])
    if amount > 0:
        t = _threshold(o, 4, small.shape[1], small.shape[2])
        small = small + t[None] * amount * 0.25
    weight = o.asarray([0.3, 0.59, 0.11])[:, None, None, None]
    d = (((small[:, None] - pal[:, :, None, None]) ** 2) * weight).sum(0)  # (n, hh, ww)
    small = pal[:, d.argmin(0)]
    return _blocky(o, small, s, s, h, w)


CHARSETS = {
    "standard": " .:;ocxOX%#@",  # no line glyphs: they read as stripes
    "dense": " .'`^,:;Il!i~+_-?][}{1)(|tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
    "binary": " 01",
    "blocks": " .oO@",
}
_ATLAS: dict[Any, Any] = {}


def ascii_atlas(charset: str, ch: int, cw: int) -> Any:
    """The charset's glyphs (bundled Inter Bold, white on black, 0–1) as (n, ch, cw), ordered
    from the least ink to the most (so brightness maps onto them in order). Drawn four times
    larger and box-filtered down, so small cells stay legible."""
    key = (charset, ch, cw)
    if key not in _ATLAS:
        from PIL import Image, ImageDraw, ImageFont

        try:
            font = ImageFont.truetype(str(FONT), max(4, round(ch * 3.4)))
        except OSError:
            font = ImageFont.load_default()
        glyphs = []
        for char in dict.fromkeys(CHARSETS[charset]):
            img = Image.new("L", (cw * 4, ch * 4), 0)
            d = ImageDraw.Draw(img)
            left, top, right, bottom = d.textbbox((0, 0), char, font=font)
            at = ((cw * 4 - (right - left)) / 2 - left, (ch * 4 - (bottom - top)) / 2 - top)
            d.text(at, char, fill=255, font=font)
            glyphs.append(np.asarray(img.resize((cw, ch), Image.Resampling.BOX), np.float32) / 255)
        glyphs.sort(key=lambda g: float(g.mean()))
        if len(_ATLAS) >= 32:  # a keyed cell size makes one per value: the oldest go
            _ATLAS.pop(next(iter(_ATLAS)))
        _ATLAS[key] = np.stack(glyphs)
    return _ATLAS[key]


@kernel("ascii")
def ascii_art(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """ASCII: the picture as characters on a grid of ``size`` (cell height), each cell's
    brightness choosing a glyph by its ink (bundled font); ``stretch``: the frame's own
    brightness range (2nd–98th percentile of the cells) spread over the whole ramp; ``mono`` in
    one ink colour or ``color`` with each cell's own colour, on a dark background."""
    _, h, w = x.shape
    xp = o.xp
    ch = max(6, round(float(p["size"]) * min(h, w)))
    cw = max(4, round(ch * 0.6))
    cells = _cells(o, x, ch, cw)
    atlas = ascii_atlas(p["charset"], ch, cw)
    n = atlas.shape[0]
    y = luma(cells).clip(0, 1)
    if p.get("stretch", False):
        lo, hi = np.percentile(o.numpy(y), [2, 98])
        if hi - lo > 0.02:  # a flat frame stays as it is
            y = ((y - float(lo)) / float(hi - lo)).clip(0, 1)
    idx = xp.floor(y**0.6 * (n - 0.001))  # mid-tones get ink
    ys, xs = o.grid(h, w)
    cy = (ys // ch).clip(0, cells.shape[1] - 1)
    cx = (xs // cw).clip(0, cells.shape[2] - 1)
    glyph_of = o.gather(idx[None], cy, cx)[0]
    flat = o.asarray(atlas.reshape(1, n * ch, cw))
    glyph = o.gather(flat, glyph_of * ch + ys % ch, xs % cw)
    bg = o.asarray(hex_rgb(p["background"]))[:, None, None]
    if p["mode"] == "color":
        tone = o.gather(cells, cy, cx)
        ink = tone / (xp.maximum(xp.maximum(tone[0], tone[1]), tone[2])[None] + 1e-3)
    else:
        ink = o.asarray(hex_rgb(p["ink"]))[:, None, None]
    return bg + (ink - bg) * glyph


@kernel("crt")
def crt(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """CRT: a slightly curved screen with dark rounded corners, a soft glow, scanlines (dark
    gaps between beams that widen where it is bright) and an RGB phosphor mask (aperture
    grille stripes or a slot mask; the triad's mean is kept)."""
    _, h, w = x.shape
    xp = o.xp
    short = min(h, w)
    ys, xs = o.grid(h, w)
    k = float(p["curvature"]) * 0.12
    if k > 0:
        u, v = (xs - (w - 1) / 2) / (w / 2), (ys - (h - 1) / 2) / (h / 2)
        bend = 1 + k * (u * u + v * v)
        x = sample(o, x, (h - 1) / 2 + v * bend * h / 2, (w - 1) / 2 + u * bend * w / 2, "zero")
        edge = xp.maximum(xp.abs(u * bend), xp.abs(v * bend))
        x = x * ((1 - edge) * 40).clip(0, 1)[None]
    glow = float(p["glow"])
    if glow > 0:
        x = screen(x, blur(o, x, 0.006 * short) * 0.6 * glow)
    pitch = max(3.0, h / float(p["lines"]))
    phase = (ys / pitch) % 1.0
    width = 0.22 + 0.25 * luma(x)
    beam = xp.exp(-(((phase - 0.5) / width) ** 2))
    scan = float(p["scanlines"])
    x = x * (1 - scan + scan * beam * 1.25)[None]
    mask = float(p["mask"])
    if mask > 0 and p["phosphor"] != "none":
        stripe = max(1, round(short / 360))
        col = xs // stripe
        if p["phosphor"] == "slot":  # every other row of slots offset by half a triad
            col = xp.floor(col + 1.5 * ((ys // (2 * stripe)) % 2))
        col = col % 3
        x = x * xp.stack([1 - mask + mask * 3 * (col == c) for c in range(3)])
    return x.clip(0, 1)


@kernel("night_vision")
def night_vision(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Night vision: the light amplified (``gain``), green phosphor, tube noise and sparkle
    that change every frame (seeded), a glow on bright points and the round tube edge."""
    _, h, w = x.shape
    xp = o.xp
    short = min(h, w)
    rng = np.random.default_rng([f.seed, f.index])
    y = (luma(blur(o, x, 0.0015 * short)) * float(p["gain"])).clip(0, 1) ** 0.8
    noise = float(p["noise"])
    if noise > 0:
        grain = o.asarray(rng.standard_normal((h, w)).astype(np.float32))
        y = y + grain * 0.07 * noise * (1.2 - y)
        sparks = (rng.random((1, h, w)) > 1 - 0.0004 * noise).astype(np.float32)
        y = y + blur(o, o.asarray(sparks), 0.8)[0] * 4
    y = y.clip(0, 1)
    glow = float(p["glow"])
    if glow > 0:
        halo = blur(o, ((y - 0.7) / 0.3).clip(0, 1)[None], 0.02 * short)[0]
        y = 1 - (1 - y) * (1 - halo * glow)
    out = ramp(xp, y, ["#000500", "#0f3d12", "#4fbf3a", "#c8ff9a", "#f4fff0"])
    ys, xs = o.grid(h, w)
    r = xp.sqrt(((xs - (w - 1) / 2) / (h / 2)) ** 2 + ((ys - (h - 1) / 2) / (h / 2)) ** 2)
    tube = float(p["tube"])
    rim = ((1.15 * tube + 2.5 * (1 - tube) - r) * 12).clip(0, 1)
    return out * (rim * (1 - 0.25 * r * r).clip(0, 1))[None]


# --- print -------------------------------------------------------------------------------------


@kernel("risograph")
def risograph(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Risograph: the picture separated into two or three spot inks (each ink's coverage from
    what it absorbs in each channel, least squares), every layer laid with a coarse stochastic
    grain (seeded; where ``pattern`` puts it — ``boil`` by default: a new grain every ``boil``
    frames and each ink a little off its place every frame, like a printed flip book) and a
    little off register, the layers evenly spread around a seeded direction; the inks multiply
    on the paper."""
    _, h, w = x.shape
    short = min(h, w)
    paper = np.array(hex_rgb(p["paper"]), np.float32).clip(1e-3, None)
    inks = np.array([hex_rgb(c) for c in p["inks"]], np.float32)
    absorb = (1 - inks / paper).clip(0, 1).T  # (3 channels, k inks)
    take = np.linalg.pinv(absorb)  # (k, 3)
    lack = 1 - (x / o.asarray(paper)[:, None, None]).clip(0, 1)
    grain = max(1.0, float(p["grain_size"]) * short)
    layers = pattern_layers(o, x, p, f, default="boil")
    boil = p.get("pattern", "boil") == "boil"
    start = np.random.default_rng([f.seed, 3]).uniform(0, 2 * math.pi)
    reach = float(p["misregister"]) * 0.01 * short
    out = o.asarray(paper)[:, None, None] * 1
    for k in range(len(inks)):
        cover = sum(float(take[k, c]) * lack[c] for c in range(3)).clip(0, 1) ** 1.25
        angle = start + 2 * math.pi * k / len(inks)
        dx, dy = reach * math.cos(angle), reach * math.sin(angle)
        if boil:  # each ink lands a little off its place on every print
            jy, jx = np.random.default_rng([f.seed, 23, f.index, k]).normal(0, JITTER * short, 2)
            dx, dy = dx + float(jx), dy + float(jy)
        if abs(dx) + abs(dy) > 0.05:
            cover = shifted(o, cover[None], dx, dy)[0]
        # Smooth noise (std 1, ~0.6 × grain px across), as clumps uniform in 0–1.
        z = texture(o, layers, _table(f.seed + k, 256), 0.75 * grain, (h, w), angle=1.3 * k)
        dots = ((cover - 1 / (1 + o.xp.exp(-1.702 * z))) * 5 + 0.5).clip(0, 1)
        laid = cover + float(p["grain"]) * (dots - cover)
        out = out * (1 - laid[None] * o.asarray(absorb[:, k])[:, None, None])
    return out.clip(0, 1)


@kernel("engrave")
def engrave(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Engraving or linocut. ``engraving``: parallel lines at ``angle``, thin in the lights and
    swelling into the shadows, bent a little by the tone like a burin following form;
    ``linocut``: the picture cut into ink and paper (two tones), with gouge strokes where the
    tone is near the cut — white strokes in the ink, ink strokes in the paper. The lines lie
    where ``pattern`` puts them (``content``: along the camera path)."""
    _, h, w = x.shape
    xp = o.xp
    short = min(h, w)
    tone = blur(o, luma(x)[None], max(0.5, 0.0015 * short))[0].clip(0, 1)
    period = max(3.0, float(p["spacing"]) * short)
    a = math.radians(float(p["angle"]))
    dark = 1 - tone
    coords = pattern_grid(o, x, p, f)
    u = (coords[1] * math.cos(a) + coords[0] * math.sin(a)) / period + 0.35 * tone
    tri = xp.abs(u - xp.floor(u) - 0.5) * 2  # 0 at a line's centre, 1 between lines
    if p["style"] == "linocut":
        cut = ((dark - 0.5) * 10 + 0.5).clip(0, 1)
        near = (1 - xp.abs(dark - 0.5) * 3.3).clip(0, 1)
        gouge = ((0.3 - tri) * period / 2).clip(0, 1) * near
        ink = cut * (1 - gouge) + (1 - cut) * gouge
    else:
        ink = ((dark**0.9 - tri) * period / 2 + 0.5).clip(0, 1)
    paper = o.asarray(hex_rgb(p["paper"]))[:, None, None]
    pen = o.asarray(hex_rgb(p["ink"]))[:, None, None]
    return paper + (pen - paper) * ink[None]


# --- geometry ----------------------------------------------------------------------------------


def _polar(o: Any, h: int, w: int, center: list[float]) -> tuple[Any, Any, float, float]:
    xp = o.xp
    ys, xs = o.grid(h, w)
    cx, cy = float(center[0]) * (w - 1), float(center[1]) * (h - 1)
    dx, dy = xs - cx, ys - cy
    return xp.sqrt(dx * dx + dy * dy), xp.arctan2(dy, dx), cx, cy


@kernel("kaleidoscope")
def kaleidoscope(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Kaleidoscope: one wedge of the picture mirrored around ``center`` into ``segments``
    (turning with ``spin``), zoomed in by ``zoom``."""
    _, h, w = x.shape
    xp = o.xp
    r, theta, cx, cy = _polar(o, h, w, p["center"])
    seg = 2 * math.pi / int(p["segments"])
    turn = math.radians(float(p["angle"])) + 2 * math.pi * float(p["spin"]) * f.index / f.fps
    a = (theta - turn) % seg
    a = xp.minimum(a, seg - a) + turn
    rr = r / float(p["zoom"])
    return sample(o, x, cy + rr * xp.sin(a), cx + rr * xp.cos(a))


@kernel("mirror")
def mirror(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Mirror: one half of the picture reflected onto the other (or a quarter into all four)."""
    _, h, w = x.shape
    xp = o.xp
    ys, xs = o.grid(h, w)
    mode = p["mode"]
    if mode in ("left", "quad"):
        xs = xp.minimum(xs, w - 1 - xs)
    elif mode == "right":
        xs = xp.maximum(xs, w - 1 - xs)
    if mode in ("top", "quad"):
        ys = xp.minimum(ys, h - 1 - ys)
    elif mode == "bottom":
        ys = xp.maximum(ys, h - 1 - ys)
    return o.gather(x, ys, xs)


@kernel("swirl")
def swirl(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Swirl: the picture turned around ``center`` by ``angle`` turns at the middle, less and
    less out to ``radius`` (untouched beyond it); ``speed`` turns it over time."""
    _, h, w = x.shape
    xp = o.xp
    r, theta, cx, cy = _polar(o, h, w, p["center"])
    reach = float(p["radius"]) * min(h, w)
    turns = float(p["angle"]) + float(p["speed"]) * f.index / f.fps
    a = theta - 2 * math.pi * turns * (1 - r / reach).clip(0, 1) ** 2
    return sample(o, x, cy + r * xp.sin(a), cx + r * xp.cos(a))


_HEAT: dict[int, Any] = {}


def _heat_table(seed: int) -> Any:
    """Two smooth tileable noise fields (2, 64, 64), std 1."""
    if seed not in _HEAT:
        rng = np.random.default_rng(seed)
        f = np.fft.fftfreq(64).astype(np.float32)
        g = np.exp(-2 * math.pi**2 * 2.0**2 * (f[:, None] ** 2 + f[None, :] ** 2))
        fields = [np.fft.ifft2(np.fft.fft2(rng.standard_normal((64, 64))) * g).real
                  for _ in range(2)]  # fmt: skip
        if len(_HEAT) > 16:
            _HEAT.clear()
        _HEAT[seed] = np.stack([a / a.std() for a in fields]).astype(np.float32)
    return _HEAT[seed]


@kernel("wave")
def wave(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Wave or heat distortion. ``wave``: rows (``horizontal``) or columns moved by a sine of
    ``wavelength``, travelling at ``speed``; ``heat``: a shimmer of smooth seeded noise rising
    through the picture."""
    _, h, w = x.shape
    xp = o.xp
    short = min(h, w)
    ys, xs = o.grid(h, w)
    amp = float(p["amplitude"]) * short
    length = max(2.0, float(p["wavelength"]) * short)
    t = f.index / f.fps
    if p["kind"] == "heat":
        table = o.asarray(_heat_table(f.seed))
        n = table.shape[-1]
        gy, gx = (ys / length + float(p["speed"]) * t) * 8, xs / length * 8
        y0, x0 = xp.floor(gy), xp.floor(gx)
        wy, wx = gy - y0, gx - x0
        val = 0
        for dy, dx, k in ((0, 0, (1 - wy) * (1 - wx)), (0, 1, (1 - wy) * wx),
                          (1, 0, wy * (1 - wx)), (1, 1, wy * wx)):  # fmt: skip
            val = val + o.gather(table, (y0 + dy) % n, (x0 + dx) % n) * k
        return sample(o, x, ys + amp * 0.5 * val[1], xs + amp * val[0])
    phase = 2 * math.pi * float(p["speed"]) * t
    if p["direction"] == "horizontal":
        return sample(o, x, ys, xs + amp * xp.sin(2 * math.pi * ys / length - phase))
    return sample(o, x, ys + amp * xp.sin(2 * math.pi * xs / length - phase), xs)


@kernel("prism")
def prism(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Prism filter: ``copies`` images of the picture fanned out along ``angle`` by ``spread``,
    each tinted a different hue (``rainbow``; the tints add up to white, so the whole stays
    neutral), mixed over the picture by ``mix``."""
    _, h, w = x.shape
    k = int(p["copies"])
    step = float(p["spread"]) * min(h, w)
    a = math.radians(float(p["angle"]))
    rainbow = float(p["rainbow"])
    total, weight = 0 * x, 0
    for j in range(k):
        off = (j - (k - 1) / 2) * step
        tint = o.asarray([1 + (c - 1) * rainbow for c in hue_rgb(360 * j / k)])[:, None, None]
        total = total + shifted(o, x, off * math.cos(a), off * math.sin(a)) * tint
        weight = weight + tint
    return x + (total / weight - x) * float(p["mix"])
