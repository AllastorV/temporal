"""Frame kernels of the effects: one function per effect, run by the frame processor
(``fx_runner.py``) on numpy or torch arrays. Imports nothing from Cutwright: the same file runs
in Cutwright's own Python (numpy, the CPU fallback and the tests) and in the objects
environment (torch, CUDA when present).

A kernel is ``fn(o, x, p, f) -> x``:

- ``o``: the backend (``NumpyOps`` / ``TorchOps``): ``o.xp`` is numpy or torch, plus the few
  operations whose spelling differs (blur, sampling, padding, arrays from numpy);
- ``x``: the frame, float32 RGB, channels first (3, H, W), display-encoded 0–1;
- ``p``: the effect's params at this frame (keys already applied);
- ``f``: ``Frame`` — ``index`` (the source frame number on the record's grid, the same in a
  preview and a render), ``fps``, ``seed`` (clip id + effect id + its seed param).

Randomness comes from ``numpy.random.default_rng([f.seed, f.index])`` (per frame) or
``default_rng(f.seed)`` (per effect, e.g. particle starting points), drawn with numpy so both
backends see the same numbers. ``@kernel("name")`` registers a kernel and refuses a second one
with the same name (a silently replaced effect was a PixEditor bug).

Colour effects that depend on the pixel alone are written once as ``@pixel`` functions on
(3, …) arrays: the ffmpeg path bakes them into a 3D LUT, the frame processor runs them directly.

``f.state`` is a dict the effect keeps for the whole job (frames come in order; effects that
look back declare ``FxDef.history`` and get warm-up frames first, so a still computes frame k
from the same inputs as the render). ``f.mask`` is the effect's scope mask at this frame (or
None); ``REGION`` kernels place themselves by it and are not blended by it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

if __package__:  # in Cutwright's process
    from cutwright.modules.fx.runners.fx_base import (
        BLENDS,
        KERNELS,
        LUMA,
        PIXELS,
        REGION,
        Frame,
        NumpyOps,
        TorchOps,
        _ray,
        blend,
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
else:  # in the objects environment: the files next to the runner, by their bare names
    from fx_base import (  # type: ignore[no-redef]  # noqa: F401
        BLENDS,
        KERNELS,
        LUMA,
        PIXELS,
        REGION,
        Frame,
        NumpyOps,
        TorchOps,
        _ray,
        blend,
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


# --- FX-B: colour ------------------------------------------------------------------------------


def _tone(xp: Any, x: Any, weight: Any, hue: float, strength: float) -> Any:
    """A luma-neutral push of ``x`` towards ``hue`` (the hue's colour minus its own luma)."""
    c = hue_rgb(hue)
    y = sum(k * v for k, v in zip(LUMA, c, strict=True))
    return xp.stack([x[i] + weight * (strength * (c[i] - y)) for i in range(3)])


@pixel("split_tone")
def split_tone(xp: Any, x: Any, p: dict[str, Any]) -> Any:
    """Shadows towards one hue, highlights towards another; the pivot grey (0.5, moved by
    ``balance``) stays as it is and luma does not change (before clipping)."""
    y = luma(x)
    pivot = 0.5 - 0.35 * float(p["balance"])
    shadows = ((pivot - y) / pivot).clip(0, 1)
    highlights = ((y - pivot) / (1 - pivot)).clip(0, 1)
    k = 0.25 * float(p["amount"])
    out = _tone(xp, x, shadows, float(p["shadows"]), k * float(p["shadows_sat"]))
    return _tone(xp, out, highlights, float(p["highlights"]), k * float(p["highlights_sat"]))


@pixel("gradient_map")
def gradient_map(xp: Any, x: Any, p: dict[str, Any]) -> Any:
    """Luma mapped onto a colour ramp (piecewise linear), mixed by ``amount``."""
    y = luma(x).clip(0, 1)
    colors = [hex_rgb(c) for c in p["colors"]]
    n = len(colors)
    pos = p.get("positions") or [i / (n - 1) for i in range(n)]
    ramp = [colors[0][ch] + 0 * y for ch in range(3)]
    for i in range(n - 1):
        t = ((y - pos[i]) / max(pos[i + 1] - pos[i], 1e-6)).clip(0, 1)
        ramp = [ramp[ch] + (colors[i + 1][ch] - colors[i][ch]) * t for ch in range(3)]
    return x + float(p["amount"]) * (xp.stack(ramp) - x)


# --- FX-B: photographic ------------------------------------------------------------------------


@kernel("clarity")
def clarity(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Mid-tone local contrast: the detail over a Gaussian of ``radius`` × short side, weighted
    by 4v(1 − v) so blacks and whites stay where they are."""
    sigma = max(0.5, float(p["radius"]) * min(x.shape[1:]))
    k = 2 * float(p["amount"])
    return (x + k * 4 * x * (1 - x) * (x - blur(o, x, sigma))).clip(0, 1)


@kernel("orton")
def orton(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """The picture screened with itself, blurred and multiplied back, mixed in by ``amount``."""
    sigma = max(0.5, float(p["size"]) * min(x.shape[1:]))
    bright = screen(x, x)
    glow = bright * blur(o, bright, sigma)
    return (x + float(p["amount"]) * (glow - x)).clip(0, 1)


BLOOM_TIGHT = 0.6  # weight of the tight glow (a quarter of ``size``) over the wide one


@kernel("bloom")
def bloom(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Only what is over the threshold spreads, at two scales: a wide haze and a tight glow
    that makes small lights shine; scaled after the blur (no clip before it), tinted, screened
    over."""
    t, i = float(p["threshold"]), float(p["intensity"])
    sigma = max(0.5, float(p["size"]) * min(x.shape[1:]))
    tint = o.asarray(hex_rgb(p["tint"]))[:, None, None]
    over = ((x - t) / (1 - t)).clip(0, 1)
    glow = blur(o, over, sigma) + BLOOM_TIGHT * blur(o, over, max(0.5, sigma / 4))
    return screen(x, (glow * (2 * i) * tint).clip(0, 1))


@kernel("dispersion")
def dispersion(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Lateral chromatic aberration: red magnified, blue shrunk about ``center`` (the shift
    grows with the radius, 0 at the centre); ``prism`` spreads it into a rainbow fringe."""
    _, h, w = x.shape
    cx, cy = float(p["center"][0]) * (w - 1), float(p["center"][1]) * (h - 1)
    reach = max(math.hypot(cx, cy), math.hypot(w - 1 - cx, cy), math.hypot(cx, h - 1 - cy),
                math.hypot(w - 1 - cx, h - 1 - cy), 1.0)  # fmt: skip
    m = float(p["amount"]) * 0.02 * math.hypot(w, h) / reach  # relative magnification
    prism = float(p["prism"])
    width = 0.05 + 0.6 * prism
    taps = [-1 + 2 * j / 8 for j in range(9)]
    weights = [[math.exp(-(((e - c) / width) ** 2)) for e in taps] for c in (1.0, 0.0, -1.0)]
    ys, xs = o.grid(h, w)
    out = [0 * x[0], 0 * x[0], 0 * x[0]]
    for j, e in enumerate(taps):
        share = [weights[c][j] / sum(weights[c]) for c in range(3)]
        if max(share) < 1e-4:
            continue
        s = 1 + e * (1 + prism) * m
        moved = x if s == 1 else sample(o, x, cy + (ys - cy) / s, cx + (xs - cx) / s)
        for c in range(3):
            if share[c] > 1e-4:
                out[c] = out[c] + moved[c] * share[c]
    return o.xp.stack(out).clip(0, 1)


PARTICLE_COUNT = {"dust": 120, "bokeh": 25, "glints": 50}


def particle_layer(h: int, w: int, p: dict[str, Any], f: Frame, tint: Any = None) -> Any:
    """The particles of frame ``f`` as one float32 (3, h, w) layer (numpy on both backends, so
    they land on the same pixels): starting points, sizes and speeds from the effect's seed,
    positions from the frame's time, wrapped around the frame with a margin. Each particle has
    the ``color`` param's colour, or ``tint(px, py) -> (n, 3)``'s."""
    rng = np.random.default_rng(f.seed)
    kind = p["kind"]
    n = int(p.get("count") or PARTICLE_COUNT[kind])
    start = rng.random((n, 2))
    scale, speed = rng.uniform(0.5, 1.5, n), rng.uniform(0.6, 1.4, n)
    phase, sway = rng.random(n), rng.random(n)
    t = f.index / f.fps
    short = min(h, w)
    angle = math.radians(float(p["direction"]))
    travel = float(p["drift"]) * t * speed * short  # px
    margin = 0.1
    px = (start[:, 0] * (1 + 2 * margin) * w + math.cos(angle) * travel
          + 0.01 * short * np.sin(2 * math.pi * (0.15 * t + sway)))  # fmt: skip
    py = start[:, 1] * (1 + 2 * margin) * h + math.sin(angle) * travel
    px = px % ((1 + 2 * margin) * w) - margin * w
    py = py % ((1 + 2 * margin) * h) - margin * h
    radius = float(p["size"]) * short * scale
    colors = tint(px, py) if tint is not None else np.tile(hex_rgb(p["color"]), (n, 1))
    layer = np.zeros((3, h, w), np.float32)
    for i in range(n):
        r = max(0.5, float(radius[i]) * (3 if kind == "bokeh" else 1))
        reach = math.ceil(r * (4 if kind == "glints" else 2)) + 1
        x0, x1 = max(0, int(px[i]) - reach), min(w, int(px[i]) + reach + 1)
        y0, y1 = max(0, int(py[i]) - reach), min(h, int(py[i]) + reach + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        dy, dx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy = dx - px[i], dy - py[i]
        d2 = dx * dx + dy * dy
        if kind == "dust":
            sprite = np.exp(-d2 / (2 * (r / 2) ** 2))
        elif kind == "bokeh":  # a soft-edged disc with a slight rim
            dist = np.sqrt(d2)
            rim = np.exp(-(((dist - 0.85 * r) / (0.12 * r)) ** 2))
            sprite = 0.35 / (1 + np.exp((dist - r) / (0.2 * r))) * (1 + 0.5 * rim)
        else:
            twinkle = 0.55 + 0.45 * math.sin(2 * math.pi * (1.3 * t + phase[i]))
            core = np.exp(-d2 / (2 * (r / 2) ** 2))
            across = 2 * (0.25 * r) ** 2
            arms = (np.exp(-np.abs(dx) / (1.5 * r)) * np.exp(-dy * dy / across)
                    + np.exp(-np.abs(dy) / (1.5 * r)) * np.exp(-dx * dx / across))  # fmt: skip
            sprite = twinkle * (core + 0.6 * arms)
        layer[:, y0:y1, x0:x1] += sprite.astype(np.float32) * np.float32(colors[i])[:, None, None]
    return layer


def scene_tint(o: Any, x: Any, p: dict[str, Any]) -> Any:
    """``tint`` for bokeh: each disc takes the colour of the scene's lights around it (a
    small copy of the frame, light-weighted and blurred), at full value, mixed with the
    ``color`` param by ``scene``."""
    _, h, w = x.shape
    q = max(1, min(h, w) // 48)
    small = np.asarray(o.numpy(downscale(o, x, q)), np.float32)
    weight = small.max(0)[None] ** 4 + 1e-4
    no, sigma = NumpyOps(), max(1.0, min(small.shape[1:]) / 12)
    lit = blur(no, small * weight, sigma) / blur(no, weight, sigma)
    k, base = float(p.get("scene", 0.8)), np.asarray(hex_rgb(p["color"]))

    def tint(px: Any, py: Any) -> Any:
        iy = np.clip((py / q).astype(int), 0, lit.shape[1] - 1)
        ix = np.clip((px / q).astype(int), 0, lit.shape[2] - 1)
        c = lit[:, iy, ix].T
        return (1 - k) * base + k * c / np.maximum(c.max(1, keepdims=True), 0.05)

    return tint


@kernel("particles")
def particles(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Dust, bokeh or glints added over the picture (screen: it only ever brightens)."""
    _, h, w = x.shape
    tint = scene_tint(o, x, p) if p["kind"] == "bokeh" and p.get("scene", 0.8) > 0 else None
    layer = o.asarray(particle_layer(h, w, p, f, tint))
    return screen(x, (layer * float(p["brightness"])).clip(0, 1))


def _span(o: Any, s: Any, dx: float, dy: float, n: float) -> Any:
    """The sum of ``s`` over about ``n`` px both ways along (dx, dy) (doubling steps)."""
    out, span = s, 1
    while span < n:
        out = out + shifted(o, out, dx * span, dy * span) + shifted(o, out, -dx * span, -dy * span)
        span *= 3
    return out


@kernel("star")
def star(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Star filter: small points of light over ``threshold`` (their brightest channel, so a
    red lamp counts) drawn out into ``points`` thin arms at ``angle``.

    - Only point-like lights: their energy at a small radius is several times that at a 4×
      radius (a large bright area has the same at both) — ``size`` is the largest that counts.
    - Thin: each arm starts from a one-pixel ridge through the light's centre that carries the
      light's energy across the arm (square-root scaled: a bigger, brighter light makes a longer,
      brighter arm, not a wider one).
    - The arms fall off along their length, keep the light's colour, and their brightness is
      softly limited (hue kept). Worked at up to ~360 px on the short side, then scaled back."""
    _, h, w = x.shape
    xp = o.xp
    t = float(p["threshold"])
    over = ((peak(o, x) - t) / (1 - t)).clip(0, 1)
    q = max(1, min(h, w) // 360)
    light, amount = downscale(o, x * over[None], q), downscale(o, over[None], q)
    s1 = max(0.5, float(p.get("size", 0.03)) * min(light.shape[1:]) / 6)
    near, wide = blur(o, amount, s1), blur(o, amount, 4 * s1)
    src = light * ((near / (wide + 1e-4) - 2) / 6).clip(0, 1)
    length = float(p["length"]) * min(h, w) / q
    lines = int(p["points"]) // 2
    acc = 0 * src
    for j in range(lines):
        theta = math.radians(float(p["angle"]) + j * 180 / lines)
        dx, dy = math.cos(theta), math.sin(theta)
        top = near[0] + 1e-5  # ties count both ways: an even-width light gets a 2 px ridge
        ridge = ((top >= shifted(o, near, -dy, dx)[0]) & (top >= shifted(o, near, dy, -dx)[0])
                 & (near[0] > 1e-3))  # fmt: skip
        energy = _span(o, src, -dy, dx, 8 * s1)
        line = energy * ridge[None] / xp.sqrt(peak(o, energy).clip(1e-3, None))[None]
        acc = acc + _ray(o, line, dx, dy, length) + _ray(o, line, -dx, -dy, length) - 2 * line
    top = peak(o, acc).clip(0, None)
    soft = (1 - xp.exp(-float(p["intensity"]) * top)) / (top + 1e-6)  # hue kept
    arms = upscale(o, acc * soft[None], h, w)
    return screen(x, arms.clip(0, 1))


# --- FX-C and later: kernels in sibling files (they register themselves on import) --------------

if __package__:
    from cutwright.modules.fx.runners import (
        fx_kernels_art,
        fx_kernels_cells,
        fx_kernels_geo,
        fx_kernels_optics,
        fx_kernels_paint,
        fx_kernels_portrait,
        fx_kernels_signal,
        fx_kernels_stylize,
    )
else:
    import fx_kernels_art  # type: ignore[no-redef]  # noqa: F401
    import fx_kernels_cells  # type: ignore[no-redef]  # noqa: F401
    import fx_kernels_geo  # type: ignore[no-redef]  # noqa: F401
    import fx_kernels_optics  # type: ignore[no-redef]  # noqa: F401
    import fx_kernels_paint  # type: ignore[no-redef]  # noqa: F401
    import fx_kernels_portrait  # type: ignore[no-redef]  # noqa: F401
    import fx_kernels_signal  # type: ignore[no-redef]  # noqa: F401
    import fx_kernels_stylize  # type: ignore[no-redef]  # noqa: F401
