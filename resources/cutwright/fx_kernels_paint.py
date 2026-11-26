"""Frame kernels of the FX-E effects that move with the picture: optical flow, carried
textures, the painterly effects (oil, watercolor, ink, toon) and the time effects (echo,
step_print, slit_scan, datamosh). Same rules as ``fx_kernels``: imports nothing from Cutwright,
numpy or torch through ``o``, randomness from numpy.

Steady over time:

- ``flow`` is our own pyramidal Lucas–Kanade on luma, written with the backend ops (the
  objects environment has no OpenCV): a regularised solve, so flat or noisy areas get no
  motion, and a soft dead zone under ~0.2–0.45 px, so a static noisy scene has almost none;
- painterly kernels read their input steadied by the previous input frame warped along the
  flow (a 2-tap average where the two agree);
- random patterns (paper, pigment, hatching, brush strokes) are evaluated at **carried
  coordinates**: two coordinate layers moved along the flow every frame and reset to the screen
  grid in turn at fixed source frame numbers (every ``CARRY`` frames), cross-faded. A static
  scene keeps the screen grid (no flicker); a panning one carries its texture along; a scene
  cut (the flow cannot explain the new frame) starts afresh. The state at a frame depends only
  on the ``CARRY`` frames before it, which the processor warms up (``FxDef.history``), so a
  still equals the render.
- ``pattern_layers`` gives every random texture (these and risograph, frosted) its pattern's
  coordinates by the ``pattern`` param: ``content`` (carried), ``boil`` (a new seeded pattern
  every ``boil`` frames, jittered a little every frame) or ``screen``;
- a **periodic** pattern (halftone screen, Bayer table, engraving and hatching lines) never
  mixes carried layers: two drifted copies of it are out of phase, and their mix fades in and
  out every ``CARRY / 2`` frames (a flash). ``pattern_grid`` lays it as one grid; in
  ``content`` mode moved rigidly along the media's camera path (``Frame.camera``, measured
  once per file by ``fx.camera``), a function of the source frame alone.

Time effects keep past frames as CPU float16 copies (no VRAM held) and work on source frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

if __package__:  # in Cutwright's process
    from cutwright.modules.fx.runners.fx_base import (
        Frame,
        blur,
        downscale,
        hex_rgb,
        kernel,
        luma,
        sample,
        upscale,
    )
    from cutwright.modules.fx.runners.fx_constants import CARRY, JITTER
else:  # in the objects environment: the files next to the runner, by their bare names
    from fx_base import (  # type: ignore[no-redef]
        Frame,
        blur,
        downscale,
        hex_rgb,
        kernel,
        luma,
        sample,
        upscale,
    )
    from fx_constants import CARRY, JITTER  # type: ignore[no-redef]

FLOW_SIZE = 256  # the flow is worked out at about this many px on the short side
DEAD_ZONE = (0.2, 0.45)  # px: motion below the first is none, full from the second
CUT = 0.1  # mean luma left after the flow's warp above which the frame is a new shot


# --- motion ------------------------------------------------------------------------------------


def _grad(o: Any, a: Any) -> tuple[Any, Any]:
    """Central differences (gy, gx) of ``a`` (c, h, w), edges mirrored."""
    pad = o.pad(a, 1, 1)
    gy = (pad[:, 2:, 1:-1] - pad[:, :-2, 1:-1]) * 0.5
    return gy, (pad[:, 1:-1, 2:] - pad[:, 1:-1, :-2]) * 0.5


def flow(o: Any, cur: Any, prev: Any) -> tuple[Any, Any]:
    """Backward optical flow (fy, fx), each (h, w) in px: ``cur(p) ≈ prev(p + f(p))``.

    Pyramidal Lucas–Kanade on luma at ≤ ``FLOW_SIZE`` px, three iterations per level,
    Gaussian windows; the normal equations are regularised so flat and noise-only areas stay
    still; motion under ``DEAD_ZONE`` is dropped (a static noisy scene has almost none)."""
    preview_flow = getattr(o, "backward_flow", None)
    if preview_flow is not None:
        return preview_flow(cur, prev, DEAD_ZONE)
    xp = o.xp
    h, w = cur.shape
    q = max(1, math.ceil(min(h, w) / FLOW_SIZE))
    pyramid = [(downscale(o, cur[None], q), downscale(o, prev[None], q))]
    while min(pyramid[-1][0].shape[1:]) >= 48 and len(pyramid) < 4:
        a, b = pyramid[-1]
        pyramid.append((downscale(o, a, 2), downscale(o, b, 2)))
    fy = fx = None
    for a, b in reversed(pyramid):
        _, lh, lw = a.shape
        if fy is None:
            fy, fx = 0 * a, 0 * a
        else:
            sy, sx = lh / fy.shape[1], lw / fy.shape[2]
            fy, fx = upscale(o, fy, lh, lw) * sy, upscale(o, fx, lh, lw) * sx
        ys, xs = o.grid(lh, lw)
        for _ in range(3):
            warped = sample(o, b, ys + fy[0], xs + fx[0])
            gy, gx = _grad(o, (warped + a) * 0.5)
            e = a - warped
            terms = xp.stack([(gx * gx)[0], (gx * gy)[0], (gy * gy)[0], (gx * e)[0], (gy * e)[0]])
            sxx, sxy, syy, sxe, sye = blur(o, terms, 3.0)
            reg = 1e-3 + 1.5 * float((e * e).mean())  # noise in the residual: hold still
            sxx, syy = sxx + reg, syy + reg
            det = sxx * syy - sxy * sxy
            fx = fx + ((syy * sxe - sxy * sye) / det).clip(-2, 2)[None]
            fy = fy + ((sxx * sye - sxy * sxe) / det).clip(-2, 2)[None]
    _, lh, lw = fy.shape
    fy, fx = upscale(o, fy, h, w)[0] * (h / lh), upscale(o, fx, h, w)[0] * (w / lw)
    mag = xp.sqrt(fy * fy + fx * fx)
    keep = ((mag - DEAD_ZONE[0]) / (DEAD_ZONE[1] - DEAD_ZONE[0])).clip(0, 1)
    return fy * keep, fx * keep


@dataclass
class Motion:
    prev: Any  # the previous input frame warped onto this one (None on a copy's first frame)
    fy: Any
    fx: Any
    layers: Any  # ((coords (2, h, w), weight), …) for carried textures, or None


def motion(o: Any, x: Any, f: Frame, carry: bool = False) -> Motion:
    """This frame's motion against the previous one (``f.state``; frames come in order, a gap
    or a scene cut starts afresh) and, with ``carry``, the carried texture coordinates."""
    xp = o.xp
    _, h, w = x.shape
    st = f.state.setdefault("motion", {})
    gray = luma(x)
    fresh = st.get("index") != f.index - 1 or st.get("size") != (h, w)
    ys, xs = o.grid(h, w)
    prev = fy = fx = None
    if not fresh:
        fy, fx = flow(o, gray, st["gray"])
        seen = sample(o, st["gray"][None], ys + fy, xs + fx)[0]
        if float(o.numpy(xp.abs(gray - seen).mean())) > CUT:  # another shot
            fresh, fy, fx = True, None, None
        else:
            prev = sample(o, st["x"], ys + fy, xs + fx)
    layers = None
    if carry:
        here = xp.stack([ys, xs])
        if fresh or "a" not in st:
            a, b = here, here
        else:
            a, b = (sample(o, st[k], ys + fy, xs + fx) for k in ("a", "b"))
        phase = f.index % CARRY
        a = here if phase == 0 else a
        b = here if phase == CARRY // 2 else b
        wa = 1 - abs(phase - CARRY / 2) / (CARRY / 2)
        st["a"], st["b"] = a, b
        layers = ((a, wa), (b, 1 - wa))
    st.update(index=f.index, size=(h, w), gray=gray, x=x)
    return Motion(prev, fy, fx, layers)


def pattern_layers(o: Any, x: Any, p: dict[str, Any], f: Frame, m: Motion | None = None,
                   default: str = "content") -> Any:  # fmt: skip
    """Where the effect reads its pattern this frame, ((coords (2, h, w), weight), …), by
    ``p["pattern"]`` (``default`` when unset): ``content`` — carried along the motion (``m``'s
    layers, else this frame's ``motion``); ``boil`` — the screen grid turned a few degrees and
    moved to a new seeded place every ``p["boil"]`` frames (a periodic pattern too looks drawn
    anew) and jittered a little every frame, as printed or hand-drawn animation; ``screen`` —
    the screen grid."""
    _, h, w = x.shape
    mode = p.get("pattern", default)
    if mode == "content":
        return (m if m is not None and m.layers is not None else motion(o, x, f, True)).layers
    ys, xs = o.grid(h, w)
    if mode == "boil":
        place = np.random.default_rng([f.seed, 17, f.index // max(1, int(p.get("boil", 2)))])
        turn = math.radians(place.uniform(2, 6) * place.choice((-1.0, 1.0)))
        oy, ox = place.uniform(0, 4096, 2)
        jy, jx = np.random.default_rng([f.seed, 19, f.index]).normal(0, JITTER * min(h, w), 2)
        c, s = math.cos(turn), math.sin(turn)
        ys, xs = (ys * c + xs * s + float(oy + jy), xs * c - ys * s + float(ox + jx))
    return ((o.xp.stack([ys, xs]), 1.0),)


def pattern_grid(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Where a **periodic** pattern (halftone screen, Bayer table, engraved or hatched lines) is
    read this frame, coords (2, h, w): one grid, never carried layers — two drifted copies of
    a periodic pattern are out of phase and their mix fades in and out. ``content``: the screen
    grid moved rigidly along the camera path (``f.camera`` = [turn, x, y] px: frame pixel p is
    at c + R(turn)(p − c) + (x, y) on the media's first frame; the screen grid without it);
    ``boil`` and ``screen`` as ``pattern_layers``."""
    _, h, w = x.shape
    if p.get("pattern", "content") != "content":
        return pattern_layers(o, x, p, f)[0][0]
    ys, xs = o.grid(h, w)
    if f.camera is None:
        return o.xp.stack([ys, xs])
    turn, mx, my = (float(v) for v in f.camera)
    c, s = math.cos(turn), math.sin(turn)
    dy, dx = ys - (h - 1) / 2, xs - (w - 1) / 2
    return o.xp.stack([dx * s + dy * c + (h - 1) / 2 + my, dx * c - dy * s + (w - 1) / 2 + mx])


def steady(o: Any, x: Any, m: Motion, limit: float = 0.12) -> Any:
    """``x`` averaged with the previous frame warped onto it where they agree (noise and
    shimmer halve; a change of content wins)."""
    if m.prev is None:
        return x
    agree = (1 - o.xp.abs(x - m.prev).mean(0) / limit).clip(0, 1) * 0.5
    return x + (m.prev - x) * agree[None]


def noise_table(seed: int, n: int = 128) -> Any:
    """A tileable field of smooth Gaussian noise (numpy (1, n, n), std 1)."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n, n)).astype(np.float32)
    f = np.fft.fftfreq(n).astype(np.float32)
    smooth = np.fft.ifft2(np.fft.fft2(white) * np.exp(-2 * math.pi**2 * 0.8**2 * (
        f[:, None] ** 2 + f[None, :] ** 2))).real  # fmt: skip
    return (smooth / smooth.std())[None].astype(np.float32)


def _lookup(o: Any, table: Any, gy: Any, gx: Any) -> Any:
    """Bilinear read of a tileable table (1, n, n) at (gy, gx), wrapping."""
    n = table.shape[-1]
    xp = o.xp
    y0, x0 = xp.floor(gy), xp.floor(gx)
    wy, wx = gy - y0, gx - x0
    out = 0
    for dy, dx, k in ((0, 0, (1 - wy) * (1 - wx)), (0, 1, (1 - wy) * wx), (1, 0, wy * (1 - wx)),
                      (1, 1, wy * wx)):  # fmt: skip
        out = out + o.gather(table, (y0 + dy) % n, (x0 + dx) % n)[0] * k
    return out


def texture(o: Any, layers: Any, table: Any, scale: float, shape: tuple[int, int],
            angle: float = 0.0, stretch: float = 1.0) -> Any:  # fmt: skip
    """Noise (std ≈ 1, (h, w)) at ``scale`` px per cell, read at ``layers``' coordinates
    (``pattern_layers``; None: the screen grid), the layers mixed so the noise keeps its
    spread; optionally turned by ``angle`` and stretched along it."""
    h, w = shape
    table = table if not isinstance(table, np.ndarray) else o.asarray(table)
    layers = layers if layers is not None else ((o.xp.stack(o.grid(h, w)), 1.0),)
    c, s = math.cos(angle), math.sin(angle)
    total, norm = 0, 0.0
    for coords, wgt in layers:
        if wgt <= 0:
            continue
        ys, xs = coords[0], coords[1]
        u, v = (xs * c + ys * s) / (scale * stretch), (-xs * s + ys * c) / scale
        total = total + _lookup(o, table, v, u) * wgt
        norm += wgt * wgt
    return total / math.sqrt(norm)


# --- painterly ---------------------------------------------------------------------------------

ZETA = 0.33  # polynomial sector weights (Kyprianidis et al.): overlap at the centre …
SECTORS = 8
ETA = (ZETA + math.cos(math.pi / SECTORS)) / math.sin(math.pi / SECTORS) ** 2  # … and width


def orientation(o: Any, x: Any, sigma: float) -> tuple[Any, Any, Any]:
    """The smoothed structure tensor's flow direction (tx, ty) — along edges — and anisotropy
    A (0 flat … 1 a clean edge)."""
    xp = o.xp
    gy, gx = _grad(o, x)
    e, f_, g = blur(o, xp.stack([(gx * gx).sum(0), (gx * gy).sum(0), (gy * gy).sum(0)]), sigma)
    root = xp.sqrt((e - g) ** 2 + 4 * f_ * f_)
    l1, l2 = (e + g + root) / 2, (e + g - root) / 2
    tx, ty = l1 - e, -f_
    norm = xp.sqrt(tx * tx + ty * ty)
    flat = (norm < 1e-9) * 1.0
    tx, ty = tx / (norm + flat) + flat, ty / (norm + flat)
    return tx, ty, (l1 - l2) / (l1 + l2 + 1e-9)


def kuwahara(o: Any, x: Any, radius: float, sharpness: float) -> Any:
    """Anisotropic Kuwahara filter with polynomial sector weights: each pixel takes the mean
    of the most uniform of eight sectors of an ellipse along the local structure (long along
    edges), so areas go flat like paint and edges stay. A sector's uniformity is its luma
    variance; offsets beyond 4 per side are strided over a slightly blurred copy (the cost
    stays ~50 offsets at any size)."""
    xp = o.xp
    _, h, w = x.shape
    tx, ty, aniso = orientation(o, x, max(1.0, radius / 3))
    a = radius * (1 + aniso)  # along the edge
    b = radius / (1 + aniso)  # across it
    reach = max(1, math.ceil(2 * radius))
    stride = max(1, math.ceil(reach / 4))
    src = blur(o, x, stride / 2) if stride > 1 else x
    src = xp.stack([src[0], src[1], src[2], luma(src)])  # RGB + luma
    py, px = min(reach, h - 1), min(reach, w - 1)
    pad = o.pad(src, py, px)
    angles = np.arange(SECTORS) * 2 * math.pi / SECTORS
    ck = o.asarray(np.cos(angles))[:, None, None]
    sk = o.asarray(np.sin(angles))[:, None, None]
    weights = 0 * xp.stack([tx] * SECTORS)
    sums = 0 * xp.stack([src] * SECTORS)  # (sectors, RGB + luma, h, w)
    squares = 0 * weights
    steps = range(-(reach // stride) * stride, reach + 1, stride)
    for dy in steps:
        for dx in steps:
            if abs(dy) > py or abs(dx) > px or dy * dy + dx * dx > reach * reach:
                continue
            u = (dx * tx + dy * ty) / a
            v = (-dx * ty + dy * tx) / b
            r2 = u * u + v * v
            g = xp.exp(-2 * r2) * (r2 <= 1)
            uk, vk = ck * u + sk * v, -sk * u + ck * v
            wk = (uk + ZETA - ETA * vk * vk).clip(0, None) ** 2 * g
            near = pad[:, py + dy : py + dy + h, px + dx : px + dx + w]
            weights += wk
            sums += wk[:, None] * near[None]
            squares += wk * (near[3] * near[3])[None]
    weights = weights + 1e-6
    mk = sums[:, :3] / weights[:, None]
    lk = sums[:, 3] / weights
    var = 3 * (squares / weights - lk * lk).clip(0, None)  # ~ the sum of the channels'
    alpha = 1 / (1 + (255 * var) ** (sharpness / 2))
    return (mk * alpha[:, None]).sum(0) / alpha.sum(0)[None]


TABLES: dict[tuple[int, int], Any] = {}


def _table(seed: int, n: int = 128) -> Any:
    """``noise_table(seed, n)``, kept (a bigger ``n`` repeats less often for a fine grain)."""
    if (seed, n) not in TABLES:
        if len(TABLES) > 32:
            TABLES.clear()
        TABLES[(seed, n)] = noise_table(seed, n)
    return TABLES[(seed, n)]


@kernel("oil")
def oil(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Oil paint: an anisotropic Kuwahara filter (flat strokes along the picture's structure,
    edges kept), worked at up to 1080 px on the short side; ``texture`` adds brush strokes —
    noise (where ``pattern`` puts it) smeared along the stroke direction and lit like relief."""
    _, h, w = x.shape
    tex = float(p["texture"])
    m = motion(o, x, f, carry=tex > 0 and p.get("pattern", "content") == "content")
    x0 = steady(o, x, m)
    q = max(1, math.ceil(min(h, w) / 1080))
    work = downscale(o, x0, q) if q > 1 else x0
    short = min(work.shape[1:])
    out = kuwahara(o, work, max(1.0, float(p["radius"]) * short), float(p["sharpness"]))
    out = upscale(o, out, h, w) if q > 1 else out
    if tex > 0:
        tx, ty, _ = orientation(o, out, max(1.0, 0.004 * min(h, w)))
        size = max(1.0, 0.0025 * min(h, w))
        noise = o.asarray(_table(f.seed))
        strokes = 0
        layers = pattern_layers(o, x, p, f, m)
        norm = math.sqrt(sum(wgt * wgt for _, wgt in layers))
        for coords, wgt in layers:  # noise smeared along the stroke (a short line integral)
            if wgt <= 0:
                continue
            line = 0
            for j in range(-4, 5):
                line = line + _lookup(o, noise, (coords[0] + j * 1.5 * size * ty) / size,
                                      (coords[1] + j * 1.5 * size * tx) / size)  # fmt: skip
            strokes = strokes + line / 3 * wgt
        strokes = strokes / norm
        gy, gx = _grad(o, strokes[None])
        relief = (gx + gy)[0] * 0.5  # light from the top-left
        out = out * (1 + tex * (0.05 * strokes + 0.12 * relief))[None]
    y = luma(out)[None]
    return (y + float(p["saturation"]) * (out - y)).clip(0, 1)


@kernel("watercolor")
def watercolor(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Watercolour: the picture simplified into washes with wobbly edges, pigment darkening
    where washes end, pigment pooling in the paper's grain (granulation), uneven wash density
    and a paper texture; the patterns where ``pattern`` puts them (carried along the motion by
    default)."""
    _, h, w = x.shape
    xp = o.xp
    short = min(h, w)
    m = motion(o, x, f, carry=p.get("pattern", "content") == "content")
    x0 = steady(o, x, m)
    layers = pattern_layers(o, x, p, f, m)
    table = o.asarray(_table(f.seed))
    bleed = float(p["bleed"]) * 0.006 * short
    if bleed > 0.05:
        ys, xs = o.grid(h, w)
        dy = texture(o, layers, table, 0.04 * short, (h, w))
        dx = texture(o, layers, table, 0.04 * short, (h, w), angle=1.3)
        x0 = sample(o, x0, ys + bleed * dy, xs + bleed * dx)
    soft = kuwahara(o, x0, max(1.0, float(p["abstraction"]) * short), 4)
    soft = blur(o, soft, 0.0012 * short).clip(0.004, 1) ** 0.65  # washes are lighter, softer
    density = -xp.log(soft)  # pigment, as optical density (Beer–Lambert)
    edge = xp.abs(soft - blur(o, soft, max(1.0, 0.006 * short))).mean(0)
    fine = texture(o, layers, table, max(1.5, 0.004 * short), (h, w), angle=0.7)
    wash = texture(o, layers, table, 0.07 * short, (h, w), angle=2.1)
    gain = (1 + 0.6 * float(p["edges"]) * (edge * 12).clip(0, 1)  # pooled at the wash's edge
            + 0.12 * wash + 0.14 * float(p["granulation"]) * fine)  # fmt: skip
    paper = texture(o, layers, table, max(1.0, 0.006 * short), (h, w), angle=0.3, stretch=2.5)
    sheet = o.asarray(hex_rgb(p["paper_color"]))[:, None, None]
    sheet = sheet * (1 + 0.03 * float(p["paper"]) * (paper + 0.5 * fine))[None]
    return (sheet * xp.exp(-density * gain.clip(0.3, None)[None])).clip(0, 1)


def xdog(o: Any, y: Any, sigma: float, lines: float) -> Any:
    """Ink of an extended difference of Gaussians (0 paper … 1 ink): dark lines where the
    picture darkens across an edge, soft-thresholded so they do not crawl."""
    xp = o.xp
    dog = blur(o, y[None], sigma)[0] - 0.98 * blur(o, y[None], 1.6 * sigma)[0]
    eps = -0.004 * (1.5 - lines)
    return (-xp.tanh(60 * (dog - eps))).clip(0, 1) * min(1.0, 0.4 + lines)


@kernel("ink")
def ink(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Ink drawing: XDoG lines along the picture's edges, hatching in the darker tones (a second
    crossing layer in the darkest), on paper; the hatching where ``pattern`` puts it (carried
    along the motion by default)."""
    _, h, w = x.shape
    xp = o.xp
    short = min(h, w)
    hatching = float(p["hatching"])
    y = luma(steady(o, x, motion(o, x, f)))
    ink_ = xdog(o, y, max(0.6, float(p["width"]) * short), float(p["lines"]))
    if hatching > 0:
        tone = blur(o, y[None], max(1.0, 0.004 * short))[0]
        period = max(3.0, float(p["spacing"]) * short)
        coords = pattern_grid(o, x, p, f)
        for depth, turn in ((0.45, 0.0), (0.22, math.pi / 2)):
            cover = ((depth - tone) / depth * (0.5 + hatching)).clip(0, 0.9)
            angle = math.radians(float(p["angle"])) + turn
            c, s = math.cos(angle), math.sin(angle)
            wave = xp.cos(2 * math.pi * (coords[1] * c + coords[0] * s) / period)
            stroke = ((wave - (1 - 2 * cover)) * period / 4).clip(0, 1) * (cover > 0.02)
            ink_ = xp.maximum(ink_, stroke * 0.85)
    paper = o.asarray(hex_rgb(p["paper"]))[:, None, None]
    pen = o.asarray(hex_rgb(p["ink"]))[:, None, None]
    return paper + (pen - paper) * ink_[None]


@kernel("toon")
def toon(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Cel / toon shading: the picture smoothed, its luma put on ``levels`` flat steps (soft
    step edges, so they do not flicker), colour kept and lifted, dark outlines (XDoG)."""
    _, h, w = x.shape
    xp = o.xp
    short = min(h, w)
    m = motion(o, x, f)
    x0 = blur(o, steady(o, x, m), max(0.5, float(p["smooth"]) * short))
    y = luma(x0).clip(1e-3, 1)
    n = int(p["levels"])
    k = y * n - 0.5
    base = xp.floor(k)
    t = ((k - base - 0.5) / 0.12 + 0.5).clip(0, 1)
    flat = ((base + t * t * (3 - 2 * t) + 0.5) / n).clip(0.02, 1)
    out = x0 * (flat / y)[None]
    grey = luma(out)[None]
    out = grey + float(p["saturation"]) * (out - grey)
    lines = xdog(o, luma(x0), max(0.6, 0.0025 * short), float(p["lines"]))
    return (out * (1 - lines[None])).clip(0, 1)


# --- time --------------------------------------------------------------------------------------


def remember(o: Any, x: Any, f: Frame, keep: int) -> dict[int, Any]:
    """The last ``keep`` input frames {source index: CPU float16 copy}, this one included (a gap
    in the frame numbers starts afresh)."""
    ring = f.state.get("ring", [])
    if ring and ring[-1][0] != f.index - 1:
        ring = []
    ring = [*ring, (f.index, np.asarray(o.numpy(x), np.float16))][-max(1, keep) :]
    f.state["ring"] = ring
    return dict(ring)


@kernel("echo")
def echo(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Echo: ``frames`` earlier frames (every ``spacing`` frames) over the picture, fading by
    ``decay`` each; ``average`` (ghosts), ``lighten`` (light trails) or ``screen``."""
    xp = o.xp
    n, gap, decay = int(p["frames"]), int(p["spacing"]), float(p["decay"])
    past = remember(o, x, f, n * gap + 1)
    mode = p["mode"]
    total, weights = x * 1, 1.0
    for j in range(1, n + 1):
        old = past.get(f.index - j * gap)
        if old is None:
            break
        img, k = o.asarray(old), decay**j
        if mode == "average":
            total, weights = total + img * k, weights + k
        elif mode == "lighten":
            total = xp.maximum(total, img * k)
        else:
            total = 1 - (1 - total) * (1 - img * k)
    return (total / weights).clip(0, 1)


def _is_new(seed: int, i: int, step: int, irregular: bool) -> bool:
    if not irregular:
        return i % step == 0
    return i % (2 * step) == 0 or bool(np.random.default_rng([seed, i, 7]).random() < 1 / step)


@kernel("step_print")
def step_print(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Step printing: a new picture every ``step`` source frames (``irregular``: at seeded,
    uneven moments), held in between; each new picture is the average of the last
    ``shutter`` × step frames (a long shutter: warm, smeared motion). ``shutter`` 0 is a hard
    strobe / stutter."""
    step, irregular = int(p["step"]), bool(p.get("irregular", False))
    span = max(1, round(float(p["shutter"]) * step))
    past = remember(o, x, f, 2 * step + span + 1)
    start = f.index
    while not _is_new(f.seed, start, step, irregular) and start > f.index - 2 * step:
        start -= 1
    if f.state.get("block") == start and "held" in f.state:
        return o.asarray(f.state["held"])
    frames = [past[i] for i in range(start - span + 1, start + 1) if i in past]
    if not frames:  # the copy began inside the block: the oldest frame there is
        frames = [past[min(past)]]
    held = np.mean(np.stack([a.astype(np.float32) for a in frames]), 0)
    f.state["block"], f.state["held"] = start, held
    return o.asarray(held)


@kernel("slit_scan")
def slit_scan(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Slit-scan: each row (or column) shows the picture from a different moment — ``now`` at
    one edge, ``span`` frames ago at the other, blended between frames."""
    _, h, w = x.shape
    n = int(p["span"])
    past = remember(o, x, f, n + 1)
    oldest = min(past)
    direction = p["direction"]
    lines = h if direction in ("down", "up") else w
    age = np.arange(lines) / max(1, lines - 1) * n
    if direction in ("up", "left"):
        age = age[::-1]
    t0 = f.index - np.floor(age).astype(int)
    frac = (age - np.floor(age)).astype(np.float32)
    t1 = np.maximum(t0 - 1, oldest)
    t0 = np.maximum(t0, oldest)
    # Only the one row (column) each line takes from its two frames: stacking every remembered
    # frame as float32 cost ~2.5 GB per 4K frame to read one line of each.
    if direction in ("down", "up"):
        a = np.stack([past[t][:, r, :] for r, t in enumerate(t0)], 1).astype(np.float32)
        b = np.stack([past[t][:, r, :] for r, t in enumerate(t1)], 1).astype(np.float32)
        out = a * (1 - frac)[None, :, None] + b * frac[None, :, None]
    else:
        a = np.stack([past[t][:, :, c] for c, t in enumerate(t0)], 2).astype(np.float32)
        b = np.stack([past[t][:, :, c] for c, t in enumerate(t1)], 2).astype(np.float32)
        out = a * (1 - frac)[None, None, :] + b * frac[None, None, :]
    return o.asarray(out)


@kernel("datamosh")
def datamosh(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Datamosh: the key frame's pixels (one every ``period`` source frames) are not refreshed
    but dragged along the motion of the frames after it, in macroblocks of ``block`` — the
    picture melts into the movement. Mixed in by ``amount``."""
    _, h, w = x.shape
    m = motion(o, x, f)
    held = f.state.get("held")
    if held is None or m.prev is None or f.index % int(p["period"]) == 0:
        held = x
    else:
        fy, fx = m.fy, m.fx
        size = round(float(p["block"]) * min(h, w))
        ys, xs = o.grid(h, w)
        if size >= 2:  # one motion per macroblock, as a codec would
            small = downscale(o, o.xp.stack([fy, fx]), size)
            iy = (ys // size).clip(0, small.shape[1] - 1)
            ix = (xs // size).clip(0, small.shape[2] - 1)
            fy, fx = o.gather(small, iy, ix)
        held = sample(o, held, ys + fy, xs + fx)
    f.state["held"] = held
    return x + (held - x) * float(p["amount"])
