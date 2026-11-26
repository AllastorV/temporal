"""Shared NumPy/Torch optical kernels, including alpha-correct spatial integration."""

import math
from typing import Any

if __package__:
    from .fx_base import Frame, blur, hex_rgb, kernel, luma, sample, screen
    from .fx_kernels_paint import flow
else:
    from fx_base import Frame, blur, hex_rgb, kernel, luma, sample, screen
    from fx_kernels_paint import flow


def _gaussian(o: Any, x: Any, sigma: float) -> Any:
    # At sub-two-pixel sigma, a continuous frequency response has negative lobes after
    # sampling. Use the normalized discrete Gaussian so clipping cannot create energy.
    if sigma == 0:
        return x
    if sigma >= 2:
        return blur(o, x, sigma)
    n = max(1, math.ceil(4 * sigma))
    weights = [math.exp(-j * j / (2 * sigma * sigma)) for j in range(n + 1)]
    norm = weights[0] + 2 * sum(weights[1:])
    h, w = x.shape[1:]
    py, px = min(h - 1, n), min(w - 1, n)
    padded = o.pad(x, py, px)
    hh, ww = padded.shape[1:]

    def response(freq: Any) -> Any:
        value = freq * 0 + weights[0]
        for j in range(1, n + 1):
            value = value + 2 * weights[j] * o.xp.cos(2 * math.pi * j * freq)
        return value / norm

    g = response(o.fftfreq(hh))[:, None] * response(o.rfftfreq(ww))[None]
    return o.fft_blur(padded, g, (hh, ww))[:, py : py + h, px : px + w]


def _spatial(o: Any, x: Any, f: Frame, mix: float, transform: Any) -> Any:
    if mix == 0:
        return x
    if f.alpha is None:
        return (x * (1 - mix) + transform(x) * mix).clip(0, 1)
    alpha = f.alpha
    if alpha.ndim == 2:
        alpha = alpha[None]
    premul = x * alpha
    out_alpha = (alpha * (1 - mix) + transform(alpha) * mix).clip(0, 1)
    out = premul * (1 - mix) + transform(premul) * mix
    f.state["alpha_replace"] = out_alpha[0]
    return (out / out_alpha.clip(1e-6, None)).clip(0, 1)


@kernel("gaussian_blur")
def gaussian(o: Any, x: Any, p: dict, f: Frame) -> Any:
    sigma = float(p["radius"]) * min(x.shape[1:]) / 1080
    return _spatial(o, x, f, float(p["mix"]), lambda a: _gaussian(o, a, sigma))


def _integrate(o: Any, a: Any, yy: Any, xx: Any, dy: Any, dx: Any, n: int) -> Any:
    acc = a * 0
    # Midpoint quadrature avoids overweighting the two ends of the shutter interval.
    for j in range(n):
        t = (j + 0.5) / n - 0.5
        acc = acc + sample(o, a, yy + t * dy, xx + t * dx)
    return acc / n


def _line_taps(dy: float, dx: float, n: int) -> dict[tuple[int, int], float]:
    """``_integrate``'s samples along a constant offset as whole-pixel taps: each midpoint
    sample is bilinear between four pixels with the same weights everywhere (the offset is the
    same for every pixel), so the n samples add up to a few dozen taps."""
    taps: dict[tuple[int, int], float] = {}
    for j in range(n):
        t = (j + 0.5) / n - 0.5
        ky, kx = math.floor(t * dy), math.floor(t * dx)
        wy, wx = t * dy - ky, t * dx - kx
        for oy, ox, k in ((0, 0, (1 - wy) * (1 - wx)), (0, 1, (1 - wy) * wx),
                          (1, 0, wy * (1 - wx)), (1, 1, wy * wx)):  # fmt: skip
            if k:
                key = (ky + oy, kx + ox)
                taps[key] = taps.get(key, 0.0) + k / n
    return taps


@kernel("directional_blur")
def directional(o: Any, x: Any, p: dict, f: Frame) -> Any:
    distance = float(p["distance"]) * min(x.shape[1:]) / 1080
    angle = math.radians(float(p["angle"]))
    taps = _line_taps(distance * math.sin(angle), distance * math.cos(angle), int(p["samples"]))
    return _spatial(o, x, f, float(p["mix"]), lambda a: o.edge_taps(a, taps))


@kernel("radial_blur")
def radial(o: Any, x: Any, p: dict, f: Frame) -> Any:
    if float(p["strength"]) == 0:
        return x
    h, w = x.shape[1:]
    yy, xx = o.grid(h, w)
    cx, cy = float(p["center"][0]) * (w - 1), float(p["center"][1]) * (h - 1)
    dx, dy = xx - cx, yy - cy
    n, strength = int(p["samples"]), float(p["strength"])

    def transform(a: Any) -> Any:
        acc = a * 0
        for j in range(n):
            t = ((j + 0.5) / n - 0.5) * strength
            if p["mode"] == "spin":
                theta = t * math.pi / 2
                gx = cx + dx * math.cos(theta) - dy * math.sin(theta)
                gy = cy + dx * math.sin(theta) + dy * math.cos(theta)
            else:
                gx, gy = cx + dx * (1 + t * 0.75), cy + dy * (1 + t * 0.75)
            acc = acc + sample(o, a, gy, gx)
        return acc / n

    return _spatial(o, x, f, float(p["mix"]), transform)


@kernel("motion_blur")
def motion(o: Any, x: Any, p: dict, f: Frame) -> Any:
    previous = f.state.get("motion_previous")
    index = f.state.get("motion_index")
    f.state["motion_previous"] = x.clone() if hasattr(x, "clone") else x.copy()
    f.state["motion_index"] = f.index
    shutter = float(p["shutter_angle"]) / 360
    if (
        previous is None
        or index != f.index - 1
        or previous.shape != x.shape
        or min(x.shape[1:]) < 8
        or shutter == 0
        or float(p["mix"]) == 0
    ):
        return x
    cur, prev = luma(x), luma(previous)
    if float(o.numpy(abs(cur - prev).mean())) > float(p["scene_cut"]):
        return x
    fy, fx = flow(o, cur, prev)
    cap = float(p["max_motion"]) * min(x.shape[1:]) / 1080
    length = (fy * fy + fx * fx + 1e-12) ** 0.5
    scale = (cap / length).clip(0, 1) * shutter
    yy, xx = o.grid(*x.shape[1:])
    return _spatial(
        o,
        x,
        f,
        float(p["mix"]),
        lambda a: _integrate(o, a, yy, xx, fy * scale, fx * scale, int(p["samples"])),
    )


@kernel("flash")
def flash(o: Any, x: Any, p: dict, f: Frame) -> Any:
    if float(p["strength"]) == 0:
        return x
    time = f.index / f.fps - float(p.get("_origin", 0)) - float(p["start"])
    if time < 0:
        return x
    hz, duration = float(p["repeat_hz"]), float(p["duration"])
    if hz > 0:
        time %= 1 / hz
        duration = min(duration, 1 / hz)
    if time >= duration:
        return x
    ramp = duration * float(p["fade"]) / 2
    envelope = min(1.0, time / ramp, (duration - time) / ramp) if ramp > 0 else 1.0
    envelope = envelope * envelope * (3 - 2 * envelope)
    color = o.asarray(hex_rgb(p["color"]))[:, None, None]
    return screen(x, color * float(p["strength"]) * envelope).clip(0, 1)


@kernel("glow")
def glow(o: Any, x: Any, p: dict, f: Frame) -> Any:
    amount = float(p["intensity"]) * float(p["mix"])
    if amount == 0:
        return x
    light = luma(x)
    knee, threshold = float(p["knee"]), float(p["threshold"])
    weight = ((light - threshold + knee) / (2 * knee)).clip(0, 1)
    weight = weight * weight * (3 - 2 * weight)
    source = x * weight[None]
    if f.alpha is not None:
        source = source * f.alpha
    sigma = float(p["radius"]) * min(x.shape[1:]) / 1080
    color = o.asarray(hex_rgb(p["color"]))[:, None, None]
    spread = (blur(o, source, sigma) * amount * color).clip(0, 1)
    if f.alpha is not None:
        alpha = f.alpha if f.alpha.ndim == 2 else f.alpha[0]
        energy = o.xp.maximum(o.xp.maximum(spread[0], spread[1]), spread[2])
        coverage = screen(alpha, energy).clip(0, 1)
        f.state["alpha_replace"] = coverage
        return (screen(x * alpha[None], spread) / coverage[None].clip(1e-6, 1)).clip(0, 1)
    return screen(x, spread).clip(0, 1)
