"""Shared cell analysis, camera-anchored grids and periodic scalar fields."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

if cv2 is not None and (threads := int(os.environ.get("CUTWRIGHT_CORE_THREADS", "0"))) > 0:
    cv2.setNumThreads(threads)

if __package__:
    from cutwright.modules.fx.runners.fx_base import NumpyOps, blur, hblur, hex_rgb, sample, upscale
else:
    from fx_base import NumpyOps, blur, hblur, hex_rgb, sample, upscale

_BASE_OPS = NumpyOps()


class CellOps(NumpyOps):
    def sample_bilinear(self, x: Any, gy: Any, gx: Any, border: str) -> Any:
        if np.ndim(gy) != 2 or np.ndim(gx) != 2:
            return None
        gy, gx = np.broadcast_arrays(gy, gx)
        channels, h, w = x.shape
        y0, x0 = np.floor(gy).astype(np.int32), np.floor(gx).astype(np.int32)
        fy, fx = gy - y0.astype(gy.dtype), gx - x0.astype(gx.dtype)
        out = np.zeros((channels, *gy.shape), dtype=np.result_type(x.dtype, gy.dtype, gx.dtype))
        flat = x.reshape(channels, -1)
        for dy, dx, weight in (
            (0, 0, (1 - fy) * (1 - fx)),
            (0, 1, (1 - fy) * fx),
            (1, 0, fy * (1 - fx)),
            (1, 1, fy * fx),
        ):
            yy, xx = y0 + dy, x0 + dx
            if border == "zero":
                weight = weight * ((yy >= 0) & (yy < h) & (xx >= 0) & (xx < w))
            indices = np.clip(yy, 0, h - 1) * w + np.clip(xx, 0, w - 1)
            out += np.take(flat, indices, axis=1) * weight[None]
        return out

    def gaussian_blur(self, x: Any, sigma: float) -> Any:
        return gaussian(x, sigma)

    def horizontal_blur(self, x: Any, sigma: float) -> Any:
        if cv2 is not None and sigma >= 1:
            size = 2 * math.ceil(4 * sigma) + 1
            return np.stack(
                [
                    cv2.GaussianBlur(plane, (size, 1), sigma, borderType=cv2.BORDER_REFLECT_101)
                    for plane in x
                ]
            )
        return hblur(_BASE_OPS, x, sigma)


OPS = CellOps()
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
# Per-level bloom tints, finest halo first: warm runs amber to deep red, prism magenta to blue.
BLOOM_PALETTES = {
    "warm": np.array(
        [[1, 0.62, 0.22], [1, 0.45, 0.15], [0.9, 0.18, 0.08], [0.7, 0.1, 0.08]], np.float32
    )[:, :, None, None],
    "prism": np.array(
        [[1, 0.22, 0.8], [0.95, 0.25, 0.95], [0.45, 0.45, 1], [0.3, 0.55, 1]], np.float32
    )[:, :, None, None],
}


def streak_taps(length: float, angle: float) -> dict[tuple[int, int], float]:
    """Bilinear taps about one pixel apart, so a streak is a smooth trail rather than copies.

    Falloff along the streak and its total energy follow the original 22-tap design."""
    count = max(22, math.ceil(length) + 1)
    unit = 21 / (count - 1)
    taps: dict[tuple[int, int], float] = {}
    for step in range(count):
        distance = step * length / (count - 1)
        sy, sx = -distance * math.sin(angle), -distance * math.cos(angle)
        iy, ix = math.floor(sy), math.floor(sx)
        fy, fx = sy - iy, sx - ix
        for dy, dx, weight in (
            (0, 0, (1 - fy) * (1 - fx)),
            (0, 1, (1 - fy) * fx),
            (1, 0, fy * (1 - fx)),
            (1, 1, fy * fx),
        ):
            key = iy + dy, ix + dx
            taps[key] = taps.get(key, 0) + weight * 0.88 ** (step * unit) * unit
    return taps


def rgb(color: str) -> Any:
    return np.array(hex_rgb(color), np.float32)[:, None, None]


def lum(x: Any) -> Any:
    return (x * LUMA[:, None, None]).sum(0)


def gradient(x: Any) -> tuple[Any, Any]:
    return tuple(
        np.gradient(x, axis=axis) if size > 1 else np.zeros_like(x)
        for axis, size in enumerate(x.shape)
    )


def smooth(a: float, b: float, x: Any) -> Any:
    v = np.clip((x - a) / max(b - a, 1e-6), 0, 1)
    return v * v * (3 - 2 * v)


def noise(y: Any, x: Any, seed: int = 0) -> Any:
    return np.mod(np.sin(x * 127.1 + y * 311.7 + seed * 0.137) * 43758.5453, 1).astype(np.float32)


def clock(p: dict[str, Any], f: Any) -> float:
    return float(p.get("_t", f.index / f.fps))


def field(y: Any, x: Any, p: dict[str, Any], t: float) -> Any:
    phase = 2 * math.pi * ((t / float(p.get("loop_seconds", 5))) % 1)
    scale = float(p.get("field_scale", 4))
    xx, yy = (x - 0.5) * scale, (y - 0.5) * scale
    r, a = np.hypot(xx, yy), np.arctan2(yy, xx)
    mode = p.get("motion", "still")
    if mode == "still":
        return np.zeros_like(xx)
    if mode == "ripple":
        v = np.sin(2 * math.pi * r - phase)
    elif mode == "sweep":
        v = np.sin(2 * math.pi * (xx * math.cos(phase) + yy * math.sin(phase)))
    elif mode == "spiral":
        v = np.sin(2 * math.pi * r + 3 * a - phase)
    elif mode == "flow":
        v = np.sin(xx * 2 + np.cos(yy * 2 + phase)) * np.cos(yy * 2 - phase)
    elif mode == "contour":
        v = np.sin(4 * (np.sin(xx + phase) + np.cos(yy - phase)))
    elif mode == "warp":
        v = np.sin(xx * 3 + 2 * np.sin(yy * 2 + phase) - phase)
    elif mode == "chladni":
        v = np.cos(xx * 3 + phase) * np.cos(yy * 4) - np.cos(yy * 3 - phase) * np.cos(xx * 4)
    elif mode == "moire":
        v = np.sin(xx * 9 + phase) * np.sin(yy * 9 - phase)
    else:
        v = (
            np.cos(3 * np.hypot(xx - 1, yy) - phase) + np.cos(3 * np.hypot(xx + 1, yy) + phase)
        ) / 2
    return v.astype(np.float32)


@dataclass
class Grid:
    colors: Any
    tone: Any
    iy: Any
    ix: Any
    py: Any
    px: Any
    cy: Any
    cx: Any
    ch: int
    cw: int

    def expand(self, a: Any) -> Any:
        return a[..., self.iy, self.ix]


def grid(x: Any, p: dict[str, Any], f: Any, aspect: float = 1, *, fractional: bool = False) -> Grid:
    _, h, w = x.shape
    cell = float(p.get("cell_size", 20)) * w / float(p.get("reference_width", 1200))
    ch = max(2, round(cell))
    cw = max(2, round(ch * aspect))
    columns = int(p.get("columns", 0))
    if columns:
        cw = max(2, round(w / columns))
        ch = max(2, round(cw / aspect))
    if fractional:
        ch = max(2, cell)
        cw = max(2, ch * aspect)
        if columns:
            cw = max(2, w / columns)
            ch = max(2, cw / aspect)
        if p.get("_rows"):
            ch = h / max(1, int(p["_rows"]))
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    angle, dx, dy = (
        f.camera if f.camera is not None and p.get("pattern", "content") == "content" else (0, 0, 0)
    )
    c, s = math.cos(angle), math.sin(angle)
    center_x, center_y = (w - 1) / 2, (h - 1) / 2
    wx = c * (xx - center_x) - s * (yy - center_y) + center_x + dx
    wy = s * (xx - center_x) + c * (yy - center_y) + center_y + dy
    x0, y0 = math.floor(float(wx.min()) / cw), math.floor(float(wy.min()) / ch)
    ix = np.floor(wx / cw).astype(np.int32) - x0
    iy = np.floor(wy / ch).astype(np.int32) - y0
    gy, gx = np.mgrid[: int(iy.max()) + 1, : int(ix.max()) + 1].astype(np.float32)
    ax, ay = ((gx + x0 + 0.5) * cw - center_x - dx, (gy + y0 + 0.5) * ch - center_y - dy)
    sx, sy = c * ax + s * ay + center_x, -s * ax + c * ay + center_y
    colors = (
        sum(
            sample(OPS, x, sy - s * ox * cw + c * oy * ch, sx + c * ox * cw + s * oy * ch)
            for oy, ox in ((0, 0), (-0.25, 0), (0.25, 0), (0, -0.25), (0, 0.25))
        )
        / 5
    )
    tone = np.clip(
        (lum(colors) - 0.5) * float(p.get("contrast", 1)) + 0.5 + float(p.get("brightness", 0)),
        0,
        1,
    ) ** (1 / float(p.get("gamma", 1)))
    drive = float(p.get("field_amount", 0.35))
    drive += float(p.get("audio_amount", 0.5)) * float(p.get("_bass", 0))
    v = field((gy + y0) * ch / h, (gx + x0) * cw / w, p, clock(p, f))
    mode = p.get("field_blend", "offset")
    if mode == "replace" and p.get("motion", "still") != "still":
        tone = (1 - drive) * tone + drive * (v + 1) / 2
    elif mode == "multiply":
        tone *= 1 + drive * v
    else:
        tone += drive * v
    return Grid(
        colors,
        np.clip(tone, 0, 1),
        iy,
        ix,
        wy % ch / ch,
        wx % cw / cw,
        gy + y0,
        gx + x0,
        round(ch),
        round(cw),
    )


def ink(g: Grid, p: dict[str, Any]) -> Any:
    if p.get("color") == "source":
        y = lum(g.colors)[None]
        color = np.clip(y + (g.colors - y) * float(p.get("saturation", 1)), 0, 1)
        return np.clip(color + float(p.get("lift", 0)), 0, 1) * float(p.get("gain", 1))
    if p.get("color") == "ember":
        v = g.tone
        return np.stack([0.6 + v * 0.4, 0.15 + v * 0.75, 0.12 + v * 0.45]) * float(p.get("gain", 1))
    return np.broadcast_to(rgb(p.get("ink", "#f2f2f2")), g.colors.shape) * float(p.get("gain", 1))


def zone(x: Any, p: dict[str, Any]) -> Any:
    y = lum(x)
    k = float(p.get("softness", 0.25))
    mode = p.get("appears", "all")
    if mode == "shadows":
        return 1 - smooth(0.3 - k / 2, 0.3 + k / 2, y)
    if mode == "highlights":
        return smooth(0.65 - k / 2, 0.65 + k / 2, y)
    if mode == "midtones":
        return smooth(0.25 - k / 2, 0.25 + k / 2, y) * (1 - smooth(0.75 - k / 2, 0.75 + k / 2, y))
    if mode == "edges":
        gy, gx = np.gradient(y) if min(y.shape) > 1 else (0 * y, 0 * y)
        return smooth(0.01, 0.01 + k / 3, np.hypot(gx, gy))
    return np.ones_like(y)


def resize(x: Any, h: int, w: int) -> Any:
    if x.shape[1:] == (h, w):
        return x
    if cv2 is not None:
        return np.stack([cv2.resize(plane, (w, h), interpolation=cv2.INTER_LINEAR) for plane in x])
    return upscale(OPS, x, h, w)


def gaussian(x: Any, radius: float) -> Any:
    if cv2 is not None and radius >= 1:
        size = 2 * math.ceil(4 * radius) + 1
        return np.stack(
            [
                cv2.GaussianBlur(plane, (size, size), radius, borderType=cv2.BORDER_REFLECT_101)
                for plane in x
            ]
        )
    return blur(_BASE_OPS, x, max(0.1, radius))


def zero_taps(x: Any, taps: dict[tuple[int, int], float]) -> Any:
    """Constant bilinear shifts without rebuilding pixel coordinates for every tap."""
    _, h, w = x.shape
    out = x * 0
    for (dy, dx), weight in taps.items():
        y0, y1 = max(0, -dy), min(h, h - dy)
        x0, x1 = max(0, -dx), min(w, w - dx)
        if y1 > y0 and x1 > x0:
            out[:, y0:y1, x0:x1] += x[:, y0 + dy : y1 + dy, x0 + dx : x1 + dx] * weight
    return out
