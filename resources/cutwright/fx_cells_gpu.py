"""Device-resident cell drawing; fonts and seeded cell choices match the CPU renderer."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

if __package__:
    from cutwright.modules.fx.runners import fx_cells as C
    from cutwright.modules.fx.runners.fx_base import blur, hblur, sample
    from cutwright.modules.fx.runners.fx_glyphs import atlas
else:
    import fx_cells as C
    from fx_base import blur, hblur, sample
    from fx_glyphs import atlas


def rgb(x: Any, value: str) -> Any:
    return torch.as_tensor(C.rgb(value), device=x.device)


def lum(x: Any) -> Any:
    return x[0] * 0.2126 + x[1] * 0.7152 + x[2] * 0.0722


def smooth(a: float, b: float, x: Any) -> Any:
    v = ((x - a).double() / float(np.float32(max(b - a, 1e-6)))).float().clamp(0, 1)
    return v * v * (3 - 2 * v)


def linear(x: Any) -> Any:
    return torch.where(x <= 0.04045, x / 12.92, ((x.clamp_min(0) + 0.055) / 1.055) ** 2.4)


def display(x: Any) -> Any:
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp_min(0) ** (1 / 2.4) - 0.055)


def gradient(x: Any) -> tuple[Any, Any]:
    return tuple(
        torch.gradient(x, dim=axis)[0] if size > 1 else torch.zeros_like(x)
        for axis, size in enumerate(x.shape)
    )


def resize(x: Any, h: int, w: int) -> Any:
    if x.shape[1:] == (h, w):
        return x
    return F.interpolate(x[None], size=(h, w), mode="bilinear", align_corners=False)[0]


def cell_sample(x: Any, gy: Any, gx: Any) -> Any:
    """Pixel-coordinate taps avoid normalized-grid rounding before temporal thresholds."""
    _, h, w = x.shape
    y0, x0 = torch.floor(gy).long(), torch.floor(gx).long()
    fy, fx = gy - y0.float(), gx - x0.float()
    out = torch.zeros((x.shape[0], *gy.shape), device=x.device)
    for dy, dx, weight in (
        (0, 0, (1 - fy) * (1 - fx)),
        (0, 1, (1 - fy) * fx),
        (1, 0, fy * (1 - fx)),
        (1, 1, fy * fx),
    ):
        out = out + x[:, (y0 + dy).clamp(0, h - 1), (x0 + dx).clamp(0, w - 1)] * weight[None]
    return out


def gaussian(o: Any, x: Any, sigma: float, *, horizontal: bool = False) -> Any:
    if x.is_cuda:
        # Animated radii/shapes must not retain thousands of cuFFT plans in the worker.
        cache = torch.backends.cuda.cufft_plan_cache[x.device.index]
        if cache.max_size > 64:
            cache.max_size = 64
    if sigma < 1:
        return (hblur if horizontal else blur)(o, x, max(0.1, sigma))
    # Reflect indices also cover kernels wider than the image and singleton axes.
    radius = math.ceil(4 * sigma)
    positions = torch.arange(-radius, radius + 1, device=x.device, dtype=torch.float64)
    weights = torch.exp(-0.5 * (positions / sigma) ** 2)
    weights = (weights / weights.sum()).float()
    out = x
    for axis in (2,) if horizontal else (2, 1):
        size = out.shape[axis]
        indices = torch.arange(-radius, size + radius, device=x.device)
        if size == 1:
            indices = indices * 0
        else:
            indices = indices.remainder(2 * (size - 1))
            indices = torch.minimum(indices, 2 * (size - 1) - indices)
        padded = out.index_select(axis, indices)
        if sigma >= 3:
            # Large spatial convolutions can request gigabytes of cuDNN workspace.
            # A fixed fast length reuses plans while a blur radius is animated.
            length = 1 << (padded.shape[axis] - 1).bit_length()
            transfer = torch.zeros(length, device=x.device)
            transfer[: len(weights)] = weights
            transfer = torch.roll(transfer, -radius)
            shape = [1, 1, 1]
            shape[axis] = length // 2 + 1
            spectrum = torch.fft.rfft(padded, n=length, dim=axis)
            spectrum = spectrum * torch.fft.rfft(transfer).reshape(shape)
            filtered = torch.fft.irfft(spectrum, n=length, dim=axis)
            out = filtered.narrow(axis, radius, size)
            continue
        kernel = weights.reshape(1, 1, 1, -1) if axis == 2 else weights.reshape(1, 1, -1, 1)
        # Batch channels instead of grouped convolution: identical scalar filtering.
        with torch.backends.cudnn.flags(allow_tf32=False):
            out = F.conv2d(padded[:, None], kernel)[:, 0]
    return out


class Grid:
    def __init__(
        self, o: Any, x: Any, p: dict[str, Any], f: Any, aspect: float = 1, fractional: bool = False
    ) -> None:
        _, h, w = x.shape
        cell = float(p.get("cell_size", 20)) * w / float(p.get("reference_width", 1200))
        ch = max(2, cell if fractional else round(cell))
        cw = max(2, ch * aspect if fractional else round(ch * aspect))
        columns = int(p.get("columns", 0))
        if columns:
            cw = max(2, w / columns if fractional else round(w / columns))
            ch = max(2, cw / aspect if fractional else round(cw / aspect))
        if fractional and p.get("_rows"):
            ch = h / max(1, int(p["_rows"]))
        yy, xx = o.grid(h, w)
        angle, dx, dy = (
            f.camera
            if f.camera is not None and p.get("pattern", "content") == "content"
            else (0, 0, 0)
        )
        c, s = math.cos(angle), math.sin(angle)
        cx, cy = (w - 1) / 2, (h - 1) / 2
        wx = c * (xx - cx) - s * (yy - cy) + cx + dx
        wy = s * (xx - cx) + c * (yy - cy) + cy + dy
        corners_x = np.array([0, w - 1, 0, w - 1], np.float32)
        corners_y = np.array([0, 0, h - 1, h - 1], np.float32)
        corners_wx = c * (corners_x - cx) - s * (corners_y - cy) + cx + dx
        corners_wy = s * (corners_x - cx) + c * (corners_y - cy) + cy + dy
        x0, y0 = math.floor(float(corners_wx.min()) / cw), math.floor(float(corners_wy.min()) / ch)
        # CUDA scalar division uses a reciprocal; cell boundaries need NumPy rounding.
        self.ix = torch.floor((wx.double() / float(np.float32(cw))).float()).long() - x0
        self.iy = torch.floor((wy.double() / float(np.float32(ch))).float()).long() - y0
        rows = int(np.floor(corners_wy.max() / ch)) - y0 + 1
        cols = int(np.floor(corners_wx.max() / cw)) - x0 + 1
        gy, gx = np.mgrid[:rows, :cols].astype(np.float32)
        self.cy, self.cx = gy + y0, gx + x0
        ax, ay = (gx + x0 + 0.5) * cw - cx - dx, (gy + y0 + 0.5) * ch - cy - dy
        sx, sy = c * ax + s * ay + cx, -s * ax + c * ay + cy
        # Small sampling coordinates retain NumPy rounding and seeded animation choices.
        colors = sum(
            cell_sample(
                x,
                o.asarray(sy - s * ox * cw + c * oy * ch),
                o.asarray(sx + c * ox * cw + s * oy * ch),
            )
            for oy, ox in ((0, 0), (-0.25, 0), (0.25, 0), (0, -0.25), (0, 0.25))
        )
        self.colors = (colors.double() / 5).float()
        tone = (
            (lum(self.colors) - 0.5) * float(p.get("contrast", 1))
            + 0.5
            + float(p.get("brightness", 0))
        ).clamp(0, 1) ** (1 / float(p.get("gamma", 1)))
        drive = float(p.get("field_amount", 0.35))
        drive += float(p.get("audio_amount", 0.5)) * float(p.get("_bass", 0))
        field = o.asarray(C.field(self.cy * ch / h, self.cx * cw / w, p, C.clock(p, f)))
        mode = p.get("field_blend", "offset")
        if mode == "replace" and p.get("motion", "still") != "still":
            tone = (1 - drive) * tone + drive * (field + 1) / 2
        elif mode == "multiply":
            tone = tone * (1 + drive * field)
        else:
            tone = tone + drive * field
        self.tone = tone.clamp(0, 1)
        self.py, self.px = wy.remainder(ch) / ch, wx.remainder(cw) / cw
        self.ch, self.cw = round(ch), round(cw)

    def expand(self, a: Any) -> Any:
        return a[..., self.iy, self.ix]

    def noise(self, seed: int) -> Any:
        return torch.as_tensor(C.noise(self.cy, self.cx, seed), device=self.colors.device)


def ink(g: Grid, p: dict[str, Any]) -> Any:
    if p.get("color") == "source":
        y = lum(g.colors)[None]
        color = (y + (g.colors - y) * float(p.get("saturation", 1))).clamp(0, 1)
        return (color + float(p.get("lift", 0))).clamp(0, 1) * float(p.get("gain", 1))
    if p.get("color") == "ember":
        v = g.tone
        return torch.stack([0.6 + v * 0.4, 0.15 + v * 0.75, 0.12 + v * 0.45]) * p.get("gain", 1)
    return rgb(g.colors, p.get("ink", "#f2f2f2")).expand_as(g.colors) * float(p.get("gain", 1))


def glyph(g: Grid, indices: Any, chars: str, **settings: Any) -> Any:
    table = torch.as_tensor(atlas(chars, g.ch, g.cw, **settings), device=g.colors.device)
    ids = g.expand(indices).long().clamp(0, len(chars) - 1)
    yy = (g.py * table.shape[1]).long().clamp(0, table.shape[1] - 1)
    xx = (g.px * table.shape[2]).long().clamp(0, table.shape[2] - 1)
    return table[ids, yy, xx]


def phrase_dither(o: Any, x: Any, p: dict[str, Any], f: Any) -> Any:
    g = Grid(o, x, p, f)
    phrase = p.get("phrase", "NOWBOARDINGPLATFORMNINE") or " "
    columns = math.ceil(x.shape[2] / g.cw)
    index = (g.cy * columns + g.cx).astype(np.int32) % len(phrase)
    opacity = np.ones_like(g.cy)
    if p.get("animate", True):
        tick = math.floor(C.clock(p, f) * float(p.get("speed", 1.2)) * 9)
        n = C.noise(g.cy, g.cx, 91)
        swap = C.noise(n * 53, tick, 7)
        amount = float(p.get("animation_amount", 0.35))
        index = (index + (swap < 0.3 * amount) * (1 + np.floor(swap * 887))) % len(phrase)
        opacity = np.where(C.noise(n * 191, tick, 29) < 0.25 * amount, 0.2, 1).astype(np.float32)
    letters = glyph(g, o.asarray(index), phrase) * g.expand(o.asarray(opacity))
    _, h, w = x.shape
    yy, xx = o.grid(h, w)
    unit = max(1, float(p.get("dither_size", 3)) * w / float(p.get("reference_width", 1200)))
    bayer = o.asarray([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])
    threshold = (bayer[(yy / unit).long() % 4, (xx / unit).long() % 4] + 0.5) / 16
    tone = (
        (lum(x) ** (1 / float(p.get("gamma", 1))) - 0.5) * float(p.get("contrast", 1))
        + 0.5
        + float(p.get("brightness", 0))
    ).clamp(0, 1)
    cover = torch.where(
        g.expand(g.tone >= float(p.get("threshold", 0.57))), letters, (tone > threshold).float()
    )
    bg = rgb(x, p.get("paper", "#0c0c0c"))
    out = bg + (g.expand(ink(g, p)) - bg) * cover[None]
    if p.get("glow", 0):
        radius = float(p.get("glow_radius", 10)) * w / float(p.get("reference_width", 1200))
        out = out + gaussian(o, (out - bg).clamp_min(0), max(0.3, radius)) * p["glow"]
    return out


def ascii_cells(o: Any, x: Any, p: dict[str, Any], f: Any) -> Any:
    _, h, w = x.shape
    columns = int(p.get("columns", 0)) or max(
        1, round(w / max(2, p.get("cell_size", 20) * w / 1200 * 0.52))
    )
    rows = max(2, round(h / w * columns * 0.52))
    settings = {**p, "columns": columns, "_rows": rows, "gamma": 1, "contrast": 1, "brightness": 0}
    g = Grid(o, linear(x.clamp(0, 1)), settings, f, 0.52, fractional=True)
    raw = lum(g.colors)
    tone = (
        (raw ** (1 / float(p.get("gamma", 1))) - 0.5) * float(p.get("contrast", 1))
        + 0.5
        + float(p.get("brightness", 0))
    ).clamp(0, 1) ** float(p.get("density_bias", 1))
    tone = (tone + g.tone - raw).clamp(0, 1)
    n = g.noise(int(p.get("seed", 0)))
    tone = torch.maximum(
        tone, torch.where(n > 0.52, float(p.get("floor", 0)) * (0.026 + 0.075 * n * n), 0)
    )
    chars = p.get("charset", " .,:;irsXA253hMHGS9B&@")
    table = atlas(chars + "-\\|/", g.ch, g.cw, regular=not p.get("bold", True), stretch=True)
    ascending = table[0].mean() <= table[len(chars) - 1].mean()
    index = torch.round((tone if ascending else 1 - tone) * (len(chars) - 1)).long()
    gy, gx = gradient(raw)
    direction = torch.round(torch.atan2(gx, -gy).remainder(math.pi) / (math.pi / 4)).long() % 4
    index = torch.where(
        torch.hypot(gx, gy) * 8 * float(p.get("structure", 0)) > 0.1, len(chars) + direction, index
    )
    cover = glyph(g, index, chars + "-\\|/", regular=not p.get("bold", True), stretch=True)
    gap = float(p.get("gap", 0)) / 2
    cover = cover * ((g.px >= gap) & (g.px < 1 - gap) & (g.py >= gap) & (g.py < 1 - gap))
    color = ink(g, p) * (1 + (float(p.get("emission", 1)) - 1) * tone[None])
    bg = linear(rgb(x, p.get("paper", "#050209")))
    out = display((bg + (g.expand(color) - bg) * cover[None]).clamp_min(0))
    out = out + x * float(p.get("photo", 0))
    amber = float(p.get("amber_glow", 0))
    if amber:
        warm = smooth(0.01, 0.12, torch.minimum(g.colors[0], g.colors[1]) - g.colors[2])
        light = (out - display(bg)).clamp_min(0) * g.expand(warm)[None]
        light = lum(light)[None] * rgb(x, "#ff9b3d")
        radius = max(
            0.3, float(p.get("glow_radius", 18)) * w / float(p.get("reference_width", 1200))
        )
        halo = gaussian(o, light, radius * 0.25) * 0.7 + gaussian(o, light, radius) * 0.6
        out = out + halo.clamp_min(0) * amber
    if p.get("glow", 0):
        out = (
            out
            + gaussian(
                o, (out - display(bg)).clamp_min(0), max(0.3, p.get("glow_radius", 10) * w / 1200)
            )
            * p["glow"]
        )
    return out


def glyph_haze(o: Any, x: Any, p: dict[str, Any], f: Any) -> Any:
    g = Grid(o, x, p, f, 0.65, fractional=True)
    chars = p.get("charset", " .,:;irsXA253hMHGS9B&@")
    regular = not p.get("bold", True)
    table = atlas(chars, g.ch, g.cw, regular=regular)
    raw = lum(g.colors)
    gamma, contrast = float(p.get("gamma", 1)), float(p.get("contrast", 1))
    before = ((raw - 0.5) * contrast + 0.5 + p.get("brightness", 0)).clamp(0, 1) ** (1 / gamma)
    tone = (
        (raw ** (1 / gamma) - 0.5) * contrast + 0.5 + p.get("brightness", 0) + g.tone - before
    ).clamp(0, 1)
    if table[0].mean() > table[-1].mean():
        tone = 1 - tone
    coverage = glyph(g, torch.floor(tone * (len(chars) - 0.001)), chars, regular=regular)
    gap = float(p.get("gap", 0)) / 2
    coverage = coverage * ((g.px >= gap) & (g.px < 1 - gap) & (g.py >= gap) & (g.py < 1 - gap))
    bg = rgb(x, p.get("paper", "#0c0c0c"))
    out = bg + (g.expand(ink(g, p)) * 1.15 - bg) * coverage[None] + x * p.get("photo", 0)
    if p.get("glow", 0):
        out = (
            out
            + gaussian(
                o, (out - bg).clamp_min(0), max(0.3, p.get("glow_radius", 10) * x.shape[2] / 1200)
            )
            * p["glow"]
        )
    return out


def thermal_reveal(o: Any, x: Any, p: dict[str, Any], f: Any) -> Any:
    scale = x.shape[2] / float(p.get("reference_width", 1200))
    t = C.clock(p, f)
    y = lum(gaussian(o, x, max(0.25, float(p.get("soften", 16)) * scale)))
    detail = float(p.get("detail", 0.22))
    y = y * (1 - detail) + lum(x) * detail
    y = ((y - 0.5) * float(p.get("contrast", 1)) + 0.5 + p.get("brightness", 0)).clamp(0, 1) ** (
        1 / float(p.get("gamma", 1))
    )
    phase = (
        (1 - y) * p.get("cycles", 1.2)
        + p.get("phase", 0.67)
        + t * p.get("drift", 0.06)
        + math.sin(t / p.get("loop_seconds", 5) * 2 * math.pi) * p.get("breathe", 0.12)
    ).remainder(1)
    palette = o.asarray(np.array([C.rgb(v)[:, 0, 0] for v in p["colors"]]))
    pos = phase * len(palette)
    index = torch.floor(pos).long() % len(palette)
    mix = pos - torch.floor(pos)
    mix = mix * mix * (3 - 2 * mix)
    color = (
        palette[index] * (1 - mix[..., None]) + palette[(index + 1) % len(palette)] * mix[..., None]
    ).permute(2, 0, 1)
    progress = float(p.get("progress", 0))
    if p.get("automatic", True):
        time = t / float(p.get("duration", 3.433333))
        if time < 0.2913:
            progress = 1.13 * max(0, time / 0.2913)
        elif time <= 0.4951:
            progress = 1.13
        else:
            progress = 1.13 + 0.87 * float(np.clip((time - 0.4951) / (0.7476 - 0.4951), 0, 1))
    soft = float(p.get("soft", 0.18))
    stage = progress / 1.13 if progress <= 1.13 else (progress - 1.13) / 0.87
    amount = smooth(-soft, soft, stage * (1 + 2 * soft) - soft - y)
    if progress > 1.13:
        amount = 1 - amount
    if progress <= 0 or progress >= 2:
        amount = amount * 0
    return x + (color - x) * amount[None]


def halftone_cells(o: Any, x: Any, p: dict[str, Any], f: Any) -> Any:
    _, h, w = x.shape
    cell = max(2, float(p.get("cell_size", 22)) * w / float(p.get("reference_width", 1200)))
    yy, xx = o.grid(h, w)
    angle = math.radians(float(p.get("angle", 64)))
    c, s = math.cos(angle), math.sin(angle)
    u, v = xx * c + yy * s, -xx * s + yy * c
    cx, cy = (torch.floor(u / cell) + 0.5) * cell, (torch.floor(v / cell) + 0.5) * cell
    y = ((lum(x) - 0.5) * p.get("contrast", 1) + 0.5 + p.get("brightness", 0)).clamp(0, 1) ** (
        1 / p.get("gamma", 1)
    )
    tone = sample(o, y[None], cx * s + cy * c, cx * c - cy * s)[0] ** 2.2
    radius = tone**1.15 * cell * 0.45 * p.get("dot_scale", 1.17)
    distance = torch.hypot(u.remainder(cell) - cell / 2, v.remainder(cell) - cell / 2)
    cover = (radius - distance + 0.5).clamp(0, 1)
    light = rgb(x, p.get("ink", "#ffb54d")) * (cover * (0.35 + tone**0.5) * 0.8)[None]
    light = light + rgb(x, "#fff2a0") * (cover * tone * p.get("dot_light", 0.37) * 0.12)[None]
    halo = gaussian(o, light, max(0.5, cell * 0.65)) * p.get("edge_glow", 0.53) * 3
    return rgb(x, p.get("paper", "#0b0705")) * 0.3 + light + halo * rgb(x, "#ff6430")


def tone_stripes(o: Any, x: Any, p: dict[str, Any], f: Any) -> Any:
    g = Grid(o, x, p, f, 1.4)
    color = torch.where(
        (g.tone > 0.7)[None],
        rgb(x, p.get("light", "#bcbcbc")),
        torch.where(
            (g.tone > 0.4)[None],
            rgb(x, p.get("medium", "#787878")),
            rgb(x, p.get("dark", "#343434")),
        ),
    )
    coverage = (g.py - 0.5).abs() < p.get("stripe_height", 0.8) / 2
    deep = g.expand(g.tone < p.get("shadow", 0))
    mark = ((g.py - 0.5).abs() < 0.1) & ((g.px - 0.5).abs() < 0.1)
    coverage = torch.where(deep, mark, coverage)
    bg = rgb(x, p.get("paper", "#161616"))
    out = bg + (g.expand(color) - bg) * coverage[None]
    if p.get("axis"):
        out[:, :, x.shape[2] // 2] = rgb(x, p.get("ink", "#ffffff"))[:, 0]
    return out


def edge_sparks(o: Any, x: Any, p: dict[str, Any], f: Any) -> Any:
    scale = x.shape[2] / float(p.get("reference_width", 1200))
    g = Grid(
        o,
        gaussian(o, x, max(0.4, scale * 2)),
        {**p, "cell_size": max(2, 12 / p.get("density", 4))},
        f,
    )
    gy, gx = gradient(g.tone)
    threshold = float(p.get("threshold", 0.09))
    edge = smooth(threshold * 0.35, threshold * 0.35 + 0.16, torch.hypot(gx, gy))
    n = g.noise(int(p.get("seed", 10)))
    hit = n < edge * 0.36 + g.tone * p.get("dust", 0.25) * 0.003
    phase = 2 * math.pi * C.clock(p, f) / p.get("loop_seconds", 5)
    energy = hit * (0.7 + 0.3 * g.tone) * (1 + p.get("twinkle", 0.34) * torch.sin(n * 100 + phase))
    radius = p.get("particle_size", 3.1) * scale / g.ch * 0.5
    cover = ((radius - torch.hypot(g.px - 0.5, g.py - 0.5)) * g.ch + 0.5).clamp(0, 1)
    particles = (
        rgb(x, p.get("ink", "#ffe2ba")) * (g.expand(energy) * cover)[None] * p.get("gain", 1.6)
    )
    particles = particles + gaussian(o, particles, max(0.5, scale * 2)) * 1.3
    return rgb(x, p.get("paper", "#030309")) + x * p.get("underlay", 0.02) + particles


def tone_atlas(o: Any, x: Any, p: dict[str, Any], f: Any) -> Any:
    kind = p.get("style", "dots")
    if kind == "terrain":
        return contour_terrain(o, x, p, f)
    if kind == "lines":
        return contour_lines(o, x, p, f)
    size = float(p.get("glyph_size", 21)) * (1 if kind == "blocks" else 0.8)
    settings = {**p, "cell_size": size, "gamma": 1, "contrast": 1, "brightness": 0}
    g = Grid(o, x, settings, f)
    tone = lum(g.colors)
    dy, dx = gradient(tone)
    luma = lum(x)
    micro = (luma - gaussian(o, luma[None], max(0.5, x.shape[2] / 600))[0]).abs()
    detail = Grid(o, micro[None].expand(3, -1, -1), settings, f).colors[0]
    n = g.noise(int(p.get("seed", 0)))
    energy = ((torch.hypot(dx, dy) + detail * 2.5) * 3.5 - 0.055).clamp(0, 0.6) * (0.5 + 0.5 * n)
    color, bg = rgb(x, p.get("ink", "#c9c7c0")), rgb(x, p.get("paper", "#1c1c1e"))
    if kind == "blocks":
        coverage = (g.px > 0.06) & (g.px < 0.94) & (g.py > 0.06) & (g.py < 0.94)
        key = (tuple(x.shape), str(x.device))
        if f.state.get("texture_key") != key:
            yy, xx = np.mgrid[: x.shape[1], : x.shape[2]]
            f.state.update(texture_key=key, texture=o.asarray(0.1 + C.noise(yy, xx, 47) * 0.9))
        out = (
            bg * (0.6 + 0.1 * coverage)[None]
            + color * g.expand(energy)[None] * (coverage * f.state["texture"])[None]
        )
    else:
        coverage = glyph(g, (n * 4).long(), "+xo.", scale=1.25)
        level = (energy * 3).clamp(0, 0.95) * p.get("line_opacity", 0.9)
        out = bg * 0.03 + color * g.expand(level)[None] * coverage[None]
    label = p.get("label_text", "PLATFORM")
    if label:
        labels = (n < p.get("labels", 0.25) * 0.025) & (energy > 0.1)
        index = (g.noise(71) * len(label)).long()
        a = glyph(g, index, label) * g.expand(labels)
        out = out + rgb(x, "#ffe5c2") * a[None] * 0.8
    probability = 0.006 if kind == "blocks" else 0.02
    selected = (n > 1 - p.get("heroes", 0.55) * probability) & (energy > 0.1)
    hot = g.expand(selected) * (
        coverage if kind == "blocks" else ((g.px - 0.5).abs() < 0.4) & ((g.py - 0.5).abs() < 0.4)
    )
    core = rgb(x, p.get("accent", "#ef282d")) ** 2 * hot[None] * 2.8
    out = out + core + gaussian(o, core, max(1, g.ch * (0.6 if kind == "blocks" else 0.4))) * 2.5
    if kind == "dots":
        warm = g.expand((energy > 0.18) & (n < 0.32))
        out = out + rgb(x, "#ffe4aa") * (coverage * warm)[None] * 1.5 * p.get("line_opacity", 0.9)
        square = ((g.px - 0.5).abs() < 0.36) & ((g.py - 0.5).abs() < 0.36)
        lit = g.expand((energy > 0.18) & (n < 0.1)) * square
        blobs = rgb(x, "#ffe4aa") * lit[None] * p.get("glow", 0)
        out = out + blobs + gaussian(o, blobs, max(1, g.ch * 0.9)) * 3
    return out


def contour_lines(o: Any, x: Any, p: dict[str, Any], f: Any) -> Any:
    _, h, w = x.shape
    scale = w / float(p.get("reference_width", 1200))
    y = lum(gaussian(o, x, max(0.25, p.get("smoothing", 1.65) * scale)))
    y = ((y - 0.5) * p.get("contrast", 1) + 0.5).clamp(0, 1)
    y = y ** (1 / p.get("gamma", 1)) * int(p.get("bands", 16))
    gy, gx = gradient(y)
    strength = torch.hypot(gx, gy)
    distance = ((y + 0.5).remainder(1) - 0.5).abs() / strength.clamp_min(0.001)
    line = (1 - smooth(0.15, 1.05, distance)) * smooth(0.0001, 0.003, strength)
    split = max(1, round(scale * 4))
    channels = torch.stack([torch.roll(line, split, 1), line, torch.roll(line, -split, 1)]) * rgb(
        x, p.get("ink", "#d8ff3e")
    )
    g = Grid(o, x, {**p, "cell_size": 16}, f)
    chosen = (g.noise(25) > 0.997) & (g.tone > 0.55)
    sparks = g.expand(chosen) * torch.exp(-((g.px - 0.5) ** 2 + (g.py - 0.5) ** 2) * 20)
    out = rgb(x, p.get("paper", "#161616")) * 0.5 + channels * 0.85 + sparks[None]
    amount = float(p.get("horizontal_glitch", 0))
    if amount <= 0:
        return out
    length = max(2, p.get("glitch_length", 120) * scale)
    row = np.floor(np.arange(h, dtype=np.float32)[:, None] / max(1, round(2 * scale)))
    cycle = (C.clock(p, f) / p.get("loop_seconds", 5)) % 1
    ticks = max(1, round(p.get("glitch_speed", 6) * p.get("loop_seconds", 5)))
    tick = math.floor(cycle * ticks)
    seed = int(p.get("seed", 0)) + 43
    gate = o.asarray(C.noise(row, tick, seed) > 1 - amount * 0.1).bool()
    offset = (C.noise(row, tick, seed + 11) * 2 - 1) * length * 1.8
    offset += np.sin(cycle * 2 * math.pi * ticks + row) * scale * 2
    yy, xx = o.grid(h, w)
    echo = sample(o, channels, yy, xx - o.asarray(offset), "zero") * gate[None]
    radius = max(1, length * 0.2)
    echo = gaussian(o, echo, radius, horizontal=True) * max(1, radius * 0.8)
    bins = np.floor(np.arange(w, dtype=np.float32) / max(4, length * 0.6)).astype(int)
    runs = (
        C.noise(row, np.arange(int(bins.max()) + 1, dtype=np.float32)[None] + tick, seed + 29) > 0.7
    )
    runs = o.asarray(runs).bool()[:, torch.as_tensor(bins, device=x.device)]
    return out + echo.clamp_min(0) * (runs & gate)[None] * amount


def contour_terrain(o: Any, x: Any, p: dict[str, Any], f: Any) -> Any:
    if __package__:
        from cutwright.modules.fx.runners.fx_cells_reference import box_blur, terrain_markers
    else:
        from fx_cells_reference import box_blur, terrain_markers
    _, h, w = x.shape
    scale = w / float(p.get("reference_width", 1200))
    step = max(1, float(p.get("cell_size", 6)) * scale)
    gh, gw = max(8, round(h / step)), max(8, round(w / step))
    source = o.numpy(resize(linear(x.clamp(0, 1)), gh, gw))
    terrain = box_blur(C.lum(source), max(1, round(float(p.get("smoothing", 4)))))
    dy, dx = C.gradient(terrain)
    yy, xx = o.grid(h, w)
    sy, sx = yy * gh / h, xx * gw / w
    tone = sample(o, o.asarray(terrain[None]), sy, sx)[0]
    gx = sample(o, o.asarray(dx[None]), sy, sx)[0] * gw / w
    gy = sample(o, o.asarray(dy[None]), sy, sx)[0] * gh / h
    slope = torch.hypot(gx, gy).clamp_min(1e-6)
    bands, multi = int(p.get("bands", 4)), int(p.get("lines_per_band", 3))
    levels = [
        (b + 0.5) / bands + (j - (multi - 1) / 2) * p.get("band_spread", 1.4) * 0.01
        for b in range(bands)
        for j in range(multi)
    ]
    levels = [v for v in levels if 0.005 < v < 0.995]
    width = max(0.25, 0.75 * scale)
    targets = o.asarray(sorted(levels))
    alpha = torch.zeros_like(tone)
    if levels:
        reach = slope * (width + 0.75)
        low = torch.searchsorted(targets, (tone - reach).contiguous())
        high = torch.searchsorted(targets, (tone + reach).contiguous(), right=True)
        for offset in range(int((high - low).max())):
            ids = low + offset
            distance = (tone - targets[ids.clamp_max(len(levels) - 1)]).abs() / slope
            cover = (1 - smooth(max(0, width - 0.75), width + 0.75, distance)) * (ids < high)
            alpha = 1 - (1 - alpha) * (1 - cover * p.get("line_opacity", 0.53))
    bg, color = rgb(x, p.get("paper", "#1c1c1e")), rgb(x, p.get("ink", "#c9c7c0"))
    out = bg + (color - bg) * alpha[None]
    painted, glow = terrain_markers(h, w, p, terrain, dy, dx, levels)
    layer = o.asarray(painted.transpose(2, 0, 1))
    out = out * (1 - layer[3:4]) + layer[:3] * layer[3:4]
    if np.any(glow):
        radius = max(1, float(p.get("glow", 0.45)) * 14 * scale)
        out = out + gaussian(o, o.asarray(glow), radius) * radius * 2
    return out


def cell_finish(o: Any, x: Any, p: dict[str, Any], f: Any) -> Any:
    _, h, w = x.shape
    ratio = min(1, 320 / max(h, w))
    small = resize(x, max(2, round(h * ratio)), max(2, round(w * ratio)))
    threshold, knee = float(p.get("threshold", 0.35)), float(p.get("knee", 0.5))
    brightness = small.amax(0)
    light = small * smooth(threshold - knee * 0.5, threshold + knee * 0.5, brightness)[None]
    halos = torch.zeros_like(small)
    palette = C.BLOOM_PALETTES.get(p.get("bloom_palette", "mono"))
    for k, (sigma, weight) in enumerate(((0.8, 0.45), (2.5, 0.3), (7.5, 0.17), (20, 0.08))):
        halo = gaussian(o, light, sigma * p.get("radius", 1.8)) * weight
        if palette is not None:
            halo = halo * torch.as_tensor(palette[k], device=x.device)
        halos = halos + halo
    tint = rgb(x, p.get("bloom_tint", "#ffffff")) if palette is None else 1
    finish = halos * tint * p.get("bloom", 0.6)
    strength = float(p.get("streaks", 0))
    if strength:
        threshold = float(p.get("streak_threshold", 0.55))
        light = small * smooth(max(0, threshold - 0.1), threshold + 0.1, brightness)[None]
        angle = math.radians(float(p.get("angle", 18)))
        length = float(p.get("length", 0.42)) * small.shape[2]
        streak = C.zero_taps(light, C.streak_taps(length, angle))
        finish = finish + streak * rgb(x, p.get("tint", "#ff6beb")) * strength * 0.25
    out = x + resize(finish, h, w)
    if p.get("trails", 0):
        # Finite history is at most 320 px wide and remains bounded in host memory.
        if __package__:
            from cutwright.modules.fx.runners.fx_cells_reference import finite_max
        else:
            from fx_cells_reference import finite_max
        memory = finite_max(o.numpy(small), f, float(p["trails"]) * 0.995)
        out = torch.maximum(out, resize(o.asarray(memory), h, w))
    out = out * 2 ** float(p.get("exposure", 0))
    y = lum(out)[None]
    return y + (out - y) * float(p.get("saturation", 1))


def motion_ink(o: Any, x: Any, p: dict[str, Any], f: Any) -> Any:
    if __package__:
        from cutwright.modules.fx.runners.fx_cells_reference import motion_ink as analyze
    else:
        from fx_cells_reference import motion_ink as analyze
    # Keep exact finite ink expiry and lossless host compression; draw its material on CUDA.
    return analyze(o.numpy(x), p, f, gpu=(o, x))


def trace_material(o: Any, x: Any, memory: Any, p: dict[str, Any], f: Any) -> Any:
    _, h, w = x.shape
    out = resize(o.asarray(memory), h, w)
    detailed = p.get("fine_detail", False)
    material = p.get("material", "dither")
    if material == "dither":
        size = max(1, p.get("cell_size", 9) * w / p.get("reference_width", 1200) * 0.18)
        rows = (np.arange(h) / size).astype(int) % 4
        cols = (np.arange(w) / size).astype(int) % 4
        table = o.asarray([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])
        a = (
            table[
                torch.as_tensor(rows, device=x.device)[:, None],
                torch.as_tensor(cols, device=x.device)[None],
            ]
            + (0.5 if detailed else 0)
        ) / 16
        level = (lum(out) * 1.38).clamp(0, 1) if detailed else out.amax(0)
        gain = 0.56 + 0.44 * level[None] if detailed else (level[None] + 0.35).clamp_max(1)
        out = (
            out
            / (out.amax(0) if detailed else level)[None].clamp_min(1e-5)
            * (level > a)[None]
            * gain
        )
    elif material == "ascii":
        g = Grid(o, out, {**p, "cell_size": max(12, p.get("cell_size", 9))}, f, 0.6)
        chars = " .:-=+*#%@"
        out = out * glyph(g, torch.floor(g.tone * (len(chars) - 0.001)), chars)[None]
    elif material == "halftone":
        g = Grid(o, out, p, f)
        radius = g.expand(g.tone.sqrt() * 0.48)
        coverage = smooth(-0.5, 0.5, (radius - torch.hypot(g.py - 0.5, g.px - 0.5)) * g.ch)
        out = out * coverage[None]
    return out + x * float(p.get("source", 0))


def covered(g: Grid, p: dict[str, Any]) -> Any:
    gap = float(p.get("gap", 0)) / 2
    return (g.px >= gap) & (g.px <= 1 - gap) & (g.py >= gap) & (g.py <= 1 - gap)


def glyph_blocks(o: Any, x: Any, p: dict[str, Any], f: Any) -> Any:
    g = Grid(o, x, p, f)
    n = g.noise(f.seed)
    threshold = float(p.get("threshold", 0.6))
    lit = smooth(threshold - 0.12, threshold + 0.08, g.tone)
    block = (g.py < p.get("block_height", 0.85)) & covered(g, p)
    chars = p.get("charset", "0123456789")
    digit = glyph(g, torch.floor(n * len(chars)), chars)
    a = g.expand(lit) * torch.where(g.expand(n < p.get("digits", 0.25)), digit, block.float())
    bg = rgb(x, p.get("paper", "#0c0c0c"))
    return bg + (g.expand(ink(g, p)) - bg) * a[None]


def code_blocks(o: Any, x: Any, p: dict[str, Any], f: Any) -> Any:
    g = Grid(o, x, p, f)
    chars = p.get("charset", "0X8SG>")
    t = C.clock(p, f)
    n = C.noise(g.cy, g.cx, f.seed)
    tick = np.floor(t * p.get("flicker", 4.9) + n * 11)
    index = ((n * 811 + tick) % len(chars)).astype(np.int32)
    a = glyph(g, o.asarray(index), chars)
    alpha = (
        p.get("opacity_min", 0) + (p.get("opacity_max", 0.68) - p.get("opacity_min", 0)) * g.tone
    )
    phase = 2 * math.pi * t / p.get("loop_seconds", 5)
    alpha = alpha * o.asarray(0.75 + 0.25 * np.sin(n * 7 + phase * round(p.get("flow", 2.2))))
    bg = rgb(x, p.get("paper", "#0c0c0c"))
    out = g.expand(bg + (g.colors * p.get("gain", 1) - bg) * alpha[None])
    pick = o.asarray(C.noise(g.cy + 17, g.cx, f.seed))
    color = torch.where((pick < 0.18)[None], rgb(x, p.get("accent_a", "#7adfff")), ink(g, p))
    color = torch.where(
        ((pick >= 0.18) & (pick < 0.36))[None], rgb(x, p.get("accent_b", "#ff7a4a")), color
    )
    a = a * g.expand(o.asarray(0.15 + 0.55 * n))
    out = out + (g.expand(color) - out) * a[None]
    coverage = covered(g, p)
    return out * coverage + bg * ~coverage


def riso_cells(o: Any, x: Any, p: dict[str, Any], f: Any) -> Any:
    _, h, w = x.shape
    scale = w / float(p.get("reference_width", 1200))
    cell = max(2, p.get("cell_size", 12) * scale)
    yy, xx = o.grid(h, w)
    shifted = sample(o, x, yy - p.get("drift_y", 13.5) * scale, xx - p.get("drift_x", 8) * scale)

    def dots(source: Any, degrees: float) -> Any:
        angle = math.radians(degrees)
        u, v = (
            xx * math.cos(angle) + yy * math.sin(angle),
            -xx * math.sin(angle) + yy * math.cos(angle),
        )
        cx, cy = torch.floor(u / cell) * cell + cell / 2, torch.floor(v / cell) * cell + cell / 2
        tone = sample(
            o,
            source[None],
            cx * math.sin(angle) + cy * math.cos(angle),
            cx * math.cos(angle) - cy * math.sin(angle),
        )[0]
        radius = tone.clamp(0, 1).sqrt() * cell * 0.48 * 1.7
        distance = torch.hypot(u.remainder(cell) - cell / 2, v.remainder(cell) - cell / 2)
        return (radius - distance + 0.5).clamp(0, 1)

    first, second = dots(1 - x[0], 15), dots(1 - shifted[2], 75)
    bleed = float(p.get("bleed", 0.65))
    first = (first + gaussian(o, first[None], max(0.1, scale * bleed))[0] * bleed * 0.2).clamp(0, 1)
    second = (second + gaussian(o, second[None], max(0.1, scale * bleed))[0] * bleed * 0.2).clamp(
        0, 1
    )
    out = rgb(x, p.get("paper", "#0c0c0c")) * (
        1 - first[None] * (1 - rgb(x, p.get("ink", "#ff453a")))
    )
    out = out * (1 - second[None] * (1 - rgb(x, p.get("ink2", "#1d4ed8"))))
    key = (tuple(x.shape), int(p.get("seed", 0)), str(x.device), max(1, round(scale)))
    if f.state.get("fibre_key") != key:
        ny, nx = np.mgrid[:h, :w].astype(np.float32)
        fibre = C.noise(ny // max(1, round(scale)), nx, int(p.get("seed", 0))) - 0.5
        f.state.update(fibre_key=key, fibre=o.asarray(fibre))
    return out * (1 + p.get("fibre", 0.7) * f.state["fibre"][None] * 0.15)


def cell_vhs(o: Any, x: Any, p: dict[str, Any], f: Any) -> Any:
    if __package__:
        from cutwright.modules.fx.runners.fx_base import KERNELS
    else:
        from fx_base import KERNELS
    filtered = KERNELS["vhs"](o, x, p, f)
    return x + filtered if p.get("layer_mode") == "add" else filtered


DRAWERS = {
    "motion_ink": motion_ink,
    "cell_finish": cell_finish,
    "ascii_cells": ascii_cells,
    "glyph_haze": glyph_haze,
    "phrase_dither": phrase_dither,
    "thermal_reveal": thermal_reveal,
    "halftone_cells": halftone_cells,
    "tone_stripes": tone_stripes,
    "edge_sparks": edge_sparks,
    "tone_atlas": tone_atlas,
    "glyph_blocks": glyph_blocks,
    "code_blocks": code_blocks,
    "riso_cells": riso_cells,
    "cell_vhs": cell_vhs,
}


def draw(o: Any, name: str, x: Any, p: dict[str, Any], f: Any) -> Any:
    fn = DRAWERS.get(name)
    if fn is None or p.get("render_style", "reference") != "reference":
        return None
    with torch.no_grad():
        image = torch.nan_to_num(x).float()
        result = fn(o, image, p, f)
        if result is None:
            return None
        mode, k = p.get("appears", "all"), float(p.get("softness", 0.25))
        if mode == "all":
            return (image + (result - image)).clamp(0, 1)
        y = lum(image)
        if mode == "shadows":
            amount = 1 - smooth(0.3 - k / 2, 0.3 + k / 2, y)
        elif mode == "highlights":
            amount = smooth(0.65 - k / 2, 0.65 + k / 2, y)
        elif mode == "midtones":
            amount = smooth(0.25 - k / 2, 0.25 + k / 2, y) * (
                1 - smooth(0.75 - k / 2, 0.75 + k / 2, y)
            )
        else:
            gy, gx = gradient(y)
            amount = smooth(0.01, 0.01 + k / 3, torch.hypot(gx, gy))
        return (image + (result - image) * amount[None]).clamp(0, 1)
