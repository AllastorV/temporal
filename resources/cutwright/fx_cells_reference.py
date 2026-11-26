"""Reference cell materials, independently reconstructed from measured behavior."""

from __future__ import annotations

import math
import zlib
from collections import deque
from typing import Any

import numpy as np

if __package__:
    from cutwright.modules.fx.runners import fx_cells as C
    from cutwright.modules.fx.runners.fx_base import hblur
    from cutwright.modules.fx.runners.fx_glyphs import FONT, atlas
else:
    import fx_cells as C
    from fx_base import hblur
    from fx_glyphs import FONT, atlas


def linear(x: Any) -> Any:
    return np.where(x <= 0.04045, x / 12.92, ((np.maximum(x, 0) + 0.055) / 1.055) ** 2.4)


def display(x: Any) -> Any:
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.maximum(x, 0) ** (1 / 2.4) - 0.055)


def glyph(g: C.Grid, indices: Any, chars: str, *, regular: bool = False, scale: float = 1) -> Any:
    table = atlas(chars, g.ch, g.cw, regular=regular, scale=scale)
    ids = np.clip(g.expand(indices).astype(np.int32), 0, len(chars) - 1)
    yy = np.clip((g.py * table.shape[1]).astype(np.int32), 0, table.shape[1] - 1)
    xx = np.clip((g.px * table.shape[2]).astype(np.int32), 0, table.shape[2] - 1)
    return table[ids, yy, xx]


def phrase_dither(x: Any, p: dict[str, Any], f: Any) -> Any:
    g = C.grid(x, p, f)
    phrase = p.get("phrase", "NOWBOARDINGPLATFORMNINE") or " "
    columns = math.ceil(x.shape[2] / g.cw)
    index = (g.cy * columns + g.cx).astype(np.int32) % len(phrase)
    opacity = np.ones_like(g.tone)
    if p.get("animate", True):
        tick = math.floor(C.clock(p, f) * float(p.get("speed", 1.2)) * 9)
        n = C.noise(g.cy, g.cx, 91)
        swap = C.noise(n * 53, tick, 7)
        amount = float(p.get("animation_amount", 0.35))
        index = (index + (swap < 0.3 * amount) * (1 + np.floor(swap * 887))) % len(phrase)
        blink = C.noise(n * 191, tick, 29) < 0.25 * amount
        opacity = np.where(blink, 0.2, 1).astype(np.float32)
    letters = glyph(g, index, phrase) * g.expand(opacity)
    text_cells = g.expand(g.tone >= float(p.get("threshold", 0.57)))
    h, w = x.shape[1:]
    yy, xx = np.mgrid[:h, :w]
    unit = max(1, float(p.get("dither_size", 3)) * w / float(p.get("reference_width", 1200)))
    bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32)
    threshold = (bayer[(yy / unit).astype(int) % 4, (xx / unit).astype(int) % 4] + 0.5) / 16
    tone = np.clip(
        (C.lum(x) ** (1 / float(p.get("gamma", 1))) - 0.5) * float(p.get("contrast", 1))
        + 0.5
        + float(p.get("brightness", 0)),
        0,
        1,
    )
    cover = np.where(text_cells, letters, tone > threshold)
    bg = C.rgb(p.get("paper", "#0c0c0c"))
    out = bg + (g.expand(C.ink(g, p)) - bg) * cover[None]
    glow = float(p.get("glow", 0))
    if glow:
        radius = float(p.get("glow_radius", 10)) * w / float(p.get("reference_width", 1200))
        out += C.gaussian(np.maximum(out - bg, 0), max(0.3, radius)) * glow
    return out


def box_blur(x: Any, radius: int) -> Any:
    out = x.astype(np.float32)
    for axis in (0, 1, 0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        source = np.pad(out, pad, mode="edge")
        cumulative = np.cumsum(source, axis=axis, dtype=np.float64)
        cumulative = np.concatenate(
            [np.take(cumulative, [0], axis=axis) * 0, cumulative], axis=axis
        )
        size = 2 * radius + 1
        out = (
            np.take(cumulative, range(size, cumulative.shape[axis]), axis=axis)
            - np.take(cumulative, range(cumulative.shape[axis] - size), axis=axis)
        ) / size
    return out.astype(np.float32)


def contour_terrain(x: Any, p: dict[str, Any], f: Any) -> Any:
    _, h, w = x.shape
    scale = w / float(p.get("reference_width", 1200))
    step = max(1, float(p.get("cell_size", 6)) * scale)
    gh, gw = max(8, round(h / step)), max(8, round(w / step))
    source = C.resize(linear(np.clip(x, 0, 1)), gh, gw)
    terrain = box_blur(C.lum(source), max(1, round(float(p.get("smoothing", 4)))))
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    sy, sx = yy * gh / h, xx * gw / w
    tone = C.sample(C.OPS, terrain[None], sy, sx)[0]
    dy, dx = C.gradient(terrain)
    gx = C.sample(C.OPS, dx[None], sy, sx)[0] * gw / w
    gy = C.sample(C.OPS, dy[None], sy, sx)[0] * gh / h
    slope = np.maximum(np.hypot(gx, gy), 1e-6)
    bands, multi = int(p.get("bands", 4)), int(p.get("lines_per_band", 3))
    levels = [
        (b + 0.5) / bands + (j - (multi - 1) / 2) * float(p.get("band_spread", 1.4)) * 0.01
        for b in range(bands)
        for j in range(multi)
    ]
    levels = [v for v in levels if 0.005 < v < 0.995]
    line_width = max(0.25, 0.75 * scale)
    alpha = terrain_coverage(tone, slope, levels, line_width, float(p.get("line_opacity", 0.53)))
    ink, bg = C.rgb(p.get("ink", "#c9c7c0")), C.rgb(p.get("paper", "#1c1c1e"))
    out = np.broadcast_to(bg, x.shape).copy() + (ink - bg) * alpha[None]
    painted, glow = terrain_markers(h, w, p, terrain, dy, dx, levels)
    out = (
        out * (1 - painted[..., 3][None])
        + painted[..., :3].transpose(2, 0, 1) * painted[..., 3][None]
    )
    if np.any(glow):
        radius = max(1, float(p.get("glow", 0.45)) * 14 * scale)
        out += C.gaussian(glow, radius) * radius * 2
    return out


def terrain_markers(
    h: int, w: int, p: dict[str, Any], terrain: Any, dy: Any, dx: Any, levels: list[float]
) -> tuple[Any, Any]:
    from PIL import Image, ImageDraw, ImageFont

    gh, gw = terrain.shape
    scale = w / float(p.get("reference_width", 1200))
    layer = Image.new("RGBA", (w, h))
    draw = ImageDraw.Draw(layer)
    size = max(3, float(p.get("glyph_size", 21)) * scale)
    spacing = max(2, float(p.get("marker_spacing", 7)) * size)
    font = ImageFont.truetype(str(FONT), max(3, round(size * 0.8)))
    glyphs = "01+x=-'./"
    label = p.get("label_text", "PLATFORM")
    glow = np.zeros((3, h, w), np.float32)
    seed = int(p.get("seed", 0))

    def read(a: Any, y: float, z: float) -> float:
        sy = float(np.float32(y * gh / h))
        sx = float(np.float32(z * gw / w))
        iy, ix = math.floor(sy), math.floor(sx)
        fy, fx = sy - iy, sx - ix
        y0, y1 = min(gh - 1, max(0, iy)), min(gh - 1, max(0, iy + 1))
        x0, x1 = min(gw - 1, max(0, ix)), min(gw - 1, max(0, ix + 1))
        return float(
            (a[y0, x0] * (1 - fx) + a[y0, x1] * fx) * (1 - fy)
            + (a[y1, x0] * (1 - fx) + a[y1, x1] * fx) * fy
        )

    for band_id, level in enumerate(levels):
        for row in range(math.ceil(h / spacing)):
            for col in range(math.ceil(w / spacing)):
                n = np.array(
                    [float(C.noise(row + band_id * 0.37, col, seed + k)) for k in (41, 59, 73, 97)]
                )
                px, py = (col + 0.12 + 0.76 * n[0]) * spacing, (row + 0.12 + 0.76 * n[1]) * spacing

                tx, ty = read(dx, py, px) * gw / w, read(dy, py, px) * gh / h
                norm = tx * tx + ty * ty
                if norm < 1e-9:
                    continue
                change = (level - read(terrain, py, px)) / norm
                ox, oy = tx * change, ty * change
                if math.hypot(ox, oy) > spacing * 0.72:
                    continue
                px, py = px + ox, py + oy
                if (
                    not 0 <= px < w
                    or not 0 <= py < h
                    or abs(read(terrain, py, px) - level) > 0.035
                    or n[1] < 0.16
                ):
                    continue
                tangent = np.array([-ty, tx]) / math.sqrt(norm)
                hero = n[3] < float(p.get("heroes", 0.55)) * 0.22
                color = C.rgb(p.get("accent", "#ef282d") if hero else p.get("ink", "#c9c7c0"))[
                    :, 0, 0
                ]
                fill = (*tuple((color * 255).astype(int)), 242)
                if hero:
                    iy, ix = round(py), round(px)
                    if 0 <= iy < h and 0 <= ix < w:
                        glow[:, iy, ix] += color * 5
                if n[2] < float(p.get("boxes", 0.3)):
                    bw, bh = size * (1.1 + n[0] * 1.1), size * (0.55 + n[1] * 0.35)
                    if abs(tangent[1]) > abs(tangent[0]) * 1.4:
                        bw, bh = bh, bw
                    draw.rectangle((px - bw / 2, py - bh / 2, px + bw / 2, py + bh / 2), fill=fill)
                else:
                    count = 1 + int(n[0] ** 2 * 4)
                    word = glyphs[int(n[0] * len(glyphs)) % len(glyphs)] * count
                    if n[2] < float(p.get("boxes", 0.3)) + float(p.get("labels", 0.25)) * 0.3:
                        word = (
                            label[int(n[0] * max(1, len(label) - 3)) :][:3]
                            if label
                            else str(round(level * 100))
                        )
                    draw.text((px, py), word, font=font, anchor="mm", fill=fill, stroke_width=0)
    painted = np.asarray(layer, np.float32) / 255
    return painted, glow


def terrain_coverage(
    tone: Any, slope: Any, levels: list[float], width: float, opacity: float
) -> Any:
    """Evaluate only isolines within this pixel's antialiased support."""
    targets = np.sort(np.asarray(levels, np.float32))
    if not len(targets):
        return np.zeros_like(tone)
    reach = slope * (width + 0.75)
    low = np.searchsorted(targets, tone - reach, side="left")
    high = np.searchsorted(targets, tone + reach, side="right")
    alpha = np.zeros_like(tone)
    for offset in range(int((high - low).max())):
        selected = low + offset < high
        distance = np.abs(tone[selected] - targets[(low + offset)[selected]]) / slope[selected]
        cover = 1 - C.smooth(max(0, width - 0.75), width + 0.75, distance)
        alpha[selected] = 1 - (1 - alpha[selected]) * (1 - cover * opacity)
    return alpha


def hsv(hue: Any, saturation: float) -> Any:
    h = hue % 1 * 6
    a = 1 - saturation
    b = 1 - saturation * (h % 1)
    c = 1 - saturation * (1 - h % 1)
    i = np.floor(h).astype(int) % 6
    return np.stack(
        [
            np.choose(i, [np.ones_like(h), b, a + h * 0, a + h * 0, c, np.ones_like(h)]),
            np.choose(i, [c, np.ones_like(h), np.ones_like(h), b, a + h * 0, a + h * 0]),
            np.choose(i, [a + h * 0, a + h * 0, c, np.ones_like(h), np.ones_like(h), b]),
        ]
    )


def motion_ink(x: Any, p: dict[str, Any], f: Any, *, gpu: tuple[Any, Any] | None = None) -> Any:
    _, h, w = x.shape
    detailed = p.get("fine_detail", False)
    ratio = min(1, (1280 if detailed else 320) / w)
    small = C.resize(np.clip(x, 0, 1), max(4, round(h * ratio)), max(4, round(w * ratio)))
    state = f.state.setdefault("reference_trace", {})
    if (
        state.get("index", f.index - 1) != f.index - 1
        or state.get("shape", small.shape) != small.shape
    ):
        state.clear()
    state.update(index=f.index, shape=small.shape)
    if "anchors" not in state:
        state["anchors"] = [small.copy() for _ in range(3)]
    anchors = state["anchors"]
    camera = f.camera if p.get("camera_compensate", True) else None
    cameras = state.setdefault("cameras", [camera] * 3)
    ticks = state.setdefault("ticks", [-1, -1, -1])
    t = f.index / f.fps
    differences = []
    aligned = [camera_align(a, old, camera, ratio) for a, old in zip(anchors, cameras, strict=True)]
    middle = aligned[1]
    for j, (period, weight) in enumerate(((0.075, 0.7), (0.28, 0.9), (0.95, 0.45))):
        delta = small - aligned[j]
        difference = np.maximum(np.abs(C.lum(delta)), np.sqrt((delta * delta).sum(0)) * 0.474)
        differences.append(difference * weight)
        tick = math.floor(t / period)
        if tick != ticks[j]:
            anchors[j], ticks[j] = small.copy(), tick
            cameras[j] = camera
    energy = np.maximum.reduce(differences)
    p68, p90 = np.percentile(energy, [68, 90])
    threshold = float(
        np.clip(
            0.145
            - float(p.get("sensitivity", 0.74)) ** 0.72 * 0.132
            + p68 * (0.45 + float(p.get("noise_reject", 0.62)) * 1.85),
            0.006,
            0.34,
        )
    )
    if (energy > threshold).mean() > 0.58 or (p68 > 0.13 and p90 > 0.24):
        state.clear()
        return (gpu[1] if gpu is not None else x) * float(p.get("source", 0))
    body = float(p.get("body", 0.24))
    if body:
        energy = np.maximum.reduce(
            [
                energy,
                np.roll(energy, 1, 0) * body,
                np.roll(energy, -1, 0) * body,
                np.roll(energy, 1, 1) * body,
                np.roll(energy, -1, 1) * body,
            ]
        )
    gate = (
        C.smooth(
            threshold - max(0.003, threshold * 0.42),
            threshold + max(0.003, threshold * 0.42),
            energy,
        )
        ** 0.68
    )
    before, after = C.lum(middle), C.lum(small)
    gy, gx = C.gradient((before + after) / 2)
    dt = after - before
    norm = gx * gx + gy * gy + 0.0025
    u, v = -dt * gx / norm, -dt * gy / norm
    color = hsv(np.arctan2(v, u) / (2 * math.pi) + 0.5, 0.72)
    if detailed:
        gy = np.roll(after, -2, 0) - np.roll(after, 2, 0)
        gx = np.roll(after, -2, 1) - np.roll(after, 2, 1)
        norm = gx * gx + gy * gy + 0.0025
        u, v = -dt * gx / norm, -dt * gy / norm
        speed = np.clip(np.hypot(u, v) * 0.42, 0, 1)
        color = hsv(np.arctan2(v, u) / (2 * math.pi) + 0.5, 0.72)
        level = np.clip(
            (energy - threshold + max(0.003, threshold * 0.42))
            * (float(p.get("gain", 0.48)) * 10 + float(p.get("sensitivity", 0.74)) * 1.5),
            0,
            1,
        )
        energy = gate * level**0.68 * (0.72 + (0.1 + after * 0.38) * 0.45)
        energy *= 0.58 + speed * 0.72
    else:
        energy = gate * (
            1
            - np.exp(
                -energy
                * (float(p.get("gain", 0.48)) * 10 + float(p.get("sensitivity", 0.74)) * 1.5)
            )
        )
    if not p.get("direction_color", True):
        color = np.broadcast_to(C.rgb(p.get("ink", "#d8ff3e")), small.shape)
    tint = float(p.get("source_tint", 0))
    if tint:
        color = color * (1 - tint) + small / np.maximum(small.max(0), 0.05)[None] * tint
    marks = color * energy[None]
    count = max(1, round((float(p.get("trail_seconds", 1.1)) + 0.95) * f.fps))
    decay = math.exp(-1 / max(1, float(p.get("trail_seconds", 1.1)) * f.fps))
    memory = ink_memory(
        state, marks, energy, count, decay, float(p.get("ink_strength", 0.82)), detailed
    )
    if gpu is not None:
        if __package__:
            from cutwright.modules.fx.runners.fx_cells_gpu import trace_material
        else:
            from fx_cells_gpu import trace_material
        return trace_material(gpu[0], gpu[1], memory, p, f)
    out = C.resize(memory, h, w)
    material = p.get("material", "dither")
    if material == "dither":
        size = max(
            1, float(p.get("cell_size", 9)) * w / float(p.get("reference_width", 1200)) * 0.18
        )
        yy, xx = np.mgrid[:h, :w]
        table = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32)
        level = np.clip(C.lum(out) * 1.38, 0, 1) if detailed else out.max(0)
        a = (
            table[(yy / size).astype(int) % 4, (xx / size).astype(int) % 4]
            + (0.5 if detailed else 0)
        ) / 16
        out = (
            out
            / np.maximum((out.max(0) if detailed else level)[None], 1e-5)
            * (level > a)[None]
            * (0.56 + 0.44 * level[None] if detailed else np.minimum(1, level[None] + 0.35))
        )
    elif material == "ascii":
        g = C.grid(out, {**p, "cell_size": max(12, p.get("cell_size", 9))}, f, 0.6)
        chars = " .:-=+*#%@"
        coverage = glyph(g, np.floor(g.tone * (len(chars) - 0.001)), chars)
        out *= coverage[None]
    elif material == "halftone":
        g = C.grid(out, p, f)
        radius = g.expand(np.sqrt(g.tone) * 0.48)
        coverage = C.smooth(-0.5, 0.5, (radius - np.hypot(g.py - 0.5, g.px - 0.5)) * g.ch)
        out *= coverage[None]
    return out + x * float(p.get("source", 0))


def ink_memory(
    state: dict[str, Any],
    marks: Any,
    energy: Any,
    count: int,
    decay: float,
    strength: float,
    detailed: bool,
) -> Any:
    """An exact finite recurrence; log weights recover after high-activity underflow."""
    weight = np.maximum(0.001, (1 - energy) * decay)
    queue = state.setdefault("memory", deque())
    if state.get("count", count) != count:
        queue.clear()
        state.pop("ink", None)
        state.pop("product", None)
        state.pop("log_product", None)
    log_weight = np.log(weight.astype(np.float64))
    log_product = state.get("log_product", 0) + log_weight
    previous = state["ink"] * weight[None] if "ink" in state else np.zeros_like(marks)
    injection = marks * strength
    if detailed:
        injection = np.maximum(injection * energy[None], injection * 0.82 - previous)
    memory = previous + injection
    if len(queue) == count:
        old_weight, old_ink, old_strength = queue.popleft()
        if isinstance(old_weight, bytes):
            old_weight = np.frombuffer(zlib.decompress(old_weight), np.float32).reshape(
                energy.shape
            )
            old_ink = np.frombuffer(zlib.decompress(old_ink), np.float32).reshape(marks.shape)
        log_product -= np.log(old_weight.astype(np.float64))
        memory -= np.exp(log_product)[None] * old_ink * old_strength
    saved_ink = injection if detailed else marks
    if detailed and energy.size >= 100_000:
        saved_weight = zlib.compress(weight.tobytes(), 1)
        saved_ink = zlib.compress(saved_ink.tobytes(), 1)
    else:
        saved_weight, saved_ink = weight.copy(), saved_ink.copy()
    queue.append((saved_weight, saved_ink, 1 if detailed else strength))
    memory = np.maximum(memory, 0)
    state.update(ink=memory, log_product=log_product, count=count)
    return memory


def camera_align(image: Any, previous: Any, current: Any, scale: float) -> Any:
    """Read an anchor in this frame's coordinates; the camera path uses full-frame pixels."""
    if previous is None or current is None or np.array_equal(previous, current):
        return image
    _, h, w = image.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    angle, dx, dy = current
    old_angle, old_dx, old_dy = previous
    c, s = math.cos(angle), math.sin(angle)
    world_x = c * (xx - cx) - s * (yy - cy) + (dx - old_dx) * scale
    world_y = s * (xx - cx) + c * (yy - cy) + (dy - old_dy) * scale
    c, s = math.cos(old_angle), math.sin(old_angle)
    return C.sample(C.OPS, image, -s * world_x + c * world_y + cy, c * world_x + s * world_y + cx)


def finite_max(value: Any, f: Any, decay: float, count: int = 180) -> Any:
    """A decaying max with an exact finite window, including after still/segment warm-up."""
    state = f.state.setdefault("reference_finish", {})
    if (
        state.get("index", f.index - 1) != f.index - 1
        or state.get("shape", value.shape) != value.shape
    ):
        state.clear()
    queue = state.setdefault("queue", deque())
    log_decay = state.get("log_decay", 0.0) + math.log(max(decay, 1e-12))
    old = state.get("value", 0 * value) * decay
    owner = np.where(value >= old, f.index, state.get("owner", f.index))
    out = np.maximum(value, old)
    queue.append((f.index, value.copy(), log_decay))
    while queue[0][0] <= f.index - count:
        queue.popleft()
    expired = owner <= f.index - count
    if np.any(expired):
        best, winners = np.zeros(expired.sum(), np.float32), np.full(expired.sum(), queue[0][0])
        for i, value, born in queue:
            candidate = value[expired] * math.exp(log_decay - born)
            winners = np.where(candidate > best, i, winners)
            best = np.maximum(best, candidate)
        out[expired], owner[expired] = best, winners
    state.update(
        index=f.index, shape=value.shape, value=out.copy(), owner=owner, log_decay=log_decay
    )
    return out


def tone_atlas(x: Any, p: dict[str, Any], f: Any) -> Any:
    if p.get("style") == "terrain":
        return contour_terrain(x, p, f)
    if p.get("style") == "lines":
        return contour_lines(x, p, f)
    kind = p.get("style", "dots")
    size = float(p.get("glyph_size", 21)) * (1 if kind == "blocks" else 0.8)
    settings = {**p, "cell_size": size, "gamma": 1, "contrast": 1, "brightness": 0}
    g = C.grid(x, settings, f)
    tone = C.lum(g.colors)
    dy, dx = C.gradient(tone)
    edge = np.hypot(dx, dy)
    luma = C.lum(x)
    micro = np.abs(luma - C.gaussian(luma[None], max(0.5, x.shape[2] / 600))[0])
    detail = C.grid(np.stack([micro] * 3), settings, f).colors[0]
    n = C.noise(g.cy, g.cx, int(p.get("seed", 0)))
    energy = np.clip((edge + detail * 2.5) * 3.5 - 0.055, 0, 0.6) * (0.5 + 0.5 * n)
    ink, bg = C.rgb(p.get("ink", "#c9c7c0")), C.rgb(p.get("paper", "#1c1c1e"))
    if kind == "blocks":
        coverage = (g.px > 0.06) & (g.px < 0.94) & (g.py > 0.06) & (g.py < 0.94)
        yy, xx = np.mgrid[: x.shape[1], : x.shape[2]]
        texture = 0.1 + C.noise(yy, xx, 47) * 0.9
        out = (
            bg * (0.6 + 0.1 * coverage)[None]
            + ink * g.expand(energy)[None] * (coverage * texture)[None]
        )
    else:
        index = (n * 4).astype(int)
        coverage = glyph(g, index, "+xo.", scale=1.25)
        level = np.clip(energy * 3, 0, 0.95) * float(p.get("line_opacity", 0.9))
        out = bg * 0.03 + ink * g.expand(level)[None] * coverage[None]
    label = p.get("label_text", "PLATFORM")
    if label:
        labels = (n < float(p.get("labels", 0.25)) * 0.025) & (energy > 0.1)
        index = (C.noise(g.cy, g.cx, 71) * len(label)).astype(int)
        a = glyph(g, index, label) * g.expand(labels)
        out += C.rgb("#ffe5c2") * a[None] * 0.8
    probability = 0.006 if kind == "blocks" else 0.02
    selected = (n > 1 - float(p.get("heroes", 0.55)) * probability) & (energy > 0.1)
    yy, xx = g.py - 0.5, g.px - 0.5
    hot = g.expand(selected) * (
        coverage if kind == "blocks" else (np.abs(xx) < 0.4) & (np.abs(yy) < 0.4)
    )
    accent = C.rgb(p.get("accent", "#ef282d")) ** 2
    core = accent * hot[None] * 2.8
    out += core
    radius = max(1, g.ch * (0.6 if kind == "blocks" else 0.4))
    out += C.gaussian(core, radius) * 2.5
    if kind == "dots":
        warm = g.expand((energy > 0.18) & (n < 0.32))
        out += C.rgb("#ffe4aa") * (coverage * warm)[None] * 1.5 * float(p.get("line_opacity", 0.9))
        # glow lights a few warm cells as filled squares with a soft halo.
        lit = g.expand((energy > 0.18) & (n < 0.1)) * ((np.abs(xx) < 0.36) & (np.abs(yy) < 0.36))
        blobs = C.rgb("#ffe4aa") * lit[None] * float(p.get("glow", 0))
        out += blobs + C.gaussian(blobs, max(1, g.ch * 0.9)) * 3
    return out


def contour_lines(x: Any, p: dict[str, Any], f: Any) -> Any:
    scale = x.shape[2] / float(p.get("reference_width", 1200))
    y = C.lum(C.gaussian(x, max(0.25, float(p.get("smoothing", 1.65)) * scale)))
    y = np.clip((y - 0.5) * float(p.get("contrast", 1)) + 0.5, 0, 1)
    y = y ** (1 / float(p.get("gamma", 1))) * int(p.get("bands", 16))
    gy, gx = C.gradient(y)
    gradient = np.hypot(gx, gy)
    slope = np.maximum(gradient, 0.001)
    distance = np.abs((y + 0.5) % 1 - 0.5) / slope
    line = 1 - C.smooth(0.15, 1.05, distance)
    line *= C.smooth(0.0001, 0.003, gradient)
    split = max(1, round(scale * 4))
    ink = C.rgb(p.get("ink", "#d8ff3e"))
    channels = np.stack([np.roll(line, split, 1), line, np.roll(line, -split, 1)]) * ink
    g = C.grid(x, {**p, "cell_size": 16}, f)
    chosen = (C.noise(g.cy, g.cx, 25) > 0.997) & (g.tone > 0.55)
    sparks = g.expand(chosen) * np.exp(-((g.px - 0.5) ** 2 + (g.py - 0.5) ** 2) * 20)
    return (
        C.rgb(p.get("paper", "#161616")) * 0.5
        + channels * 0.85
        + sparks[None]
        + horizontal_fragments(channels, p, f)
    )


def horizontal_fragments(channels: Any, p: dict[str, Any], f: Any) -> Any:
    """Thin displaced contour echoes, broken into moving horizontal runs."""
    amount = float(p.get("horizontal_glitch", 0))
    if amount <= 0:
        return channels * 0
    _, h, w = channels.shape
    scale = w / float(p.get("reference_width", 1200))
    length = max(2, float(p.get("glitch_length", 120)) * scale)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    row = np.floor(yy / max(1, round(2 * scale)))
    cycle = (C.clock(p, f) / float(p.get("loop_seconds", 5))) % 1
    ticks = max(1, round(float(p.get("glitch_speed", 6)) * p.get("loop_seconds", 5)))
    tick = math.floor(cycle * ticks)
    seed = int(p.get("seed", 0)) + 43
    n = C.noise(row, tick, seed)
    gate = n > 1 - amount * 0.1
    offset = (C.noise(row, tick, seed + 11) * 2 - 1) * length * 1.8
    offset += np.sin(cycle * 2 * math.pi * ticks + row) * scale * 2
    echo = C.sample(C.OPS, channels, yy, xx - offset, "zero") * gate[None]
    radius = max(1, length * 0.2)
    echo = hblur(C.OPS, echo, radius) * max(1, radius * 0.8)
    runs = C.noise(row, np.floor(xx / max(4, length * 0.6)) + tick, seed + 29) > 0.7
    return np.maximum(echo, 0) * (runs & gate)[None] * amount


def ascii_cells(x: Any, p: dict[str, Any], f: Any) -> Any:
    """Tone-ranked glyphs with a shared baseline and directional edge marks."""
    _, h, w = x.shape
    columns = int(p.get("columns", 0)) or max(
        1, round(w / max(2, p.get("cell_size", 20) * w / 1200 * 0.52))
    )
    rows = max(2, round(h / w * columns * 0.52))
    settings = {**p, "columns": columns, "_rows": rows, "gamma": 1, "contrast": 1, "brightness": 0}
    g = C.grid(linear(np.clip(x, 0, 1)), settings, f, 0.52, fractional=True)
    raw = C.lum(g.colors)
    tone = np.clip(
        (raw ** (1 / float(p.get("gamma", 1))) - 0.5) * float(p.get("contrast", 1))
        + 0.5
        + float(p.get("brightness", 0)),
        0,
        1,
    ) ** float(p.get("density_bias", 1))
    tone = np.clip(tone + g.tone - raw, 0, 1)
    floor = float(p.get("floor", 0))
    n = C.noise(g.cy, g.cx, int(p.get("seed", 0)))
    tone = np.maximum(tone, np.where(n > 0.52, floor * (0.026 + 0.075 * n * n), 0))
    chars = p.get("charset", " .,:;irsXA253hMHGS9B&@")
    regular = not p.get("bold", True)
    table = atlas(chars + "-\\|/", g.ch, g.cw, regular=regular, stretch=True)
    ascending = table[0].mean() <= table[len(chars) - 1].mean()
    index = np.rint((tone if ascending else 1 - tone) * (len(chars) - 1)).astype(int)
    gy, gx = C.gradient(raw)
    magnitude = np.hypot(gx, gy) * 8
    direction = np.rint((np.arctan2(gx, -gy) % math.pi) / (math.pi / 4)).astype(int) % 4
    index = np.where(magnitude * float(p.get("structure", 0)) > 0.1, len(chars) + direction, index)
    yy = np.clip((g.py * table.shape[1]).astype(int), 0, table.shape[1] - 1)
    xx = np.clip((g.px * table.shape[2]).astype(int), 0, table.shape[2] - 1)
    cover = table[g.expand(index), yy, xx]
    gap = float(p.get("gap", 0)) / 2
    cover *= (g.px >= gap) & (g.px < 1 - gap) & (g.py >= gap) & (g.py < 1 - gap)
    color = C.ink(g, p) * (1 + (float(p.get("emission", 1)) - 1) * tone[None])
    bg = linear(C.rgb(p.get("paper", "#050209")))
    out = display(np.maximum(bg + (g.expand(color) - bg) * cover[None], 0))
    out += x * float(p.get("photo", 0))
    amber = float(p.get("amber_glow", 0))
    if amber:
        warm = C.smooth(0.01, 0.12, np.minimum(g.colors[0], g.colors[1]) - g.colors[2])
        light = np.maximum(out - display(bg), 0) * g.expand(warm)[None]
        light = C.lum(light)[None] * C.rgb("#ff9b3d")
        radius = max(
            0.3, float(p.get("glow_radius", 18)) * w / float(p.get("reference_width", 1200))
        )
        halo = C.gaussian(light, radius * 0.25) * 0.7 + C.gaussian(light, radius) * 0.6
        out += np.maximum(halo, 0) * amber
    glow = float(p.get("glow", 0))
    if glow:
        out += (
            C.gaussian(
                np.maximum(out - display(bg), 0), max(0.3, p.get("glow_radius", 10) * w / 1200)
            )
            * glow
        )
    return out


def ghost_cells(x: Any, p: dict[str, Any], f: Any, aspect: float = 0.65) -> Any:
    g = C.grid(x, p, f, aspect, fractional=True)
    regular = not p.get("bold", True)
    chars = p.get("charset", " .,:;irsXA253hMHGS9B&@")
    table = atlas(chars, g.ch, g.cw, regular=regular)
    ascending = table[0].mean() <= table[-1].mean()
    gamma, contrast = float(p.get("gamma", 1)), float(p.get("contrast", 1))
    raw = C.lum(g.colors)
    before = np.clip((raw - 0.5) * contrast + 0.5 + p.get("brightness", 0), 0, 1) ** (1 / gamma)
    tone = np.clip(
        (raw ** (1 / gamma) - 0.5) * contrast + 0.5 + p.get("brightness", 0) + g.tone - before, 0, 1
    )
    tone = tone if ascending else 1 - tone
    coverage = glyph(g, np.floor(tone * (len(chars) - 0.001)), chars, regular=regular)
    gap = float(p.get("gap", 0)) / 2
    coverage *= (g.px >= gap) & (g.px < 1 - gap) & (g.py >= gap) & (g.py < 1 - gap)
    bg = C.rgb(p.get("paper", "#0c0c0c"))
    color = g.expand(C.ink(g, p)) * 1.15
    out = bg + (color - bg) * coverage[None] + x * float(p.get("photo", 0))
    glow = float(p.get("glow", 0))
    if glow:
        out += (
            C.gaussian(
                np.maximum(out - bg, 0), max(0.3, p.get("glow_radius", 10) * x.shape[2] / 1200)
            )
            * glow
        )
    return out


def glyph_haze(x: Any, p: dict[str, Any], f: Any) -> Any:
    return ghost_cells(x, p, f)


def thermal_reveal(x: Any, p: dict[str, Any], f: Any) -> Any:
    scale = x.shape[2] / float(p.get("reference_width", 1200))
    t = C.clock(p, f)
    y = C.lum(C.gaussian(x, max(0.25, float(p.get("soften", 16)) * scale)))
    detail = float(p.get("detail", 0.22))
    y = y * (1 - detail) + C.lum(x) * detail
    y = np.clip(
        (y - 0.5) * float(p.get("contrast", 1)) + 0.5 + float(p.get("brightness", 0)), 0, 1
    ) ** (1 / float(p.get("gamma", 1)))
    phase = (
        (1 - y) * float(p.get("cycles", 1.2))
        + float(p.get("phase", 0.67))
        + t * float(p.get("drift", 0.06))
        + math.sin(t / float(p.get("loop_seconds", 5)) * 2 * math.pi)
        * float(p.get("breathe", 0.12))
    ) % 1
    palette = np.array([C.rgb(v)[:, 0, 0] for v in p["colors"]])
    pos = phase * len(palette)
    index = np.floor(pos).astype(int) % len(palette)
    mix = pos - np.floor(pos)
    mix = mix * mix * (3 - 2 * mix)
    color = (
        palette[index] * (1 - mix[..., None]) + palette[(index + 1) % len(palette)] * mix[..., None]
    ).transpose(2, 0, 1)
    progress = float(p.get("progress", 0))
    if p.get("automatic", True):
        time = t / float(p.get("duration", 3.433333))
        if time < 0.2913:
            progress = 1.13 * max(0, time / 0.2913)
        elif time <= 0.4951:
            progress = 1.13
        else:
            step = float(np.clip((time - 0.4951) / (0.7476 - 0.4951), 0, 1))
            progress = 1.13 + 0.87 * step
    soft = float(p.get("soft", 0.18))
    stage = progress / 1.13 if progress <= 1.13 else (progress - 1.13) / 0.87
    edge = stage * (1 + 2 * soft) - soft
    amount = C.smooth(-soft, soft, edge - y)
    if progress > 1.13:
        amount = 1 - amount
    if progress <= 0 or progress >= 2:
        amount *= 0
    return x + (color - x) * amount[None]


def halftone_cells(x: Any, p: dict[str, Any], f: Any) -> Any:
    _, h, w = x.shape
    scale = w / float(p.get("reference_width", 1200))
    cell = max(2, float(p.get("cell_size", 22)) * scale)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    angle = math.radians(float(p.get("angle", 64)))
    c, s = math.cos(angle), math.sin(angle)
    u, v = xx * c + yy * s, -xx * s + yy * c
    cx, cy = (np.floor(u / cell) + 0.5) * cell, (np.floor(v / cell) + 0.5) * cell
    y = np.clip(
        (C.lum(x) - 0.5) * float(p.get("contrast", 1)) + 0.5 + float(p.get("brightness", 0)), 0, 1
    ) ** (1 / float(p.get("gamma", 1)))
    tone = C.sample(C.OPS, y[None], cx * s + cy * c, cx * c - cy * s)[0] ** 2.2
    radius = tone**1.15 * cell * 0.45 * float(p.get("dot_scale", 1.17))
    distance = np.hypot(u % cell - cell / 2, v % cell - cell / 2)
    cover = np.clip(radius - distance + 0.5, 0, 1)
    ink, bg = C.rgb(p.get("ink", "#ffb54d")), C.rgb(p.get("paper", "#0b0705"))
    light = ink * (cover * (0.35 + tone**0.5) * 0.8)[None]
    light += C.rgb("#fff2a0") * (cover * tone * float(p.get("dot_light", 0.37)) * 0.12)[None]
    halo = C.gaussian(light, max(0.5, cell * 0.65)) * float(p.get("edge_glow", 0.53)) * 3
    halo *= C.rgb("#ff6430")
    return bg * 0.3 + light + halo


def tone_stripes(x: Any, p: dict[str, Any], f: Any) -> Any:
    g = C.grid(x, p, f, 1.4)
    color = np.where(
        (g.tone > 0.7)[None],
        C.rgb(p.get("light", "#bcbcbc")),
        np.where(
            (g.tone > 0.4)[None], C.rgb(p.get("medium", "#787878")), C.rgb(p.get("dark", "#343434"))
        ),
    )
    coverage = np.abs(g.py - 0.5) < float(p.get("stripe_height", 0.8)) / 2
    # Cells below the shadow tone drop their bar and keep only a small dark mark.
    deep = g.expand(g.tone < float(p.get("shadow", 0)))
    mark = (np.abs(g.py - 0.5) < 0.1) & (np.abs(g.px - 0.5) < 0.1)
    coverage = np.where(deep, mark, coverage)
    bg = C.rgb(p.get("paper", "#161616"))
    out = bg + (g.expand(color) - bg) * coverage[None]
    if p.get("axis"):
        out[:, :, x.shape[2] // 2] = C.rgb(p.get("ink", "#ffffff"))[:, 0]
    return out


def edge_sparks(x: Any, p: dict[str, Any], f: Any) -> Any:
    scale = x.shape[2] / float(p.get("reference_width", 1200))
    filtered = C.gaussian(x, max(0.4, scale * 2))
    g = C.grid(filtered, {**p, "cell_size": max(2, 12 / p.get("density", 4))}, f)
    gy, gx = C.gradient(g.tone)
    threshold = float(p.get("threshold", 0.09))
    edge = C.smooth(threshold * 0.35, threshold * 0.35 + 0.16, np.hypot(gx, gy))
    n = C.noise(g.cy, g.cx, int(p.get("seed", 10)))
    hit = n < edge * 0.36 + g.tone * float(p.get("dust", 0.25)) * 0.003
    phase = 2 * math.pi * C.clock(p, f) / float(p.get("loop_seconds", 5))
    energy = hit * (0.7 + 0.3 * g.tone) * (1 + p.get("twinkle", 0.34) * np.sin(n * 100 + phase))
    radius = float(p.get("particle_size", 3.1)) * scale / g.ch * 0.5
    cover = np.clip((radius - np.hypot(g.px - 0.5, g.py - 0.5)) * g.ch + 0.5, 0, 1)
    particles = (
        C.rgb(p.get("ink", "#ffe2ba"))
        * (g.expand(energy) * cover)[None]
        * float(p.get("gain", 1.6))
    )
    particles += C.gaussian(particles, max(0.5, scale * 2)) * 1.3
    return C.rgb(p.get("paper", "#030309")) + x * p.get("underlay", 0.02) + particles


def cell_finish(x: Any, p: dict[str, Any], f: Any) -> Any:
    _, h, w = x.shape
    ratio = min(1, 320 / max(h, w))
    small = C.resize(x, max(2, round(h * ratio)), max(2, round(w * ratio)))
    threshold, knee = float(p.get("threshold", 0.35)), float(p.get("knee", 0.5))
    brightness = small.max(0)
    mask = C.smooth(threshold - knee * 0.5, threshold + knee * 0.5, brightness)
    light = small * mask[None]
    halos = 0 * small
    palette = C.BLOOM_PALETTES.get(p.get("bloom_palette", "mono"))
    for k, (sigma, weight) in enumerate(((0.8, 0.45), (2.5, 0.3), (7.5, 0.17), (20, 0.08))):
        halo = C.gaussian(light, sigma * float(p.get("radius", 1.8))) * weight
        halos += halo if palette is None else halo * palette[k]
    tint = C.rgb(p.get("bloom_tint", "#ffffff")) if palette is None else 1
    finish = halos * tint * float(p.get("bloom", 0.6))
    strength = float(p.get("streaks", 0))
    if strength:
        threshold = float(p.get("streak_threshold", 0.55))
        light = small * C.smooth(max(0, threshold - 0.1), threshold + 0.1, brightness)[None]
        angle = math.radians(float(p.get("angle", 18)))
        length = float(p.get("length", 0.42)) * small.shape[2]
        streak = C.zero_taps(light, C.streak_taps(length, angle))
        finish += streak * C.rgb(p.get("tint", "#ff6beb")) * strength * 0.25
    out = x + C.resize(finish, h, w)
    decay = float(p.get("trails", 0))
    if decay:
        small = finite_max(small, f, decay * 0.995)
        out = np.maximum(out, C.resize(small, h, w))
    out *= 2 ** float(p.get("exposure", 0))
    y = C.lum(out)[None]
    return y + (out - y) * float(p.get("saturation", 1))


DRAWERS = {
    "phrase_dither": phrase_dither,
    "tone_atlas": tone_atlas,
    "motion_ink": motion_ink,
    "cell_finish": cell_finish,
    "ascii_cells": ascii_cells,
    "glyph_haze": glyph_haze,
    "thermal_reveal": thermal_reveal,
    "halftone_cells": halftone_cells,
    "tone_stripes": tone_stripes,
    "edge_sparks": edge_sparks,
}
