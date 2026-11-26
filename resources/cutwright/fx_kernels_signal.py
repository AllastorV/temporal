"""Temporal and analog reference effects, shared by numpy and torch workers."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import numpy as np

if __package__:
    from cutwright.modules.fx.runners.fx_base import (
        Frame,
        blur,
        downscale,
        hex_rgb,
        kernel,
        luma,
        sample,
        screen,
        upscale,
    )
    from cutwright.modules.fx.runners.fx_kernels_art import crt
    from cutwright.modules.fx.runners.fx_kernels_paint import flow
else:
    from fx_base import (  # type: ignore[no-redef]
        Frame,
        blur,
        downscale,
        hex_rgb,
        kernel,
        luma,
        sample,
        screen,
        upscale,
    )
    from fx_kernels_art import crt  # type: ignore[no-redef]
    from fx_kernels_paint import flow  # type: ignore[no-redef]


def _noise(o: Any, seed: int, shape: tuple[int, ...]) -> Any:
    return o.asarray(np.random.default_rng(seed & 0xFFFFFFFF).random(shape, dtype=np.float32))


@kernel("pixel_scan")
def pixel_scan(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    _, h, w = x.shape
    xp = o.xp
    short = min(h, w)
    ys, xs = o.grid(h, w)
    theta = math.radians(float(p["angle"]))
    dx, dy = math.cos(theta), math.sin(theta)
    length = abs(dx) * w + abs(dy) * h
    along = ((xs - w / 2) * dx + (ys - h / 2) * dy) / length + 0.5
    across = -(xs - w / 2) * dy + (ys - h / 2) * dx
    t = f.index / f.fps - float(p.get("_origin", 0)) - float(p["offset"])
    progress = min(1.0, max(0.0, t / float(p["duration"]))) if p["auto"] else float(p["progress"])
    if p["reverse"]:
        along = 1 - along
        dx, dy = -dx, -dy
    size = max(1.0, short / float(p["blocks"]))
    count = max(4, math.ceil((w + h) / size) + 4)
    tick = math.floor(t * float(p["speed"]))
    rnd = _noise(o, f.seed + tick * 3571, (2, count, 1))
    # Hashing the row index keeps one coherent strip along the scan direction.
    idx = ((across / size) % count).clip(0, count - 1)
    row = o.gather(rnd, idx, xp.zeros_like(idx))
    width = float(p["width"])
    distance = along - progress + (row[0] - 0.5) * width * float(p["scatter"])
    edge = xp.exp(-((distance / max(width * 0.35, 1e-5)) ** 2))
    reveal = (-distance / max(1 / length, width * 0.05) + 0.5).clip(0, 1)
    if progress <= 0:
        reveal, edge = reveal * 0, edge * 0
    elif progress >= 1:
        reveal, edge = reveal * 0 + 1, edge * 0
    shift = (row[1] - 0.5) * float(p["stretch"]) * short * edge
    stretched = sample(o, x, ys + shift * dy, xs + shift * dx)
    chroma = float(p["chromatic"]) * short * 0.01 * edge
    red = sample(o, stretched, ys, xs + chroma)[0]
    blue = sample(o, stretched, ys, xs - chroma)[2]
    stretched = xp.stack([red, stretched[1], blue])
    visible = 1 - reveal if p["mode"] == "erase" else edge if p["mode"] == "band" else reveal
    out = stretched * visible[None]
    color = o.asarray(hex_rgb(p["color"]))[:, None, None]
    light_mask = edge if f.alpha is None else edge * f.alpha
    light = light_mask[None] * color * float(p["glow"])
    spread = float(p["glow_size"]) * short
    if spread > 0:
        light = light + blur(o, light, spread) * 1.8
    if f.alpha is not None:
        energy = xp.maximum(xp.maximum(light[0], light[1]), light[2]).clip(0, 1)
        alpha = screen(f.alpha * visible, energy)
        f.state["alpha_replace"] = alpha
        out = screen(out * f.alpha[None], light.clip(0, 1)) / alpha[None].clip(1e-6, 1)
        return out.clip(0, 1)
    return screen(out, light.clip(0, 1)).clip(0, 1)


def _history(o: Any, x: Any, f: Frame, count: int, scale: int) -> dict[int, Any]:
    small = downscale(o, x, scale)
    ring = f.state.get("ring", [])
    if ring and (ring[-1][0] != f.index - 1 or ring[-1][1].shape != small.shape):
        ring = []
    ring.append((f.index, np.asarray(o.numpy(small), np.float16)))
    f.state["ring"] = ring[-(count + 1) :]
    return dict(f.state["ring"])


def ghost(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    _, h, w = x.shape
    xp = o.xp
    frames, spacing = int(p["frames"]), int(p["spacing"])
    past = _history(o, x, f, frames * spacing, int(p.get("history_scale", 1)))
    ys, xs = o.grid(h, w)
    mode = p["mode"]
    theta = math.radians(float(p["angle"]))
    dx, dy = math.cos(theta), math.sin(theta)
    span = max(1, int(p["slices"]))
    axis = xs / w if mode == "vertical" else (xs / w * dx + ys / h * dy)
    phase = (xp.floor(axis * span) % frames) + 1
    out = x
    for k in range(frames, 0, -1):
        old = past.get(f.index - k * spacing)
        if old is None:
            continue
        old = upscale(o, o.asarray(old), h, w)
        distance = float(p["stretch"]) * min(h, w) * k / frames
        if mode == "smear":
            old = sample(o, old, ys - distance * dy, xs - distance * dx)
        elif mode == "depth":
            depth = p.get("_depth")
            if depth is not None:
                bend = (depth - 0.5) * float(p["depth_amount"]) * distance
                old = sample(o, old, ys + bend * dy, xs + bend * dx)
        mask = xp.abs(luma(old) - luma(x))
        mask = ((mask - float(p["threshold"])) / max(0.02, float(p["threshold"]))).clip(0, 1)
        if not p["motion_only"]:
            mask = mask * 0 + 1
        sigma = float(p["softness"]) * min(h, w)
        if sigma > 0:
            mask = blur(o, mask[None], sigma)[0]
        if mode in ("vertical", "angled"):
            mask = mask * (phase == k)
        alpha = mask[None] * float(p["amount"]) * float(p["decay"]) ** (k - 1)
        top = (
            screen(out, old)
            if p["blend"] == "screen"
            else xp.maximum(out, old)
            if p["blend"] == "lighten"
            else old
        )
        out = out + (top - out) * alpha
    return out.clip(0, 1)


def _mosh_cells(o: Any, seed: int, gh: int, gw: int) -> Any:
    """Contiguous groups of codec tiles, not a full-screen salt-and-pepper grid."""
    cells = _noise(o, seed, (1, max(2, math.ceil(gh / 4)), max(2, math.ceil(gw / 4))))
    cells = upscale(o, cells, gh, gw)
    return (cells - cells.min()) / (cells.max() - cells.min()).clip(1e-6, 1)


def mosh(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Motion prediction with an imperfect reference frame, not random UV distortion.

    The reference's clear subjects come from fresh prediction residuals. A scene cut
    deliberately omits that one residual, like a missing intra frame: the old shot
    remains while subsequent moving detail is drawn into it.
    """
    _, h, w = x.shape
    xp, st = o.xp, f.state
    short, mode = min(h, w), p["mode"]
    ys, xs = o.grid(h, w)
    gray = downscale(o, luma(x)[None], max(1, math.ceil(short / 192)))
    fresh = st.get("index") != f.index - 1 or st.get("size") != (h, w)
    cut = not fresh and float(o.numpy(xp.abs(gray - st["gray"]).mean())) > 0.12
    reset = fresh or f.index % int(p["period"]) == 0 or (cut and p["cut_reset"])
    active = None
    if reset:
        held = xp.zeros_like(x) if mode == "trails" else x
        velocity = xp.zeros_like(xp.concatenate([gray, gray], axis=0))
    else:
        fy, fx = flow(o, gray[0], st["gray"][0])
        measured = xp.stack([fy, fx])
        memory = float(p.get("vector_memory", 0.25))
        velocity = st["velocity"] if cut else st["velocity"] * memory + measured * (1 - memory)
        scale = o.asarray([h / gray.shape[1], w / gray.shape[2]])[:, None, None]
        original = upscale(o, measured, h, w) * scale
        vectors = upscale(o, velocity, h, w) * scale
        size = max(2, round(float(p["block"]) * short))
        if mode not in ("liquid", "trails") and float(p["block"]) > 0:
            blocks = downscale(o, vectors, size)
            vectors = o.gather(
                blocks,
                (ys // size).clip(0, blocks.shape[1] - 1),
                (xs // size).clip(0, blocks.shape[2] - 1),
            )
        magnitude = xp.sqrt(vectors[0] ** 2 + vectors[1] ** 2) / short
        active = (magnitude / max(0.0002, float(p["threshold"]))).clip(0, 1)
        offset = vectors * float(p["motion"]) * active[None]
        if mode == "blocks":
            offset = xp.floor(offset + 0.5)
        held = sample(o, st["held"], ys + offset[0], xs + offset[1])
        predicted = sample(o, st["source"], ys + original[0], xs + original[1])
        residual = x - predicted
        if mode == "trails":
            # Local prediction errors reveal fragments; still areas stay dark.
            reveal = (
                (xp.abs(residual).mean(axis=0) - max(0.01, float(p["threshold"]) * 8)) / 0.05
            ).clip(0, 1)
            if cut:
                reveal = reveal * 0
            held = xp.maximum(
                held * float(p["decay"]), x * reveal[None] * float(p.get("residual", 1))
            )
        elif not cut:
            held = held + residual * float(p.get("residual", 1))
        refresh = float(p["refresh"])
        if mode in ("feedback", "blocks", "chroma"):
            gh, gw = math.ceil(h / size), math.ceil(w / size)
            cells = _mosh_cells(o, f.seed + (f.index // 3) * 3571, gh, gw)
            patches = o.gather(cells, (ys // size).clip(0, gh - 1), (xs // size).clip(0, gw - 1))
            refresh = (patches < refresh) * 1.0
        held = (held * (1 - refresh) + x * refresh).clip(0, 1)
    st.update(index=f.index, size=(h, w), gray=gray, source=x, velocity=velocity, held=held)
    display = (held + x * 0.04).clip(0, 1) if mode == "trails" else held
    if mode == "chroma" and active is not None:
        # Separation belongs to this displayed frame, never to the feedback buffer.
        # Repeated RGB offsets would turn the whole picture into a growing rainbow.
        separation = float(p["chromatic"]) * short * 0.004 * active
        display = xp.stack(
            [
                sample(o, display, ys, xs + separation)[0],
                display[1],
                sample(o, display, ys, xs - separation)[2],
            ]
        )
    grey = luma(display)[None]
    display = (grey + (display - grey) * float(p["saturation"])).clip(0, 1)
    if mode in ("feedback", "blocks", "chroma"):
        # Codec-like corruption replaces tiles with opaque stale texture. Mixing
        # the entire two pictures would make a dissolve instead of the reference's
        # intact foreground and sharply retained fragments.
        size = max(2, round(max(0.012, float(p["block"])) * short))
        gh, gw = math.ceil(h / size), math.ceil(w / size)
        cells = _mosh_cells(o, f.seed + (f.index // int(p["period"])) * 1741, gh, gw)
        coverage = o.gather(cells, (ys // size).clip(0, gh - 1), (xs // size).clip(0, gw - 1))
        return xp.where(coverage < float(p["amount"]), display, x).clip(0, 1)
    return (x + (display - x) * float(p["amount"])).clip(0, 1)


def analog(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    _, h, w = x.shape
    xp = o.xp
    ys, xs = o.grid(h, w)
    short = min(h, w)
    t = f.index / f.fps * float(p["speed"])
    mode = p["mode"]
    rnd = _noise(o, f.seed + math.floor(t * 24) * 113, (3, h, 1))
    band = xp.sin(ys / h * float(p["bands"]) * math.tau + t * 3)
    jitter = float(p["jitter"]) * short * 0.025
    shift = (band * 0.3 + (rnd[0] - 0.5) * 0.7) * jitter
    if mode == "circuit":
        shift = shift + xp.sin(ys / h * 11 + luma(x) * 8 + t) * short * float(p["jitter"]) * 0.06
    if mode == "cable":
        shift = shift + (xp.abs(band) > 0.96) * (rnd[1] - 0.5) * short * float(p["dropout"]) * 0.3
    roll = float(p["roll"]) * t * h * 0.1
    out = sample(o, x, (ys + roll) % h, xs + shift)
    chroma = float(p["chromatic"]) * short * 0.012
    out = xp.stack([sample(o, out, ys, xs + chroma)[0], out[1], sample(o, out, ys, xs - chroma)[2]])
    bleed = float(p["bleed"])
    if bleed > 0:
        y = luma(out)[None]
        color = out - y
        out = y + color * (1 - bleed) + blur(o, color, max(0.5, bleed * short * 0.007)) * bleed
    if mode == "shutter":
        shutter = (ys / h * int(p["bands"]) + t * 2) % 1 < 0.55
        out = out * (0.25 + 0.75 * shutter)[None]
    elif mode == "chroma":
        phase = xp.sin(ys / h * int(p["bands"]) + t * 4)
        out = xp.stack([out[0] * (0.8 + 0.3 * phase), out[1], out[2] * (0.8 - 0.3 * phase)])
    elif mode == "enders":
        out = sample(o, out, ys, xs + xp.floor(band * 3) * short * 0.012)
        phase = xp.floor((ys / h * int(p["bands"]) + t * 2) % 3)
        out = out * xp.stack([0.6 + 1.2 * (phase == c) for c in range(3)])
    elif mode == "handicam":
        out = blur(o, out, max(0.4, short * 0.0008))
    noise = _noise(o, f.seed + math.floor(t * 30) * 701, (1, h, w)) - 0.5
    out = out + noise * float(p["noise"]) * (0.5 if mode == "cable" else 0.18)
    if mode == "cable":
        colored = _noise(
            o, f.seed + math.floor(t * 12) * 2029, (3, max(2, h // 12), max(2, w // 16))
        )
        out = out + (upscale(o, colored, h, w) - 0.5) * float(p["noise"]) * 0.8
    dropout = (rnd[2] > 1 - float(p["dropout"]) * 0.035) * (0.4 + noise[0] * 0.6)
    out = (out + dropout[None]).clip(0, 1)
    out = crt(o, out, p, f)
    return (x + (out - x) * float(p["amount"])).clip(0, 1)


def _blob_edges(rows: list[dict[str, Any]], mode: str, count: int) -> list[tuple[int, int]]:
    """Unique edges, ordered by persistent IDs rather than detection area."""
    order = sorted(range(len(rows)), key=lambda i: rows[i]["id"])
    if mode == "chain":
        return list(pairwise(order))
    if mode == "mesh":
        return [(a, b) for i, a in enumerate(order) for b in order[i + 1 :]]
    centers = [((r["box"][0] + r["box"][2]) / 2, (r["box"][1] + r["box"][3]) / 2) for r in rows]
    edges = set()
    for a in order:
        cx, cy = centers[a]
        neighbors = sorted(
            (b for b in order if b != a),
            key=lambda b: ((centers[b][0] - cx) ** 2 + (centers[b][1] - cy) ** 2, rows[b]["id"]),
        )
        for b in neighbors[:count]:
            edges.add(tuple(sorted((a, b))))
    return sorted(edges, key=lambda edge: tuple(sorted(rows[i]["id"] for i in edge)))


def _blob_label(row: dict[str, Any], p: dict[str, Any]) -> str:
    cx = (row["box"][0] + row["box"][2]) / 2
    cy = (row["box"][1] + row["box"][3]) / 2
    ident = f"{p.get('label_prefix', 'NODE_')}{row['id']}"
    coordinates = f"x:{cx:.3f} y:{cy:.3f}"
    mode = p.get("label_mode", "id")
    return (
        coordinates
        if mode == "coordinates"
        else f"{ident} {coordinates}"
        if mode == "id_coordinates"
        else ident
    )


@kernel("blob_track")
def blob_track(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    _, h, w = x.shape
    rows = p.get("_blobs", [])
    out = x * float(p["background"])
    overlay = Image.new("RGBA", (w, h))
    draw = ImageDraw.Draw(overlay)
    color = tuple(round(v * 255) for v in hex_rgb(p["color"]))
    line = max(1, round(float(p["thickness"]) * min(h, w)))
    font = ImageFont.load_default(size=max(7, round(min(h, w) * float(p.get("label_size", 0.016)))))
    point = max(1, round(min(h, w) * float(p.get("point_size", 0.003))))
    radius = round(min(h, w) * float(p.get("corner_radius", 0.006)))
    centers = []
    alphas = []
    for row in rows:
        x0, y0, x1, y1 = row["box"]
        centers.append((round((x0 + x1) / 2 * (w - 1)), round((y0 + y1) / 2 * (h - 1))))
        alphas.append(max(0.1, 1 - row["lost"] / max(1, int(p["lifetime"]) + 1)))
    # Connections sit behind labels and nodes; lost tracks fade together with their edges.
    if p["connections"]:
        for ia, ib in _blob_edges(
            rows, p.get("connection_mode", "nearest"), int(p.get("connection_count", 2))
        ):
            opacity = float(p.get("connection_opacity", 0.65)) * min(alphas[ia], alphas[ib])
            draw.line((*centers[ia], *centers[ib]), fill=(*color, round(255 * opacity)), width=line)
    for ri, row in enumerate(rows):
        x0, y0, x1, y1 = row["box"]
        a, b, c, d = (
            round(x0 * (w - 1)),
            round(y0 * (h - 1)),
            round(x1 * (w - 1)),
            round(y1 * (h - 1)),
        )
        alpha = max(0.1, 1 - row["lost"] / max(1, int(p["lifetime"]) + 1))
        ink = (*color, round(255 * alpha))
        cx, cy = centers[ri]
        if p["fill"] != "none" and c > a and d > b:
            patch = x[:, b:d, a:c]
            if p["fill"] == "pixel":
                patch = upscale(
                    o, downscale(o, patch, max(2, min(c - a, d - b) // 8)), d - b, c - a
                )
            elif p["fill"] == "glitch":
                yy, xx = o.grid(d - b, c - a)
                offset = math.sin(f.index * 0.3 + row["id"] + f.seed % 127) * (c - a) * 0.18
                patch = sample(o, patch, yy, xx + offset)
            elif p["fill"] == "tint":
                patch = o.asarray(hex_rgb(p["color"]))[:, None, None] + patch * 0
            else:
                yy, xx = o.grid(d - b, c - a)
                stripe = ((xx + yy) % max(4, 4 * line)) < max(2, 2 * line)
                patch = (
                    patch * 0.4 + o.asarray(hex_rgb(p["color"]))[:, None, None] * stripe[None] * 0.6
                )
            weight = float(p["fill_amount"]) * alpha
            out = out.copy() if isinstance(out, np.ndarray) else out.clone()
            out[:, b:d, a:c] = out[:, b:d, a:c] * (1 - weight) + patch * weight
        if p["style"] == "nodes":
            draw.rounded_rectangle((a, b, c, d), radius=radius, outline=ink, width=line)
            draw.ellipse((cx - point, cy - point, cx + point, cy + point), fill=ink)
        elif p["style"] == "points":
            extent = point * 3
            a, b, c, d = cx - extent, cy - extent, cx + extent, cy + extent
            draw.rectangle((a, b, c, d), outline=ink, width=line)
            draw.ellipse((cx - point, cy - point, cx + point, cy + point), fill=ink)
        elif p["style"] == "box":
            draw.rectangle((a, b, c, d), outline=ink, width=line)
        elif p["style"] == "ellipse":
            draw.ellipse((a, b, c, d), outline=ink, width=line)
        elif p["style"] == "cross":
            draw.line((a, cy, c, cy), fill=ink, width=line)
            draw.line((cx, b, cx, d), fill=ink, width=line)
        else:
            length = max(3, min(c - a, d - b) // 4)
            for xx, yy, sx, sy in ((a, b, 1, 1), (c, b, -1, 1), (a, d, 1, -1), (c, d, -1, -1)):
                draw.line(
                    (xx + sx * length, yy, xx, yy, xx, yy + sy * length), fill=ink, width=line
                )
        if p["labels"]:
            label = _blob_label(row, p)
            bounds = draw.textbbox((0, 0), label, font=font)
            tx = max(0, min(w - (bounds[2] - bounds[0]), a + 2 * line))
            ty = b - font.size - 2 if b >= font.size + 2 else b + 2 * line
            draw.text((tx, max(0, min(h - font.size, ty))), label, fill=ink, font=font)
        if int(p["trails"]) > 0 and len(row["trail"]) > 1:
            draw.line(
                [
                    (round(tx * (w - 1)), round(ty * (h - 1)))
                    for tx, ty in row["trail"][-int(p["trails"]) :]
                ],
                fill=ink,
                width=line,
            )
    rgba = np.asarray(overlay).astype(np.float32) / 255
    alpha = o.asarray(rgba[:, :, 3])[None]
    return (out * (1 - alpha) + o.asarray(rgba[:, :, :3].transpose(2, 0, 1)) * alpha).clip(0, 1)


for _name in ("ghost", "ghost_vertical", "ghost_smear", "ghost_angled", "ghost_depth"):
    kernel(_name)(ghost)
for _name in ("mosh", "mosh_blocks", "mosh_trails", "mosh_liquid", "mosh_chroma"):
    kernel(_name)(mosh)
for _name in (
    "crt_analog",
    "crt_circuit",
    "crt_shutter",
    "crt_chroma",
    "crt_enders",
    "crt_handicam",
    "crt_cable",
):
    kernel(_name)(analog)
