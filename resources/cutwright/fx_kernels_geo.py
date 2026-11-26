"""Frame kernels of the geometric effects (family ``geometrik``): the structure (``fx_geo``) drawn
over the picture. Same rules as ``fx_kernels``: imports nothing from Cutwright, randomness from
numpy (``fx_geo.phase_of``), the result through ``o``. The field and the drawing are numpy (see
``fx_geo``); only the final picture goes back to the backend.

Every kernel here is a ``REGION`` kernel: it gets the entry's scope mask and places itself by it
(``subject``, ``fx_geo.finish``; ``off`` blends by the scope as the processor would), and keeps
faces clear by ``face``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

if __package__:  # in Cutwright's process
    from cutwright.modules.fx.runners import fx_geo as G
    from cutwright.modules.fx.runners.fx_base import (
        KERNELS,
        REGION,
        Frame,
        hex_rgb,
        kernel,
        sample,
    )
else:  # in the objects environment: the files next to the runner, by their bare names
    import fx_geo as G  # type: ignore[no-redef]
    from fx_base import KERNELS, REGION, Frame, hex_rgb, kernel, sample  # type: ignore[no-redef]

LADDER = {"hero": 1.0, "feature": 0.7, "secondary": 0.45, "micro": 0.25}  # symbol size by role
ROLES = tuple(LADDER)
SYMBOLS = ("arc", "bracket", "cross", "diamond", "tick", "ring", "dot")


@kernel("geo_field")
def geo_field(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """The field itself: every node a dot, the anchors ringed, over the dimmed picture (what
    ``cw see fx-field`` shows; as an effect, a technical overlay)."""
    X = G.numpy_rgb(o, x)
    _, h, w = X.shape
    short = min(h, w)
    mk = G.masks(o, p, f, h, w)
    layers, fld = G.geo_layers(o, x, p, f, X, mk)
    color, accent = hex_rgb(p.get("color", "#3a4bff")), hex_rgb(p.get("accent", "#ffffff"))
    canvas = G.Canvas(h, w)
    for layer in layers:
        if layer.weight <= 0 or not len(layer.xy):
            continue
        moved = G.animate(layer, fld, p, f)
        alpha = layer.weight * moved.alpha
        canvas.discs(moved.xy, short * float(p.get("size", 0.006)) * moved.scale, color, alpha)
        rim = layer.anchors
        if len(rim):
            canvas.rings(moved.xy[rim], short * 0.03 * moved.scale[rim], max(1.0, short * 0.002),
                         accent, alpha[rim])  # fmt: skip
    premul, cover = canvas.layer()
    y = G.over(G.material(X, {"dim": 0.35, **p}, mk), premul, cover)
    return o.asarray(G.finish(X, y, p, mk))


# --- constellation -----------------------------------------------------------------------------


def roles(layer: G.Layer) -> Any:
    """Role index per node (``ROLES``): the first two anchors hero, the other anchors feature,
    the best 40 % of the rest secondary, the others micro."""
    n = len(layer.xy)
    out = np.full(n, 3, np.int64)
    rest = np.setdiff1d(np.arange(n), layer.anchors, assume_unique=False)
    out[rest[: math.ceil(0.4 * len(rest))]] = 2  # nodes are best first
    out[layer.anchors[2:]] = 1
    out[layer.anchors[:2]] = 0
    return out


def _build(layer: G.Layer, fld: G.Field, p: dict[str, Any], f: Frame) -> dict[str, Any]:
    """What a layer keeps while it lives (worked out once, when it was built): roles, symbols,
    links and the circle chains' directions."""
    sig = [p.get(k) for k in ("links", "reach", "support", "symbols", "circles", "seed")]
    ex = layer.extra
    if ex.get("sig") == sig:
        return ex
    h, w = fld.size
    short = min(h, w)
    kinds = [s for s in p.get("symbols") or SYMBOLS if s in SYMBOLS] or ["dot"]
    pick = (G.phase_of(layer.keys, f.seed, 11) * len(kinds)).astype(np.int64)
    reach = float(p.get("reach", 0.2)) * short
    pairs = G.links(fld, layer, p.get("links", "delaunay"), reach, float(p.get("support", 0)))
    chains = []
    r0 = float(p.get("size", 0.03)) * short * 1.4
    for hub in layer.anchors[:2] if int(p.get("circles", 0)) > 0 else []:
        chains.append((int(hub), G.emptiest(layer.xy, layer.xy[hub], 6 * r0, fld.size)))
    ex.clear()
    ex.update(sig=sig, roles=roles(layer), symbol=np.asarray(kinds)[pick], pairs=pairs,
              chains=chains)  # fmt: skip
    return ex


def _symbols(canvas: G.Canvas, kind: str, c: Any, r: Any, tan: Any, ph: Any, width: float,
             rgb: Any, alpha: Any) -> None:  # fmt: skip
    """``kind`` drawn at centres ``c`` (n, 2), sizes ``r`` (n,)."""
    if not len(c):
        return
    if kind == "dot":
        canvas.discs(c, 0.35 * r, rgb, alpha)
    elif kind == "ring":
        canvas.rings(c, r, width, rgb, alpha)
    elif kind == "arc":  # an open circle, turned by the node's phase
        a0 = 2 * math.pi * ph
        canvas.rings(c, r, width, rgb, alpha, arc=(a0, a0 + 1.5 * math.pi))
    elif kind == "tick":  # along the edge
        canvas.lines(c - tan * r[:, None] * 0.6, c + tan * r[:, None] * 0.6, width, rgb, alpha)
    elif kind == "cross":
        for u in ((1.0, 0.0), (0.0, 1.0)):
            d = np.asarray(u) * (0.7 * r)[:, None]
            canvas.lines(c - d, c + d, width, rgb, alpha)
    elif kind == "diamond":
        tips = [np.asarray(v) for v in ((1, 0), (0, 1), (-1, 0), (0, -1))]
        for t0, t1 in zip(tips, tips[1:] + tips[:1], strict=True):
            canvas.lines(c + t0 * r[:, None], c + t1 * r[:, None], width, rgb, alpha)
    elif kind == "bracket":  # four corner Ls round the node
        for sx in (-1, 1):
            for sy in (-1, 1):
                corner = c + np.stack([sx * r, sy * r], 1)
                arm = 0.45 * r
                for d in (np.stack([-sx * arm, 0 * r], 1), np.stack([0 * r, -sy * arm], 1)):
                    canvas.lines(corner, corner + d, width, rgb, alpha)


@kernel("constellation")
def constellation(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Symbols on the picture's edge points (sized by role), links between them, circle chains
    from the anchors with their meeting points, an optional frame; moved by the motion block."""
    X = G.numpy_rgb(o, x)
    _, h, w = X.shape
    short = min(h, w)
    mk = G.masks(o, p, f, h, w)
    layers, fld = G.geo_layers(o, x, p, f, X, mk)
    color, accent = hex_rgb(p.get("color", "#f4f1ea")), hex_rgb(p.get("accent", "#3a4bff"))
    width = max(0.35, float(p.get("line_width", 0.0015)) * short)
    size = float(p.get("size", 0.03)) * short
    t = G.clock(p, f)
    speed = float(p.get("speed", 30)) / 100
    grow = G.reveal(p, f)
    dash_px = float(p.get("dash", 0)) * short
    dash = (dash_px, dash_px) if dash_px > 0 else None
    canvas = G.Canvas(h, w)
    for layer in layers:
        if layer.weight <= 0 or not len(layer.xy):
            continue
        ex = _build(layer, fld, p, f)
        mv = G.animate(layer, fld, p, f)
        alpha = layer.weight * mv.alpha
        pairs = ex["pairs"]
        if len(pairs) and grow > 0:
            a, b = mv.xy[pairs[:, 0]], mv.xy[pairs[:, 1]]
            b = a + (b - a) * grow
            seen = np.minimum(mv.alpha[pairs[:, 0]], mv.alpha[pairs[:, 1]])
            la = layer.weight * min(1.0, 0.8 * mv.links) * seen
            flow = t * speed * short * 0.2  # a dash runs along its line
            if p.get("links") == "orthogonal":  # L elbows: across, then down
                elbow = np.stack([b[:, 0], a[:, 1]], 1)
                canvas.lines(a, elbow, width, color, la, dash, flow)
                canvas.lines(elbow, b, width, color, la, dash, flow)
            else:
                canvas.lines(a, b, width, color, la, dash, flow)
            if p.get("runner", False):
                ph = G.phase_of(pairs[:, 0] * 4099 + pairs[:, 1], f.seed, 5)
                k = ((t * (0.2 + 0.8 * speed) + ph) % 1.0)[:, None]
                canvas.discs(a + (b - a) * k, 1.6 * width, accent, la)
        role = ex["roles"]
        r = size * np.asarray([LADDER[ROLES[i]] for i in role]) * mv.scale
        tan = G.tangents(fld, mv.xy)
        ph = G.phase_of(layer.keys, f.seed, 13)
        for kind in SYMBOLS:
            sel = ex["symbol"] == kind
            if not sel.any():
                continue
            hot = sel & (role == 0)
            cool = sel & (role != 0)
            _symbols(canvas, kind, mv.xy[cool], r[cool], tan[cool], ph[cool], width, color,
                     alpha[cool])  # fmt: skip
            _symbols(canvas, kind, mv.xy[hot], r[hot], tan[hot], ph[hot], 1.5 * width, accent,
                     alpha[hot])  # fmt: skip
        for hub, u in ex["chains"]:
            centres, meet = G.circle_chain(mv.xy[hub], u, size * 1.4 * mv.scale[hub],
                                           int(p.get("circles", 0)))  # fmt: skip
            canvas.rings(centres, size * 1.4 * mv.scale[hub] * 0.78 ** np.arange(len(centres)),
                         width, accent, alpha[hub] * grow)  # fmt: skip
            if p.get("intersections", True) and len(meet):
                canvas.discs(meet, 1.8 * width, color, alpha[hub] * grow)
    if p.get("frame", False):
        m = 0.04 * short
        corners = np.array([[m, m], [w - 1 - m, m], [w - 1 - m, h - 1 - m], [m, h - 1 - m]])
        canvas.polyline(corners, width, color, 0.8, closed=True)
    premul, cover = canvas.layer()
    return o.asarray(G.finish(X, G.over(G.material(X, p, mk), premul, cover), p, mk))


# --- contours ----------------------------------------------------------------------------------


def _tints(p: dict[str, Any], X: Any, pts: Any, k: Any, n: int) -> Any:
    """Each line's colour (n_lines, 3) by ``tint``: one colour, a ramp from colour to accent by
    level ``k`` of ``n``, or the picture's own colour where the line runs (lifted)."""
    color, accent = np.asarray(hex_rgb(p.get("color", "#f4f1ea"))), np.asarray(
        hex_rgb(p.get("accent", "#3a4bff")))  # fmt: skip
    mode = p.get("tint", "color")
    if mode == "levels":
        u = (np.asarray(k, np.float64) / max(n - 1, 1))[:, None]
        return color + (accent - color) * u
    if mode == "source" and len(pts):
        _, h, w = X.shape
        ix = np.clip(np.rint(pts[:, 0]).astype(np.int64), 0, w - 1)
        iy = np.clip(np.rint(pts[:, 1]).astype(np.int64), 0, h - 1)
        c = X[:, iy, ix].T
        return 0.35 + 0.65 * c / np.maximum(c.max(1, keepdims=True), 0.05)
    return np.broadcast_to(color, (len(pts), 3))


@kernel("contours")
def contours(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """``topographic``: iso-luma lines (marching squares on the field's luma, smoothed by
    ``detail``), every fourth one heavier; ``flow``: evenly spaced lines along the iso-luma
    tangent from seeds on the camera grid. ``drift`` slides the levels through the tones (or a
    dash along the flow lines), ``pulse`` breathes the width, ``wave`` ripples the lines; bass
    thickens, treble adds detail."""
    X = G.numpy_rgb(o, x)
    _, h, w = X.shape
    short = min(h, w)
    mk = G.masks(o, p, f, h, w)
    fld = G.geo_field(o, x, {**p, "max_nodes": 0}, f, X, *G.node_weights(p, mk))
    snd = G.sound(p)
    t, s = G.clock(p, f), float(p.get("speed", 30)) / 100
    a = float(p.get("amount", 30)) / 100
    mode = p.get("motion", "none")
    width = max(0.35, float(p.get("line_width", 0.0015)) * short) * (1 + 1.5 * snd.bass)
    if mode == "pulse":
        width *= 1 + a * (0.5 + 0.5 * math.sin(2 * math.pi * (0.5 + 2 * s) * t))
    canvas = G.Canvas(h, w)
    q = fld.q
    if p.get("style", "topographic") == "flow":
        spacing = float(p.get("spacing", 0.025)) * short
        seeds, _ = G.camera_lattice(h, w, f.camera, spacing)
        strength = fld.at(fld.mag, seeds[:, 0], seeds[:, 1]) if len(seeds) else np.zeros(0)
        seeds = seeds[np.argsort(-strength, kind="stable")]
        lines = G.streamlines(fld, seeds, spacing)
        dash_px = float(p.get("dash", 0)) * short
        dash = (dash_px, dash_px) if dash_px > 0 else None
        run = -t * (0.05 + 0.5 * s) * short if mode == "drift" else 0.0
        tints = _tints(p, X, np.asarray([ln[len(ln) // 2] for ln in lines]).reshape(-1, 2),
                       np.zeros(len(lines)), 1)  # fmt: skip
        for line, rgb in zip(lines, tints, strict=True):
            if mode == "wave":
                line = line + np.stack([0 * line[:, 0], a * 0.01 * short * np.sin(
                    2 * math.pi * (s * t - line[:, 0] / (0.4 * short)))], 1)  # fmt: skip
            canvas.polyline(line, width, rgb, 1.0, dash=dash, phase=run)
    else:
        detail = min(1.0, float(p.get("detail", 0.5)) + 0.5 * snd.high)
        luma = G.smooth(fld.luma, max(0.0, (1 - detail) * 0.012 * short / q))
        n = int(p.get("levels", 12))
        offset = (t * (0.05 + 0.5 * s) * (0.3 + 0.7 * a)) % 1.0 if mode == "drift" else 0.0
        for k in range(n):
            level = (k + 0.5 + offset) / n
            ga, gb = G.iso_segments(luma, level)
            if not len(ga):
                continue
            fa, fb = (ga + 0.5) * q - 0.5, (gb + 0.5) * q - 0.5
            if mode == "wave":
                for pts in (fa, fb):
                    pts[:, 1] += a * 0.01 * short * np.sin(2 * math.pi * (s * t - pts[:, 0] / (
                        0.4 * short)))  # fmt: skip
            heavy = p.get("index", True) and k % 4 == 3
            rgb = _tints(p, X, (fa + fb) / 2, np.full(len(fa), k), n)
            canvas.lines(fa, fb, width * (1.8 if heavy else 1.0), rgb, 1.0)
    premul, cover = canvas.layer()
    return o.asarray(G.finish(X, G.over(G.material(X, p, mk), premul, cover), p, mk))


# --- cells: pixel memory, horizontal echo ------------------------------------------------------


def cell_signal(X: Any, s: int, edge: float) -> tuple[Any, Any]:
    """Cell means (3, hh, ww) of ``s`` px cells and their signal: in a dark frame a bright cell,
    in a bright frame a dark one (the cell's luma against the frame's mean), plus ``edge`` × the
    cells' Sobel."""
    _, h, w = X.shape
    hh, ww = max(1, h // s), max(1, w // s)
    small = X[:, : hh * s, : ww * s].reshape(3, hh, s, ww, s).mean((2, 4))
    y = G.LUMA[0] * small[0] + G.LUMA[1] * small[1] + G.LUMA[2] * small[2]
    mean = float(y.mean())
    sig = y - mean if mean < 0.5 else mean - y
    gx, gy = G.sobel(y)
    return small, sig + edge * np.hypot(gx, gy) * 2


def blocky(a: Any, s: int, h: int, w: int) -> Any:
    """Cells (c, hh, ww) back to h × w, each a flat block (the last cell runs on to the edge)."""
    iy = np.minimum(np.arange(h) // s, a.shape[1] - 1)
    ix = np.minimum(np.arange(w) // s, a.shape[2] - 1)
    return a[:, iy][:, :, ix]


def _step(p: dict[str, Any], f: Frame) -> int:
    """The motion's tick (a new draw of the random flips, glyphs, jitter): 1–12 a second."""
    if p.get("motion", "none") == "none":
        return 0
    return math.floor(G.clock(p, f) * (1 + 11 * float(p.get("speed", 30)) / 100))


def _glyph_cells(canvas: G.Canvas, iy: Any, ix: Any, chars: Any, s: int, rgb: Any,
                 alpha: Any) -> None:  # fmt: skip
    """Glyph ``chars[i]`` (a GLYPHS key) centred in cell (iy[i], ix[i])."""
    px = max(0.5, s / 7)
    left, top = (s - 3 * px) / 2, (s - 5 * px) / 2
    rgb = np.broadcast_to(np.asarray(rgb, np.float64).reshape(-1, 3), (len(iy), 3))
    alpha = np.broadcast_to(np.asarray(alpha, np.float64), (len(iy),))
    for ch in np.unique(chars):
        sel = chars == ch
        for k, bit in enumerate(G.GLYPHS[str(ch)]):
            if bit != "1":
                continue
            a = np.stack([ix[sel] * s + left + (k % 3) * px, iy[sel] * s + top + (k // 3) * px], 1)
            canvas.rects(a, a + px, rgb[sel], alpha[sel])


def _windows(img: Any, X: Any, layer: G.Layer, fld: G.Field, p: dict[str, Any], f: Frame,
             canvas: G.Canvas) -> None:  # fmt: skip
    """Memory windows on ``img`` (in place): a zoomed crop round each of the layer's first
    ``windows`` anchors, set off toward the emptiest side, pasted with the layer's weight; its
    frame, a label and a guide line to the anchor go on ``canvas``."""
    _, h, w = img.shape
    short = min(h, w)
    count, zoom = int(p.get("windows", 0)), float(p.get("zoom", 2.5))
    mv = G.animate(layer, fld, p, f)
    color, accent = hex_rgb(p.get("color", "#f4f1ea")), hex_rgb(p.get("accent", "#3a4bff"))
    ww, wh = 0.26 * short, 0.17 * short
    line = max(0.6, float(p.get("line_width", 0.0015)) * short)
    for n, hub in enumerate(layer.anchors[:count]):
        at = mv.xy[hub]
        u = G.emptiest(mv.xy, at, 0.35 * short, (h, w))
        cx = float(np.clip(at[0] + u[0] * 0.24 * short, ww / 2 + 2, w - ww / 2 - 3))
        cy = float(np.clip(at[1] + u[1] * 0.24 * short, wh / 2 + 2, h - wh / 2 - 3))
        x0, y0 = int(cx - ww / 2), int(cy - wh / 2)
        x1, y1 = x0 + int(ww), y0 + int(wh)
        sx = np.clip(np.rint(at[0] + (np.arange(x0, x1) - cx) / zoom).astype(np.int64), 0, w - 1)
        sy = np.clip(np.rint(at[1] + (np.arange(y0, y1) - cy) / zoom).astype(np.int64), 0, h - 1)
        crop = X[:, sy][:, :, sx]
        k = layer.weight * float(mv.alpha[hub])
        img[:, y0:y1, x0:x1] += (crop - img[:, y0:y1, x0:x1]) * k
        box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64) - 0.5
        canvas.polyline(box, line, color, k, closed=True)
        corner = box[np.argmin(np.hypot(*(box - at).T))]
        canvas.lines([at], [corner], line, accent, k)
        canvas.rings([at], 0.012 * short, line, accent, k)
        px = max(1.0, 0.0055 * short)
        canvas.rects([[x0 - 0.5, y0 - 7 * px]], [[x0 - 0.5 + 34 * px, y0 - 0.5]], accent, k)
        label = f"0X{int(layer.keys[hub]) % 4096:03X} {n + 1}"
        canvas.text(label, (x0 + px, y0 - 6 * px), px, (1.0, 1.0, 1.0), k)


@kernel("pixel_memory")
def pixel_memory(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Pixel memory: the frame as cells, lit where the cell stands out (``cell_signal`` over
    ``threshold``), drawn as solid cells (``archive``) or glyphs (``glyphs``); ``entropy`` flips
    cells at random on each motion tick; ``windows`` zoomed crops round the anchors, labelled."""
    X = G.numpy_rgb(o, x)
    _, h, w = X.shape
    short = min(h, w)
    mk = G.masks(o, p, f, h, w)
    s = max(2, round(float(p.get("cell", 0.02)) * short))
    _, sig = cell_signal(X, s, float(p.get("edges", 0.5)))
    hh, ww = sig.shape
    snd = G.sound(p)
    thr = max(0.0, float(p.get("signal", 0.12)) - 0.1 * snd.beat)
    step = _step(p, f)
    keys = np.arange(hh * ww).reshape(hh, ww)
    active = sig > thr
    entropy = float(p.get("entropy", 0.15)) + 0.3 * snd.high
    if entropy > 0:
        active ^= G.phase_of(keys * 7 + step * 1000003, f.seed, 21) < entropy * 0.25
    color, accent = hex_rgb(p.get("color", "#f4f1ea")), hex_rgb(p.get("accent", "#3a4bff"))
    iy, ix = np.nonzero(active)
    strong = sig[iy, ix] > thr + 0.6 * (1 - thr)
    rgb = np.where(strong[:, None], np.asarray(accent), np.asarray(color))
    cells = G.Canvas(h, w)
    if p.get("style", "archive") == "glyphs":
        pick = (G.phase_of(keys[iy, ix] + step * 7919, f.seed, 23) * 16).astype(np.int64)
        _glyph_cells(cells, iy, ix, np.asarray(list("0123456789ABCDEF"))[pick], s, rgb, 1.0)
    else:
        g = 0.07 * s
        a = np.stack([ix * s + g, iy * s + g], 1)
        cells.rects(a, a + s - 2 * g, rgb, 1.0)
    premul, cover = cells.layer()
    img = G.over(G.material(X, {"dim": 0.6, **p}, mk), premul, cover)
    if int(p.get("windows", 0)) > 0:
        layers, fld = G.geo_layers(o, x, p, f, X, mk)
        marks = G.Canvas(h, w)
        for layer in layers:
            if layer.weight > 0:
                _windows(img, X, layer, fld, p, f, marks)
        premul, cover = marks.layer()
        img = G.over(img, premul, cover)
    return o.asarray(G.finish(X, img, p, mk))


def trail(val: Any, col: Any, fb: Any) -> tuple[Any, Any]:
    """Along each row (to the right), the decayed running maximum of ``val`` (hh, ww) —
    max over k of val[j − k] · fb^k, ``fb`` per row — and the colour of the cell it came from:
    log₂(ww) doubling steps (max-plus with a linear cost composes exactly)."""
    val, col = val.copy(), col.copy()
    span = 1
    _, ww = val.shape
    while span < ww:
        cand = np.zeros_like(val)
        cand[:, span:] = val[:, :-span] * (fb[:, None] ** span)
        moved = np.zeros_like(col)
        moved[:, :, span:] = col[:, :, :-span]
        take = cand > val
        val = np.where(take, cand, val)
        col = np.where(take[None], moved, col)
        span *= 2
    return val, col


@kernel("horizontal_echo")
def horizontal_echo(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Horizontal echo (in space, not time): cells that stand out stretch sideways in their own
    colour, fading by ``feedback`` per cell; ``steps`` bands the fade, ``jitter`` shifts rows on
    each motion tick, ``strips`` gives some rows a long drag. ``pulse`` breathes the length,
    ``glitch`` flips rows' direction; bass lengthens."""
    X = G.numpy_rgb(o, x)
    _, h, w = X.shape
    short = min(h, w)
    mk = G.masks(o, p, f, h, w)
    s = max(2, round(float(p.get("cell", 0.012)) * short))
    small, sig = cell_signal(X, s, float(p.get("edges", 0.3)))
    hh = sig.shape[0]
    thr = float(p.get("threshold", 0.15))
    val = ((sig - thr) / max(1e-6, 1 - thr)).clip(0, 1)
    snd = G.sound(p)
    t, sp, a = G.clock(p, f), float(p.get("speed", 30)) / 100, float(p.get("amount", 30)) / 100
    mode = p.get("motion", "none")
    step = _step(p, f)
    length = 1 + 1.5 * snd.bass
    if mode == "pulse":
        length *= 1 + a * 0.8 * math.sin(2 * math.pi * (0.3 + 1.5 * sp) * t)
    fb = np.full(hh, float(p.get("feedback", 0.92)) ** (1 / max(length, 0.05)))
    rows = np.arange(hh)
    drag = G.phase_of(rows, f.seed, 31) < float(p.get("strips", 0.15)) * 0.5
    fb = np.where(drag, np.maximum(fb, 0.985), fb)
    direction = p.get("direction", "right")
    flip = (G.phase_of(rows * 13 + step * 7919, f.seed, 33) < a * 0.5) if mode == "glitch" else (
        np.zeros(hh, bool))  # fmt: skip
    out_v, out_c = np.zeros_like(val), np.zeros_like(small)
    for way in ("right", "left") if direction == "both" else (direction,):
        go_left = np.full(hh, way == "left") ^ flip
        v = np.where(go_left[:, None], val[:, ::-1], val)
        c = np.where(go_left[None, :, None], small[:, :, ::-1], small)
        tv, tc = trail(v, c, fb)
        tv = np.where(go_left[:, None], tv[:, ::-1], tv)
        tc = np.where(go_left[None, :, None], tc[:, :, ::-1], tc)
        take = tv > out_v
        out_v, out_c = np.where(take, tv, out_v), np.where(take[None], tc, out_c)
    steps = int(p.get("steps", 0))
    if steps > 1:
        out_v = np.ceil(out_v * steps) / steps
    jitter = float(p.get("jitter", 0.0))
    if jitter > 0:
        shift = np.rint((G.phase_of(rows * 31 + step * 104729, f.seed, 35) - 0.5) * 2 * jitter
                        * 4).astype(np.int64)  # fmt: skip
        for r in np.flatnonzero(shift):
            out_v[r], out_c[:, r] = np.roll(out_v[r], shift[r]), np.roll(out_c[:, r], shift[r], 1)
    base = G.material(X, p, mk)
    alpha = blocky(out_v[None], s, h, w)
    color = blocky(out_c, s, h, w)
    return o.asarray(G.finish(X, base + (color - base) * alpha, p, mk))


# --- grid system -------------------------------------------------------------------------------


def bsp(w: int, h: int, depth: int, focus: tuple[float, float], avoid: float, seed: int,
        min_side: float) -> list[tuple[float, float, float, float]]:  # fmt: skip
    """The frame split ``depth`` times (each cell along its longer side, at 30–70 % by the seed),
    never through the band ``avoid`` px round the focus point (the cut moves out of it, or the
    cell stays whole): rectangles (x0, y0, x1, y1)."""
    cells = [(0.0, 0.0, float(w), float(h))]
    fx, fy = focus
    for level in range(depth):
        nxt = []
        for i, (x0, y0, x1, y1) in enumerate(cells):
            wide = (x1 - x0) >= (y1 - y0)
            lo, hi = (x0, x1) if wide else (y0, y1)
            if hi - lo < 2 * min_side:
                nxt.append((x0, y0, x1, y1))
                continue
            r = 0.3 + 0.4 * float(G.phase_of(np.array([level * 1000 + i]), seed, 41)[0])
            cut = lo + (hi - lo) * r
            f_at, inside = (fx, y0 <= fy <= y1) if wide else (fy, x0 <= fx <= x1)
            if inside and abs(cut - f_at) < avoid:
                room = (lo + min_side, hi - min_side)
                options = [c for c in (f_at - avoid, f_at + avoid) if room[0] <= c <= room[1]]
                if not options:
                    nxt.append((x0, y0, x1, y1))
                    continue
                cut = min(options, key=lambda c: abs(c - cut))
            nxt += ([(x0, y0, cut, y1), (cut, y0, x1, y1)] if wide
                    else [(x0, y0, x1, cut), (x0, cut, x1, y1)])  # fmt: skip
        cells = nxt
    return cells


GRID_MIX = {  # cell kinds and their shares by mode: source, collage, tone, empty, frame, bar
    "editorial": (0.5, 0.0, 0.2, 0.0, 0.3, 0.0),
    "fragment": (0.0, 0.6, 0.2, 0.1, 0.1, 0.0),
    "data": (0.25, 0.0, 0.15, 0.1, 0.0, 0.5),
}
GRID_DEPTH = {"editorial": 3, "fragment": 5, "data": 4}


@kernel("grid_system")
def grid_system(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Grid system: ``measure`` lays a column/row grid with ticks; ``editorial`` / ``fragment``
    / ``data`` split the frame (BSP, around the focus point) into cells that keep the source,
    show a moved crop of it (collage), go flat to their mean tone, empty, framed or a bar of
    their brightness. ``drift`` slides the collage crops, ``glitch`` reshuffles the cells on
    each tick, ``scan`` sweeps a highlighted column."""
    X = G.numpy_rgb(o, x)
    _, h, w = X.shape
    short = min(h, w)
    mk = G.masks(o, p, f, h, w)
    color, accent = hex_rgb(p.get("color", "#f4f1ea")), hex_rgb(p.get("accent", "#3a4bff"))
    paper = np.asarray(hex_rgb(p.get("paper", "#e9e4d8")), np.float32)[:, None, None]
    line = max(0.5, float(p.get("line_width", 0.0015)) * short)
    t, sp, a = G.clock(p, f), float(p.get("speed", 30)) / 100, float(p.get("amount", 30)) / 100
    mode, motion = p.get("mode", "editorial"), p.get("motion", "none")
    img = G.material(X, p, mk).copy()
    canvas = G.Canvas(h, w)
    if mode == "measure":
        cols = int(p.get("columns", 12))
        rows = int(p.get("rows", 0)) or max(1, round(cols * h / w))
        xs, ys = np.arange(1, cols) * w / cols, np.arange(1, rows) * h / rows
        canvas.lines(np.stack([xs, 0 * xs], 1), np.stack([xs, 0 * xs + h - 1], 1), line, color, 0.6)
        canvas.lines(np.stack([0 * ys, ys], 1), np.stack([0 * ys + w - 1, ys], 1), line, color, 0.6)
        px = max(1.0, 0.005 * short)
        for i, gx in enumerate(xs):
            canvas.text(str(i + 1), (gx + 2 * px, 2 * px), px, color, 0.9)
        if motion == "scan":
            k = math.floor((t * (0.5 + 3 * sp)) % cols)
            canvas.rects([[k * w / cols, 0]], [[(k + 1) * w / cols, h]], accent, 0.15 + 0.3 * a)
    else:
        depth = int(p.get("depth", 0)) or GRID_DEPTH.get(mode, 4)
        focus = p.get("focus") or [0.5, 0.45]
        avoid = float(p.get("avoid", 0.12)) * short
        cells = bsp(w, h, depth, (focus[0] * w, focus[1] * h), avoid, f.seed, 0.06 * short)
        mix = np.cumsum(GRID_MIX.get(mode, GRID_MIX["editorial"]))
        tick = _step(p, f) if motion == "glitch" else 0
        gutter = float(p.get("gutter", 0.006)) * short / 2
        for i, (x0, y0, x1, y1) in enumerate(cells):
            r = float(G.phase_of(np.array([i * 31 + tick * 7919]), f.seed, 43)[0])
            kind = int(np.searchsorted(mix, r * mix[-1], side="right"))
            ix0, iy0 = int(x0 + gutter), int(y0 + gutter)
            ix1, iy1 = int(x1 - gutter), int(y1 - gutter)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            region = (slice(None), slice(iy0, iy1), slice(ix0, ix1))
            if kind == 1:  # collage: a crop from elsewhere (drifting)
                ph = float(G.phase_of(np.array([i]), f.seed, 45)[0])
                dx = (ph - 0.5) * 0.4 * short + (a * 0.08 * short * math.sin(2 * math.pi * (
                    sp * t + ph)) if motion == "drift" else 0.0)  # fmt: skip
                dy = (float(G.phase_of(np.array([i]), f.seed, 47)[0]) - 0.5) * 0.4 * short
                sy = np.clip(np.arange(iy0, iy1) + int(dy), 0, h - 1)
                sx = np.clip(np.arange(ix0, ix1) + int(dx), 0, w - 1)
                img[region] = X[:, sy][:, :, sx]
            elif kind == 2:  # flat tone
                img[region] = X[region].mean((1, 2), keepdims=True)
            elif kind == 3:  # empty
                img[region] = paper
            elif kind == 5:  # a bar of the cell's brightness, from the bottom
                y = G.LUMA[0] * X[0, iy0:iy1, ix0:ix1] + G.LUMA[1] * X[1, iy0:iy1, ix0:ix1] + (
                    G.LUMA[2] * X[2, iy0:iy1, ix0:ix1])  # fmt: skip
                top = y1 - (y1 - y0) * float(y.mean())
                img[region] = paper * 0.15
                canvas.rects([[x0 + 2 * gutter, top]], [[x1 - 2 * gutter, y1 - gutter]], accent,
                             0.9)  # fmt: skip
            outline = np.array([[ix0, iy0], [ix1, iy0], [ix1, iy1], [ix0, iy1]], np.float64) - 0.5
            canvas.polyline(outline, line * (1.6 if kind == 4 else 1.0), color,
                            0.9 if kind == 4 else 0.5, closed=True)  # fmt: skip
    premul, cover = canvas.layer()
    return o.asarray(G.finish(X, G.over(img, premul, cover), p, mk))


# --- curtain -----------------------------------------------------------------------------------


@kernel("curtain")
def curtain(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Spectral curtain: in every few columns the strongest horizontal ridge (largest vertical
    Sobel), its colour drawn out down, up or both ways as a fading streak, screened over.
    ``pulse`` breathes the length, ``wave`` sways the streaks; bass lengthens, treble
    flickers them."""
    X = G.numpy_rgb(o, x)
    _, h, w = X.shape
    short = min(h, w)
    mk = G.masks(o, p, f, h, w)
    fld = G.geo_field(o, x, {**p, "max_nodes": 0}, f, X, *G.node_weights(p, mk))
    q = fld.q
    snd = G.sound(p)
    t, sp, a = G.clock(p, f), float(p.get("speed", 30)) / 100, float(p.get("amount", 30)) / 100
    mode = p.get("motion", "none")
    length = float(p.get("length", 0.35)) * h * (1 + 0.8 * snd.bass)
    if mode == "pulse":
        length *= 1 + a * 0.6 * math.sin(2 * math.pi * (0.3 + 1.5 * sp) * t)
    gap = max(1, round((1.05 - float(p.get("density", 0.5))) * 0.03 * short))  # px
    stripe = max(1, round(gap * float(p.get("width", 0.8))))
    xs = np.arange(gap // 2, w, gap)
    fx = np.clip(((xs + 0.5) / q - 0.5).round().astype(np.int64), 0, fld.gy.shape[1] - 1)
    gy = np.abs(fld.gy[:, fx])
    row = np.argmax(gy, 0)
    keep = gy[row, np.arange(len(fx))] > fld.threshold
    xs, ry = xs[keep], ((row[keep] + 0.5) * q - 0.5).clip(0, h - 1)
    lum = G.LUMA[0] * X[0] + G.LUMA[1] * X[1] + G.LUMA[2] * X[2]
    above = np.clip(np.rint(ry - 1.5 * q).astype(np.int64), 0, h - 1)
    under = np.clip(np.rint(ry + 1.5 * q).astype(np.int64), 0, h - 1)
    lit = np.where(lum[above, xs] >= lum[under, xs], above, under)  # the ridge's light side
    colors = X[:, lit, xs]
    inten = float(p.get("intensity", 0.8)) * (1 - 0.35 * snd.high * G.phase_of(
        xs + _step(p, f) * 7919, f.seed, 51))  # fmt: skip
    ys = np.arange(h, dtype=np.float64)[:, None]
    d = ys - ry[None]
    way = p.get("direction", "down")
    d = d if way == "down" else -d if way == "up" else np.abs(d)
    fade = (1 - d / max(length, 1.0)).clip(0, None) ** 1.5
    prof = np.where((d >= 0) & (d <= length), fade, 0.0) * inten
    sway = (a * 0.02 * short * np.sin(2 * math.pi * (sp * t) + xs * 0.05)) if mode == "wave" else (
        np.zeros(len(xs)))  # fmt: skip
    streak = np.zeros((3, h, w), np.float32)
    for i, x0 in enumerate(xs):
        c0 = int(np.clip(x0 - stripe // 2 + round(float(sway[i])), 0, w - 1))
        c1 = min(w, c0 + stripe)
        column = (prof[:, i, None] * colors[:, i, None, None]).astype(np.float32)  # (3, h, 1)
        streak[:, :, c0:c1] = np.maximum(streak[:, :, c0:c1], column)
    base = G.material(X, p, mk)
    return o.asarray(G.finish(X, 1 - (1 - base) * (1 - streak), p, mk))


# --- portrait window ---------------------------------------------------------------------------


@kernel("portrait_window")
def portrait_window(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """A window that follows the main face (``_faces``; the ``center`` when there is none):
    outside blurred and dimmed, inside the source or the ``halftone`` / ``engrave`` kernels;
    a thin frame, corner ticks and a label. ``pulse`` breathes the window."""
    X = G.numpy_rgb(o, x)
    _, h, w = X.shape
    short = min(h, w)
    mk = G.masks(o, {**p, "face": "off"}, f, h, w)
    faces = p.get("_faces") or []
    pad = float(p.get("pad", 0.8))
    if faces:
        cx, cy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        bw, bh = fw * w * (1 + 2 * pad), fh * h * (1 + 2 * pad)
        cx, cy = cx * w, cy * h
    else:
        c = p.get("center") or [0.5, 0.42]
        cx, cy, bw, bh = c[0] * w, c[1] * h, 0.38 * short, 0.46 * short
    t, sp, a = G.clock(p, f), float(p.get("speed", 30)) / 100, float(p.get("amount", 30)) / 100
    if p.get("motion", "none") == "pulse":
        k = 1 + a * 0.08 * math.sin(2 * math.pi * (0.3 + 1.5 * sp) * t)
        bw, bh = bw * k, bh * k
    x0, x1 = int(max(0, cx - bw / 2)), int(min(w, cx + bw / 2))
    y0, y1 = int(max(0, cy - bh / 2)), int(min(h, cy + bh / 2))
    outside = G.smooth_rgb(X, float(p.get("blur", 0.02)) * short) * (1 - float(p.get("dim", 0.3)))
    inside_kind, look = p.get("inside", "source"), p.get("_inside")
    if inside_kind in ("halftone", "engrave") and inside_kind in KERNELS and look:
        inner = np.asarray(o.numpy(KERNELS[inside_kind](o, x, look, f)), np.float32)
    else:
        inner = X
    img = outside.copy()
    if x1 > x0 and y1 > y0:
        img[:, y0:y1, x0:x1] = inner[:, y0:y1, x0:x1]
    canvas = G.Canvas(h, w)
    color = hex_rgb(p.get("color", "#f4f1ea"))
    line = max(0.6, float(p.get("line_width", 0.0015)) * short)
    box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64) - 0.5
    canvas.polyline(box, line, color, 0.9, closed=True)
    tick = 0.03 * short
    for (bx, by), (sx, sy) in zip(box, ((-1, -1), (1, -1), (1, 1), (-1, 1)), strict=True):
        canvas.lines([[bx, by], [bx, by]], [[bx + sx * tick, by], [bx, by + sy * tick]],
                     2 * line, color, 1.0)  # fmt: skip
    px = max(1.0, 0.005 * short)
    canvas.text(f"{int(x0):04d}:{int(y0):04d}", (x0, y1 + 2 * px), px, color, 0.9)
    premul, cover = canvas.layer()
    return o.asarray(G.finish(X, G.over(img, premul, cover), p, mk))


# --- corner pin --------------------------------------------------------------------------------


def homography(src: list[tuple[float, float]], dst: list[tuple[float, float]]) -> Any:
    """The 3 × 3 projective map taking the four points ``src`` to ``dst`` (h33 = 1)."""
    rows, rhs = [], []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs += [u, v]
    h = np.linalg.solve(np.asarray(rows, np.float64), np.asarray(rhs, np.float64))
    return np.append(h, 1.0).reshape(3, 3)


@kernel("corner_pin")
def corner_pin(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Corner pin (screen replacement): the clip's whole picture laid into the quad ``corners``
    (top left, top right, bottom right, bottom left; fractions of the timeline's frame
    ``_frame``) by a perspective map, sampled bilinearly from the output side. Outside the quad
    the clip is transparent: its alpha goes to the processor (``f.state["alpha"]``), so on an
    upper track the picture below shows. Worked in the clip's own pixels, the timeline's fit
    (scale, centred) undone first, so the quad lands where it was drawn on the timeline."""
    _, h, w = x.shape
    W, H = (float(v) for v in (p.get("_frame") or (w, h)))
    k = min(W / w, H / h)
    ox, oy = (W - w * k) / 2, (H - h * k) / 2
    c = [float(v) for v in p["corners"]]
    quad = [((c[2 * i] * W - ox) / k, (c[2 * i + 1] * H - oy) / k) for i in range(4)]
    try:  # python floats: they multiply numpy and torch arrays alike
        m = homography(quad, [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)),
                              (0.0, float(h))]).tolist()  # fmt: skip
    except np.linalg.LinAlgError:  # a degenerate quad (three corners on a line): nothing shows
        f.state["alpha"] = x[0] * 0
        return x * 0
    ys, xs = o.grid(h, w)
    px, py = xs + 0.5, ys + 0.5  # pixel centres, continuous coordinates
    den = m[2][0] * px + m[2][1] * py + m[2][2]
    safe = o.xp.where(den > 1e-9, den, 1.0)
    u = (m[0][0] * px + m[0][1] * py + m[0][2]) / safe - 0.5
    v = (m[1][0] * px + m[1][1] * py + m[1][2]) / safe - 0.5
    picture = sample(o, x, v, u, "zero")
    cover = sample(o, x[:1] * 0 + 1, v, u, "zero")[0] * (den > 1e-9)  # soft at the quad's edges
    f.state["alpha"] = cover
    straight = o.xp.where(cover[None] > 1e-4, picture / cover.clip(1e-4, None)[None], 0.0)
    return straight.clip(0, 1)


REGION.update({"geo_field", "constellation", "contours", "pixel_memory", "horizontal_echo",
               "grid_system", "curtain", "portrait_window", "corner_pin"})  # fmt: skip
