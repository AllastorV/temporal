"""Frame kernels of the portrait and light effects: skin refinement (``portre``) and a virtual
light (``relight``). Same rules as
``fx_kernels``: imports nothing from Cutwright but the leaf helpers (``fx_base``, ``fx_geo``),
numpy or torch through ``o``.

``face_refine`` (Resolve's Face Refinement, a beauty pass): an edge-preserving smoothing — the
self-guided filter, each channel its own guide, so a contour (eyes, lips, jaw) keeps its edge and
a pore or a blemish under ``eps`` goes — put back over the frame only where the picture is skin
(a soft ellipse in Cb/Cr around skin tones, light to dark) and, with ``area: faces``, inside the
faces analysis' ellipses. ``texture`` adds the finest detail back so skin never turns plastic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

if __package__:  # in Cutwright's process
    from cutwright.modules.fx.runners import fx_geo as G
    from cutwright.modules.fx.runners.fx_base import (
        REGION,
        Frame,
        blur,
        downscale,
        hex_rgb,
        kernel,
        luma,
        shifted,
        upscale,
    )
else:  # in the objects environment: the files next to the runner, by their bare names
    import fx_geo as G  # type: ignore[no-redef]
    from fx_base import (  # type: ignore[no-redef]
        REGION,
        Frame,
        blur,
        downscale,
        hex_rgb,
        kernel,
        luma,
        shifted,
        upscale,
    )

# Skin in BT.709 Cb/Cr (0–1, 0.5 grey): light and dark skin both sit near this centre.
SKIN_CB, SKIN_CR, SKIN_RB, SKIN_RR = 0.43, 0.59, 0.065, 0.055


def skin_mask(o: Any, x: Any) -> Any:
    """0–1 per pixel: how skin-like the colour is (not too dark, not clipped)."""
    y = luma(x)
    cb = (x[2] - y) * 0.5389 + 0.5
    cr = (x[0] - y) * 0.6350 + 0.5
    d = ((cb - SKIN_CB) / SKIN_RB) ** 2 + ((cr - SKIN_CR) / SKIN_RR) ** 2
    lit = ((y - 0.06) / 0.08).clip(0, 1) * ((0.97 - y) / 0.05).clip(0, 1)
    return (1.4 - d).clip(0, 1) * lit


def guided(o: Any, c: Any, sigma: float, eps: float) -> Any:
    """The self-guided filter of one channel (h, w): flat where the local variance is under
    ``eps``, the channel itself across a strong edge."""
    m = blur(o, c[None], sigma)[0]
    v = blur(o, (c * c)[None], sigma)[0] - m * m
    a = v / (v + eps)
    b = m - a * m
    return blur(o, a[None], sigma)[0] * c + blur(o, b[None], sigma)[0]


@kernel("face_refine")
def face_refine(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    _, h, w = x.shape
    short = min(h, w)
    amount = float(p.get("amount", 0.5))
    sigma = max(1.0, float(p.get("smoothing", 0.006)) * short)
    eps = (0.015 + 0.05 * amount) ** 2  # how large a blemish may be and still go
    smooth = o.xp.stack([guided(o, x[c], sigma, eps) for c in range(3)])
    fine = x - blur(o, x, max(0.6, 0.0012 * short))
    smooth = smooth + float(p.get("texture", 0.35)) * fine
    m = blur(o, skin_mask(o, x)[None], max(1.0, 0.004 * short))[0]
    if p.get("area", "faces") == "faces" and p.get("_faces"):
        m = m * o.asarray(G.face_mask({**p, "face": "reduce"}, h, w))
    if f.mask is not None:  # a scope mask (person, a drawn one): only inside it
        m = m * f.mask if p.get("_scope", "inside") != "outside" else m * (1 - f.mask)
    return (x + (smooth - x) * (m * amount)[None]).clip(0, 1)


REGION.add("face_refine")  # it places itself by the scope mask (with the skin mask)


def push_pull(o: Any, x: Any, valid: Any) -> Any:
    """Holes (``valid`` 0) filled from around them: area averages down a pyramid weighted by
    what is valid, then up again, each level's holes taking the coarser level's picture."""
    levels = []
    cur, ok = x, valid
    while min(cur.shape[-2:]) > 8:
        levels.append((cur, ok))
        weight = downscale(o, ok[None], 2)[0]
        cur = downscale(o, cur * ok[None], 2) / weight.clip(1e-6, None)[None]
        ok = (weight > 1e-3) * 1.0
    fill = cur
    for picture, have in reversed(levels):
        up = upscale(o, fill, *picture.shape[-2:])
        fill = picture * have[None] + up * (1 - have[None])
    return fill


_LAMA: dict[str, Any] = {}  # the loaded TorchScript model by path: once per worker
LAMA_SIDE = 1024  # the hole's box is filled at most this large on its long side


def lama_fill(o: Any, x: Any, hole: Any, path: str) -> Any:
    """LaMa (big-lama TorchScript: image and 0/1 mask in, 0–1 image out) filling ``hole``
    (h, w; 1 = to fill) in its bounding box plus a margin, sized to a multiple of 8; None off
    torch or with no hole."""
    if getattr(o.xp, "__name__", "") != "torch":
        return None
    torch = o.xp
    inside = np.asarray(o.numpy(hole)) > 0.5
    rows, cols = np.nonzero(inside.any(1))[0], np.nonzero(inside.any(0))[0]
    if not len(rows):
        return None
    model = _LAMA.get(path)
    if model is None:
        model = _LAMA[path] = torch.jit.load(path, map_location=x.device).eval()
    h, w = inside.shape
    m = max(16, int(rows[-1] - rows[0] + cols[-1] - cols[0]) // 2)  # context around the hole
    y0, y1 = max(0, rows[0] - m), min(h, rows[-1] + m + 1)
    x0, x1 = max(0, cols[0] - m), min(w, cols[-1] + m + 1)
    ch, cw = y1 - y0, x1 - x0
    scale = min(1.0, LAMA_SIDE / max(ch, cw))
    size = (max(8, int(ch * scale) // 8 * 8), max(8, int(cw * scale) // 8 * 8))
    resize = torch.nn.functional.interpolate
    with torch.no_grad():
        img = resize(x[None, :, y0:y1, x0:x1], size=size, mode="bilinear", align_corners=False)
        cut = resize(hole[None, None, y0:y1, x0:x1], size=size, mode="bilinear",
                     align_corners=False)  # fmt: skip
        out = model(img, (cut > 0.01).float())
        out = resize(out, size=(ch, cw), mode="bilinear", align_corners=False)[0]
    fill = x.clone()
    fill[:, y0:y1, x0:x1] = out.clamp(0, 1)
    return fill


@kernel("object_remove")
def object_remove(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """Object removal: the scope mask's area (the object: ``mask.object`` or a drawn mask)
    refilled. ``auto``: from a clean plate — the background seen there in earlier frames,
    gathered while the mask was off it (a locked-off shot: a passer-by, a cable) — and where it
    was never seen, from around (``push_pull``); ``model``: the same, but what was never seen
    from LaMa (``_lama``, torch only; else from around); ``spatial``: from around only. Its
    ``memory`` frames are its history, so a still equals the render. Shortcut: LaMa fills each
    frame alone (averaged over frames against flicker); a moving camera needs a video
    inpainting model (ProPainter)."""
    if f.mask is None:
        return x
    _, h, w = x.shape
    short = min(h, w)
    sigma = max(1.0, float(p.get("grow", 0.006)) * short)  # a little past the mask's edge
    region = (blur(o, f.mask[None], sigma)[0] * 3).clip(0, 1)
    clear = (region < 0.02) * 1.0
    around = push_pull(o, x, clear)
    fill = around
    mode = p.get("mode", "auto")
    if mode in ("auto", "model"):
        st = f.state.setdefault("plate", {})
        fresh = st.get("size") != (h, w) or st.get("index") != f.index - 1
        if fresh:
            st.update(plate=x * clear[None], seen=clear, lama=None)
        else:  # a decaying average where the background shows (grain settles), held elsewhere
            k = clear * 0.2
            st["plate"] = st["plate"] * (1 - k[None]) + x * k[None]
            st["seen"] = st["seen"] * (1 - k) + k
        st.update(size=(h, w), index=f.index)
        known = (st["seen"] > 0.01) * 1.0
        if mode == "model" and p.get("_lama"):
            painted = lama_fill(o, x, (region > 0.02) * (1 - known), str(p["_lama"]))
            if painted is not None:
                held = st.get("lama")
                st["lama"] = painted if held is None else held * 0.7 + painted * 0.3
                around = st["lama"]
        plate = st["plate"] / st["seen"].clip(1e-3, None)[None]
        fill = plate * known[None] + around * (1 - known[None])
    return (x * (1 - region[None]) + fill * region[None]).clip(0, 1)


REGION.add("object_remove")


@kernel("relight")
def relight(o: Any, x: Any, p: dict[str, Any], f: Frame) -> Any:
    """A virtual light (Resolve's Relight): a lamp at ``position`` (frame fractions) ``height``
    (short-side share) above the picture, its light falling off over ``radius``; surfaces turn
    to it by normals read from the depth's slope (``_depth``: Depth Anything V2, larger =
    nearer; ``surface``: 0 flat, 1 strongly shaped) and ``falloff`` dims the far. Without a
    depth the smoothed luma's slope shapes them (an approximation). ``ambient`` dims what the
    lamp does not reach; ``color`` tints the light."""
    xp = o.xp
    _, h, w = x.shape
    short = min(h, w)
    lx, ly = (float(v) for v in p.get("position", (0.3, 0.3)))
    depth = p.get("_depth")
    if depth is not None:  # a 0–1 relative depth: its slope per pixel, scaled to the frame
        y = blur(o, depth[None], max(1.0, 0.004 * short))
        k = float(p.get("surface", 0.5)) * 2 * short
    else:
        y = blur(o, luma(x)[None], max(1.0, 0.01 * short))
        k = float(p.get("surface", 0.5)) * 40
    gx = (shifted(o, y, -1, 0) - shifted(o, y, 1, 0))[0] / 2  # zero beyond the edge: shortcut
    gy = (shifted(o, y, 0, -1) - shifted(o, y, 0, 1))[0] / 2
    nx, ny = -gx * k, -gy * k
    ys, xs = o.grid(h, w)
    ax, ay = lx * w - xs, ly * h - ys
    az = float(p.get("height", 0.5)) * short
    shade = ((nx * ax + ny * ay + az) / (xp.sqrt(nx * nx + ny * ny + 1)
             * xp.sqrt(ax * ax + ay * ay + az * az))).clip(0, 1)  # fmt: skip
    d2 = (ax * ax + ay * ay) / (float(p.get("radius", 0.5)) * short) ** 2
    reach = 1 / (1 + d2)
    if depth is not None:  # the far gets less of the lamp
        reach = reach * (1 - float(p.get("falloff", 0.5)) * (1 - depth))
    tint = o.asarray(np.asarray(hex_rgb(p.get("color", "#ffe2b0")), np.float32)[:, None, None])
    lamp = float(p.get("intensity", 0.6)) * (shade * reach)[None] * tint
    dim = 1 - float(p.get("ambient", 0.2)) * (1 - reach)[None]
    return (x * dim * (1 + lamp)).clip(0, 1)
