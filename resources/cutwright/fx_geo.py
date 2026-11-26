"""The geometric effects' shared machinery (``fx.geo``): a structure read from the picture and
drawn over it. A leaf helper like ``fx_kernels_paint``: it registers no kernel and imports only
``fx_base``, ``fx_constants`` and ``fx_kernels_paint`` (the kernels are in ``fx_kernels_geo``).

**Field** (``geo_field``): luma at about ``FIELD_SIZE`` px on the short side, lightly blurred,
its Sobel gradient, and the **nodes**: per cell of a grid the strongest edge point (Sobel
magnitude plus a little local contrast, edges under ``threshold`` never), refined across the
edge to a sub-pixel place, sorted by score, kept ``0.7`` cell apart, at most ``max_nodes``; the
strongest ones far apart are the ``ANCHORS`` anchors. The cells are laid on the media's camera
path (``Frame.camera``, as a periodic pattern's grid): on a pan the same content points win the
same cells, so the structure rides the picture instead of boiling.

``follow`` decides how the structure lives in time (``geo_layers``):

- ``frame``: built anew every frame (moving subjects boil, on purpose);
- ``track``: two layers of nodes carried along the optical flow (``fx_kernels_paint.motion``)
  and nudged back onto their edge, each built anew in turn every ``CARRY`` frames and
  cross-faded (the carried textures' scheme): a frame depends on the ``CARRY`` frames before it
  (``FxDef.history``), so a still equals the render; a scene cut builds both anew;
- ``hold``: the nodes of one frame (``_hold``, worked out by the render on the clip's frame at
  ``hold_at``) stand still.

**Drawing** (``Canvas``): our own anti-aliased splatting. Every primitive becomes samples
``STEP`` px apart (across its width too), each carrying its share of the area, accumulated with
bilinear weights by ``bincount``; a dash is a test on the sample's arc length. numpy only: the
objects environment has no OpenCV, the field is small and the drawing is scattered.

Time, motion and sound come in the params: ``_t`` (clip seconds; the source time when absent),
``_bass`` / ``_mid`` / ``_high`` / ``_beat`` (0–1: the render's band envelopes), ``_hold``.
Randomness is a hash of the node's cell on the camera grid and the effect's seed, so a node
keeps its phase while it lives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

if __package__:  # in Cutwright's process
    from cutwright.modules.fx.runners.fx_base import LUMA, NumpyOps, blur
    from cutwright.modules.fx.runners.fx_constants import CARRY
    from cutwright.modules.fx.runners.fx_kernels_paint import motion
else:  # in the objects environment: the files next to the runner, by their bare names
    from fx_base import LUMA, NumpyOps, blur  # type: ignore[no-redef]
    from fx_constants import CARRY  # type: ignore[no-redef]
    from fx_kernels_paint import motion  # type: ignore[no-redef]

FIELD_SIZE = 540  # the field is worked out at about this many px on the short side
ANCHORS = 7
ANCHOR_GAP = 0.18  # anchors at least this far apart (short-side fraction)
NODE_GAP = 0.7  # nodes at least this many cells apart
STEP = 0.5  # px between drawing samples
_NO = NumpyOps()


def numpy_rgb(o: Any, x: Any) -> Any:
    return np.asarray(o.numpy(x), np.float32)


def cell_px(p: dict[str, Any], short: int) -> float:
    """The grid's cell (px) for ``density`` 0–1: about 12 % of the short side down to 3 %."""
    return short * (0.12 - 0.09 * float(p.get("density", 0.5)))


def _down(y: Any, q: int) -> Any:
    if q <= 1:
        return y
    h, w = y.shape[0] // q * q, y.shape[1] // q * q
    return y[:h, :w].reshape(h // q, q, w // q, q).mean((1, 3))


def smooth(a: Any, sigma: float) -> Any:
    """Gaussian blur of a 2-D numpy array (sigma in its px)."""
    return blur(_NO, np.asarray(a, np.float32)[None], sigma)[0]


def smooth_rgb(X: Any, sigma: float) -> Any:
    """Gaussian blur of a (3, h, w) numpy picture."""
    return np.asarray(blur(_NO, np.asarray(X, np.float32), sigma), np.float32)


def sobel(y: Any) -> tuple[Any, Any]:
    """(gx, gy) of a 2-D array, scaled so a 0 → 1 step gives 0.5 on both sides of it."""
    p = np.pad(y, 1, mode="edge")
    gx = p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:] - p[:-2, :-2] - 2 * p[1:-1, :-2] - p[2:, :-2]
    gy = p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:] - p[:-2, :-2] - 2 * p[:-2, 1:-1] - p[:-2, 2:]
    return gx / 8, gy / 8


def bilinear(a: Any, x: Any, y: Any) -> Any:
    """``a`` (h, w) read at float px (x, y), clamped to the edge."""
    h, w = a.shape
    x = np.clip(np.asarray(x, np.float64), 0, w - 1)
    y = np.clip(np.asarray(y, np.float64), 0, h - 1)
    x0 = np.minimum(np.floor(x).astype(np.int64), w - 2) if w > 1 else np.zeros_like(x, np.int64)
    y0 = np.minimum(np.floor(y).astype(np.int64), h - 2) if h > 1 else np.zeros_like(y, np.int64)
    fx, fy = x - x0, y - y0
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    return ((a[y0, x0] * (1 - fx) + a[y0, x1] * fx) * (1 - fy)
            + (a[y1, x0] * (1 - fx) + a[y1, x1] * fx) * fy)  # fmt: skip


def on_camera_grid(px: Any, py: Any, h: int, w: int, camera: Any) -> tuple[Any, Any]:
    """Frame px (x, y) as px on the media's first frame (``Frame.camera`` = [turn, x, y]: frame
    pixel p is at c + R(turn)(p − c) + (x, y) there, as ``fx_kernels_paint.pattern_grid``)."""
    if camera is None:
        return px, py
    turn, mx, my = (float(v) for v in camera)
    c, s = math.cos(turn), math.sin(turn)
    dy, dx = py - (h - 1) / 2, px - (w - 1) / 2
    return dx * c - dy * s + (w - 1) / 2 + mx, dx * s + dy * c + (h - 1) / 2 + my


def camera_lattice(h: int, w: int, camera: Any, spacing: float) -> tuple[Any, Any]:
    """Points ``spacing`` px apart on the media's first frame (the camera grid) that fall in
    this frame, as frame px (n, 2), and their lattice keys: seeds that ride the picture."""
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], np.float64)
    gx, gy = on_camera_grid(corners[:, 0], corners[:, 1], h, w, camera)
    i0, i1 = math.floor(min(gx) / spacing), math.ceil(max(gx) / spacing)
    j0, j1 = math.floor(min(gy) / spacing), math.ceil(max(gy) / spacing)
    jj, ii = np.mgrid[j0 : j1 + 1, i0 : i1 + 1]
    qx, qy = (ii.ravel() + 0.5) * spacing, (jj.ravel() + 0.5) * spacing
    if camera is None:
        px, py = qx, qy
    else:  # p = c + Rᵀ(q − m − c)
        turn, mx, my = (float(v) for v in camera)
        c, s = math.cos(turn), math.sin(turn)
        dx, dy = qx - mx - (w - 1) / 2, qy - my - (h - 1) / 2
        px, py = dx * c + dy * s + (w - 1) / 2, -dx * s + dy * c + (h - 1) / 2
    inside = (px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)
    keys = (jj.ravel() + 2**20) * 2**21 + (ii.ravel() + 2**20)
    return np.stack([px, py], 1)[inside], keys[inside]


def phase_of(keys: Any, seed: int, salt: int = 0) -> Any:
    """A number in [0, 1) per key (a node's cell), steady for the same key, seed and salt."""
    k = np.asarray(keys, np.float64).astype(np.int64).astype(np.uint64)
    k = k * np.uint64(0x9E3779B97F4A7C15) + np.uint64((int(seed) * 1000003 + salt) & 0xFFFFFFFF)
    k ^= k >> np.uint64(31)
    k *= np.uint64(0xBF58476D1CE4E5B9)
    k ^= k >> np.uint64(29)
    return (k >> np.uint64(11)).astype(np.float64) / float(2**53)


# --- the field ---------------------------------------------------------------------------------


@dataclass
class Field:
    """One frame's field. Maps are at field scale (``q`` frame px per field px); node places
    are frame px [x, y], best first."""

    q: int
    size: tuple[int, int]  # the frame (h, w)
    cell: float  # frame px
    luma: Any
    gx: Any
    gy: Any
    mag: Any
    score: Any
    threshold: float
    xy: Any = None  # (n, 2)
    s: Any = None  # (n,)
    keys: Any = None  # (n,) the node's cell on the camera grid
    anchors: Any = None  # indices into xy

    def at(self, a: Any, px: Any, py: Any) -> Any:
        """A field map read at frame px."""
        q = self.q
        return bilinear(a, (np.asarray(px) + 0.5) / q - 0.5, (np.asarray(py) + 0.5) / q - 0.5)


def _offset(a: Any, b: Any, c: Any) -> Any:
    """The parabola's peak through (−1, a), (0, b), (1, c), within ±0.5."""
    d = a - 2 * b + c
    safe = np.where(d < -1e-9, d, -1.0)
    return np.where(d < -1e-9, 0.5 * (a - c) / safe, 0.0).clip(-0.5, 0.5)


def _refine(fld: Field, iy: Any, ix: Any) -> tuple[Any, Any]:
    """Field px (iy, ix) as frame px, moved across the edge (along the gradient's main axis)
    to the magnitude's sub-pixel peak."""
    m = fld.mag
    along_x = np.abs(fld.gx[iy, ix]) >= np.abs(fld.gy[iy, ix])
    ox = np.where(along_x, _offset(m[iy, ix - 1], m[iy, ix], m[iy, ix + 1]), 0.0)
    oy = np.where(along_x, 0.0, _offset(m[iy - 1, ix], m[iy, ix], m[iy + 1, ix]))
    q = fld.q
    return (ix + ox + 0.5) * q - 0.5, (iy + oy + 0.5) * q - 0.5


def spaced(xy: Any, gap: float, limit: int) -> Any:
    """Indices of ``xy`` (best first) taken greedily at least ``gap`` px apart, ``limit`` at
    most."""
    picked = np.zeros((max(limit, 1), 2), np.float64)
    out: list[int] = []
    g2 = gap * gap
    for i in range(len(xy)):
        if len(out) >= limit:
            break
        if out and float(((picked[: len(out)] - xy[i]) ** 2).sum(1).min()) < g2:
            continue
        picked[len(out)] = xy[i]
        out.append(i)
    return np.asarray(out, np.int64)


def anchors_of(xy: Any, short: int) -> Any:
    """The best nodes (``xy`` best first) at least ``ANCHOR_GAP`` × short side apart, up to
    ``ANCHORS``."""
    return spaced(xy, ANCHOR_GAP * short, ANCHORS)


def geo_field(o: Any, x: Any, p: dict[str, Any], f: Any, rgb: Any = None,
              weight: Any = None, boost: Any = None) -> Field:  # fmt: skip
    """The frame's field and nodes (``rgb``: the frame as numpy, when the caller has it).
    ``weight`` (frame-sized, 0–1) scales the node score (a face kept clear), ``boost`` adds to
    it (a subject's outline emphasised)."""
    X = numpy_rgb(o, x) if rgb is None else rgb
    _, h, w = X.shape
    short = min(h, w)
    q = max(1, round(short / getattr(o, "field_size", FIELD_SIZE)))
    y = smooth(_down(LUMA[0] * X[0] + LUMA[1] * X[1] + LUMA[2] * X[2], q), 0.7)
    gx, gy = sobel(y)
    mag = np.hypot(gx, gy)
    cell = cell_px(p, short)
    sigma = max(1.0, cell / q / 3)
    mean = smooth(y, sigma)
    contrast = np.sqrt((smooth(y * y, sigma) - mean * mean).clip(0, None))
    threshold = float(p.get("threshold", 0.04))
    score = mag + 0.3 * contrast
    if boost is not None:
        b = _down(boost, q)[: y.shape[0], : y.shape[1]]
        score, mag = score + b, mag + b
    if weight is not None:
        k = _down(weight, q)[: y.shape[0], : y.shape[1]]
        score, mag = score * k, mag * k
    fld = Field(q, (h, w), cell, y, gx, gy, mag, score, threshold)
    valid = mag > threshold
    valid[[0, -1], :] = False
    valid[:, [0, -1]] = False
    iy, ix = np.nonzero(valid)
    if not len(iy):
        return _nodes(fld, np.zeros((0, 2)), np.zeros(0), np.zeros(0, np.int64), short)
    gxs, gys = on_camera_grid((ix + 0.5) * q - 0.5, (iy + 0.5) * q - 0.5, h, w, f.camera)
    cu, cv = np.floor(gxs / cell).astype(np.int64), np.floor(gys / cell).astype(np.int64)
    keys = (cv + 2**20) * 2**21 + (cu + 2**20)
    tie = ((gxs * 0.618034 + gys * 0.381966) / cell) % 1 * 1e-6  # rides the content too
    score = fld.score[iy, ix] + tie
    order = np.lexsort((-score, keys))
    first = np.ones(len(order), bool)
    first[1:] = keys[order][1:] != keys[order][:-1]
    win = order[first]
    iy, ix, keys, score = iy[win], ix[win], keys[win], fld.score[iy[win], ix[win]]
    px, py = _refine(fld, iy, ix)
    best = np.argsort(-score, kind="stable")
    xy = np.stack([px, py], 1)[best]
    keep = spaced(xy, NODE_GAP * cell, int(p.get("max_nodes", 160)))
    return _nodes(fld, xy[keep], score[best][keep], keys[best][keep], short)


def _nodes(fld: Field, xy: Any, s: Any, keys: Any, short: int) -> Field:
    fld.xy = np.asarray(xy, np.float64).reshape(-1, 2)
    fld.s, fld.keys = np.asarray(s, np.float64), np.asarray(keys, np.int64)
    fld.anchors = anchors_of(fld.xy, short)
    return fld


# --- subject and faces -------------------------------------------------------------------------


@dataclass
class Masks:
    """This frame's subject (the effect's scope mask, 0–1) and faces (soft ellipses), frame-sized
    numpy, None when not there."""

    subject: Any = None
    face: Any = None


def face_mask(p: dict[str, Any], h: int, w: int) -> Any:
    """Soft ellipses over ``_faces`` ([[cx, cy, w, h], …] as fractions of the frame), a little
    larger than the box (hair, chin)."""
    out = np.zeros((h, w), np.float32)
    for cx, cy, fw, fh in p.get("_faces") or []:
        rx, ry = max(1.0, 0.65 * fw * w), max(1.0, 0.75 * fh * h)
        x0, x1 = max(0, int(cx * w - 1.3 * rx)), min(w, int(cx * w + 1.3 * rx) + 1)
        y0, y1 = max(0, int(cy * h - 1.3 * ry)), min(h, int(cy * h + 1.3 * ry) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        r = np.hypot((xs - cx * w) / rx, (ys - cy * h) / ry)
        out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], ((1.15 - r) / 0.3).clip(0, 1))
    return out


def masks(o: Any, p: dict[str, Any], f: Any, h: int, w: int) -> Masks:
    subject = np.asarray(o.numpy(f.mask), np.float32) if f.mask is not None else None
    face = face_mask(p, h, w) if p.get("face", "off") != "off" and p.get("_faces") else None
    return Masks(subject, face)


def node_weights(p: dict[str, Any], mk: Masks) -> tuple[Any, Any]:
    """(weight, boost) for ``geo_field``: ``face`` reduce / exclude keeps nodes off faces;
    ``subject: emphasize`` adds the subject's outline to the node score."""
    weight = boost = None
    if mk.face is not None:  # exclude: no node where the face mask is over half
        exclude = p.get("face") == "exclude"
        weight = (1 - 2 * mk.face).clip(0, 1) if exclude else 1 - 0.6 * mk.face
    if mk.subject is not None and p.get("subject", "off") == "emphasize":
        gx, gy = sobel(smooth(mk.subject, 1.5))
        boost = np.hypot(gx, gy) * 2.0
    return weight, boost


def material(X: Any, p: dict[str, Any], mk: Masks) -> Any:
    """What a node effect draws on: the ``backdrop``, or inside an isolated subject its
    ``material`` — ``refract`` (the picture bent along its own luma gradient, glass-like) or
    ``blocks`` (coarse pixel blocks)."""
    base = backdrop(X, p)
    mode = p.get("material", "look")
    if mk.subject is None or p.get("subject", "off") != "isolate" or mode == "look":
        return base
    _, h, w = X.shape
    short = min(h, w)
    if mode == "blocks":
        s = max(2, round(0.02 * short))
        hh, ww = h // s, w // s
        small = X[:, : hh * s, : ww * s].reshape(3, hh, s, ww, s).mean((2, 4))
        iy, ix = np.minimum(np.arange(h) // s, hh - 1), np.minimum(np.arange(w) // s, ww - 1)
        mat = small[:, iy][:, :, ix]
    else:
        y = smooth(LUMA[0] * X[0] + LUMA[1] * X[1] + LUMA[2] * X[2], max(1.0, 0.006 * short))
        gx, gy = sobel(y)
        k = 0.03 * short / 0.05
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
        sx, sy = xs + (gx * k).clip(-0.04 * short, 0.04 * short), ys + (gy * k).clip(
            -0.04 * short, 0.04 * short)  # fmt: skip
        mat = np.stack([bilinear(X[c], sx, sy) for c in range(3)]).astype(np.float32)
    m = mk.subject[None]
    return base + (mat - base) * m


def finish(X: Any, y: Any, p: dict[str, Any], mk: Masks) -> Any:
    """The effect's picture ``y`` against the one it was given ``X``: faces (``face``: reduce
    keeps 40 % of the effect there, exclude none), then the subject by ``subject`` — off: the
    entry's scope as the processor would (``_scope`` inside / outside), protect: the subject
    stays as it was, isolate: only the subject gets the effect, emphasize: the whole frame."""

    def mix(keep: Any) -> Any:  # X where keep is 1, y where 0: exact at both ends
        k = keep[None]
        return X * k + y * (1 - k)

    if mk.face is not None:
        y = mix(mk.face * (1.0 if p.get("face") == "exclude" else 0.6))
    m = mk.subject
    if m is not None:
        mode, scope = p.get("subject", "off"), p.get("_scope", "inside")
        if (mode == "off" and scope == "outside") or mode == "protect":
            y = mix(m)
        elif mode in ("off", "isolate"):
            y = mix(1 - m)
    return np.clip(y, 0, 1)


# --- in time -----------------------------------------------------------------------------------


@dataclass
class Layer:
    """Nodes drawn with ``weight`` (two cross-faded layers in ``track``); ``extra`` is the
    effect's own data worked out when the layer was built (links, roles), carried with it."""

    xy: Any
    s: Any
    keys: Any
    anchors: Any
    weight: float
    born: int
    extra: dict[str, Any] = field(default_factory=dict)


def _built(fld: Field, index: int) -> Layer:
    return Layer(fld.xy.copy(), fld.s.copy(), fld.keys.copy(), fld.anchors.copy(), 1.0, index)


def _carried(layer: Layer, fy: Any, fx: Any, fld: Field) -> Layer:
    """The layer's nodes moved along the backward flow (``cur(p) ≈ prev(p + f(p))``: a node at
    q goes to p with p + f(p) = q, two fixed-point steps) and nudged halfway to their edge's
    sub-pixel peak where they still sit on an edge."""
    px, py = layer.xy[:, 0], layer.xy[:, 1]
    nx, ny = px, py
    for _ in range(2):
        nx, ny = px - bilinear(fx, nx, ny), py - bilinear(fy, nx, ny)
    q = fld.q
    fh, fw = fld.mag.shape
    ix = np.clip(np.rint((nx + 0.5) / q - 0.5).astype(np.int64), 1, fw - 2)
    iy = np.clip(np.rint((ny + 0.5) / q - 0.5).astype(np.int64), 1, fh - 2)
    sx, sy = _refine(fld, iy, ix)
    along_x = np.abs(fld.gx[iy, ix]) >= np.abs(fld.gy[iy, ix])
    on_edge = fld.mag[iy, ix] > fld.threshold
    nx = np.where(on_edge & along_x, nx + 0.5 * (sx - nx), nx)
    ny = np.where(on_edge & ~along_x, ny + 0.5 * (sy - ny), ny)
    return Layer(np.stack([nx, ny], 1), layer.s, layer.keys, layer.anchors, layer.weight,
                 layer.born, layer.extra)  # fmt: skip


def held(p: dict[str, Any], fld: Field, f: Any) -> Layer:
    """``_hold`` ([[x, y, score, key], …], best first) as a still layer."""
    rows = np.asarray(p["_hold"], np.float64).reshape(-1, 4)
    xy = rows[:, :2]
    keep = f.state.setdefault("geo_hold", {})
    if keep.get("rows") is None or not np.array_equal(keep["rows"], rows):
        keep.update(rows=rows, extra={})
    return Layer(xy, rows[:, 2], rows[:, 3].astype(np.int64), anchors_of(xy, min(fld.size)),
                 1.0, -1, keep["extra"])  # fmt: skip


def geo_layers(o: Any, x: Any, p: dict[str, Any], f: Any, rgb: Any = None,
               mk: Masks | None = None) -> tuple[list[Layer], Field]:  # fmt: skip
    """The layers to draw this frame and the frame's field (see the module's docstring); with
    ``mk`` a face is kept clear of nodes and a subject's outline draws them (``node_weights``)."""
    X = numpy_rgb(o, x) if rgb is None else rgb
    weight, boost = node_weights(p, mk) if mk is not None else (None, None)
    def analyse():
        return geo_field(o, x, p, f, X, weight, boost)
    fld = o.field(analyse) if hasattr(o, "field") else analyse()
    follow = p.get("follow", "track")
    if follow == "hold" and p.get("_hold"):
        return [held(p, fld, f)], fld
    if follow != "track":
        return [_built(fld, f.index)], fld
    st = f.state.setdefault("geo", {})
    m = motion(o, x, f)
    phase = f.index % CARRY
    if m.fy is None or "layers" not in st:  # a fresh start, a gap or a scene cut
        layers = [_built(fld, f.index), _built(fld, f.index)]
    else:
        fy, fx = numpy_rgb(o, m.fy), numpy_rgb(o, m.fx)
        layers = [_carried(layer, fy, fx, fld) for layer in st["layers"]]
        if phase == 0:
            layers[0] = _built(fld, f.index)
        if phase == CARRY // 2:
            layers[1] = _built(fld, f.index)
    wa = 1 - abs(phase - CARRY / 2) / (CARRY / 2)
    layers[0].weight, layers[1].weight = wa, 1 - wa
    st["layers"] = layers
    return layers, fld


def hold_rows(fld: Field) -> list[list[float]]:
    """The field's nodes as ``_hold`` rows (what the render passes for ``follow: hold``)."""
    return [[round(float(x), 3), round(float(y), 3), round(float(s), 5), float(k)]
            for (x, y), s, k in zip(fld.xy, fld.s, fld.keys, strict=True)]  # fmt: skip


# --- motion and sound --------------------------------------------------------------------------


def clock(p: dict[str, Any], f: Any) -> float:
    """Clip seconds (``_t``), else source seconds."""
    return float(p["_t"]) if "_t" in p else f.index / f.fps


@dataclass
class Sound:
    bass: float = 0.0
    mid: float = 0.0
    high: float = 0.0
    beat: float = 0.0


def sound(p: dict[str, Any]) -> Sound:
    """The band envelopes at this frame, scaled by ``audio_amount`` (all 0 when off)."""
    if p.get("audio", "off") == "off":
        return Sound()
    k = float(p.get("audio_amount", 50)) / 100
    return Sound(*(k * float(p.get(f"_{n}", 0.0)) for n in ("bass", "mid", "high", "beat")))


@dataclass
class Moved:
    """A layer's nodes this frame after the effect's own motion: places, how visible (0–1) and
    how big (× the effect's size) each is, and a gain on the links."""

    xy: Any
    alpha: Any
    scale: Any
    links: float = 1.0


def animate(layer: Layer, fld: Field, p: dict[str, Any], f: Any) -> Moved:
    """``motion`` applied to a layer (``speed`` / ``amount`` 0–100, sound on top: bass → size,
    mid → links, beat → pulse). Every node's phase comes from its cell (``phase_of``)."""
    h, w = fld.size
    short = min(h, w)
    n = len(layer.xy)
    mode = p.get("motion", "none")
    a, s = float(p.get("amount", 30)) / 100, float(p.get("speed", 30)) / 100
    t = clock(p, f)
    ph = phase_of(layer.keys, f.seed)
    xy = layer.xy.astype(np.float64).copy()
    alpha, scale = np.ones(n), np.ones(n)
    snd = sound(p)
    cell = fld.cell
    if mode == "drift" and n:  # along the edge the node sits on
        gx, gy = fld.at(fld.gx, xy[:, 0], xy[:, 1]), fld.at(fld.gy, xy[:, 0], xy[:, 1])
        norm = np.hypot(gx, gy) + 1e-9
        k = a * cell * 0.6 * np.sin(2 * math.pi * (0.6 * s * t + ph))
        xy += np.stack([-gy / norm * k, gx / norm * k], 1)
    elif mode == "orbit":
        ang = 2 * math.pi * (0.8 * s * t + ph)
        xy += a * cell * 0.4 * np.stack([np.cos(ang), np.sin(ang)], 1)
    elif mode == "wave":
        xy[:, 1] += a * cell * 0.6 * np.sin(2 * math.pi * (s * t - xy[:, 0] / (0.5 * short)))
    elif mode == "glitch":
        band = np.floor(xy[:, 1] / (short / 14)).astype(np.int64)
        step = math.floor(t * (2 + 10 * s))
        r = phase_of(band * 7919 + step, f.seed, 3)
        r2 = phase_of(band * 104729 + step, f.seed, 4)
        xy[:, 0] += np.where(r < 0.15 + 0.35 * a, (r2 - 0.5) * 2 * a * short * 0.06, 0.0)
    elif mode == "pulse":
        scale *= 1 + a * 0.6 * (0.5 + 0.5 * math.sin(2 * math.pi * (0.5 + 2 * s) * t))
    elif mode == "scan":
        line = (t * (0.1 + 0.6 * s)) % 1.0 * h
        lit = np.exp(-(((xy[:, 1] - line) / (0.05 * h)) ** 2))
        alpha *= 0.35 + 0.65 * (1 - a) + a * 0.65 * lit
        scale *= 1 + a * 0.8 * lit
    elif mode in ("assemble", "sequence"):
        progress = min(1.0, t * (0.15 + 1.35 * s))
        if mode == "assemble":  # top to bottom
            alpha *= ((progress * 1.25 - xy[:, 1] / h) / 0.25).clip(0, 1)
        else:  # best first
            alpha *= (progress * (n + 4) - np.arange(n)).clip(0, 1)
    scale *= 1 + 0.8 * snd.bass + 0.5 * snd.beat
    return Moved(xy, alpha, scale, links=1 + 1.2 * snd.mid)


def reveal(p: dict[str, Any], f: Any) -> float:
    """How far a drawn line has grown (0–1): ``assemble`` grows the links after the nodes."""
    if p.get("motion", "none") != "assemble":
        return 1.0
    s = float(p.get("speed", 30)) / 100
    return float(min(1.0, max(0.0, clock(p, f) * (0.15 + 1.35 * s) * 1.25 - 0.4) / 0.6))


# --- drawing -----------------------------------------------------------------------------------


# Our own 3 × 5 pixel glyphs (rows top to bottom, 3 bits each) for labels and character clouds.
GLYPHS = {
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111",
    "3": "111001111001111", "4": "101101111001001", "5": "111100111001111",
    "6": "111100111101111", "7": "111001001001001", "8": "111101111101111",
    "9": "111101111001111", "A": "010101111101101", "B": "110101110101110",
    "C": "111100100100111", "D": "110101101101110", "E": "111100111100111",
    "F": "111100111100100", "X": "000000101010101", ":": "000010000010000",
    ".": "000000000000010", "-": "000000111000000", "+": "000010111010000",
    "/": "001001010100100", "#": "101111101111101", "=": "000111000111000",
}  # fmt: skip


def _per(v: Any, n: int) -> Any:
    return np.broadcast_to(np.asarray(v, np.float64), (n,)).astype(np.float64)


def _rgb(v: Any, n: int) -> Any:
    a = np.asarray(v, np.float64)
    return np.broadcast_to(a.reshape(-1, 3) if a.ndim == 2 else a.reshape(1, 3), (n, 3))


class Canvas:
    """Anti-aliased drawing by splatting (see the module's docstring). Coordinates are frame px,
    pixel centres on integers; colours are display-encoded RGB 0–1."""

    def __init__(self, h: int, w: int) -> None:
        self.h, self.w = h, w
        self.parts: list[tuple[Any, Any, Any, Any]] = []

    def samples(self, xs: Any, ys: Any, wt: Any, rgb: Any) -> None:
        keep = wt > 0
        if keep.any():
            self.parts.append((xs[keep], ys[keep], wt[keep], rgb[keep]))

    def lines(self, a: Any, b: Any, width: Any, rgb: Any, alpha: Any = 1.0,
              dash: tuple[float, float] | None = None, phase: Any = 0.0) -> None:  # fmt: skip
        """Segments a → b ((n, 2) each) ``width`` px wide (butt ends); ``dash`` (on, off) px
        along each segment from its start, moved on by ``phase`` px."""
        a = np.asarray(a, np.float64).reshape(-1, 2)
        b = np.asarray(b, np.float64).reshape(-1, 2)
        n = len(a)
        if not n:
            return
        width, alpha, phase = _per(width, n), _per(alpha, n), _per(phase, n)
        d = b - a
        length = np.hypot(d[:, 0], d[:, 1])
        ok = (length > 1e-6) & (alpha > 0) & (width > 0)
        if not ok.any():
            return
        a, d, length, rgb = a[ok], d[ok], length[ok], _rgb(rgb, n)[ok]
        width, alpha, phase = width[ok], alpha[ok], phase[ok]
        ns = np.maximum(1, np.ceil(length / STEP)).astype(np.int64)
        na = np.maximum(1, np.round(width / STEP)).astype(np.int64)
        total = ns * na
        seg = np.repeat(np.arange(len(a)), total)
        idx = np.arange(int(total.sum())) - np.repeat(np.cumsum(total) - total, total)
        along = (idx // na[seg] + 0.5) / ns[seg]
        across = ((idx % na[seg] + 0.5) / na[seg] - 0.5) * np.maximum(width[seg], 1.0)
        nrm = np.stack([-d[:, 1], d[:, 0]], 1) / length[:, None]
        pos = a[seg] + d[seg] * along[:, None] + nrm[seg] * across[:, None]
        wt = (length / ns * width / na * alpha)[seg]
        if dash is not None:
            on, off = dash
            wt = wt * (((along * length[seg] + phase[seg]) % (on + off)) < on)
        self.samples(pos[:, 0], pos[:, 1], wt, rgb[seg])

    def polyline(self, pts: Any, width: float, rgb: Any, alpha: float = 1.0, closed: bool = False,
                 dash: tuple[float, float] | None = None, phase: float = 0.0) -> None:  # fmt: skip
        """Connected segments through ``pts`` (k, 2); a dash runs on along the whole line."""
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        if closed and len(pts) > 2:
            pts = np.vstack([pts, pts[:1]])
        if len(pts) < 2:
            return
        a, b = pts[:-1], pts[1:]
        run = np.concatenate([[0.0], np.cumsum(np.hypot(*(b - a).T))[:-1]])
        self.lines(a, b, width, rgb, alpha, dash, phase + run)

    def discs(self, c: Any, r: Any, rgb: Any, alpha: Any = 1.0) -> None:
        """Filled discs, centres ``c`` (n, 2), radii ``r``."""
        c = np.asarray(c, np.float64).reshape(-1, 2)
        n = len(c)
        if not n:
            return
        r, alpha, rgb = _per(r, n), _per(alpha, n), _rgb(rgb, n)
        ok = (r > 0) & (alpha > 0)
        c, r, alpha, rgb = c[ok], r[ok], alpha[ok], rgb[ok]
        if not len(c):
            return
        tiny = r < STEP
        if tiny.any():
            self.samples(c[tiny, 0], c[tiny, 1], math.pi * r[tiny] ** 2 * alpha[tiny], rgb[tiny])
        c, r, alpha, rgb = c[~tiny], r[~tiny], alpha[~tiny], rgb[~tiny]
        if not len(c):
            return
        k = (2 * np.ceil(r / STEP)).astype(np.int64)  # samples a side
        total = k * k
        seg = np.repeat(np.arange(len(c)), total)
        idx = np.arange(int(total.sum())) - np.repeat(np.cumsum(total) - total, total)
        ox = ((idx % k[seg]) + 0.5) * STEP - k[seg] * STEP / 2
        oy = ((idx // k[seg]) + 0.5) * STEP - k[seg] * STEP / 2
        inside = ox * ox + oy * oy <= r[seg] ** 2
        self.samples(c[seg, 0] + ox, c[seg, 1] + oy, STEP * STEP * alpha[seg] * inside, rgb[seg])

    def rings(self, c: Any, r: Any, width: Any, rgb: Any, alpha: Any = 1.0,
              arc: tuple[Any, Any] | None = None, dash: tuple[float, float] | None = None,
              phase: Any = 0.0) -> None:  # fmt: skip
        """Circles (or arcs from ``arc[0]`` to ``arc[1]`` radians, each scalar or (n,)) as
        polylines fine enough to look round."""
        c = np.asarray(c, np.float64).reshape(-1, 2)
        n = len(c)
        if not n:
            return
        r, width, alpha, phase = _per(r, n), _per(width, n), _per(alpha, n), _per(phase, n)
        a0 = _per(arc[0], n) if arc is not None else np.zeros(n)
        a1 = _per(arc[1], n) if arc is not None else np.full(n, 2 * math.pi)
        rgb = _rgb(rgb, n)
        m = np.clip(np.ceil(np.abs(a1 - a0) * r / 3), 8, 256).astype(np.int64)
        seg = np.repeat(np.arange(n), m)
        j = np.arange(int(m.sum())) - np.repeat(np.cumsum(m) - m, m)
        t0 = a0[seg] + (a1 - a0)[seg] * j / m[seg]
        t1 = a0[seg] + (a1 - a0)[seg] * (j + 1) / m[seg]
        p0 = c[seg] + r[seg, None] * np.stack([np.cos(t0), np.sin(t0)], 1)
        p1 = c[seg] + r[seg, None] * np.stack([np.cos(t1), np.sin(t1)], 1)
        run = phase[seg] + r[seg] * np.abs(t0 - a0[seg])
        self.lines(p0, p1, width[seg], rgb[seg], alpha[seg], dash, run)

    def rects(self, a: Any, b: Any, rgb: Any, alpha: Any = 1.0) -> None:
        """Filled rectangles from corner ``a`` to ``b`` ((n, 2) each, a ≤ b), sampled like
        discs (edges anti-aliased by the splat)."""
        a = np.asarray(a, np.float64).reshape(-1, 2)
        b = np.asarray(b, np.float64).reshape(-1, 2)
        n = len(a)
        if not n:
            return
        alpha, rgb = _per(alpha, n), _rgb(rgb, n)
        size = np.maximum(b - a, 0.0)
        kx = np.maximum(1, np.ceil(size[:, 0] / STEP)).astype(np.int64)
        ky = np.maximum(1, np.ceil(size[:, 1] / STEP)).astype(np.int64)
        total = kx * ky
        seg = np.repeat(np.arange(n), total)
        idx = np.arange(int(total.sum())) - np.repeat(np.cumsum(total) - total, total)
        fx = ((idx % kx[seg]) + 0.5) / kx[seg]
        fy = ((idx // kx[seg]) + 0.5) / ky[seg]
        area = (size[:, 0] * size[:, 1] / total * alpha)[seg]
        self.samples(a[seg, 0] + size[seg, 0] * fx - 0.5, a[seg, 1] + size[seg, 1] * fy - 0.5,
                     area, rgb[seg])  # fmt: skip

    def text(self, s: str, at: Any, px: float, rgb: Any, alpha: float = 1.0) -> None:
        """``s`` in the built-in 3 × 5 glyphs (``GLYPHS``), top-left at ``at``, each glyph
        pixel ``px`` wide; unknown characters are blank."""
        tl = []
        x0, y0 = float(at[0]), float(at[1])
        for i, ch in enumerate(s.upper()):
            bits = GLYPHS.get(ch)
            if bits is None:
                continue
            for k, bit in enumerate(bits):
                if bit == "1":
                    tl.append((x0 + (4 * i + k % 3) * px, y0 + (k // 3) * px))
        if tl:
            a = np.asarray(tl)
            self.rects(a, a + px, rgb, alpha)

    def layer(self) -> tuple[Any, Any]:
        """(premultiplied RGB (3, h, w), coverage (h, w)), numpy float32."""
        h, w = self.h, self.w
        if not self.parts:
            return np.zeros((3, h, w), np.float32), np.zeros((h, w), np.float32)
        xs, ys, wt, rgb = (np.concatenate(v) for v in zip(*self.parts, strict=True))
        x0, y0 = np.floor(xs), np.floor(ys)
        fx, fy = xs - x0, ys - y0
        idx, val, col = [], [], []
        for dy, dx, k in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)),
                          (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):  # fmt: skip
            xi, yi = x0 + dx, y0 + dy
            ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            idx.append((yi[ok] * w + xi[ok]).astype(np.int64))
            val.append(wt[ok] * k[ok])
            col.append(rgb[ok])
        i, v, c = np.concatenate(idx), np.concatenate(val), np.concatenate(col)
        cov = np.bincount(i, v, h * w)
        rgb_sum = np.stack([np.bincount(i, v * c[:, ch], h * w) for ch in range(3)])
        alpha = np.minimum(cov, 1.0)
        premul = rgb_sum / np.maximum(cov, 1e-9) * alpha
        return premul.reshape(3, h, w).astype(np.float32), alpha.reshape(h, w).astype(np.float32)


def over(base: Any, premul: Any, alpha: Any) -> Any:
    return premul + base * (1 - alpha[None])


def backdrop(X: Any, p: dict[str, Any]) -> Any:
    """What the structure is drawn on (``base``): the picture dimmed by ``dim`` (``source``), a
    duotone of its luma between ``ink`` and ``paper``, or black."""
    mode = p.get("base", "source")
    if mode == "black":
        return np.zeros_like(X)
    if mode == "duotone":
        ink, paper = (np.asarray(_hex(p.get(k, d)), np.float32)[:, None, None]
                      for k, d in (("ink", "#0e0f1a"), ("paper", "#e9e4d8")))  # fmt: skip
        y = (LUMA[0] * X[0] + LUMA[1] * X[1] + LUMA[2] * X[2])[None]
        return ink + (paper - ink) * y
    return X * (1 - float(p.get("dim", 0.0)))


def _hex(value: str) -> tuple[float, float, float]:
    v = value.lstrip("#")
    return int(v[0:2], 16) / 255, int(v[2:4], 16) / 255, int(v[4:6], 16) / 255


# --- links -------------------------------------------------------------------------------------


def _circumcircle(pts: Any, tri: Any) -> tuple[Any, Any, Any]:
    """Centres (cx, cy) and squared radii of triangles ``tri`` (m, 3) over ``pts``."""
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    d = 2 * (a[:, 0] * (b[:, 1] - c[:, 1]) + b[:, 0] * (c[:, 1] - a[:, 1])
             + c[:, 0] * (a[:, 1] - b[:, 1]))  # fmt: skip
    d = np.where(np.abs(d) < 1e-12, 1e-12, d)
    sa, sb, sc = (a * a).sum(1), (b * b).sum(1), (c * c).sum(1)
    cx = (sa * (b[:, 1] - c[:, 1]) + sb * (c[:, 1] - a[:, 1]) + sc * (a[:, 1] - b[:, 1])) / d
    cy = (sa * (c[:, 0] - b[:, 0]) + sb * (a[:, 0] - c[:, 0]) + sc * (b[:, 0] - a[:, 0])) / d
    return cx, cy, (a[:, 0] - cx) ** 2 + (a[:, 1] - cy) ** 2


def delaunay(xy: Any) -> tuple[Any, Any]:
    """Our own Bowyer–Watson: (triangles (m, 3), edges (k, 2) with i < j, sorted) of ``xy``.
    Points are inserted in their given order; a super-triangle far outside holds them."""
    pts = np.asarray(xy, np.float64).reshape(-1, 2)
    n = len(pts)
    if n < 3:
        edges = np.asarray([[0, 1]] if n == 2 else [], np.int64).reshape(-1, 2)
        return np.zeros((0, 3), np.int64), edges
    lo, hi = pts.min(0), pts.max(0)
    span = float(max(hi - lo)) + 1.0
    mid = (lo + hi) / 2
    big = np.array([[mid[0] - 20 * span, mid[1] - span], [mid[0] + 20 * span, mid[1] - span],
                    [mid[0], mid[1] + 20 * span]])  # fmt: skip
    allp = np.vstack([pts, big])
    tri = np.array([[n, n + 1, n + 2]], np.int64)
    cx, cy, r2 = _circumcircle(allp, tri)
    for i in range(n):
        px, py = allp[i]
        bad = (px - cx) ** 2 + (py - cy) ** 2 < r2 * (1 - 1e-12)
        if not bad.any():
            continue
        edges = np.sort(np.concatenate([tri[bad][:, [0, 1]], tri[bad][:, [1, 2]],
                                        tri[bad][:, [2, 0]]]), 1)  # fmt: skip
        uniq, count = np.unique(edges, axis=0, return_counts=True)
        rim = uniq[count == 1]
        new = np.column_stack([rim, np.full(len(rim), i)])
        ncx, ncy, nr2 = _circumcircle(allp, new)
        tri = np.vstack([tri[~bad], new])
        cx, cy, r2 = (np.concatenate([v[~bad], nv]) for v, nv in ((cx, ncx), (cy, ncy), (r2, nr2)))
    tri = tri[(tri < n).all(1)]
    edges = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1)
    return tri, np.unique(edges, axis=0) if len(edges) else edges.reshape(-1, 2)


def support(fld: Field, xy: Any, pairs: Any, samples: int = 12) -> Any:
    """Per link, the share of points along it (ends left out) that lie on an edge (Sobel over
    half the field's threshold)."""
    if not len(pairs):
        return np.zeros(0)
    a, b = xy[pairs[:, 0]], xy[pairs[:, 1]]
    k = (np.arange(samples) + 1) / (samples + 1)
    px = a[:, 0, None] + (b[:, 0] - a[:, 0])[:, None] * k[None]
    py = a[:, 1, None] + (b[:, 1] - a[:, 1])[:, None] * k[None]
    return (fld.at(fld.mag, px, py) > 0.5 * fld.threshold).mean(1)


def tangents(fld: Field, xy: Any) -> Any:
    """Unit vectors along the edge at each place (across the gradient)."""
    gx, gy = fld.at(fld.gx, xy[:, 0], xy[:, 1]), fld.at(fld.gy, xy[:, 0], xy[:, 1])
    norm = np.hypot(gx, gy) + 1e-9
    return np.stack([-gy / norm, gx / norm], 1)


def links(fld: Field, layer: Layer, kind: str, reach: float, need: float) -> Any:
    """Index pairs (k, 2) of the links of ``kind`` between the layer's nodes, none longer than
    ``reach`` px (``radial``: 1.5 ×), kept when at least ``need`` of each lies on an edge:

    - ``near``: each node to its two nearest; ``delaunay``: the triangulation's edges;
    - ``chain``: each node to its nearest neighbour lying along its edge (|cos| > 0.85), so
      chains follow the contours; ``radial``: each anchor to its eight nearest nodes;
    - ``orthogonal``: as ``near`` (drawn as L elbows); ``mixed``: every other Delaunay edge,
      the chains and the first anchor's rays."""
    xy = layer.xy
    n = len(xy)
    if n < 2 or kind == "none":
        return np.zeros((0, 2), np.int64)
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    np.fill_diagonal(d, np.inf)
    out: list[Any] = []
    if kind in ("near", "orthogonal"):
        nearest = np.argsort(d, 1, kind="stable")[:, :2]
        out.append(np.column_stack([np.repeat(np.arange(n), nearest.shape[1]), nearest.ravel()]))
    if kind == "delaunay":
        out.append(delaunay(xy)[1])
    if kind == "mixed":
        out.append(delaunay(xy)[1][::2])
    if kind in ("chain", "mixed"):
        tan = tangents(fld, xy)
        rel = xy[None, :, :] - xy[:, None, :]
        cos = np.abs((rel * tan[:, None, :]).sum(2)) / (d + 1e-9)
        along = np.where(cos > 0.85, d, np.inf)
        best = np.argmin(along, 1)
        ok = np.isfinite(along[np.arange(n), best])
        out.append(np.column_stack([np.arange(n)[ok], best[ok]]))
    if kind in ("radial", "mixed"):
        hubs = layer.anchors if kind == "radial" else layer.anchors[:1]
        for hub in hubs:
            near = [j for j in np.argsort(d[hub], kind="stable")[:8] if d[hub, j] <= 1.5 * reach]
            out.extend(np.array([[hub, j]]) for j in near)
    pairs = np.concatenate(out) if out else np.zeros((0, 2), np.int64)
    pairs = np.unique(np.sort(pairs.astype(np.int64), 1), axis=0) if len(pairs) else pairs
    limit = reach * (1.5 if kind == "radial" else 1.0)
    length = d[pairs[:, 0], pairs[:, 1]] if len(pairs) else np.zeros(0)
    keep = length <= (limit if kind != "mixed" else 1.5 * reach)
    pairs = pairs[keep]
    if need > 0 and len(pairs):
        pairs = pairs[support(fld, xy, pairs) >= need]
    return pairs


def emptiest(xy: Any, at: Any, reach: float, size: tuple[int, int], tries: int = 16) -> Any:
    """The unit direction from ``at`` with the fewest nodes in a 45° cone within ``reach``
    (directions leaving the frame within ``reach`` lose), the first of equals."""
    h, w = size
    rel = xy - at
    dist = np.hypot(rel[:, 0], rel[:, 1])
    near = (dist > 1e-6) & (dist <= reach)
    best, out = math.inf, np.array([1.0, 0.0])
    for k in range(tries):
        th = 2 * math.pi * k / tries
        u = np.array([math.cos(th), math.sin(th)])
        cos = (rel[near] @ u) / dist[near] if near.any() else np.zeros(0)
        count = float((cos > math.cos(math.pi / 8)).sum())
        end = at + u * reach
        if not (0 <= end[0] <= w - 1 and 0 <= end[1] <= h - 1):
            count += 1000
        if count < best:
            best, out = count, u
    return out


# --- contours ----------------------------------------------------------------------------------

# Marching squares: corners v0 top-left, v1 top-right, v2 bottom-right, v3 bottom-left (bit i
# set when corner i is above the level); edges 0 top, 1 right, 2 bottom, 3 left. The saddles (5,
# 10) are split by the cell's centre value: ``SADDLE[case][centre above]``.
CASES: dict[int, tuple[tuple[int, int], ...]] = {
    1: ((3, 0),), 2: ((0, 1),), 3: ((3, 1),), 4: ((1, 2),), 6: ((0, 2),), 7: ((3, 2),),
    8: ((2, 3),), 9: ((0, 2),), 11: ((1, 2),), 12: ((1, 3),), 13: ((0, 1),), 14: ((3, 0),),
}  # fmt: skip
SADDLE = {5: {True: ((0, 1), (2, 3)), False: ((3, 0), (1, 2))},
          10: {True: ((3, 0), (1, 2)), False: ((0, 1), (2, 3))}}  # fmt: skip


def iso_segments(v: Any, level: float) -> tuple[Any, Any]:
    """The ``level`` line of grid ``v`` (h, w) as segments (a, b), each (k, 2) [x, y] in grid
    px. Neighbouring cells compute a shared edge's point from the same two values, so the
    segments meet exactly (no gaps, no T-joints)."""
    v0, v1, v2, v3 = v[:-1, :-1], v[:-1, 1:], v[1:, 1:], v[1:, :-1]
    case = (v0 > level) * 1 + (v1 > level) * 2 + (v2 > level) * 4 + (v3 > level) * 8
    centre = (v0 + v1 + v2 + v3) / 4 > level
    ys, xs = np.mgrid[0 : v0.shape[0], 0 : v0.shape[1]].astype(np.float64)

    def cross(a: Any, b: Any) -> Any:
        d = b - a
        return np.where(np.abs(d) > 1e-12, (level - a) / np.where(np.abs(d) > 1e-12, d, 1.0), 0.5)

    points = (
        np.stack([xs + cross(v0, v1), ys], -1),  # top
        np.stack([xs + 1, ys + cross(v1, v2)], -1),  # right
        np.stack([xs + cross(v3, v2), ys + 1], -1),  # bottom
        np.stack([xs, ys + cross(v0, v3)], -1),  # left
    )
    out_a, out_b = [], []
    for c in range(1, 15):
        if c in SADDLE:
            for above in (True, False):
                sel = (case == c) & (centre == above)
                for e0, e1 in SADDLE[c][above]:
                    out_a.append(points[e0][sel])
                    out_b.append(points[e1][sel])
            continue
        sel = case == c
        for e0, e1 in CASES[c]:
            out_a.append(points[e0][sel])
            out_b.append(points[e1][sel])
    a, b = np.concatenate(out_a).reshape(-1, 2), np.concatenate(out_b).reshape(-1, 2)
    # A corner exactly on the level gives zero-length pieces in the cells round it: dropped, so
    # the real segments through that corner meet in pairs as everywhere else.
    real = np.abs(a - b).max(1) > 1e-12
    return a[real], b[real]


def streamlines(fld: Field, seeds: Any, spacing: float, steps: int = 400) -> list[Any]:
    """Lines along the iso-luma tangent (across the gradient) from ``seeds`` (frame px), both
    ways, RK2 in steps of a third of ``spacing``; a line stops on a weak gradient, at the frame's
    edge or after ``steps``. Taken in seed order, a line is kept when most of it runs where no
    kept line passed within ``spacing`` / 2 (an occupancy grid): evenly spaced lines."""
    h, w = fld.size
    seeds = np.asarray(seeds, np.float64).reshape(-1, 2)
    if not len(seeds):
        return []
    step = spacing / 3

    def tangent(p: Any) -> Any:
        return tangents(fld, p)

    halves = []
    for sign in (1.0, -1.0):
        pos = seeds.copy()
        alive = np.ones(len(pos), bool)
        trail = [pos.copy()]
        heading = tangent(pos) * sign
        for _ in range(steps // 2):
            t1 = tangent(pos)
            t1 *= np.sign((t1 * heading).sum(1, keepdims=True) + 1e-12)  # keep going one way
            mid = pos + t1 * step / 2
            t2 = tangent(mid)
            t2 *= np.sign((t2 * t1).sum(1, keepdims=True) + 1e-12)
            nxt = pos + t2 * step
            weak = fld.at(fld.mag, nxt[:, 0], nxt[:, 1]) < 0.5 * fld.threshold
            out = (nxt[:, 0] < 0) | (nxt[:, 0] > w - 1) | (nxt[:, 1] < 0) | (nxt[:, 1] > h - 1)
            alive &= ~(weak | out)
            pos = np.where(alive[:, None], nxt, pos)
            heading = np.where(alive[:, None], t2, heading)
            trail.append(np.where(alive[:, None], pos, np.nan))
            if not alive.any():
                break
        halves.append(np.stack(trail, 1))  # (seeds, steps, 2)
    cell = max(1.0, spacing / 2)
    gw, gh = int(w / cell) + 1, int(h / cell) + 1
    taken = np.zeros((gh, gw), bool)
    lines = []
    for i in range(len(seeds)):
        back, fwd = halves[1][i][::-1], halves[0][i][1:]
        line = np.vstack([back, fwd])
        line = line[~np.isnan(line[:, 0])]
        if len(line) < 4:
            continue
        gx = np.clip((line[:, 0] / cell).astype(np.int64), 0, gw - 1)
        gy = np.clip((line[:, 1] / cell).astype(np.int64), 0, gh - 1)
        if taken[gy, gx].mean() > 0.3:
            continue
        lines.append(line)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                taken[np.clip(gy + dy, 0, gh - 1), np.clip(gx + dx, 0, gw - 1)] = True
    return lines


def circle_chain(at: Any, u: Any, r0: float, count: int) -> tuple[Any, Any]:
    """Overlapping circles shrinking by 0.78 from ``at`` along ``u`` (centres (count, 2), radii)
    and where each meets the next ((m, 2) points)."""
    radii = r0 * 0.78 ** np.arange(count)
    centres = [np.asarray(at, np.float64)]
    for i in range(1, count):
        centres.append(centres[-1] + u * 0.7 * (radii[i - 1] + radii[i]))
    centres = np.asarray(centres).reshape(-1, 2)
    meet = []
    for i in range(count - 1):
        c1, c2, r1, r2 = centres[i], centres[i + 1], radii[i], radii[i + 1]
        d = float(np.hypot(*(c2 - c1)))
        a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
        hgt = math.sqrt(max(r1 * r1 - a * a, 0.0))
        e = (c2 - c1) / d
        nrm = np.array([-e[1], e[0]])
        meet += [c1 + a * e + hgt * nrm, c1 + a * e - hgt * nrm]
    return centres, np.asarray(meet).reshape(-1, 2)
