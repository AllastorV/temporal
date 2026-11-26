"""Noise reduction of a clip's frames: motion-compensated temporal averaging, then non-local
means. Written once for both backends (numpy here, torch in the objects environment), with the
frame processor's job protocol (``fx_runner`` hands it jobs of ``kind: "denoise"``). Imports
only the leaf ``fx_base`` and the optical flow of ``fx_kernels_paint``.

- **Noise level** ``noise``: per frame, from its luma alone (the median absolute diagonal Haar
  detail / 0.6745). Strengths are relative to it, so "60" cleans faint grain and heavy noise
  alike and leaves a clean picture alone.
- **Temporal** ``temporal``: every neighbour (up to ``radius`` frames each way) is warped onto
  this frame along the optical flow (``fx_kernels_paint.flow``: pyramidal Lucas–Kanade, still
  where the picture is flat or only noisy) and weighed per pixel by how well its 5 × 5 patch
  agrees with this one's: where the flow is wrong (an occlusion, a moving edge the coarse flow
  smears) the neighbour drops out — less cleaning there, never a ghost.
- **Spatial** ``_nlm``: non-local means over a 7 × 7 search window with 5 × 5 patches (all three
  planes), separate strengths for luma and chroma.

Weights are ``exp(−d / s)`` with ``d`` the patch distance in units of the noise's expected
distance (2σ²) and ``s`` = strength × ``GAIN`` / 100: patches that differ only by noise weigh
``exp(−1 / s)`` (0.69 at strength 60), real structure ~0. Frames are planes in 0–1: Y'CbCr
(luma first) or, with ``rgb``, R'G'B' worked on in an opponent space (BT.709 luma, B − Y,
R − Y). A frame's result depends only on the frames of its window: a still, a range render and
the full render agree.
"""

from __future__ import annotations

import contextlib
import queue
import subprocess
import threading
from collections.abc import Iterable
from typing import Any

import numpy as np

if __package__:  # in Cutwright's process
    from cutwright.modules.fx.runners.fx_base import (
        LUMA,
        blur,
        downscale,
        popen_pipes,
        sample,
        upscale,
    )
    from cutwright.modules.fx.runners.fx_kernels_paint import DEAD_ZONE, FLOW_SIZE, _grad
else:  # in the objects environment: the files next to the runner, by their bare names
    from fx_base import LUMA, blur, downscale, popen_pipes, sample, upscale  # type: ignore[no-redef]
    from fx_kernels_paint import DEAD_ZONE, FLOW_SIZE, _grad  # type: ignore[no-redef]

PATCH = 2  # patches of (2·PATCH + 1)² pixels
SEARCH = 3  # the spatial search window: (2·SEARCH + 1)² offsets
GAIN = 4.5  # strength 100 → s = 4.5: noise-only patches weigh exp(−1/4.5) = 0.8
FLOOR = 1 / 1023  # the smallest noise level assumed (a 10-bit step): a clean frame stays put
CB, CR = 1.8556, 1.5748  # BT.709: Cb = (B − Y) / 1.8556, Cr = (R − Y) / 1.5748


def box(o: Any, x: Any, r: int = PATCH) -> Any:
    """Mean of each (h, w) plane of ``x`` (c, h, w) over (2r + 1)² pixels, edges mirrored."""
    _, h, w = x.shape
    pad = o.pad(x, r, r)
    rows = pad[:, :, 0:w]
    for k in range(1, 2 * r + 1):
        rows = rows + pad[:, :, k : k + w]
    out = rows[:, 0:h]
    for k in range(1, 2 * r + 1):
        out = out + rows[:, k : k + h]
    return out / (2 * r + 1) ** 2


def noise(o: Any, y: Any) -> float:
    """The noise level (σ, 0–1) of a luma plane (h, w): the robust spread of its finest
    diagonal detail, which a smooth picture hardly has."""
    h, w = y.shape
    a = y[0 : h // 2 * 2 : 2, 0 : w // 2 * 2 : 2]
    b = y[0 : h // 2 * 2 : 2, 1 : w // 2 * 2 : 2]
    c = y[1 : h // 2 * 2 : 2, 0 : w // 2 * 2 : 2]
    d = y[1 : h // 2 * 2 : 2, 1 : w // 2 * 2 : 2]
    detail = np.abs(np.asarray(o.numpy(a - b - c + d))) * 0.5  # (the same median on both)
    return max(FLOOR, float(np.median(detail)) / 0.6745)


def _opponent(o: Any, x: Any) -> Any:
    y = LUMA[0] * x[0] + LUMA[1] * x[1] + LUMA[2] * x[2]
    return o.xp.stack([y, (x[2] - y) / CB + 0.5, (x[0] - y) / CR + 0.5])


def _rgb(o: Any, x: Any) -> Any:
    r = x[0] + (x[2] - 0.5) * CR
    b = x[0] + (x[1] - 0.5) * CB
    g = (x[0] - LUMA[0] * r - LUMA[2] * b) / LUMA[1]
    return o.xp.stack([r, g, b])


def _weight(o: Any, d: Any, strength: float) -> Any:
    return o.xp.exp(-d / (strength * GAIN / 100))


def _bsample(o: Any, x: Any, gy: Any, gx: Any) -> Any:
    """Plane i of ``x`` (n, h, w) read at its own positions ``gy[i]``, ``gx[i]`` (bilinear,
    edges clamped): the planes stacked into one tall image, read once."""
    n, h, w = x.shape
    tall = x.reshape(1, n * h, w)
    base = o.asarray(np.arange(n, dtype=np.float32) * h)[:, None, None]
    y0, x0 = o.xp.floor(gy), o.xp.floor(gx)
    wy, wx = gy - y0, gx - x0
    out = 0
    for dy, dx, k in ((0, 0, (1 - wy) * (1 - wx)), (0, 1, (1 - wy) * wx),
                      (1, 0, wy * (1 - wx)), (1, 1, wy * wx)):  # fmt: skip
        yy = (y0 + dy).clip(0, h - 1) + base
        out = out + o.gather(tall, yy, (x0 + dx).clip(0, w - 1))[0] * k
    return out


def flows(o: Any, cur: Any, others: Any) -> tuple[Any, Any]:
    """Backward optical flow (fy, fx), each (n, h, w) px, of luma ``cur`` (h, w) against each
    of ``others`` (n, h, w): ``cur(p) ≈ others[i](p + f_i(p))``. ``fx_kernels_paint.flow``
    (pyramidal Lucas–Kanade at ≤ ``FLOW_SIZE`` px, regularised, a dead zone) for all the
    neighbours in one pass (on the GPU the small levels cost launches, not pixels), each pair
    regularised by its own residual."""
    xp = o.xp
    n, h, w = others.shape
    q = max(1, -(-min(h, w) // FLOW_SIZE))
    pyramid = [(downscale(o, cur[None], q), downscale(o, others, q))]
    while min(pyramid[-1][0].shape[1:]) >= 48 and len(pyramid) < 4:
        a, b = pyramid[-1]
        pyramid.append((downscale(o, a, 2), downscale(o, b, 2)))
    fy = fx = None
    for a, b in reversed(pyramid):
        _, lh, lw = b.shape
        if fy is None:
            fy, fx = 0 * b, 0 * b
        else:
            sy, sx = lh / fy.shape[1], lw / fy.shape[2]
            fy, fx = upscale(o, fy, lh, lw) * sy, upscale(o, fx, lh, lw) * sx
        ys, xs = o.grid(lh, lw)
        for _ in range(3):
            warped = _bsample(o, b, ys + fy, xs + fx)
            gy, gx = _grad(o, (warped + a) * 0.5)
            e = a - warped
            terms = xp.stack([gx * gx, gx * gy, gy * gy, gx * e, gy * e]).reshape(5 * n, lh, lw)
            sxx, sxy, syy, sxe, sye = blur(o, terms, 3.0).reshape(5, n, lh, lw)
            reg = 1e-3 + 1.5 * (e * e).mean(-1).mean(-1)[:, None, None]  # noise: hold still
            sxx, syy = sxx + reg, syy + reg
            det = sxx * syy - sxy * sxy
            fx = fx + ((syy * sxe - sxy * sye) / det).clip(-2, 2)
            fy = fy + ((sxx * sye - sxy * sxe) / det).clip(-2, 2)
    _, lh, lw = fy.shape
    fy, fx = upscale(o, fy, h, w) * (h / lh), upscale(o, fx, h, w) * (w / lw)
    mag = xp.sqrt(fy * fy + fx * fx)
    keep = ((mag - DEAD_ZONE[0]) / (DEAD_ZONE[1] - DEAD_ZONE[0])).clip(0, 1)
    return fy * keep, fx * keep


def _temporal(
    o: Any, x: Any, others: list[Any], strength: float, sigma: float, motion_threshold: float = 50
) -> Any:
    if not others:
        return x
    _, h, w = x.shape
    ys, xs = o.grid(h, w)
    total, weights = x, 1.0
    scale = 1 / (2 * sigma * sigma * 3)  # the patch distance of pure noise: 1
    fy, fx = flows(o, x[0], o.xp.stack([f[0] for f in others]))
    for i, other in enumerate(others):
        warped = sample(o, other, ys + fy[i], xs + fx[i])
        d = box(o, ((warped - x) ** 2).sum(0)[None])[0] * scale
        k = _weight(o, d, strength)
        # Noise-only distances cluster around one. Reject disagreement beyond that noise
        # floor more conservatively when motion tolerance is low (occlusions / cuts).
        penalty = (d - 1.5).clip(0, None) * (1 - motion_threshold / 100) * 5
        k = k * o.xp.exp(-penalty)
        total = total + warped * k[None]
        weights = weights + k
    return total / weights[None]


def _nlm(o: Any, x: Any, luma: float, chroma: float, sigma: float) -> Any:
    """Non-local means. The patch distance of offset −d at p is that of +d at p − d: each
    pair of offsets costs one distance map, made on the frame grown by the search radius."""
    _, h, w = x.shape
    s = SEARCH
    pad = o.pad(x, 2 * s, 2 * s)
    grown = pad[:, s : s + h + 2 * s, s : s + w + 2 * s]
    scale = 1 / (2 * sigma * sigma * 3)
    acc_l, acc_c, sum_l, sum_c = x[0], x[1:], 1.0, 1.0
    half = [(dy, dx) for dy in range(0, s + 1) for dx in range(-s, s + 1) if (dy, dx) > (0, 0)]
    for dy, dx in half:
        there = pad[:, s + dy : s + dy + h + 2 * s, s + dx : s + dx + w + 2 * s]
        dist = box(o, ((grown - there) ** 2).sum(0)[None])[0] * scale
        for sy, sx in ((dy, dx), (-dy, -dx)):
            # +d: the map at p; −d: the map at p − d (the pair's other end).
            oy, ox = (s, s) if (sy, sx) == (dy, dx) else (s - dy, s - dx)
            d = dist[oy : oy + h, ox : ox + w]
            moved = pad[:, 2 * s + sy : 2 * s + sy + h, 2 * s + sx : 2 * s + sx + w]
            k = _weight(o, d, luma) if luma > 0 else None
            if k is not None:
                acc_l, sum_l = acc_l + moved[0] * k, sum_l + k
            if chroma > 0:
                k = k if chroma == luma else _weight(o, d, chroma)
                acc_c, sum_c = acc_c + moved[1:] * k[None], sum_c + k
    out_l = acc_l / sum_l if luma > 0 else x[0]
    out_c = acc_c / sum_c if chroma > 0 else x[1:]
    return o.xp.stack([out_l, out_c[0], out_c[1]])


def denoise(o: Any, window: list[Any], i: int, p: dict[str, Any]) -> Any:
    """Frame ``window[i]`` (3, h, w) cleaned with the other frames of ``window`` as its
    temporal neighbours. ``p``: ``luma``, ``chroma``, ``temporal`` (0–100), ``rgb``."""
    rgb = bool(p.get("rgb"))
    original = window[i]
    blend = float(p.get("blend", 0))
    if blend >= 1:
        return original.clip(0, 1)
    frames = [_opponent(o, f) if rgb else f for f in window]
    x = frames[i]
    if float(p.get("temporal", 0)) > 0:
        others = [f for j, f in enumerate(frames) if j != i]
        x = _temporal(
            o, x, others, float(p["temporal"]), noise(o, x[0]), float(p.get("motion_threshold", 50))
        )
    luma, chroma = float(p.get("luma", 0)), float(p.get("chroma", 0))
    if luma > 0 or chroma > 0:
        x = _nlm(o, x, luma, chroma, noise(o, x[0]))
    cleaned = _rgb(o, x) if rgb else x
    return (cleaned * (1 - blend) + original * blend).clip(0, 1)


def _drain(pipe: Any) -> Any:
    chunks: list[bytes] = []
    thread = threading.Thread(target=lambda: chunks.extend(iter(lambda: pipe.read(65536), b"")),
                              daemon=True)  # fmt: skip
    thread.start()

    def text() -> str:
        thread.join(timeout=30)
        return b"".join(chunks).decode(errors="replace")

    return text


def process(job: dict[str, Any], frames: Iterable[bytes], o: Any) -> dict[str, Any]:
    """The job's frames (``frames`` of them, planar 16-bit ``pix_fmt``) through ``denoise``;
    frames ``skip`` … ``skip + count − 1`` are written to ``job["out"]``, each with up to
    ``radius`` neighbours on each side among the frames given (the job reaches past its own
    range by ``radius`` frames: only the media's ends cut a window short). Reading, cleaning
    and writing overlap (a thread each side): at 1080p the pipes and conversions take about
    as long as the GPU's work. ``keep`` (ascending, repeats allowed) instead: only those frames
    are cleaned and written, at ``out_fps`` (the timeline's frames of a 50p clip on 25p)."""
    w, h, n = int(job["width"]), int(job["height"]), int(job["frames"])
    skip, count = int(job.get("skip", 0)), int(job.get("count", n))
    keep = job.get("keep")
    wanted: dict[int, int] = {}
    for i in keep or ():
        wanted[i] = wanted.get(i, 0) + 1
    if keep is not None:
        count = len(keep)
    p = {k: job[k] for k in ("luma", "chroma", "temporal", "rgb")}
    p |= {k: job[k] for k in ("blend", "motion_threshold") if k in job}
    r = int(job["radius"]) if float(p["temporal"]) > 0 else 0
    encoder = popen_pipes(
        [job["ffmpeg"], "-hide_banner", "-nostdin", "-v", "error", "-y", "-f", "rawvideo",
         "-pix_fmt", job["pix_fmt"], "-s", f"{w}x{h}",
         "-r", str(job.get("out_fps") or f"{float(job['fps']):.6f}"),
         "-i", "-", *job["out_args"], job["out"]],
        stdin=True, stderr=subprocess.PIPE,
    )  # fmt: skip
    log = _drain(encoder.stderr)
    order = [1, 2, 0] if p["rgb"] else [0, 1, 2]  # gbrp comes and goes as G, B, R
    back = [2, 0, 1] if p["rgb"] else [0, 1, 2]
    packed = str(getattr(o, "device", "")).startswith("cuda")
    given: queue.Queue = queue.Queue(maxsize=2)
    done: queue.Queue = queue.Queue(maxsize=2)
    failed: list[BaseException] = []
    stop = threading.Event()
    written, last = [0], [b""]

    def read() -> None:
        try:
            for raw in frames:
                if packed:
                    given.put(raw)  # Upload the original 16-bit bytes on the processing thread.
                else:
                    planes = np.frombuffer(raw, "<u2").reshape(3, h, w)[back]
                    given.put(planes.astype(np.float32) / 65535)
                if stop.is_set():  # the caller reads the rest of the job's frames
                    break
        except BaseException as exc:
            failed.append(exc)
        given.put(None)

    def write() -> None:
        broken = False
        while (out := done.get()) is not None:
            if failed:
                continue  # draining: the job fails
            try:
                if packed:
                    # Frame packing speaks GBR; a YUV job keeps its original plane order.
                    data = o.dump_frame(out if p["rgb"] else out[[2, 0, 1]], None)
                else:
                    data = (
                        (np.clip(o.numpy(out), 0, 1) * 65535 + 0.5).astype("<u2")[order].tobytes()
                    )
            except BaseException as exc:
                failed.append(exc)
                continue
            if not broken:
                try:
                    encoder.stdin.write(data)
                except OSError:
                    broken = True  # the encoder stopped: its stderr says why
            last[0] = data
            written[0] += 1

    reader, writer = threading.Thread(target=read, daemon=True), threading.Thread(target=write)
    reader.start()
    writer.start()
    ring: dict[int, Any] = {}

    def emit(i: int) -> None:
        times = wanted.get(i, 0) if keep is not None else int(skip <= i < skip + count)
        if times:
            near = [j for j in range(i - r, i + r + 1) if j in ring]
            out = denoise(o, [ring[j] for j in near], near.index(i), p)
            for _ in range(times):
                done.put(out)

    got = 0
    try:
        while not failed and (planes := given.get()) is not None:
            if packed:
                picture, _ = o.load_frame(planes, 3, h, w, host_rounding=True)
                ring[got] = picture if p["rgb"] else picture[[1, 2, 0]]
            else:
                ring[got] = o.asarray(planes)
            got += 1
            emit(got - 1 - r)  # the frame whose last neighbour just came
            ring.pop(got - 1 - 2 * r, None)  # no frame still to come needs it
        for i in range(max(0, got - r), got):  # the last frames: no neighbours after them
            emit(i)
    finally:
        stop.set()
        while reader.is_alive():  # it may wait to hand over a frame
            with contextlib.suppress(queue.Empty):
                given.get(timeout=0.05)
        done.put(None)
        writer.join()
    if failed:  # the encoder too (fx_runner.process does it in a finally): not left to the GC
        with contextlib.suppress(OSError, ValueError):
            encoder.kill()
            encoder.stdin.close()
        encoder.wait()
        raise failed[0]
    with contextlib.suppress(OSError):
        while written[0] and written[0] < count:  # the frames ran out: the copy holds the last
            encoder.stdin.write(last[0])
            written[0] += 1
        encoder.stdin.close()
    if encoder.wait() != 0 or not written[0]:
        raise RuntimeError("ffmpeg: " + (log().strip()[-400:] or "kare yazılamadı"))
    return {"frames": written[0], "device": getattr(o, "device", "numpy")}
