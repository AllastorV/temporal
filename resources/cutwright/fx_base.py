"""The frame kernels' shared base: the kernel registries (``KERNELS``, ``PIXELS``, ``REGION``,
``@kernel``, ``@pixel``), ``Frame``, the backends (``NumpyOps``, ``TorchOps``) and the operations
written once for both (blur, sampling, blend modes …). A leaf module: it imports no kernel file,
so the kernel files (``fx_kernels``, ``fx_kernels_art``, ``fx_kernels_paint``,
``fx_kernels_stylize``) import it in any order, in Cutwright's process (package names) and in the
objects environment (bare names next to the runner). ``fx_kernels`` re-exports all of it."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

KERNELS: dict[str, Callable[..., Any]] = {}
PIXELS: dict[str, Callable[..., Any]] = {}
REGION: set[str] = set()  # kernels that use the scope mask themselves (not blended by it)
LUMA = (0.2126, 0.7152, 0.0722)


def kernel(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def add(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in KERNELS:
            raise ValueError(f"fx kernel '{name}' is defined twice")
        KERNELS[name] = fn
        return fn

    return add


def pixel(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """A per-pixel colour function ``fn(xp, rgb, p) -> rgb`` on (3, …) arrays; it is also the
    effect's kernel."""

    def add(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in PIXELS:
            raise ValueError(f"fx pixel function '{name}' is defined twice")
        PIXELS[name] = fn
        kernel(name)(lambda o, x, p, f: fn(o.xp, x, p).clip(0, 1))
        return fn

    return add


@dataclass(frozen=True)
class Frame:
    index: int
    fps: float
    seed: int
    state: dict[str, Any] = field(default_factory=dict)
    mask: Any = None
    camera: Any = None  # [turn, x, y] px: this frame's place on the camera path (fx.camera)
    alpha: Any = None


def popen_pipes(cmd: list[str], *, stdin: bool = False, stdout: bool = False,
                size: int = 1 << 20, **kwargs: Any) -> Any:  # fmt: skip
    """``subprocess.Popen`` whose stdin / stdout pipe (``True``) has a ``size``-byte buffer. A
    Windows pipe holds 4 KB by default and a 4K 16-bit frame is 50 MB: three such hops moved
    ~300 MB/s (6 frames/s); a 1 MB buffer moves ~1.9 GB/s."""
    import os
    import subprocess

    if os.name == "nt":
        import _winapi
        import msvcrt

        def make() -> tuple[int, int]:
            read, write = _winapi.CreatePipe(None, size)
            return msvcrt.open_osfhandle(read, os.O_RDONLY), msvcrt.open_osfhandle(write, 0)
    else:
        make = os.pipe
    out = make() if stdout else None
    inp = make() if stdin else None
    if out:
        kwargs["stdout"] = out[1]
    if inp:
        kwargs["stdin"] = inp[0]
    try:
        if __package__ and __package__.startswith("cutwright."):
            from cutwright.core.resources import popen

            proc = popen(cmd, **kwargs)
        else:
            threads = os.environ.get("CUTWRIGHT_CORE_THREADS")
            if threads and str(cmd[0]).lower().endswith(("ffmpeg", "ffmpeg.exe")):
                cmd = [
                    cmd[0],
                    "-filter_threads",
                    threads,
                    "-filter_complex_threads",
                    threads,
                    *cmd[1:-1],
                    "-threads",
                    threads,
                    cmd[-1],
                ]
            proc = subprocess.Popen(cmd, **kwargs)
    except BaseException:
        for fd in (*(out or ()), *(inp or ())):
            os.close(fd)
        raise
    if out:
        os.close(out[1])  # (the child has its copy)
        proc.stdout = os.fdopen(out[0], "rb")
    if inp:
        os.close(inp[0])
        proc.stdin = os.fdopen(inp[1], "wb")
    return proc


# --- backends ----------------------------------------------------------------------------------


class NumpyOps:
    """numpy, in Cutwright's own process (no GPU; the reference the torch backend matches)."""

    xp = np

    def asarray(self, a: Any) -> Any:
        return np.asarray(a, dtype=np.float32)

    def numpy(self, x: Any) -> Any:
        return np.asarray(x)

    def load_frame(
        self, raw: bytes, channels: int, h: int, w: int, *, host_rounding: bool = False
    ) -> tuple[Any, Any]:
        """A raw planar 16-bit G, B, R (A) frame as (RGB float 0–1, coverage or None)."""
        planes = np.frombuffer(raw, "<u2").reshape(channels, h, w)
        coverage = self.asarray(planes[3].astype(np.float32) / 65535) if channels == 4 else None
        return self.asarray(planes[[2, 0, 1]].astype(np.float32) / 65535), coverage

    def dump_frame(self, x: Any, coverage: Any) -> bytes:
        """The inverse of ``load_frame``."""
        rgb = (np.clip(self.numpy(x), 0, 1) * 65535 + 0.5).astype("<u2")
        out = [rgb[1], rgb[2], rgb[0]]
        if coverage is not None:
            out.append((np.clip(self.numpy(coverage), 0, 1) * 65535 + 0.5).astype("<u2"))
        return np.stack(out).tobytes()

    def edge_taps(self, x: Any, taps: dict[tuple[int, int], float]) -> Any:
        """``sum(w * x[clip(y + dy), clip(x + dx)])`` over ``taps`` {(dy, dx): w}: whole-pixel
        reads with the edge held (the bilinear samples of a constant offset, added up)."""
        _, h, w = x.shape
        lo_y, hi_y = min(0, *(t[0] for t in taps)), max(0, *(t[0] for t in taps))
        lo_x, hi_x = min(0, *(t[1] for t in taps)), max(0, *(t[1] for t in taps))
        padded = np.pad(x, ((0, 0), (-lo_y, hi_y), (-lo_x, hi_x)), mode="edge")
        out = x * 0
        for (dy, dx), weight in taps.items():
            y0, x0 = dy - lo_y, dx - lo_x
            out = out + padded[:, y0 : y0 + h, x0 : x0 + w] * np.float32(weight)
        return out

    def pad(self, x: Any, py: int, px: int) -> Any:
        return np.pad(x, ((0, 0), (py, py), (px, px)), mode="reflect")

    def fftfreq(self, n: int) -> Any:
        return np.fft.fftfreq(n).astype(np.float32)

    def rfftfreq(self, n: int) -> Any:
        return np.fft.rfftfreq(n).astype(np.float32)

    def fft_blur(self, x: Any, g: Any, shape: tuple[int, int]) -> Any:
        return np.fft.irfft2(np.fft.rfft2(x) * g, s=shape).astype(np.float32)

    def gather(self, x: Any, iy: Any, ix: Any) -> Any:
        return x[:, iy.astype(np.int64), ix.astype(np.int64)]

    def grid(self, h: int, w: int) -> tuple[Any, Any]:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        return ys, xs

    def argsort(self, key: Any) -> Any:
        """Stable, along the last axis."""
        return np.argsort(key, axis=-1, kind="stable")

    def cumsum(self, x: Any) -> Any:
        return np.cumsum(x, axis=-1)

    def take(self, x: Any, idx: Any) -> Any:
        """``x`` (C, H, W) reordered along the last axis by ``idx`` (H, W)."""
        return np.take_along_axis(x, idx[None], axis=-1)


class TorchOps:
    """torch on ``device`` (CUDA when present)."""

    def __init__(self, device: str = "cuda") -> None:
        import torch

        self.xp = torch
        self.device = device

    def asarray(self, a: Any) -> Any:
        return self.xp.as_tensor(np.asarray(a, dtype=np.float32), device=self.device)

    def numpy(self, x: Any) -> Any:
        return x.detach().float().cpu().numpy()

    def load_frame(
        self, raw: bytes, channels: int, h: int, w: int, *, host_rounding: bool = False
    ) -> tuple[Any, Any]:
        """As NumpyOps'; the 16-bit planes go up as they are and are converted on the device."""
        import warnings

        with warnings.catch_warnings():  # (the bytes are read-only; the copy to the device is made)
            warnings.simplefilter("ignore")
            raw16 = self.xp.as_tensor(np.frombuffer(raw, "<i2").reshape(channels, h, w),
                                      device=self.device)  # fmt: skip
        planes = raw16.to(self.xp.int32) & 0xFFFF
        if host_rounding:
            # Match numpy's correctly rounded float32 division before upload. CUDA's
            # scalar float division uses a rounded reciprocal (512/65536 inputs differ).
            normalized = (planes.double() / 65535).float()
            return normalized[[2, 0, 1]], normalized[3] if channels == 4 else None
        coverage = planes[3].float() / 65535 if channels == 4 else None
        return planes[[2, 0, 1]].float() / 65535, coverage

    def dump_frame(self, x: Any, coverage: Any) -> bytes:
        """As NumpyOps'; packed to 16 bits on the device (a NaN is 0)."""
        torch = self.xp
        planes = [x[1], x[2], x[0]] + ([coverage] if coverage is not None else [])
        stacked = torch.nan_to_num(torch.stack(planes).float()).clamp(0, 1)
        v = (stacked * 65535 + 0.5).to(torch.int32)
        return torch.where(v >= 32768, v - 65536, v).to(torch.int16).cpu().numpy().tobytes()

    def edge_taps(self, x: Any, taps: dict[tuple[int, int], float]) -> Any:
        """As NumpyOps': one depthwise convolution of the edge-held frame by the taps."""
        import torch.nn.functional as F

        lo_y, hi_y = min(0, *(t[0] for t in taps)), max(0, *(t[0] for t in taps))
        lo_x, hi_x = min(0, *(t[1] for t in taps)), max(0, *(t[1] for t in taps))
        kernel = self.xp.zeros((hi_y - lo_y + 1, hi_x - lo_x + 1), dtype=self.xp.float32)
        for (dy, dx), weight in taps.items():
            kernel[dy - lo_y, dx - lo_x] += weight
        c = x.shape[0]
        padded = F.pad(x[None], (-lo_x, hi_x, -lo_y, hi_y), mode="replicate")
        return F.conv2d(padded, kernel.to(self.device)[None, None].repeat(c, 1, 1, 1), groups=c)[0]

    def sample_bilinear(self, x: Any, gy: Any, gx: Any, border: str) -> Any:
        """``sample`` in one ``grid_sample`` (bilinear, the edge held or 0 outside): None for
        shapes it does not cover (the caller then gathers)."""
        import torch.nn.functional as F

        _, h, w = x.shape
        if not (hasattr(gy, "dim") and hasattr(gx, "dim")) or gy.dim() != 2 or gx.dim() != 2:
            return None
        if min(h, w) < 2:
            return None
        gy, gx = self.xp.broadcast_tensors(gy, gx)
        grid = self.xp.stack([gx * (2.0 / (w - 1)) - 1.0, gy * (2.0 / (h - 1)) - 1.0], dim=-1)
        return F.grid_sample(x[None], grid[None], mode="bilinear", align_corners=True,
                             padding_mode="zeros" if border == "zero" else "border")[0]  # fmt: skip

    def pad(self, x: Any, py: int, px: int) -> Any:
        import torch.nn.functional as F

        return F.pad(x[None], (px, px, py, py), mode="reflect")[0]

    def fftfreq(self, n: int) -> Any:
        return self.xp.fft.fftfreq(n, device=self.device)

    def rfftfreq(self, n: int) -> Any:
        return self.xp.fft.rfftfreq(n, device=self.device)

    def fft_blur(self, x: Any, g: Any, shape: tuple[int, int]) -> Any:
        if x.is_cuda:  # a plan per padded size: an animated radius must not keep thousands
            cache = self.xp.backends.cuda.cufft_plan_cache[x.device.index]
            if cache.max_size > 64:
                cache.max_size = 64
        fft = self.xp.fft
        return fft.irfft2(fft.rfft2(x) * g, s=shape)

    def gather(self, x: Any, iy: Any, ix: Any) -> Any:
        return x[:, iy.long(), ix.long()]

    def grid(self, h: int, w: int) -> tuple[Any, Any]:
        torch = self.xp
        ys = torch.arange(h, device=self.device, dtype=torch.float32)[:, None].expand(h, w)
        xs = torch.arange(w, device=self.device, dtype=torch.float32)[None, :].expand(h, w)
        return ys, xs

    def argsort(self, key: Any) -> Any:
        return self.xp.argsort(key, dim=-1, stable=True)

    def cumsum(self, x: Any) -> Any:
        return self.xp.cumsum(x, dim=-1)

    def take(self, x: Any, idx: Any) -> Any:
        return self.xp.gather(x, -1, idx[None].expand(x.shape[0], -1, -1))


# --- shared operations (written once for both backends) ---------------------------------------


def blur(o: Any, x: Any, sigma: float) -> Any:
    """Gaussian blur of each channel (sigma in px): the exact Gaussian in the frequency domain,
    the frame mirrored at its edges."""
    if sigma < 0.3:
        return x
    native = getattr(o, "gaussian_blur", None)
    if native is not None:
        return native(x, sigma)
    _, h, w = x.shape
    py, px = min(h - 1, math.ceil(3 * sigma)), min(w - 1, math.ceil(3 * sigma))
    padded = o.pad(x, py, px)
    hh, ww = padded.shape[-2:]
    fy, fx = o.fftfreq(hh), o.rfftfreq(ww)
    g = o.xp.exp(-2 * math.pi**2 * sigma**2 * (fy[:, None] ** 2 + fx[None, :] ** 2))
    return o.fft_blur(padded, g, (hh, ww))[:, py : py + h, px : px + w]


def hblur(o: Any, x: Any, sigma: float) -> Any:
    """Gaussian blur along rows only (sigma in px), the frame mirrored at its sides."""
    if sigma < 0.3:
        return x
    native = getattr(o, "horizontal_blur", None)
    if native is not None:
        return native(x, sigma)
    w = x.shape[-1]
    px = min(w - 1, math.ceil(3 * sigma))
    padded = o.pad(x, 0, px)
    hh, ww = padded.shape[-2:]
    g = o.xp.exp(-2 * math.pi**2 * sigma**2 * o.rfftfreq(ww) ** 2)[None, :]
    return o.fft_blur(padded, g, (hh, ww))[:, :, px : px + w]


def sample(o: Any, x: Any, gy: Any, gx: Any, border: str = "clamp") -> Any:
    """Each channel of ``x`` read at pixel positions (gy, gx) with bilinear weights; outside
    the frame the edge (``clamp``) or 0 (``zero``)."""
    fast = getattr(o, "sample_bilinear", None)
    if fast is not None and (out := fast(x, gy, gx, border)) is not None:
        return out
    _, h, w = x.shape
    xp = o.xp
    y0, x0 = xp.floor(gy), xp.floor(gx)
    wy, wx = gy - y0, gx - x0
    out = 0
    for dy, dx, k in ((0, 0, (1 - wy) * (1 - wx)), (0, 1, (1 - wy) * wx),
                      (1, 0, wy * (1 - wx)), (1, 1, wy * wx)):  # fmt: skip
        yy, xx = y0 + dy, x0 + dx
        v = o.gather(x, yy.clip(0, h - 1), xx.clip(0, w - 1))
        if border == "zero":
            inside = (yy >= 0) & (yy <= h - 1) & (xx >= 0) & (xx <= w - 1)
            v = v * inside
        out = out + v * k
    return out


def luma(x: Any) -> Any:
    return LUMA[0] * x[0] + LUMA[1] * x[1] + LUMA[2] * x[2]


def screen(a: Any, b: Any) -> Any:
    return 1 - (1 - a) * (1 - b)


def _soft_light(xp: Any, a: Any, b: Any) -> Any:
    """W3C soft light of ``b`` over ``a``."""
    d = xp.where(a > 0.25, xp.sqrt(a.clip(0, None)), ((16 * a - 12) * a + 4) * a)
    return xp.where(b <= 0.5, a - (1 - 2 * b) * a * (1 - a), a + (2 * b - 1) * (d - a))


def _dodge(xp: Any, a: Any, b: Any) -> Any:
    """W3C colour dodge: black stays, a white top makes any lit base white."""
    return xp.where(b >= 1, (a > 0) * 1.0, (a / (1 - b).clip(1e-6, None)).clip(None, 1))


def _burn(xp: Any, a: Any, b: Any) -> Any:
    """W3C colour burn: white stays, a black top makes any unlit base black."""
    return xp.where(a >= 1, 1.0, xp.where(b <= 0, 0.0, 1 - ((1 - a) / b.clip(1e-6, None)).clip(
        None, 1)))  # fmt: skip


def _luminosity(xp: Any, a: Any, b: Any) -> Any:
    """W3C luminosity: the base's hue and saturation with the top's luma (Rec.709 weights),
    clipped into range along the grey axis (``ClipColor``). Works on the whole (3, …) picture."""

    def lum(c: Any) -> Any:
        return LUMA[0] * c[0] + LUMA[1] * c[1] + LUMA[2] * c[2]

    c = a + (lum(b) - lum(a))[None]
    y = lum(c)[None]
    n = xp.minimum(xp.minimum(c[0], c[1]), c[2])[None]
    x = xp.maximum(xp.maximum(c[0], c[1]), c[2])[None]
    c = xp.where(n < 0, y + (c - y) * y / (y - n).clip(1e-6, None), c)
    return xp.where(x > 1, y + (c - y) * (1 - y) / (x - y).clip(1e-6, None), c)


BLENDS: dict[str, Callable[[Any, Any, Any], Any]] = {
    "normal": lambda xp, a, b: b,
    "multiply": lambda xp, a, b: a * b,
    "screen": lambda xp, a, b: screen(a, b),
    "overlay": lambda xp, a, b: xp.where(a < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b)),
    "soft_light": _soft_light,
    "add": lambda xp, a, b: (a + b).clip(None, 1),
    "lighten": lambda xp, a, b: xp.maximum(a, b),
    "darken": lambda xp, a, b: xp.minimum(a, b),
    "color_dodge": _dodge,
    "color_burn": _burn,
    "difference": lambda xp, a, b: xp.abs(a - b),
    "exclusion": lambda xp, a, b: a + b - 2 * a * b,
    "luminosity": _luminosity,  # not per channel: the ffmpeg path sends it to the frame processor
}


def blend(o: Any, base: Any, top: Any, mode: str, opacity: float) -> Any:
    """The effect's picture ``top`` over the picture it was given ``base`` by ``mode`` (the
    base decides overlay, as in a photo editor), mixed in by ``opacity``."""
    if mode == "normal" and opacity >= 1:
        return top
    return base + (BLENDS[mode](o.xp, base, top) - base) * opacity


def hex_rgb(value: str) -> tuple[float, float, float]:
    v = value.lstrip("#")
    return int(v[0:2], 16) / 255, int(v[2:4], 16) / 255, int(v[4:6], 16) / 255


def hue_rgb(degrees: float) -> tuple[float, float, float]:
    """The fully saturated colour of a hue (HSV, value 1)."""
    h = (degrees % 360) / 60
    c, x = 1.0, 1 - abs(h % 2 - 1)
    return [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)][int(h) % 6]


def shifted(o: Any, x: Any, ox: float, oy: float) -> Any:
    """``x`` moved by (ox, oy) px (bilinear), 0 where nothing moved in."""
    _, h, w = x.shape
    ys, xs = o.grid(h, w)
    return sample(o, x, ys - oy, xs - ox, "zero")


def downscale(o: Any, x: Any, q: int) -> Any:
    """Area average over q × q blocks (the frame cropped to a multiple of q)."""
    if q <= 1:
        return x
    c, h, w = x.shape
    h, w = h // q * q, w // q * q
    return x[:, :h, :w].reshape(c, h // q, q, w // q, q).mean((2, 4))


def upscale(o: Any, x: Any, h: int, w: int) -> Any:
    """Bilinear back to h × w."""
    _, sh, sw = x.shape
    if (sh, sw) == (h, w):
        return x
    ys, xs = o.grid(h, w)
    return sample(o, x, (ys + 0.5) * sh / h - 0.5, (xs + 0.5) * sw / w - 0.5)


def _ray(o: Any, s: Any, dx: float, dy: float, length: float) -> Any:
    """``s`` smeared one way along (dx, dy), falling off exponentially (e^-3 at ``length``):
    the sum of r^k shift(k) for k < 2^n, built in n doubling steps, on until the tail is
    faint (no visible end)."""
    r = math.exp(-3 / max(length, 1.0))
    out, span = s, 1
    while span < 2.5 * length:
        out = out + r**span * shifted(o, out, dx * span, dy * span)
        span *= 2
    return out


def peak(o: Any, x: Any) -> Any:
    """The brightest channel of each pixel."""
    return o.xp.maximum(o.xp.maximum(x[0], x[1]), x[2])
