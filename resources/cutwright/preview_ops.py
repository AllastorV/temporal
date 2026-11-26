"""Optional OpenCV operations for the CPU-based Cutwright preview kernels."""
from functools import lru_cache

import cv2
import numpy as np

from fx_base import NumpyOps


class PreviewOps(NumpyOps):
    field_size = 160
    frame_key = None

    def __init__(self):
        self.fields = {}

    def field(self, compute):
        key = self.frame_key
        if key is None:
            return compute()
        if key not in self.fields:
            if len(self.fields) >= 10:
                del self.fields[next(iter(self.fields))]
            self.fields[key] = compute()
        return self.fields[key]

    @lru_cache(maxsize=3)
    def grid(self, h, w):
        y, x = np.indices((h, w), dtype=np.float32)
        y.setflags(write=False)
        x.setflags(write=False)
        return y, x

    def gaussian_blur(self, x, sigma):
        return np.stack([cv2.GaussianBlur(c, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101) for c in x])

    def sample_bilinear(self, x, gy, gx, border):
        gx, gy = np.broadcast_arrays(gx, gy)
        if gx.ndim != 2 or max(gx.shape + x.shape[-2:]) >= 32767:
            return None
        mode = cv2.BORDER_CONSTANT if border == "zero" else cv2.BORDER_REPLICATE
        mx = np.ascontiguousarray(gx, dtype=np.float32)
        my = np.ascontiguousarray(gy, dtype=np.float32)
        return np.stack([cv2.remap(c, mx, my, cv2.INTER_LINEAR, borderMode=mode) for c in x])

    def backward_flow(self, current, previous, dead_zone):
        h, w = current.shape
        ratio = min(1.0, 96 / min(h, w))
        size = (max(1, round(w * ratio)), max(1, round(h * ratio)))
        def luma8(frame):
            return (cv2.resize(frame, size, interpolation=cv2.INTER_AREA).clip(0, 1) * 255).astype(np.uint8)
        return self._flow(luma8(current).tobytes(),luma8(previous).tobytes(),size,w,h,tuple(dead_zone))

    @lru_cache(maxsize=10)
    def _flow(self, current, previous, size, w, h, dead_zone):
        def plane(buffer):
            return np.frombuffer(buffer, dtype=np.uint8).reshape(size[1], size[0])
        vector = cv2.calcOpticalFlowFarneback(plane(current), plane(previous), None, 0.5, 3, 11, 2, 5, 1.1, 0)
        vector = cv2.resize(vector, (w, h), interpolation=cv2.INTER_LINEAR)
        fx = vector[:, :, 0] * (w / size[0])
        fy = vector[:, :, 1] * (h / size[1])
        keep = np.clip((np.hypot(fy, fx) - dead_zone[0]) / (dead_zone[1] - dead_zone[0]), 0, 1)
        return fy * keep, fx * keep
