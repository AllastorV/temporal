"""Small-frame region detection and deterministic nearest-region tracking."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


def components(mask: np.ndarray, minimum: float, maximum: float) -> list[list[float]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    found = []
    for sy, sx in zip(*np.where(mask), strict=True):
        if seen[sy, sx]:
            continue
        todo = deque([(int(sy), int(sx))])
        seen[sy, sx] = True
        x0 = x1 = int(sx)
        y0 = y1 = int(sy)
        area = 0
        while todo:
            y, x = todo.popleft()
            area += 1
            x0, x1, y0, y1 = min(x0, x), max(x1, x), min(y0, y), max(y1, y)
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    todo.append((yy, xx))
        if minimum <= area / (h * w) <= maximum:
            found.append([x0 / w, y0 / h, (x1 + 1) / w, (y1 + 1) / h, area / (h * w)])
    return sorted(found, key=lambda b: -b[4])


def camera_shift(current: np.ndarray, previous: np.ndarray) -> tuple[int, int]:
    a = np.fft.rfft2(current - np.median(current))
    b = np.fft.rfft2(previous - np.median(previous))
    cross = a * np.conj(b)
    correlation = np.fft.irfft2(cross / np.maximum(np.abs(cross), 1e-8), s=current.shape)
    y, x = np.unravel_index(np.argmax(correlation), current.shape)
    h, w = current.shape
    return int(y if y <= h // 2 else y - h), int(x if x <= w // 2 else x - w)


def detect(gray: np.ndarray, previous: np.ndarray | None, p: dict[str, Any]) -> list[list[float]]:
    mode = p["detect"]
    if mode == "motion":
        if previous is None:
            return []
        dy, dx = camera_shift(gray, previous) if p["camera_compensate"] else (0, 0)
        previous = np.roll(previous, (dy, dx), (0, 1))
        delta = gray - previous
        delta -= np.median(delta)
        field = np.abs(delta)
        if dy > 0:
            field[:dy] = 0
        elif dy < 0:
            field[dy:] = 0
        if dx > 0:
            field[:, :dx] = 0
        elif dx < 0:
            field[:, dx:] = 0
    elif mode == "details":
        gy, gx = np.gradient(gray)
        field = np.hypot(gy, gx)
    else:
        field = gray
    mask = field > float(p["threshold"])
    for _ in range(int(p["merge"])):
        pad = np.pad(mask, 1)
        mask = np.logical_or.reduce(
            [pad[y : y + gray.shape[0], x : x + gray.shape[1]] for y in range(3) for x in range(3)]
        )
    return components(mask, float(p["min_area"]), float(p["max_area"]))[: int(p["max_blobs"])]


def track(
    boxes: list[list[float]], state: dict[str, Any], p: dict[str, Any]
) -> list[dict[str, Any]]:
    old = state.get("tracks", [])
    pairs = []
    for i, box in enumerate(boxes):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        for j, prior in enumerate(old):
            b = prior["box"]
            distance = (cx - (b[0] + b[2]) / 2) ** 2 + (cy - (b[1] + b[3]) / 2) ** 2
            if distance < 0.12**2:
                pairs.append((distance, i, j))
    assigned, used = {}, set()
    for _, i, j in sorted(pairs):
        if i not in assigned and j not in used:
            assigned[i] = j
            used.add(j)
    tracks = []
    for i, box in enumerate(boxes):
        if i in assigned:
            prior = old[assigned[i]]
            smooth = float(p["smooth"])
            b = [a * smooth + v * (1 - smooth) for a, v in zip(prior["box"], box[:4], strict=True)]
            ident, trail = prior["id"], prior["trail"]
        else:
            ident = state.get("next", 1)
            state["next"] = ident + 1
            b, trail = box[:4], []
        trail = [*trail, [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]][-max(1, int(p["trails"])) :]
        tracks.append({"id": ident, "box": b, "trail": trail, "lost": 0})
    for j, prior in enumerate(old):
        if j not in used and prior["lost"] < int(p["lifetime"]):
            tracks.append({**prior, "lost": prior["lost"] + 1})
    state["tracks"] = tracks[: int(p["max_blobs"])]
    return state["tracks"]
