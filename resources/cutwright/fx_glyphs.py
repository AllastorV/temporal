"""A bounded, antialiased atlas of the bundled monospaced typeface."""

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from threading import RLock

import numpy as np

FONT = Path(__file__).resolve().parent / "fonts" / "JetBrainsMono-Bold.ttf"
CJK_FONT = FONT.with_name("NotoSansCJKkr-Bold.otf")
ATLAS_BUDGET = 16 * 1024 * 1024
_ATLAS_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_ATLAS_LOCK = RLock()


def is_cjk(text: str) -> bool:
    return any(0x2E80 <= ord(c) <= 0xD7AF or 0xF900 <= ord(c) <= 0xFAFF for c in text)


@lru_cache(maxsize=16)
def label_font(size: int, cjk: bool = False):
    from PIL import ImageFont

    return ImageFont.truetype(str(CJK_FONT if cjk else FONT), size)


def atlas(
    chars: str,
    height: int,
    width: int,
    *,
    regular: bool = False,
    scale: float = 1,
    stretch: bool = False,
) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    # Oversized logical cells sample a bounded raster instead of allocating off-screen glyphs.
    factor = min(
        1, 256 / max(height, width), (ATLAS_BUDGET / (4 * len(chars) * height * width)) ** 0.5
    )
    height, width = max(2, int(height * factor)), max(2, int(width * factor))
    key = (chars, height, width, regular, scale, stretch)
    with _ATLAS_LOCK:
        if key in _ATLAS_CACHE:
            _ATLAS_CACHE.move_to_end(key)
            return _ATLAS_CACHE[key]
    path = FONT.with_name("JetBrainsMono-Regular.ttf") if regular else FONT
    font = ImageFont.truetype(str(path), max(4, round(height * 4 * scale)))
    glyphs = []
    drawn = {}
    for char in chars:
        if char in drawn:
            glyphs.append(drawn[char])
            continue
        face = label_font(max(4, round(height * 4 * scale)), True) if is_cjk(char) else font
        image = Image.new("L", (width * 4, height * 4))
        draw = ImageDraw.Draw(image)
        left, top, right, bottom = draw.textbbox((0, 0), char, font=face)
        if stretch:
            advance = max(1, round(face.getlength("M")))
            image = Image.new("L", (advance, height * 4))
            ascent, descent = face.getmetrics()
            baseline = height * 4 * 0.98 * ascent / (ascent + descent)
            ImageDraw.Draw(image).text((0, baseline), char, font=face, fill=255, anchor="ls")
        else:
            draw.text(
                ((width * 4 - right - left) / 2, (height * 4 - bottom - top) / 2),
                char,
                font=face,
                fill=255,
            )
        value = (
            np.asarray(image.resize((width, height), Image.Resampling.BOX), dtype=np.float32) / 255
        )
        drawn[char] = value
        glyphs.append(value)
    result = np.stack(glyphs)
    if result.nbytes <= ATLAS_BUDGET:
        with _ATLAS_LOCK:
            if key in _ATLAS_CACHE:
                return _ATLAS_CACHE[key]
            while _ATLAS_CACHE and (
                len(_ATLAS_CACHE) >= 16
                or sum(v.nbytes for v in _ATLAS_CACHE.values()) + result.nbytes > ATLAS_BUDGET
            ):
                _ATLAS_CACHE.popitem(last=False)
            _ATLAS_CACHE[key] = result
    return result
