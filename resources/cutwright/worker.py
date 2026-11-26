"""Run the bundled Cutwright frame kernels outside the host's Python interpreter."""
import ctypes as ct
import json
import mmap
from pathlib import Path
import struct
import sys
import traceback
from ctypes import wintypes as wt

import numpy as np
from fx_kernels import Frame, KERNELS, NumpyOps
from fx_base import TorchOps

PRESETS = json.loads(Path(__file__).with_name("presets.json").read_text(encoding="utf-8"))
GROUPS = ("analog", "geometric", "art", "mosh")
HEADER = struct.Struct("<8id")
OFFSET = 4096

def main():
    tag, capacity, parent_id = sys.argv[1:4]
    api = ct.WinDLL("kernel32", use_last_error=True)
    api.OpenEventW.argtypes = [wt.DWORD, wt.BOOL, wt.LPCWSTR]
    api.OpenEventW.restype = wt.HANDLE
    api.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    api.OpenProcess.restype = wt.HANDLE
    api.SetEvent.argtypes = [wt.HANDLE]
    api.WaitForMultipleObjects.argtypes = [wt.DWORD, ct.POINTER(wt.HANDLE), wt.BOOL, wt.DWORD]
    request = api.OpenEventW(0x100000, False, tag + "_request")
    done = api.OpenEventW(2, False, tag + "_done")
    parent = api.OpenProcess(0x100000, False, int(parent_id))
    if not all((request, done, parent)):
        raise OSError(ct.get_last_error())
    handles = (wt.HANDLE * 2)(request, parent)
    ops = NumpyOps()
    cuda = None
    if "--preview" in sys.argv[4:]:
        try:
            from preview_ops import PreviewOps
            ops = PreviewOps()
        except (ImportError, OSError):
            pass
        try:
            import torch
            if torch.cuda.is_available():
                cuda = TorchOps("cuda")
        except (ImportError, OSError, RuntimeError):
            pass
    state = {}
    with mmap.mmap(-1, int(capacity), tagname=tag) as mapping:
        while api.WaitForMultipleObjects(2, handles, False, 0xffffffff) == 0:
            command, group, preset, w, h, index, seed, reset, fps = HEADER.unpack_from(mapping)
            if command == 0:
                break
            try:
                if reset:
                    state = {}
                effect = PRESETS[GROUPS[group]][preset]
                frame = np.ndarray((h,w,4), np.float32, buffer=mapping, offset=OFFSET)
                # OFX rows run bottom up. Cutwright image coordinates run top down.
                alpha = np.ascontiguousarray(frame[::-1,:,3])
                rgb = frame[::-1,:,:3].transpose(2,0,1).copy()
                rgb /= np.maximum(alpha, 1e-6)[None]
                np.clip(rgb, 0, 1, out=rgb)
                backend = cuda if cuda and group != 1 else ops
                f = Frame(index, fps, seed, state=state, alpha=backend.asarray(alpha))
                image = backend.asarray(rgb)
                geometric_history = group == 1 and preset in (0, 1, 3) and hasattr(ops, "field")
                if geometric_history:
                    ops.frame_key = (index, seed, fps, preset, w, h, frame.tobytes())
                if command == 1 and geometric_history:
                    # Only tracking state contributes to the next frame. All drawing is local.
                    from fx_geo import geo_layers, masks
                    params = effect["params"]
                    geo_layers(backend, image, params, f, rgb, masks(backend, params, f, h, w))
                else:
                    out = KERNELS[effect["effect"]](backend, image, effect["params"], f)
                if command == 2:
                    frame[::-1,:,:3] = np.moveaxis(np.clip(backend.numpy(out), 0, 1) * alpha, 0, -1)
                if geometric_history:
                    ops.frame_key = None
                del frame
                struct.pack_into("<i", mapping, 0, 1)
            except Exception:
                message = traceback.format_exc().encode("utf-8", errors="replace")[-1800:]
                mapping[64:65 + len(message)] = message + b"\0"
                struct.pack_into("<i", mapping, 0, -1)
            api.SetEvent(done)

if __name__ == "__main__":
    main()
