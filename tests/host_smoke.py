import json
import os
import subprocess
import sys
import tempfile

sys.path.append(r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Modules")
os.environ.setdefault("RESOLVE_SCRIPT_LIB", r"H:\Programlar\Davinchi Resolve\fusionscript.dll")
import DaVinciResolveScript as dvr
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(ROOT, "tests", "media", "moving_square.mov")
OUT = os.path.join(ROOT, "tests", "host_smoke", os.path.splitext(os.path.basename(MEDIA))[0])
MODULES = ["time_shift", "slow_shutter", "echo", "prism", "stutter", "pixel", "double_exposure", "time_displacement",
           "halation", "print_lab", "diffusion_lab", "camera_motion", "refraction_lab", "procedural_motion",
           "motion_tear", "datamosh", "blob_tracking", "multi_optic"]
FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def read_exr(path):
    raw = os.path.join(tempfile.gettempdir(), "tp_smoke.raw")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", path],
                           capture_output=True, text=True, check=True)
    s = json.loads(probe.stdout)["streams"][0]
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path, "-f", "rawvideo", "-pix_fmt", "gbrpf32le", raw], check=True)
    return np.fromfile(raw, dtype=np.float32).reshape(3, s["height"], s["width"])


def main():
    os.makedirs(OUT, exist_ok=True)
    resolve = dvr.scriptapp("Resolve")
    if resolve is None:
        print("FAIL: cannot connect to Resolve")
        return 2
    pm = resolve.GetProjectManager()
    cur = pm.GetCurrentProject()
    if cur is not None and cur.GetName() == "TEMPORAL_P0_smoke":
        project = cur
    else:
        project = pm.LoadProject("TEMPORAL_P0_smoke") or pm.CreateProject("TEMPORAL_P0_smoke")
    mp = project.GetMediaPool()
    clips = [c for c in mp.GetRootFolder().GetClipList() if c.GetName() == os.path.basename(MEDIA)]
    if not clips:
        clips = mp.ImportMedia([MEDIA]) or resolve.GetMediaStorage().AddItemListToMediaPool([MEDIA])
    if not clips:
        print("FAIL: could not import " + MEDIA)
        return 2
    name = "smoke_all_%d" % project.GetTimelineCount()
    tl = mp.CreateTimelineFromClips(name, [clips[0]])
    project.SetCurrentTimeline(tl)
    item = tl.GetItemListInTrack("video", 1)[0]
    comp = item.AddFusionComp()
    media_in = comp.FindTool("MediaIn1")
    media_out = comp.FindTool("MediaOut1")
    ref = comp.AddTool("Saver", -32768, -32768)
    ref.Clip = os.path.join(OUT, "source_.exr")
    ref.ConnectInput("Input", media_in)
    results = {}
    first = None
    for i, mid in enumerate(MODULES):
        tool = comp.AddTool("ofx.com.temporal." + mid, -32768, -32768)
        if tool is None:
            results[mid] = {"load": False}
            continue
        tool.ConnectInput("Source", media_in)
        saver = comp.AddTool("Saver", -32768, -32768)
        saver.Clip = os.path.join(OUT, mid + "_.exr")
        saver.ConnectInput("Input", tool)
        first = first or tool
        results[mid] = {"load": True, "inputs": len(tool.GetInputList())}
    if first is not None:
        media_out.ConnectInput("Input", first)
    comp.Render({"Start": FRAME, "End": FRAME, "Wait": True})
    src = read_exr(os.path.join(OUT, "source_%04d.exr" % FRAME))
    ok = True
    for mid in MODULES:
        r = results[mid]
        path = os.path.join(OUT, "%s_%04d.exr" % (mid, FRAME))
        if not r["load"] or not os.path.exists(path):
            r["rendered"] = False
            ok = False
            continue
        img = read_exr(path)
        r["rendered"] = True
        r["finite"] = bool(np.isfinite(img).all())
        r["mean_abs_diff_vs_source"] = round(float(np.abs(img - src).mean()), 5)
        r["mean"] = round(float(img.mean()), 5)
        ok = ok and r["finite"] and r["mean_abs_diff_vs_source"] > 0
    print(json.dumps({"frame": FRAME, "timeline": name, "results": results}, indent=1))
    pm.SaveProject()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
