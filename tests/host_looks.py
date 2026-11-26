import json
import os
import sys

sys.path.append(r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Modules")
os.environ.setdefault("RESOLVE_SCRIPT_LIB", r"H:\Programlar\Davinchi Resolve\fusionscript.dll")
import DaVinciResolveScript as dvr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME = int(sys.argv[1])
MEDIA_NAME = sys.argv[2]
MODULE = sys.argv[3]
VARIANTS = json.load(open(sys.argv[4], encoding="utf-8")) if sys.argv[4].endswith(".json") else json.loads(sys.argv[4])
OUT = os.path.join(ROOT, "tests", "host_looks", MODULE)


def main():
    os.makedirs(OUT, exist_ok=True)
    resolve = dvr.scriptapp("Resolve")
    pm = resolve.GetProjectManager()
    project = pm.GetCurrentProject()
    if project is None or project.GetName() != "TEMPORAL_P0_smoke":
        project = pm.LoadProject("TEMPORAL_P0_smoke")
    mp = project.GetMediaPool()
    clips = [c for c in mp.GetRootFolder().GetClipList() if c.GetName() == MEDIA_NAME]
    if not clips:
        print("FAIL: clip not in media pool: " + MEDIA_NAME)
        return 2
    tl = mp.CreateTimelineFromClips("looks_%s_%d" % (MODULE, project.GetTimelineCount()), [clips[0]])
    project.SetCurrentTimeline(tl)
    item = tl.GetItemListInTrack("video", 1)[0]
    comp = item.AddFusionComp()
    media_in = comp.FindTool("MediaIn1")
    media_out = comp.FindTool("MediaOut1")
    ref = comp.AddTool("Saver", -32768, -32768)
    ref.Clip = os.path.join(OUT, "source_.exr")
    ref.ConnectInput("Input", media_in)
    first = None
    for name, settings in VARIANTS.items():
        settings = dict(settings)
        mod = settings.pop("__module", MODULE)
        tool = comp.AddTool("ofx.com.temporal." + mod, -32768, -32768)
        tool.ConnectInput("Source", media_in)
        for k, v in settings.items():
            tool.SetInput(k, v)
        saver = comp.AddTool("Saver", -32768, -32768)
        saver.Clip = os.path.join(OUT, name + "_.exr")
        saver.ConnectInput("Input", tool)
        first = first or tool
    media_out.ConnectInput("Input", first)
    comp.Render({"Start": FRAME, "End": FRAME, "Wait": True})
    missing = [n for n in VARIANTS if not os.path.exists(os.path.join(OUT, "%s_%04d.exr" % (n, FRAME)))]
    print(json.dumps({"out": OUT, "missing": missing}))
    pm.SaveProject()
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
