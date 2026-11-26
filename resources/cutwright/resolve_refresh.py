import os
import sys
import threading

modules = os.path.join(os.environ.get("PROGRAMDATA", r"C:\ProgramData"), r"Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Modules")
sys.path.append(modules)
import DaVinciResolveScript as script


def shifted(timecode, fps, delta):
    separator = ";" if ";" in timecode else ":"
    hours, minutes, seconds, frames = (int(part) for part in timecode.replace(";", ":").split(":"))
    base = int(round(fps))
    total = ((hours * 60 + minutes) * 60 + seconds) * base + frames + delta
    frames, total = total % base, total // base
    seconds, total = total % 60, total // 60
    minutes, hours = total % 60, total // 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{frames:02d}"


def refresh(resolve):
    if resolve.GetCurrentPage() == "fusion":
        return
    timeline = resolve.GetProjectManager().GetCurrentProject().GetCurrentTimeline()
    if not timeline:
        return
    here = timeline.GetCurrentTimecode()
    fps = float(timeline.GetSetting("timelineFrameRate") or 24)
    moved = shifted(here, fps, 1)
    timeline.SetCurrentTimecode(moved)
    if timeline.GetCurrentTimecode() == moved:
        timeline.SetCurrentTimecode(here)


def main():
    pending = threading.Event()
    closed = threading.Event()

    def listen():
        while sys.stdin.readline():
            pending.set()
        closed.set()
        pending.set()

    threading.Thread(target=listen, daemon=True).start()
    resolve = None
    while True:
        pending.wait()
        if closed.is_set():
            return
        closed.wait(0.03)
        pending.clear()
        try:
            if resolve is None:
                resolve = script.scriptapp("Resolve")
            if resolve is not None:
                refresh(resolve)
        except Exception:
            resolve = None


if __name__ == "__main__":
    main()
