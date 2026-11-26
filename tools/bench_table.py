import re
import sys

raw, md = sys.argv[1], sys.argv[2]
rows = {}
for line in open(raw, encoding="utf-8", errors="ignore"):
    p = line.split()
    if len(p) == 5 and p[1] in ("1080p", "UHD"):
        rows.setdefault(p[0], {})[p[1]] = (p[2], p[3])
table = "| Module | frames | 1080p ms | UHD 3840x2160 ms |\n|---|---|---|---|\n"
for m, v in rows.items():
    table += "| %s | %s | %s | %s |\n" % (m, v["UHD"][0], v["1080p"][1], v["UHD"][1])
text = open(md, encoding="utf-8").read()
text = re.sub(r"\| Module \| frames.*?\n\n", lambda _: table + "\n", text, flags=re.S)
text = text.replace("after the QA round-1 fixes", "after the QA fixes and the Print Lab re-photography rewrite")
text = text.replace("time_shift is the slowest because", "Print Lab is the heaviest at UHD: it prints a supersampled page (up to 10 MP) and re-photographs it through mip-mapped depth of field. time_shift is next because")
open(md, "w", encoding="utf-8").write(text)
print(table)
