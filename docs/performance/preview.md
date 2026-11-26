# 1080p preview checkpoint

The target on the RTX 3070 Ti is at most 33.33 ms per frame with a UHD source, a 1920 × 1080 timeline at 30 fps, full playback resolution and render cache off. The target is still open.

The diagnostic CSV records an initial in-process sweep of 111 modes and presets across 22 effects. It used 1920 × 1080 decoded input and output, the interactive flag, OFX render scale 0.5, one separately timed initial frame and eight subsequent frames per case. The effective scale of the actual full-resolution Resolve timeline was later measured at 1.0. Use the initial sweep to locate bottlenecks.

In Resolve Studio 21.2 the Color-page test with Motion Tear applied displayed 13 fps during playback. The normal-size render calls had interactive and draft flags both zero, also seen in the Fusion file render. The automatic quality choice preserves full detail when those flags are absent. The UI check predates the CUDA flow solver in this checkpoint. The solver passed its CPU parity test across three quality settings. The updated in-process full-quality Motion Tear run at scale 1.0 measured 40.17 ms median and 47.15 ms worst across 30 consecutive frames.

## Repeat the benchmark

Build the CMake target `tp_preview_bench` in the project build directory so the Cutwright worker can locate bundle resources. The optional CMake file path `TP_PREVIEW_PYTHON` or environment variable `TEMPORAL_PREVIEW_PYTHON` selects a Python installation with CUDA PyTorch and OpenCV. Full-quality Cutwright uses the regular Python installation.

```powershell
ffmpeg -ss 1 -i input.mp4 -vf "fps=30,scale=1920:1080:flags=bicubic,vflip" -frames:v 84 -an -pix_fmt rgb24 -f rawvideo preview_1080.rgb
cmake --build build --target tp_preview_bench
./build/tp_preview_bench.exe report.csv 30 all
```

The current runner uses OFX render scale 1.0. Use `default` for default settings, `all` for the primary mode or preset options, and `full` for default settings with preview disabled. An optional effect ID selects one module. Input frames are in memory before timing. Decode, viewer display and disk writes are outside the timed section. The first frame and the slowest measured frame are reported separately.

## Full-resolution measurements for 43de940

At OFX scale 1.0, the explicit preview path met the 33.33 ms per-frame budget in 76 of 111 settings over 30 consecutive frames. Including the first frame, 73 settings met the budget. With the full-quality path used in the tested Resolve Color playback, 7 of 22 default settings met the budget over six consecutive frames, including their first frame. These figures cover effect processing in memory. See `preview-1080p.csv` and `full-1080p.csv` for the individual results. The release test suite passed all 153 tests.

## Resolve playback at Half resolution (2026-11-23)

Measured in Resolve Studio with diagnostic logging. Playback and Deliver send the same render arguments: interactive, draft and sequential flags are all zero, and `beginSequenceRender` arrives once per frame with a one-frame range. The plugin's preview path therefore never runs in Resolve, and no OFX signal tells playback from export.

Resolve's own **Playback > Timeline Playback Resolution > Half** does the job instead. During playback the Color page passes a 960 × 540 image and the Fusion page passes render scale 0.5, while Deliver still renders 1920 × 1080. The frame rectangle is now taken from the source image's own region of definition. Before that, the Fusion page at Half scaled its already reduced RoD a second time and drew a 2× crop of the top-left quarter.

`tp_preview_bench report.csv 12 full "" preview_1080.rgb 2` measures the Half case on a 4K source scaled to 1080. With preview disabled, 19 of 23 effects stayed under 33.33 ms on every measured frame at 960 × 540, against 10 of 23 at 1920 × 1080. CRT Analog, Geometric, Cutwright Art and Cutwright Mosh run through Python workers and take 0.3 to 4.4 s per frame at Half, so they need a native port to reach 30 fps. See `full-540-half.csv` and `full-1080-recheck.csv`.

On the Color page at Half, Resolve reports render scale 1.0 and a 960 × 540 project size, so pixel-sized controls such as blur radii look twice as large during playback. Deliver is unaffected.

A 24 or 25 fps budget (41.67 or 40 ms) changes neither count: the 19 Half results already hold 33.33 ms on every measured frame, and the four Python-worker effects stay 8× to 100× over. At 1920 × 1080, Prism (42.8 ms worst) and Motion Tear (45.7 ms) are the nearest misses. Lens Character showed two 79 ms frames in the sweep, which ran right after the Python effects; on its own, three 40-frame runs stayed at or below 17.4 ms at 1080 and 5.3 ms at Half, with a one-time first frame of 95 to 147 ms.

## Resolve playback on a 1080p timeline (2026-11-23)

Resolve Studio, Edit page, 1920 × 1080 / 30 fps timeline, a 4K H.264 30 fps source, render cache off, one effect per 4-second clip. The figure is the number of frames Resolve actually rendered per second of real playback, capped at the 30 fps timeline rate. Double Exposure and Lens Character are not offered to `TimelineItem.AddEffect`; their rows use the in-memory benchmark. Multi-Optic Studio was not logged.

| Effect | Playback Full | Playback Half |
| --- | ---: | ---: |
| Time Shift | 19 | 30 |
| Slow Shutter | 13 | 30 |
| Echo | 8 | 29 |
| Prism | 15 | 30 |
| Stutter | 30 | 30 |
| Pixel | 30 | 30 |
| Time Displacement | 4 | 3–8 |
| Halation | 27 | 30 |
| Print Lab | 12 | 30 |
| Diffusion Lab | 30 | 30 |
| Camera Motion Lab | 30 | 30 |
| Refraction Lab | 30 | 30 |
| Procedural Motion | 30 | 30 |
| Motion Tear | 21 | 30 |
| Datamosh | 9 | 30 |
| Blob Tracking | 7 | 9–11 |
| Multi-Optic | 27 | 30 |
| CRT / Analog | 0.9 | 3.7 |
| Geometric | ~0.1 (benchmark) | 0.3 |
| Cutwright Art | ~0.4 (benchmark) | 1.2 |
| Cutwright Mosh | 0.1 | 0.3 |
| Lens Character (benchmark) | 30 | 30 |
| Double Exposure (benchmark) | 30 | 30 |

The CRT / Analog, Geometric, Cutwright Art and Cutwright Mosh rows above were measured with the Python worker. All four are native since the CUDA port (`docs/CUDA_PLAN.md`, stages 1–4); their in-memory times at Half are now 7–60 ms per frame, but Resolve playback has not been measured again.

At 24 fps or better: 10 of 23 at Full, 17 of 23 at Half. Time Displacement and Blob Tracking compute in 16–19 ms at Half in memory but read many neighbouring frames, so long-GOP 4K decode limits them in Resolve; optimized or proxy media should help them. The Python-worker effects need a native port.
