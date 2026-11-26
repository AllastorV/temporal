#include "modules.h"
#include "rain_engine.h"
#include "scatter_ops.h"
#include "../gpu/gpu.h"

namespace tp {

namespace {

ParamValues rainLook(const ModuleDef& module, int look) {
    if (look < 1 || look > 6) return {};
    ParamValues p = defaultValues(module);
    for (const char* id : {"studio_look", "encoding", "mix", "seed", "output"}) p.v.erase(id);
    struct Look { double amount, minSize, maxSize, slideSize, slide, speed, wind, wiggle, trails, mist, focus, defocus, refraction; };
    static const Look looks[6] = {
        {0.6, 2.5, 18, 14, 0.10, 0.4, 0, 0.4, 0.4, 0.0, 0.0, 20, 1.0},
        {1.0, 4.0, 36, 24, 0.30, 0.6, 0, 0.5, 0.6, 0.0, 0.0, 28, 1.0},
        {1.6, 3.0, 44, 26, 0.55, 0.9, 0, 0.7, 0.8, 0.1, 0.0, 36, 1.1},
        {1.2, 3.0, 32, 20, 0.35, 1.0, 38, 0.35, 0.7, 0.0, 0.0, 28, 1.0},
        {0.8, 2.5, 26, 18, 0.40, 0.4, 0, 0.6, 0.9, 0.85, 0.0, 32, 1.0},
        {0.09, 20, 110, 160, 0.15, 0.3, 0, 0.4, 0.3, 0.0, 1.0, 40, 0.8},
    };
    const Look& l = looks[look - 1];
    p.set("amount", l.amount); p.set("min_size", l.minSize); p.set("max_size", l.maxSize); p.set("slide_size", l.slideSize);
    p.set("slide", l.slide); p.set("slide_speed", l.speed); p.set("wind", l.wind); p.set("wiggle", l.wiggle);
    p.set("trails", l.trails); p.set("mist", l.mist); p.set("focus", l.focus);
    p.set("defocus", l.defocus); p.set("refraction", l.refraction);
    return p;
}

void rainDiskBlur(const ImageView& src, OwnedImage& out, const Rect& R, float radius) {
    out.reset(R);
    if (gpu::rainDiskBlur(src, R, radius, out.view)) return;
    const int f = std::clamp(static_cast<int>(radius / 5.f), 1, 16);
    const DlLevel lo = dlDown(src, f);
    const Rect lb = lo.img.view.bounds;
    const float rl = radius / f;
    const int taps = std::clamp(static_cast<int>(std::ceil(1.3f * rl * rl)), 12, 96);
    const float ca = -0.73736887f, sa = 0.67549029f;
    DlLevel disk;
    disk.f = f;
    disk.img.reset(lb);
    parallelRows(lb.y1, lb.y2, [&](int y) {
        for (int x = lb.x1; x < lb.x2; ++x) {
            RGBA acc{};
            float u = 1.f, v = 0.f;
            for (int k = 0; k < taps; ++k) {
                const float rr = rl * std::sqrt((k + 0.5f) / taps);
                acc += lo.img.view.bilinear(x + 0.5f + rr * u, y + 0.5f + rr * v, SampleEdge::Clamp);
                const float nu = u * ca - v * sa;
                v = u * sa + v * ca;
                u = nu;
            }
            disk.img.view.store(x, y, acc * (1.f / taps));
        }
    });
    out.reset(R);
    parallelRows(R.y1, R.y2, [&](int y) {
        for (int x = R.x1; x < R.x2; ++x) out.view.store(x, y, f == 1 ? disk.img.view.load(x - R.x1 + lb.x1, y - R.y1 + lb.y1) : disk.sample(R, x, y));
    });
}

}

ModuleDef makeRainGlass() {
    ModuleDef m;
    m.id = "rain_glass";
    m.cuda = CudaMode::Device;
    m.name = "Rain Glass";
    m.family = "Optics";
    m.description = "Rain on a window or a lens: a deterministic drop engine lands drops, merges the ones that touch and runs large drops down the glass in stick-slip paths that swallow the drops in their way, leave beads and clear lanes through condensation. Each drop is a small lens showing an inverted, minified view of the scene, with a thin dark rim, a glint and a caustic. Focus moves between the glass and the scene.";
    m.temporal = false;
    m.usesEncoding = true;
    m.params = {
        Choice("studio_look", "Load Look", {"Choose a Look", "Drizzle", "Steady Rain", "Downpour", "Windy Glass", "Misted Window", "Lens Splash"}, 0,
               "Loads an editable starting point into the rain controls."),
        Dbl("amount", "Rain", 1.0, 0.0, 3.0, 0.0, 2.0, Unit::None, "How many drops land on the glass.", "Drops"),
        Dbl("min_size", "Min Drop Size", 4.0, 0.5, 200.0, 0.5, 20.0, Unit::Pixels, "Diameter of the smallest drops in full-resolution pixels. Small drops are the most common.", "Drops"),
        Dbl("max_size", "Max Drop Size", 36.0, 1.0, 400.0, 2.0, 80.0, Unit::Pixels, "Diameter of the largest drops that land, in full-resolution pixels.", "Drops"),
        Dbl("slide_size", "Slide Above", 24.0, 1.0, 400.0, 2.0, 80.0, Unit::Pixels, "Diameter above which a drop is too heavy to cling to the glass and always runs down, including drops that grow this big by merging.", "Drops"),
        Int("seed", "Seed", 1, 1, 9999, 1, 999, Unit::None, "Changes the drop layout.", "Drops"),
        Dbl("tempo", "Tempo", 1.0, 0.05, 4.0, 0.1, 3.0, Unit::None, "Speed of the rain: how fast drops land, run and evaporate.", "Drops"),
        Dbl("slide", "Running Drops", 0.3, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Share of the drops between half the Slide Above size and the Slide Above size that break free and run early.", "Running"),
        Dbl("slide_speed", "Run Speed", 0.6, 0.0, 3.0, 0.0, 2.0, Unit::None, "Speed of the slips between the pauses of a running drop.", "Running"),
        Dbl("wind", "Wind", 0.0, -75.0, 75.0, -60.0, 60.0, Unit::Degrees, "Tilts the direction drops run, as on a moving car window. Heavy drops follow gravity more.", "Running"),
        Dbl("wiggle", "Wiggle", 0.5, 0.0, 2.0, 0.0, 1.5, Unit::None, "How sharply a running drop changes direction after each pause.", "Running"),
        Dbl("trails", "Trails", 0.6, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Small beads left behind a running drop.", "Running"),
        Dbl("mist", "Condensation", 0.0, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Fog of fine droplets on the glass. Running drops wipe clear lanes that slowly fog again.", "Glass"),
        Dbl("refog", "Refog Time", 4.0, 0.0, 30.0, 0.0, 12.0, Unit::Seconds, "Seconds a wiped lane takes to fog over again. 0 keeps lanes clear.", "Glass"),
        Dbl("focus", "Focus", 0.0, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Where the lens focuses: 0 on the drops on the glass, 1 on the scene behind. Values between rack the focus.", "Depth of Field"),
        Dbl("defocus", "Defocus", 28.0, 0.0, 200.0, 0.0, 80.0, Unit::Pixels, "Blur of whatever is out of focus, as a disk diameter in full-resolution pixels.", "Depth of Field"),
        Dbl("refraction", "Refraction", 1.0, 0.0, 3.0, 0.0, 2.0, Unit::None, "How wide a view each drop gathers. The view inside a drop is upside down and smaller.", "Shading"),
        Dbl("rim", "Rim", 0.5, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Thin dark edge where the drop surface turns steep.", "Shading"),
        Dbl("highlight", "Highlight", 0.6, 0.0, 3.0, 0.0, 2.0, Unit::None, "Specular glint of the light on each drop.", "Shading"),
        Dbl("caustic", "Caustic", 0.35, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Bright crescent on the side of a drop away from the light.", "Shading"),
        Dbl("light_angle", "Light Angle", 135.0, -180.0, 180.0, -180.0, 180.0, Unit::Degrees, "Direction the light comes from, counter-clockwise from the right.", "Shading"),
        Choice("output", "Output", {"Composite", "Water Matte"}, 0, "Water Matte outputs the drops as a white-on-black matte.", "Shading"),
    };
    m.loadLook = rainLook;
    m.framesNeeded = [](const ParamValues&, double) { return FrameWindow{}; };
    m.render = [](const RenderContext& c) {
        const ParamValues& p = c.p();
        const ImageView src = c.source();
        if (!src.valid()) {
            requireHost(c);
            parallelRows(c.window.y1, c.window.y2, [&](int y) { for (int x = c.window.x1; x < c.window.x2; ++x) c.dst.store(x, y, RGBA{}); });
            return;
        }
        const Encoding enc = c.encoding;
        const float fh = static_cast<float>(std::max(1, c.frame.h()));
        RainSettings s;
        s.seconds = (c.time - c.first) / std::max(1.0, c.fps);
        s.aspect = static_cast<float>(c.frame.w()) / fh;
        s.amount = std::max(0.f, p.f("amount"));
        s.maxR = 0.5f * c.px(std::max(0.5f, p.f("max_size"))) / fh;
        s.minR = std::min(s.maxR, 0.5f * c.px(std::max(0.25f, p.f("min_size"))) / fh);
        s.slideR = 0.5f * c.px(std::max(0.5f, p.f("slide_size"))) / fh;
        s.slide = clampf(p.f("slide"), 0.f, 1.f);
        s.slideSpeed = std::max(0.f, p.f("slide_speed"));
        s.wind = clampf(p.f("wind"), -75.f, 75.f);
        s.wiggle = std::max(0.f, p.f("wiggle"));
        s.trails = clampf(p.f("trails"), 0.f, 1.f);
        s.refog = std::max(0.f, p.f("refog"));
        s.tempo = std::max(0.05f, p.f("tempo"));
        s.shutter = 0.5f / static_cast<float>(std::max(1.0, c.fps));
        s.seed = static_cast<uint32_t>(std::max(1, p.i("seed")));
        std::vector<RainDrop> drops;
        std::vector<RainWipe> wipes;
        rainSimulate(s, drops, wipes);

        Rect R = intersect(c.frame, src.bounds);
        if (R.empty()) R = src.bounds;
        const int W = R.w(), H = R.h();
        const auto toPx = [&](float u, float v, float& x, float& y) { x = c.frame.x1 + u * fh; y = c.frame.y1 + v * fh; };
        const int band = 32, bands = (H + band - 1) / band;
        std::vector<std::vector<int>> dropBins(static_cast<size_t>(bands)), wipeBins(static_cast<size_t>(bands));
        const auto bin = [&](std::vector<std::vector<int>>& bins, int index, float y0, float y1) {
            if (y1 < R.y1 || y0 >= R.y2) return;
            const int a = std::clamp(static_cast<int>(std::floor((y0 - R.y1) / band)), 0, bands - 1);
            const int b = std::clamp(static_cast<int>(std::floor((y1 - R.y1) / band)), 0, bands - 1);
            for (int k = a; k <= b; ++k) bins[static_cast<size_t>(k)].push_back(index);
        };
        RainSplat S;
        S.fx1 = static_cast<float>(c.frame.x1); S.fy1 = static_cast<float>(c.frame.y1); S.fh = fh;
        S.rx1 = R.x1; S.ry1 = R.y1; S.rx2 = R.x2; S.ry2 = R.y2;
        S.band = band; S.bands = bands;
        const auto reachOf = [&](const RainDrop& d) { return S.reach(d); };
        for (size_t i = 0; i < drops.size(); ++i) {
            float x, y;
            toPx(drops[i].x, drops[i].y, x, y);
            bin(dropBins, static_cast<int>(i), y - reachOf(drops[i]), y + reachOf(drops[i]));
        }
        for (size_t i = 0; i < wipes.size(); ++i) {
            float ya, yb, xa;
            toPx(wipes[i].x0, wipes[i].y0, xa, ya);
            toPx(wipes[i].x1, wipes[i].y1, xa, yb);
            const float reach = wipes[i].r * fh + 2.f;
            bin(wipeBins, static_cast<int>(i), std::min(ya, yb) - reach, std::max(ya, yb) + reach);
        }
        std::vector<int> dropStart, dropList, wipeStart, wipeList;
        for (int k = 0; k < bands; ++k) {
            dropStart.push_back(static_cast<int>(dropList.size()));
            wipeStart.push_back(static_cast<int>(wipeList.size()));
            dropList.insert(dropList.end(), dropBins[static_cast<size_t>(k)].begin(), dropBins[static_cast<size_t>(k)].end());
            wipeList.insert(wipeList.end(), wipeBins[static_cast<size_t>(k)].begin(), wipeBins[static_cast<size_t>(k)].end());
        }
        dropStart.push_back(static_cast<int>(dropList.size()));
        wipeStart.push_back(static_cast<int>(wipeList.size()));
        S.drops = drops.data(); S.dropCount = static_cast<int>(drops.size());
        S.wipes = wipes.data(); S.wipeCount = static_cast<int>(wipes.size());
        S.dropStart = dropStart.data(); S.dropList = dropList.data(); S.dropListCount = static_cast<int>(dropList.size());
        S.wipeStart = wipeStart.data(); S.wipeList = wipeList.data(); S.wipeListCount = static_cast<int>(wipeList.size());
        const size_t cells = static_cast<size_t>(W) * H;
        const float focus = clampf(p.f("focus"), 0.f, 1.f);
        const float defocus = 0.5f * c.px(std::max(0.f, p.f("defocus")));
        const float sceneBlur = defocus * (1.f - focus), dropBlur = defocus * focus;
        const float refraction = std::max(0.f, p.f("refraction"));
        const float mist = clampf(p.f("mist"), 0.f, 1.f);
        const bool matte = p.i("output") == 1;
        const bool blurDrops = dropBlur > 0.5f && !matte;
        RainShade P;
        P.fh = fh;
        P.mist = mist;
        P.rim = clampf(p.f("rim"), 0.f, 1.f);
        P.highlight = std::max(0.f, p.f("highlight"));
        P.caustic = clampf(p.f("caustic"), 0.f, 1.f);
        P.magnify = 1.5f + 3.f * refraction;
        P.dot = std::max(1.2f, c.px(2.2f));
        P.seed = s.seed;
        P.matte = matte;
        P.blurDrops = blurDrops;
        const float la = p.f("light_angle") * kPi / 180.f;
        P.lx = std::cos(la); P.ly = std::sin(la); P.lz = 1.2f;
        const float ln = 1.f / std::sqrt(P.lx * P.lx + P.ly * P.ly + P.lz * P.lz);
        P.lx *= ln; P.ly *= ln; P.lz *= ln;
        P.hz0 = P.lz + 1.f;
        P.hn = 1.f / std::sqrt(P.lx * P.lx + P.ly * P.ly + P.hz0 * P.hz0);
        P.lxy = std::max(1e-4f, std::sqrt(P.lx * P.lx + P.ly * P.ly));
        const float dot = P.dot, lx = P.lx, ly = P.ly;
        std::vector<float> beadMap, glintMap;
        if (mist > 0) {
            beadMap.assign(cells, 0.f);
            glintMap.assign(cells, 0.f);
            const int cx0 = static_cast<int>(std::floor(R.x1 / dot)) - 1, cx1 = static_cast<int>(std::floor(R.x2 / dot)) + 1;
            const int cy0 = static_cast<int>(std::floor(R.y1 / dot)) - 1, cy1 = static_cast<int>(std::floor(R.y2 / dot)) + 1;
            for (int phase = 0; phase < 5; ++phase) {
                parallelRows(0, (cy1 - cy0) / 5 + 1, [&](int row) {
                    const int gy = cy0 + row * 5 + phase;
                    if (gy > cy1) return;
                    for (int gx = cx0; gx <= cx1; ++gx) {
                        const uint32_t ck = rainKey(rainKey(s.seed + 11u, static_cast<uint32_t>(gx)), static_cast<uint32_t>(gy));
                        const float px0 = (gx + rainHash(rainKey(ck, 2))) * dot, py0 = (gy + rainHash(rainKey(ck, 3))) * dot;
                        const float density = 0.6f + 0.4f * rainNoise2(px0 / (40.f * dot), py0 / (40.f * dot), s.seed + 7u);
                        if (rainHash(rainKey(ck, 1)) > 0.55f + 0.4f * density) continue;
                        const float rad = dot * (0.2f + 0.45f * rainHash(rainKey(ck, 4)) * rainHash(rainKey(ck, 5)) + 0.05f);
                        const int xa = std::max(R.x1, static_cast<int>(std::floor(px0 - rad - 1.f))), xb = std::min(R.x2, static_cast<int>(std::ceil(px0 + rad + 1.f)));
                        const int ya = std::max(R.y1, static_cast<int>(std::floor(py0 - rad - 1.f))), yb = std::min(R.y2, static_cast<int>(std::ceil(py0 + rad + 1.f)));
                        for (int y = ya; y < yb; ++y)
                            for (int x = xa; x < xb; ++x) {
                                const float ddx = x + 0.5f - px0, ddy = y + 0.5f - py0, dd = std::sqrt(ddx * ddx + ddy * ddy);
                                const float cover = 1.f - rainSmooth(rad - 0.6f, rad + 0.4f, dd);
                                const size_t k = static_cast<size_t>(y - R.y1) * W + (x - R.x1);
                                if (cover <= beadMap[k]) continue;
                                beadMap[k] = cover;
                                const float gdx = ddx - 0.35f * rad * lx, gdy = ddy - 0.35f * rad * ly;
                                glintMap[k] = (1.f - rainSmooth(0.f, 0.35f * rad + 0.3f, std::sqrt(gdx * gdx + gdy * gdy))) * cover;
                            }
                    }
                });
            }
        }
        const float innerSigma = std::max(0.5f, 0.35f * P.magnify * std::max(1.f, c.px(1.f)));
        const float fogSigma = std::max(2.f, c.px(18.f));
        const float* bead = beadMap.empty() ? nullptr : beadMap.data();
        const float* glint = glintMap.empty() ? nullptr : glintMap.data();
        const gpu::RainGlassArgs ga{src, c.dst, R, c.window, static_cast<int>(enc), P, S, bead, glint, sceneBlur, dropBlur, innerSigma, fogSigma};
        if (gpu::rainGlass(ga)) return;
        requireHost(c);

        std::vector<float> height(cells, 0.f), wiped(cells, 0.f), wsum(cells, 0.f), wcx(cells, 0.f), wcy(cells, 0.f), wrad(cells, 0.f);
        parallelRows(R.y1, R.y2, [&](int y) {
            const size_t row = static_cast<size_t>(y - R.y1) * W;
            const int b = (y - R.y1) / band;
            const float v = (y + 0.5f - c.frame.y1) / fh;
            for (int k = dropStart[static_cast<size_t>(b)]; k < dropStart[static_cast<size_t>(b) + 1]; ++k) {
                const RainDrop& d = drops[static_cast<size_t>(dropList[static_cast<size_t>(k)])];
                const float cx = S.fx1 + d.x * fh, cy = S.fy1 + d.y * fh, reach = S.reach(d);
                if (std::fabs(y + 0.5f - cy) > reach) continue;
                const int x0 = std::max(R.x1, static_cast<int>(cx - reach)), x1 = std::min(R.x2, static_cast<int>(cx + reach) + 1);
                for (int x = x0; x < x1; ++x) {
                    const float h = rainDropHeight(d, (x + 0.5f - S.fx1) / fh, v);
                    if (h <= 0.f) continue;
                    const size_t q = row + (x - R.x1);
                    rainAddDrop(d, h, cx, cy, fh, height[q], wsum[q], wcx[q], wcy[q], wrad[q]);
                }
            }
            for (int k = wipeStart[static_cast<size_t>(b)]; k < wipeStart[static_cast<size_t>(b) + 1]; ++k) {
                const RainWipe& w = wipes[static_cast<size_t>(wipeList[static_cast<size_t>(k)])];
                float ax, ay, bx, by, rad;
                int x0, x1;
                rainWipeSpan(S, w, ax, ay, bx, by, rad, x0, x1);
                for (int x = x0; x < x1; ++x) {
                    const float clear = rainWipeClear(w, ax, ay, bx, by, rad, x, y);
                    float& cell = wiped[row + (x - R.x1)];
                    if (clear > cell) cell = clear;
                }
            }
        });
        RainMaps M;
        M.height = height.data(); M.wiped = wiped.data(); M.wsum = wsum.data();
        M.wcx = wcx.data(); M.wcy = wcy.data(); M.wrad = wrad.data();
        M.bead = bead; M.glint = glint;
        M.x1 = R.x1; M.y1 = R.y1; M.w = W; M.h = H;

        OwnedImage linear(R);
        parallelRows(R.y1, R.y2, [&](int y) {
            for (int x = R.x1; x < R.x2; ++x) linear.view.store(x, y, decode(src.load(x, y), enc));
        });
        OwnedImage back;
        if (sceneBlur > 0.5f) rainDiskBlur(linear.view, back, R, sceneBlur);
        else back = linear;
        OwnedImage inner = linear;
        gaussianBlur(inner, innerSigma);
        OwnedImage fog;
        if (mist > 0) {
            fog = back;
            gaussianBlur(fog, fogSigma);
        }
        const auto toPL = [](const RGBA& v) { return PL4{v.r, v.g, v.b, v.a}; };
        const auto backAt = [&](int x, int y) { return toPL(back.view.load(x, y)); };
        const auto fogAt = [&](int x, int y) { return toPL(fog.view.load(x, y)); };
        const auto innerAt = [&](float x, float y) { return toPL(inner.view.bilinear(x, y, SampleEdge::Mirror)); };
        const auto bentAt = [&](float x, float y) { return toPL(back.view.bilinear(x, y, SampleEdge::Mirror)); };
        OwnedImage layer, glass;
        if (blurDrops) { layer.reset(R); glass.reset(R); }
        const Rect area = blurDrops ? R : intersect(c.window, R);
        parallelRows(area.y1, area.y2, [&](int y) {
            for (int x = area.x1; x < area.x2; ++x) {
                PL4 g;
                const PL4 o = rainShadePixel(P, M, x, y, backAt, fogAt, innerAt, bentAt, g);
                const RGBA out{o.r, o.g, o.b, o.a};
                if (blurDrops) {
                    layer.view.store(x, y, out);
                    glass.view.store(x, y, RGBA{g.r, g.g, g.b, g.a});
                } else if (c.window.contains(x, y)) {
                    c.dst.store(x, y, matte ? out : encode(out, enc));
                }
            }
        });
        if (!blurDrops) return;
        OwnedImage soft;
        rainDiskBlur(layer.view, soft, R, dropBlur);
        parallelRows(c.window.y1, c.window.y2, [&](int y) {
            for (int x = c.window.x1; x < c.window.x2; ++x) {
                const RGBA bg = glass.view.at(x, y, SampleEdge::Clamp), d = soft.view.at(x, y, SampleEdge::Clamp);
                const RGBA o{bg.r * (1.f - d.a) + d.r, bg.g * (1.f - d.a) + d.g, bg.b * (1.f - d.a) + d.b, bg.a};
                c.dst.store(x, y, encode(o, enc));
            }
        });
    };
    return m;
}

}
