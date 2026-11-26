#include "modules.h"
#include "../gpu/gpu.h"

namespace tp {

namespace {

float timeShiftMask(float l, float coverage) {
    float th = 1.f - coverage;
    return smoothstepf(th - 0.25f, th, l);
}

float timeShiftFade(float f) { return std::pow(std::max(0.f, 1.f - 0.92f * f), 1.3f); }

}

ModuleDef makeTimeShift() {
    ModuleDef m;
    m.id = "time_shift";
    m.cuda = CudaMode::Device;
    m.name = "Time Shift";
    m.family = "Temporal";
    m.description = "Older frames are displaced in proportion to their age, so moving and bright content leaves a directional, vertical or radial smear of its past.";
    m.temporal = true;
    m.usesEncoding = true;
    m.params = {
        Choice("mode", "Mode", {"Directional Smear", "Pull Up/Down", "Radial Drag"}, 0,
               "Directional Smear: past samples slide along Angle. Pull Up/Down: past samples slide vertically. Radial Drag: past samples slide away from (or toward) Center."),
        Int("offset_frames", "Offset", 8, 1, 32, 1, 32, Unit::Frames, "Temporal depth in frames: the oldest sample is this many frames back."),
        Choice("combine", "Combine", {"Light Trails", "Average", "Shadow Trails"}, 0,
               "Light Trails: bright past content drags behind itself and is only laid over the frame where brighter, keeping its colour and fading with age. Shadow Trails: the same for dark subjects on bright backgrounds (Coverage then selects by darkness). Average: normalized mix of the current frame and its displaced past."),
        Dbl("streak_length", "Streak Length", 60.0, -2000.0, 2000.0, -400.0, 400.0, Unit::Pixels,
            "Spatial displacement of the oldest sample in full-resolution pixels; younger samples move proportionally less. Negative flips direction (in Pull Up/Down negative pulls up, in Radial Drag negative drags inward)."),
        Dbl("angle", "Angle", 0.0, -360.0, 360.0, -180.0, 180.0, Unit::Degrees, "Directional Smear only: direction the past slides toward, in degrees counter-clockwise from the right."),
        Int("detail", "Detail", 12, 2, 32, 2, 32, Unit::None, "Number of temporal samples spread evenly across the depth (capped at Offset). Higher is smoother and slower."),
        Dbl("light_response", "Light Response", 1.0, 0.0, 4.0, 0.0, 4.0, Unit::None, "Extra weight for bright past samples: weight = 1 + response x luma^2, so highlights streak more."),
        Dbl("coverage", "Coverage", 0.55, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Which past content smears: luma threshold 1 - coverage with a soft knee. 1 smears everything, 0 smears only highlights above 1.0 (HDR). The current frame is always kept."),
        Point("center", "Center", 0.5, 0.5, "Radial Drag only: origin of the drag, normalized frame coordinates."),
    };
    m.framesNeeded = [](const ParamValues& p, double) {
        FrameWindow w;
        w.before = std::clamp(p.i("offset_frames"), 1, 32);
        return w;
    };
    m.render = [](const RenderContext& c) {
        const ParamValues& p = c.p();
        const int mode = p.i("mode");
        const int depth = std::clamp(p.i("offset_frames"), 1, 32);
        const int samples = std::min(std::clamp(p.i("detail"), 2, 32), depth);
        const float diag = std::max(1.f, std::hypot(static_cast<float>(c.frame.w()), static_cast<float>(c.frame.h())));
        const float len = clampf(finiteOr(c.px(p.f("streak_length")), 0.f), -diag, diag);
        const float ang = p.f("angle") * kPi / 180.f;
        const float lr = clampf(p.f("light_response"), 0.f, 4.f);
        const float cov = saturate(p.f("coverage"));
        double cxd, cyd;
        c.pointToPixels(p.vec("center"), cxd, cyd);
        const float cx = static_cast<float>(cxd), cy = static_cast<float>(cyd);
        float dx = 0.f, dy = 0.f;
        if (mode == 0) { dx = std::cos(ang) * len; dy = std::sin(ang) * len; }
        else if (mode == 1) dy = -len;
        const float radial = len / diag;

        const int combine = p.i("combine");
        struct Tap { ImageView img; float f0, f1; };
        std::vector<Tap> taps;
        taps.reserve(static_cast<size_t>(samples));
        float prevF = 0.f;
        for (int k = 1; k <= samples; ++k) {
            int age = static_cast<int>(std::lround(static_cast<double>(k) * depth / samples));
            float f = static_cast<float>(age) / depth;
            taps.push_back({c.frameOffset(-age), prevF, f});
            prevF = f;
        }
        const ImageView cur = c.source();
        const Encoding enc = c.encoding;
        const float maxLen = mode == 2 ? std::fabs(radial) * diag * 0.5f : std::hypot(dx, dy);
        const int ds = maxLen > 8.f ? 2 : 1;
        const int totalSub = std::clamp(static_cast<int>(std::ceil(maxLen / (1.0f * ds))), samples, 96);
        constexpr int kLut = 4096;
        constexpr float kLutMax = 1.5f;
        std::vector<float> lut(kLut + 1);
        for (int i = 0; i <= kLut; ++i) lut[static_cast<size_t>(i)] = decodeValue(kLutMax * i / kLut, enc);
        auto dec = [&](const RGBA& v) -> RGBA {
            if (enc == Encoding::Linear) return v;
            RGBA o = v;
            for (int ch = 0; ch < 3; ++ch) {
                float x = v[ch];
                if (x >= 0.f && x < kLutMax) {
                    float fi = x * (kLut / kLutMax);
                    int i0 = static_cast<int>(fi);
                    float t = fi - i0;
                    o[ch] = lut[static_cast<size_t>(i0)] + (lut[static_cast<size_t>(i0) + 1] - lut[static_cast<size_t>(i0)]) * t;
                } else o[ch] = decodeValue(x, enc);
            }
            return o;
        };
        struct Sub { int tap; float f; };
        std::vector<Sub> subs;
        subs.reserve(static_cast<size_t>(totalSub));
        for (int j = 0; j < totalSub; ++j) {
            float f = (j + 1.f) / totalSub;
            int t = 0;
            while (t + 1 < static_cast<int>(taps.size()) && f > taps[static_cast<size_t>(t)].f1 + 1e-6f) ++t;
            subs.push_back({t, f});
        }

        Rect R = intersect(c.frame, cur.valid() ? cur.bounds : c.frame);
        if (R.empty()) R = c.frame;
        const int lw = (R.w() + ds - 1) / ds, lh = (R.h() + ds - 1) / ds;
        const bool shadow = combine == 2;
        if (gpu::available()) {
            gpu::TimeShiftArgs ga;
            for (const Tap& t : taps) ga.taps.push_back(t.img);
            for (const Sub& sb : subs) { ga.subTap.push_back(sb.tap); ga.subF.push_back(sb.f); }
            ga.cur = cur; ga.dst = c.dst; ga.window = c.window; ga.R = R; ga.ds = ds;
            ga.mode = mode; ga.combine = combine; ga.shadow = shadow;
            ga.dx = dx; ga.dy = dy; ga.cx = cx; ga.cy = cy; ga.radial = radial;
            ga.lr = lr; ga.cov = cov; ga.encoding = static_cast<int>(enc);
            float gn = 0.f;
            for (const Sub& sb : subs) gn += (1.f + lr) * timeShiftFade(sb.f);
            ga.norm = std::max(gn, 1e-6f);
            if (gpu::timeShift(ga)) return;
        }
        requireHost(c);
        struct Layer { OwnedImage ws; };
        std::vector<Layer> layers(taps.size());
        for (size_t k = 0; k < taps.size(); ++k) {
            Layer& L = layers[k];
            L.ws.reset(Rect{0, 0, lw, lh});
            const ImageView img = taps[k].img;
            if (!img.valid()) continue;
            parallelRows(0, lh, [&](int j) {
                const int ya = R.y1 + j * ds, yb = std::min(R.y2, ya + ds);
                for (int i = 0; i < lw; ++i) {
                    const int xa = R.x1 + i * ds, xb = std::min(R.x2, xa + ds);
                    RGBA acc;
                    float wacc = 0.f;
                    for (int y = ya; y < yb; ++y)
                        for (int x = xa; x < xb; ++x) {
                            RGBA s = dec(img.at(x, y, SampleEdge::Clamp));
                            float l = clampf(luma(s), 0.f, 64.f);
                            float lc = shadow ? 1.f - std::min(l, 1.f) : std::min(l, 1.f);
                            float w = (1.f + lr * lc * lc) * timeShiftMask(shadow ? lc : l, cov);
                            acc += s * w;
                            wacc += w;
                        }
                    float inv = 1.f / static_cast<float>((xb - xa) * (yb - ya));
                    RGBA o = acc * inv;
                    o.a = wacc * inv;
                    L.ws.view.store(i, j, o);
                }
            });
        }
        float norm = 0.f;
        for (const Sub& sb : subs) norm += (1.f + lr) * timeShiftFade(sb.f);
        norm = std::max(norm, 1e-6f);
        const float invDs = 1.f / static_cast<float>(ds);

        parallelRows(c.window.y1, c.window.y2, [&](int y) {
            const float py = y + 0.5f;
            for (int x = c.window.x1; x < c.window.x2; ++x) {
                const float px = x + 0.5f;
                float vx = dx, vy = dy;
                if (mode == 2) { vx = (px - cx) * radial; vy = (py - cy) * radial; }
                const RGBA curLin = decode(cur.at(x, y, SampleEdge::Black), enc);
                RGBA acc;
                float wsum = 0.f;
                for (const Sub& sb : subs) {
                    const Layer& L = layers[static_cast<size_t>(sb.tap)];
                    const float fade = timeShiftFade(sb.f);
                    const float lx = (px - vx * sb.f - R.x1) * invDs, ly = (py - vy * sb.f - R.y1) * invDs;
                    RGBA smp = L.ws.view.bilinear(lx, ly, SampleEdge::Clamp);
                    acc += smp * fade;
                }
                wsum = acc.a;
                RGBA out;
                if (combine != 1) {
                    RGBA col = wsum > 1e-6f ? acc * (1.f / wsum) : curLin;
                    float a = 1.f - std::exp(-2.5f * wsum / norm);
                    RGBA trail = lerp(curLin, col, a);
                    for (int ch = 0; ch < 3; ++ch) out[ch] = shadow ? std::min(curLin[ch], trail[ch]) : std::max(curLin[ch], trail[ch]);
                } else {
                    out = (curLin + acc) * (1.f / (1.f + wsum));
                }
                out.a = curLin.a;
                c.dst.store(x, y, encode(out, enc));
            }
        });
    };
    return m;
}

}
