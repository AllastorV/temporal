#include "modules.h"

namespace tp {
namespace {

FrameWindow timeDisplacementWindow(const ParamValues& p) {
    const int range = std::clamp(p.i("range"), 1, kMaxTemporalWindow);
    const int dir = p.i("direction");
    FrameWindow w;
    if (dir == 0) w.before = range;
    else if (dir == 1) w.after = range;
    else w.before = w.after = (range + 1) / 2;
    return clampWindow(w);
}

float timeDisplacementNoise(uint32_t seed, float u, float v) {
    float a = valueNoise2(seed, u, v);
    float b = valueNoise2(hashCombine(seed, 0x51u), u * 2.f + 17.3f, v * 2.f + 9.1f);
    return (a * 2.f + b) * (1.f / 3.f);
}

}

ModuleDef makeTimeDisplacement() {
    ModuleDef m;
    m.id = "time_displacement";
    m.name = "Time Displacement";
    m.family = "Temporal";
    m.description = "Per-pixel time offset driven by a luma, gradient or noise map. Each pixel reads a neighbouring frame inside a bounded window, picked by nearest frame or blended between the two nearest frames in linear light.";
    m.temporal = true;
    m.usesEncoding = true;
    m.params = {
        Choice("map_source", "Map Source", {"Luma", "Gradient", "Noise"}, 1,
               "Luma: brightness of the current frame. Gradient: linear or radial ramp. Noise: seeded 2D value noise."),
        Int("range", "Range", 16, 1, 64, 1, 64, Unit::Frames,
            "Time offset in frames reached at map value 1."),
        Choice("direction", "Direction", {"Past", "Future", "Both"}, 0,
               "Past: map 1 reads Range frames earlier. Future: Range frames later. Both: centered, map 0.5 is the current frame and 0..1 spans -Range/2..+Range/2."),
        Dbl("gradient_angle", "Gradient Angle", 90.0, -360.0, 360.0, -180.0, 180.0, Unit::Degrees,
            "Gradient: direction the linear ramp rises in. 0 = left to right, 90 = bottom to top."),
        Bool("radial", "Radial", false, "Gradient: ramp rises radially from Center instead of linearly."),
        Point("center", "Center", 0.5, 0.5, "Gradient: center of the ramp (map 0.5 for linear, map 0 for radial)."),
        Dbl("noise_scale", "Noise Scale", 160.0, 4.0, 2000.0, 16.0, 600.0, Unit::Pixels,
            "Noise: feature size of the noise in pixels."),
        Dbl("noise_evolution", "Noise Evolution", 0.0, 0.0, 4.0, 0.0, 4.0, Unit::None,
            "Noise: how fast the noise field changes over time, in cycles per second. 0 = static."),
        Int("seed", "Seed", 1, 0, 9999, 0, 9999, Unit::None, "Noise: seed of the noise pattern."),
        Dbl("contrast", "Map Contrast", 1.0, 0.0, 4.0, 0.0, 4.0, Unit::None,
            "Contrast of the map around 0.5, then clamped to 0..1. 0 = flat 0.5."),
        Bool("invert", "Invert Map", false, "Flip the map (1 - map) after contrast."),
        Dbl("map_blur", "Map Blur", 24.0, 0.0, 200.0, 0.0, 200.0, Unit::Pixels,
            "Gaussian blur radius of the map in pixels (sigma = radius / 2). Softens the time boundaries."),
        Choice("interpolation", "Interpolation", {"Nearest Frame", "Blend Frames"}, 1,
               "Nearest Frame: each pixel shows one whole frame. Blend Frames: linear blend between the two nearest frames."),
    };
    m.framesNeeded = [](const ParamValues& p, double) { return timeDisplacementWindow(p); };
    m.render = [](const RenderContext& c) {
        const ParamValues& p = c.p();
        const FrameWindow w = timeDisplacementWindow(p);
        const int range = std::clamp(p.i("range"), 1, kMaxTemporalWindow);
        const int dir = p.i("direction");
        const int source = p.i("map_source");
        const bool radial = p.b("radial");
        const bool invert = p.b("invert");
        const bool blend = p.i("interpolation") == 1;
        const float contrast = clampf(p.f("contrast"), 0.f, 4.f);
        const Encoding enc = c.encoding;

        std::vector<ImageView> frames;
        frames.reserve(static_cast<size_t>(w.before + w.after + 1));
        for (int k = -w.before; k <= w.after; ++k) frames.push_back(k == 0 ? c.source() : c.frameOffset(k));
        const ImageView cur = frames[static_cast<size_t>(w.before)];

        const float fw = static_cast<float>(std::max(1, c.frame.w())), fh = static_cast<float>(std::max(1, c.frame.h()));
        double pcx, pcy;
        c.pointToPixels(p.vec("center"), pcx, pcy);
        const float cx = static_cast<float>(pcx), cy = static_cast<float>(pcy);
        const float ang = static_cast<float>(p.d("gradient_angle")) * kPi / 180.f;
        const float ca = std::cos(ang), sa = std::sin(ang);
        const float ext = std::max(1.f, std::fabs(fw * ca) + std::fabs(fh * sa));
        const float halfDiag = std::max(1.f, 0.5f * std::hypot(fw, fh));
        const float nscale = std::max(1.f, c.px(clampf(p.f("noise_scale"), 4.f, 2000.f)));
        const double fps = std::isfinite(c.fps) && c.fps > 0 ? c.fps : 24.0;
        double z = c.time / fps * std::clamp(p.d("noise_evolution"), 0.0, 4.0);
        if (!std::isfinite(z)) z = 0;
        z = std::clamp(z, -1e7, 1e7);
        const double zfl = std::floor(z);
        const int32_t zi = static_cast<int32_t>(zfl);
        const float zt = smoothstepf(0.f, 1.f, static_cast<float>(z - zfl));
        const uint32_t seed = static_cast<uint32_t>(std::clamp(p.i("seed"), 0, 9999));
        const uint32_t seedA = hashCombine(seed, static_cast<uint32_t>(zi));
        const uint32_t seedB = hashCombine(seed, static_cast<uint32_t>(zi + 1));

        auto rawMap = [&](int x, int y) -> float {
            const float px = x + 0.5f, py = y + 0.5f;
            if (source == 0) return saturate(finiteOr(luma(decode(cur.at(x, y, SampleEdge::Clamp), enc)), 0.f));
            if (source == 1) {
                const float dx = px - cx, dy = py - cy;
                if (radial) return std::hypot(dx, dy) / halfDiag;
                return 0.5f + (dx * ca + dy * sa) / ext;
            }
            const float u = (px - c.frame.x1) / nscale, v = (py - c.frame.y1) / nscale;
            const float a = timeDisplacementNoise(seedA, u, v);
            if (zt <= 0.f) return a;
            return lerpf(a, timeDisplacementNoise(seedB, u, v), zt);
        };

        const float sigma = std::min(c.px(clampf(p.f("map_blur"), 0.f, 200.f)) * 0.5f, std::max(fw, fh));
        PixelBuffer mapPlane;
        Rect mapRect;
        const bool useBuffer = sigma > 0.35f;
        if (useBuffer) {
            const int pad = static_cast<int>(std::ceil(sigma * 3.f)) + 2;
            Rect r{c.window.x1 - pad, c.window.y1 - pad, c.window.x2 + pad, c.window.y2 + pad};
            Rect clipped = intersect(r, c.frame);
            if (clipped.empty()) clipped = c.window;
            clipped = Rect{std::min(clipped.x1, c.window.x1), std::min(clipped.y1, c.window.y1), std::max(clipped.x2, c.window.x2), std::max(clipped.y2, c.window.y2)};
            mapRect = clipped;
            mapPlane.resize(static_cast<size_t>(clipped.w()) * clipped.h());
            parallelRows(clipped.y1, clipped.y2, [&](int y) {
                float* row = mapPlane.data() + static_cast<size_t>(y - clipped.y1) * clipped.w();
                for (int x = clipped.x1; x < clipped.x2; ++x) row[x - clipped.x1] = rawMap(x, y);
            });
            gaussianBlurPlane(mapPlane.data(), clipped.w(), clipped.h(), sigma);
        }

        const float rangeF = static_cast<float>(range);
        parallelRows(c.window.y1, c.window.y2, [&](int y) {
            for (int x = c.window.x1; x < c.window.x2; ++x) {
                float mv = useBuffer ? mapPlane[static_cast<size_t>(std::clamp(y, mapRect.y1, mapRect.y2 - 1) - mapRect.y1) * mapRect.w() + (std::clamp(x, mapRect.x1, mapRect.x2 - 1) - mapRect.x1)] : rawMap(x, y);
                mv = saturate(finiteOr((mv - 0.5f) * contrast + 0.5f, 0.5f));
                if (invert) mv = 1.f - mv;
                float o = dir == 0 ? -mv * rangeF : (dir == 1 ? mv * rangeF : (mv - 0.5f) * rangeF);
                o = clampf(o, static_cast<float>(-w.before), static_cast<float>(w.after));
                if (!blend) {
                    int k = std::clamp(static_cast<int>(std::lround(o)), -w.before, w.after);
                    c.dst.store(x, y, frames[static_cast<size_t>(k + w.before)].at(x, y, SampleEdge::Black));
                    continue;
                }
                const float fl = std::floor(o);
                const int k0 = std::clamp(static_cast<int>(fl), -w.before, w.after);
                const float t = o - fl;
                const RGBA a = frames[static_cast<size_t>(k0 + w.before)].at(x, y, SampleEdge::Black);
                if (t <= 1e-6f || k0 >= w.after) { c.dst.store(x, y, a); continue; }
                const RGBA b = frames[static_cast<size_t>(k0 + 1 + w.before)].at(x, y, SampleEdge::Black);
                c.dst.store(x, y, encode(lerp(decode(a, enc), decode(b, enc), t), enc));
            }
        });
    };
    return m;
}

}
