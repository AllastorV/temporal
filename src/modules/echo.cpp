#include "modules.h"

namespace tp {
namespace {

const RGBA kEchoBlack{};

const float* echoRow(const ImageView& v, int y) {
    return v.valid() && y >= v.bounds.y1 && y < v.bounds.y2 ? v.px(v.bounds.x1, y) : nullptr;
}

const RGBA& echoTap(const ImageView& v, const float* row, int x) {
    return row && x >= v.bounds.x1 && x < v.bounds.x2 ? reinterpret_cast<const RGBA*>(row)[x - v.bounds.x1] : kEchoBlack;
}

RGBA echoAffine(const ImageView& v, const Affine& inv, int x, int y) {
    double sx, sy;
    inv.apply(x + 0.5, y + 0.5, sx, sy);
    const float fx = static_cast<float>(sx) - 0.5f, fy = static_cast<float>(sy) - 0.5f;
    const float flx = std::floor(fx), fly = std::floor(fy);
    if (flx >= v.bounds.x1 && flx < v.bounds.x2 - 1 && fly >= v.bounds.y1 && fly < v.bounds.y2 - 1) {
        const int x0 = static_cast<int>(flx), y0 = static_cast<int>(fly);
        const float tx = fx - flx, ty = fy - fly;
        const RGBA* r0 = reinterpret_cast<const RGBA*>(v.px(x0, y0));
        const RGBA* r1 = reinterpret_cast<const RGBA*>(v.px(x0, y0 + 1));
        return lerp(lerp(r0[0], r0[1], tx), lerp(r1[0], r1[1], tx), ty);
    }
    return v.bilinear(static_cast<float>(sx), static_cast<float>(sy), SampleEdge::Black);
}

}

ModuleDef makeEcho() {
    ModuleDef m;
    m.id = "echo";
    m.name = "Echo";
    m.family = "Temporal";
    m.description = "Layers earlier (or later) frames over the current one as ghosts, glass reflections, stretched trails or a recursive tunnel.";
    m.temporal = true;
    m.usesEncoding = true;
    m.params = {
        Choice("mode", "Mode", {"Ghosts", "Glass", "Stretch", "Recursive"}, 0,
               "Ghosts: plain trails. Glass: each echo scaled/rotated about the center. Stretch: echoes stretched along an axis. Recursive: echo-of-echo series weighted by Feedback."),
        Choice("direction", "Direction", {"Past", "Future"}, 0, "Take echoes from earlier or later frames."),
        Choice("blend", "Blend", {"Average", "Lighten", "Add", "Darken"}, 3,
               "How echoes combine with the current frame. Darken: each echo is laid over the frame only where it is darker, giving distinct stacked silhouettes on bright backgrounds. Lighten: the same for bright subjects on dark backgrounds. Average: soft trails. Add: accumulating light."),
        Int("count", "Count", 6, 1, 32, 1, 16, Unit::None, "Number of echoes."),
        Dbl("spacing", "Spacing", 2.0, 0.5, 16.0, 0.5, 8.0, Unit::Frames, "Frames between consecutive echoes."),
        Dbl("decay", "Decay", 0.6, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Weight multiplier per echo (echo k weighs decay^k)."),
        Dbl("scale", "Scale", 1.0, 0.25, 4.0, 0.8, 1.25, Unit::None, "Scale applied per echo, compounding (Glass, Stretch, Recursive)."),
        Dbl("rotation", "Rotation", 0.0, -180.0, 180.0, -30.0, 30.0, Unit::Degrees, "Rotation per echo (Glass, Recursive) or stretch axis (Stretch)."),
        Dbl("feedback", "Feedback", 0.7, 0.0, 0.99, 0.0, 0.99, Unit::Normalized, "Recursive mode only: weight of echo k is feedback^k."),
    };
    m.framesNeeded = [](const ParamValues& p, double) {
        int reach = static_cast<int>(std::ceil(p.i("count") * p.d("spacing")));
        FrameWindow w;
        if (p.i("direction") == 0) w.before = reach; else w.after = reach;
        return w;
    };
    m.render = [](const RenderContext& c) {
        const ParamValues& p = c.p();
        const int mode = p.i("mode"), blend = p.i("blend");
        const int n = std::clamp(p.i("count"), 1, 32);
        const double sign = p.i("direction") == 0 ? -1.0 : 1.0;
        const double spacing = p.d("spacing");
        const double scl = p.d("scale");
        const double rot = p.d("rotation") * kPi / 180.0;
        const float decay = p.f("decay"), feedback = p.f("feedback");
        const double cx = (c.frame.x1 + c.frame.x2) * 0.5, cy = (c.frame.y1 + c.frame.y2) * 0.5;

        struct Layer { ImageView img; Affine inv; float w; bool identity; };
        std::vector<Layer> layers;
        layers.reserve(static_cast<size_t>(n) + 1);
        layers.push_back({c.source(), Affine{}, 1.f, true});
        for (int k = 1; k <= n; ++k) {
            int off = static_cast<int>(std::lround(sign * k * spacing));
            Affine t;
            bool identity = true;
            if (mode == 1 || mode == 3) {
                t = Affine::about(cx, cy, Affine::rotate(rot * k) * Affine::scale(std::pow(scl, k)));
                identity = false;
            } else if (mode == 2) {
                t = Affine::about(cx, cy, Affine::rotate(rot) * Affine::scale(std::pow(scl, k), 1.0) * Affine::rotate(-rot));
                identity = false;
            }
            float w = mode == 3 ? std::pow(feedback, static_cast<float>(k)) : std::pow(decay, static_cast<float>(k));
            layers.push_back({c.frameOffset(off), t.inverse(), w, identity});
        }

        float wsum = 1.f;
        std::vector<Layer> active;
        active.reserve(layers.size());
        for (size_t k = 1; k < layers.size(); ++k) {
            wsum += layers[k].w;
            if (layers[k].img.valid() && layers[k].w > 0.f) active.push_back(layers[k]);
        }
        const float norm = blend == 0 && wsum > 0.f ? 1.f / wsum : 1.f;
        const ImageView& src = layers[0].img;
        const Encoding enc = c.encoding;
        parallelRows(c.window.y1, c.window.y2, [&](int y) {
            constexpr int kChunk = 256;
            RGBA tap[kChunk];
            const float* srcRow = echoRow(src, y);
            RGBA* out = reinterpret_cast<RGBA*>(c.dst.px(c.window.x1, y));
            for (int xa = c.window.x1; xa < c.window.x2; xa += kChunk) {
                const int cnt = std::min(kChunk, c.window.x2 - xa);
                RGBA* acc = out + (xa - c.window.x1);
                for (int i = 0; i < cnt; ++i) acc[i] = echoTap(src, srcRow, xa + i);
                decodeRow(acc, acc, cnt, enc);
                for (const Layer& L : active) {
                    if (L.identity) {
                        const float* row = echoRow(L.img, y);
                        for (int i = 0; i < cnt; ++i) tap[i] = echoTap(L.img, row, xa + i);
                    } else {
                        for (int i = 0; i < cnt; ++i) tap[i] = echoAffine(L.img, L.inv, xa + i, y);
                    }
                    decodeRow(tap, tap, cnt, enc);
                    if (blend == 0 || blend == 2) {
                        for (int i = 0; i < cnt; ++i) acc[i] += tap[i] * L.w;
                    } else {
                        const float wk = std::min(1.f, L.w);
                        for (int i = 0; i < cnt; ++i)
                            for (int ch = 0; ch < 3; ++ch) {
                                const float d = blend == 1 ? std::max(acc[i][ch], tap[i][ch]) : std::min(acc[i][ch], tap[i][ch]);
                                acc[i][ch] = lerpf(acc[i][ch], d, wk);
                            }
                    }
                }
                if (blend == 0 && wsum > 0.f)
                    for (int i = 0; i < cnt; ++i) acc[i] *= norm;
                encodeRow(acc, acc, cnt, enc);
            }
        });
    };
    return m;
}

}
