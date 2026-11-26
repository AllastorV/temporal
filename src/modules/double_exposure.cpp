#include "modules.h"

namespace tp {
namespace {

FrameWindow doubleExposureWindow(const ParamValues& p) {
    const int off = std::clamp(p.i("frame_offset"), -kMaxTemporalWindow, kMaxTemporalWindow);
    FrameWindow w;
    w.before = std::max(0, -off);
    w.after = std::max(0, off);
    return clampWindow(w);
}

const RGBA kDoubleExposureBlack{};

const float* doubleExposureRow(const ImageView& v, int y) {
    return v.valid() && y >= v.bounds.y1 && y < v.bounds.y2 ? v.px(v.bounds.x1, y) : nullptr;
}

bool doubleExposureInRow(const ImageView& v, const float* row, int x) {
    return row && x >= v.bounds.x1 && x < v.bounds.x2;
}

const RGBA& doubleExposureTap(const ImageView& v, const float* row, int x) {
    return doubleExposureInRow(v, row, x) ? reinterpret_cast<const RGBA*>(row)[x - v.bounds.x1] : kDoubleExposureBlack;
}

const RGBA& doubleExposureClamped(const ImageView& v, int x, int y) {
    if (!v.valid()) return kDoubleExposureBlack;
    return *reinterpret_cast<const RGBA*>(v.px(std::clamp(x, v.bounds.x1, v.bounds.x2 - 1), std::clamp(y, v.bounds.y1, v.bounds.y2 - 1)));
}

RGBA doubleExposureBilinear(const ImageView& v, float sx, float sy) {
    const float fx = sx - 0.5f, fy = sy - 0.5f;
    const float flx = std::floor(fx), fly = std::floor(fy);
    if (flx >= v.bounds.x1 && flx < v.bounds.x2 - 1 && fly >= v.bounds.y1 && fly < v.bounds.y2 - 1) {
        const int x0 = static_cast<int>(flx), y0 = static_cast<int>(fly);
        const float tx = fx - flx, ty = fy - fly;
        const RGBA* r0 = reinterpret_cast<const RGBA*>(v.px(x0, y0));
        const RGBA* r1 = reinterpret_cast<const RGBA*>(v.px(x0, y0 + 1));
        return lerp(lerp(r0[0], r0[1], tx), lerp(r1[0], r1[1], tx), ty);
    }
    return v.bilinear(sx, sy, SampleEdge::Black);
}

float doubleExposureScreen(float a, float b) {
    const float hi = std::max(a, b), lo = std::min(a, b);
    return hi + lo * (1.f - saturate(hi));
}

float doubleExposureAxisCoverage(float s, int lo, int hi) {
    const float v = s - 0.5f;
    const float fl = std::floor(v);
    if (!(fl > -1e6f && fl < 1e6f)) return 0.f;
    const int i0 = static_cast<int>(fl);
    const float t = v - fl;
    float c = 0.f;
    if (i0 >= lo && i0 < hi) c += 1.f - t;
    if (i0 + 1 >= lo && i0 + 1 < hi) c += t;
    return c;
}

}

ModuleDef makeDoubleExposure() {
    ModuleDef m;
    m.id = "double_exposure";
    m.name = "Double Exposure";
    m.family = "Compositing";
    m.description = "Blend a connected Second Source in linear light. With no auxiliary clip, Frame Offset selects a second exposure from the primary clip.";
    m.temporal = true;
    m.usesEncoding = true;
    m.params = {
        Int("frame_offset", "Frame Offset", -12, -64, 64, -64, 64, Unit::Frames,
            "Frame the second exposure is taken from, relative to the current frame. Negative = earlier."),
        Dbl("exposure", "Second Exposure", 0.0, -6.0, 6.0, -6.0, 6.0, Unit::None,
            "Gain on the second exposure in stops (x2 per stop)."),
        Choice("blend", "Blend", {"Add", "Screen", "Lighten", "Multiply", "Average"}, 4,
               "How the two exposures combine, in linear light with premultiplied alpha. Screen keeps HDR highlights instead of folding them."),
        Dbl("balance", "Balance", 0.5, 0.0, 1.0, 0.0, 1.0, Unit::Normalized,
            "Average: weight of the second exposure (0 = first only, 1 = second only). Other blends: below 0.5 fades the second exposure out, above 0.5 brightens it (x2 at 1)."),
        Choice("key_source", "Key Source", {"None", "Current Luma", "Second Luma"}, 0,
               "Silhouette key masking the second exposure. The key is 1 inside dark areas (luma below Threshold), like unexposed film taking a second exposure."),
        Dbl("key_threshold", "Key Threshold", 0.25, 0.0, 2.0, 0.0, 1.0, Unit::None,
            "Linear luma where the key switches from silhouette (below) to background (above)."),
        Dbl("feather", "Key Feather", 0.1, 0.0, 1.0, 0.0, 1.0, Unit::None,
            "Width in linear luma of the soft transition around the threshold."),
        Bool("invert_key", "Invert Key", false, "Show the second exposure in the bright areas instead of the dark silhouette."),
        Dbl("key_blur", "Key Blur", 0.0, 0.0, 200.0, 0.0, 200.0, Unit::Pixels,
            "Gaussian blur radius of the key matte in pixels (sigma = radius / 2)."),
        Dbl("second_scale", "Second Scale", 1.0, 0.1, 4.0, 0.25, 2.0, Unit::None,
            "Scale of the second exposure about its center."),
        Dbl("second_rotation", "Second Rotation", 0.0, -360.0, 360.0, -180.0, 180.0, Unit::Degrees,
            "Rotation of the second exposure about its center."),
        Point("second_offset", "Second Position", 0.5, 0.5, "Where the center of the second exposure is placed. 0.5, 0.5 = centered."),
        Bool("flip_horizontal", "Flip Horizontal", false, "Mirror the second exposure left to right."),
    };
    m.framesNeeded = [](const ParamValues& p, double) { return doubleExposureWindow(p); };
    m.render = [](const RenderContext& c) {
        const ParamValues& p = c.p();
        const int off = std::clamp(p.i("frame_offset"), -kMaxTemporalWindow, kMaxTemporalWindow);
        const ImageView A = c.source();
        const ImageView B = c.secondSource.valid() ? c.secondSource : (off == 0 ? A : c.frameOffset(off));
        const int blend = p.i("blend");
        const int keySrc = p.i("key_source");
        const float gain = std::exp2(clampf(p.f("exposure"), -6.f, 6.f));
        const float balance = saturate(p.f("balance"));
        const float thr = clampf(p.f("key_threshold"), 0.f, 2.f);
        const float feather = saturate(p.f("feather"));
        const bool invertKey = p.b("invert_key");

        const double scl = std::clamp(p.d("second_scale"), 0.1, 4.0);
        const double rotDeg = std::clamp(p.d("second_rotation"), -360.0, 360.0);
        const bool flip = p.b("flip_horizontal");
        double tx, ty;
        c.pointToPixels(p.vec("second_offset"), tx, ty);
        const double fcx = (c.frame.x1 + c.frame.x2) * 0.5, fcy = (c.frame.y1 + c.frame.y2) * 0.5;
        const bool identity = scl == 1.0 && std::fmod(rotDeg, 360.0) == 0.0 && !flip && std::fabs(tx - fcx) < 1e-9 && std::fabs(ty - fcy) < 1e-9;
        const Affine fwd = Affine::translate(tx, ty) * Affine::rotate(rotDeg * kPi / 180.0) * Affine::scale(flip ? -scl : scl, scl) * Affine::translate(-fcx, -fcy);
        const Affine inv = fwd.inverse();

        const Encoding enc = c.encoding;
        auto secondRaw = [&](int x, int y, const float* bRow, float& cov) -> RGBA {
            if (!B.valid()) { cov = 0.f; return {}; }
            if (identity) {
                cov = doubleExposureInRow(B, bRow, x) ? 1.f : 0.f;
                return doubleExposureTap(B, bRow, x);
            }
            double sx, sy;
            inv.apply(x + 0.5, y + 0.5, sx, sy);
            cov = doubleExposureAxisCoverage(static_cast<float>(sx), B.bounds.x1, B.bounds.x2) *
                  doubleExposureAxisCoverage(static_cast<float>(sy), B.bounds.y1, B.bounds.y2);
            return doubleExposureBilinear(B, static_cast<float>(sx), static_cast<float>(sy));
        };
        auto keyOf = [&](float L) -> float {
            float k = 1.f - smoothstepf(thr - feather * 0.5f, thr + feather * 0.5f, finiteOr(L, 0.f));
            return invertKey ? 1.f - k : k;
        };
        auto rawKey = [&](int x, int y, const float* bRow) -> float {
            if (keySrc == 0) return 1.f;
            if (keySrc == 1) return keyOf(luma(decode(doubleExposureClamped(A, x, y), enc)));
            float cv;
            const RGBA b = secondRaw(x, y, bRow, cv);
            return keyOf(luma(B.valid() ? decode(b, enc) : b));
        };

        const float fw = static_cast<float>(std::max(1, c.frame.w())), fh = static_cast<float>(std::max(1, c.frame.h()));
        const float sigma = std::min(c.px(clampf(p.f("key_blur"), 0.f, 200.f)) * 0.5f, std::max(fw, fh));
        const bool useBuffer = keySrc != 0 && sigma > 0.35f;
        OwnedImage keyImg;
        if (useBuffer) {
            const int pad = static_cast<int>(std::ceil(sigma * 3.f)) + 2;
            Rect r = intersect(Rect{c.window.x1 - pad, c.window.y1 - pad, c.window.x2 + pad, c.window.y2 + pad}, c.frame);
            if (r.empty()) r = c.window;
            r = Rect{std::min(r.x1, c.window.x1), std::min(r.y1, c.window.y1), std::max(r.x2, c.window.x2), std::max(r.y2, c.window.y2)};
            keyImg.reset(r);
            parallelRows(r.y1, r.y2, [&](int y) {
                const float* bRow = identity ? doubleExposureRow(B, y) : nullptr;
                for (int x = r.x1; x < r.x2; ++x) keyImg.view.px(x, y)[0] = rawKey(x, y, bRow);
            });
            gaussianBlur(keyImg, sigma);
        }

        const float k2 = 2.f * balance;
        parallelRows(c.window.y1, c.window.y2, [&](int y) {
            constexpr int kChunk = 256;
            RGBA bv[kChunk];
            float cov[kChunk];
            const float* aRow = doubleExposureRow(A, y);
            const float* bRow = identity ? doubleExposureRow(B, y) : nullptr;
            const float* keyRow = useBuffer ? keyImg.view.px(c.window.x1, y) : nullptr;
            RGBA* out = reinterpret_cast<RGBA*>(c.dst.px(c.window.x1, y));
            for (int xa = c.window.x1; xa < c.window.x2; xa += kChunk) {
                const int cnt = std::min(kChunk, c.window.x2 - xa);
                RGBA* av = out + (xa - c.window.x1);
                for (int i = 0; i < cnt; ++i) {
                    av[i] = doubleExposureTap(A, aRow, xa + i);
                    bv[i] = secondRaw(xa + i, y, bRow, cov[i]);
                }
                decodeRow(av, av, cnt, enc);
                if (B.valid()) decodeRow(bv, bv, cnt, enc);
                for (int i = 0; i < cnt; ++i) {
                    const int x = xa + i;
                    const RGBA a = av[i];
                    RGBA b = bv[i];
                    const float key = saturate(useBuffer ? keyRow[4 * (x - c.window.x1)]
                                             : keySrc == 1 && doubleExposureInRow(A, aRow, x) ? keyOf(luma(a))
                                             : keySrc == 2 ? keyOf(luma(b))
                                             : rawKey(x, y, bRow));
                    b.r *= gain; b.g *= gain; b.b *= gain;
                    RGBA o;
                    if (blend == 4) {
                        o = a + (b - a * cov[i]) * (balance * key);
                    } else {
                        const float wgt = k2 * key;
                        RGBA s{b.r * wgt, b.g * wgt, b.b * wgt, b.a * std::min(1.f, wgt)};
                        const float aa = saturate(a.a), sa = saturate(s.a);
                        const float union_ = aa + sa - aa * sa;
                        if (blend == 0) {
                            o = RGBA{a.r + s.r, a.g + s.g, a.b + s.b, union_};
                        } else {
                            const float ia = aa > 1e-6f ? 1.f / aa : 0.f, is = sa > 1e-6f ? 1.f / sa : 0.f;
                            for (int ch = 0; ch < 3; ++ch) {
                                const float ca = a[ch] * ia, cs = s[ch] * is;
                                float f;
                                if (blend == 1) f = doubleExposureScreen(ca, cs);
                                else if (blend == 2) f = std::max(ca, cs);
                                else f = ca * cs;
                                o[ch] = s[ch] * (1.f - aa) + a[ch] * (1.f - sa) + aa * sa * f;
                            }
                            o.a = union_;
                        }
                    }
                    av[i] = o;
                }
                encodeRow(av, av, cnt, enc);
            }
        });
    };
    return m;
}

}
