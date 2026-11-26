#include "modules.h"
#include "scatter_ops.h"
#include "../gpu/gpu.h"

namespace tp {

namespace {

DlLevel dlBlurLevel(const ImageView& sel, float sigma) {
    const int f = std::clamp(static_cast<int>(std::floor(sigma / 2.5f)), 1, 64);
    DlLevel L = dlDown(sel, f);
    gaussianBlur(L.img, sigma / static_cast<float>(L.f));
    return L;
}

RGBA dlTint(const RGBA& v, const RGBA& t) { return {v.r * t.r, v.g * t.g, v.b * t.b, v.a}; }

}

ModuleDef makeDiffusionLab() {
    ModuleDef m;
    m.id = "diffusion_lab";
    m.cuda = CudaMode::Device;
    m.name = "Diffusion Lab";
    m.family = "Optics";
    m.description = "Multi-scale, highlight-selective light scatter emulating diffusion filters: white mist, black mist, glimmer stars, fog and anamorphic streaks. Computed in linear light.";
    m.temporal = false;
    m.usesEncoding = true;
    m.params = {
        Choice("mode", "Mode", {"White Mist", "Black Mist", "Glimmer", "Fog", "Streak", "Pearl", "Bloom", "Black Soft", "Bronze Glimmer", "Petroleum Jelly"}, 0,
               "White Mist: energy-preserving scatter of the whole image. Black Mist: only highlights scatter, shadows stay dense. Glimmer: star glints plus tight bloom. Fog: very wide scatter with a tinted veil. Streak: anamorphic directional streaks from highlights. Pearl: pearlescent mist, cool in the shadows and warm in the highlights. Bloom: wide additive glow from highlights. Black Soft: softens mid-tone detail while keeping blacks and highlights crisp, no halo. Bronze Glimmer: warm star glints. Petroleum Jelly: sharp centre, smeared and glowing frame edges like vaseline on the lens."),
        Dbl("strength", "Strength", 0.35, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Amount of scattered light. 0 returns the source."),
        Dbl("scatter_radius", "Scatter Radius", 30.0, 0.0, 800.0, 0.0, 200.0, Unit::Pixels, "Base scatter radius in full-resolution pixels. Scatter is a sum of three gaussians at 1x, 3x and 9x this radius (weights 0.5, 0.3, 0.2)."),
        Dbl("threshold", "Threshold", 0.6, 0.0, 8.0, 0.0, 2.0, Unit::None, "Linear luma above which light scatters (Black Mist, Glimmer, Streak). Soft knee of half the threshold."),
        Dbl("shadow_protection", "Shadow Protection", 0.5, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Black Mist only: keeps halos out of dark areas. At 1 pure black stays black."),
        Dbl("focal_length", "Focal Length", 50.0, 12.0, 300.0, 12.0, 300.0, Unit::None, "Lens focal length in mm. Artistic approximation: effective radius = scatter radius x (focal length / 50) x sqrt(2.8 / aperture)."),
        Dbl("aperture", "Aperture", 2.8, 1.0, 22.0, 1.0, 22.0, Unit::None, "Lens f-number. Artistic approximation: wider apertures (smaller numbers) scatter more, see Focal Length."),
        Color("tint", "Tint", 1.0, 1.0, 1.0, "Color of the scattered light (the veil color in Fog)."),
        Dbl("streak_angle", "Streak Angle", 0.0, -360.0, 360.0, -180.0, 180.0, Unit::Degrees, "Direction of streaks (Streak) and of the star arms (Glimmer), counter-clockwise from horizontal."),
        Dbl("streak_length", "Streak Length", 200.0, 0.0, 2000.0, 0.0, 600.0, Unit::Pixels, "Half-length of streaks and star arms in full-resolution pixels."),
    };
    m.framesNeeded = [](const ParamValues&, double) { return FrameWindow{}; };
    m.render = [](const RenderContext& c) {
        const ParamValues& p = c.p();
        const ImageView src = c.source();
        const float k = saturate(p.f("strength"));
        if (k <= 0.f || !src.valid()) {
            requireHost(c);
            parallelRows(c.window.y1, c.window.y2, [&](int y) {
                for (int x = c.window.x1; x < c.window.x2; ++x) c.dst.store(x, y, src.at(x, y, SampleEdge::Black));
            });
            return;
        }
        const int mode = p.i("mode");
        const Encoding enc = c.encoding;
        const float thr = std::max(0.f, p.f("threshold"));
        const float sp = saturate(p.f("shadow_protection"));
        const float fl = clampf(p.f("focal_length"), 12.f, 300.f);
        const float ap = clampf(p.f("aperture"), 1.f, 22.f);
        const auto tv = p.vec("tint");
        const RGBA tint{static_cast<float>(tv[0]), static_cast<float>(tv[1]), static_cast<float>(tv[2]), 1.f};
        const float ang = p.f("streak_angle") * kPi / 180.f;
        const float diag = std::sqrt(static_cast<float>(c.frame.w()) * c.frame.w() + static_cast<float>(c.frame.h()) * c.frame.h());
        const float rEff = std::min(std::max(0.f, c.px(p.f("scatter_radius"))) * (fl / 50.f) * std::sqrt(2.8f / ap), diag);
        const float len = std::min(std::max(0.f, c.px(p.f("streak_length"))), diag);

        Rect R = intersect(c.frame, src.bounds);
        if (R.empty()) R = src.bounds;
        if (gpu::available() && gpu::diffusion({src,c.dst,c.frame,R,c.window,mode,static_cast<int>(enc),
                                               thr,rEff,len,ang,diag,k,sp,tint})) return;
        requireHost(c);
        const bool all = mode == 0 || mode == 3 || mode == 5 || mode == 7 || mode == 9;
        OwnedImage sel(R);
        parallelRows(R.y1, R.y2, [&](int y) {
            for (int x = R.x1; x < R.x2; ++x) {
                RGBA lin = decode(src.load(x, y), enc);
                sel.view.store(x, y, all ? lin : dlHighlight(lin, thr));
            }
        });

        std::vector<DlLevel> levels;
        std::vector<float> weights;
        DlLevel star1, star2;
        if (mode == 0 || mode == 1 || mode == 3 || mode == 5 || mode == 6 || mode == 9) {
            const float base = mode == 3 ? rEff * 3.f : (mode == 6 ? rEff * 2.f : (mode == 9 ? rEff * 1.5f : rEff));
            const float mult[3] = {1.f, 3.f, 9.f}, wt[3] = {0.5f, 0.3f, 0.2f};
            for (int i = 0; i < 3; ++i) {
                levels.push_back(dlBlurLevel(sel.view, std::min(base * mult[i], diag) * 0.5f));
                weights.push_back(wt[i]);
            }
            if (mode == 9) star1 = dlDirBlur(sel.view, ang, len * 0.4f);
        } else if (mode == 7) {
            levels.push_back(dlBlurLevel(sel.view, std::min(rEff * 0.3f, diag) * 0.5f));
            weights.push_back(1.f);
        } else if (mode == 2 || mode == 8) {
            levels.push_back(dlBlurLevel(sel.view, rEff * 0.125f));
            weights.push_back(1.f);
            star1 = dlDirBlur(sel.view, ang, len);
            star2 = dlDirBlur(sel.view, ang + kPi * 0.5f, len);
        } else {
            star1 = dlDirBlur(sel.view, ang, len);
        }

        float lumaMean = 0.f;
        if (mode == 3) {
            std::vector<double> rows(static_cast<size_t>(R.h()), 0.0);
            parallelRows(R.y1, R.y2, [&](int y) {
                double s = 0;
                for (int x = R.x1; x < R.x2; ++x) s += std::max(0.f, luma(sel.view.load(x, y)));
                rows[y - R.y1] = s;
            });
            double tot = 0;
            for (double v : rows) tot += v;
            lumaMean = static_cast<float>(tot / std::max(1.0, static_cast<double>(R.w()) * R.h()));
            lumaMean = std::min(lumaMean, 1e4f);
        }

        parallelRows(c.window.y1, c.window.y2, [&](int y) {
            for (int x = c.window.x1; x < c.window.x2; ++x) {
                RGBA lin = decode(src.at(x, y, SampleEdge::Black), enc);
                RGBA S;
                for (size_t i = 0; i < levels.size(); ++i) S += levels[i].sample(R, x, y) * weights[i];
                RGBA o;
                switch (mode) {
                case 0: {
                    RGBA t = dlTint(S, tint);
                    o = lin * (1.f - k) + t * k;
                    break;
                }
                case 1: {
                    float protect = lerpf(1.f, smoothstepf(0.f, 0.18f, luma(lin)), sp);
                    RGBA h = dlTint(S, tint) * (k * protect);
                    o = RGBA{lin.r + h.r, lin.g + h.g, lin.b + h.b, lin.a};
                    break;
                }
                case 2:
                case 8: {
                    RGBA st = (star1.sample(R, x, y, SampleEdge::Black) + star2.sample(R, x, y, SampleEdge::Black)) * 0.5f;
                    RGBA h = dlTint(st * 2.f * 0.6f + S * 0.4f, tint) * k;
                    if (mode == 8) h = RGBA{h.r * 1.0f, h.g * 0.72f, h.b * 0.42f, h.a};
                    o = RGBA{lin.r + h.r, lin.g + h.g, lin.b + h.b, lin.a};
                    break;
                }
                case 5: {
                    float ls = saturate(luma(S));
                    RGBA pt{lerpf(0.95f, 1.05f, ls), lerpf(0.98f, 1.0f, ls), lerpf(1.06f, 0.94f, ls), 1.f};
                    o = lin * (1.f - k) + dlTint(dlTint(S, pt), tint) * k;
                    o.a = lin.a;
                    break;
                }
                case 6: {
                    RGBA h = dlTint(S, tint) * (1.5f * k);
                    o = RGBA{lin.r + h.r, lin.g + h.g, lin.b + h.b, lin.a};
                    break;
                }
                case 7: {
                    float l = luma(lin);
                    float mids = smoothstepf(0.02f, 0.15f, l) * (1.f - smoothstepf(0.6f, 1.2f, l));
                    o = lerp(lin, S, k * mids);
                    o.a = lin.a;
                    break;
                }
                case 9: {
                    float nx = (x + 0.5f - (c.frame.x1 + 0.5f * c.frame.w())) / (0.5f * c.frame.w());
                    float ny = (y + 0.5f - (c.frame.y1 + 0.5f * c.frame.h())) / (0.5f * c.frame.h());
                    float m = smoothstepf(0.25f, 1.1f, std::sqrt(nx * nx + ny * ny));
                    RGBA smear = star1.sample(R, x, y, SampleEdge::Clamp) * 0.5f + S * 0.5f;
                    o = lerp(lin, dlTint(smear, tint), saturate(m * k * 1.6f));
                    o = o + dlTint(S, tint) * (0.15f * k * m);
                    o.a = lin.a;
                    break;
                }
                case 3: {
                    RGBA veil{tint.r * lumaMean, tint.g * lumaMean, tint.b * lumaMean, lin.a};
                    o = lin * (1.f - k * 0.6f) + (S * 0.7f + veil * 0.3f) * k;
                    o.a = lerpf(lin.a, S.a, 0.7f * k);
                    break;
                }
                default: {
                    RGBA h = dlTint(star1.sample(R, x, y, SampleEdge::Black), tint) * (2.f * k);
                    o = RGBA{lin.r + h.r, lin.g + h.g, lin.b + h.b, lin.a};
                    break;
                }
                }
                c.dst.store(x, y, encode(o, enc));
            }
        });
    };
    return m;
}

}
