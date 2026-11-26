#include "modules.h"
#include "../gpu/gpu.h"

namespace tp {

namespace {

RGBA halHighlight(const RGBA& lin, float t, float w) {
    RGBA c{std::max(0.f, lin.r), std::max(0.f, lin.g), std::max(0.f, lin.b), 0.f};
    float l = luma(c);
    if (l <= 1e-8f) return {};
    float s = w <= 1e-6f ? (l > t ? 1.f : 0.f) : smoothstepf(t - w, t + w, l);
    if (s <= 0.f) return {};
    return {c.r * s, c.g * s, c.b * s, 0.f};
}

}

ModuleDef makeHalation() {
    ModuleDef m;
    m.id = "halation";
    m.cuda = CudaMode::Device;
    m.name = "Temporal Halation";
    m.family = "Optics";
    m.description = "Film-style halation with memory: highlights from the current and earlier frames are accumulated with decay, blurred, tinted and added back as glowing exposure trails.";
    m.temporal = true;
    m.usesEncoding = true;
    m.params = {
        Dbl("threshold", "Threshold", 0.5, 0.0, 8.0, 0.0, 2.0, Unit::None, "Linear luma above which light halates; the full brightness of qualifying highlights feeds the glow. 1.0 is diffuse white; values above 1 only catch HDR highlights."),
        Dbl("knee", "Knee", 0.25, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Width of the soft transition around the threshold, as a fraction of the threshold. 0 is a hard cut."),
        Dbl("radius", "Radius", 30.0, 0.0, 800.0, 0.0, 150.0, Unit::Pixels, "Spread of the halation glow in full-resolution pixels (about two gaussian standard deviations)."),
        Dbl("decay", "Decay", 0.55, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Weight multiplier per past frame: frame k back weighs decay^k. 0 uses only the current frame."),
        Int("duration", "Duration", 8, 1, 32, 1, 32, Unit::Frames, "Number of frames (current plus earlier) whose highlights are remembered. Weights are normalized so duration does not change brightness."),
        Choice("quality", "Sampling", {"Fast", "Full"}, 0, "Fast samples four highlight locations per blur cell on large frames. Full integrates every highlight."),
        Color("tint", "Tint", 1.0, 0.35, 0.12, "Color of the halation light, multiplied with the highlights."),
        Dbl("intensity", "Intensity", 1.0, 0.0, 8.0, 0.0, 2.0, Unit::None, "Gain of the added halation light."),
        Bool("halo_only", "Halo Only", false, "Output only the halation layer (alpha from its brightness) for compositing over the plate."),
    };
    m.framesNeeded = [](const ParamValues& p, double) {
        FrameWindow w;
        w.before = p.d("intensity") > 0 && p.d("decay") > 0 ? std::clamp(p.i("duration"), 1, 32) - 1 : 0;
        return w;
    };
    m.render = [](const RenderContext& c) {
        const ParamValues& p = c.p();
        const float thr = std::max(0.f, p.f("threshold"));
        const float kw = saturate(p.f("knee")) * thr;
        const float decay = saturate(p.f("decay"));
        const int dur = std::clamp(p.i("duration"), 1, 32);
        const float inten = std::max(0.f, p.f("intensity"));
        const auto tv = p.vec("tint");
        const RGBA tint{static_cast<float>(tv[0]), static_cast<float>(tv[1]), static_cast<float>(tv[2]), 0.f};
        const bool haloOnly = p.b("halo_only");
        const Encoding enc = c.encoding;

        const ImageView cur = c.source();
        struct Layer { ImageView img; float w; };
        std::vector<Layer> layers;
        float wsum = 0.f, wk = 1.f;
        for (int k = 0; k < (inten > 0.f ? dur : 1); ++k) {
            layers.push_back({k == 0 ? cur : c.frameOffset(-k), wk});
            wsum += wk;
            wk *= decay;
            if (wk < 1e-6f) break;
        }
        for (auto& L : layers) L.w /= wsum;

        Rect R = cur.valid() ? intersect(c.frame, cur.bounds) : c.frame;
        if (R.empty()) R = c.window;
        const float diag = std::sqrt(static_cast<float>(c.frame.w()) * c.frame.w() + static_cast<float>(c.frame.h()) * c.frame.h());
        const float sigma = std::min(std::max(0.f, c.px(p.f("radius"))) * 0.5f, diag);
        const int minCell = c.preview && p.i("quality") == 0 && sigma >= 4.f ?
            std::clamp(R.w()/120, 4, 16) : 1;
        const int f = std::clamp(static_cast<int>(std::floor(sigma / 2.5f)), minCell, 64);
        const int lw = (R.w() + f - 1) / f, lh = (R.h() + f - 1) / f;
        OwnedImage low(Rect{0, 0, lw, lh});

        const bool active = inten > 0.f;
        if (active) {
            std::vector<ImageView> gl;
            std::vector<float> gw;
            for (const auto& L : layers) { gl.push_back(L.img); gw.push_back(L.w); }
            const bool sampled = p.i("quality") == 0 && f >= 4 && R.w() >= 720;
            if (sampled) {
                requireHost(c);
                parallelRows(0, lh, [&](int j) {
                    for (int i = 0; i < lw; ++i) {
                        RGBA sum;
                        for (const auto& layer : layers) {
                            if (!layer.img.valid()) continue;
                            for (int sy : {1, 3}) for (int sx : {1, 3}) {
                                int x = std::min(R.x2 - 1, R.x1 + i * f + sx * f / 4);
                                int y = std::min(R.y2 - 1, R.y1 + j * f + sy * f / 4);
                                sum += halHighlight(decode(layer.img.at(x,y,SampleEdge::Black), enc),thr,kw) * (0.25f * layer.w);
                            }
                        }
                        low.view.store(i,j,sum);
                    }
                });
            } else if (!(gpu::available() && gpu::halationAccum(gl, gw, R, f, thr, kw, static_cast<int>(enc), low.view))) {
                requireHost(c);
                parallelRows(0, lh, [&](int j) {
                    const int ya = R.y1 + j * f, yb = std::min(R.y2, ya + f);
                    for (int i = 0; i < lw; ++i) {
                        const int xa = R.x1 + i * f, xb = std::min(R.x2, xa + f);
                        RGBA acc;
                        for (const auto& L : layers) {
                            if (!L.img.valid()) continue;
                            RGBA s;
                            for (int y = ya; y < yb; ++y)
                                for (int x = xa; x < xb; ++x) s += halHighlight(decode(L.img.at(x, y, SampleEdge::Black), enc), thr, kw);
                            acc += s * L.w;
                        }
                        low.view.store(i, j, acc * (1.f / static_cast<float>((xb - xa) * (yb - ya))));
                    }
                });
            }
            OwnedImage wide = low;
            gaussianBlur(low, sigma / static_cast<float>(f));
            gaussianBlur(wide, std::min(sigma * 3.f, diag) / static_cast<float>(f));
            parallelRows(0, lh, [&](int j) {
                for (int i = 0; i < lw; ++i) low.view.store(i, j, low.view.load(i, j) * 0.6f + wide.view.load(i, j) * 0.4f);
            });
        }

        if (active && gpu::available() && cur.valid() &&
            gpu::halationComposite(cur, low.view, c.dst, c.window, R, f, tint, inten, haloOnly, static_cast<int>(enc))) return;
        requireHost(c);
        const float invf = 1.f / static_cast<float>(f);
        parallelRows(c.window.y1, c.window.y2, [&](int y) {
            for (int x = c.window.x1; x < c.window.x2; ++x) {
                RGBA s = cur.valid() ? cur.at(x, y, SampleEdge::Black) : RGBA{};
                RGBA h;
                if (active) {
                    RGBA b = low.view.bilinear((x - R.x1 + 0.5f) * invf, (y - R.y1 + 0.5f) * invf, SampleEdge::Clamp);
                    h = RGBA{b.r * tint.r, b.g * tint.g, b.b * tint.b, 0.f} * inten;
                }
                if (haloOnly) {
                    h.a = saturate(std::max(h.r, std::max(h.g, h.b)));
                    c.dst.store(x, y, encode(h, enc));
                } else if (h.r == 0.f && h.g == 0.f && h.b == 0.f) {
                    c.dst.store(x, y, s);
                } else {
                    RGBA lin = decode(s, enc);
                    c.dst.store(x, y, encode(RGBA{lin.r + h.r, lin.g + h.g, lin.b + h.b, lin.a}, enc));
                }
            }
        });
    };
    return m;
}

}
