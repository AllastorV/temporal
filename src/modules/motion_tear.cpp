#include "modules.h"
#include "../core/flow.h"
#include "../gpu/gpu.h"

namespace tp {

ModuleDef makeMotionTear() {
    ModuleDef m;
    m.id = "motion_tear";
    m.cuda = CudaMode::Bridged;
    m.name = "Motion Tear";
    m.family = "Temporal";
    m.description = "Flow-driven slow-shutter tear: every pixel is exposed along its own optical-flow motion vector, so moving subjects drag a leading or trailing exposure while the background stays sharp; highlights streak harder and can be warmed.";
    m.temporal = true;
    m.usesEncoding = true;
    m.params = {
        Choice("mode", "Mode", {"From Motion", "Fixed Direction"}, 0,
               "From Motion: optical flow between the previous and current frame drives the tear. Fixed Direction: every pixel tears along Angle by Shutter frames of a constant speed (no analysis)."),
        Dbl("shutter", "Shutter", 4.0, 0.0, 32.0, 0.0, 16.0, Unit::Frames, "Exposure length in frames of motion: the streak covers this many frames worth of each pixel's own movement."),
        Dbl("amount", "Amount", 1.0, 0.0, 4.0, 0.0, 4.0, Unit::None, "Overall strength of the tear (multiplies the streak length and its opacity up to 1)."),
        Choice("sharp_at", "Sharp At", {"End", "Middle", "Start"}, 0,
               "Where the sharp image sits on the streak: End = trailing tear behind the subject, Start = leading tear ahead of it, Middle = both ways."),
        Dbl("softness", "Softness", 0.5, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Falloff along the streak: 0 keeps a hard uniform exposure, 1 fades smoothly toward the far end."),
        Dbl("highlights", "Highlights", 0.4, 0.0, 4.0, 0.0, 2.0, Unit::None, "Linear luma above which samples streak extra bright (light response); 0 disables."),
        Dbl("warmth", "Warmth", 0.0, -1.0, 1.0, -1.0, 1.0, Unit::Normalized, "Colour of the tear: positive warms the trail, negative cools it."),
        Dbl("motion_bias", "Motion Bias", 1.0, 0.0, 4.0, 0.0, 4.0, Unit::None, "Exaggerates or calms the measured motion before tearing."),
        Dbl("threshold", "Motion Threshold", 0.3, 0.0, 10.0, 0.0, 4.0, Unit::Pixels, "Motion below this many pixels per frame is treated as static (keeps a still background crisp)."),
        Dbl("angle", "Angle", 0.0, -360.0, 360.0, -180.0, 180.0, Unit::Degrees, "Fixed Direction only: direction of the motion that tears."),
        Dbl("speed", "Speed", 8.0, 0.0, 200.0, 0.0, 60.0, Unit::Pixels, "Fixed Direction only: assumed motion in full-resolution pixels per frame."),
        Int("samples", "Samples", 24, 4, 96, 4, 64, Unit::None, "Exposure samples along each streak; higher is smoother and slower."),
        Choice("quality", "Analysis", {"Draft", "Standard", "Fine"}, 1, "Optical-flow resolution and iterations."),
    };
    m.framesNeeded = [](const ParamValues& p, double) {
        FrameWindow w;
        if (p.i("mode") == 0) w.before = 1;
        return w;
    };
    m.render = [](const RenderContext& c) {
        const ParamValues& p = c.p();
        const ImageView cur = c.source();
        if (!cur.valid()) {
            parallelRows(c.window.y1, c.window.y2, [&](int y) { for (int x = c.window.x1; x < c.window.x2; ++x) c.dst.store(x, y, RGBA{}); });
            return;
        }
        const Encoding enc = c.encoding;
        const int mode = p.i("mode");
        const float diag = std::max(1.f, std::hypot(static_cast<float>(c.frame.w()), static_cast<float>(c.frame.h())));
        const float amount = clampf(p.f("amount"), 0.f, 4.f);
        const float shutter = clampf(p.f("shutter"), 0.f, 32.f) * amount;
        const int sharp = p.i("sharp_at");
        const float soft = saturate(p.f("softness"));
        const float hl = std::max(0.f, p.f("highlights"));
        const float warm = clampf(p.f("warmth"), -1.f, 1.f);
        const float bias = clampf(p.f("motion_bias"), 0.f, 4.f);
        const float thr = std::max(0.f, c.px(p.f("threshold")));
        const int N = std::clamp(p.i("samples"), 4, 96);
        Rect R = intersect(c.frame, cur.bounds);
        if (R.empty()) R = cur.bounds;
        FlowField flow;
        if (mode == 0) {
            ImageView prev = c.frameOffset(-1);
            const float analysisScale = c.preview ? std::max(float(c.scale), std::min(R.w(),R.h())/180.f) : float(c.scale);
            if (prev.valid()) flow = computeFlow(prev, cur, R, analysisScale, enc, std::clamp(p.i("quality"), 0, 2));
        }
        const float fa = p.f("angle") * kPi / 180.f;
        const float fdx = std::cos(fa) * c.px(p.f("speed")), fdy = std::sin(fa) * c.px(p.f("speed"));
        OwnedImage lin(R);
        parallelRows(R.y1, R.y2, [&](int y) { for (int x = R.x1; x < R.x2; ++x) lin.view.store(x, y, decode(cur.load(x, y), enc)); });
        const float s0 = sharp == 0 ? 0.f : (sharp == 1 ? -0.5f : -1.f);
        const int tile = std::max(4, static_cast<int>(std::lround(16.0 * c.scale)));
        const int tw = (R.w() + tile - 1) / tile, th = (R.h() + tile - 1) / tile;
        std::vector<float> tmu(static_cast<size_t>(tw) * th, 0.f), tmv(tmu.size(), 0.f), nmu(tmu.size(), 0.f), nmv(tmu.size(), 0.f);
        if (mode == 0 && flow.w > 0) {
            parallelRows(0, th, [&](int j) {
                for (int i = 0; i < tw; ++i) {
                    float bu = 0.f, bv = 0.f, bm = 0.f;
                    for (int y = R.y1 + j * tile; y < std::min(R.y2, R.y1 + (j + 1) * tile); y += 2)
                        for (int x = R.x1 + i * tile; x < std::min(R.x2, R.x1 + (i + 1) * tile); x += 2) {
                            float u, v;
                            flow.sample(x + 0.5f, y + 0.5f, u, v);
                            float mm = u * u + v * v;
                            if (mm > bm) { bm = mm; bu = u; bv = v; }
                        }
                    tmu[static_cast<size_t>(j) * tw + i] = bu; tmv[static_cast<size_t>(j) * tw + i] = bv;
                }
            });
            parallelRows(0, th, [&](int j) {
                for (int i = 0; i < tw; ++i) {
                    float bu = 0.f, bv = 0.f, weight = 0.f;
                    for (int dj = -1; dj <= 1; ++dj)
                        for (int di = -1; di <= 1; ++di) {
                            int ii = std::clamp(i + di, 0, tw - 1), jj = std::clamp(j + dj, 0, th - 1);
                            float u = tmu[static_cast<size_t>(jj) * tw + ii], v = tmv[static_cast<size_t>(jj) * tw + ii];
                            // Continuous weights prevent neighbouring lights swapping the winning vector.
                            float w = std::min(std::hypot(u,v), static_cast<float>(tile)) / (1.f+di*di+dj*dj);
                            bu += u*w; bv += v*w; weight += w;
                        }
                    if (weight > 0.f) {bu/=weight; bv/=weight;}
                    nmu[static_cast<size_t>(j) * tw + i] = bu; nmv[static_cast<size_t>(j) * tw + i] = bv;
                }
            });
        }
        // Wide sample spacing aliases repeated lights. Prefilter only the long exposure taps.
        OwnedImage filtered;
        if (shutter * std::max(1.f,bias) > 5.f) {
            filtered = lin;
            gaussianBlur(filtered, std::max(0.8f,c.px(1.4f)));
        }
        const RGBA warmTint{1.f + 0.35f * warm, 1.f + 0.05f * warm, 1.f - 0.4f * warm, 1.f};
        if (gpu::available()) {
            const gpu::TearVectorGrid measured{flow.u.data(), flow.v.data(), flow.w, flow.h,
                                               flow.ox, flow.oy, flow.scale, flow.scale};
            const gpu::TearVectorGrid nearby{nmu.data(), nmv.data(), tw, th, R.x1, R.y1,
                                             static_cast<float>(tile), 1.f};
            const gpu::TearExposure exposure{fdx, fdy, bias, thr, shutter, diag, s0, soft,
                                             hl, amount, warmTint, N, static_cast<int>(enc), mode == 0};
            if (gpu::motionTear(lin.view, filtered.view, c.dst, c.window, measured, nearby, exposure))
                return;
        }
        parallelRows(c.window.y1, c.window.y2, [&](int y) {
            for (int x = c.window.x1; x < c.window.x2; ++x) {
                const float px = x + 0.5f, py = y + 0.5f;
                float vx = fdx, vy = fdy;
                if (mode == 0) {
                    flow.sample(px, py, vx, vy);
                    float gx = (px - R.x1) / tile - 0.5f, gy = (py - R.y1) / tile - 0.5f;
                    float flx = std::floor(gx), fly = std::floor(gy);
                    int i0 = static_cast<int>(flx), j0 = static_cast<int>(fly);
                    float tx = gx - flx, ty = gy - fly;
                    auto T = [&](const std::vector<float>& a, int i, int j) { return a[static_cast<size_t>(std::clamp(j, 0, th - 1)) * tw + std::clamp(i, 0, tw - 1)]; };
                    float nu = lerpf(lerpf(T(nmu, i0, j0), T(nmu, i0 + 1, j0), tx), lerpf(T(nmu, i0, j0 + 1), T(nmu, i0 + 1, j0 + 1), tx), ty);
                    float nv = lerpf(lerpf(T(nmv, i0, j0), T(nmv, i0 + 1, j0), tx), lerpf(T(nmv, i0, j0 + 1), T(nmv, i0 + 1, j0 + 1), tx), ty);
                    float own = std::hypot(vx, vy), nm = std::hypot(nu, nv);
                    float wOwn = nm > 1e-4f ? smoothstepf(0.2f, 0.6f, own / nm) : 1.f;
                    vx = lerpf(nu, vx, wOwn); vy = lerpf(nv, vy, wOwn);
                }
                vx *= bias; vy *= bias;
                const float mag = std::hypot(vx, vy);
                const RGBA base = lin.view.at(x, y, SampleEdge::Black);
                if (!(mag > thr) || shutter <= 0.f) { c.dst.store(x, y, encode(base, enc)); continue; }
                float lx = vx * shutter, ly = vy * shutter;
                const float len = std::hypot(lx, ly);
                if (len > diag) { lx *= diag / len; ly *= diag / len; }
                const float gate = smoothstepf(thr, thr * 2.f + 0.5f, mag);
                RGBA acc;
                float wsum = 0.f;
                RGBA hot;
                const int Np = std::clamp(static_cast<int>(std::ceil(std::hypot(lx, ly) / 1.5f)), 4, N);
                float footprint = std::hypot(lx,ly)/Np;
                float lowpass = filtered.view.valid() ? smoothstepf(1.5f,3.5f,footprint) : 0.f;
                for (int k = 0; k < Np; ++k) {
                    const float s = s0 + (k + 0.5f) / Np;
                    const float d = std::fabs(s);
                    const float w = lerpf(1.f, std::exp(-3.f * d), soft);
                    RGBA smp = lin.view.bilinear(px + lx * s, py + ly * s, SampleEdge::Clamp);
                    if (lowpass > 0.f) smp=lerp(smp,filtered.view.bilinear(px+lx*s,py+ly*s,SampleEdge::Clamp),lowpass);
                    acc += smp * w;
                    wsum += w;
                    if (hl > 0.f) {
                        float l = luma(smp);
                        float e = std::max(0.f, l - hl);
                        hot += smp * (w * e / std::max(l, 1e-4f));
                    }
                }
                RGBA trail = acc * (1.f / wsum);
                trail = RGBA{trail.r * warmTint.r, trail.g * warmTint.g, trail.b * warmTint.b, trail.a};
                hot = hot * (amount / std::max(1.f, wsum * 0.25f));
                RGBA o = lerp(base, trail, saturate(gate * std::min(1.f, amount)));
                o = RGBA{o.r + hot.r * warmTint.r, o.g + hot.g * warmTint.g, o.b + hot.b * warmTint.b, o.a};
                c.dst.store(x, y, encode(o, enc));
            }
        });
    };
    return m;
}

}
