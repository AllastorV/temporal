#include "modules.h"

namespace tp {

namespace {

const RGBA kSlowShutterBlack{};

const float* slowShutterRow(const ImageView& v, int y) {
    return v.valid() && y >= v.bounds.y1 && y < v.bounds.y2 ? v.px(v.bounds.x1, y) : nullptr;
}

const RGBA& slowShutterTap(const ImageView& v, const float* row, int x) {
    return row && x >= v.bounds.x1 && x < v.bounds.x2 ? reinterpret_cast<const RGBA*>(row)[x - v.bounds.x1] : kSlowShutterBlack;
}

struct SlowShutterSpan { double lo, hi, period; };

SlowShutterSpan slowShutterSpan(const ParamValues& p, double fps) {
    double f = std::isfinite(fps) && fps > 0 ? fps : 24.0;
    double vfps = std::clamp(p.d("virtual_fps"), 1.0, 120.0);
    double ang = std::clamp(p.d("shutter_angle"), 1.0, 360.0);
    double period = std::min(f / vfps, 1e4);
    double dur = std::min(static_cast<double>(kMaxTemporalWindow), period * ang / 360.0);
    int sync = p.i("sync");
    if (sync == 1) return {-dur * 0.5, dur * 0.5, period};
    if (sync == 2) return {0.0, dur, period};
    return {-dur, 0.0, period};
}

FrameWindow slowShutterWindow(const ParamValues& p, double fps) {
    SlowShutterSpan s = slowShutterSpan(p, fps);
    double back = -s.lo + (p.b("stepped") ? s.period : 0.0);
    FrameWindow w;
    w.before = static_cast<int>(std::ceil(std::min(back, 1e3) - 1e-9));
    w.after = static_cast<int>(std::ceil(s.hi - 1e-9));
    return clampWindow(w);
}

}

ModuleDef makeSlowShutter() {
    ModuleDef m;
    m.id = "slow_shutter";
    m.name = "Slow Shutter";
    m.family = "Temporal";
    m.description = "Simulates a slower virtual camera by accumulating exposure over several source frames: smooth motion blur or crisp decaying trails.";
    m.temporal = true;
    m.usesEncoding = true;
    m.params = {
        Choice("mode", "Mode", {"Smooth", "Hard Trails"}, 0,
               "Smooth: normalized weighted average over the exposure (motion blur). Hard Trails: the average plus crisp ghosts, each frame weighted decay^age and combined with lighten."),
        Dbl("virtual_fps", "Virtual FPS", 6.0, 1.0, 120.0, 1.0, 60.0, Unit::FPS, "Frame rate of the simulated camera. Lower means a longer exposure per virtual frame."),
        Dbl("shutter_angle", "Shutter Angle", 360.0, 1.0, 360.0, 1.0, 360.0, Unit::Degrees, "Fraction of the virtual frame period the shutter is open (360 = fully open). Exposure = source fps / virtual fps x angle / 360 frames, capped at 64."),
        Int("sample_count", "Samples", 16, 1, 64, 1, 64, Unit::None, "Maximum number of time samples across the exposure; fewer than the exposure length gives an evenly spaced subset."),
        Choice("sync", "Sync", {"Trailing", "Centered", "Leading"}, 0, "Exposure interval ends at, is centered on, or starts at the current frame."),
        Bool("stepped", "Stepped", false, "Hold each virtual frame so the result steps at Virtual FPS like a real low frame rate camera (anchored at frame 0)."),
        Dbl("decay", "Decay", 0.85, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Hard Trails only: brightness multiplier per frame of age away from the current frame."),
        Dbl("softness", "Softness", 0.3, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Shutter taper: 0 is a hard box exposure, 1 a full cosine fade in and out across the interval."),
        Dbl("exposure_curve", "Exposure Curve", 1.0, 0.25, 4.0, 0.25, 4.0, Unit::None, "Weight shaping position^(curve - 1) along the interval (0 = earliest, 1 = latest): above 1 favors later samples, below 1 earlier ones, 1 is neutral."),
        Dbl("exposure", "Exposure", 0.0, -4.0, 4.0, -4.0, 4.0, Unit::None, "Gain on the result in stops."),
    };
    m.framesNeeded = [](const ParamValues& p, double fps) { return slowShutterWindow(p, fps); };
    m.render = [](const RenderContext& c) {
        const ParamValues& p = c.p();
        const bool hard = p.i("mode") == 1;
        const SlowShutterSpan s = slowShutterSpan(p, c.fps);
        const FrameWindow win = slowShutterWindow(p, c.fps);
        const double rt = std::round(c.time);
        const double ref = p.b("stepped") ? std::floor(c.time / s.period + 1e-9) * s.period : c.time;
        const double a = ref + s.lo, b = ref + s.hi;
        const int n = std::clamp(std::min(p.i("sample_count"), static_cast<int>(std::ceil(b - a - 1e-9))), 1, 64);
        const float soft = saturate(p.f("softness"));
        const float curve = clampf(p.f("exposure_curve"), 0.25f, 4.f);
        const float decay = saturate(p.f("decay"));
        const float gain = std::exp2(clampf(p.f("exposure"), -4.f, 4.f));
        const int lo = static_cast<int>(rt) - win.before, hi = static_cast<int>(rt) + win.after;

        std::vector<double> weight(static_cast<size_t>(hi - lo + 1), 0.0);
        for (int i = 0; i < n; ++i) {
            double u = (i + 0.5) / n;
            double t = std::clamp(a + u * (b - a), static_cast<double>(lo), static_cast<double>(hi));
            double hann = 0.5 - 0.5 * std::cos(2.0 * kPi * u);
            double w = (1.0 - soft * (1.0 - hann)) * std::pow(u, curve - 1.0);
            double f0 = std::floor(t);
            double fr = t - f0;
            int k = static_cast<int>(f0) - lo;
            weight[static_cast<size_t>(k)] += w * (1.0 - fr);
            if (fr > 0.0 && k + 1 < static_cast<int>(weight.size())) weight[static_cast<size_t>(k + 1)] += w * fr;
        }
        double total = 0.0;
        for (double w : weight) total += w;
        if (!(total > 0.0)) { weight.assign(weight.size(), 0.0); weight[static_cast<size_t>(rt - lo)] = 1.0; total = 1.0; }

        struct Tap { ImageView img; float w, trail; };
        std::vector<Tap> taps;
        for (int k = 0; k < static_cast<int>(weight.size()); ++k) {
            if (weight[static_cast<size_t>(k)] <= 0.0) continue;
            double age = std::fabs(lo + k - ref);
            taps.push_back({c.frameAt(lo + k), static_cast<float>(weight[static_cast<size_t>(k)] / total), static_cast<float>(std::pow(decay, age))});
        }
        const Encoding enc = c.encoding;
        parallelRows(c.window.y1, c.window.y2, [&](int y) {
            constexpr int kChunk = 256;
            RGBA v[kChunk], lit[kChunk];
            RGBA* out = reinterpret_cast<RGBA*>(c.dst.px(c.window.x1, y));
            for (int xa = c.window.x1; xa < c.window.x2; xa += kChunk) {
                const int cnt = std::min(kChunk, c.window.x2 - xa);
                RGBA* base = out + (xa - c.window.x1);
                for (int i = 0; i < cnt; ++i) {
                    base[i] = RGBA{};
                    lit[i] = RGBA{-3e38f, -3e38f, -3e38f, -3e38f};
                }
                for (const Tap& t : taps) {
                    const float* row = slowShutterRow(t.img, y);
                    for (int i = 0; i < cnt; ++i) v[i] = slowShutterTap(t.img, row, xa + i);
                    decodeRow(v, v, cnt, enc);
                    for (int i = 0; i < cnt; ++i) base[i] += v[i] * t.w;
                    if (hard)
                        for (int i = 0; i < cnt; ++i) {
                            const RGBA g = v[i] * t.trail;
                            for (int ch = 0; ch < 4; ++ch) lit[i][ch] = std::max(lit[i][ch], g[ch]);
                        }
                }
                for (int i = 0; i < cnt; ++i) {
                    if (hard)
                        for (int ch = 0; ch < 4; ++ch) base[i][ch] = std::max(base[i][ch], lit[i][ch]);
                    base[i].r *= gain; base[i].g *= gain; base[i].b *= gain;
                }
                encodeRow(base, base, cnt, enc);
            }
        });
    };
    return m;
}

}
