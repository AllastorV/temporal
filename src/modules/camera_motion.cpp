#include "modules.h"
#include "../gpu/gpu.h"

namespace tp {

namespace {

struct CameraMotionSample {
    Affine inv;
};

double cameraMotionEase(double p, int easing) {
    p = std::clamp(p, 0.0, 1.0);
    switch (easing) {
    case 1: return p * p * p;
    case 2: { double q = 1.0 - p; return 1.0 - q * q * q; }
    case 3: return p < 0.5 ? 4.0 * p * p * p : 1.0 - std::pow(-2.0 * p + 2.0, 3.0) * 0.5;
    case 4: {
        const double k = 1.0 / (1.0 - std::exp2(-10.0));
        if (p < 0.5) return 0.5 * (std::exp2(20.0 * p - 10.0) - std::exp2(-10.0)) * k;
        return 1.0 - 0.5 * (std::exp2(-20.0 * p + 10.0) - std::exp2(-10.0)) * k;
    }
    default: return p;
    }
}

double cameraMotionRebound(double q, double rebound) {
    if (q <= 0.0 || q >= 1.0 || rebound <= 0.0) return 0.0;
    return rebound * 0.6 * std::sin(2.0 * kPi * q) * std::exp(-4.0 * q) * (1.0 - q);
}


}

ModuleDef makeCameraMotion() {
    ModuleDef m;
    m.id = "camera_motion";
    m.cuda = CudaMode::Native;
    m.name = "Camera Motion Lab";
    m.family = "Motion";
    m.description = "Virtual 2D camera moves on the current frame: snap zoom, push, whip pan, handheld shake and a subject-safe single-plate dolly approximation, with eased timing, rebound and sub-frame motion blur.";
    m.temporal = false;
    m.usesEncoding = true;
    m.params = {
        Choice("mode", "Mode", {"Snap Zoom", "Push", "Whip", "Handheld", "Dolly Zoom", "Slow Pull", "Snap and Back", "Punch In", "Pulse", "Reveal", "Arc In", "Crane Rise"}, 0,
               "Snap Zoom: eased zoom to Zoom. Push: slow linear dolly-in. Whip: eased pan of Distance along Angle. Handheld: seeded shake. Dolly Zoom: a gentle, uniform change of framing about the selected pivot. Slow Pull: starts at Zoom and pulls back linearly to 1. Snap and Back: eases in to Zoom over the first half of Duration and back out over the second. Punch In: hard cut to Zoom at Start Frame with optional rebound. Pulse: repeating punch every Duration frames that decays back each time. Reveal: starts at Zoom and eases out to the full frame.", "Move"),
        Point("center", "Start Point / Pivot", 0.5, 0.5, "Normalized pivot of zoom, rotation and the dolly-zoom warp (0..1 across the frame).", "Move"),
        Bool("use_path", "Use Framing Path", false, "Move the start point to the end point during the timing interval.", "Move"),
        Point("end_point", "End Point", 0.5, 0.5, "Screen position reached by the starting pivot.", "Move"),
        Int("end_frame", "End Frame", 24, -100000, 100000, 0, 500, Unit::Frames, "Last frame of a framing path.", "Timing"),
        Int("start_frame", "Start Frame", 0, -100000, 100000, 0, 500, Unit::Frames, "Frame where the move starts. Before it the move is at its start state.", "Timing"),
        Dbl("duration", "Duration", 12.0, 1.0, 480.0, 1.0, 60.0, Unit::Frames, "Length of the move in frames. Rebound decays over another Duration.", "Timing"),
        Choice("easing", "Easing", {"Linear", "Ease In", "Ease Out", "Ease In-Out", "Expo"}, 3,
               "Progress curve for Snap Zoom, Whip and Dolly Zoom. Expo is an exponential ease-in-out.", "Timing"),
        Dbl("rebound", "Rebound", 0.15, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Damped overshoot after the target is reached (Snap Zoom, Whip, Dolly Zoom). 0 = none.", "Timing"),
        Dbl("zoom", "Zoom", 2.0, 0.25, 8.0, 0.5, 4.0, Unit::None, "Target scale factor (Snap Zoom, Push) or periphery scale (Dolly Zoom). 1 = no zoom.", "Move"),
        Dbl("distance", "Distance", 1920.0, -8000.0, 8000.0, -3000.0, 3000.0, Unit::Pixels, "Whip pan travel in full-resolution pixels.", "Move"),
        Dbl("angle", "Angle", 0.0, -360.0, 360.0, -180.0, 180.0, Unit::Degrees, "Whip pan direction in degrees (0 = image moves right, 90 = down).", "Move"),
        Dbl("amplitude", "Amplitude", 12.0, 0.0, 500.0, 0.0, 80.0, Unit::Pixels, "Handheld positional shake amplitude in full-resolution pixels; also drives a small zoom jitter.", "Handheld"),
        Dbl("rotation", "Rotation", 0.0, -180.0, 180.0, -15.0, 15.0, Unit::Degrees, "Handheld roll amplitude in degrees, or extra roll added over the move in the other modes.", "Move"),
        Dbl("frequency", "Frequency", 1.2, 0.05, 20.0, 0.1, 6.0, Unit::None, "Handheld shake frequency in Hz (cycles per second of clip time).", "Handheld"),
        Int("seed", "Seed", 1, 0, 100000, 0, 1000, Unit::None, "Random seed of the handheld shake.", "Handheld"),
        Dbl("breathing", "Breathing", 0.0, 0.0, 0.2, 0.0, 0.2, Unit::Normalized, "Slow 0.25 Hz sinusoidal zoom pulse added to every mode (fraction of scale).", "Move"),
        Dbl("shutter", "Shutter", 180.0, 0.0, 720.0, 0.0, 360.0, Unit::Degrees, "Shutter angle of the motion blur. 0 = sharp; 360 = blur over one whole frame interval.", "Motion Blur"),
        Int("motion_samples", "Motion Samples", 12, 1, 64, 1, 32, Unit::None, "Sub-frame transform samples averaged across the shutter interval.", "Motion Blur"),
        Choice("outside", "Outside", {"Clamp", "Mirror", "Black"}, 1, "How areas revealed outside the source frame are filled.", "Motion Blur"),
    };
    m.framesNeeded = [](const ParamValues&, double) { return FrameWindow{}; };
    m.render = [](const RenderContext& c) {
        const ParamValues& p = c.p();
        const int mode = p.i("mode"), easing = p.i("easing");
        const SampleEdge edge = static_cast<SampleEdge>(std::clamp(p.i("outside"), 0, 2));
        const double start = p.d("start_frame");
        const double duration = std::max(1.0, p.b("use_path") ? p.d("end_frame") - start : p.d("duration"));
        const double rebound = std::clamp(p.d("rebound"), 0.0, 1.0);
        const double zoom = std::clamp(p.d("zoom"), 0.25, 8.0);
        const double logZoom = std::log(zoom);
        const double distance = c.px(p.f("distance"));
        const double angle = p.d("angle") * kPi / 180.0;
        const double amplitude = c.px(p.f("amplitude"));
        const double rotation = p.d("rotation") * kPi / 180.0;
        const double frequency = std::clamp(p.d("frequency"), 0.05, 20.0);
        const uint32_t seed = static_cast<uint32_t>(p.i("seed"));
        const double breathing = std::clamp(p.d("breathing"), 0.0, 0.2);
        const double shutter = std::clamp(p.d("shutter"), 0.0, 720.0) / 360.0;
        const int samples = shutter > 0.0 ? std::clamp(p.i("motion_samples"), 1, 64) : 1;
        const double fps = c.fps > 0.0 ? c.fps : 24.0;
        double cx, cy;
        c.pointToPixels(p.vec("center"), cx, cy);
        if (mode == 4) {
            // Anchor the sample under the selected pixel exactly even on even-sized frames.
            cx = std::floor(cx) + 0.5; cy = std::floor(cy) + 0.5;
        }
        const double halfDiag = std::max(1.0, 0.5 * std::hypot(static_cast<double>(c.frame.w()), static_cast<double>(c.frame.h())));
        const double frameW = std::max(1, c.frame.w());
        double endX, endY;
        c.pointToPixels(p.vec("end_point"), endX, endY);

        auto sampleAt = [&](double t) {
            CameraMotionSample s;
            double raw = (t - start) / duration;
            double pl = std::clamp(raw, 0.0, 1.0);
            double e = cameraMotionEase(pl, easing) + cameraMotionRebound(raw - 1.0, rebound);
            double scale = 1.0, rot = 0.0, ox = 0.0, oy = 0.0;
            if (mode == 0) { scale = std::exp(logZoom * e); rot = rotation * e; }
            else if (mode == 1) { scale = 1.0 + (zoom - 1.0) * pl; rot = rotation * pl; }
            else if (mode == 2) { ox = distance * e * std::cos(angle); oy = distance * e * std::sin(angle); rot = rotation * e; }
            else if (mode == 3) {
                double ts = t / fps * frequency;
                ox = amplitude * fbm1(hashCombine(seed, 11u), ts, 3);
                oy = amplitude * fbm1(hashCombine(seed, 23u), ts, 3);
                rot = rotation * fbm1(hashCombine(seed, 37u), ts * 0.8, 2);
                scale = 1.0 + 0.5 * amplitude / frameW * fbm1(hashCombine(seed, 53u), ts * 0.5, 2);
            } else if (mode == 5) {
                scale = zoom + (1.0 - zoom) * pl;
                rot = rotation * (1.0 - pl);
            } else if (mode == 6) {
                double q = std::clamp(raw, 0.0, 1.0);
                double e2 = q < 0.5 ? cameraMotionEase(q * 2.0, easing) : cameraMotionEase(1.0 - (q - 0.5) * 2.0, easing);
                scale = std::exp(logZoom * e2);
                rot = rotation * e2;
            } else if (mode == 7) {
                double e2 = raw >= 0.0 ? 1.0 + cameraMotionRebound(raw, rebound) : 0.0;
                scale = std::exp(logZoom * e2);
                rot = rotation * e2;
            } else if (mode == 8) {
                double e2 = raw >= 0.0 ? std::exp(-6.0 * (raw - std::floor(raw))) : 0.0;
                scale = std::exp(logZoom * e2);
                rot = rotation * e2;
            } else if (mode == 9) {
                scale = std::exp(logZoom * (1.0 - e));
                rot = rotation * (1.0 - e);
            } else {
                // A single plate has no depth map. Keep people on a similarity transform.
                scale = std::exp(logZoom * (mode == 4 ? 0.35 : 1.0) * e);
                if (mode == 10) {
                    ox = 0.08 * frameW * std::sin(pl * kPi);
                    oy = 0.025 * c.frame.h() * std::sin(2.0 * pl * kPi);
                }
                if (mode == 11) oy = -0.12 * c.frame.h() * e;
                rot = rotation * e;
            }
            if (p.b("use_path")) {
                const double progress = cameraMotionEase(pl, easing);
                ox += (endX - cx) * progress;
                oy += (endY - cy) * progress;
            }
            if (breathing > 0.0) scale *= 1.0 + breathing * std::sin(2.0 * kPi * 0.25 * t / fps);
            scale = std::max(scale, 1e-3);
            Affine fwd = Affine::translate(ox, oy) * Affine::about(cx, cy, Affine::rotate(rot) * Affine::scale(scale));
            s.inv = fwd.inverse();
            return s;
        };

        std::vector<CameraMotionSample> list;
        list.reserve(static_cast<size_t>(samples));
        for (int k = 0; k < samples; ++k) {
            double off = samples > 1 ? ((k + 0.5) / samples - 0.5) * shutter : 0.0;
            // Do not integrate across the discontinuity of a hard cut.
            double sampleTime = c.time + off;
            if (mode == 7 && ((sampleTime < start) != (c.time < start))) sampleTime = c.time;
            list.push_back(sampleAt(sampleTime));
        }
        bool allSame = true;
        for (const auto& s : list) {
            const Affine& a = s.inv; const Affine& b = list[0].inv;
            if (a.a != b.a || a.b != b.b || a.c != b.c || a.d != b.d || a.tx != b.tx || a.ty != b.ty) { allSame = false; break; }
        }
        if (allSame) list.resize(1);

        ImageView src = c.source();
        if (!src.valid() && c.cudaRender) throw std::runtime_error("Missing native CUDA source image");
        if (!src.valid()) {
            parallelRows(c.window.y1, c.window.y2, [&](int y) {
                for (int x = c.window.x1; x < c.window.x2; ++x) c.dst.store(x, y, RGBA{});
            });
            return;
        }
        const Encoding enc = c.encoding;
        if (c.cudaRender || gpu::available()) {
            gpu::MotionArgs ga;
            ga.src = src; ga.dst = c.dst; ga.window = c.window;
            for (const auto& s : list) ga.samples.push_back({float(s.inv.a), float(s.inv.b), float(s.inv.c), float(s.inv.d), float(s.inv.tx), float(s.inv.ty), 0.f});
            ga.cx = float(cx); ga.cy = float(cy); ga.halfDiag = float(halfDiag);
            ga.edge = static_cast<int>(edge); ga.encoding = static_cast<int>(enc);
            if (c.cudaRender) {
                if (!gpu::motionNative(ga,c.p().f("mix"),c.cudaStream)) throw std::runtime_error("Native CUDA motion render failed");
                return;
            }
            if (gpu::motionResample(ga)) return;
        }
        OwnedImage lin;
        if (enc != Encoding::Linear) {
            lin.reset(src.bounds);
            parallelRows(src.bounds.y1, src.bounds.y2, [&](int y) {
                for (int x = src.bounds.x1; x < src.bounds.x2; ++x) lin.view.store(x, y, decode(src.load(x, y), enc));
            });
            src = lin.view;
        }
        const float inv = 1.f / static_cast<float>(list.size());
        parallelRows(c.window.y1, c.window.y2, [&](int y) {
            for (int x = c.window.x1; x < c.window.x2; ++x) {
                RGBA acc;
                for (const auto& s : list) {
                    acc += sampleAffine(src, s.inv, x, y, edge);
                }
                c.dst.store(x, y, encode(acc * inv, enc));
            }
        });
    };
    return m;
}

}
