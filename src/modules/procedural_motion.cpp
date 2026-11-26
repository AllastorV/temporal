#include "modules.h"
#include "../gpu/gpu.h"

namespace tp {

namespace {

struct ProceduralMotionAxes {
    double x = 0, y = 0, z = 0, rx = 0, ry = 0, rz = 0;
};

double proceduralMotionCos(double deg) {
    double v = std::cos(deg * kPi / 180.0);
    if (std::fabs(v) < 0.02) v = v < 0.0 ? -0.02 : 0.02;
    return v;
}

}

ModuleDef makeProceduralMotion() {
    ModuleDef m;
    m.id = "procedural_motion";
    m.cuda = CudaMode::Native;
    m.name = "Procedural Motion";
    m.family = "Motion";
    m.description = "Seeded, time-based generators (circle, figure eight, door swing, drift, handheld, noise, spring) driving a six-axis 2D transform of the current frame with sub-frame motion blur. Tilt axes are a 2D cosine-foreshortening approximation.";
    m.temporal = false;
    m.usesEncoding = true;
    m.params = {
        Choice("mode", "Mode", {"Circle", "Figure Eight", "Door Swing", "Drift", "Handheld", "Noise", "Spring"}, 4,
               "Circle and Figure Eight: periodic paths. Door Swing: Y tilt about a vertical hinge. Drift: slow noise. Handheld: multi-octave shake. Noise: single-octave noise. Spring: damped oscillation from Start Frame.", "Generator"),
        Choice("hinge", "Hinge", {"Left", "Right"}, 0, "Door Swing only: which frame edge stays fixed.", "Generator"),
        Dbl("amplitude", "Amplitude", 40.0, 0.0, 2000.0, 0.0, 300.0, Unit::Pixels, "Translation amplitude in full-resolution pixels (X and Y axes).", "Amounts"),
        Dbl("rotation_amount", "Rotation Amount", 5.0, 0.0, 180.0, 0.0, 90.0, Unit::Degrees, "Angle amplitude in degrees for roll and tilt axes; maximum swing angle in Door Swing.", "Amounts"),
        Dbl("zoom_amount", "Zoom Amount", 0.1, 0.0, 2.0, 0.0, 1.0, Unit::None, "Zoom amplitude in stops: scale = 2^(zoom_amount * axis_z * signal).", "Amounts"),
        Dbl("frequency", "Frequency", 0.5, 0.01, 20.0, 0.05, 4.0, Unit::None, "Motion frequency in Hz (cycles per second); noise cells per second for noise modes.", "Generator"),
        Dbl("phase", "Phase", 0.0, -360.0, 360.0, -180.0, 180.0, Unit::Degrees, "Phase offset of the generator in degrees of one cycle.", "Generator"),
        Dbl("smoothing", "Smoothing", 0.5, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Handheld only: more smoothing uses fewer noise octaves (1 = one octave, 0 = five).", "Generator"),
        Int("seed", "Seed", 1, 0, 100000, 0, 1000, Unit::None, "Random seed of the noise-based modes.", "Generator"),
        Dbl("loop", "Loop", 0.0, 0.0, 60.0, 0.0, 10.0, Unit::Seconds, "When above 0 the motion repeats exactly with this period in seconds (frequency is rounded to whole cycles per loop). 0 = off.", "Generator"),
        Int("start_frame", "Start Frame", 0, -100000, 100000, 0, 500, Unit::Frames, "Spring only: frame where the oscillation is triggered.", "Spring"),
        Dbl("damping", "Damping", 3.0, 0.0, 20.0, 0.0, 10.0, Unit::None, "Spring only: exponential decay rate per second.", "Spring"),
        Dbl("axis_x", "Axis X", 1.0, 0.0, 2.0, 0.0, 2.0, Unit::None, "Weight of horizontal translation.", "Axes"),
        Dbl("axis_y", "Axis Y", 1.0, 0.0, 2.0, 0.0, 2.0, Unit::None, "Weight of vertical translation.", "Axes"),
        Dbl("axis_z", "Axis Z", 0.1, 0.0, 2.0, 0.0, 2.0, Unit::None, "Weight of zoom.", "Axes"),
        Dbl("axis_rx", "Axis Tilt X", 0.0, 0.0, 2.0, 0.0, 2.0, Unit::None, "Weight of X tilt, rendered as cosine foreshortening of the image height.", "Axes"),
        Dbl("axis_ry", "Axis Tilt Y", 0.0, 0.0, 2.0, 0.0, 2.0, Unit::None, "Weight of Y tilt, rendered as cosine foreshortening of the image width.", "Axes"),
        Dbl("axis_rz", "Axis Roll", 0.2, 0.0, 2.0, 0.0, 2.0, Unit::None, "Weight of roll about the pivot.", "Axes"),
        Point("center", "Pivot", 0.5, 0.5, "Normalized pivot of zoom, roll and tilt (0..1 across the frame).", "Axes"),
        Dbl("shutter", "Shutter", 180.0, 0.0, 720.0, 0.0, 360.0, Unit::Degrees, "Shutter angle of the motion blur. 0 = sharp; 360 = blur over one whole frame interval.", "Motion Blur"),
        Int("motion_samples", "Motion Samples", 8, 1, 64, 1, 32, Unit::None, "Sub-frame transform samples averaged across the shutter interval.", "Motion Blur"),
        Choice("outside", "Outside", {"Clamp", "Mirror", "Black"}, 0, "How areas revealed outside the source frame are filled.", "Motion Blur"),
    };
    m.framesNeeded = [](const ParamValues&, double) { return FrameWindow{}; };
    m.render = [](const RenderContext& c) {
        const ParamValues& p = c.p();
        const int mode = p.i("mode");
        const SampleEdge edge = static_cast<SampleEdge>(std::clamp(p.i("outside"), 0, 2));
        const double amp = c.px(p.f("amplitude"));
        const double rotAmt = std::clamp(p.d("rotation_amount"), 0.0, 180.0);
        const double zoomAmt = std::clamp(p.d("zoom_amount"), 0.0, 2.0);
        const double loop = std::clamp(p.d("loop"), 0.0, 60.0);
        double freq = std::clamp(p.d("frequency"), 0.01, 20.0);
        const double phase = p.d("phase") / 360.0;
        const double smoothing = std::clamp(p.d("smoothing"), 0.0, 1.0);
        const uint32_t seed = static_cast<uint32_t>(p.i("seed"));
        const double damping = std::clamp(p.d("damping"), 0.0, 20.0);
        const double wx = std::clamp(p.d("axis_x"), 0.0, 2.0), wy = std::clamp(p.d("axis_y"), 0.0, 2.0), wz = std::clamp(p.d("axis_z"), 0.0, 2.0);
        const double wrx = std::clamp(p.d("axis_rx"), 0.0, 2.0), wry = std::clamp(p.d("axis_ry"), 0.0, 2.0), wrz = std::clamp(p.d("axis_rz"), 0.0, 2.0);
        const double shutter = std::clamp(p.d("shutter"), 0.0, 720.0) / 360.0;
        const int samples = shutter > 0.0 ? std::clamp(p.i("motion_samples"), 1, 64) : 1;
        const double fps = c.fps > 0.0 ? c.fps : 24.0;
        const double startSec = p.d("start_frame") / fps;
        const bool looping = loop > 0.0;
        const double loopFrames = loop * fps;
        if (looping && mode != 3 && mode != 4 && mode != 5) freq = std::max(1.0, std::round(freq * loop)) / loop;
        const double hingeX = p.i("hinge") == 0 ? c.frame.x1 : c.frame.x2;
        const int octaves = 1 + static_cast<int>(std::lround((1.0 - smoothing) * 4.0));
        double cx, cy;
        c.pointToPixels(p.vec("center"), cx, cy);

        auto noise = [&](uint32_t axis, double ts, double rate, int oct) -> double {
            uint32_t s = hashCombine(seed, axis);
            if (!looping) return fbm1(s, ts * rate * freq + phase, oct);
            double th = 2.0 * kPi * (ts / loop + phase);
            double r = rate * freq * loop / (2.0 * kPi);
            double sum = 0.0, a = 1.0, norm = 0.0;
            for (int o = 0; o < oct; ++o) {
                double ro = r * std::exp2(o);
                sum += a * (valueNoise2(hashCombine(s, static_cast<uint32_t>(o)), static_cast<float>(ro * std::cos(th) + 17.0 * o), static_cast<float>(ro * std::sin(th) + 31.0 * o)) * 2.0 - 1.0);
                norm += a; a *= 0.5;
            }
            return sum / norm;
        };

        auto sampleAt = [&](double t) {
            double tf = t;
            if (looping && loopFrames > 0.0) { tf = std::fmod(t, loopFrames); if (tf < 0.0) tf += loopFrames; }
            const double ts = tf / fps;
            const double th = 2.0 * kPi * (freq * ts + phase);
            Affine fwd;
            if (mode == 2) {
                double ang = rotAmt * 0.5 * (1.0 - std::cos(th));
                fwd = Affine::about(hingeX, cy, Affine::scale(proceduralMotionCos(ang), 1.0));
                return fwd.inverse();
            }
            ProceduralMotionAxes g;
            switch (mode) {
            case 0: g = {std::cos(th), std::sin(th), std::sin(th), std::sin(th), std::cos(th), std::sin(th)}; break;
            case 1: g = {std::sin(th), std::sin(2.0 * th), std::cos(2.0 * th), std::sin(2.0 * th), std::sin(th), std::sin(th)}; break;
            case 3: g = {noise(1, ts, 1.0, 2), noise(2, ts, 1.0, 2), 0.5 * noise(3, ts, 0.5, 2), 0, 0, 0}; break;
            case 4: g = {noise(1, ts, 1.0, octaves), noise(2, ts, 1.0, octaves), noise(3, ts, 0.5, octaves), 0, 0, noise(6, ts, 0.8, octaves)}; break;
            case 5: g = {noise(1, ts, 1.0, 1), noise(2, ts, 1.0, 1), noise(3, ts, 1.0, 1), noise(4, ts, 1.0, 1), noise(5, ts, 1.0, 1), noise(6, ts, 1.0, 1)}; break;
            default: {
                double tau = ts - startSec;
                if (looping) { tau = std::fmod(tau, loop); if (tau < 0.0) tau += loop; }
                double v = tau < 0.0 ? 0.0 : std::exp(-damping * tau) * std::cos(2.0 * kPi * (freq * tau + phase));
                g = {v, v, v, v, v, v};
            }
            }
            double z = zoomAmt * wz * g.z;
            double sxy = std::exp2(z);
            double sx = sxy * proceduralMotionCos(rotAmt * wry * g.ry);
            double sy = sxy * proceduralMotionCos(rotAmt * wrx * g.rx);
            double rz = rotAmt * wrz * g.rz * kPi / 180.0;
            fwd = Affine::translate(amp * wx * g.x, amp * wy * g.y) * Affine::about(cx, cy, Affine::rotate(rz) * Affine::scale(sx, sy));
            return fwd.inverse();
        };

        std::vector<Affine> list;
        list.reserve(static_cast<size_t>(samples));
        for (int k = 0; k < samples; ++k) {
            double off = samples > 1 ? ((k + 0.5) / samples - 0.5) * shutter : 0.0;
            list.push_back(sampleAt(c.time + off));
        }
        bool allSame = true;
        for (const Affine& a : list) {
            const Affine& b = list[0];
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
            for (const Affine& a : list) ga.samples.push_back({float(a.a), float(a.b), float(a.c), float(a.d), float(a.tx), float(a.ty), 0.f});
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
                for (const Affine& a : list) acc += sampleAffine(src, a, x, y, edge);
                c.dst.store(x, y, encode(acc * inv, enc));
            }
        });
    };
    return m;
}

}
