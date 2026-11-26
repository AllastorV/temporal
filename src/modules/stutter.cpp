#include "modules.h"

namespace tp {
namespace {

long long stutterFloorDiv(long long a, long long b) {
    long long q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

double stutterBeat(const ParamValues& p, double fps) {
    double f = std::isfinite(fps) && fps > 0.0 ? fps : 24.0;
    double bpm = std::clamp(p.d("bpm"), 20.0, 300.0);
    double b = f * 60.0 / bpm;
    return std::isfinite(b) ? std::clamp(b, 1.0, 1e6) : 1.0;
}

FrameWindow stutterWindow(const ParamValues& p, double fps) {
    FrameWindow w;
    const int mode = p.i("mode");
    const int hold = std::clamp(p.i("hold"), 1, 64);
    const int rep = std::clamp(p.i("repeat"), 1, 16);
    if (mode == 0) {
        w.before = hold - 1;
    } else if (mode == 1) {
        const int len = std::clamp(p.i("loop_length"), 2, 64);
        w.before = std::min(kMaxTemporalWindow, len * rep - 1);
        w.after = len - 1;
    } else {
        const double beat = stutterBeat(p, fps);
        const int beatReach = static_cast<int>(std::min(1e6, std::ceil(beat)));
        w.before = std::min({kMaxTemporalWindow, hold * (rep - 1), beatReach});
    }
    return clampWindow(w);
}

long long stutterSource(const ParamValues& p, double fps, long long t) {
    const int mode = p.i("mode");
    const long long ref = std::clamp(p.i("reference_frame"), -100000, 100000);
    const long long d = t - ref;
    const int hold = std::clamp(p.i("hold"), 1, 64);
    const int rep = std::clamp(p.i("repeat"), 1, 16);
    if (mode == 0) return ref + stutterFloorDiv(d, hold) * hold;
    if (mode == 1) {
        const long long len = std::clamp(p.i("loop_length"), 2, 64);
        const long long span = len * rep;
        const long long seg = stutterFloorDiv(d, span);
        const long long j = d - seg * span;
        const long long pass = j / len, i = j % len;
        const long long off = (pass % 2 == 0) ? len - 1 - i : i;
        return ref + seg * span + off;
    }
    const double beat = stutterBeat(p, fps);
    const double bi = std::floor(static_cast<double>(d) / beat);
    const long long beatStart = ref + static_cast<long long>(std::ceil(bi * beat - 1e-9));
    const long long j = std::max(0LL, t - beatStart);
    const float chance = static_cast<float>(std::clamp(p.d("chance"), 0.0, 1.0));
    const uint32_t seed = static_cast<uint32_t>(std::clamp(p.i("seed"), 0, 9999));
    const int32_t beatIndex = static_cast<int32_t>(static_cast<long long>(bi));
    if (!(hashFloat(seed, beatIndex, 0x5747) < chance)) return t;
    const long long nextBeat = ref + static_cast<long long>(std::ceil((bi + 1) * beat - 1e-9));
    const long long slice = std::min<long long>(hold, std::max(1LL, (nextBeat - beatStart) / rep));
    if (j < slice * rep) return beatStart + j % slice;
    return t;
}

}

ModuleDef makeStutter() {
    ModuleDef m;
    m.id = "stutter";
    m.name = "Stutter";
    m.family = "Temporal";
    m.description = "Shows exactly one source frame per output frame through a deterministic time remap: held frames, backwards loops or BPM-locked repeated slices. No blending.";
    m.temporal = true;
    m.usesEncoding = false;
    m.params = {
        Choice("mode", "Mode", {"Frame Hold", "Reverse Loop", "Rhythmic Repeat"}, 0,
               "Frame Hold: freeze every Hold frames. Reverse Loop: play each Loop Length segment backwards (ping-pong when Repeat > 1). Rhythmic Repeat: repeat a slice at the start of each beat."),
        Int("reference_frame", "Reference Frame", 0, -100000, 100000, 0, 1000, Unit::Frames,
            "Absolute frame number that anchors the hold, loop and beat grid."),
        Int("hold", "Hold", 4, 1, 64, 1, 64, Unit::Frames,
            "Frame Hold: frames each held image lasts (1 = untouched). Rhythmic Repeat: length of the repeated slice in frames."),
        Int("repeat", "Repeat", 2, 1, 16, 1, 16, Unit::None,
            "Reverse Loop: number of ping-pong passes per segment. Rhythmic Repeat: how many times the slice plays at the start of a beat. Reach is capped at 64 frames."),
        Int("loop_length", "Loop Length", 8, 2, 64, 2, 64, Unit::Frames,
            "Reverse Loop: length in frames of each segment played backwards."),
        Dbl("bpm", "BPM", 120.0, 20.0, 300.0, 20.0, 300.0, Unit::BPM,
            "Rhythmic Repeat: tempo. One beat lasts fps * 60 / BPM frames."),
        Dbl("chance", "Chance", 1.0, 0.0, 1.0, 0.0, 1.0, Unit::Normalized,
            "Rhythmic Repeat: probability that a given beat stutters, decided per beat by Seed."),
        Int("seed", "Seed", 1, 0, 9999, 0, 9999, Unit::None,
            "Seed for the per-beat Chance decision."),
    };
    m.framesNeeded = [](const ParamValues& p, double fps) { return stutterWindow(p, fps); };
    m.render = [](const RenderContext& c) {
        const ParamValues& p = c.p();
        const long long t = std::llround(c.time);
        const FrameWindow w = stutterWindow(p, c.fps);
        const long long src = std::clamp(stutterSource(p, c.fps, t), t - w.before, t + w.after);
        const ImageView img = c.frameAt(static_cast<double>(src));
        parallelRows(c.window.y1, c.window.y2, [&](int y) {
            for (int x = c.window.x1; x < c.window.x2; ++x) c.dst.store(x, y, img.at(x, y, SampleEdge::Black));
        });
    };
    return m;
}

}
