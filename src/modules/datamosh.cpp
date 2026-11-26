#include "modules.h"
#include "../core/flow.h"

namespace tp {

namespace {

int datamoshAnchor(int mode, int t, int ref, int K) {
    if (mode == 2) return ref - 1;
    int n = t - ref;
    int q = n >= 0 ? n / K : -((-n + K - 1) / K);
    return ref + q * K;
}

}

ModuleDef makeDatamosh() {
    ModuleDef m;
    m.id = "datamosh";
    m.name = "Datamosh";
    m.family = "Digital";
    m.description = "Codec-free datamosh simulation: the clean frame at each reset (the I-frame) is dragged by the measured motion of the following frames in macroblocks, so old pixels smear with new movement; plus seeded block corruption. Each frame is rebuilt from its reset anchor, so seeking is deterministic. This is a simulation, not bitstream datamosh.";
    m.temporal = true;
    m.usesEncoding = false;
    m.params = {
        Choice("mode", "Mode", {"P-Frame Smear", "Block Corruption", "Scene Transition"}, 0,
               "P-Frame Smear: periodic I-frames every Reset Interval, motion smears the image in between. Block Corruption: clean motion plus glitched blocks. Scene Transition: from Reference Frame on, the previous frame's pixels are carried by the new shot's motion for Reset Interval frames."),
        Int("reference_frame", "Reference Frame", 0, -100000, 100000, 0, 1000, Unit::Frames, "Absolute frame that anchors the I-frame grid, or the cut frame in Scene Transition."),
        Int("reset_interval", "Reset Interval", 12, 2, 24, 2, 24, Unit::Frames, "Frames between clean I-frames (length of the smear)."),
        Dbl("block_size", "Block Size", 16.0, 4.0, 128.0, 4.0, 64.0, Unit::Pixels, "Macroblock size in full-resolution pixels; motion is quantized per block."),
        Dbl("persistence", "Persistence", 0.9, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "How strongly old pixels stick: 1 = pure smear, lower lets blocks refresh with the real frame."),
        Dbl("flow_scale", "Flow Scale", 1.0, 0.0, 4.0, 0.0, 4.0, Unit::None, "Multiplier on the motion vectors."),
        Dbl("feedback", "Feedback", 0.0, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Motion acceleration through the GOP: vectors grow by this fraction per frame since the reset."),
        Dbl("corruption", "Corruption", 0.06, 0.0, 1.0, 0.0, 0.5, Unit::Normalized, "Probability that a macroblock glitches (shift, duplicate or channel slip)."),
        Int("seed", "Seed", 1, 0, 100000, 0, 1000, Unit::None, "Seed of the block decisions."),
        Choice("quality", "Analysis", {"Draft", "Standard"}, 0, "Optical-flow quality for the motion vectors."),
    };
    m.framesNeeded = [](const ParamValues& p, double) {
        FrameWindow w;
        w.before = p.i("mode") == 1 ? 0 : std::clamp(p.i("reset_interval"), 2, 24);
        return w;
    };
    m.render = [](const RenderContext& c) {
        const ParamValues& p = c.p();
        const int mode = p.i("mode");
        const int K = std::clamp(p.i("reset_interval"), 2, 24);
        const int t = static_cast<int>(std::lround(c.time));
        const int ref = p.i("reference_frame");
        const ImageView cur = c.source();
        if (!cur.valid()) {
            parallelRows(c.window.y1, c.window.y2, [&](int y) { for (int x = c.window.x1; x < c.window.x2; ++x) c.dst.store(x, y, RGBA{}); });
            return;
        }
        Rect R = intersect(c.frame, cur.bounds);
        if (R.empty()) R = cur.bounds;
        const float bs = std::max(2.f, c.px(p.f("block_size")));
        const int bw = static_cast<int>(std::ceil(R.w() / bs)), bh = static_cast<int>(std::ceil(R.h() / bs));
        const float persist = saturate(p.f("persistence"));
        const float fscale = clampf(p.f("flow_scale"), 0.f, 4.f);
        const float fb = saturate(p.f("feedback"));
        const float corrupt = saturate(p.f("corruption"));
        const uint32_t seed = static_cast<uint32_t>(p.i("seed"));
        int anchor = datamoshAnchor(mode, t, ref, K);
        int steps = t - anchor;
        bool active = steps > 0 && steps <= K;
        if (mode == 2 && (t < ref || t >= ref + K)) active = false;
        if (mode == 1) active = false;

        OwnedImage img(R);
        if (!active) {
            copyRect(cur, img.view, R);
        } else {
            const bool draft = p.i("quality") == 0;
            std::vector<ImageView> fr;
            if (draft) {
                fr = {c.frameOffset(anchor - t), cur};
            } else {
                for (int k = 0; k <= steps; ++k) {
                    if (c.isAborted()) return;
                    fr.push_back(c.frameOffset(anchor + k - t));
                }
            }
            ImageView base = fr[0].valid() ? fr[0] : cur;
            copyRect(base, img.view, R);
            OwnedImage tmp(R);
            std::vector<float> bu(static_cast<size_t>(bw) * bh), bv(bu.size());
            for (int k = 1; k < static_cast<int>(fr.size()); ++k) {
                if (c.isAborted()) return;
                if (!fr[static_cast<size_t>(k)].valid() || !fr[static_cast<size_t>(k - 1)].valid()) continue;
                const float analysisSize = c.preview ? 180.f : 640.f;
                const float analysisScale = draft || c.preview ? std::max(float(c.scale), std::min(R.w(),R.h()) / analysisSize) : float(c.scale);
                FlowField fl = computeFlow(fr[k-1], fr[k], R, analysisScale, Encoding::Linear, p.i("quality") == 1 ? 1 : 0);
                const float gain = fscale * (1.f + fb * static_cast<float>((draft ? steps : k) - 1));
                parallelRows(0, bh, [&](int j) {
                    for (int i = 0; i < bw; ++i) {
                        float su = 0.f, sv = 0.f;
                        int n = 0;
                        for (int yy = 0; yy < 4; ++yy)
                            for (int xx = 0; xx < 4; ++xx) {
                                float u, v;
                                fl.sample(R.x1 + (i + (xx + 0.5f) / 4.f) * bs, R.y1 + (j + (yy + 0.5f) / 4.f) * bs, u, v);
                                su += u; sv += v; ++n;
                            }
                        bu[static_cast<size_t>(j) * bw + i] = std::round(su / n * gain);
                        bv[static_cast<size_t>(j) * bw + i] = std::round(sv / n * gain);
                    }
                });
                const ImageView now = fr[static_cast<size_t>(k)];
                const int frameNo = draft ? t : anchor + k;
                parallelRows(R.y1, R.y2, [&](int y) {
                    for (int x = R.x1; x < R.x2; ++x) {
                        int i = std::clamp(static_cast<int>((x - R.x1) / bs), 0, bw - 1), j = std::clamp(static_cast<int>((y - R.y1) / bs), 0, bh - 1);
                        float refresh = hashFloat(seed, i / 3, j / 3, frameNo / 3);
                        if (refresh > persist + (1.f - persist) * 0.5f) { tmp.view.store(x, y, now.at(x, y, SampleEdge::Clamp)); continue; }
                        float u = bu[static_cast<size_t>(j) * bw + i], v = bv[static_cast<size_t>(j) * bw + i];
                        tmp.view.store(x, y, img.view.at(static_cast<int>(x - u), static_cast<int>(y - v), SampleEdge::Clamp));
                    }
                });
                std::swap(img.buf, tmp.buf);
                img.view.data = img.buf.data();
                tmp.view.data = tmp.buf.data();
            }
        }

        const bool doCorrupt = corrupt > 0.f && (mode == 1 || active);
        parallelRows(c.window.y1, c.window.y2, [&](int y) {
            for (int x = c.window.x1; x < c.window.x2; ++x) {
                RGBA o = img.view.at(x, y, SampleEdge::Clamp);
                if (doCorrupt) {
                    int i = std::clamp(static_cast<int>((x - R.x1) / bs), 0, bw - 1), j = std::clamp(static_cast<int>((y - R.y1) / bs), 0, bh - 1);
                    if (hashFloat(seed, i, j, t * 7 + 3) < corrupt) {
                        float kind = hashFloat(seed, i, j, t * 7 + 4);
                        int sx = static_cast<int>(std::lround((hashFloat(seed, i, j, t * 7 + 5) - 0.5f) * 4.f)) * static_cast<int>(bs);
                        int sy = static_cast<int>(std::lround((hashFloat(seed, i, j, t * 7 + 6) - 0.5f) * 2.f)) * static_cast<int>(bs);
                        if (kind < 0.45f) o = img.view.at(x + sx, y + sy, SampleEdge::Clamp);
                        else if (kind < 0.75f) {
                            int bx = R.x1 + static_cast<int>(i * bs), by = R.y1 + static_cast<int>(j * bs);
                            o = img.view.at(bx + static_cast<int>(bs) / 2, by + static_cast<int>(bs) / 2, SampleEdge::Clamp);
                        } else {
                            RGBA a = img.view.at(x + static_cast<int>(bs) / 4, y, SampleEdge::Clamp);
                            o = RGBA{a.r, o.g, img.view.at(x - static_cast<int>(bs) / 4, y, SampleEdge::Clamp).b, o.a};
                        }
                    }
                }
                c.dst.store(x, y, o);
            }
        });
    };
    return m;
}

}
