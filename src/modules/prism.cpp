#include "modules.h"

namespace tp {

namespace {

struct PrismTap { ImageView img; float w; };

void prismOffsets(const ParamValues& p, int off[3]) {
    const bool temporal = p.i("mode") != 1;
    const char* ids[3] = {"red_offset", "green_offset", "blue_offset"};
    for (int ch = 0; ch < 3; ++ch) off[ch] = temporal ? std::clamp(p.i(ids[ch]), -32, 32) : 0;
}

}

ModuleDef makePrism() {
    ModuleDef m;
    m.id = "prism";
    m.name = "Prism";
    m.family = "Temporal";
    m.description = "Splits red, green and blue in time and space: each channel reads its own frame offset, with radial chromatic dispersion and per-channel blur.";
    m.temporal = true;
    m.usesEncoding = true;
    m.params = {
        Choice("mode", "Mode", {"Temporal RGB", "Spatial Dispersion", "Both"}, 2,
               "Temporal RGB: each channel reads its own frame offset. Spatial Dispersion: radial chromatic displacement only. Both: the two combined."),
        Int("red_offset", "Red Offset", -4, -32, 32, -16, 16, Unit::Frames, "Frame offset of the red channel (negative = past)."),
        Int("green_offset", "Green Offset", 0, -32, 32, -16, 16, Unit::Frames, "Frame offset of the green channel (negative = past)."),
        Int("blue_offset", "Blue Offset", 4, -32, 32, -16, 16, Unit::Frames, "Frame offset of the blue channel (negative = past)."),
        Dbl("spread", "Spread", 8.0, 0.0, 400.0, 0.0, 60.0, Unit::Pixels,
            "Radial chromatic displacement in full-resolution pixels at half the frame diagonal away from Center, growing linearly with distance: red is pushed outward, blue inward, green stays."),
        Dbl("blur", "Blur", 0.0, 0.0, 100.0, 0.0, 20.0, Unit::Pixels,
            "Gaussian sigma in full-resolution pixels for the most displaced channel; each channel is blurred in proportion to its frame offset (temporal) or spread (spatial) relative to the largest."),
        Dbl("decay", "Trail Decay", 0.0, 0.0, 1.0, 0.0, 1.0, Unit::Normalized,
            "0: each channel reads a single frame. Above 0: each channel averages every frame from the current one to its offset, weighted decay^k from the far end, giving colored trails."),
        Point("center", "Center", 0.5, 0.5, "Origin of the radial dispersion, normalized frame coordinates."),
    };
    m.framesNeeded = [](const ParamValues& p, double) {
        int off[3];
        prismOffsets(p, off);
        FrameWindow w;
        for (int o : off) { w.before = std::max(w.before, -o); w.after = std::max(w.after, o); }
        return w;
    };
    m.render = [](const RenderContext& c) {
        const ParamValues& p = c.p();
        const int mode = p.i("mode");
        const bool spatial = mode != 0;
        int off[3];
        prismOffsets(p, off);
        const float decay = saturate(p.f("decay"));
        const float diag = std::max(1.f, std::hypot(static_cast<float>(c.frame.w()), static_cast<float>(c.frame.h())));
        const float spread = spatial ? clampf(finiteOr(c.px(p.f("spread")), 0.f), 0.f, diag) : 0.f;
        const float k = spread / (0.5f * diag);
        const float blur = clampf(finiteOr(c.px(p.f("blur")), 0.f), 0.f, diag);
        double cxd, cyd;
        c.pointToPixels(p.vec("center"), cxd, cyd);
        const float cx = static_cast<float>(cxd), cy = static_cast<float>(cyd);
        const float dsgn[3] = {k, 0.f, -k};
        const int maxo = std::max({std::abs(off[0]), std::abs(off[1]), std::abs(off[2])});

        std::vector<PrismTap> taps[3];
        float sigma[3];
        for (int ch = 0; ch < 3; ++ch) {
            const int o = off[ch], n = std::abs(o), sg = o < 0 ? -1 : 1;
            if (decay <= 0.f || n == 0) taps[ch].push_back({c.frameOffset(o), 1.f});
            else {
                float sum = 0.f;
                for (int j = 0; j <= n; ++j) {
                    float w = std::pow(decay, static_cast<float>(n - j));
                    taps[ch].push_back({c.frameOffset(sg * j), w});
                    sum += w;
                }
                for (PrismTap& t : taps[ch]) t.w /= sum;
            }
            float tf = maxo > 0 ? static_cast<float>(n) / maxo : 0.f;
            float sf = (spread > 0.f && ch != 1) ? 1.f : 0.f;
            sigma[ch] = blur * std::max(tf, sf);
        }
        const Encoding enc = c.encoding;

        OwnedImage packs[3];
        float packSigma[3] = {0.f, 0.f, 0.f};
        int packOf[3] = {-1, -1, -1}, laneOf[3] = {0, 0, 0}, packUsed[3] = {0, 0, 0}, packs_n = 0;
        bool planar[3] = {false, false, false};
        for (int ch = 0; ch < 3; ++ch) {
            if (!(sigma[ch] > 0.35f) && !(dsgn[ch] != 0.f && taps[ch].size() > 1)) continue;
            planar[ch] = true;
            int k = 0;
            while (k < packs_n && !(packUsed[k] == 1 && packSigma[k] == sigma[ch])) ++k;
            if (k == packs_n) { packSigma[k] = sigma[ch]; packs[k].reset(c.frame); ++packs_n; }
            packOf[ch] = k;
            laneOf[ch] = 2 * packUsed[k]++;
        }
        if (packs_n > 0) {
            size_t planeTaps = 0;
            for (int ch = 0; ch < 3; ++ch) if (planar[ch]) planeTaps = std::max(planeTaps, taps[ch].size());
            parallelRows(c.frame.y1, c.frame.y2, [&](int y) {
                for (int x = c.frame.x1; x < c.frame.x2; ++x) {
                    float v[3] = {0.f, 0.f, 0.f}, al[3] = {0.f, 0.f, 0.f};
                    for (size_t j = 0; j < planeTaps; ++j) {
                        RGBA q3;
                        for (int ch = 0; ch < 3; ++ch) {
                            if (!planar[ch] || j >= taps[ch].size()) continue;
                            const RGBA q = taps[ch][j].img.at(x, y, SampleEdge::Black);
                            q3[ch] = q[ch];
                            al[ch] += taps[ch][j].w * q.a;
                        }
                        const RGBA d = decode(q3, enc);
                        for (int ch = 0; ch < 3; ++ch)
                            if (planar[ch] && j < taps[ch].size()) v[ch] += taps[ch][j].w * d[ch];
                    }
                    RGBA out[3];
                    for (int ch = 0; ch < 3; ++ch) {
                        if (!planar[ch]) continue;
                        out[packOf[ch]][laneOf[ch]] = v[ch];
                        out[packOf[ch]][laneOf[ch] + 1] = al[ch];
                    }
                    for (int k = 0; k < packs_n; ++k) packs[k].view.store(x, y, out[k]);
                }
            });
            for (int k = 0; k < packs_n; ++k) gaussianBlur(packs[k], packSigma[k]);
        }

        parallelRows(c.window.y1, c.window.y2, [&](int y) {
            const float py = y + 0.5f;
            for (int x = c.window.x1; x < c.window.x2; ++x) {
                const float px = x + 0.5f;
                float v[3] = {0.f, 0.f, 0.f}, a[3] = {0.f, 0.f, 0.f}, sx[3], sy[3];
                size_t rawTaps = 0;
                for (int ch = 0; ch < 3; ++ch) {
                    sx[ch] = px - dsgn[ch] * (px - cx);
                    sy[ch] = py - dsgn[ch] * (py - cy);
                    if (planar[ch]) {
                        const ImageView& pv = packs[packOf[ch]].view;
                        RGBA s = dsgn[ch] != 0.f ? pv.bilinear(sx[ch], sy[ch], SampleEdge::Clamp) : pv.at(x, y, SampleEdge::Clamp);
                        v[ch] = s[laneOf[ch]]; a[ch] = s[laneOf[ch] + 1];
                    } else rawTaps = std::max(rawTaps, taps[ch].size());
                }
                for (size_t j = 0; j < rawTaps; ++j) {
                    RGBA q3;
                    for (int ch = 0; ch < 3; ++ch) {
                        if (planar[ch] || j >= taps[ch].size()) continue;
                        const PrismTap& t = taps[ch][j];
                        const RGBA q = dsgn[ch] != 0.f ? t.img.bilinear(sx[ch], sy[ch], SampleEdge::Clamp) : t.img.at(x, y, SampleEdge::Black);
                        q3[ch] = q[ch];
                        a[ch] += t.w * q.a;
                    }
                    const RGBA d = decode(q3, enc);
                    for (int ch = 0; ch < 3; ++ch)
                        if (!planar[ch] && j < taps[ch].size()) v[ch] += taps[ch][j].w * d[ch];
                }
                RGBA out = encode(RGBA{v[0], v[1], v[2], 0.f}, enc);
                out.a = std::max({0.f, saturate(a[0]), saturate(a[1]), saturate(a[2])});
                c.dst.store(x, y, out);
            }
        });
    };
    return m;
}

}
