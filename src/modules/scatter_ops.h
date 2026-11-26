#pragma once
#include "modules.h"

namespace tp {

struct DlLevel {
    OwnedImage img;
    int f = 1;
    RGBA sample(const Rect& R, int x, int y, SampleEdge e = SampleEdge::Clamp) const {
        const float inv = 1.f / static_cast<float>(f);
        return img.view.bilinear((x - R.x1 + 0.5f) * inv, (y - R.y1 + 0.5f) * inv, e);
    }
};

inline DlLevel dlDown(const ImageView& sel, int f) {
    DlLevel L;
    L.f = std::max(1, f);
    const Rect& R = sel.bounds;
    const int lw = (R.w() + L.f - 1) / L.f, lh = (R.h() + L.f - 1) / L.f;
    L.img.reset(Rect{0, 0, lw, lh});
    parallelRows(0, lh, [&](int j) {
        const int ya = R.y1 + j * L.f, yb = std::min(R.y2, ya + L.f);
        for (int i = 0; i < lw; ++i) {
            const int xa = R.x1 + i * L.f, xb = std::min(R.x2, xa + L.f);
            RGBA s;
            for (int y = ya; y < yb; ++y)
                for (int x = xa; x < xb; ++x) s += sel.load(x, y);
            L.img.view.store(i, j, s * (1.f / static_cast<float>((xb - xa) * (yb - ya))));
        }
    });
    return L;
}

inline DlLevel dlDirBlur(const ImageView& sel, float angleRad, float lengthPx, int maxTaps = 32, float decay = 3.f) {
    lengthPx = std::max(0.f, lengthPx);
    const int f = std::clamp(static_cast<int>(std::ceil(lengthPx / maxTaps)), 1, 64);
    DlLevel lo = dlDown(sel, f);
    const int taps = std::clamp(static_cast<int>(std::ceil(lengthPx / f)), 1, maxTaps);
    const float dx = std::cos(angleRad), dy = -std::sin(angleRad);
    const int n = 2 * taps + 1;
    std::vector<int> ox(static_cast<size_t>(n)), oy(static_cast<size_t>(n));
    std::vector<float> w(static_cast<size_t>(n) * 4);
    float ws = 0.f;
    for (int t = -taps; t <= taps; ++t) ws += std::exp(-decay * std::fabs(static_cast<float>(t)) / taps);
    for (int t = -taps; t <= taps; ++t) {
        const float v = std::exp(-decay * std::fabs(static_cast<float>(t)) / taps) / ws;
        const float px = t * dx, py = t * dy;
        const float flx = std::floor(px), fly = std::floor(py);
        const float tx = px - flx, ty = py - fly;
        const size_t k = static_cast<size_t>(t + taps);
        ox[k] = static_cast<int>(flx);
        oy[k] = static_cast<int>(fly);
        w[k * 4 + 0] = v * (1.f - tx) * (1.f - ty);
        w[k * 4 + 1] = v * tx * (1.f - ty);
        w[k * 4 + 2] = v * (1.f - tx) * ty;
        w[k * 4 + 3] = v * tx * ty;
    }
    const Rect lb = lo.img.view.bounds;
    const int lw = lb.w(), lh = lb.h(), P = taps + 2, pw = lw + 2 * P, ph = lh + 2 * P;
    std::vector<float> pad(static_cast<size_t>(pw) * ph * 4, 0.f);
    parallelRows(0, lh, [&](int j) {
        const float* s = lo.img.view.px(lb.x1, lb.y1 + j);
        std::copy(s, s + static_cast<size_t>(lw) * 4, pad.begin() + (static_cast<size_t>(j + P) * pw + P) * 4);
    });
    DlLevel out;
    out.f = lo.f;
    out.img.reset(lb);
    const size_t stride = static_cast<size_t>(pw) * 4;
    parallelRows(0, lh, [&](int j) {
        for (int i = 0; i < lw; ++i) {
            float a[4] = {0.f, 0.f, 0.f, 0.f};
            for (int k = 0; k < n; ++k) {
                const float* p = pad.data() + (static_cast<size_t>(j + P + oy[k]) * pw + static_cast<size_t>(i + P + ox[k])) * 4;
                const float* q = p + stride;
                const float* wk = w.data() + static_cast<size_t>(k) * 4;
                for (int ch = 0; ch < 4; ++ch) a[ch] += wk[0] * p[ch] + wk[1] * p[4 + ch] + wk[2] * q[ch] + wk[3] * q[4 + ch];
            }
            out.img.view.store(lb.x1 + i, lb.y1 + j, RGBA{a[0], a[1], a[2], a[3]});
        }
    });
    return out;
}

inline RGBA dlHighlight(const RGBA& lin, float t) {
    RGBA c{std::max(0.f, lin.r), std::max(0.f, lin.g), std::max(0.f, lin.b), 0.f};
    float l = luma(c);
    if (l <= 1e-8f) return {};
    float w = 0.5f * t, e;
    if (w <= 1e-6f) e = std::max(0.f, l - t);
    else if (l <= t - w) e = 0.f;
    else if (l >= t + w) e = l - t;
    else { float d = l - t + w; e = d * d / (4.f * w); }
    float s = e / l;
    return {c.r * s, c.g * s, c.b * s, 0.f};
}

}
