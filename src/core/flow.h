#pragma once
#include "core.h"

namespace tp {

template <class F>
inline void flowRows(int n, int work, F&& fn) {
    if (work < 120000) { for (int i = 0; i < n; ++i) fn(i); return; }
    parallelRows(0, n, fn);
}

struct GrayImage {
    int w = 0, h = 0;
    std::vector<float> v;
    void reset(int W, int H) { w = W; h = H; v.assign(static_cast<size_t>(W) * H, 0.f); }
    float at(int x, int y) const {
        x = std::clamp(x, 0, w - 1); y = std::clamp(y, 0, h - 1);
        return v[static_cast<size_t>(y) * w + x];
    }
    float bilinear(float fx, float fy) const {
        float x = fx - 0.5f, y = fy - 0.5f;
        float flx = std::floor(clampf(x, -1e6f, 1e6f)), fly = std::floor(clampf(y, -1e6f, 1e6f));
        int x0 = static_cast<int>(flx), y0 = static_cast<int>(fly);
        float tx = x - flx, ty = y - fly;
        return lerpf(lerpf(at(x0, y0), at(x0 + 1, y0), tx), lerpf(at(x0, y0 + 1), at(x0 + 1, y0 + 1), tx), ty);
    }
};

struct FlowField {
    int w = 0, h = 0;
    float scale = 1.f;
    int ox = 0, oy = 0;
    std::vector<float> u, v;
    void reset(int W, int H) { w = W; h = H; u.assign(static_cast<size_t>(W) * H, 0.f); v.assign(static_cast<size_t>(W) * H, 0.f); }
    void sample(float px, float py, float& fu, float& fv) const {
        if (w == 0) { fu = fv = 0.f; return; }
        float gx = (px - ox) / scale, gy = (py - oy) / scale;
        float x = gx - 0.5f, y = gy - 0.5f;
        float flx = std::floor(clampf(x, -1e6f, 1e6f)), fly = std::floor(clampf(y, -1e6f, 1e6f));
        int x0 = static_cast<int>(flx), y0 = static_cast<int>(fly);
        float tx = x - flx, ty = y - fly;
        auto U = [&](int xx, int yy) { xx = std::clamp(xx, 0, w - 1); yy = std::clamp(yy, 0, h - 1); return u[static_cast<size_t>(yy) * w + xx]; };
        auto V = [&](int xx, int yy) { xx = std::clamp(xx, 0, w - 1); yy = std::clamp(yy, 0, h - 1); return v[static_cast<size_t>(yy) * w + xx]; };
        fu = lerpf(lerpf(U(x0, y0), U(x0 + 1, y0), tx), lerpf(U(x0, y0 + 1), U(x0 + 1, y0 + 1), tx), ty) * scale;
        fv = lerpf(lerpf(V(x0, y0), V(x0 + 1, y0), tx), lerpf(V(x0, y0 + 1), V(x0 + 1, y0 + 1), tx), ty) * scale;
    }
};

namespace gpu {
bool available();
bool solveFlow(const std::vector<GrayImage>& previous, const std::vector<GrayImage>& current,
               int iterations, FlowField& output);
}

inline GrayImage flowGray(const ImageView& img, const Rect& r, int f, Encoding) {
    GrayImage g;
    const int W = std::max(1, (r.w() + f - 1) / f), H = std::max(1, (r.h() + f - 1) / f);
    g.reset(W, H);
    if (!img.valid()) return g;
    flowRows(H, W * H * f * f, [&](int j) {
        for (int i = 0; i < W; ++i) {
            float s = 0.f;
            int n = 0;
            for (int y = r.y1 + j * f; y < std::min(r.y2, r.y1 + (j + 1) * f); ++y)
                for (int x = r.x1 + i * f; x < std::min(r.x2, r.x1 + (i + 1) * f); ++x) {
                    const float* q = img.px(std::clamp(x, img.bounds.x1, img.bounds.x2 - 1), std::clamp(y, img.bounds.y1, img.bounds.y2 - 1));
                    s += clampf(0.2126f * q[0] + 0.7152f * q[1] + 0.0722f * q[2], -1.f, 16.f);
                    ++n;
                }
            g.v[static_cast<size_t>(j) * W + i] = n ? s / n : 0.f;
        }
    });
    return g;
}

inline GrayImage flowDown(const GrayImage& a) {
    GrayImage b;
    b.reset(std::max(1, (a.w + 1) / 2), std::max(1, (a.h + 1) / 2));
    flowRows(b.h, b.w * b.h, [&](int j) {
        for (int i = 0; i < b.w; ++i)
            b.v[static_cast<size_t>(j) * b.w + i] = 0.25f * (a.at(2 * i, 2 * j) + a.at(2 * i + 1, 2 * j) + a.at(2 * i, 2 * j + 1) + a.at(2 * i + 1, 2 * j + 1));
    });
    return b;
}

inline void flowBox(std::vector<float>& d, int w, int h, int r) {
    if (r <= 0) return;
    std::vector<float> t(d.size());
    flowRows(h, w * h, [&](int y) {
        float acc = 0.f;
        for (int k = -r; k <= r; ++k) acc += d[static_cast<size_t>(y) * w + std::clamp(k, 0, w - 1)];
        for (int x = 0; x < w; ++x) {
            t[static_cast<size_t>(y) * w + x] = acc / (2 * r + 1);
            acc += d[static_cast<size_t>(y) * w + std::clamp(x + r + 1, 0, w - 1)] - d[static_cast<size_t>(y) * w + std::clamp(x - r, 0, w - 1)];
        }
    });
    flowRows(w, w * h, [&](int x) {
        float acc = 0.f;
        for (int k = -r; k <= r; ++k) acc += t[static_cast<size_t>(std::clamp(k, 0, h - 1)) * w + x];
        for (int y = 0; y < h; ++y) {
            d[static_cast<size_t>(y) * w + x] = acc / (2 * r + 1);
            acc += t[static_cast<size_t>(std::clamp(y + r + 1, 0, h - 1)) * w + x] - t[static_cast<size_t>(std::clamp(y - r, 0, h - 1)) * w + x];
        }
    });
}

inline FlowField computeFlow(const ImageView& prev, const ImageView& cur, const Rect& r, float renderScale, Encoding enc, int quality) {
    FlowField out;
    const int f = std::max(1, static_cast<int>(std::lround(4.0 * std::max(0.25, static_cast<double>(renderScale)) / (quality >= 2 ? 2.0 : 1.0))));
    GrayImage g0 = flowGray(prev, r, f, enc), g1 = flowGray(cur, r, f, enc);
    std::vector<GrayImage> p0{g0}, p1{g1};
    while (p0.back().w > 24 && p0.back().h > 24 && p0.size() < 6) { p0.push_back(flowDown(p0.back())); p1.push_back(flowDown(p1.back())); }
    const int iterations = quality >= 1 ? 4 : 3;
    if (g0.v.size() >= 4096 && gpu::available() && gpu::solveFlow(p0,p1,iterations,out)) {
        out.scale = static_cast<float>(f);
        out.ox = r.x1; out.oy = r.y1;
        return out;
    }
    std::vector<float> U, V;
    int cw = 0, ch = 0;
    for (int lv = static_cast<int>(p0.size()) - 1; lv >= 0; --lv) {
        const GrayImage& A = p0[static_cast<size_t>(lv)];
        const GrayImage& B = p1[static_cast<size_t>(lv)];
        const int w = A.w, h = A.h;
        std::vector<float> nu(static_cast<size_t>(w) * h, 0.f), nv(nu.size(), 0.f);
        if (!U.empty()) {
            flowRows(h, w * h, [&](int y) {
                for (int x = 0; x < w; ++x) {
                    int sx = std::clamp(x / 2, 0, cw - 1), sy = std::clamp(y / 2, 0, ch - 1);
                    nu[static_cast<size_t>(y) * w + x] = 2.f * U[static_cast<size_t>(sy) * cw + sx];
                    nv[static_cast<size_t>(y) * w + x] = 2.f * V[static_cast<size_t>(sy) * cw + sx];
                }
            });
        }
        for (int it = 0; it < iterations; ++it) {
            std::vector<float> xx(nu.size()), xy(nu.size()), yy(nu.size()), xt(nu.size()), yt(nu.size());
            flowRows(h, w * h, [&](int y) {
                for (int x = 0; x < w; ++x) {
                    size_t k = static_cast<size_t>(y) * w + x;
                    float px = x + 0.5f - nu[k], py = y + 0.5f - nv[k];
                    float a = A.bilinear(px, py);
                    float gx = 0.25f * (B.at(x + 1, y) - B.at(x - 1, y) + A.bilinear(px + 1.f, py) - A.bilinear(px - 1.f, py));
                    float gy = 0.25f * (B.at(x, y + 1) - B.at(x, y - 1) + A.bilinear(px, py + 1.f) - A.bilinear(px, py - 1.f));
                    float gt = B.at(x, y) - a;
                    xx[k] = gx * gx; xy[k] = gx * gy; yy[k] = gy * gy; xt[k] = gx * gt; yt[k] = gy * gt;
                }
            });
            const int rad = 3;
            flowBox(xx, w, h, rad); flowBox(xy, w, h, rad); flowBox(yy, w, h, rad); flowBox(xt, w, h, rad); flowBox(yt, w, h, rad);
            flowRows(h, w * h, [&](int y) {
                for (int x = 0; x < w; ++x) {
                    size_t k = static_cast<size_t>(y) * w + x;
                    const float lam = 1e-4f;
                    float a11 = xx[k] + lam, a12 = xy[k], a22 = yy[k] + lam;
                    float det = a11 * a22 - a12 * a12;
                    if (!(det > 1e-12f)) continue;
                    float du = -(a22 * xt[k] - a12 * yt[k]) / det;
                    float dv = -(a11 * yt[k] - a12 * xt[k]) / det;
                    du = clampf(du, -2.f, 2.f); dv = clampf(dv, -2.f, 2.f);
                    nu[k] += du; nv[k] += dv;
                }
            });
            flowBox(nu, w, h, 1);
            flowBox(nv, w, h, 1);
        }
        U.swap(nu); V.swap(nv); cw = w; ch = h;
    }
    {
        const GrayImage& A = p0[0];
        const GrayImage& B = p1[0];
        const float cap = 0.12f * static_cast<float>(std::max(cw, ch));
        std::vector<float> keep(U.size(), 1.f);
        flowRows(ch, cw * ch, [&](int y) {
            for (int x = 0; x < cw; ++x) {
                size_t k = static_cast<size_t>(y) * cw + x;
                float e = std::fabs(B.at(x, y) - A.bilinear(x + 0.5f - U[k], y + 0.5f - V[k]));
                float mag = std::hypot(U[k], V[k]);
                float tol = 0.04f + 0.08f * std::fabs(B.at(x, y));
                if (e > tol || mag > cap || !std::isfinite(mag)) keep[k] = 0.f;
            }
        });
        flowBox(keep, cw, ch, 1);
        for (size_t k = 0; k < U.size(); ++k) {
            float w = smoothstepf(0.5f, 0.9f, keep[k]);
            U[k] *= w; V[k] *= w;
        }
    }
    out.reset(cw, ch);
    out.u = U; out.v = V;
    out.scale = static_cast<float>(f);
    out.ox = r.x1; out.oy = r.y1;
    for (size_t k = 0; k < out.u.size(); ++k) {
        if (!std::isfinite(out.u[k])) out.u[k] = 0.f;
        if (!std::isfinite(out.v[k])) out.v[k] = 0.f;
    }
    return out;
}

}
