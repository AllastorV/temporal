#pragma once
#include "../core/core.h"
#include "../core/numpy_random.h"
#include "../core/cutwright_host.h"
#include "../gpu/gpu.h"
#include <cmath>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace tp::cwart {

inline std::array<double, 3> hexRgb(const char* hex) {
    const auto part = [&](int i) { return std::strtol(std::string(hex + 1 + 2 * i, 2).c_str(), nullptr, 16) / 255.0; };
    return {part(0), part(1), part(2)};
}

template <class F>
void forEachUnpremultiplied(const ImageView& exchange, F&& fn) {
    const int w = exchange.bounds.w(), h = exchange.bounds.h();
    parallelRows(0, h, [&](int y) {
        float* row = exchange.px(0, y);
        for (int x = 0; x < w; ++x) {
            float* c = row + 4 * x;
            const float a = std::max(c[3], 1e-6f);
            float rgb[3] = {std::clamp(c[0] / a, 0.f, 1.f), std::clamp(c[1] / a, 0.f, 1.f), std::clamp(c[2] / a, 0.f, 1.f)};
            fn(rgb);
            for (int k = 0; k < 3; ++k) c[k] = std::clamp(rgb[k], 0.f, 1.f) * c[3];
        }
    });
}

inline void thermal(const ImageView& exchange) {
    static const char* iron[] = {"#000000", "#1c0050", "#7a0a8c", "#d4282d", "#f5871e", "#fde26a", "#ffffff"};
    std::array<std::array<double, 3>, 7> stops;
    for (int i = 0; i < 7; ++i) stops[i] = hexRgb(iron[i]);
    forEachUnpremultiplied(exchange, [&](float* x) {
        const float y = std::clamp(0.2126f * x[0] + 0.7152f * x[1] + 0.0722f * x[2], 0.f, 1.f);
        float out[3] = {float(stops[0][0]), float(stops[0][1]), float(stops[0][2])};
        for (int i = 0; i < 6; ++i) {
            const float t = std::clamp(y * 6.f - float(i), 0.f, 1.f);
            for (int ch = 0; ch < 3; ++ch) out[ch] = out[ch] + float(stops[i + 1][ch] - stops[i][ch]) * t;
        }
        for (int ch = 0; ch < 3; ++ch) x[ch] = std::clamp(x[ch] + (out[ch] - x[ch]) * 1.f, 0.f, 1.f);
    });
}

struct TopDown {
    int w = 0, h = 0;
    std::vector<float> rgb, alpha;
    explicit TopDown(const ImageView& exchange) : w(exchange.bounds.w()), h(exchange.bounds.h()), rgb(size_t(3) * w * h), alpha(size_t(w) * h) {
        parallelRows(0, h, [&](int Y) {
            const float* row = exchange.px(0, h - 1 - Y);
            for (int X = 0; X < w; ++X) {
                const float* c = row + 4 * X;
                const float a = std::max(c[3], 1e-6f);
                for (int k = 0; k < 3; ++k) rgb[(size_t(k) * h + Y) * w + X] = std::clamp(c[k] / a, 0.f, 1.f);
                alpha[size_t(Y) * w + X] = c[3];
            }
        });
    }
    float& at(int c, int Y, int X) { return rgb[(size_t(c) * h + Y) * w + X]; }
    void store(const ImageView& exchange) const {
        parallelRows(0, h, [&](int Y) {
            float* row = exchange.px(0, h - 1 - Y);
            for (int X = 0; X < w; ++X) {
                const float a = alpha[size_t(Y) * w + X];
                for (int k = 0; k < 3; ++k) row[4 * X + k] = std::clamp(rgb[(size_t(k) * h + Y) * w + X], 0.f, 1.f) * a;
                row[4 * X + 3] = a;
            }
        });
    }
};

inline int pyRound(double v) { return static_cast<int>(std::nearbyint(v)); }

inline void pixelArt(const ImageView& exchange) {
    TopDown img(exchange);
    const int w = img.w, h = img.h;
    const int s = std::max(2, pyRound(0.012 * std::min(h, w)));
    const int hh = std::max(1, h / s), ww = std::max(1, w / s);
    static const char* pastel[] = {"#2b2d42", "#5c6784", "#a3b1c6", "#f2efe9", "#e07a5f", "#f2cc8f", "#81b29a", "#3d405b"};
    float pal[8][3];
    for (int i = 0; i < 8; ++i) {
        const auto c = hexRgb(pastel[i]);
        for (int k = 0; k < 3; ++k) pal[i][k] = float(c[k]);
    }
    static const int bayer4[4][4] = {{0, 8, 2, 10}, {12, 4, 14, 6}, {3, 11, 1, 9}, {15, 7, 13, 5}};
    std::vector<int> choice(size_t(hh) * ww);
    parallelRows(0, hh, [&](int cy) {
        for (int cx = 0; cx < ww; ++cx) {
            float small[3];
            for (int k = 0; k < 3; ++k) {
                double sum = 0;
                for (int dy = 0; dy < s; ++dy)
                    for (int dx = 0; dx < s; ++dx) sum += img.at(k, std::min(cy * s + dy, h - 1), std::min(cx * s + dx, w - 1));
                small[k] = float(sum / (double(s) * s));
            }
            const float t = float((bayer4[cy % 4][cx % 4] + 0.5) / 16.0 - 0.5);
            for (float& v : small) v = v + t * 0.5f * 0.25f;
            int best = 0;
            float bestD = 0;
            for (int i = 0; i < 8; ++i) {
                const float dr = small[0] - pal[i][0], dg = small[1] - pal[i][1], db = small[2] - pal[i][2];
                const float d = dr * dr * 0.3f + dg * dg * 0.59f + db * db * 0.11f;
                if (i == 0 || d < bestD) { best = i; bestD = d; }
            }
            choice[size_t(cy) * ww + cx] = best;
        }
    });
    parallelRows(0, h, [&](int Y) {
        const int cy = std::min(Y / s, hh - 1);
        for (int X = 0; X < w; ++X) {
            const int c = choice[size_t(cy) * ww + std::min(X / s, ww - 1)];
            for (int k = 0; k < 3; ++k) img.at(k, Y, X) = pal[c][k];
        }
    });
    img.store(exchange);
}

inline void particles(const ImageView& exchange, int index, double fps, long long seed) {
    TopDown img(exchange);
    const int w = img.w, h = img.h, n = 120;
    NumpyPcg64 rng{static_cast<uint32_t>(seed)};
    std::vector<double> start(2 * n), scale(n), speed(n), phase(n), sway(n);
    for (double& v : start) v = rng.nextDouble();
    for (double& v : scale) v = 0.5 + 1.0 * rng.nextDouble();
    for (double& v : speed) v = 0.6 + (1.4 - 0.6) * rng.nextDouble();
    for (double& v : phase) v = rng.nextDouble();
    for (double& v : sway) v = rng.nextDouble();
    const double t = index / fps, shortSide = std::min(h, w), pi = 3.14159265358979323846;
    const double angle = 270.0 * pi / 180.0, margin = 0.1, size = 0.006, drift = 0.03;
    const auto color = hexRgb("#fff4e0");
    const float col[3] = {float(color[0]), float(color[1]), float(color[2])};
    std::vector<float> layer(size_t(3) * w * h, 0.f);
    const auto pyMod = [](double a, double b) { const double r = std::fmod(a, b); return r < 0 ? r + b : r; };
    for (int i = 0; i < n; ++i) {
        const double travel = drift * t * speed[i] * shortSide;
        double px = start[2 * i] * (1 + 2 * margin) * w + std::cos(angle) * travel + 0.01 * shortSide * std::sin(2 * pi * (0.15 * t + sway[i]));
        double py = start[2 * i + 1] * (1 + 2 * margin) * h + std::sin(angle) * travel;
        px = pyMod(px, (1 + 2 * margin) * w) - margin * w;
        py = pyMod(py, (1 + 2 * margin) * h) - margin * h;
        const double r = std::max(0.5, size * shortSide * scale[i]);
        const int reach = static_cast<int>(std::ceil(r * 2)) + 1;
        const int ix = static_cast<int>(px), iy = static_cast<int>(py);
        const int x0 = std::max(0, ix - reach), x1 = std::min(w, ix + reach + 1);
        const int y0 = std::max(0, iy - reach), y1 = std::min(h, iy + reach + 1);
        if (x0 >= x1 || y0 >= y1) continue;
        const double denom = 2 * (r / 2) * (r / 2);
        for (int Y = y0; Y < y1; ++Y)
            for (int X = x0; X < x1; ++X) {
                const double dx = double(float(X)) - px, dy = double(float(Y)) - py;
                const float sprite = float(std::exp(-(dx * dx + dy * dy) / denom));
                for (int k = 0; k < 3; ++k) layer[(size_t(k) * h + Y) * w + X] += sprite * col[k];
            }
    }
    parallelRows(0, h, [&](int Y) {
        for (int X = 0; X < w; ++X)
            for (int k = 0; k < 3; ++k) {
                const float b = std::clamp(layer[(size_t(k) * h + Y) * w + X] * 0.6f, 0.f, 1.f);
                float& a = img.at(k, Y, X);
                a = 1.f - (1.f - a) * (1.f - b);
            }
    });
    img.store(exchange);
}

using Planes = std::vector<cw::Plane>;

inline float sampleZero(const cw::Plane& p, float gy, float gx) {
    const float y0 = std::floor(gy), x0 = std::floor(gx), wy = gy - y0, wx = gx - x0;
    float out = 0;
    const float ks[4] = {(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx};
    const float dy[4] = {0, 0, 1, 1}, dx[4] = {0, 1, 0, 1};
    for (int k = 0; k < 4; ++k) {
        const float yy = y0 + dy[k], xx = x0 + dx[k];
        if (yy < 0 || yy > p.h - 1 || xx < 0 || xx > p.w - 1) continue;
        out = out + p.at(int(yy), int(xx)) * ks[k];
    }
    return out;
}

inline cw::Plane shifted(const cw::Plane& p, double ox, double oy) {
    cw::Plane o(p.h, p.w);
    const float fx = float(ox), fy = float(oy);
    const float rx = std::nearbyint(fx), ry = std::nearbyint(fy);
    if (std::fabs(fx - rx) < 1e-6f && std::fabs(fy - ry) < 1e-6f && std::fabs(rx) < 1e6f && std::fabs(ry) < 1e6f) {
        const int sx = int(rx), sy = int(ry);
        parallelRows(0, p.h, [&](int y) {
            const int yy = y - sy;
            if (yy < 0 || yy >= p.h) return;
            const int x0 = std::max(0, sx), x1 = std::min(p.w, p.w + sx);
            for (int x = x0; x < x1; ++x) o.at(y, x) = p.at(yy, x - sx);
        });
        return o;
    }
    parallelRows(0, p.h, [&](int y) {
        for (int x = 0; x < p.w; ++x) o.at(y, x) = sampleZero(p, float(y) - fy, float(x) - fx);
    });
    return o;
}

inline Planes shifted(const Planes& p, double ox, double oy) {
    Planes o;
    for (const auto& c : p) o.push_back(shifted(c, ox, oy));
    return o;
}

inline void addScaled(Planes& a, const Planes& b, float k) {
    for (size_t c = 0; c < a.size(); ++c)
        for (size_t i = 0; i < a[c].v.size(); ++i) a[c].v[i] = a[c].v[i] + k * b[c].v[i];
}

struct Shift {
    float fx, fy;
    int sx, sy;
    bool integral;
    Shift(double ox, double oy) : fx(float(ox)), fy(float(oy)) {
        const float rx = std::nearbyint(fx), ry = std::nearbyint(fy);
        integral = std::fabs(fx - rx) < 1e-6f && std::fabs(fy - ry) < 1e-6f && std::fabs(rx) < 1e6f && std::fabs(ry) < 1e6f;
        sx = integral ? int(rx) : 0;
        sy = integral ? int(ry) : 0;
    }
    float at(const cw::Plane& p, int y, int x) const {
        if (!integral) return sampleZero(p, float(y) - fy, float(x) - fx);
        const int yy = y - sy, xx = x - sx;
        return yy < 0 || yy >= p.h || xx < 0 || xx >= p.w ? 0.f : p.at(yy, xx);
    }
};

inline Planes span(const Planes& s, double dx, double dy, double n) {
    Planes out = s, next = s;
    for (long long step = 1; step < n; step *= 3) {
        const Shift a(dx * step, dy * step), b(-dx * step, -dy * step);
        for (size_t c = 0; c < out.size(); ++c)
            parallelRows(0, out[c].h, [&](int y) {
                for (int x = 0; x < out[c].w; ++x) next[c].at(y, x) = out[c].at(y, x) + a.at(out[c], y, x) + b.at(out[c], y, x);
            });
        std::swap(out, next);
    }
    return out;
}

inline Planes ray(const Planes& s, double dx, double dy, double length) {
    const double r = std::exp(-3 / std::max(length, 1.0));
    Planes out = s, next = s;
    for (long long step = 1; step < 2.5 * length; step *= 2) {
        const Shift m(dx * step, dy * step);
        const float k = float(std::pow(r, double(step)));
        for (size_t c = 0; c < out.size(); ++c)
            parallelRows(0, out[c].h, [&](int y) {
                for (int x = 0; x < out[c].w; ++x) next[c].at(y, x) = out[c].at(y, x) + k * m.at(out[c], y, x);
            });
        std::swap(out, next);
    }
    return out;
}

inline void star(const ImageView& exchange) {
    TopDown img(exchange);
    const int w = img.w, h = img.h;
    const double threshold = 0.85;
    cw::Plane over(h, w);
    Planes lit(3, cw::Plane(h, w));
    parallelRows(0, h, [&](int Y) {
        for (int X = 0; X < w; ++X) {
            const float pk = std::max(std::max(img.at(0, Y, X), img.at(1, Y, X)), img.at(2, Y, X));
            const float o = std::clamp((pk - float(threshold)) / float(1 - threshold), 0.f, 1.f);
            over.at(Y, X) = o;
            for (int k = 0; k < 3; ++k) lit[k].at(Y, X) = img.at(k, Y, X) * o;
        }
    });
    const int q = std::max(1, std::min(h, w) / 360);
    Planes light;
    for (const auto& c : lit) light.push_back(cw::downscale(c, q));
    const cw::Plane amount = cw::downscale(over, q);
    const int sh = amount.h, sw = amount.w;
    const double s1 = std::max(0.5, 0.03 * std::min(sh, sw) / 6);
    const double length = 0.08 * std::min(h, w) / q;
    const double pi = 3.14159265358979323846;
    const size_t count = size_t(sh) * sw;
    Planes acc(3, cw::Plane(sh, sw));
    const bool onGpu = gpu::cutwrightStar(amount, light, s1, length, acc);
    cw::Plane near, wide;
    Planes src = light;
    if (!onGpu) {
        acc.assign(3, cw::Plane(sh, sw));
        near = cw::blur(amount, float(s1));
        wide = cw::blur(amount, float(4 * s1));
        for (size_t i = 0; i < count; ++i) {
            const float k = std::clamp((near.v[i] / (wide.v[i] + 1e-4f) - 2.f) / 6.f, 0.f, 1.f);
            for (auto& c : src) c.v[i] = c.v[i] * k;
        }
    }
    for (int j = 0; j < 2 && !onGpu; ++j) {
        const double theta = (0.0 + j * 180.0 / 2) * pi / 180.0;
        const double dx = std::cos(theta), dy = std::sin(theta);
        const cw::Plane a = shifted(near, -dy, dx), b = shifted(near, dy, -dx);
        Planes energy = span(src, -dy, dx, 8 * s1);
        Planes line(3, cw::Plane(sh, sw));
        for (size_t i = 0; i < near.v.size(); ++i) {
            const float top = near.v[i] + 1e-5f;
            const bool ridge = top >= a.v[i] && top >= b.v[i] && near.v[i] > 1e-3f;
            const float pk = std::max(std::max(energy[0].v[i], energy[1].v[i]), energy[2].v[i]);
            const float norm = std::sqrt(std::max(pk, 1e-3f));
            for (int c = 0; c < 3; ++c) line[c].v[i] = energy[c].v[i] * (ridge ? 1.f : 0.f) / norm;
        }
        const Planes forward = ray(line, dx, dy, length), backward = ray(line, -dx, -dy, length);
        for (int c = 0; c < 3; ++c)
            for (size_t i = 0; i < near.v.size(); ++i)
                acc[c].v[i] = acc[c].v[i] + forward[c].v[i] + backward[c].v[i] - 2 * line[c].v[i];
    }
    Planes armsSmall(3, cw::Plane(sh, sw));
    for (size_t i = 0; i < count; ++i) {
        const float top = std::max(std::max(std::max(acc[0].v[i], acc[1].v[i]), acc[2].v[i]), 0.f);
        const float soft = (1 - std::exp(-0.35f * top)) / (top + 1e-6f);
        for (int c = 0; c < 3; ++c) armsSmall[c].v[i] = acc[c].v[i] * soft;
    }
    for (int c = 0; c < 3; ++c) {
        const cw::Plane arms = cw::upscale(armsSmall[c], h, w);
        parallelRows(0, h, [&](int Y) {
            for (int X = 0; X < w; ++X) {
                float& v = img.at(c, Y, X);
                v = 1.f - (1.f - v) * (1.f - std::clamp(arms.at(Y, X), 0.f, 1.f));
            }
        });
    }
    img.store(exchange);
}

struct MotionInkState {
    bool valid = false;
    int index = -1, sh = 0, sw = 0;
    Planes anchors[3];
    int ticks[3] = {-1, -1, -1};
    Planes ink;
    bool hasInk = false;
    std::vector<double> logProduct;
    struct Saved { std::vector<float> weight; Planes marks; };
    std::vector<Saved> queue;
    int count = -1;
};

inline float lum3(float r, float g, float b) { return r * 0.2126f + g * 0.7152f + b * 0.0722f; }

inline double percentile(std::vector<float>& v, double q) {
    const double idx = q / 100.0 * double(v.size() - 1);
    const size_t lo = size_t(std::floor(idx));
    const double t = idx - double(lo);
    std::nth_element(v.begin(), v.begin() + lo, v.end());
    const float a = v[lo];
    const float b = lo + 1 < v.size() ? *std::min_element(v.begin() + lo + 1, v.end()) : a;
    const float diff = b - a;
    return t >= 0.5 ? double(b) - double(diff) * (1 - t) : double(a) + double(diff) * t;
}

inline std::vector<float> gradientAxis(const cw::Plane& p, bool rows) {
    std::vector<float> g(p.v.size());
    const int n = rows ? p.h : p.w;
    parallelRows(0, p.h, [&](int y) {
        for (int x = 0; x < p.w; ++x) {
            const int i = rows ? y : x;
            const auto at = [&](int k) { return rows ? p.at(k, x) : p.at(y, k); };
            float v;
            if (n < 2) v = 0;
            else if (i == 0) v = at(1) - at(0);
            else if (i == n - 1) v = at(n - 1) - at(n - 2);
            else v = (at(i + 1) - at(i - 1)) / 2.0f;
            g[size_t(y) * p.w + x] = v;
        }
    });
    return g;
}

inline Planes TopDownPlanes(const ImageView& exchange) {
    TopDown img(exchange);
    Planes out(3, cw::Plane(img.h, img.w));
    for (int c = 0; c < 3; ++c)
        for (int Y = 0; Y < img.h; ++Y)
            for (int X = 0; X < img.w; ++X) out[c].at(Y, X) = std::clamp(img.at(c, Y, X), 0.f, 1.f);
    return out;
}

inline Planes resizeFromExchange(const ImageView& exchange, int sh, int sw) {
    const int w = exchange.bounds.w(), h = exchange.bounds.h();
    const auto px = [&](int Y, int X, int c) {
        const float* p = exchange.px(X, h - 1 - Y);
        return std::clamp(std::clamp(p[c] / std::max(p[3], 1e-6f), 0.f, 1.f), 0.f, 1.f);
    };
    Planes out(3, cw::Plane(sh, sw));
    parallelRows(0, sh, [&](int y) {
        for (int x = 0; x < sw; ++x) {
            const float gy = (float(y) + 0.5f) * h / sh - 0.5f, gx = (float(x) + 0.5f) * w / sw - 0.5f;
            const float y0 = std::floor(gy), x0 = std::floor(gx), wy = gy - y0, wx = gx - x0;
            const int ya = int(std::clamp(y0, 0.f, h - 1.f)), yb = int(std::clamp(y0 + 1.f, 0.f, h - 1.f));
            const int xa = int(std::clamp(x0, 0.f, w - 1.f)), xb = int(std::clamp(x0 + 1.f, 0.f, w - 1.f));
            for (int c = 0; c < 3; ++c)
                out[c].at(y, x) = px(ya, xa, c) * ((1 - wy) * (1 - wx)) + px(ya, xb, c) * ((1 - wy) * wx) +
                                  px(yb, xa, c) * (wy * (1 - wx)) + px(yb, xb, c) * (wy * wx);
        }
    });
    return out;
}

inline void motionInk(MotionInkState& st, const ImageView& exchange, int index, double fps, bool output) {
    const int w = exchange.bounds.w(), h = exchange.bounds.h();
    const double ratio = std::min(1.0, 320.0 / w);
    const int sh = std::max(4, pyRound(h * ratio)), sw = std::max(4, pyRound(w * ratio));
    const Planes small = sh == h && sw == w ? TopDownPlanes(exchange) : resizeFromExchange(exchange, sh, sw);
    if (st.valid && (st.index != index - 1 || st.sh != sh || st.sw != sw)) st = MotionInkState{};
    st.valid = true;
    st.index = index;
    st.sh = sh;
    st.sw = sw;
    if (st.anchors[0].empty())
        for (auto& a : st.anchors) a = small;
    const double t = index / fps;
    const size_t n = size_t(sh) * sw;
    std::vector<float> energy(n, 0.f);
    const double periods[3] = {0.075, 0.28, 0.95};
    const float weights[3] = {0.7f, 0.9f, 0.45f};
    const Planes middle = st.anchors[1];
    for (int j = 0; j < 3; ++j) {
        parallelRows(0, sh, [&](int row) {
            for (size_t i = size_t(row) * sw; i < size_t(row + 1) * sw; ++i) {
                const float dr = small[0].v[i] - st.anchors[j][0].v[i], dg = small[1].v[i] - st.anchors[j][1].v[i], db = small[2].v[i] - st.anchors[j][2].v[i];
                const float d = std::max(std::fabs(lum3(dr, dg, db)), std::sqrt(dr * dr + dg * dg + db * db) * 0.474f) * weights[j];
                energy[i] = j == 0 ? d : std::max(energy[i], d);
            }
        });
        const int tick = int(std::floor(t / periods[j]));
        if (tick != st.ticks[j]) { st.anchors[j] = small; st.ticks[j] = tick; }
    }
    std::vector<float> ranked = energy;
    const double p68 = percentile(ranked, 68), p90 = percentile(ranked, 90);
    const double threshold = std::clamp(0.145 - std::pow(0.74, 0.72) * 0.132 + p68 * (0.45 + 0.62 * 1.85), 0.006, 0.34);
    size_t over = 0;
    for (float e : energy) over += e > float(threshold);
    if (double(over) / double(n) > 0.58 || (p68 > 0.13 && p90 > 0.24)) {
        st = MotionInkState{};
        if (output) {
            TopDown img(exchange);
            for (auto& v : img.rgb) v = 0;
            img.store(exchange);
        }
        return;
    }
    const float body = 0.24f;
    std::vector<float> grown(n);
    parallelRows(0, sh, [&](int y) {
        for (int x = 0; x < sw; ++x) {
            const auto e = [&](int yy, int xx) { return energy[size_t((yy + sh) % sh) * sw + (xx + sw) % sw]; };
            float v = e(y, x);
            v = std::max(v, e(y - 1, x) * body);
            v = std::max(v, e(y + 1, x) * body);
            v = std::max(v, e(y, x - 1) * body);
            v = std::max(v, e(y, x + 1) * body);
            grown[size_t(y) * sw + x] = v;
        }
    });
    energy = grown;
    const double spread = std::max(0.003, threshold * 0.42);
    const float lo = float(threshold - spread);
    const float denom = float(std::max((threshold + spread) - (threshold - spread), 1e-6));
    cw::Plane avg(sh, sw), before(sh, sw), after(sh, sw);
    parallelRows(0, sh, [&](int row) {
        for (size_t i = size_t(row) * sw; i < size_t(row + 1) * sw; ++i) {
            before.v[i] = lum3(middle[0].v[i], middle[1].v[i], middle[2].v[i]);
            after.v[i] = lum3(small[0].v[i], small[1].v[i], small[2].v[i]);
            avg.v[i] = (before.v[i] + after.v[i]) / 2.0f;
        }
    });
    const auto gy = gradientAxis(avg, true), gx = gradientAxis(avg, false);
    const float gain = float(1.0 * 10 + 0.74 * 1.5);
    Planes marks(3, cw::Plane(sh, sw));
    parallelRows(0, sh, [&](int row) {
    for (size_t i = size_t(row) * sw; i < size_t(row + 1) * sw; ++i) {
        const float v0 = std::clamp((energy[i] - lo) / denom, 0.f, 1.f);
        const float gate = std::pow(v0 * v0 * (3 - 2 * v0), 0.68f);
        const float dt = after.v[i] - before.v[i];
        const float norm = gx[i] * gx[i] + gy[i] * gy[i] + 0.0025f;
        const float u = -dt * gx[i] / norm, v = -dt * gy[i] / norm;
        const float hue = std::atan2(v, u) / float(2 * 3.14159265358979323846) + 0.5f;
        float hm = std::fmod(hue, 1.f);
        if (hm < 0) hm += 1.f;
        const float hh = hm * 6;
        float frac = std::fmod(hh, 1.f);
        if (frac < 0) frac += 1.f;
        const float a = float(1.0 - 0.72), b = 1 - 0.72f * frac, cc = 1 - 0.72f * (1 - frac);
        int k = int(std::floor(hh)) % 6;
        if (k < 0) k += 6;
        const float rgb[6][3] = {{1, cc, a}, {b, 1, a}, {a, 1, cc}, {a, b, 1}, {cc, a, 1}, {1, a, b}};
        const float en = gate * (1 - std::exp(-energy[i] * gain));
        energy[i] = en;
        for (int c = 0; c < 3; ++c) marks[c].v[i] = rgb[k][c] * en;
    }
    });
    const int count = std::max(1, pyRound((1.1 + 0.95) * fps));
    const double decay = std::exp(-1 / std::max(1.0, 1.1 * fps));
    if (st.count != count) {
        st.queue.clear();
        st.hasInk = false;
        st.logProduct.assign(n, 0.0);
    }
    if (st.logProduct.size() != n) st.logProduct.assign(n, 0.0);
    std::vector<float> weight(n);
    const MotionInkState::Saved* old = int(st.queue.size()) == count ? &st.queue.front() : nullptr;
    Planes memory(3, cw::Plane(sh, sw));
    parallelRows(0, sh, [&](int row) {
        for (size_t i = size_t(row) * sw; i < size_t(row + 1) * sw; ++i) {
            weight[i] = std::max(0.001f, (1 - energy[i]) * float(decay));
            st.logProduct[i] += std::log(double(weight[i]));
            for (int c = 0; c < 3; ++c) memory[c].v[i] = (st.hasInk ? st.ink[c].v[i] * weight[i] : 0.f) + marks[c].v[i] * 0.82f;
            if (old) {
                st.logProduct[i] -= std::log(double(old->weight[i]));
                const float keep = float(std::exp(st.logProduct[i]));
                for (int c = 0; c < 3; ++c) memory[c].v[i] -= keep * old->marks[c].v[i] * 0.82f;
            }
            for (int c = 0; c < 3; ++c) memory[c].v[i] = std::max(memory[c].v[i], 0.f);
        }
    });
    if (old) st.queue.erase(st.queue.begin());
    st.queue.push_back({std::move(weight), std::move(marks)});
    st.ink = std::move(memory);
    st.hasInk = true;
    st.count = count;
    if (!output) return;
    const Planes& ink = st.ink;
    const double size = std::max(1.0, 20.0 * w / 1200.0 * 0.18);
    static const float table[4][4] = {{0, 8, 2, 10}, {12, 4, 14, 6}, {3, 11, 1, 9}, {15, 7, 13, 5}};
    const bool same = sh == h && sw == w;
    parallelRows(0, h, [&](int Y) {
        float* line = exchange.px(0, h - 1 - Y);
        const float gy = (float(Y) + 0.5f) * sh / h - 0.5f;
        for (int X = 0; X < w; ++X) {
            const float gx = (float(X) + 0.5f) * sw / w - 0.5f;
            float src[3];
            for (int c = 0; c < 3; ++c) src[c] = same ? ink[c].at(Y, X) : cw::sampleClamp(ink[c], gy, gx);
            const float level = std::max(std::max(src[0], src[1]), src[2]);
            const float a = table[int(Y / size) % 4][int(X / size) % 4] / 16;
            const float k = (level > a ? 1.f : 0.f) * std::min(1.f, level + 0.35f);
            const float d = std::max(level, 1e-5f);
            float* px = line + 4 * X;
            const float alpha = px[3], div = std::max(alpha, 1e-6f);
            for (int c = 0; c < 3; ++c) {
                const float cur = std::clamp(px[c] / div, 0.f, 1.f);
                const float v = src[c] / d * k + cur * 0.f;
                px[c] = std::clamp(std::clamp(cur + (v - cur) * 1.f, 0.f, 1.f), 0.f, 1.f) * alpha;
            }
            px[3] = alpha;
        }
    });
}

inline bool native(int preset, const ImageView& exchange, int index, double fps, long long seed) {
    if (preset == 3) { star(exchange); return true; }
    if (preset == 6) { thermal(exchange); return true; }
    if (preset == 4) { particles(exchange, index, fps, seed); return true; }
    if (preset == 5) { pixelArt(exchange); return true; }
    return false;
}

}
