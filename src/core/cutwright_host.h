#pragma once
#include "core.h"
#include <algorithm>
#include <cmath>
#include <mutex>
#include <utility>
#include <vector>

namespace tp::cw {

struct FftAxis {
    std::vector<float> weights;
    int radius = 0, pad = 0, period = 1;
};

inline FftAxis fftGauss(float sigma, int n) {
    static std::mutex cacheLock;
    static std::vector<std::pair<std::pair<float, int>, FftAxis>> cache;
    std::lock_guard<std::mutex> guard(cacheLock);
    for (const auto& e : cache) if (e.first.first == sigma && e.first.second == n) return e.second;
    FftAxis a;
    a.pad = std::min(n - 1, static_cast<int>(std::ceil(3 * double(sigma))));
    a.period = n + 2 * a.pad;
    const double pi = 3.14159265358979323846;
    std::vector<double> g(a.period), k(a.period / 2 + 1);
    for (int f = 0; f < a.period; ++f) {
        const double q = double(f < (a.period + 1) / 2 ? f : f - a.period) / a.period;
        g[f] = std::exp(-2 * pi * pi * double(sigma) * sigma * q * q);
    }
    for (int m = 0; m <= a.period / 2; ++m) {
        double v = 0;
        for (int f = 0; f < a.period; ++f) v += g[f] * std::cos(2 * pi * double(f) * m / a.period);
        k[m] = v / a.period;
    }
    a.radius = a.period / 2;
    while (a.radius > 1 && std::fabs(k[a.radius]) < 1e-6) --a.radius;
    for (int m = -a.radius; m <= a.radius; ++m) a.weights.push_back(static_cast<float>(k[std::abs(m)]));
    if (cache.size() > 64) cache.clear();
    cache.push_back({{sigma, n}, a});
    return a;
}

template <class T>
class Pooled {
    static std::mutex& lock() {
        static std::mutex m;
        return m;
    }
    static std::vector<T>& idle() {
        static std::vector<T> v;
        return v;
    }
public:
    T item;
    Pooled() {
        std::lock_guard<std::mutex> guard(lock());
        if (!idle().empty()) {
            item = std::move(idle().back());
            idle().pop_back();
        }
    }
    ~Pooled() {
        std::lock_guard<std::mutex> guard(lock());
        if (idle().size() < 2) idle().push_back(std::move(item));
    }
    Pooled(const Pooled&) = delete;
    Pooled& operator=(const Pooled&) = delete;
    static size_t idleCount() {
        std::lock_guard<std::mutex> guard(lock());
        return idle().size();
    }
};

template <class T>
inline void fit(std::vector<T>& v, size_t n) {
    v.resize(n);
    if (v.capacity() > 2 * n) v.shrink_to_fit();
}

inline int reflect(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i > n - 1) i = i < 0 ? -i : 2 * (n - 1) - i;
    return i;
}

struct Plane {
    int h = 0, w = 0;
    std::vector<float> v;
    Plane() = default;
    Plane(int hh, int ww, float fill = 0) : h(hh), w(ww), v(size_t(hh) * ww, fill) {}
    float& at(int y, int x) { return v[size_t(y) * w + x]; }
    float at(int y, int x) const { return v[size_t(y) * w + x]; }
};

inline Plane blur(const Plane& p, float sigma) {
    if (sigma < 0.3f) return p;
    const FftAxis ax = fftGauss(sigma, p.w), ay = fftGauss(sigma, p.h);
    Plane t(p.h, p.w), o(p.h, p.w);
    parallelRows(0, p.h, [&](int y) {
        for (int x = 0; x < p.w; ++x) {
            float s = 0;
            for (int m = -ax.radius; m <= ax.radius; ++m) {
                int j = (x + ax.pad - m) % ax.period;
                if (j < 0) j += ax.period;
                s += p.at(y, reflect(j - ax.pad, p.w)) * ax.weights[m + ax.radius];
            }
            t.at(y, x) = s;
        }
    });
    parallelRows(0, p.h, [&](int y) {
        for (int x = 0; x < p.w; ++x) {
            float s = 0;
            for (int m = -ay.radius; m <= ay.radius; ++m) {
                int j = (y + ay.pad - m) % ay.period;
                if (j < 0) j += ay.period;
                s += t.at(reflect(j - ay.pad, p.h), x) * ay.weights[m + ay.radius];
            }
            o.at(y, x) = s;
        }
    });
    return o;
}

inline Plane downscale(const Plane& p, int q) {
    if (q <= 1) return p;
    Plane o(p.h / q, p.w / q);
    parallelRows(0, o.h, [&](int y) {
        for (int x = 0; x < o.w; ++x) {
            float s = 0;
            for (int dy = 0; dy < q; ++dy)
                for (int dx = 0; dx < q; ++dx) s += p.at(y * q + dy, x * q + dx);
            o.at(y, x) = s / float(q * q);
        }
    });
    return o;
}

inline float sampleClamp(const Plane& p, float gy, float gx) {
    const float y0 = std::floor(gy), x0 = std::floor(gx), wy = gy - y0, wx = gx - x0;
    const int ya = int(std::clamp(y0, 0.f, p.h - 1.f)), yb = int(std::clamp(y0 + 1.f, 0.f, p.h - 1.f));
    const int xa = int(std::clamp(x0, 0.f, p.w - 1.f)), xb = int(std::clamp(x0 + 1.f, 0.f, p.w - 1.f));
    return p.at(ya, xa) * ((1 - wy) * (1 - wx)) + p.at(ya, xb) * ((1 - wy) * wx) + p.at(yb, xa) * (wy * (1 - wx)) + p.at(yb, xb) * (wy * wx);
}

inline Plane upscale(const Plane& p, int h, int w) {
    if (p.h == h && p.w == w) return p;
    Plane o(h, w);
    parallelRows(0, h, [&](int y) {
        for (int x = 0; x < w; ++x)
            o.at(y, x) = sampleClamp(p, (float(y) + 0.5f) * p.h / h - 0.5f, (float(x) + 0.5f) * p.w / w - 0.5f);
    });
    return o;
}

}
