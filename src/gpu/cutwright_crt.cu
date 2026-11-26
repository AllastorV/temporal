#include "gpu_common.cuh"
#include "../core/numpy_random.h"
#include "../core/cutwright_host.h"
#include <algorithm>
#include <cmath>

namespace tp::gpu {

namespace {

__device__ inline float4 crtAt(const float4* img, int w, int h, int X, int Y) {
    return img[(size_t)Y * w + X];
}

__global__ void crtUnpremult(const float4* exchange, float4* rgb, int w, int h) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    float4 c = exchange[(size_t)(h - 1 - Y) * w + X];
    float a = fmaxf(c.w, 1e-6f);
    rgb[(size_t)Y * w + X] = make_float4(dclamp(c.x / a, 0.f, 1.f), dclamp(c.y / a, 0.f, 1.f), dclamp(c.z / a, 0.f, 1.f), c.w);
}

__global__ void crtCurve(const float4* rgb, float4* out, int w, int h, float k) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    float u = ((float)X - (w - 1) * 0.5f) / (w * 0.5f), v = ((float)Y - (h - 1) * 0.5f) / (h * 0.5f);
    float bend = 1.f + k * (u * u + v * v);
    float gy = (h - 1) * 0.5f + v * bend * h * 0.5f, gx = (w - 1) * 0.5f + u * bend * w * 0.5f;
    float y0 = floorf(gy), x0 = floorf(gx), wy = gy - y0, wx = gx - x0;
    float4 acc = make_float4(0, 0, 0, 0);
    const float ks[4] = {(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx};
    const int dy[4] = {0, 0, 1, 1}, dx[4] = {0, 1, 0, 1};
    for (int t = 0; t < 4; ++t) {
        float yy = y0 + dy[t], xx = x0 + dx[t];
        if (yy < 0 || yy > h - 1 || xx < 0 || xx > w - 1) continue;
        float4 s = crtAt(rgb, w, h, (int)xx, (int)yy);
        acc.x += s.x * ks[t]; acc.y += s.y * ks[t]; acc.z += s.z * ks[t];
    }
    float edge = fmaxf(fabsf(u * bend), fabsf(v * bend));
    float fade = dclamp((1.f - edge) * 40.f, 0.f, 1.f);
    out[(size_t)Y * w + X] = make_float4(acc.x * fade, acc.y * fade, acc.z * fade, 0.f);
}

__device__ inline int crtReflect(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i > n - 1) i = i < 0 ? -i : 2 * (n - 1) - i;
    return i;
}

__global__ void crtBlur(const float4* in, float4* out, int w, int h, const float* weights, int radius, int horizontal, int pad, int period) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    const int n = horizontal ? w : h, i = horizontal ? X : Y;
    float4 acc = make_float4(0, 0, 0, 0);
    for (int m = -radius; m <= radius; ++m) {
        int j = (i + pad - m) % period;
        if (j < 0) j += period;
        const int k = crtReflect(j - pad, n);
        float4 s = horizontal ? crtAt(in, w, h, k, Y) : crtAt(in, w, h, X, k);
        float g = weights[m + radius];
        acc.x += s.x * g; acc.y += s.y * g; acc.z += s.z * g;
    }
    out[(size_t)Y * w + X] = acc;
}

__global__ void crtFinish(const float4* curved, const float4* glow, const float4* rgb, float4* exchange, int w, int h,
                          float glowAmount, float pitch, float scan, float mask, int phosphor, int stripe, float amount) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    float4 c = crtAt(curved, w, h, X, Y);
    if (glow) {
        float4 g = crtAt(glow, w, h, X, Y);
        float s = 0.6f * glowAmount;
        c = make_float4(1.f - (1.f - c.x) * (1.f - g.x * s), 1.f - (1.f - c.y) * (1.f - g.y * s), 1.f - (1.f - c.z) * (1.f - g.z * s), 0.f);
    }
    float phase = fmodf((float)Y / pitch, 1.f);
    float width = 0.22f + 0.25f * (0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z);
    float d = (phase - 0.5f) / width;
    float beam = expf(-d * d);
    float k = 1.f - scan + scan * beam * 1.25f;
    c.x *= k; c.y *= k; c.z *= k;
    if (mask > 0.f && phosphor != 0) {
        float col = floorf((float)X / stripe);
        if (phosphor == 2) col = floorf(col + 1.5f * fmodf(floorf((float)Y / (2 * stripe)), 2.f));
        col = fmodf(col, 3.f);
        c.x *= 1.f - mask + mask * 3.f * (col == 0.f);
        c.y *= 1.f - mask + mask * 3.f * (col == 1.f);
        c.z *= 1.f - mask + mask * 3.f * (col == 2.f);
    }
    const float4 o = crtAt(rgb, w, h, X, Y);
    c = make_float4(dclamp(c.x, 0.f, 1.f), dclamp(c.y, 0.f, 1.f), dclamp(c.z, 0.f, 1.f), 0.f);
    c = make_float4(dclamp(o.x + (c.x - o.x) * amount, 0.f, 1.f), dclamp(o.y + (c.y - o.y) * amount, 0.f, 1.f),
                    dclamp(o.z + (c.z - o.z) * amount, 0.f, 1.f), 0.f);
    exchange[(size_t)(h - 1 - Y) * w + X] = make_float4(c.x * o.w, c.y * o.w, c.z * o.w, o.w);
}

__device__ inline float4 sampleClamp(const float4* img, int w, int h, float gy, float gx) {
    float y0 = floorf(gy), x0 = floorf(gx), wy = gy - y0, wx = gx - x0;
    int ya = (int)dclamp(y0, 0.f, h - 1.f), yb = (int)dclamp(y0 + 1.f, 0.f, h - 1.f);
    int xa = (int)dclamp(x0, 0.f, w - 1.f), xb = (int)dclamp(x0 + 1.f, 0.f, w - 1.f);
    float4 a = crtAt(img, w, h, xa, ya), b = crtAt(img, w, h, xb, ya), c = crtAt(img, w, h, xa, yb), d = crtAt(img, w, h, xb, yb);
    float k00 = (1 - wy) * (1 - wx), k01 = (1 - wy) * wx, k10 = wy * (1 - wx), k11 = wy * wx;
    return make_float4(a.x * k00 + b.x * k01 + c.x * k10 + d.x * k11, a.y * k00 + b.y * k01 + c.y * k10 + d.y * k11,
                       a.z * k00 + b.z * k01 + c.z * k10 + d.z * k11, 0.f);
}

__device__ inline float pyMod(float a, float b) {
    float r = fmodf(a, b);
    return r < 0 ? r + b : r;
}

__global__ void analogShift(const float4* src, float4* out, const float* rnd, int w, int h, int mode,
                            float bands, float t, float t3, float jitterPx, float circuitPx, float cablePx, float roll) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    float band = sinf((float)Y / h * bands * 6.28318530717958647692f + t3);
    float shift = (band * 0.3f + (rnd[Y] - 0.5f) * 0.7f) * jitterPx;
    if (mode == 1) shift += sinf((float)Y / h * 11.f + dluma(crtAt(src, w, h, X, Y)) * 8.f + t) * circuitPx;
    if (mode == 6) shift += (fabsf(band) > 0.96f ? 1.f : 0.f) * (rnd[h + Y] - 0.5f) * cablePx;
    out[(size_t)Y * w + X] = sampleClamp(src, w, h, pyMod((float)Y + roll, (float)h), (float)X + shift);
}

__global__ void analogChroma(const float4* src, float4* out, int w, int h, float chroma) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    out[(size_t)Y * w + X] = make_float4(sampleClamp(src, w, h, (float)Y, X + chroma).x, crtAt(src, w, h, X, Y).y,
                                         sampleClamp(src, w, h, (float)Y, X - chroma).z, 0.f);
}

__global__ void analogColor(const float4* src, float4* out, int w, int h) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    float4 c = crtAt(src, w, h, X, Y);
    float y = dluma(c);
    out[(size_t)Y * w + X] = make_float4(c.x - y, c.y - y, c.z - y, 0.f);
}

__global__ void analogBleed(const float4* src, const float4* color, const float4* blurred, float4* out, int w, int h, float bleed) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    float y = dluma(crtAt(src, w, h, X, Y));
    float4 c = crtAt(color, w, h, X, Y), b = crtAt(blurred, w, h, X, Y);
    out[(size_t)Y * w + X] = make_float4(y + c.x * (1 - bleed) + b.x * bleed, y + c.y * (1 - bleed) + b.y * bleed,
                                         y + c.z * (1 - bleed) + b.z * bleed, 0.f);
}

__global__ void analogMode(const float4* src, float4* out, int w, int h, int mode, float bands, float t2, float t3, float t4, float endersPx) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    float4 c = crtAt(src, w, h, X, Y);
    if (mode == 2) {
        float k = 0.25f + 0.75f * (pyMod((float)Y / h * bands + t2, 1.f) < 0.55f ? 1.f : 0.f);
        c.x *= k; c.y *= k; c.z *= k;
    } else if (mode == 3) {
        float phase = sinf((float)Y / h * bands + t4);
        c.x *= 0.8f + 0.3f * phase; c.z *= 0.8f - 0.3f * phase;
    } else if (mode == 4) {
        float band = sinf((float)Y / h * bands * 6.28318530717958647692f + t3);
        c = sampleClamp(src, w, h, (float)Y, X + floorf(band * 3.f) * endersPx);
        float phase = floorf(pyMod((float)Y / h * bands + t2, 3.f));
        c.x *= 0.6f + 1.2f * (phase == 0.f); c.y *= 0.6f + 1.2f * (phase == 1.f); c.z *= 0.6f + 1.2f * (phase == 2.f);
    }
    out[(size_t)Y * w + X] = c;
}

__global__ void analogNoise(float4* img, const float* noise, const float* rnd, const float* colored, int w, int h, int cw, int ch,
                            float noiseGain, float coloredGain, float dropoutLimit) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    float4 c = crtAt(img, w, h, X, Y);
    float n = noise[(size_t)Y * w + X] - 0.5f;
    c.x += n * noiseGain; c.y += n * noiseGain; c.z += n * noiseGain;
    if (colored) {
        float gy = ((float)Y + 0.5f) * ch / h - 0.5f, gx = ((float)X + 0.5f) * cw / w - 0.5f;
        float y0 = floorf(gy), x0 = floorf(gx), wy = gy - y0, wx = gx - x0;
        int ya = (int)dclamp(y0, 0.f, ch - 1.f), yb = (int)dclamp(y0 + 1.f, 0.f, ch - 1.f);
        int xa = (int)dclamp(x0, 0.f, cw - 1.f), xb = (int)dclamp(x0 + 1.f, 0.f, cw - 1.f);
        float v[3];
        for (int k = 0; k < 3; ++k) {
            const float* pl = colored + (size_t)k * cw * ch;
            v[k] = pl[ya * cw + xa] * (1 - wy) * (1 - wx) + pl[ya * cw + xb] * (1 - wy) * wx +
                   pl[yb * cw + xa] * wy * (1 - wx) + pl[yb * cw + xb] * wy * wx;
        }
        c.x += (v[0] - 0.5f) * coloredGain; c.y += (v[1] - 0.5f) * coloredGain; c.z += (v[2] - 0.5f) * coloredGain;
    }
    float drop = (rnd[2 * h + Y] > dropoutLimit ? 1.f : 0.f) * (0.4f + n * 0.6f);
    img[(size_t)Y * w + X] = make_float4(dclamp(c.x + drop, 0.f, 1.f), dclamp(c.y + drop, 0.f, 1.f), dclamp(c.z + drop, 0.f, 1.f), 0.f);
}

using cw::FftAxis;
using cw::fftGauss;

struct CrtGpu {
    DeviceBuffer images[8], weights[3], tables[3];
    cudaStream_t s = nullptr;
    int w = 0, h = 0;
    size_t bytes = 0;
    dim3 blk{16, 16}, grd;
    bool ok = true;
    static bool fit(DeviceBuffer& b, size_t n) { return b.bytes >= n || b.alloc(n); }
    float4* img(int i) { return static_cast<float4*>(images[i].ptr); }
    bool begin(const ImageView& exchange) {
        ok = true;
        w = exchange.bounds.w(); h = exchange.bounds.h();
        bytes = size_t(w) * h * sizeof(float4);
        grd = dim3((w + 15) / 16, (h + 15) / 16);
        if (cudaStreamCreate(&s) != cudaSuccess) { cudaGetLastError(); s = nullptr; return ok = false; }
        for (auto& b : images) ok = ok && fit(b, bytes);
        ok = ok && cudaMemcpyAsync(images[0].ptr, exchange.data, bytes, cudaMemcpyHostToDevice, s) == cudaSuccess;
        if (ok) crtUnpremult<<<grd, blk, 0, s>>>(img(0), img(1), w, h);
        return ok;
    }
    const float* table(int slot, const std::vector<float>& v) {
        ok = ok && fit(tables[slot], v.size() * sizeof(float)) &&
             cudaMemcpyAsync(tables[slot].ptr, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice, s) == cudaSuccess;
        return static_cast<const float*>(tables[slot].ptr);
    }
    void blur(int slot, int src, int tmp, int dst, float sigma) {
        const FftAxis ax = fftGauss(sigma, w), ay = fftGauss(sigma, h);
        std::vector<float> both(ax.weights);
        both.insert(both.end(), ay.weights.begin(), ay.weights.end());
        ok = ok && fit(weights[slot], both.size() * sizeof(float)) &&
             cudaMemcpyAsync(weights[slot].ptr, both.data(), both.size() * sizeof(float), cudaMemcpyHostToDevice, s) == cudaSuccess;
        if (!ok) return;
        const float* wx = static_cast<float*>(weights[slot].ptr);
        crtBlur<<<grd, blk, 0, s>>>(img(src), img(tmp), w, h, wx, ax.radius, 1, ax.pad, ax.period);
        crtBlur<<<grd, blk, 0, s>>>(img(tmp), img(dst), w, h, wx + ax.weights.size(), ay.radius, 0, ay.pad, ay.period);
        cudaStreamSynchronize(s);
    }
    void crt(const CrtParams& p, int src, float amount) {
        if (!ok) return;
        const int shortSide = std::min(w, h);
        const float sigma = 0.006f * shortSide;
        const bool glow = p.glow > 0 && sigma >= 0.3f;
        crtCurve<<<grd, blk, 0, s>>>(img(src), img(6), w, h, p.curvature * 0.12f);
        if (glow) blur(0, 6, 7, 5, sigma);
        const float pitch = fmaxf(3.f, h / p.lines);
        const int stripe = std::max(1, static_cast<int>(std::nearbyint(shortSide / 360.0)));
        if (ok) crtFinish<<<grd, blk, 0, s>>>(img(6), glow ? img(5) : nullptr, img(1), img(0), w, h,
                                              p.glow, pitch, p.scanlines, p.mask, p.phosphor, stripe, amount);
    }
    bool end(const ImageView& exchange) {
        ok = ok && cudaGetLastError() == cudaSuccess &&
             cudaMemcpyAsync(exchange.data, images[0].ptr, bytes, cudaMemcpyDeviceToHost, s) == cudaSuccess;
        if (s) {
            if (cudaStreamSynchronize(s) != cudaSuccess) ok = false;
            cudaStreamDestroy(s);
            s = nullptr;
        }
        if (!ok) cudaGetLastError();
        return ok;
    }
};

CrtGpu& crtGpu() {
    static CrtGpu g;
    return g;
}

bool usable(const ImageView& exchange) {
    return available() && exchange.bounds.w() > 0 && exchange.bounds.h() > 0 &&
           exchange.rowBytes == static_cast<ptrdiff_t>(exchange.bounds.w()) * 16;
}

}

bool cutwrightCrt(const CrtParams& p, const ImageView& exchange) {
    if (!usable(exchange)) return false;
    std::lock_guard<std::mutex> lock(deviceMutex());
    CrtGpu& g = crtGpu();
    if (g.begin(exchange)) g.crt(p, 1, 1.f);
    return g.end(exchange);
}

bool cutwrightAnalog(const AnalogParams& p, const ImageView& exchange) {
    if (!usable(exchange)) return false;
    const int w = exchange.bounds.w(), h = exchange.bounds.h(), shortSide = std::min(w, h);
    const double t = p.index / p.fps * p.speed;
    const auto seedAt = [&](double rate, long long mul) {
        return uint32_t(static_cast<long long>(p.seed) + static_cast<long long>(std::floor(t * rate)) * mul);
    };
    const auto rnd = numpyRandomFloat32(seedAt(24, 113), size_t(3) * h);
    const auto noise = numpyRandomFloat32(seedAt(30, 701), size_t(h) * w);
    const int ch = std::max(2, h / 12), cw = std::max(2, w / 16);
    const auto colored = p.mode == 6 ? numpyRandomFloat32(seedAt(12, 2029), size_t(3) * ch * cw) : std::vector<float>{};
    std::lock_guard<std::mutex> lock(deviceMutex());
    CrtGpu& g = crtGpu();
    if (!g.begin(exchange)) return g.end(exchange);
    const float* dRnd = g.table(0, rnd);
    const float* dNoise = g.table(1, noise);
    const float* dColored = p.mode == 6 ? g.table(2, colored) : nullptr;
    if (!g.ok) return g.end(exchange);
    const float tf = float(t);
    analogShift<<<g.grd, g.blk, 0, g.s>>>(g.img(1), g.img(2), dRnd, w, h, p.mode, p.bands, tf, float(t * 3),
                                          p.jitter * shortSide * 0.025f, shortSide * p.jitter * 0.06f,
                                          shortSide * p.dropout * 0.3f, float(p.roll * t * h * 0.1));
    analogChroma<<<g.grd, g.blk, 0, g.s>>>(g.img(2), g.img(3), w, h, p.chromatic * shortSide * 0.012f);
    int cur = 3;
    if (p.bleed > 0) {
        analogColor<<<g.grd, g.blk, 0, g.s>>>(g.img(3), g.img(4), w, h);
        g.blur(1, 4, 5, 6, std::max(0.5f, p.bleed * shortSide * 0.007f));
        if (g.ok) analogBleed<<<g.grd, g.blk, 0, g.s>>>(g.img(3), g.img(4), g.img(6), g.img(2), w, h, p.bleed);
        cur = 2;
    }
    if (p.mode >= 2 && p.mode <= 4) {
        const int next = cur == 2 ? 3 : 2;
        analogMode<<<g.grd, g.blk, 0, g.s>>>(g.img(cur), g.img(next), w, h, p.mode, p.bands, float(t * 2), float(t * 3), float(t * 4), shortSide * 0.012f);
        cur = next;
    } else if (p.mode == 5) {
        const int next = cur == 2 ? 3 : 2;
        g.blur(2, cur, 4, next, std::max(0.4f, shortSide * 0.0008f));
        cur = next;
    }
    if (g.ok) analogNoise<<<g.grd, g.blk, 0, g.s>>>(g.img(cur), dNoise, dRnd, dColored, w, h, cw, ch,
                                                    p.noise * (p.mode == 6 ? 0.5f : 0.18f), p.noise * 0.8f,
                                                    1.f - p.dropout * 0.035f);
    g.crt(p.crt, cur, p.amount);
    return g.end(exchange);
}

}
