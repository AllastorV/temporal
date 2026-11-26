#include "gpu_common.cuh"
#include "../core/cutwright_host.h"
#include "../core/numpy_random.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace tp::gpu {

namespace {

__device__ inline uint32_t vhsHash(uint32_t v) {
    const uint32_t state = v * 747796405u + 2891336453u;
    const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ inline float vhsUniform(uint32_t key, uint32_t i) {
    return (float(vhsHash(key ^ vhsHash(i)) >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

__device__ inline float vhsNormal(uint32_t key, uint32_t i) {
    const float u1 = vhsUniform(key, 2 * i), u2 = vhsUniform(key ^ 0x9e3779b9u, 2 * i + 1);
    return sqrtf(-2.f * logf(u1)) * cosf(6.28318530717958647692f * u2);
}

__device__ inline int vhsReflect(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i > n - 1) i = i < 0 ? -i : 2 * (n - 1) - i;
    return i;
}

__global__ void vhsUnpremult(const float4* exchange, float* rgb, float* alpha, int w, int h) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    const float4 c = exchange[(size_t)(h - 1 - Y) * w + X];
    const float a = fmaxf(c.w, 1e-6f);
    const size_t n = (size_t)w * h, i = (size_t)Y * w + X;
    rgb[i] = dclamp(c.x / a, 0.f, 1.f);
    rgb[n + i] = dclamp(c.y / a, 0.f, 1.f);
    rgb[2 * n + i] = dclamp(c.z / a, 0.f, 1.f);
    alpha[i] = c.w;
}

__global__ void vhsDisplace(const float* in, float* out, const float* dx, float dy, int w, int h) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    const float gy = (float)Y - dy, gx = (float)X - dx[Y];
    const float y0 = floorf(gy), x0 = floorf(gx), fy = gy - y0, fx = gx - x0;
    const size_t n = (size_t)w * h;
    const float ks[4] = {(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx};
    const int oy[4] = {0, 0, 1, 1}, ox[4] = {0, 1, 0, 1};
    for (int c = 0; c < 3; ++c) {
        float acc = 0;
        for (int k = 0; k < 4; ++k) {
            const int yy = min(max((int)y0 + oy[k], 0), h - 1), xx = min(max((int)x0 + ox[k], 0), w - 1);
            acc += in[c * n + (size_t)yy * w + xx] * ks[k];
        }
        out[c * n + (size_t)Y * w + X] = acc;
    }
}

__global__ void vhsSplit(const float* rgb, float* y, float* uv, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float r = rgb[i], g = rgb[n + i], b = rgb[2 * n + i];
    const float l = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    y[i] = l;
    uv[i] = b - l;
    uv[n + i] = r - l;
}

__global__ void vhsBlur(const float* in, float* out, int planes, int h, int w, const float* weights, int radius, int horizontal, int pad, int period) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int n = horizontal ? w : h, i = horizontal ? x : y;
    const size_t plane = (size_t)h * w;
    for (int c = 0; c < planes; ++c) {
        float acc = 0;
        for (int m = -radius; m <= radius; ++m) {
            int j = (i + pad - m) % period;
            if (j < 0) j += period;
            const int k = vhsReflect(j - pad, n);
            acc += in[c * plane + (horizontal ? (size_t)y * w + k : (size_t)k * w + x)] * weights[m + radius];
        }
        out[c * plane + (size_t)y * w + x] = acc;
    }
}

__global__ void vhsSharpen(float* y, const float* soft, const float* softBlur, float sharpen, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = soft[i] + 1.5f * sharpen * (soft[i] - softBlur[i]);
    y[i] = 0.05f + 0.9f * v;
}

__global__ void vhsShiftChroma(const float* in, float* out, float shift, float saturation, int w, int h) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    const float gx = (float)X - shift;
    const float x0 = floorf(gx), fx = gx - x0;
    const size_t n = (size_t)w * h;
    for (int c = 0; c < 2; ++c) {
        float acc = 0;
        const int xa = (int)x0, xb = xa + 1;
        if (xa >= 0 && xa < w) acc += in[c * n + (size_t)Y * w + xa] * (1 - fx);
        if (xb >= 0 && xb < w) acc += in[c * n + (size_t)Y * w + xb] * fx;
        out[c * n + (size_t)Y * w + X] = acc * saturation;
    }
}

__global__ void vhsNoisePlanes(float* grain, float* dots, float* streaks, uint32_t key, float density, int bandStart, float bandDensity, int n, int w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    grain[i] = vhsNormal(key ^ 0x1111u, i);
    const int row = i / w;
    dots[i] = vhsUniform(key ^ 0x2222u, i) > 1.f - density ? 1.f : 0.f;
    dots[n + i] = row >= bandStart && vhsUniform(key ^ 0x3333u, i) > 1.f - bandDensity ? 1.f : 0.f;
    streaks[i] = vhsNormal(key ^ 0x4444u, i);
    streaks[n + i] = vhsNormal(key ^ 0x5555u, i);
}

__global__ void vhsCompose(const float* y, const float* uv, const float* rowNoise, const float* grain, const float* dash, const float* bandDash,
                           const float* streaks, const float* alpha, float4* exchange, float noise, float dashGain, float bandGain,
                           float bandLift, int bandStart, int w, int h) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    const size_t n = (size_t)w * h, i = (size_t)Y * w + X;
    float l = y[i];
    float u = uv[i], v = uv[n + i];
    if (noise > 0) {
        l = l + rowNoise[Y] + grain[i] * 0.1f * noise;
        l = l + dash[i] * dashGain;
        u = u + streaks[i] * 0.3f * noise;
        v = v + streaks[n + i] * 0.3f * noise;
    }
    if (Y >= bandStart) l = l + (bandDash[n + i] * bandGain + bandLift);
    const float r = v + l, b = u + l;
    const float g = (l - 0.2126f * r - 0.0722f * b) / 0.7152f;
    const float a = alpha[i];
    exchange[(size_t)(h - 1 - Y) * w + X] = make_float4(dclamp(r, 0.f, 1.f) * a, dclamp(g, 0.f, 1.f) * a, dclamp(b, 0.f, 1.f) * a, a);
}

struct VhsGpu {
    DeviceBuffer exchange, rgb, rgb2, alpha, y, soft, tmp, tmp2, uv, uv2, grain, dots, streaks, rows, dx, weights;
};

VhsGpu& vhsGpu() {
    static VhsGpu g;
    return g;
}

bool fitV(DeviceBuffer& b, size_t n) { return b.bytes >= n || b.alloc(n); }

int roundHalfEven(double v) { return static_cast<int>(std::nearbyint(v)); }

}

bool cutwrightVhs(const VhsParams& p, const ImageView& exchange) {
    const int w = exchange.bounds.w(), h = exchange.bounds.h();
    if (!available() || w < 2 || h < 2 || exchange.rowBytes != static_cast<ptrdiff_t>(w) * 16) return false;
    const double s = h / 480.0, t = p.index / p.fps, pi = 3.14159265358979323846;
    NumpyPcg64 rng(std::vector<uint32_t>{uint32_t(p.seed), uint32_t(p.index)});
    const double phase = rng.nextDouble();
    std::vector<double> wobble(h);
    for (int Y = 0; Y < h; ++Y) wobble[Y] = p.jitter * 0.8 * s * std::sin(2 * pi * (double(Y) / h * 3 + 0.7 * t + phase));
    const double dy = p.jitter * s * rng.uniform(-1, 1);
    std::vector<float> rows(h, 0.f);
    int band = 0;
    if (p.tracking > 0) {
        band = std::max(2, roundHalfEven(h * (0.015 + 0.045 * p.tracking)));
        const double reach = p.tracking * w * rng.uniform(0.025, 0.05) * (rng.boundedLemire(1) ? 1.0 : -1.0);
        for (int i = 0; i < band; ++i) rows[h - band + i] = float(reach * std::pow((i + 1.0) / band, 1.5));
    }
    std::vector<float> dx(h);
    bool moved = dy != 0;
    for (int Y = 0; Y < h; ++Y) {
        dx[Y] = float(wobble[Y] + double(rows[Y]));
        moved = moved || dx[Y] != 0;
    }
    std::mt19937 noiseRng(uint32_t(p.seed) * 2654435761u ^ uint32_t(p.index) * 40503u);
    std::normal_distribution<float> normal;
    std::vector<float> rowNoise(h);
    for (float& v : rowNoise) v = normal(noiseRng) * 0.03f * p.noise;
    const uint32_t key = uint32_t(p.seed) * 0x9e3779b1u ^ uint32_t(p.index) * 0x85ebca77u;

    std::lock_guard<std::mutex> lock(deviceMutex());
    VhsGpu& g = vhsGpu();
    const size_t n = size_t(w) * h;
    cudaStream_t st = nullptr;
    if (cudaStreamCreate(&st) != cudaSuccess) { cudaGetLastError(); return false; }
    bool ok = fitV(g.exchange, n * 16) && fitV(g.rgb, n * 12) && fitV(g.rgb2, n * 12) && fitV(g.alpha, n * 4) &&
              fitV(g.y, n * 4) && fitV(g.soft, n * 4) && fitV(g.tmp, n * 8) && fitV(g.tmp2, n * 8) && fitV(g.uv, n * 8) &&
              fitV(g.uv2, n * 8) && fitV(g.grain, n * 4) && fitV(g.dots, n * 8) && fitV(g.streaks, n * 8) &&
              fitV(g.rows, h * 4) && fitV(g.dx, h * 4);
    ok = ok && cudaMemcpyAsync(g.exchange.ptr, exchange.data, n * 16, cudaMemcpyHostToDevice, st) == cudaSuccess &&
         cudaMemcpyAsync(g.rows.ptr, rowNoise.data(), h * 4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
         cudaMemcpyAsync(g.dx.ptr, dx.data(), h * 4, cudaMemcpyHostToDevice, st) == cudaSuccess;
    auto f = [](DeviceBuffer& b) { return static_cast<float*>(b.ptr); };
    const dim3 blk(16, 16), grd((w + 15) / 16, (h + 15) / 16);
    const int lin = int((n + 255) / 256);
    const auto blur1d = [&](const float* in, float* out, int planes, float sigma, bool horizontal) {
        if (!ok) return;
        if (sigma < 0.3f) {
            ok = cudaMemcpyAsync(out, in, size_t(planes) * n * 4, cudaMemcpyDeviceToDevice, st) == cudaSuccess;
            return;
        }
        const cw::FftAxis a = cw::fftGauss(sigma, horizontal ? w : h);
        ok = cudaStreamSynchronize(st) == cudaSuccess && fitV(g.weights, a.weights.size() * 4) &&
             cudaMemcpyAsync(g.weights.ptr, a.weights.data(), a.weights.size() * 4, cudaMemcpyHostToDevice, st) == cudaSuccess;
        if (ok) vhsBlur<<<grd, blk, 0, st>>>(in, out, planes, h, w, f(g.weights), a.radius, horizontal ? 1 : 0, a.pad, a.period);
    };
    if (ok) {
        vhsUnpremult<<<grd, blk, 0, st>>>(static_cast<float4*>(g.exchange.ptr), f(g.rgb), f(g.alpha), w, h);
        const float* rgb = f(g.rgb);
        if (moved) {
            vhsDisplace<<<grd, blk, 0, st>>>(f(g.rgb), f(g.rgb2), f(g.dx), float(dy), w, h);
            rgb = f(g.rgb2);
        }
        vhsSplit<<<lin, 256, 0, st>>>(rgb, f(g.y), f(g.uv), int(n));
        blur1d(f(g.y), f(g.tmp), 1, float(std::max(0.3, 0.0028 * w)), true);
        blur1d(f(g.tmp), f(g.tmp2), 1, float(0.6 * s), true);
        blur1d(f(g.tmp2), f(g.soft), 1, float(0.6 * s), false);
        blur1d(f(g.soft), f(g.tmp), 1, float(std::max(0.6, 0.006 * w)), true);
        if (ok) vhsSharpen<<<lin, 256, 0, st>>>(f(g.y), f(g.soft), f(g.tmp), p.sharpen, int(n));
        blur1d(f(g.uv), f(g.uv2), 2, float((0.006 + 0.022 * p.chromaBlur) * w), true);
        if (ok) vhsShiftChroma<<<grd, blk, 0, st>>>(f(g.uv2), f(g.uv), float(0.012 * w * p.chromaBlur), p.saturation, w, h);
        const float density = float(12 * p.noise / (480.0 * w)), bandDensity = float(0.012 * p.tracking);
        if (ok) vhsNoisePlanes<<<lin, 256, 0, st>>>(f(g.grain), f(g.dots), f(g.streaks), key, density, h - band, bandDensity, int(n), w);
        blur1d(f(g.grain), f(g.soft), 1, float(1.2 * s), true);
        blur1d(f(g.dots), f(g.tmp2), 1, float(4 * s), true);
        blur1d(f(g.dots) + n, f(g.tmp2) + n, 1, float(6 * s), true);
        blur1d(f(g.streaks), f(g.tmp), 2, float(6 * s), true);
        if (ok) {
            vhsCompose<<<grd, blk, 0, st>>>(f(g.y), f(g.uv), f(g.rows), f(g.soft), f(g.tmp2), f(g.tmp2), f(g.tmp), f(g.alpha),
                                            static_cast<float4*>(g.exchange.ptr), p.noise, float(2.5 * 4 * s), float(2.5 * 6 * s),
                                            float(0.05 * p.tracking), band ? h - band : h, w, h);
            ok = cudaGetLastError() == cudaSuccess &&
                 cudaMemcpyAsync(exchange.data, g.exchange.ptr, n * 16, cudaMemcpyDeviceToHost, st) == cudaSuccess;
        }
    }
    if (cudaStreamSynchronize(st) != cudaSuccess) ok = false;
    cudaStreamDestroy(st);
    if (!ok) cudaGetLastError();
    return ok;
}

}
