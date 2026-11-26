#include "gpu_common.cuh"
#include "../core/cutwright_host.h"
#include <algorithm>
#include <cmath>

namespace tp::gpu {

namespace {

__device__ inline int inkReflect(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i > n - 1) i = i < 0 ? -i : 2 * (n - 1) - i;
    return i;
}

__device__ inline float inkSample(const float* p, int h, int w, float gy, float gx) {
    const float y0 = floorf(gy), x0 = floorf(gx), wy = gy - y0, wx = gx - x0;
    const int ya = (int)dclamp(y0, 0.f, h - 1.f), yb = (int)dclamp(y0 + 1.f, 0.f, h - 1.f);
    const int xa = (int)dclamp(x0, 0.f, w - 1.f), xb = (int)dclamp(x0 + 1.f, 0.f, w - 1.f);
    return p[(size_t)ya * w + xa] * ((1 - wy) * (1 - wx)) + p[(size_t)ya * w + xb] * ((1 - wy) * wx) +
           p[(size_t)yb * w + xa] * (wy * (1 - wx)) + p[(size_t)yb * w + xb] * (wy * wx);
}

__global__ void inkUnpremult(const float4* exchange, float* x, float* alpha, float* gray, int w, int h) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    const float4 c = exchange[(size_t)(h - 1 - Y) * w + X];
    const float a = fmaxf(c.w, 1e-6f);
    const size_t n = (size_t)w * h, i = (size_t)Y * w + X;
    const float r = dclamp(c.x / a, 0.f, 1.f), g = dclamp(c.y / a, 0.f, 1.f), b = dclamp(c.z / a, 0.f, 1.f);
    x[i] = r;
    x[n + i] = g;
    x[2 * n + i] = b;
    alpha[i] = c.w;
    gray[i] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

__global__ void inkSeen(const float* gray, const float* prevGray, const float* fy, const float* fx, double* sum, int w, int h) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    const size_t i = (size_t)Y * w + X;
    const float seen = inkSample(prevGray, h, w, (float)Y + fy[i], (float)X + fx[i]);
    atomicAdd(sum, double(fabsf(gray[i] - seen)));
}

__global__ void inkSteady(const float* x, const float* prevX, const float* fy, const float* fx, float* y, int w, int h) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    const size_t n = (size_t)w * h, i = (size_t)Y * w + X;
    float c[3], p[3];
    for (int k = 0; k < 3; ++k) {
        c[k] = x[k * n + i];
        p[k] = inkSample(prevX + k * n, h, w, (float)Y + fy[i], (float)X + fx[i]);
    }
    const float agree = dclamp(1.f - (fabsf(c[0] - p[0]) + fabsf(c[1] - p[1]) + fabsf(c[2] - p[2])) / 3.f / 0.12f, 0.f, 1.f) * 0.5f;
    for (int k = 0; k < 3; ++k) c[k] = c[k] + (p[k] - c[k]) * agree;
    y[i] = 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2];
}

__global__ void inkBlur(const float* in, float* out, int h, int w, const float* weights, int radius, int horizontal, int pad, int period) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int n = horizontal ? w : h, i = horizontal ? x : y;
    float acc = 0;
    for (int m = -radius; m <= radius; ++m) {
        int j = (i + pad - m) % period;
        if (j < 0) j += period;
        const int k = inkReflect(j - pad, n);
        acc += in[horizontal ? (size_t)y * w + k : (size_t)k * w + x] * weights[m + radius];
    }
    out[(size_t)y * w + x] = acc;
}

struct InkCompose {
    float eps, lineGain, hatching, period, waveScale;
    float c[2], s[2];
    float ink[3], paper[3];
};

__global__ void inkCompose(const float* b1, const float* b2, const float* tone, const float* alpha, float4* exchange, int w, int h, InkCompose a) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    const size_t i = (size_t)Y * w + X;
    const float dog = b1[i] - 0.98f * b2[i];
    float ink = dclamp(-tanhf(60.f * (dog - a.eps)), 0.f, 1.f) * a.lineGain;
    if (a.hatching > 0) {
        const float depths[2] = {0.45f, 0.22f};
        for (int k = 0; k < 2; ++k) {
            const float cover = dclamp((depths[k] - tone[i]) / depths[k] * (0.5f + a.hatching), 0.f, 0.9f);
            const float wave = cosf(a.waveScale * ((float)X * a.c[k] + (float)Y * a.s[k]) / a.period);
            const float stroke = dclamp((wave - (1.f - 2.f * cover)) * a.period / 4.f, 0.f, 1.f) * (cover > 0.02f ? 1.f : 0.f);
            ink = fmaxf(ink, stroke * 0.85f);
        }
    }
    const float al = alpha[i];
    float out[3];
    for (int k = 0; k < 3; ++k) out[k] = dclamp(a.paper[k] + (a.ink[k] - a.paper[k]) * ink, 0.f, 1.f) * al;
    exchange[(size_t)(h - 1 - Y) * w + X] = make_float4(out[0], out[1], out[2], al);
}

struct InkState {
    DeviceBuffer exchange, x, prevX, alpha, gray, prevGray, fy, fx, y, b1, b2, tone, tmp, weights, sum;
    cw::Plane hostGray, hostPrevGray;
    int index = -1, h = 0, w = 0;
};

InkState& inkState() {
    static InkState s;
    return s;
}

bool fitI(DeviceBuffer& b, size_t n) { return b.bytes >= n || b.alloc(n); }

}

bool cutwrightInk(const InkParams& p, const ImageView& exchange, bool reset, bool output) {
    const int w = exchange.bounds.w(), h = exchange.bounds.h();
    if (!available() || w < 2 || h < 2 || exchange.rowBytes != static_cast<ptrdiff_t>(w) * 16) return false;
    InkState& st = inkState();
    const size_t n = size_t(w) * h;
    if (reset) st.index = -1;
    const bool fresh = st.index != p.index - 1 || st.h != h || st.w != w;
    auto f = [](DeviceBuffer& b) { return static_cast<float*>(b.ptr); };
    const dim3 blk(16, 16), grd((w + 15) / 16, (h + 15) / 16);
    bool ok = true;
    {
        std::lock_guard<std::mutex> lock(deviceMutex());
        cudaStream_t s = nullptr;
        if (cudaStreamCreate(&s) != cudaSuccess) { cudaGetLastError(); return false; }
        ok = fitI(st.exchange, n * 16) && fitI(st.x, n * 12) && fitI(st.prevX, n * 12) && fitI(st.alpha, n * 4) &&
             fitI(st.gray, n * 4) && fitI(st.prevGray, n * 4) && fitI(st.fy, n * 4) && fitI(st.fx, n * 4) && fitI(st.y, n * 4) &&
             fitI(st.b1, n * 4) && fitI(st.b2, n * 4) && fitI(st.tone, n * 4) && fitI(st.tmp, n * 4) && fitI(st.sum, sizeof(double));
        ok = ok && cudaMemcpyAsync(st.exchange.ptr, exchange.data, n * 16, cudaMemcpyHostToDevice, s) == cudaSuccess;
        if (ok) inkUnpremult<<<grd, blk, 0, s>>>(static_cast<float4*>(st.exchange.ptr), f(st.x), f(st.alpha), f(st.gray), w, h);
        st.hostGray = cw::Plane(h, w);
        ok = ok && (!output || fresh || cudaMemcpyAsync(st.hostGray.v.data(), st.gray.ptr, n * 4, cudaMemcpyDeviceToHost, s) == cudaSuccess);
        if (cudaStreamSynchronize(s) != cudaSuccess) ok = false;
        cudaStreamDestroy(s);
    }
    if (!ok) { cudaGetLastError(); st.index = -1; return false; }
    cw::Plane fy, fx;
    bool havePrev = false;
    if (output && !fresh) {
        if (!cutwrightFlow(st.hostGray, st.hostPrevGray, fy, fx)) { st.index = -1; return false; }
        havePrev = true;
    }
    {
        std::lock_guard<std::mutex> lock(deviceMutex());
        cudaStream_t s = nullptr;
        if (cudaStreamCreate(&s) != cudaSuccess) { cudaGetLastError(); st.index = -1; return false; }
        if (output) {
            if (havePrev) {
                double sum = 0;
                ok = cudaMemcpyAsync(st.fy.ptr, fy.v.data(), n * 4, cudaMemcpyHostToDevice, s) == cudaSuccess &&
                     cudaMemcpyAsync(st.fx.ptr, fx.v.data(), n * 4, cudaMemcpyHostToDevice, s) == cudaSuccess &&
                     cudaMemsetAsync(st.sum.ptr, 0, sizeof(double), s) == cudaSuccess;
                if (ok) inkSeen<<<grd, blk, 0, s>>>(f(st.gray), f(st.prevGray), f(st.fy), f(st.fx), static_cast<double*>(st.sum.ptr), w, h);
                ok = ok && cudaMemcpyAsync(&sum, st.sum.ptr, sizeof(double), cudaMemcpyDeviceToHost, s) == cudaSuccess &&
                     cudaStreamSynchronize(s) == cudaSuccess;
                if (ok && float(sum / double(n)) > 0.1f) havePrev = false;
            }
            if (ok) {
                if (havePrev) inkSteady<<<grd, blk, 0, s>>>(f(st.x), f(st.prevX), f(st.fy), f(st.fx), f(st.y), w, h);
                else ok = cudaMemcpyAsync(st.y.ptr, st.gray.ptr, n * 4, cudaMemcpyDeviceToDevice, s) == cudaSuccess;
            }
            const int shortSide = std::min(w, h);
            const auto blur = [&](DeviceBuffer& in, DeviceBuffer& out, float sigma) {
                if (!ok) return;
                if (sigma < 0.3f) {
                    ok = cudaMemcpyAsync(out.ptr, in.ptr, n * 4, cudaMemcpyDeviceToDevice, s) == cudaSuccess;
                    return;
                }
                const cw::FftAxis ax = cw::fftGauss(sigma, w), ay = cw::fftGauss(sigma, h);
                std::vector<float> both(ax.weights);
                both.insert(both.end(), ay.weights.begin(), ay.weights.end());
                ok = cudaStreamSynchronize(s) == cudaSuccess && fitI(st.weights, both.size() * 4) &&
                     cudaMemcpyAsync(st.weights.ptr, both.data(), both.size() * 4, cudaMemcpyHostToDevice, s) == cudaSuccess;
                if (!ok) return;
                inkBlur<<<grd, blk, 0, s>>>(f(in), f(st.tmp), h, w, f(st.weights), ax.radius, 1, ax.pad, ax.period);
                inkBlur<<<grd, blk, 0, s>>>(f(st.tmp), f(out), h, w, f(st.weights) + ax.weights.size(), ay.radius, 0, ay.pad, ay.period);
            };
            const double sigma = std::max(0.6, double(p.width) * shortSide);
            blur(st.y, st.b1, float(sigma));
            blur(st.y, st.b2, float(1.6 * sigma));
            if (p.hatching > 0) blur(st.y, st.tone, float(std::max(1.0, 0.004 * shortSide)));
            if (ok) {
                InkCompose a{};
                a.eps = float(-0.004 * (1.5 - double(p.lines)));
                a.lineGain = float(std::min(1.0, 0.4 + double(p.lines)));
                a.hatching = p.hatching;
                a.period = float(std::max(3.0, double(p.spacing) * shortSide));
                a.waveScale = float(2 * 3.14159265358979323846);
                for (int k = 0; k < 2; ++k) {
                    const double angle = double(p.angle) * 3.14159265358979323846 / 180.0 + (k ? 3.14159265358979323846 / 2 : 0.0);
                    a.c[k] = float(std::cos(angle));
                    a.s[k] = float(std::sin(angle));
                }
                for (int k = 0; k < 3; ++k) { a.ink[k] = p.ink[k]; a.paper[k] = p.paper[k]; }
                inkCompose<<<grd, blk, 0, s>>>(f(st.b1), f(st.b2), f(st.tone), f(st.alpha), static_cast<float4*>(st.exchange.ptr), w, h, a);
                ok = cudaGetLastError() == cudaSuccess &&
                     cudaMemcpyAsync(exchange.data, st.exchange.ptr, n * 16, cudaMemcpyDeviceToHost, s) == cudaSuccess;
            }
        }
        ok = ok && cudaMemcpyAsync(st.prevX.ptr, st.x.ptr, n * 12, cudaMemcpyDeviceToDevice, s) == cudaSuccess &&
             cudaMemcpyAsync(st.prevGray.ptr, st.gray.ptr, n * 4, cudaMemcpyDeviceToDevice, s) == cudaSuccess;
        if (!output || fresh) ok = ok && cudaMemcpyAsync(st.hostGray.v.data(), st.gray.ptr, n * 4, cudaMemcpyDeviceToHost, s) == cudaSuccess;
        if (cudaStreamSynchronize(s) != cudaSuccess) ok = false;
        cudaStreamDestroy(s);
    }
    if (!ok) { cudaGetLastError(); st.index = -1; return false; }
    st.hostPrevGray = st.hostGray;
    st.index = p.index;
    st.h = h;
    st.w = w;
    return true;
}

}
