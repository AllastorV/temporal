#include "gpu_common.cuh"
#include <vector>

namespace tp::gpu {

namespace {

__device__ inline int blurReflect(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i > n - 1) i = i < 0 ? -i : 2 * (n - 1) - i;
    return i;
}

__global__ void blurRows(const float* in, float* out, const float* wts, int w, int h, int radius, int pad, int period) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const float* row = in + size_t(y) * w;
    float s = 0;
    for (int m = -radius; m <= radius; ++m) {
        int j = (x + pad - m) % period;
        if (j < 0) j += period;
        s += row[blurReflect(j - pad, w)] * wts[m + radius];
    }
    out[size_t(y) * w + x] = s;
}

__global__ void blurCols(const float* in, float* out, const float* wts, int w, int h, int radius, int pad, int period) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float s = 0;
    for (int m = -radius; m <= radius; ++m) {
        int j = (y + pad - m) % period;
        if (j < 0) j += period;
        s += in[size_t(blurReflect(j - pad, h)) * w + x] * wts[m + radius];
    }
    out[size_t(y) * w + x] = s;
}

struct BlurGpu {
    float* a = nullptr;
    float* b = nullptr;
    float* wts = nullptr;
    size_t cap = 0, wcap = 0;
    cudaStream_t s = nullptr;
    bool reserve(size_t n, size_t taps) {
        if (!s && cudaStreamCreate(&s) != cudaSuccess) { s = nullptr; return false; }
        if (n > cap) {
            cudaFree(a);
            cudaFree(b);
            a = b = nullptr;
            cap = 0;
            if (cudaMalloc(&a, n * sizeof(float)) != cudaSuccess || cudaMalloc(&b, n * sizeof(float)) != cudaSuccess) return false;
            cap = n;
        }
        if (taps > wcap) {
            cudaFree(wts);
            wts = nullptr;
            wcap = 0;
            if (cudaMalloc(&wts, taps * sizeof(float)) != cudaSuccess) return false;
            wcap = taps;
        }
        return true;
    }
};

BlurGpu& blurGpu() {
    static BlurGpu g;
    return g;
}

}

bool planeBlur(const cw::Plane& in, float sigma, cw::Plane& out) {
    if (!available() || in.w < 1 || in.h < 1 || sigma < 0.3f) return false;
    const cw::FftAxis ax = cw::fftGauss(sigma, in.w), ay = cw::fftGauss(sigma, in.h);
    std::vector<float> weights(ax.weights);
    weights.insert(weights.end(), ay.weights.begin(), ay.weights.end());
    const size_t n = size_t(in.w) * in.h;
    std::lock_guard<std::mutex> lock(deviceMutex());
    BlurGpu& g = blurGpu();
    if (!g.reserve(n, weights.size())) { cudaGetLastError(); return false; }
    out = cw::Plane(in.h, in.w);
    const dim3 blk(32, 8), grd((in.w + 31) / 32, (in.h + 7) / 8);
    cudaMemcpyAsync(g.a, in.v.data(), n * sizeof(float), cudaMemcpyHostToDevice, g.s);
    cudaMemcpyAsync(g.wts, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice, g.s);
    blurRows<<<grd, blk, 0, g.s>>>(g.a, g.b, g.wts, in.w, in.h, ax.radius, ax.pad, ax.period);
    blurCols<<<grd, blk, 0, g.s>>>(g.b, g.a, g.wts + ax.weights.size(), in.w, in.h, ay.radius, ay.pad, ay.period);
    if (cudaGetLastError() != cudaSuccess) return false;
    cudaMemcpyAsync(out.v.data(), g.a, n * sizeof(float), cudaMemcpyDeviceToHost, g.s);
    const bool ok = cudaStreamSynchronize(g.s) == cudaSuccess;
    if (!ok) cudaGetLastError();
    return ok;
}

}
