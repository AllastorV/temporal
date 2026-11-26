#include "gpu_common.cuh"
#include "../core/cutwright_host.h"
#include <algorithm>
#include <cmath>

namespace tp::gpu {

namespace {

struct ShiftArg {
    float fx, fy;
    int sx, sy, integral;
};

ShiftArg makeShift(double ox, double oy) {
    ShiftArg a{float(ox), float(oy), 0, 0, 0};
    const float rx = std::nearbyint(a.fx), ry = std::nearbyint(a.fy);
    if (std::fabs(a.fx - rx) < 1e-6f && std::fabs(a.fy - ry) < 1e-6f && std::fabs(rx) < 1e6f && std::fabs(ry) < 1e6f) {
        a.integral = 1;
        a.sx = int(rx);
        a.sy = int(ry);
    }
    return a;
}

__device__ inline float zeroSample(const float* p, int h, int w, float gy, float gx) {
    const float y0 = floorf(gy), x0 = floorf(gx), wy = gy - y0, wx = gx - x0;
    const float ks[4] = {(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx};
    const float dy[4] = {0, 0, 1, 1}, dx[4] = {0, 1, 0, 1};
    float out = 0;
    for (int k = 0; k < 4; ++k) {
        const float yy = y0 + dy[k], xx = x0 + dx[k];
        if (yy < 0 || yy > h - 1 || xx < 0 || xx > w - 1) continue;
        out = out + p[(size_t)yy * w + (int)xx] * ks[k];
    }
    return out;
}

__device__ inline float shiftAt(const float* p, int h, int w, int y, int x, ShiftArg s) {
    if (!s.integral) return zeroSample(p, h, w, (float)y - s.fy, (float)x - s.fx);
    const int yy = y - s.sy, xx = x - s.sx;
    return yy < 0 || yy >= h || xx < 0 || xx >= w ? 0.f : p[(size_t)yy * w + xx];
}

__device__ inline int starReflect(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i > n - 1) i = i < 0 ? -i : 2 * (n - 1) - i;
    return i;
}

__global__ void starBlur(const float* in, float* out, int h, int w, const float* weights, int radius, int horizontal, int pad, int period) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int n = horizontal ? w : h, i = horizontal ? x : y;
    float acc = 0;
    for (int m = -radius; m <= radius; ++m) {
        int j = (i + pad - m) % period;
        if (j < 0) j += period;
        const int k = starReflect(j - pad, n);
        acc += in[horizontal ? (size_t)y * w + k : (size_t)k * w + x] * weights[m + radius];
    }
    out[(size_t)y * w + x] = acc;
}

__global__ void starSrc(const float* light, const float* near, const float* wide, float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float k = dclamp((near[i] / (wide[i] + 1e-4f) - 2.f) / 6.f, 0.f, 1.f);
    for (int c = 0; c < 3; ++c) src[(size_t)c * n + i] = light[(size_t)c * n + i] * k;
}

__global__ void starSpan(const float* in, float* out, int h, int w, ShiftArg a, ShiftArg b) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t n = (size_t)h * w, i = (size_t)y * w + x;
    for (int c = 0; c < 3; ++c) {
        const float* p = in + c * n;
        out[c * n + i] = p[i] + shiftAt(p, h, w, y, x, a) + shiftAt(p, h, w, y, x, b);
    }
}

__global__ void starRay(const float* in, float* out, int h, int w, ShiftArg m, float k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t n = (size_t)h * w, i = (size_t)y * w + x;
    for (int c = 0; c < 3; ++c) {
        const float* p = in + c * n;
        out[c * n + i] = p[i] + k * shiftAt(p, h, w, y, x, m);
    }
}

__global__ void starLine(const float* energy, const float* near, float* line, int h, int w, ShiftArg a, ShiftArg b) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t n = (size_t)h * w, i = (size_t)y * w + x;
    const float top = near[i] + 1e-5f;
    const bool ridge = top >= shiftAt(near, h, w, y, x, a) && top >= shiftAt(near, h, w, y, x, b) && near[i] > 1e-3f;
    const float pk = fmaxf(fmaxf(energy[i], energy[n + i]), energy[2 * n + i]);
    const float norm = sqrtf(fmaxf(pk, 1e-3f));
    for (int c = 0; c < 3; ++c) line[c * n + i] = energy[c * n + i] * (ridge ? 1.f : 0.f) / norm;
}

__global__ void starAcc(float* acc, const float* forward, const float* backward, const float* line, int n3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n3) return;
    acc[i] = acc[i] + forward[i] + backward[i] - 2 * line[i];
}

struct StarGpu {
    DeviceBuffer amount, near, wide, tmp, light, src, a, b, line, fwd, bwd, acc, weights;
};

StarGpu& starGpu() {
    static StarGpu g;
    return g;
}

bool fitS(DeviceBuffer& b, size_t n) { return b.bytes >= n || b.alloc(n); }

}

bool cutwrightStar(const cw::Plane& amount, const std::vector<cw::Plane>& light, double s1, double length, std::vector<cw::Plane>& acc) {
    if (!available() || light.size() != 3) return false;
    const int h = amount.h, w = amount.w;
    const size_t n = size_t(h) * w;
    std::lock_guard<std::mutex> lock(deviceMutex());
    StarGpu& g = starGpu();
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) { cudaGetLastError(); return false; }
    bool ok = fitS(g.amount, n * 4) && fitS(g.near, n * 4) && fitS(g.wide, n * 4) && fitS(g.tmp, n * 4) &&
              fitS(g.light, n * 12) && fitS(g.src, n * 12) && fitS(g.a, n * 12) && fitS(g.b, n * 12) &&
              fitS(g.line, n * 12) && fitS(g.fwd, n * 12) && fitS(g.bwd, n * 12) && fitS(g.acc, n * 12);
    ok = ok && cudaMemcpyAsync(g.amount.ptr, amount.v.data(), n * 4, cudaMemcpyHostToDevice, s) == cudaSuccess;
    for (int c = 0; ok && c < 3; ++c)
        ok = cudaMemcpyAsync(static_cast<float*>(g.light.ptr) + c * n, light[c].v.data(), n * 4, cudaMemcpyHostToDevice, s) == cudaSuccess;
    auto f = [](DeviceBuffer& b) { return static_cast<float*>(b.ptr); };
    const dim3 blk(16, 16), grd((w + 15) / 16, (h + 15) / 16);
    const auto blur = [&](float sigma, DeviceBuffer& out) {
        if (sigma < 0.3f) {
            ok = ok && cudaMemcpyAsync(out.ptr, g.amount.ptr, n * 4, cudaMemcpyDeviceToDevice, s) == cudaSuccess;
            return;
        }
        const cw::FftAxis ax = cw::fftGauss(sigma, w), ay = cw::fftGauss(sigma, h);
        std::vector<float> both(ax.weights);
        both.insert(both.end(), ay.weights.begin(), ay.weights.end());
        ok = ok && cudaStreamSynchronize(s) == cudaSuccess && fitS(g.weights, both.size() * 4) &&
             cudaMemcpyAsync(g.weights.ptr, both.data(), both.size() * 4, cudaMemcpyHostToDevice, s) == cudaSuccess;
        if (!ok) return;
        starBlur<<<grd, blk, 0, s>>>(f(g.amount), f(g.tmp), h, w, f(g.weights), ax.radius, 1, ax.pad, ax.period);
        starBlur<<<grd, blk, 0, s>>>(f(g.tmp), f(out), h, w, f(g.weights) + ax.weights.size(), ay.radius, 0, ay.pad, ay.period);
        ok = cudaStreamSynchronize(s) == cudaSuccess;
    };
    blur(float(s1), g.near);
    blur(float(4 * s1), g.wide);
    if (ok) {
        const int n1 = int(n), n3 = int(3 * n);
        starSrc<<<(n1 + 255) / 256, 256, 0, s>>>(f(g.light), f(g.near), f(g.wide), f(g.src), n1);
        ok = cudaMemsetAsync(g.acc.ptr, 0, n * 12, s) == cudaSuccess;
        const double pi = 3.14159265358979323846;
        for (int j = 0; j < 2 && ok; ++j) {
            const double theta = (0.0 + j * 180.0 / 2) * pi / 180.0;
            const double dx = std::cos(theta), dy = std::sin(theta);
            ok = cudaMemcpyAsync(g.a.ptr, g.src.ptr, n * 12, cudaMemcpyDeviceToDevice, s) == cudaSuccess;
            DeviceBuffer* cur = &g.a;
            DeviceBuffer* nxt = &g.b;
            for (long long step = 1; step < 8 * s1; step *= 3) {
                starSpan<<<grd, blk, 0, s>>>(f(*cur), f(*nxt), h, w, makeShift(-dy * step, dx * step), makeShift(dy * step, -dx * step));
                std::swap(cur, nxt);
            }
            starLine<<<grd, blk, 0, s>>>(f(*cur), f(g.near), f(g.line), h, w, makeShift(-dy, dx), makeShift(dy, -dx));
            const double r = std::exp(-3 / std::max(length, 1.0));
            for (int dir = 0; dir < 2 && ok; ++dir) {
                const double sx = dir == 0 ? dx : -dx, sy = dir == 0 ? dy : -dy;
                DeviceBuffer& target = dir == 0 ? g.fwd : g.bwd;
                ok = cudaMemcpyAsync(target.ptr, g.line.ptr, n * 12, cudaMemcpyDeviceToDevice, s) == cudaSuccess;
                DeviceBuffer* rc = &target;
                DeviceBuffer* rn = &g.a;
                for (long long step = 1; step < 2.5 * length; step *= 2) {
                    starRay<<<grd, blk, 0, s>>>(f(*rc), f(*rn), h, w, makeShift(sx * step, sy * step), float(std::pow(r, double(step))));
                    std::swap(rc, rn);
                }
                if (rc != &target) ok = cudaMemcpyAsync(target.ptr, rc->ptr, n * 12, cudaMemcpyDeviceToDevice, s) == cudaSuccess;
            }
            if (ok) starAcc<<<(n3 + 255) / 256, 256, 0, s>>>(f(g.acc), f(g.fwd), f(g.bwd), f(g.line), n3);
        }
    }
    acc.assign(3, cw::Plane(h, w));
    for (int c = 0; ok && c < 3; ++c)
        ok = cudaMemcpyAsync(acc[c].v.data(), static_cast<float*>(g.acc.ptr) + c * n, n * 4, cudaMemcpyDeviceToHost, s) == cudaSuccess;
    if (cudaStreamSynchronize(s) != cudaSuccess) ok = false;
    cudaStreamDestroy(s);
    if (!ok || cudaGetLastError() != cudaSuccess) return false;
    return true;
}

}
