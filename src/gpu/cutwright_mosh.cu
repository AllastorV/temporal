#include "gpu_common.cuh"
#include "../core/cutwright_host.h"
#include "../core/numpy_random.h"
#include <algorithm>
#include <cmath>

namespace tp::gpu {

namespace {

struct Field {
    const float* y;
    const float* x;
    int h, w;
};

__device__ inline float fieldAt(const float* f, int w, int y, int x) { return f[(size_t)y * w + x]; }

__device__ inline float fieldSample(const float* f, int h, int w, float gy, float gx) {
    float y0 = floorf(gy), x0 = floorf(gx), wy = gy - y0, wx = gx - x0;
    int ya = (int)dclamp(y0, 0.f, h - 1.f), yb = (int)dclamp(y0 + 1.f, 0.f, h - 1.f);
    int xa = (int)dclamp(x0, 0.f, w - 1.f), xb = (int)dclamp(x0 + 1.f, 0.f, w - 1.f);
    return fieldAt(f, w, ya, xa) * ((1 - wy) * (1 - wx)) + fieldAt(f, w, ya, xb) * ((1 - wy) * wx) +
           fieldAt(f, w, yb, xa) * (wy * (1 - wx)) + fieldAt(f, w, yb, xb) * (wy * wx);
}

__device__ inline float fieldUp(const float* f, int fh, int fw, int H, int W, int Y, int X) {
    if (fh == H && fw == W) return fieldAt(f, fw, Y, X);
    return fieldSample(f, fh, fw, ((float)Y + 0.5f) * fh / H - 0.5f, ((float)X + 0.5f) * fw / W - 0.5f);
}

__device__ inline float4 imgSample(const float4* img, int w, int h, float gy, float gx) {
    float y0 = floorf(gy), x0 = floorf(gx), wy = gy - y0, wx = gx - x0;
    int ya = (int)dclamp(y0, 0.f, h - 1.f), yb = (int)dclamp(y0 + 1.f, 0.f, h - 1.f);
    int xa = (int)dclamp(x0, 0.f, w - 1.f), xb = (int)dclamp(x0 + 1.f, 0.f, w - 1.f);
    float4 a = img[(size_t)ya * w + xa], b = img[(size_t)ya * w + xb], c = img[(size_t)yb * w + xa], d = img[(size_t)yb * w + xb];
    float k00 = (1 - wy) * (1 - wx), k01 = (1 - wy) * wx, k10 = wy * (1 - wx), k11 = wy * wx;
    return make_float4(a.x * k00 + b.x * k01 + c.x * k10 + d.x * k11, a.y * k00 + b.y * k01 + c.y * k10 + d.y * k11,
                       a.z * k00 + b.z * k01 + c.z * k10 + d.z * k11, 0.f);
}

__global__ void moshUnpremult(const float4* exchange, float4* rgb, int w, int h) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    float4 c = exchange[(size_t)(h - 1 - Y) * w + X];
    float a = fmaxf(c.w, 1e-6f);
    rgb[(size_t)Y * w + X] = make_float4(dclamp(c.x / a, 0.f, 1.f), dclamp(c.y / a, 0.f, 1.f), dclamp(c.z / a, 0.f, 1.f), c.w);
}

__global__ void moshBlocks(Field vel, float sy, float sx, float* by, float* bx, int H, int W, int size, int bh, int bw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= bw || j >= bh) return;
    float ay = 0, ax = 0;
    for (int dy = 0; dy < size; ++dy)
        for (int dx = 0; dx < size; ++dx) {
            int Y = j * size + dy, X = i * size + dx;
            ay += fieldUp(vel.y, vel.h, vel.w, H, W, Y, X) * sy;
            ax += fieldUp(vel.x, vel.h, vel.w, H, W, Y, X) * sx;
        }
    by[(size_t)j * bw + i] = ay / float(size * size);
    bx[(size_t)j * bw + i] = ax / float(size * size);
}

struct StepArgs {
    Field measured, velocity;
    const float* blockY;
    const float* blockX;
    int bh, bw, size;
    const float* cells;
    int ch, cw;
    float scaleY, scaleX, shortSide, threshold, motion, decay, residual, refresh, revealFloor;
    int mode, blocks, cut;
};

__global__ void moshStep(const float4* x, const float4* heldPrev, const float4* sourcePrev, float4* heldOut, float* activeOut,
                         int w, int h, StepArgs a) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    const size_t idx = (size_t)Y * w + X;
    float oy = fieldUp(a.measured.y, a.measured.h, a.measured.w, h, w, Y, X) * a.scaleY;
    float ox = fieldUp(a.measured.x, a.measured.h, a.measured.w, h, w, Y, X) * a.scaleX;
    float vy, vx;
    if (a.blocks) {
        int by = min(max(Y / a.size, 0), a.bh - 1), bx = min(max(X / a.size, 0), a.bw - 1);
        vy = a.blockY[(size_t)by * a.bw + bx];
        vx = a.blockX[(size_t)by * a.bw + bx];
    } else {
        vy = fieldUp(a.velocity.y, a.velocity.h, a.velocity.w, h, w, Y, X) * a.scaleY;
        vx = fieldUp(a.velocity.x, a.velocity.h, a.velocity.w, h, w, Y, X) * a.scaleX;
    }
    float magnitude = sqrtf(vy * vy + vx * vx) / a.shortSide;
    float active = dclamp(magnitude / a.threshold, 0.f, 1.f);
    float offY = vy * a.motion * active, offX = vx * a.motion * active;
    if (a.mode == 2) { offY = floorf(offY + 0.5f); offX = floorf(offX + 0.5f); }
    float4 held = imgSample(heldPrev, w, h, (float)Y + offY, (float)X + offX);
    float4 predicted = imgSample(sourcePrev, w, h, (float)Y + oy, (float)X + ox);
    float4 cur = x[idx];
    float4 res = make_float4(cur.x - predicted.x, cur.y - predicted.y, cur.z - predicted.z, 0.f);
    if (a.mode == 0) {
        float reveal = dclamp(((fabsf(res.x) + fabsf(res.y) + fabsf(res.z)) / 3.f - a.revealFloor) / 0.05f, 0.f, 1.f);
        if (a.cut) reveal = 0.f;
        held = make_float4(fmaxf(held.x * a.decay, cur.x * reveal * a.residual), fmaxf(held.y * a.decay, cur.y * reveal * a.residual),
                           fmaxf(held.z * a.decay, cur.z * reveal * a.residual), 0.f);
    } else if (!a.cut) {
        held = make_float4(held.x + res.x * a.residual, held.y + res.y * a.residual, held.z + res.z * a.residual, 0.f);
    }
    float refresh = a.refresh;
    if (a.cells) {
        int cy = min(max(Y / a.size, 0), a.ch - 1), cx = min(max(X / a.size, 0), a.cw - 1);
        refresh = a.cells[(size_t)cy * a.cw + cx] < a.refresh ? 1.f : 0.f;
    }
    heldOut[idx] = make_float4(dclamp(held.x * (1 - refresh) + cur.x * refresh, 0.f, 1.f), dclamp(held.y * (1 - refresh) + cur.y * refresh, 0.f, 1.f),
                               dclamp(held.z * (1 - refresh) + cur.z * refresh, 0.f, 1.f), 0.f);
    activeOut[idx] = active;
}

__global__ void moshDisplay(const float4* held, const float4* x, float4* display, int w, int h, int trails) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    const size_t idx = (size_t)Y * w + X;
    float4 c = held[idx];
    if (trails) {
        float4 s = x[idx];
        c = make_float4(dclamp(c.x + s.x * 0.04f, 0.f, 1.f), dclamp(c.y + s.y * 0.04f, 0.f, 1.f), dclamp(c.z + s.z * 0.04f, 0.f, 1.f), 0.f);
    }
    display[idx] = c;
}

__global__ void moshOutput(const float4* display, const float* active, const float4* x, const float* coverage, float4* exchange,
                           int w, int h, float separationPx, float saturation, float amount, int size, int ch, int cw) {
    int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= w || Y >= h) return;
    const size_t idx = (size_t)Y * w + X;
    float4 c = display[idx];
    if (active) {
        float sep = separationPx * active[idx];
        c.x = imgSample(display, w, h, (float)Y, X + sep).x;
        c.z = imgSample(display, w, h, (float)Y, X - sep).z;
    }
    float grey = dluma(c);
    c = make_float4(dclamp(grey + (c.x - grey) * saturation, 0.f, 1.f), dclamp(grey + (c.y - grey) * saturation, 0.f, 1.f),
                    dclamp(grey + (c.z - grey) * saturation, 0.f, 1.f), 0.f);
    float4 o = x[idx];
    if (coverage) {
        int cy = min(max(Y / size, 0), ch - 1), cx = min(max(X / size, 0), cw - 1);
        if (!(coverage[(size_t)cy * cw + cx] < amount)) c = o;
        c = make_float4(dclamp(c.x, 0.f, 1.f), dclamp(c.y, 0.f, 1.f), dclamp(c.z, 0.f, 1.f), 0.f);
    } else {
        c = make_float4(dclamp(o.x + (c.x - o.x) * amount, 0.f, 1.f), dclamp(o.y + (c.y - o.y) * amount, 0.f, 1.f),
                        dclamp(o.z + (c.z - o.z) * amount, 0.f, 1.f), 0.f);
    }
    exchange[(size_t)(h - 1 - Y) * w + X] = make_float4(c.x * o.w, c.y * o.w, c.z * o.w, o.w);
}

__device__ inline int lkReflect(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i > n - 1) i = i < 0 ? -i : 2 * (n - 1) - i;
    return i;
}

__global__ void lkWarp(const float* a, const float* b, const float* fy, const float* fx, float* avg, float* e, int h, int w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t i = (size_t)y * w + x;
    const float warped = fieldSample(b, h, w, (float)y + fy[i], (float)x + fx[i]);
    avg[i] = (warped + a[i]) * 0.5f;
    e[i] = a[i] - warped;
}

__global__ void lkTerms(const float* avg, const float* e, float* terms, double* ee, int h, int w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t i = (size_t)y * w + x, n = (size_t)h * w;
    const float gy = (avg[(size_t)lkReflect(y + 1, h) * w + x] - avg[(size_t)lkReflect(y - 1, h) * w + x]) * 0.5f;
    const float gx = (avg[(size_t)y * w + lkReflect(x + 1, w)] - avg[(size_t)y * w + lkReflect(x - 1, w)]) * 0.5f;
    const float ev = e[i];
    terms[i] = gx * gx;
    terms[n + i] = gx * gy;
    terms[2 * n + i] = gy * gy;
    terms[3 * n + i] = gx * ev;
    terms[4 * n + i] = gy * ev;
    atomicAdd(ee, double(ev * ev));
}

__global__ void planeBlur(const float* in, float* out, int planes, int h, int w, const float* weights, int radius, int horizontal, int pad, int period) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int n = horizontal ? w : h, i = horizontal ? x : y;
    const size_t plane = (size_t)h * w;
    for (int c = 0; c < planes; ++c) {
        float acc = 0;
        for (int m = -radius; m <= radius; ++m) {
            int j = (i + pad - m) % period;
            if (j < 0) j += period;
            const int k = lkReflect(j - pad, n);
            acc += in[c * plane + (horizontal ? (size_t)y * w + k : (size_t)k * w + x)] * weights[m + radius];
        }
        out[c * plane + (size_t)y * w + x] = acc;
    }
}

__global__ void lkUpdate(const float* sums, float* fy, float* fx, const double* ee, int h, int w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t i = (size_t)y * w + x, n = (size_t)h * w;
    const float reg = float(1e-3 + 1.5 * (ee[0] / (double(h) * w)));
    const float xx = sums[i] + reg, xy = sums[n + i], yy = sums[2 * n + i] + reg, xe = sums[3 * n + i], ye = sums[4 * n + i];
    const float det = xx * yy - xy * xy;
    fx[i] += dclamp((yy * xe - xy * ye) / det, -2.f, 2.f);
    fy[i] += dclamp((xx * ye - xy * xe) / det, -2.f, 2.f);
}

__global__ void planeUpscale(const float* in, int ih, int iw, float* out, int oh, int ow, float scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ow || y >= oh) return;
    out[(size_t)y * ow + x] = fieldUp(in, ih, iw, oh, ow, y, x) * scale;
}

__global__ void lkFinish(float* fy, float* fx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float mag = sqrtf(fy[i] * fy[i] + fx[i] * fx[i]);
    const float keep = dclamp((mag - 0.2f) / (0.45f - 0.2f), 0.f, 1.f);
    fy[i] *= keep;
    fx[i] *= keep;
}

struct LkGpu {
    DeviceBuffer a[4], b[4], fy, fx, fyT, fxT, avg, e, terms, tmp, sums, ee, weights;
};

LkGpu& lkGpu() {
    static LkGpu g;
    return g;
}

bool fitLk(DeviceBuffer& b, size_t n) { return b.bytes >= n || b.alloc(n); }

bool lucasKanadeGpu(const cw::Plane& cur, const cw::Plane& prev, cw::Plane& oy, cw::Plane& ox, cudaStream_t s) {
    const int h = cur.h, w = cur.w;
    const int q = std::max(1, static_cast<int>(std::ceil(std::min(h, w) / 256.0)));
    std::vector<std::pair<cw::Plane, cw::Plane>> pyramid{{cw::downscale(cur, q), cw::downscale(prev, q)}};
    while (std::min(pyramid.back().first.h, pyramid.back().first.w) >= 48 && pyramid.size() < 4)
        pyramid.push_back({cw::downscale(pyramid.back().first, 2), cw::downscale(pyramid.back().second, 2)});
    LkGpu& g = lkGpu();
    const size_t top = size_t(std::max(pyramid[0].first.h * pyramid[0].first.w, h * w));
    bool ok = fitLk(g.fy, top * 4) && fitLk(g.fx, top * 4) && fitLk(g.fyT, top * 4) && fitLk(g.fxT, top * 4) &&
              fitLk(g.avg, top * 4) && fitLk(g.e, top * 4) && fitLk(g.terms, top * 20) && fitLk(g.tmp, top * 20) &&
              fitLk(g.sums, top * 20) && fitLk(g.ee, sizeof(double));
    for (size_t k = 0; ok && k < pyramid.size(); ++k) {
        const size_t n = pyramid[k].first.v.size() * sizeof(float);
        ok = fitLk(g.a[k], n) && fitLk(g.b[k], n) &&
             cudaMemcpyAsync(g.a[k].ptr, pyramid[k].first.v.data(), n, cudaMemcpyHostToDevice, s) == cudaSuccess &&
             cudaMemcpyAsync(g.b[k].ptr, pyramid[k].second.v.data(), n, cudaMemcpyHostToDevice, s) == cudaSuccess;
    }
    if (!ok) return false;
    const dim3 blk(16, 16);
    auto f = [](DeviceBuffer& b) { return static_cast<float*>(b.ptr); };
    int lh = 0, lw = 0;
    std::vector<float> both;
    for (int k = int(pyramid.size()) - 1; k >= 0 && ok; --k) {
        const int nh = pyramid[k].first.h, nw = pyramid[k].first.w;
        const dim3 grd((nw + 15) / 16, (nh + 15) / 16);
        if (lh == 0) {
            ok = cudaMemsetAsync(g.fy.ptr, 0, size_t(nh) * nw * 4, s) == cudaSuccess && cudaMemsetAsync(g.fx.ptr, 0, size_t(nh) * nw * 4, s) == cudaSuccess;
        } else {
            planeUpscale<<<grd, blk, 0, s>>>(f(g.fy), lh, lw, f(g.fyT), nh, nw, float(double(nh) / lh));
            planeUpscale<<<grd, blk, 0, s>>>(f(g.fx), lh, lw, f(g.fxT), nh, nw, float(double(nw) / lw));
            std::swap(g.fy, g.fyT);
            std::swap(g.fx, g.fxT);
        }
        lh = nh; lw = nw;
        const cw::FftAxis ax = cw::fftGauss(3.f, lw), ay = cw::fftGauss(3.f, lh);
        if (ok) ok = cudaStreamSynchronize(s) == cudaSuccess;
        both = ax.weights;
        both.insert(both.end(), ay.weights.begin(), ay.weights.end());
        ok = ok && fitLk(g.weights, both.size() * 4) &&
             cudaMemcpyAsync(g.weights.ptr, both.data(), both.size() * 4, cudaMemcpyHostToDevice, s) == cudaSuccess;
        for (int iter = 0; iter < 3 && ok; ++iter) {
            ok = cudaMemsetAsync(g.ee.ptr, 0, sizeof(double), s) == cudaSuccess;
            lkWarp<<<grd, blk, 0, s>>>(f(g.a[k]), f(g.b[k]), f(g.fy), f(g.fx), f(g.avg), f(g.e), lh, lw);
            lkTerms<<<grd, blk, 0, s>>>(f(g.avg), f(g.e), f(g.terms), static_cast<double*>(g.ee.ptr), lh, lw);
            planeBlur<<<grd, blk, 0, s>>>(f(g.terms), f(g.tmp), 5, lh, lw, f(g.weights), ax.radius, 1, ax.pad, ax.period);
            planeBlur<<<grd, blk, 0, s>>>(f(g.tmp), f(g.sums), 5, lh, lw, f(g.weights) + ax.weights.size(), ay.radius, 0, ay.pad, ay.period);
            lkUpdate<<<grd, blk, 0, s>>>(f(g.sums), f(g.fy), f(g.fx), static_cast<double*>(g.ee.ptr), lh, lw);
        }
    }
    if (!ok) return false;
    const dim3 grd((w + 15) / 16, (h + 15) / 16);
    planeUpscale<<<grd, blk, 0, s>>>(f(g.fy), lh, lw, f(g.fyT), h, w, float(double(h) / lh));
    planeUpscale<<<grd, blk, 0, s>>>(f(g.fx), lh, lw, f(g.fxT), h, w, float(double(w) / lw));
    lkFinish<<<(h * w + 255) / 256, 256, 0, s>>>(f(g.fyT), f(g.fxT), h * w);
    oy = cw::Plane(h, w);
    ox = cw::Plane(h, w);
    ok = cudaMemcpyAsync(oy.v.data(), g.fyT.ptr, size_t(h) * w * 4, cudaMemcpyDeviceToHost, s) == cudaSuccess &&
         cudaMemcpyAsync(ox.v.data(), g.fxT.ptr, size_t(h) * w * 4, cudaMemcpyDeviceToHost, s) == cudaSuccess &&
         cudaStreamSynchronize(s) == cudaSuccess && cudaGetLastError() == cudaSuccess;
    return ok;
}

cw::Plane moshCells(uint32_t seed, int gh, int gw) {
    const int nh = std::max(2, (gh + 3) / 4), nw = std::max(2, (gw + 3) / 4);
    cw::Plane noise(nh, nw);
    noise.v = numpyRandomFloat32(seed, size_t(nh) * nw);
    cw::Plane cells = cw::upscale(noise, gh, gw);
    const auto [lo, hi] = std::minmax_element(cells.v.begin(), cells.v.end());
    const float minV = *lo, range = std::clamp(*hi - *lo, 1e-6f, 1.f);
    for (float& v : cells.v) v = (v - minV) / range;
    return cells;
}

__global__ void moshGray(const float4* x, float* gray, int w, int q, int gh, int gw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= gw || j >= gh) return;
    float s = 0;
    for (int dy = 0; dy < q; ++dy)
        for (int dx = 0; dx < q; ++dx) s += dluma(x[(size_t)(j * q + dy) * w + i * q + dx]);
    gray[(size_t)j * gw + i] = s / float(q * q);
}

int pyRound(double v) { return static_cast<int>(std::nearbyint(v)); }

struct MoshState {
    DeviceBuffer exchange, x, held, source, heldNext, display, active, grayDev;
    DeviceBuffer measuredY, measuredX, velY, velX, blockY, blockX, refreshCells, coverageCells;
    cw::Plane gray, velocityY, velocityX;
    int index = -1, h = 0, w = 0;
};

MoshState& moshState() {
    static MoshState s;
    return s;
}

struct FlowCacheEntry {
    std::vector<float> prev, cur;
    cw::Plane y, x;
};

bool cachedFlow(const cw::Plane& cur, const cw::Plane& prev, cw::Plane& oy, cw::Plane& ox, cudaStream_t s) {
    static std::vector<FlowCacheEntry> cache;
    for (const auto& e : cache)
        if (e.y.h == cur.h && e.y.w == cur.w && e.cur == cur.v && e.prev == prev.v) {
            oy = e.y;
            ox = e.x;
            return true;
        }
    if (!lucasKanadeGpu(cur, prev, oy, ox, s)) return false;
    if (cache.size() >= 16) cache.erase(cache.begin());
    cache.push_back({prev.v, cur.v, oy, ox});
    return true;
}

bool fit(DeviceBuffer& b, size_t n) { return b.bytes >= n || b.alloc(n); }

bool upload(DeviceBuffer& b, const std::vector<float>& v, cudaStream_t s) {
    return fit(b, std::max<size_t>(v.size(), 1) * sizeof(float)) &&
           (v.empty() || cudaMemcpyAsync(b.ptr, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice, s) == cudaSuccess);
}

}

bool cutwrightMosh(const MoshParams& p, const ImageView& exchange, bool reset, bool output) {
    const int w = exchange.bounds.w(), h = exchange.bounds.h();
    if (!available() || w < 1 || h < 1 || exchange.rowBytes != static_cast<ptrdiff_t>(w) * 16) return false;
    const int shortSide = std::min(w, h);
    const int q = std::max(1, static_cast<int>(std::ceil(shortSide / 192.0)));
    std::lock_guard<std::mutex> lock(deviceMutex());
    MoshState& st = moshState();
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) { cudaGetLastError(); return false; }
    const auto fail = [&]() {
        cudaStreamSynchronize(s);
        cudaStreamDestroy(s);
        cudaGetLastError();
        st.index = -1;
        return false;
    };
    const size_t bytes = size_t(w) * h * sizeof(float4);
    cw::Plane gray(h / q, w / q);
    bool ok = fit(st.exchange, bytes) && fit(st.x, bytes) && fit(st.held, bytes) && fit(st.source, bytes) &&
              fit(st.heldNext, bytes) && fit(st.display, bytes) && fit(st.active, size_t(w) * h * sizeof(float)) &&
              fit(st.grayDev, gray.v.size() * sizeof(float));
    ok = ok && cudaMemcpyAsync(st.exchange.ptr, exchange.data, bytes, cudaMemcpyHostToDevice, s) == cudaSuccess;
    const dim3 blk(16, 16), grd((w + 15) / 16, (h + 15) / 16);
    auto* ex = static_cast<float4*>(st.exchange.ptr);
    auto* x = static_cast<float4*>(st.x.ptr);
    if (ok) {
        moshUnpremult<<<grd, blk, 0, s>>>(ex, x, w, h);
        moshGray<<<dim3((gray.w + 15) / 16, (gray.h + 15) / 16), blk, 0, s>>>(x, static_cast<float*>(st.grayDev.ptr), w, q, gray.h, gray.w);
        ok = cudaMemcpyAsync(gray.v.data(), st.grayDev.ptr, gray.v.size() * sizeof(float), cudaMemcpyDeviceToHost, s) == cudaSuccess &&
             cudaStreamSynchronize(s) == cudaSuccess;
    }
    if (!ok) return fail();
    if (reset) st.index = -1;
    const bool fresh = st.index != p.index - 1 || st.h != h || st.w != w;
    bool cut = false;
    if (!fresh) {
        double sum = 0;
        for (size_t k = 0; k < gray.v.size(); ++k) sum += std::fabs(double(gray.v[k]) - st.gray.v[k]);
        cut = float(sum / gray.v.size()) > 0.12f;
    }
    const bool doReset = fresh || p.index % p.period == 0;
    const int size = std::max(2, pyRound(p.block * shortSide));
    const bool cellModes = p.mode >= 2;
    cw::Plane measuredY, measuredX, refreshCells, coverageCells;
    if (!doReset) {
        if (!cachedFlow(gray, st.gray, measuredY, measuredX, s)) return fail();
        if (!cut) {
            for (size_t k = 0; k < st.velocityY.v.size(); ++k) {
                st.velocityY.v[k] = st.velocityY.v[k] * p.memory + measuredY.v[k] * (1 - p.memory);
                st.velocityX.v[k] = st.velocityX.v[k] * p.memory + measuredX.v[k] * (1 - p.memory);
            }
        }
        if (cellModes) refreshCells = moshCells(uint32_t(p.seed + (long long)(p.index / 3) * 3571), (h + size - 1) / size, (w + size - 1) / size);
    } else {
        st.velocityY = cw::Plane(gray.h, gray.w);
        st.velocityX = cw::Plane(gray.h, gray.w);
    }
    const int size2 = std::max(2, pyRound(std::max(0.012f, p.block) * shortSide));
    if (cellModes) coverageCells = moshCells(uint32_t(p.seed + (long long)(p.index / p.period) * 1741), (h + size2 - 1) / size2, (w + size2 - 1) / size2);
    bool hasActive = false;
    if (doReset) {
        if (p.mode == 0) ok = cudaMemsetAsync(st.held.ptr, 0, bytes, s) == cudaSuccess;
        else ok = cudaMemcpyAsync(st.held.ptr, st.x.ptr, bytes, cudaMemcpyDeviceToDevice, s) == cudaSuccess;
    } else {
        ok = upload(st.measuredY, measuredY.v, s) && upload(st.measuredX, measuredX.v, s) &&
             upload(st.velY, st.velocityY.v, s) && upload(st.velX, st.velocityX.v, s);
        StepArgs a{};
        a.measured = {static_cast<float*>(st.measuredY.ptr), static_cast<float*>(st.measuredX.ptr), gray.h, gray.w};
        a.velocity = {static_cast<float*>(st.velY.ptr), static_cast<float*>(st.velX.ptr), gray.h, gray.w};
        a.scaleY = float(double(h) / gray.h);
        a.scaleX = float(double(w) / gray.w);
        a.size = size;
        a.blocks = (p.mode == 2 || p.mode == 3) && p.block > 0;
        if (ok && a.blocks) {
            a.bh = h / size; a.bw = w / size;
            ok = fit(st.blockY, size_t(a.bh) * a.bw * sizeof(float)) && fit(st.blockX, size_t(a.bh) * a.bw * sizeof(float));
            if (ok) {
                dim3 bg((a.bw + 15) / 16, (a.bh + 15) / 16);
                moshBlocks<<<bg, blk, 0, s>>>(a.velocity, a.scaleY, a.scaleX, static_cast<float*>(st.blockY.ptr),
                                              static_cast<float*>(st.blockX.ptr), h, w, size, a.bh, a.bw);
            }
            a.blockY = static_cast<float*>(st.blockY.ptr);
            a.blockX = static_cast<float*>(st.blockX.ptr);
        }
        if (ok && cellModes) {
            ok = upload(st.refreshCells, refreshCells.v, s);
            a.cells = static_cast<float*>(st.refreshCells.ptr);
            a.ch = refreshCells.h; a.cw = refreshCells.w;
        }
        a.shortSide = float(shortSide);
        a.threshold = std::max(0.0002f, p.threshold);
        a.motion = p.motion; a.decay = p.decay; a.residual = p.residual; a.refresh = p.refresh;
        a.revealFloor = std::max(0.01f, p.threshold * 8);
        a.mode = p.mode; a.cut = cut;
        if (ok) {
            moshStep<<<grd, blk, 0, s>>>(x, static_cast<float4*>(st.held.ptr), static_cast<float4*>(st.source.ptr),
                                         static_cast<float4*>(st.heldNext.ptr), static_cast<float*>(st.active.ptr), w, h, a);
            std::swap(st.held, st.heldNext);
            hasActive = true;
        }
    }
    if (ok && output) {
        moshDisplay<<<grd, blk, 0, s>>>(static_cast<float4*>(st.held.ptr), x, static_cast<float4*>(st.display.ptr), w, h, p.mode == 0);
        const float* coverage = nullptr;
        if (cellModes) {
            ok = upload(st.coverageCells, coverageCells.v, s);
            coverage = static_cast<float*>(st.coverageCells.ptr);
        }
        if (ok) moshOutput<<<grd, blk, 0, s>>>(static_cast<float4*>(st.display.ptr), p.mode == 3 && hasActive ? static_cast<float*>(st.active.ptr) : nullptr,
                                               x, coverage, ex, w, h, p.chromatic * shortSide * 0.004f, p.saturation, p.amount,
                                               size2, coverageCells.h, coverageCells.w);
        ok = ok && cudaMemcpyAsync(exchange.data, st.exchange.ptr, bytes, cudaMemcpyDeviceToHost, s) == cudaSuccess;
    }
    ok = ok && cudaMemcpyAsync(st.source.ptr, st.x.ptr, bytes, cudaMemcpyDeviceToDevice, s) == cudaSuccess;
    if (cudaStreamSynchronize(s) != cudaSuccess) ok = false;
    cudaStreamDestroy(s);
    if (!ok) { cudaGetLastError(); st.index = -1; return false; }
    st.gray = gray;
    st.index = p.index;
    st.h = h;
    st.w = w;
    return true;
}

bool cutwrightFlow(const cw::Plane& cur, const cw::Plane& prev, cw::Plane& fy, cw::Plane& fx) {
    if (!available() || cur.h != prev.h || cur.w != prev.w || cur.h < 2 || cur.w < 2) return false;
    std::lock_guard<std::mutex> lock(deviceMutex());
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) { cudaGetLastError(); return false; }
    const bool ok = lucasKanadeGpu(cur, prev, fy, fx, s);
    cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
    if (!ok) cudaGetLastError();
    return ok;
}

}
