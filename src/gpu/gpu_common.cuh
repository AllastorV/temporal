#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include "gpu.h"

namespace tp::gpu {

struct DevImage {
    float4* data = nullptr;
    int x1 = 0, y1 = 0, w = 0, h = 0;
};

__device__ inline float4 f4add(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ inline float4 f4mul(float4 a, float s) { return make_float4(a.x * s, a.y * s, a.z * s, a.w * s); }
__device__ inline float4 f4lerp(float4 a, float4 b, float t) { return make_float4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t); }
__device__ inline float dclamp(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
__device__ inline float dsmooth(float e0, float e1, float x) {
    if (e1 == e0) return x < e0 ? 0.f : 1.f;
    float t = dclamp((x - e0) / (e1 - e0), 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
}
__device__ inline float dluma(float4 c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }

__device__ inline float4 dAtClamp(const DevImage& im, int x, int y) {
    x = x < im.x1 ? im.x1 : (x >= im.x1 + im.w ? im.x1 + im.w - 1 : x);
    y = y < im.y1 ? im.y1 : (y >= im.y1 + im.h ? im.y1 + im.h - 1 : y);
    return im.data[(size_t)(y - im.y1) * im.w + (x - im.x1)];
}
__device__ inline float4 dAtBlack(const DevImage& im, int x, int y) {
    if (x < im.x1 || y < im.y1 || x >= im.x1 + im.w || y >= im.y1 + im.h) return make_float4(0, 0, 0, 0);
    return im.data[(size_t)(y - im.y1) * im.w + (x - im.x1)];
}
__device__ inline float4 dBilinearClamp(const DevImage& im, float fx, float fy) {
    float x = fx - 0.5f, y = fy - 0.5f;
    float flx = floorf(x), fly = floorf(y);
    flx = dclamp(flx, -1e6f, 1e6f); fly = dclamp(fly, -1e6f, 1e6f);
    int x0 = (int)flx, y0 = (int)fly;
    float tx = x - flx, ty = y - fly;
    float4 a = dAtClamp(im, x0, y0), b = dAtClamp(im, x0 + 1, y0), c = dAtClamp(im, x0, y0 + 1), d = dAtClamp(im, x0 + 1, y0 + 1);
    return f4lerp(f4lerp(a, b, tx), f4lerp(c, d, tx), ty);
}

__device__ inline float dDecodeValue(float v, int e) {
    switch (e) {
    case 1: return v < 0.f ? -powf(-v, 2.4f) : powf(v, 2.4f);
    case 2: return v <= 0.02740668f ? v / 10.44426855f : exp2f(v / 0.07329248f - 7.0f) - 0.0075f;
    case 3: return v <= 0.155251141552511f ? (v - 0.0729055341958355f) / 10.5402377416545f : exp2f(v * 17.52f - 9.72f);
    default: return v;
    }
}
__device__ inline float dEncodeValue(float v, int e) {
    switch (e) {
    case 1: return v < 0.f ? -powf(-v, 1.f / 2.4f) : powf(v, 1.f / 2.4f);
    case 2: return v <= 0.00262409f ? v * 10.44426855f : (log2f(v + 0.0075f) + 7.0f) * 0.07329248f;
    case 3: return v <= 0.0078125f ? 10.5402377416545f * v + 0.0729055341958355f : (log2f(v) + 9.72f) / 17.52f;
    default: return v;
    }
}
__device__ inline float4 dDecode(float4 c, int e) {
    if (e == 0) return c;
    return make_float4(dDecodeValue(c.x, e), dDecodeValue(c.y, e), dDecodeValue(c.z, e), c.w);
}
__device__ inline float4 dEncode(float4 c, int e) {
    if (e == 0) return c;
    return make_float4(dEncodeValue(c.x, e), dEncodeValue(c.y, e), dEncodeValue(c.z, e), c.w);
}

struct DeviceBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) { alloc(n); }
    bool alloc(size_t n) {
        release();
        if (n == 0) return true;
        if (cudaMalloc(&ptr, n) != cudaSuccess) { ptr = nullptr; return false; }
        bytes = n;
        return true;
    }
    void release() { if (ptr) cudaFree(ptr); ptr = nullptr; bytes = 0; }
    ~DeviceBuffer() { release(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), bytes(o.bytes) { o.ptr = nullptr; o.bytes = 0; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept { release(); ptr = o.ptr; bytes = o.bytes; o.ptr = nullptr; o.bytes = 0; return *this; }
};

struct DevLin {
    DevImage img;
    int edge = 0;
#ifdef __CUDACC__
    __device__ static int mirror(int index, int origin, int count) {
        int pos = (index - origin) % (2 * count);
        if (pos < 0) pos += 2 * count;
        return origin + (pos < count ? pos : 2 * count - 1 - pos);
    }
    __device__ float4 at(int x, int y) const {
        if (edge == 1) {
            x = mirror(x, img.x1, img.w);
            y = mirror(y, img.y1, img.h);
        }
        return dAtBlack(img, x, y);
    }
#endif
    __host__ __device__ PL4 operator()(float sx, float sy) const {
#ifdef __CUDA_ARCH__
        float4 v;
        if (edge == 0) v = dBilinearClamp(img, sx, sy);
        else {
            const float fx = sx - 0.5f, fy = sy - 0.5f;
            const int x = (int)floorf(fx), y = (int)floorf(fy);
            v = f4lerp(f4lerp(at(x, y), at(x + 1, y), fx - x),
                       f4lerp(at(x, y + 1), at(x + 1, y + 1), fx - x), fy - y);
        }
        return PL4{v.x, v.y, v.z, v.w};
#else
        return PL4{};
#endif
    }
};

bool deviceGaussian(const DevImage& in, float sigma, float4* out, float4* tmp, cudaStream_t s);
bool uploadRegion(const ImageView& src, const Rect& r, DeviceBuffer& buf, DevImage& out, cudaStream_t s);
bool downloadRegion(const DevImage& im, const ImageView& dst, const Rect& r, cudaStream_t s);
bool uploadStaged(const ImageView& src, const Rect& r, DeviceBuffer& buf, DevImage& out, cudaStream_t s);
bool downloadStaged(const DevImage& im, const ImageView& dst, const Rect& r, cudaStream_t s);
std::mutex& deviceMutex();

}
