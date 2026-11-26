#include "gpu_common.cuh"

namespace tp::gpu {

namespace {

__global__ void rainDown(DevImage src, int f, int lw, int lh, float4* low) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= lw || j >= lh) return;
    const int xa = src.x1 + i * f, xb = min(src.x1 + src.w, xa + f), ya = src.y1 + j * f, yb = min(src.y1 + src.h, ya + f);
    float4 s = make_float4(0, 0, 0, 0);
    for (int y = ya; y < yb; ++y)
        for (int x = xa; x < xb; ++x) s = f4add(s, src.data[(size_t)(y - src.y1) * src.w + (x - src.x1)]);
    low[(size_t)j * lw + i] = f4mul(s, 1.f / (float)((xb - xa) * (yb - ya)));
}

__global__ void rainDisk(DevImage low, float radius, int taps, float4* out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= low.w || y >= low.h) return;
    const float ca = -0.73736887f, sa = 0.67549029f;
    float4 acc = make_float4(0, 0, 0, 0);
    float u = 1.f, v = 0.f;
    for (int k = 0; k < taps; ++k) {
        const float rr = radius * sqrtf((k + 0.5f) / taps);
        acc = f4add(acc, dBilinearClamp(low, x + 0.5f + rr * u, y + 0.5f + rr * v));
        const float nu = u * ca - v * sa;
        v = u * sa + v * ca;
        u = nu;
    }
    out[(size_t)y * low.w + x] = f4mul(acc, 1.f / taps);
}

__global__ void rainUp(DevImage disk, int rw, int rh, int f, float4* out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= rw || y >= rh) return;
    const float inv = 1.f / (float)f;
    out[(size_t)y * rw + x] = f == 1 ? disk.data[(size_t)y * disk.w + x] : dBilinearClamp(disk, (x + 0.5f) * inv, (y + 0.5f) * inv);
}

__global__ void rainDecode(const float4* raw, float4* lin, size_t n, int enc) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) lin[i] = dDecode(raw[i], enc);
}

__global__ void rainBox(DevImage in, int r, int dx, int dy, float4* out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= in.w || y >= in.h) return;
    float4 acc = make_float4(0, 0, 0, 0);
    for (int k = -r; k <= r; ++k) acc = f4add(acc, dAtClamp(in, in.x1 + x + k * dx, in.y1 + y + k * dy));
    out[(size_t)y * in.w + x] = f4mul(acc, 1.f / (float)(2 * r + 1));
}

struct RainAt {
    DevImage img;
    __host__ __device__ PL4 operator()(int x, int y) const {
#ifdef __CUDA_ARCH__
        const float4 v = img.data[(size_t)(y - img.y1) * img.w + (x - img.x1)];
        return PL4{v.x, v.y, v.z, v.w};
#else
        (void)x; (void)y;
        return PL4{};
#endif
    }
};

__global__ void rainShadeKernel(RainShade P, RainMaps M, RainAt back, RainAt fog, DevLin inner, DevLin bent,
                                int ax1, int ay1, int aw, int ah, int enc, float4* out, float4* glass) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= aw || j >= ah) return;
    PL4 g;
    const PL4 o = rainShadePixel(P, M, ax1 + i, ay1 + j, back, fog, inner, bent, g);
    const size_t k = (size_t)j * aw + i;
    const float4 v = make_float4(o.r, o.g, o.b, o.a);
    if (P.blurDrops) {
        out[k] = v;
        glass[k] = make_float4(g.r, g.g, g.b, g.a);
    } else {
        out[k] = P.matte ? v : dEncode(v, enc);
    }
}

__global__ void rainComposite(DevImage glass, DevImage soft, int wx1, int wy1, int ww, int wh, int enc, float4* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ww || j >= wh) return;
    const float4 bg = dAtClamp(glass, wx1 + i, wy1 + j), d = dAtClamp(soft, wx1 + i, wy1 + j);
    out[(size_t)j * ww + i] = dEncode(make_float4(bg.x * (1.f - d.w) + d.x, bg.y * (1.f - d.w) + d.y, bg.z * (1.f - d.w) + d.z, bg.w), enc);
}

__global__ void rainSplatKernel(RainSplat S, int w, int h, float* height, float* wsum, float* wcx, float* wcy, float* wrad, float* wiped) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= w || j >= h) return;
    const size_t k = (size_t)j * w + i;
    float wp;
    rainSplatPixel(S, S.rx1 + i, S.ry1 + j, height[k], wsum[k], wcx[k], wcy[k], wrad[k], wp);
    if (wiped) wiped[k] = wp;
}

dim3 rainGrid(int w, int h) { return dim3((w + 15) / 16, (h + 15) / 16); }

bool diskDev(const DevImage& in, float radius, float4* out, cudaStream_t s) {
    const int f = std::max(1, std::min(16, static_cast<int>(radius / 5.f)));
    const int lw = (in.w + f - 1) / f, lh = (in.h + f - 1) / f;
    const float rl = radius / f;
    const int taps = std::max(12, std::min(96, static_cast<int>(std::ceil(1.3f * rl * rl))));
    DeviceBuffer lowBuf, diskBuf;
    if (!lowBuf.alloc((size_t)lw * lh * sizeof(float4)) || !diskBuf.alloc((size_t)lw * lh * sizeof(float4))) return false;
    const dim3 blk(16, 16);
    rainDown<<<rainGrid(lw, lh), blk, 0, s>>>(in, f, lw, lh, static_cast<float4*>(lowBuf.ptr));
    rainDisk<<<rainGrid(lw, lh), blk, 0, s>>>(DevImage{static_cast<float4*>(lowBuf.ptr), 0, 0, lw, lh}, rl, taps, static_cast<float4*>(diskBuf.ptr));
    rainUp<<<rainGrid(in.w, in.h), blk, 0, s>>>(DevImage{static_cast<float4*>(diskBuf.ptr), 0, 0, lw, lh}, in.w, in.h, f, out);
    const bool ok = cudaGetLastError() == cudaSuccess;
    cudaStreamSynchronize(s);
    return ok;
}


template <class T>
bool uploadArray(const T* host, size_t n, DeviceBuffer& buf, const T*& dev, cudaStream_t s) {
    dev = nullptr;
    if (!host || n == 0) return true;
    if (!buf.alloc(n * sizeof(T))) return false;
    dev = static_cast<const T*>(buf.ptr);
    return cudaMemcpyAsync(buf.ptr, host, n * sizeof(T), cudaMemcpyHostToDevice, s) == cudaSuccess;
}

}

bool deviceGaussian(const DevImage& in, float sigma, float4* out, float4* tmp, cudaStream_t s) {
    const size_t bytes = (size_t)in.w * in.h * sizeof(float4);
    if (cudaMemcpyAsync(out, in.data, bytes, cudaMemcpyDeviceToDevice, s) != cudaSuccess) return false;
    if (!(sigma > 0.35f)) return true;
    int radii[3];
    gaussianBoxRadii(sigma, radii);
    const dim3 blk(16, 16), gr = rainGrid(in.w, in.h);
    for (int r : radii) {
        if (r < 1) continue;
        rainBox<<<gr, blk, 0, s>>>(DevImage{out, in.x1, in.y1, in.w, in.h}, r, 1, 0, tmp);
        rainBox<<<gr, blk, 0, s>>>(DevImage{tmp, in.x1, in.y1, in.w, in.h}, r, 0, 1, out);
    }
    return cudaGetLastError() == cudaSuccess;
}

bool rainDiskBlur(const ImageView& src, const Rect& R, float radius, const ImageView& dst) {
    if (!available() || R.empty() || !src.valid() || !dst.valid()) return false;
    std::lock_guard<std::mutex> lock(deviceMutex());
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) { cudaGetLastError(); return false; }
    DeviceBuffer srcBuf, outBuf;
    DevImage in{};
    bool ok = uploadRegion(src, R, srcBuf, in, s) && in.w == R.w() && in.h == R.h() &&
              outBuf.alloc((size_t)R.w() * R.h() * sizeof(float4)) && diskDev(in, radius, static_cast<float4*>(outBuf.ptr), s) &&
              downloadRegion(DevImage{static_cast<float4*>(outBuf.ptr), R.x1, R.y1, R.w(), R.h()}, dst, R, s);
    cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
    if (!ok) { cudaGetLastError(); return false; }
    return true;
}

bool rainGlass(const RainGlassArgs& a) {
    const Rect& R = a.R;
    if (!available() || R.empty() || !a.src.valid() || !a.dst.valid()) return false;
    const Rect area = a.shade.blurDrops ? R : intersect(a.window, R);
    const Rect outRect = a.shade.blurDrops ? a.window : area;
    if (outRect.empty()) return true;
    const int W = R.w(), H = R.h();
    const size_t n = (size_t)W * H, img = n * sizeof(float4);
    std::lock_guard<std::mutex> lock(deviceMutex());
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) { cudaGetLastError(); return false; }
    DeviceBuffer rawBuf, linBuf, backBuf, innerBuf, fogBuf, tmpBuf, layerBuf, glassBuf, softBuf, outBuf;
    DeviceBuffer mapBuf[6], beadBuf, glintBuf, splatBuf[6];
    DevImage raw{};
    const bool mist = a.shade.mist > 0;
    RainSplat S = a.splat;
    RainMaps M;
    M.x1 = R.x1; M.y1 = R.y1; M.w = W; M.h = H;
    bool ok = uploadRegion(a.src, R, rawBuf, raw, s) && raw.w == W && raw.h == H && linBuf.alloc(img) && innerBuf.alloc(img) && tmpBuf.alloc(img) &&
              uploadArray(a.splat.drops, (size_t)a.splat.dropCount, splatBuf[0], S.drops, s) &&
              uploadArray(a.splat.wipes, (size_t)a.splat.wipeCount, splatBuf[1], S.wipes, s) &&
              uploadArray(a.splat.dropStart, (size_t)a.splat.bands + 1, splatBuf[2], S.dropStart, s) &&
              uploadArray(a.splat.dropList, (size_t)a.splat.dropListCount, splatBuf[3], S.dropList, s) &&
              uploadArray(a.splat.wipeStart, (size_t)a.splat.bands + 1, splatBuf[4], S.wipeStart, s) &&
              uploadArray(a.splat.wipeList, (size_t)a.splat.wipeListCount, splatBuf[5], S.wipeList, s) &&
              uploadArray(mist ? a.bead : nullptr, n, beadBuf, M.bead, s) && uploadArray(mist ? a.glint : nullptr, n, glintBuf, M.glint, s);
    ok = ok && S.dropStart && S.wipeStart && (!mist || (M.bead && M.glint));
    float* maps[6] = {};
    for (int k = 0; ok && k < 6; ++k) {
        if (k == 5 && !mist) break;
        ok = mapBuf[k].alloc(n * sizeof(float));
        maps[k] = static_cast<float*>(mapBuf[k].ptr);
    }
    if (ok) {
        rainSplatKernel<<<rainGrid(W, H), dim3(16, 16), 0, s>>>(S, W, H, maps[0], maps[1], maps[2], maps[3], maps[4], maps[5]);
        ok = cudaGetLastError() == cudaSuccess;
        M.height = maps[0]; M.wsum = maps[1]; M.wcx = maps[2]; M.wcy = maps[3]; M.wrad = maps[4]; M.wiped = maps[5];
    }
    auto* lin = static_cast<float4*>(linBuf.ptr);
    float4* back = lin;
    if (ok) {
        rainDecode<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(raw.data, lin, n, a.encoding);
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (ok && a.sceneBlur > 0.5f) {
        ok = backBuf.alloc(img) && diskDev(DevImage{lin, R.x1, R.y1, W, H}, a.sceneBlur, static_cast<float4*>(backBuf.ptr), s);
        back = static_cast<float4*>(backBuf.ptr);
    }
    ok = ok && deviceGaussian(DevImage{lin, R.x1, R.y1, W, H}, a.innerSigma, static_cast<float4*>(innerBuf.ptr), static_cast<float4*>(tmpBuf.ptr), s);
    if (ok && mist) ok = fogBuf.alloc(img) && deviceGaussian(DevImage{back, R.x1, R.y1, W, H}, a.fogSigma, static_cast<float4*>(fogBuf.ptr), static_cast<float4*>(tmpBuf.ptr), s);
    const DevImage backImg{back, R.x1, R.y1, W, H};
    if (ok) {
        const bool blur = a.shade.blurDrops;
        ok = blur ? layerBuf.alloc(img) && glassBuf.alloc(img) : outBuf.alloc((size_t)area.w() * area.h() * sizeof(float4));
        if (ok) {
            DevLin inner{DevImage{static_cast<float4*>(innerBuf.ptr), R.x1, R.y1, W, H}, 1}, bent{backImg, 1};
            rainShadeKernel<<<rainGrid(area.w(), area.h()), dim3(16, 16), 0, s>>>(a.shade, M, RainAt{backImg},
                RainAt{DevImage{static_cast<float4*>(fogBuf.ptr), R.x1, R.y1, W, H}}, inner, bent, area.x1, area.y1, area.w(), area.h(), a.encoding,
                static_cast<float4*>(blur ? layerBuf.ptr : outBuf.ptr), static_cast<float4*>(glassBuf.ptr));
            ok = cudaGetLastError() == cudaSuccess;
        }
        if (ok && blur) {
            ok = softBuf.alloc(img) && diskDev(DevImage{static_cast<float4*>(layerBuf.ptr), R.x1, R.y1, W, H}, a.dropBlur, static_cast<float4*>(softBuf.ptr), s) &&
                 outBuf.alloc((size_t)outRect.w() * outRect.h() * sizeof(float4));
            if (ok) {
                rainComposite<<<rainGrid(outRect.w(), outRect.h()), dim3(16, 16), 0, s>>>(DevImage{static_cast<float4*>(glassBuf.ptr), R.x1, R.y1, W, H},
                    DevImage{static_cast<float4*>(softBuf.ptr), R.x1, R.y1, W, H}, outRect.x1, outRect.y1, outRect.w(), outRect.h(), a.encoding,
                    static_cast<float4*>(outBuf.ptr));
                ok = cudaGetLastError() == cudaSuccess;
            }
        }
        ok = ok && downloadRegion(DevImage{static_cast<float4*>(outBuf.ptr), outRect.x1, outRect.y1, outRect.w(), outRect.h()}, a.dst, outRect, s);
    }
    cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
    if (!ok) { cudaGetLastError(); return false; }
    return true;
}

}
