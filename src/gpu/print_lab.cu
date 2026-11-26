#include "gpu_common.cuh"

namespace tp::gpu {

namespace {

struct DevTone {
    DevImage img;
    float fx1, fy1, ps;
    __host__ __device__ PrintLabRGB operator()(float qx, float qy) const {
#ifdef __CUDA_ARCH__
        float4 t = dBilinearClamp(img, fx1 + qx / ps, fy1 + qy / ps);
        return PrintLabRGB{plSat(t.x), plSat(t.y), plSat(t.z)};
#else
        return PrintLabRGB{};
#endif
    }
};

__global__ void plAlphaKernel(DevImage src, int x0, int y0, int w, int h, float4* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= w || j >= h) return;
    float alpha = dAtBlack(src, x0 + i, y0 + j).w;
    alpha = isfinite(alpha) ? fminf(1.f, fmaxf(0.f, alpha)) : 0.f;
    float4& o = out[(size_t)j * w + i];
    o = make_float4(o.x * alpha, o.y * alpha, o.z * alpha, alpha);
}

constexpr int kMaxMips = 9;

struct DevMips {
    DevImage lv[kMaxMips];
    int maxLod;
    __host__ __device__ PL4 operator()(float qx, float qy, float lod) const {
#ifdef __CUDA_ARCH__
        int l0 = (int)floorf(lod);
        float t = lod - l0;
        float sc0 = 1.f / (float)(1 << l0);
        float4 a = dBilinearClamp(lv[l0], qx * sc0, qy * sc0);
        if (t > 1e-3f && l0 < maxLod) a = f4lerp(a, dBilinearClamp(lv[l0 + 1], qx * sc0 * 0.5f, qy * sc0 * 0.5f), t);
        return PL4{a.x, a.y, a.z, a.w};
#else
        return PL4{};
#endif
    }
};

__global__ void plToneKernel(DevImage src, int fx1, int fy1, int fx2, int fy2, int rx1, int ry1, int rw, int rh, float bp, float range, int enc, float4* tone) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= rw || j >= rh) return;
    const int x = min(max(rx1 + i, fx1), fx2 - 1), y = min(max(ry1 + j, fy1), fy2 - 1);
    const float4 v = dAtClamp(src, x, y);
    const float a = isfinite(v.w) ? v.w : 0.f;
    float o[3] = {1.f, 1.f, 1.f};
    if (a > 1e-5f) {
        const float c[3] = {v.x, v.y, v.z};
        for (int ch = 0; ch < 3; ++ch) {
            const float q = c[ch] / a;
            float d = enc == 1 ? q : dEncodeValue(dDecodeValue(q, enc), 1);
            d = plSat(isfinite(d) ? d : 0.f);
            o[ch] = plSat((d - bp) / range);
        }
    }
    tone[(size_t)j * rw + i] = make_float4(o[0], o[1], o[2], 1.f);
}

__global__ void plPageKernel(PLParams P, DevTone tone, float4* page) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= P.PW || j >= P.PH) return;
    PL4 o = plPagePixel(P, i, j, tone);
    page[(size_t)j * P.PW + i] = make_float4(o.r, o.g, o.b, o.a);
}

__global__ void plMipKernel(DevImage prev, float4* next, int nw, int nh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nw || j >= nh) return;
    float4 a = f4add(f4add(dAtClamp(prev, 2 * i, 2 * j), dAtClamp(prev, 2 * i + 1, 2 * j)), f4add(dAtClamp(prev, 2 * i, 2 * j + 1), dAtClamp(prev, 2 * i + 1, 2 * j + 1)));
    next[(size_t)j * nw + i] = f4mul(a, 0.25f);
}

__global__ void plCameraKernel(PLParams P, PLCamera C, DevMips mips, int wx1, int wy1, int ww, int wh, float4* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ww || j >= wh) return;
    PL4 o = plCameraPixel(P, C, wx1 + i, wy1 + j, mips);
    out[(size_t)j * ww + i] = make_float4(o.r, o.g, o.b, 1.f);
}

}

bool printLab(const PLParams& Pin, const PLCamera& Cin, const PrintLabTone& toneIn, const ImageView& src, const ImageView& dst, const Rect& window) {
    if (!available() || window.empty() || !Pin.lut) return false;
    std::lock_guard<std::mutex> lock(deviceMutex());
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) { cudaGetLastError(); return false; }
    bool ok = true;
    PLParams P = Pin;
    PLCamera C = Cin;
    DeviceBuffer lutBuf, srcBuf, toneBuf, tmpBuf, outBuf;
    DeviceBuffer mipBufs[kMaxMips];
    DevTone tone{};
    ok = lutBuf.alloc((kPrintLabLutN + 1) * sizeof(float)) &&
         cudaMemcpyAsync(lutBuf.ptr, Pin.lut, (kPrintLabLutN + 1) * sizeof(float), cudaMemcpyHostToDevice, s) == cudaSuccess;
    P.lut = static_cast<const float*>(lutBuf.ptr);
    DevImage srcImg{};
    const Rect& TR = toneIn.region;
    const size_t toneBytes = static_cast<size_t>(TR.w()) * TR.h() * sizeof(float4);
    if (ok) ok = uploadRegion(src, toneIn.frame, srcBuf, srcImg, s) && srcImg.w > 0 && srcImg.h > 0 && toneBuf.alloc(toneBytes) && tmpBuf.alloc(toneBytes);
    if (ok) {
        auto* raw = static_cast<float4*>(tmpBuf.ptr);
        plToneKernel<<<dim3((TR.w() + 15) / 16, (TR.h() + 15) / 16), dim3(16, 16), 0, s>>>(srcImg, toneIn.frame.x1, toneIn.frame.y1, toneIn.frame.x2, toneIn.frame.y2,
            TR.x1, TR.y1, TR.w(), TR.h(), toneIn.blackPoint, toneIn.range, toneIn.encoding, raw);
        ok = cudaGetLastError() == cudaSuccess &&
             deviceGaussian(DevImage{raw, TR.x1, TR.y1, TR.w(), TR.h()}, toneIn.sigma, static_cast<float4*>(toneBuf.ptr), raw, s);
        tone.img = DevImage{static_cast<float4*>(toneBuf.ptr), TR.x1, TR.y1, TR.w(), TR.h()};
    }
    tone.fx1 = static_cast<float>(C.Fx1); tone.fy1 = static_cast<float>(C.Fy1); tone.ps = C.ps;
    DevMips mips{};
    int nl = 0;
    const dim3 blk(16, 16);
    if (ok) {
        ok = mipBufs[0].alloc(static_cast<size_t>(P.PW) * P.PH * sizeof(float4));
        if (ok) {
            mips.lv[0] = DevImage{static_cast<float4*>(mipBufs[0].ptr), 0, 0, P.PW, P.PH};
            dim3 grd((P.PW + 15) / 16, (P.PH + 15) / 16);
            plPageKernel<<<grd, blk, 0, s>>>(P, tone, static_cast<float4*>(mipBufs[0].ptr));
            ok = cudaGetLastError() == cudaSuccess;
            nl = 1;
        }
    }
    while (ok && nl < kMaxMips) {
        const DevImage& prev = mips.lv[nl - 1];
        if (prev.w <= 2 || prev.h <= 2) break;
        int nw = (prev.w + 1) / 2, nh = (prev.h + 1) / 2;
        if (!mipBufs[nl].alloc(static_cast<size_t>(nw) * nh * sizeof(float4))) { ok = false; break; }
        mips.lv[nl] = DevImage{static_cast<float4*>(mipBufs[nl].ptr), 0, 0, nw, nh};
        dim3 grd((nw + 15) / 16, (nh + 15) / 16);
        plMipKernel<<<grd, blk, 0, s>>>(prev, mips.lv[nl].data, nw, nh);
        ok = cudaGetLastError() == cudaSuccess;
        ++nl;
    }
    mips.maxLod = nl - 1;
    C.maxLod = nl - 1;
    const int ww = window.w(), wh = window.h();
    if (ok) ok = outBuf.alloc(static_cast<size_t>(ww) * wh * sizeof(float4));
    if (ok) {
        dim3 grd((ww + 15) / 16, (wh + 15) / 16);
        plCameraKernel<<<grd, blk, 0, s>>>(P, C, mips, window.x1, window.y1, ww, wh, static_cast<float4*>(outBuf.ptr));
        ok = cudaGetLastError() == cudaSuccess;
    }
    DeviceBuffer alphaBuf;
    DevImage alphaImg = srcImg;
    if (ok && !(window.x1 >= srcImg.x1 && window.y1 >= srcImg.y1 && window.x2 <= srcImg.x1 + srcImg.w && window.y2 <= srcImg.y1 + srcImg.h))
        ok = uploadRegion(src, window, alphaBuf, alphaImg, s);
    if (ok) {
        plAlphaKernel<<<dim3((ww + 15) / 16, (wh + 15) / 16), blk, 0, s>>>(alphaImg, window.x1, window.y1, ww, wh, static_cast<float4*>(outBuf.ptr));
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (ok) {
        DevImage o{static_cast<float4*>(outBuf.ptr), window.x1, window.y1, ww, wh};
        ok = downloadRegion(o, dst, window, s);
    }
    cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
    if (!ok) { cudaGetLastError(); return false; }
    return true;
}

}
