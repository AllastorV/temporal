#include "gpu_common.cuh"

namespace tp::gpu {

namespace {

__global__ void tsLayerKernel(DevImage frame, float4* layer, int lw, int lh, int rx1, int ry1, int rx2, int ry2, int ds, int enc, float lr, float cov, int shadow) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= lw || j >= lh) return;
    float4 out = make_float4(0, 0, 0, 0);
    if (frame.data) {
        int xa = rx1 + i * ds, xb = min(rx2, xa + ds), ya = ry1 + j * ds, yb = min(ry2, ya + ds);
        float th = 1.f - cov;
        float4 acc = make_float4(0, 0, 0, 0);
        float wacc = 0.f;
        for (int y = ya; y < yb; ++y)
            for (int x = xa; x < xb; ++x) {
                float4 s = dDecode(dAtClamp(frame, x, y), enc);
                float l = dclamp(dluma(s), 0.f, 64.f);
                float lc = shadow ? 1.f - fminf(l, 1.f) : fminf(l, 1.f);
                float w = (1.f + lr * lc * lc) * dsmooth(th - 0.25f, th, shadow ? lc : l);
                acc = f4add(acc, f4mul(s, w));
                wacc += w;
            }
        float inv = 1.f / (float)((xb - xa) * (yb - ya));
        out = f4mul(acc, inv);
        out.w = wacc * inv;
    }
    layer[(size_t)j * lw + i] = out;
}

__global__ void tsCompositeKernel(DevImage cur, const float4* const* layers, int lw, int lh, const int* subTap, const float* subF, int nsub,
                                  int wx1, int wy1, int ww, int wh, float4* out, int mode, int combine, int shadow,
                                  float dx, float dy, float cx, float cy, float radial, int rx1, int ry1, float invDs, float norm, int enc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ww || j >= wh) return;
    int x = wx1 + i, y = wy1 + j;
    float px = x + 0.5f, py = y + 0.5f;
    float vx = dx, vy = dy;
    if (mode == 2) { vx = (px - cx) * radial; vy = (py - cy) * radial; }
    float4 curLin = dDecode(dAtBlack(cur, x, y), enc);
    DevImage L;
    L.x1 = 0; L.y1 = 0; L.w = lw; L.h = lh;
    float4 acc = make_float4(0, 0, 0, 0);
    for (int k = 0; k < nsub; ++k) {
        float f = subF[k];
        float fade = powf(fmaxf(0.f, 1.f - 0.92f * f), 1.3f);
        L.data = const_cast<float4*>(layers[subTap[k]]);
        float lx = (px - vx * f - rx1) * invDs, ly = (py - vy * f - ry1) * invDs;
        acc = f4add(acc, f4mul(dBilinearClamp(L, lx, ly), fade));
    }
    float wsum = acc.w;
    float4 o;
    if (combine != 1) {
        float4 col = wsum > 1e-6f ? f4mul(acc, 1.f / wsum) : curLin;
        float a = 1.f - expf(-2.5f * wsum / norm);
        float4 trail = f4lerp(curLin, col, a);
        o.x = shadow ? fminf(curLin.x, trail.x) : fmaxf(curLin.x, trail.x);
        o.y = shadow ? fminf(curLin.y, trail.y) : fmaxf(curLin.y, trail.y);
        o.z = shadow ? fminf(curLin.z, trail.z) : fmaxf(curLin.z, trail.z);
    } else {
        float inv = 1.f / (1.f + wsum);
        o = make_float4((curLin.x + acc.x) * inv, (curLin.y + acc.y) * inv, (curLin.z + acc.z) * inv, 0.f);
    }
    o.w = curLin.w;
    out[(size_t)j * ww + i] = dEncode(o, enc);
}

}

bool timeShift(const TimeShiftArgs& a) {
    if (!available() || a.window.empty() || a.subTap.empty()) return false;
    std::lock_guard<std::mutex> lock(deviceMutex());
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) { cudaGetLastError(); return false; }
    bool ok = true;
    const int lw = (a.R.w() + a.ds - 1) / a.ds, lh = (a.R.h() + a.ds - 1) / a.ds;
    const size_t layerPx = static_cast<size_t>(lw) * lh;
    std::vector<float4*> layerPtrs;
    DeviceBuffer layerBuf, frameBuf, outBuf, ptrBuf, tapBuf, fBuf;
    const dim3 blk(16, 16);
    ok = layerBuf.alloc(a.taps.size() * layerPx * sizeof(float4));
    for (size_t k = 0; ok && k < a.taps.size(); ++k) {
        layerPtrs.push_back(static_cast<float4*>(layerBuf.ptr) + k * layerPx);
        DevImage fr;
        if (!uploadStaged(a.taps[k], a.R, frameBuf, fr, s)) { ok = false; break; }
        dim3 grd((lw + 15) / 16, (lh + 15) / 16);
        tsLayerKernel<<<grd, blk, 0, s>>>(fr, layerPtrs.back(), lw, lh, a.R.x1, a.R.y1, a.R.x2, a.R.y2, a.ds, a.encoding, a.lr, a.cov, a.shadow ? 1 : 0);
        if (cudaGetLastError() != cudaSuccess) ok = false;
    }
    DevImage cur;
    const int ww = a.window.w(), wh = a.window.h();
    if (ok) ok = uploadStaged(a.cur, a.cur.valid() ? a.cur.bounds : a.window, frameBuf, cur, s);
    if (ok) ok = ptrBuf.alloc(layerPtrs.size() * sizeof(float4*)) && tapBuf.alloc(a.subTap.size() * sizeof(int)) && fBuf.alloc(a.subF.size() * sizeof(float)) &&
                 outBuf.alloc(static_cast<size_t>(ww) * wh * sizeof(float4));
    if (ok) ok = cudaMemcpyAsync(ptrBuf.ptr, layerPtrs.data(), layerPtrs.size() * sizeof(float4*), cudaMemcpyHostToDevice, s) == cudaSuccess &&
                 cudaMemcpyAsync(tapBuf.ptr, a.subTap.data(), a.subTap.size() * sizeof(int), cudaMemcpyHostToDevice, s) == cudaSuccess &&
                 cudaMemcpyAsync(fBuf.ptr, a.subF.data(), a.subF.size() * sizeof(float), cudaMemcpyHostToDevice, s) == cudaSuccess;
    if (ok) {
        dim3 grd((ww + 15) / 16, (wh + 15) / 16);
        tsCompositeKernel<<<grd, blk, 0, s>>>(cur, static_cast<const float4* const*>(ptrBuf.ptr), lw, lh, static_cast<const int*>(tapBuf.ptr), static_cast<const float*>(fBuf.ptr),
                                              static_cast<int>(a.subTap.size()), a.window.x1, a.window.y1, ww, wh, static_cast<float4*>(outBuf.ptr), a.mode, a.combine, a.shadow ? 1 : 0,
                                              a.dx, a.dy, a.cx, a.cy, a.radial, a.R.x1, a.R.y1, 1.f / a.ds, a.norm, a.encoding);
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (ok) {
        DevImage o;
        o.data = static_cast<float4*>(outBuf.ptr);
        o.x1 = a.window.x1; o.y1 = a.window.y1; o.w = ww; o.h = wh;
        ok = downloadStaged(o, a.dst, a.window, s);
    }
    cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
    if (!ok) cudaGetLastError();
    return ok;
}

}
