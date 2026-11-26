#include "gpu_common.cuh"

namespace tp::gpu {

namespace {

__device__ inline float4 dHalHighlight(float4 lin, float t, float w) {
    float4 c = make_float4(fmaxf(0.f, lin.x), fmaxf(0.f, lin.y), fmaxf(0.f, lin.z), 0.f);
    float l = dluma(c);
    if (l <= 1e-8f) return make_float4(0, 0, 0, 0);
    float s = w <= 1e-6f ? (l > t ? 1.f : 0.f) : dsmooth(t - w, t + w, l);
    if (s <= 0.f) return make_float4(0, 0, 0, 0);
    return make_float4(c.x * s, c.y * s, c.z * s, 0.f);
}

__global__ void halAccumKernel(DevImage img, float wgt, int rx1, int ry1, int rx2, int ry2, int f, int lw, int lh, float thr, float kw, int enc, float4* low, bool last) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= lw || j >= lh) return;
    const int ya = ry1 + j * f, yb = min(ry2, ya + f);
    const int xa = rx1 + i * f, xb = min(rx2, xa + f);
    float4 acc = low[(size_t)j * lw + i];
    if (img.data) {
        float4 s = make_float4(0, 0, 0, 0);
        for (int y = ya; y < yb; ++y)
            for (int x = xa; x < xb; ++x) s = f4add(s, dHalHighlight(dDecode(dAtBlack(img, x, y), enc), thr, kw));
        acc = f4add(acc, f4mul(s, wgt));
    }
    if (last) acc = f4mul(acc, 1.f / (float)((xb - xa) * (yb - ya)));
    low[(size_t)j * lw + i] = acc;
}

}

bool halationAccum(const std::vector<ImageView>& layers, const std::vector<float>& weights, const Rect& R, int f, float thr, float kw, int enc, const ImageView& low) {
    if (!available() || layers.empty() || R.empty()) return false;
    std::lock_guard<std::mutex> lock(deviceMutex());
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) { cudaGetLastError(); return false; }
    const int lw = low.bounds.w(), lh = low.bounds.h();
    DeviceBuffer lowBuf, imgBuf;
    bool ok = lowBuf.alloc(static_cast<size_t>(lw) * lh * sizeof(float4)) &&
              cudaMemsetAsync(lowBuf.ptr, 0, lowBuf.bytes, s) == cudaSuccess;
    const dim3 blk(16, 16), grd((lw + 15) / 16, (lh + 15) / 16);
    for (size_t k = 0; ok && k < layers.size(); ++k) {
        DevImage img{};
        if (layers[k].valid()) ok = uploadStaged(layers[k], R, imgBuf, img, s);
        if (!ok) break;
        halAccumKernel<<<grd, blk, 0, s>>>(img, weights[k], R.x1, R.y1, R.x2, R.y2, f, lw, lh, thr, kw, enc, static_cast<float4*>(lowBuf.ptr), k + 1 == layers.size());
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (ok) {
        DevImage o{static_cast<float4*>(lowBuf.ptr), low.bounds.x1, low.bounds.y1, lw, lh};
        ok = downloadStaged(o, low, low.bounds, s);
    }
    cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
    if (!ok) { cudaGetLastError(); return false; }
    return true;
}

__global__ void compositeKernel(DevImage src, DevImage low, DevImage output, int rx, int ry,
                                float inv, float4 tint, bool only, int encoding) {
    int i = blockIdx.x*blockDim.x+threadIdx.x, j = blockIdx.y*blockDim.y+threadIdx.y;
    if (i>=output.w || j>=output.h) return;
    int x=output.x1+i,y=output.y1+j;
    float4 glow=dBilinearClamp(low,(x-rx+0.5f)*inv,(y-ry+0.5f)*inv);
    glow.x*=tint.x; glow.y*=tint.y; glow.z*=tint.z;
    float4 out;
    if (only) {
        glow.w=dclamp(fmaxf(glow.x,fmaxf(glow.y,glow.z)),0,1);
        out=dEncode(glow,encoding);
    } else {
        out=dAtBlack(src,x,y);
        if (glow.x!=0 || glow.y!=0 || glow.z!=0) {
            float4 lin=dDecode(out,encoding);
            lin.x+=glow.x;lin.y+=glow.y;lin.z+=glow.z;
            out=dEncode(lin,encoding);
        }
    }
    output.data[size_t(j)*output.w+i]=out;
}

bool halationComposite(const ImageView& cur, const ImageView& low, const ImageView& dst, const Rect& window, const Rect& region, int factor, const RGBA& tint, float gain, bool only, int encoding) {
    std::lock_guard<std::mutex> guard(deviceMutex());
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream)!=cudaSuccess) return false;
    DeviceBuffer srcBuf,lowBuf,outBuf;
    DevImage src{},blur{};
    bool ok=uploadStaged(cur,window,srcBuf,src,stream) && uploadStaged(low,low.bounds,lowBuf,blur,stream);
    const bool inPlace=src.x1==window.x1 && src.y1==window.y1 && src.w==window.w() && src.h==window.h();
    ok=ok && (inPlace || outBuf.alloc(size_t(window.w())*window.h()*sizeof(float4)));
    if (ok) {
        DevImage out{inPlace?src.data:static_cast<float4*>(outBuf.ptr),window.x1,window.y1,window.w(),window.h()};
        dim3 block(16,16), grid((out.w+15)/16,(out.h+15)/16);
        compositeKernel<<<grid,block,0,stream>>>(src,blur,out,region.x1,region.y1,1.f/factor,
            make_float4(tint.r*gain,tint.g*gain,tint.b*gain,0),only,encoding);
        ok=cudaGetLastError()==cudaSuccess && downloadStaged(out,dst,window,stream);
    }
    if(cudaStreamSynchronize(stream)!=cudaSuccess) ok=false;
    cudaStreamDestroy(stream);
    return ok;
}

}
