#include "gpu_common.cuh"

namespace tp::gpu {

namespace {

__device__ inline int dMirror(int i, int lo, int hi) {
    int n = hi - lo;
    if (n <= 1) return lo;
    int p = 2 * n;
    int k = (i - lo) % p;
    if (k < 0) k += p;
    return lo + (k < n ? k : p - 1 - k);
}

__device__ inline float4 dAtEdge(const DevImage& im, int x, int y, int edge) {
    if (x >= im.x1 && y >= im.y1 && x < im.x1 + im.w && y < im.y1 + im.h) return im.data[(size_t)(y - im.y1) * im.w + (x - im.x1)];
    if (edge == 2) return make_float4(0, 0, 0, 0);
    if (edge == 1) { x = dMirror(x, im.x1, im.x1 + im.w); y = dMirror(y, im.y1, im.y1 + im.h); }
    else { x = min(max(x, im.x1), im.x1 + im.w - 1); y = min(max(y, im.y1), im.y1 + im.h - 1); }
    return im.data[(size_t)(y - im.y1) * im.w + (x - im.x1)];
}

__device__ inline float4 dBilinearEdge(const DevImage& im, float fx, float fy, int edge) {
    if (!isfinite(fx) || !isfinite(fy)) return make_float4(0, 0, 0, 0);
    float x = fx - 0.5f, y = fy - 0.5f;
    float flx = floorf(x), fly = floorf(y);
    if (flx < -1e6f || flx > 1e6f || fly < -1e6f || fly > 1e6f) return edge == 2 ? make_float4(0, 0, 0, 0) : dAtEdge(im, 0, 0, edge);
    int x0 = (int)flx, y0 = (int)fly;
    float tx = x - flx, ty = y - fly;
    float4 a = dAtEdge(im, x0, y0, edge), b = dAtEdge(im, x0 + 1, y0, edge), c = dAtEdge(im, x0, y0 + 1, edge), d = dAtEdge(im, x0 + 1, y0 + 1, edge);
    return f4lerp(f4lerp(a, b, tx), f4lerp(c, d, tx), ty);
}

__global__ void decodeKernel(float4* data, size_t n, int enc) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = dDecode(data[i], enc);
}

__global__ void motionKernel(DevImage src, const MotionSample* samples, int ns, float cx, float cy, float halfDiag, int edge, int enc,
                             int wx1, int wy1, int ww, int wh, float4* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ww || j >= wh) return;
    float px = wx1 + i + 0.5f, py = wy1 + j + 0.5f;
    float4 acc = make_float4(0, 0, 0, 0);
    for (int k = 0; k < ns; ++k) {
        const MotionSample s = samples[k];
        float qx = s.a * px + s.b * py + s.tx, qy = s.c * px + s.d * py + s.ty;
        if (s.logS != 0.f) {
            float dx = qx - cx, dy = qy - cy;
            float r = sqrtf(dx * dx + dy * dy) / halfDiag;
            float f = expf(-s.logS * dsmooth(0.12f, 1.0f, r));
            qx = cx + dx * f; qy = cy + dy * f;
        }
        acc = f4add(acc, dBilinearEdge(src, qx, qy, edge));
    }
    out[(size_t)j * ww + i] = dEncode(f4mul(acc, 1.f / ns), enc);
}

}

bool motionResample(const MotionArgs& a) {
    if (!available() || a.window.empty() || a.samples.empty() || !a.src.valid()) return false;
    std::lock_guard<std::mutex> lock(deviceMutex());
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) { cudaGetLastError(); return false; }
    DeviceBuffer srcBuf, sBuf, outBuf;
    DevImage src;
    bool ok = uploadRegion(a.src, a.src.bounds, srcBuf, src, s);
    const int ww = a.window.w(), wh = a.window.h();
    if (ok && a.encoding != 0) {
        size_t n = (size_t)src.w * src.h;
        decodeKernel<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(src.data, n, a.encoding);
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (ok) ok = sBuf.alloc(a.samples.size() * sizeof(MotionSample)) && outBuf.alloc((size_t)ww * wh * sizeof(float4)) &&
                 cudaMemcpyAsync(sBuf.ptr, a.samples.data(), a.samples.size() * sizeof(MotionSample), cudaMemcpyHostToDevice, s) == cudaSuccess;
    if (ok) {
        dim3 blk(16, 16), grd((ww + 15) / 16, (wh + 15) / 16);
        motionKernel<<<grd, blk, 0, s>>>(src, static_cast<const MotionSample*>(sBuf.ptr), (int)a.samples.size(), a.cx, a.cy, a.halfDiag, a.edge, a.encoding,
                                          a.window.x1, a.window.y1, ww, wh, static_cast<float4*>(outBuf.ptr));
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (ok) {
        DevImage o{static_cast<float4*>(outBuf.ptr), a.window.x1, a.window.y1, ww, wh};
        ok = downloadRegion(o, a.dst, a.window, s);
    }
    cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
    if (!ok) cudaGetLastError();
    return ok;
}

namespace {
struct DevicePlane {
    float* base;
    ptrdiff_t rowBytes;
    Rect bounds;
    __device__ float4* at(int x,int y) const {
        return reinterpret_cast<float4*>(reinterpret_cast<char*>(base)+(y-bounds.y1)*rowBytes)+(x-bounds.x1);
    }
    __device__ float4 sample(int x,int y,int edge,int enc) const {
        if(x<bounds.x1 || y<bounds.y1 || x>=bounds.x2 || y>=bounds.y2) {
            if(edge==2) return make_float4(0,0,0,0);
            if(edge==1) { x=dMirror(x,bounds.x1,bounds.x2); y=dMirror(y,bounds.y1,bounds.y2); }
            else { x=min(max(x,bounds.x1),bounds.x2-1); y=min(max(y,bounds.y1),bounds.y2-1); }
        }
        return dDecode(*at(x,y),enc);
    }
};
struct NativeSamples { MotionSample values[64]; };
__global__ void nativeMotionKernel(DevicePlane src, DevicePlane dst, Rect window, NativeSamples list,
                                   int count, int edge,int enc,float mix) {
    int x=window.x1+blockIdx.x*blockDim.x+threadIdx.x;
    int y=window.y1+blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=window.x2 || y>=window.y2) return;
    float4 sum=make_float4(0,0,0,0);
    for(int k=0;k<count;++k) {
        MotionSample s=list.values[k];
        float qx=s.a*(x+0.5f)+s.b*(y+0.5f)+s.tx-0.5f;
        float qy=s.c*(x+0.5f)+s.d*(y+0.5f)+s.ty-0.5f;
        int i=__float2int_rd(qx),j=__float2int_rd(qy);
        float4 top=f4lerp(src.sample(i,j,edge,enc),src.sample(i+1,j,edge,enc),qx-i);
        float4 bottom=f4lerp(src.sample(i,j+1,edge,enc),src.sample(i+1,j+1,edge,enc),qx-i);
        sum=f4add(sum,f4lerp(top,bottom,qy-j));
    }
    float4 result=dEncode(f4mul(sum,1.f/count),enc);
    if(mix<1.f) result=f4lerp(src.sample(x,y,2,0),result,mix);
    *dst.at(x,y)=make_float4(isfinite(result.x)?result.x:0.f,isfinite(result.y)?result.y:0.f,
                              isfinite(result.z)?result.z:0.f,isfinite(result.w)?result.w:0.f);
}
}

bool motionNative(const MotionArgs& a,float mix,void* hostStream) {
    if(a.samples.empty() || a.samples.size()>64 || !a.src.valid() || !a.dst.valid()) return false;
    NativeSamples list{};
    std::copy(a.samples.begin(),a.samples.end(),list.values);
    DevicePlane src{a.src.data,a.src.rowBytes,a.src.bounds},dst{a.dst.data,a.dst.rowBytes,a.dst.bounds};
    cudaStream_t stream=static_cast<cudaStream_t>(hostStream);
    dim3 block(16,16),grid((a.window.w()+15)/16,(a.window.h()+15)/16);
    nativeMotionKernel<<<grid,block,0,stream>>>(src,dst,a.window,list,int(a.samples.size()),a.edge,a.encoding,mix);
    if(cudaGetLastError()!=cudaSuccess) return false;
    // Without a stream from the host the OFX contract requires synchronous completion.
    return hostStream != nullptr || cudaStreamSynchronize(stream)==cudaSuccess;
}

}
