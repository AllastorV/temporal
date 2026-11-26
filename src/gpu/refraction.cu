#include "gpu_common.cuh"

namespace tp::gpu {

namespace {

struct DevRaw {
    DevImage img;
    __host__ __device__ PL4 operator()(int x, int y) const {
#ifdef __CUDA_ARCH__
        float4 v = dAtBlack(img, x, y);
        return PL4{v.x, v.y, v.z, v.w};
#else
        return PL4{};
#endif
    }
};

struct DevDepth {
    DevLin depth;
    int sourceX, sourceY;
    float scaleX, scaleY;
    __host__ __device__ PL4 operator()(float x, float y) const {
        return depth(depth.img.x1 + (x - sourceX) * scaleX, depth.img.y1 + (y - sourceY) * scaleY);
    }
};

__global__ void rlDecodeKernel(const float4* raw, float4* lin, size_t n, int enc) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) lin[i] = dDecode(raw[i], enc);
}

__global__ void rlKernel(RLParams P, DevRaw raw, DevLin lin, int wx1, int wy1, int ww, int wh, float4* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ww || j >= wh) return;
    PL4 o = rlPixel(P, wx1 + i, wy1 + j, raw, lin);
    out[(size_t)j * ww + i] = make_float4(o.r, o.g, o.b, o.a);
}

__global__ void opticKernel(OpticStack stack, DevRaw raw, DevLin lin, DevLin sharp, int wx1, int wy1, int ww, int wh, int band, float4* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = band + blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ww || j >= wh) return;
    const PL4 pixel = opticStackPixel(stack, wx1 + i, wy1 + j, raw, lin, sharp);
    out[(size_t)j * ww + i] = make_float4(pixel.r, pixel.g, pixel.b, pixel.a);
}

__global__ void bokehKernel(DevLin lin, float radius, int taps, float highlight, float4* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= lin.img.w || j >= lin.img.h) return;
    const PL4 v = gsBokehPixel(lin, lin.img.x1 + i + 0.5f, lin.img.y1 + j + 0.5f, radius, taps, highlight);
    out[(size_t)j * lin.img.w + i] = make_float4(v.r, v.g, v.b, v.a);
}

__global__ void lensKernel(LensCharacterParams lens, DevLin lin, DevDepth depth,
                           int wx1, int wy1, int ww, int wh, float4* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ww || j >= wh) return;
    const PL4 pixel = lensCharacterPixel(lens, wx1 + i, wy1 + j, lin, depth);
    out[(size_t)j * ww + i] = make_float4(pixel.r, pixel.g, pixel.b, pixel.a);
}

bool renderGlass(const ImageView& src, const ImageView& dst, const Rect& window,
                 const RLParams* legacy, const OpticStack* stack) {
    if (!available() || window.empty() || !src.valid()) return false;
    const int encoding = legacy ? legacy->enc : (stack->mask ? 0 : stack->encoding);
    std::lock_guard<std::mutex> lock(deviceMutex());
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) { cudaGetLastError(); return false; }
    DeviceBuffer rawBuf, linBuf, bokehBuf, outBuf;
    DevRaw raw{};
    DevLin lin{};
    lin.edge = stack ? stack->edge : 0;
    bool ok = uploadRegion(src, src.bounds, rawBuf, raw.img, s);
    lin.img = raw.img;
    if (ok && encoding != 0) {
        size_t n = (size_t)raw.img.w * raw.img.h;
        ok = linBuf.alloc(n * sizeof(float4));
        if (ok) {
            rlDecodeKernel<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(raw.img.data, static_cast<float4*>(linBuf.ptr), n, encoding);
            ok = cudaGetLastError() == cudaSuccess;
            lin.img.data = static_cast<float4*>(linBuf.ptr);
        }
    }
    const DevLin sharp = lin;
    if (ok && stack && !stack->mask && stack->solid.shape > 0 && stack->solid.taps > 0) {
        ok = bokehBuf.alloc((size_t)lin.img.w * lin.img.h * sizeof(float4));
        if (ok) {
            dim3 blk(16, 16), grd((lin.img.w + 15) / 16, (lin.img.h + 15) / 16);
            bokehKernel<<<grd, blk, 0, s>>>(lin, stack->solid.blur, stack->solid.taps, stack->solid.highlight, static_cast<float4*>(bokehBuf.ptr));
            ok = cudaGetLastError() == cudaSuccess;
            lin.img.data = static_cast<float4*>(bokehBuf.ptr);
        }
    }
    const int ww = window.w(), wh = window.h();
    if (ok) ok = outBuf.alloc((size_t)ww * wh * sizeof(float4));
    if (ok) {
        dim3 blk(16, 16), grd((ww + 15) / 16, (wh + 15) / 16);
        if (legacy) {
            rlKernel<<<grd, blk, 0, s>>>(*legacy, raw, lin, window.x1, window.y1, ww, wh, static_cast<float4*>(outBuf.ptr));
            ok = cudaGetLastError() == cudaSuccess;
        } else {
            const int rows = stack->solid.shape > 0 && stack->solid.samples > 8 ? 256 : wh;
            for (int band = 0; ok && band < wh; band += rows) {
                const dim3 part(grd.x, (std::min(rows, wh - band) + 15) / 16);
                opticKernel<<<part, blk, 0, s>>>(*stack, raw, lin, sharp, window.x1, window.y1, ww, wh, band, static_cast<float4*>(outBuf.ptr));
                ok = cudaGetLastError() == cudaSuccess;
                if (ok && rows < wh) ok = cudaStreamSynchronize(s) == cudaSuccess;
            }
        }
    }
    if (ok) {
        DevImage o{static_cast<float4*>(outBuf.ptr), window.x1, window.y1, ww, wh};
        ok = downloadRegion(o, dst, window, s);
    }
    cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
    if (!ok) cudaGetLastError();
    return ok;
}

}

bool refraction(const RLParams& P, const ImageView& src, const ImageView& dst, const Rect& window) {
    return !P.useBlur && renderGlass(src, dst, window, &P, nullptr);
}

bool multiOptic(const OpticStack& stack, const ImageView& src, const ImageView& dst, const Rect& window) {
    return renderGlass(src, dst, window, nullptr, &stack);
}

bool lensCharacter(const LensCharacterParams& lens, const ImageView& src, const ImageView& depth,
                   const ImageView& dst, const Rect& window, SampleEdge edge) {
    if (!available() || window.empty() || !src.valid() || (lens.hasDepth && !depth.valid())) return false;
    std::lock_guard<std::mutex> lock(deviceMutex());
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) { cudaGetLastError(); return false; }
    DeviceBuffer rawBuf, linearBuf, depthBuf, outBuf;
    DevLin linear{};
    linear.edge = static_cast<int>(edge);
    bool ok = uploadRegion(src, src.bounds, rawBuf, linear.img, stream);
    if (ok && lens.encoding != 0) {
        const size_t count = static_cast<size_t>(linear.img.w) * linear.img.h;
        ok = linearBuf.alloc(count * sizeof(float4));
        if (ok) {
            auto* decoded = static_cast<float4*>(linearBuf.ptr);
            rlDecodeKernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(linear.img.data, decoded, count, lens.encoding);
            ok = cudaGetLastError() == cudaSuccess;
            linear.img.data = decoded;
        }
    }
    DevDepth depthSample{{}, src.bounds.x1, src.bounds.y1,
        static_cast<float>(depth.bounds.w()) / src.bounds.w(), static_cast<float>(depth.bounds.h()) / src.bounds.h()};
    if (ok && lens.hasDepth) ok = uploadRegion(depth, depth.bounds, depthBuf, depthSample.depth.img, stream);
    const int ww = window.w(), wh = window.h();
    if (ok) ok = outBuf.alloc(static_cast<size_t>(ww) * wh * sizeof(float4));
    if (ok) {
        const dim3 block(16, 16), grid((ww + 15) / 16, (wh + 15) / 16);
        auto* pixels = static_cast<float4*>(outBuf.ptr);
        lensKernel<<<grid, block, 0, stream>>>(lens, linear, depthSample, window.x1, window.y1, ww, wh, pixels);
        ok = cudaGetLastError() == cudaSuccess && downloadRegion({pixels, window.x1, window.y1, ww, wh}, dst, window, stream);
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    if (!ok) cudaGetLastError();
    return ok;
}

}
