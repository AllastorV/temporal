#include "gpu_common.cuh"

namespace tp::gpu {
namespace {

bool uploadGrid(TearVectorGrid& grid, DeviceBuffer& buffer, cudaStream_t stream) {
    if (!grid.width || !grid.height) return true;
    const size_t count = size_t(grid.width) * grid.height;
    const size_t bytes = count * sizeof(float);
    if (!buffer.alloc(2 * bytes)) return false;
    float* u = static_cast<float*>(buffer.ptr);
    float* v = u + count;
    if (cudaMemcpyAsync(u, grid.u, bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(v, grid.v, bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess)
        return false;
    grid.u = u;
    grid.v = v;
    return true;
}

__device__ float blend(float a, float b, float t) { return a + (b - a) * t; }

__device__ float2 sampleGrid(const TearVectorGrid& grid, float px, float py) {
    const float x = (px - grid.originX) / grid.spacing - 0.5f;
    const float y = (py - grid.originY) / grid.spacing - 0.5f;
    const int ix = __float2int_rd(x), iy = __float2int_rd(y);
    const int left = max(0, min(ix, grid.width - 1)), right = max(0, min(ix + 1, grid.width - 1));
    const int top = max(0, min(iy, grid.height - 1)) * grid.width;
    const int bottom = max(0, min(iy + 1, grid.height - 1)) * grid.width;
    const float fx = x - ix, fy = y - iy;
    auto sample = [&](const float* field) {
        return blend(blend(field[top + left], field[top + right], fx),
                     blend(field[bottom + left], field[bottom + right], fx), fy) * grid.vectorScale;
    };
    return make_float2(sample(grid.u), sample(grid.v));
}

__global__ void expose(DevImage linear, DevImage filtered, DevImage out, TearVectorGrid flow,
                       TearVectorGrid nearby, TearExposure p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= out.w || j >= out.h) return;
    const int x = out.x1 + i, y = out.y1 + j;
    const float px = x + 0.5f, py = y + 0.5f;
    float2 velocity = make_float2(p.fixedX, p.fixedY);
    if (p.fromMotion) {
        velocity = make_float2(0, 0);
        if (flow.width) {
            const float2 own = sampleGrid(flow, px, py), local = sampleGrid(nearby, px, py);
            const float magnitude = hypotf(local.x, local.y);
            const float weight = magnitude > 1e-4f ?
                dsmooth(0.2f, 0.6f, hypotf(own.x, own.y) / magnitude) : 1.f;
            velocity = make_float2(blend(local.x, own.x, weight), blend(local.y, own.y, weight));
        }
    }
    velocity.x *= p.bias;
    velocity.y *= p.bias;
    const float speed = hypotf(velocity.x, velocity.y);
    const float4 base = dAtBlack(linear, x, y);
    float4& output = out.data[size_t(j) * out.w + i];
    if (!(speed > p.threshold) || p.shutter <= 0) {
        output = dEncode(base, p.encoding);
        return;
    }
    float dx = velocity.x * p.shutter, dy = velocity.y * p.shutter;
    const float length = hypotf(dx, dy);
    if (length > p.diagonal) {
        dx *= p.diagonal / length;
        dy *= p.diagonal / length;
    }
    const float span = hypotf(dx, dy);
    const int samples = max(4, min(p.samples, __float2int_ru(span / 1.5f)));
    const float lowpass = filtered.data ? dsmooth(1.5f, 3.5f, span / samples) : 0.f;
    float4 sum{}, hot{};
    float total = 0.f;
    for (int k = 0; k < samples; ++k) {
        const float s = p.sampleStart + (k + 0.5f) / samples;
        const float sx = px + dx * s, sy = py + dy * s;
        float4 tap = dBilinearClamp(linear, sx, sy);
        if (lowpass > 0.f) tap = f4lerp(tap, dBilinearClamp(filtered, sx, sy), lowpass);
        const float weight = blend(1.f, expf(-3.f * fabsf(s)), p.softness);
        sum = f4add(sum, f4mul(tap, weight));
        total += weight;
        if (p.highlights > 0.f) {
            const float luma = dluma(tap);
            hot = f4add(hot, f4mul(tap, weight * fmaxf(0.f, luma - p.highlights) / fmaxf(luma, 1e-4f)));
        }
    }
    float4 trail = f4mul(sum, 1.f / total);
    trail.x *= p.tint.r;
    trail.y *= p.tint.g;
    trail.z *= p.tint.b;
    const float gate = dsmooth(p.threshold, 2.f * p.threshold + 0.5f, speed);
    float4 result = f4lerp(base, trail, gate * fminf(1.f, p.amount));
    hot = f4mul(hot, p.amount / fmaxf(1.f, total * 0.25f));
    result.x += hot.x * p.tint.r;
    result.y += hot.y * p.tint.g;
    result.z += hot.z * p.tint.b;
    output = dEncode(result, p.encoding);
}
}

bool motionTear(const ImageView& linear, const ImageView& filtered, const ImageView& dst,
                Rect window, TearVectorGrid flow, TearVectorGrid nearby, TearExposure exposure) {
    std::lock_guard<std::mutex> guard(deviceMutex());
    cudaStream_t stream{};
    if (cudaStreamCreate(&stream) != cudaSuccess) return false;
    DeviceBuffer input, blur, output, flowBuffer, nearbyBuffer;
    DevImage src{}, low{};
    bool ok = uploadRegion(linear, linear.bounds, input, src, stream) &&
              uploadRegion(filtered, filtered.bounds, blur, low, stream) &&
              output.alloc(size_t(window.w()) * window.h() * sizeof(float4));
    if (ok && exposure.fromMotion && flow.width)
        ok = uploadGrid(flow, flowBuffer, stream) && uploadGrid(nearby, nearbyBuffer, stream);
    if (ok) {
        DevImage out{static_cast<float4*>(output.ptr), window.x1, window.y1, window.w(), window.h()};
        dim3 block(16, 16), grid((out.w + 15) / 16, (out.h + 15) / 16);
        expose<<<grid, block, 0, stream>>>(src, low, out, flow, nearby, exposure);
        ok = cudaGetLastError() == cudaSuccess && downloadRegion(out, dst, window, stream);
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) ok = false;
    cudaStreamDestroy(stream);
    return ok;
}
}
