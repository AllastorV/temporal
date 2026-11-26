#include "gpu_common.cuh"
#include <atomic>
#include <cstdint>
#include <xmmintrin.h>

namespace tp::gpu {

namespace {
std::atomic<int> gEnabled{-1};

struct Staging {
    static constexpr size_t kChunk = size_t(8) << 20;
    std::mutex m;
    char* buf[2] = {};
    cudaEvent_t ev[2] = {};
    int state = 0;
    bool ready() {
        if (state == 0) {
            state = -1;
            for (int i = 0; i < 2; ++i)
                if (cudaHostAlloc(reinterpret_cast<void**>(&buf[i]), kChunk, cudaHostAllocDefault) != cudaSuccess ||
                    cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming) != cudaSuccess) {
                    cudaGetLastError();
                    for (int k = 0; k < 2; ++k) {
                        if (buf[k]) cudaFreeHost(buf[k]);
                        if (ev[k]) cudaEventDestroy(ev[k]);
                        buf[k] = nullptr; ev[k] = nullptr;
                    }
                    return false;
                }
            state = 1;
        }
        return state == 1;
    }
};

void streamCopy(char* d, const char* s, size_t n) {
    size_t i = 0;
    if ((reinterpret_cast<std::uintptr_t>(d) & 15) == 0)
        for (; i + 64 <= n; i += 64) {
            const __m128 a = _mm_loadu_ps(reinterpret_cast<const float*>(s + i)), b = _mm_loadu_ps(reinterpret_cast<const float*>(s + i + 16));
            const __m128 c = _mm_loadu_ps(reinterpret_cast<const float*>(s + i + 32)), e = _mm_loadu_ps(reinterpret_cast<const float*>(s + i + 48));
            _mm_stream_ps(reinterpret_cast<float*>(d + i), a);
            _mm_stream_ps(reinterpret_cast<float*>(d + i + 16), b);
            _mm_stream_ps(reinterpret_cast<float*>(d + i + 32), c);
            _mm_stream_ps(reinterpret_cast<float*>(d + i + 48), e);
        }
    std::memcpy(d + i, s + i, n - i);
    _mm_sfence();
}

Staging& staging() {
    static Staging s;
    return s;
}
}

std::mutex& deviceMutex() {
    static std::mutex m;
    return m;
}

bool available() {
    static const bool ok = [] {
        if (const char* e = std::getenv("TP_DISABLE_GPU")) if (e[0] && e[0] != '0') return false;
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) { cudaGetLastError(); return false; }
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return false;
        return prop.major >= 6;
    }();
    return ok && enabled();
}

void setEnabled(bool on) { gEnabled = on ? 1 : 0; }
bool enabled() { return gEnabled.load() != 0; }

bool uploadRegion(const ImageView& src, const Rect& r, DeviceBuffer& buf, DevImage& out, cudaStream_t s) {
    Rect q = src.valid() ? intersect(r, src.bounds) : Rect{};
    out = DevImage{};
    if (q.empty()) return true;
    size_t bytes = static_cast<size_t>(q.w()) * q.h() * sizeof(float4);
    if (buf.bytes < bytes && !buf.alloc(bytes)) return false;
    out.data = static_cast<float4*>(buf.ptr);
    out.x1 = q.x1; out.y1 = q.y1; out.w = q.w(); out.h = q.h();
    const size_t rowW = static_cast<size_t>(q.w()) * sizeof(float4);
    if (src.rowBytes > 0) {
        if (cudaMemcpy2DAsync(out.data, rowW, src.px(q.x1, q.y1), static_cast<size_t>(src.rowBytes), rowW, q.h(), cudaMemcpyDefault, s) != cudaSuccess) return false;
    } else {
        for (int y = q.y1; y < q.y2; ++y)
            if (cudaMemcpyAsync(reinterpret_cast<char*>(out.data) + static_cast<size_t>(y - q.y1) * rowW, src.px(q.x1, y), rowW, cudaMemcpyDefault, s) != cudaSuccess) return false;
    }
    return true;
}

bool downloadRegion(const DevImage& im, const ImageView& dst, const Rect& r, cudaStream_t s) {
    const size_t rowW = static_cast<size_t>(r.w()) * sizeof(float4);
    if (dst.rowBytes > 0) {
        if (cudaMemcpy2DAsync(dst.px(r.x1, r.y1), static_cast<size_t>(dst.rowBytes), im.data, rowW, rowW, r.h(), cudaMemcpyDefault, s) != cudaSuccess) return false;
    } else {
        for (int y = r.y1; y < r.y2; ++y)
            if (cudaMemcpyAsync(dst.px(r.x1, y), reinterpret_cast<const char*>(im.data) + static_cast<size_t>(y - r.y1) * rowW, rowW, cudaMemcpyDefault, s) != cudaSuccess) return false;
    }
    return cudaStreamSynchronize(s) == cudaSuccess;
}


bool syncStream(void* stream) {
    return cudaStreamSynchronize(static_cast<cudaStream_t>(stream)) == cudaSuccess;
}

bool copyImage(const ImageView& from, const ImageView& to, const Rect& r, void* stream) {
    const Rect q = intersect(intersect(r, from.bounds), to.bounds);
    if (q.empty()) return true;
    if (!from.valid() || !to.valid()) return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const size_t rowW = static_cast<size_t>(q.w()) * sizeof(float4);
    if (from.rowBytes > 0 && to.rowBytes > 0) {
        if (cudaMemcpy2DAsync(to.px(q.x1, q.y1), static_cast<size_t>(to.rowBytes), from.px(q.x1, q.y1), static_cast<size_t>(from.rowBytes), rowW, q.h(), cudaMemcpyDefault, s) != cudaSuccess) return false;
    } else {
        for (int y = q.y1; y < q.y2; ++y)
            if (cudaMemcpyAsync(to.px(q.x1, y), from.px(q.x1, y), rowW, cudaMemcpyDefault, s) != cudaSuccess) return false;
    }
    return cudaStreamSynchronize(s) == cudaSuccess;
}

namespace {
__global__ void finishKernel(const char* src, std::ptrdiff_t srcRow, Rect sb, char* dst, std::ptrdiff_t dstRow, Rect db, Rect r, float mix) {
    const int x = r.x1 + blockIdx.x * blockDim.x + threadIdx.x, y = r.y1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= r.x2 || y >= r.y2) return;
    float4* d = reinterpret_cast<float4*>(dst + static_cast<std::ptrdiff_t>(y - db.y1) * dstRow) + (x - db.x1);
    float4 v = *d;
    if (mix < 1.f) {
        float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
        if (src && x >= sb.x1 && y >= sb.y1 && x < sb.x2 && y < sb.y2) a = *(reinterpret_cast<const float4*>(src + static_cast<std::ptrdiff_t>(y - sb.y1) * srcRow) + (x - sb.x1));
        v = make_float4(__fadd_rn(a.x, __fmul_rn(__fsub_rn(v.x, a.x), mix)), __fadd_rn(a.y, __fmul_rn(__fsub_rn(v.y, a.y), mix)),
                        __fadd_rn(a.z, __fmul_rn(__fsub_rn(v.z, a.z), mix)), __fadd_rn(a.w, __fmul_rn(__fsub_rn(v.w, a.w), mix)));
    }
    *d = make_float4(isfinite(v.x) ? v.x : 0.f, isfinite(v.y) ? v.y : 0.f, isfinite(v.z) ? v.z : 0.f, isfinite(v.w) ? v.w : 0.f);
}
}

bool finishDevice(const ImageView& src, const ImageView& dst, const Rect& r, float mix, void* stream) {
    const Rect q = intersect(r, dst.bounds);
    if (q.empty()) return true;
    if (!dst.valid() || dst.rowBytes <= 0 || (src.valid() && src.rowBytes <= 0)) return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const dim3 blk(16, 16), grd((q.w() + 15) / 16, (q.h() + 15) / 16);
    finishKernel<<<grd, blk, 0, s>>>(src.valid() ? reinterpret_cast<const char*>(src.data) : nullptr, src.rowBytes, src.bounds,
                                     reinterpret_cast<char*>(dst.data), dst.rowBytes, dst.bounds, q, mix);
    return cudaGetLastError() == cudaSuccess && cudaStreamSynchronize(s) == cudaSuccess;
}

bool hostMemory(const void* p) {
    cudaPointerAttributes a{};
    if (cudaPointerGetAttributes(&a, p) != cudaSuccess) { cudaGetLastError(); return true; }
    return a.type != cudaMemoryTypeDevice && a.type != cudaMemoryTypeManaged;
}

bool uploadStaged(const ImageView& src, const Rect& r, DeviceBuffer& buf, DevImage& out, cudaStream_t s) {
    const Rect q = src.valid() ? intersect(r, src.bounds) : Rect{};
    const size_t rowW = static_cast<size_t>(q.w()) * sizeof(float4);
    Staging& g = staging();
    std::unique_lock<std::mutex> lock(g.m);
    if (q.empty() || rowW > Staging::kChunk || !g.ready() || !hostMemory(src.data)) {
        lock.unlock();
        return uploadRegion(src, r, buf, out, s);
    }
    out = DevImage{};
    const size_t bytes = rowW * q.h();
    if (buf.bytes < bytes && !buf.alloc(bytes)) return false;
    out.data = static_cast<float4*>(buf.ptr);
    out.x1 = q.x1; out.y1 = q.y1; out.w = q.w(); out.h = q.h();
    char* dev = reinterpret_cast<char*>(out.data);
    const int rows = static_cast<int>(Staging::kChunk / rowW);
    for (int y0 = q.y1, slot = 0; y0 < q.y2; y0 += rows, slot ^= 1) {
        const int y1 = std::min(q.y2, y0 + rows);
        char* stage = g.buf[slot];
        bool ok = cudaEventSynchronize(g.ev[slot]) == cudaSuccess;
        if (ok) parallelRows(y0, y1, [&](int y) { streamCopy(stage + static_cast<size_t>(y - y0) * rowW, reinterpret_cast<const char*>(src.px(q.x1, y)), rowW); });
        ok = ok && cudaMemcpyAsync(dev + static_cast<size_t>(y0 - q.y1) * rowW, stage, static_cast<size_t>(y1 - y0) * rowW, cudaMemcpyHostToDevice, s) == cudaSuccess &&
             cudaEventRecord(g.ev[slot], s) == cudaSuccess;
        if (!ok) { cudaStreamSynchronize(s); return false; }
    }
    return true;
}

bool downloadStaged(const DevImage& im, const ImageView& dst, const Rect& r, cudaStream_t s) {
    const size_t rowW = static_cast<size_t>(r.w()) * sizeof(float4);
    Staging& g = staging();
    std::unique_lock<std::mutex> lock(g.m);
    if (r.empty() || rowW > Staging::kChunk || !g.ready() || !hostMemory(dst.data)) {
        lock.unlock();
        return downloadRegion(im, dst, r, s);
    }
    const char* dev = reinterpret_cast<const char*>(im.data);
    const int rows = static_cast<int>(Staging::kChunk / rowW);
    auto issue = [&](int y0, int slot) {
        const int y1 = std::min(r.y2, y0 + rows);
        return cudaStreamWaitEvent(s, g.ev[slot], 0) == cudaSuccess &&
               cudaMemcpyAsync(g.buf[slot], dev + static_cast<size_t>(y0 - r.y1) * rowW, static_cast<size_t>(y1 - y0) * rowW, cudaMemcpyDeviceToHost, s) == cudaSuccess &&
               cudaEventRecord(g.ev[slot], s) == cudaSuccess;
    };
    bool ok = issue(r.y1, 0);
    for (int y0 = r.y1, slot = 0; ok && y0 < r.y2; y0 += rows, slot ^= 1) {
        if (y0 + rows < r.y2) ok = issue(y0 + rows, slot ^ 1);
        ok = ok && cudaEventSynchronize(g.ev[slot]) == cudaSuccess;
        if (!ok) break;
        const int y1 = std::min(r.y2, y0 + rows);
        const char* stage = g.buf[slot];
        parallelRows(y0, y1, [&](int y) { streamCopy(reinterpret_cast<char*>(dst.px(r.x1, y)), stage + static_cast<size_t>(y - y0) * rowW, rowW); });
    }
    return cudaStreamSynchronize(s) == cudaSuccess && ok;
}

}
