#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <functional>
#include <thread>
#include <mutex>
#include <exception>
#include <vector>

namespace tp {

constexpr float kPi = 3.14159265358979323846f;

struct Rect {
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    int w() const { return x2 - x1; }
    int h() const { return y2 - y1; }
    bool empty() const { return x2 <= x1 || y2 <= y1; }
    bool contains(int x, int y) const { return x >= x1 && x < x2 && y >= y1 && y < y2; }
};

inline Rect intersect(const Rect& a, const Rect& b) {
    Rect r{std::max(a.x1, b.x1), std::max(a.y1, b.y1), std::min(a.x2, b.x2), std::min(a.y2, b.y2)};
    if (r.empty()) r = Rect{};
    return r;
}

struct RGBA {
    float r = 0, g = 0, b = 0, a = 0;
    RGBA() = default;
    RGBA(float r_, float g_, float b_, float a_) : r(r_), g(g_), b(b_), a(a_) {}
    RGBA operator+(const RGBA& o) const { return {r + o.r, g + o.g, b + o.b, a + o.a}; }
    RGBA operator-(const RGBA& o) const { return {r - o.r, g - o.g, b - o.b, a - o.a}; }
    RGBA operator*(float s) const { return {r * s, g * s, b * s, a * s}; }
    RGBA& operator+=(const RGBA& o) { r += o.r; g += o.g; b += o.b; a += o.a; return *this; }
    RGBA& operator*=(float s) { r *= s; g *= s; b *= s; a *= s; return *this; }
    float& operator[](int i) { return (&r)[i]; }
    float operator[](int i) const { return (&r)[i]; }
};

inline RGBA lerp(const RGBA& a, const RGBA& b, float t) { return a + (b - a) * t; }
inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float saturate(float v) { return clampf(v, 0.f, 1.f); }
inline float smoothstepf(float e0, float e1, float x) {
    if (e1 == e0) return x < e0 ? 0.f : 1.f;
    float t = saturate((x - e0) / (e1 - e0));
    return t * t * (3.f - 2.f * t);
}
inline float luma(const RGBA& c) { return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b; }
inline float finiteOr(float v, float fallback) { return std::isfinite(v) ? v : fallback; }

enum class SampleEdge { Clamp, Mirror, Black };

inline int mirrorIndex(int i, int lo, int hi) {
    int n = hi - lo;
    if (n <= 1) return lo;
    int p = 2 * n;
    int k = (i - lo) % p;
    if (k < 0) k += p;
    return lo + (k < n ? k : p - 1 - k);
}

struct ImageView {
    float* data = nullptr;
    Rect bounds;
    std::ptrdiff_t rowBytes = 0;

    bool valid() const { return data != nullptr && !bounds.empty(); }
    float* px(int x, int y) const {
        return reinterpret_cast<float*>(reinterpret_cast<char*>(data) + static_cast<std::ptrdiff_t>(y - bounds.y1) * rowBytes) + static_cast<std::ptrdiff_t>(x - bounds.x1) * 4;
    }
    RGBA load(int x, int y) const {
        const float* p = px(x, y);
        return {p[0], p[1], p[2], p[3]};
    }
    void store(int x, int y, const RGBA& c) const {
        float* p = px(x, y);
        p[0] = c.r; p[1] = c.g; p[2] = c.b; p[3] = c.a;
    }
    RGBA at(int x, int y, SampleEdge e = SampleEdge::Clamp) const {
        if (!valid()) return {};
        if (!bounds.contains(x, y)) {
            if (e == SampleEdge::Black) return {};
            if (e == SampleEdge::Mirror) {
                x = mirrorIndex(x, bounds.x1, bounds.x2);
                y = mirrorIndex(y, bounds.y1, bounds.y2);
            } else {
                x = std::clamp(x, bounds.x1, bounds.x2 - 1);
                y = std::clamp(y, bounds.y1, bounds.y2 - 1);
            }
        }
        return load(x, y);
    }
    RGBA bilinear(float fx, float fy, SampleEdge e = SampleEdge::Clamp) const {
        if (!valid() || !std::isfinite(fx) || !std::isfinite(fy)) return {};
        float x = fx - 0.5f, y = fy - 0.5f;
        float flx = std::floor(x), fly = std::floor(y);
        if (flx < -1e6f || flx > 1e6f || fly < -1e6f || fly > 1e6f) return e == SampleEdge::Black ? RGBA{} : at(0, 0, e);
        int x0 = static_cast<int>(flx), y0 = static_cast<int>(fly);
        float tx = x - flx, ty = y - fly;
        RGBA a = at(x0, y0, e), b = at(x0 + 1, y0, e), c = at(x0, y0 + 1, e), d = at(x0 + 1, y0 + 1, e);
        return lerp(lerp(a, b, tx), lerp(c, d, tx), ty);
    }
};

template <class F>
void parallelRows(int y1, int y2, F&& fn);

void* poolAllocate(size_t bytes);
void poolRelease(void* p, size_t bytes);

template <class T>
struct UninitAlloc : std::allocator<T> {
    using value_type = T;
    template <class U> struct rebind { using other = UninitAlloc<U>; };
    UninitAlloc() = default;
    template <class U> UninitAlloc(const UninitAlloc<U>&) noexcept {}
    T* allocate(size_t n) { return static_cast<T*>(poolAllocate(n * sizeof(T))); }
    void deallocate(T* p, size_t n) noexcept { poolRelease(p, n * sizeof(T)); }
    template <class U, class... A> void construct(U* p, A&&... a) {
        if constexpr (sizeof...(A) == 0) ::new (static_cast<void*>(p)) U;
        else ::new (static_cast<void*>(p)) U(std::forward<A>(a)...);
    }
};

using PixelBuffer = std::vector<float, UninitAlloc<float>>;

struct OwnedImage {
    PixelBuffer buf;
    ImageView view;
    OwnedImage() = default;
    explicit OwnedImage(const Rect& r) { reset(r); }
    void reset(const Rect& r) {
        const int w = std::max(0, r.w()), h = std::max(0, r.h());
        buf.clear();
        buf.resize(static_cast<size_t>(w) * h * 4);
        view.data = buf.empty() ? nullptr : buf.data();
        view.bounds = r;
        view.rowBytes = static_cast<std::ptrdiff_t>(w) * 4 * sizeof(float);
        if (!buf.empty()) parallelRows(0, h, [&](int y) { std::memset(buf.data() + static_cast<size_t>(y) * w * 4, 0, static_cast<size_t>(w) * 4 * sizeof(float)); });
    }
    OwnedImage(const OwnedImage& o) { *this = o; }
    OwnedImage& operator=(const OwnedImage& o) {
        if (this == &o) return *this;
        view = o.view;
        buf.clear();
        buf.resize(o.buf.size());
        view.data = buf.empty() ? nullptr : buf.data();
        const int h = std::max(0, o.view.bounds.h());
        const size_t row = h > 0 ? o.buf.size() / static_cast<size_t>(h) : 0;
        if (!buf.empty()) parallelRows(0, h, [&](int y) { std::memcpy(buf.data() + static_cast<size_t>(y) * row, o.buf.data() + static_cast<size_t>(y) * row, row * sizeof(float)); });
        return *this;
    }
    OwnedImage(OwnedImage&&) noexcept = default;
    OwnedImage& operator=(OwnedImage&&) noexcept = default;
};

inline unsigned workerCount() {
    unsigned n = std::thread::hardware_concurrency();
    return n == 0 ? 4u : n;
}

void runRowWorkers(unsigned count, void* context, void (*work)(void*));

template <class F>
void parallelRows(int y1, int y2, F&& fn) {
    int rows = y2 - y1;
    if (rows <= 0) return;
    unsigned nthreads = std::min<unsigned>(workerCount(), static_cast<unsigned>(rows));
    if (nthreads <= 1 || rows < 8) {
        for (int y = y1; y < y2; ++y) fn(y);
        return;
    }
    std::atomic<int> next{y1};
    const int chunk = std::max(1, rows / static_cast<int>(nthreads * 8));
    auto work = [&]() {
        for (;;) {
            int s = next.fetch_add(chunk);
            if (s >= y2) break;
            int e = std::min(y2, s + chunk);
            for (int y = s; y < e; ++y) fn(y);
        }
    };
    // Worker exceptions must reach the OFX render entry point rather than terminate Resolve.
    std::exception_ptr failure;
    std::mutex errorLock;
    auto guarded = [&]() {
        try { work(); }
        catch (...) {
            next.store(y2);
            std::lock_guard<std::mutex> lock(errorLock);
            if (!failure) failure = std::current_exception();
        }
    };
    runRowWorkers(nthreads, &guarded, [](void* context) {
        (*static_cast<decltype(guarded)*>(context))();
    });
    if (failure) std::rethrow_exception(failure);
}

inline void copyRect(const ImageView& src, const ImageView& dst, const Rect& r) {
    if (r.empty()) return;
    if (src.valid() && dst.valid() && src.bounds.contains(r.x1, r.y1) && src.bounds.contains(r.x2 - 1, r.y2 - 1)) {
        parallelRows(r.y1, r.y2, [&](int y) { std::memcpy(dst.px(r.x1, y), src.px(r.x1, y), static_cast<size_t>(r.w()) * 4 * sizeof(float)); });
        return;
    }
    parallelRows(r.y1, r.y2, [&](int y) {
        for (int x = r.x1; x < r.x2; ++x) dst.store(x, y, src.at(x, y));
    });
}

enum class Encoding { Linear = 0, Gamma24 = 1, DaVinciIntermediate = 2, ACEScct = 3 };

inline float decodeValue(float v, Encoding e) {
    switch (e) {
    case Encoding::Linear: return v;
    case Encoding::Gamma24: return v < 0.f ? -std::pow(-v, 2.4f) : std::pow(v, 2.4f);
    case Encoding::DaVinciIntermediate: {
        const float A = 0.0075f, B = 7.0f, C = 0.07329248f, M = 10.44426855f, LOG_CUT = 0.02740668f;
        return v <= LOG_CUT ? v / M : std::exp2(v / C - B) - A;
    }
    case Encoding::ACEScct: {
        const float X_BRK = 0.155251141552511f;
        return v <= X_BRK ? (v - 0.0729055341958355f) / 10.5402377416545f : std::exp2(v * 17.52f - 9.72f);
    }
    }
    return v;
}

inline float encodeValue(float v, Encoding e) {
    switch (e) {
    case Encoding::Linear: return v;
    case Encoding::Gamma24: return v < 0.f ? -std::pow(-v, 1.f / 2.4f) : std::pow(v, 1.f / 2.4f);
    case Encoding::DaVinciIntermediate: {
        const float A = 0.0075f, B = 7.0f, C = 0.07329248f, M = 10.44426855f, LIN_CUT = 0.00262409f;
        return v <= LIN_CUT ? v * M : (std::log2(v + A) + B) * C;
    }
    case Encoding::ACEScct: {
        const float Y_BRK = 0.0078125f;
        return v <= Y_BRK ? 10.5402377416545f * v + 0.0729055341958355f : (std::log2(v) + 9.72f) / 17.52f;
    }
    }
    return v;
}

#if defined(__AVX2__)
__forceinline __m128 log2x4(__m128 x) {
    const __m128i b = _mm_castps_si128(x);
    __m128i e = _mm_sub_epi32(_mm_srli_epi32(b, 23), _mm_set1_epi32(127));
    __m128 m = _mm_castsi128_ps(_mm_or_si128(_mm_and_si128(b, _mm_set1_epi32(0x007fffff)), _mm_set1_epi32(0x3f800000)));
    const __m128 big = _mm_cmpgt_ps(m, _mm_set1_ps(1.41421356f));
    m = _mm_blendv_ps(m, _mm_mul_ps(m, _mm_set1_ps(0.5f)), big);
    e = _mm_sub_epi32(e, _mm_castps_si128(big));
    const __m128 one = _mm_set1_ps(1.f);
    const __m128 t = _mm_div_ps(_mm_sub_ps(m, one), _mm_add_ps(m, one)), t2 = _mm_mul_ps(t, t), t4 = _mm_mul_ps(t2, t2);
    const __m128 lo = _mm_fmadd_ps(t2, _mm_set1_ps(0.961796694f), _mm_set1_ps(2.88539008f));
    const __m128 hi = _mm_fmadd_ps(t2, _mm_set1_ps(0.412198583f), _mm_set1_ps(0.577078016f));
    return _mm_fmadd_ps(t, _mm_fmadd_ps(t4, hi, lo), _mm_cvtepi32_ps(e));
}

__forceinline __m128 exp2x4(__m128 y) {
    y = _mm_min_ps(_mm_max_ps(y, _mm_set1_ps(-126.f)), _mm_set1_ps(127.f));
    const __m128i i = _mm_cvtps_epi32(y);
    const __m128 f = _mm_sub_ps(y, _mm_cvtepi32_ps(i)), f2 = _mm_mul_ps(f, f), f4 = _mm_mul_ps(f2, f2);
    const __m128 p01 = _mm_fmadd_ps(f, _mm_set1_ps(0.693147181f), _mm_set1_ps(1.f));
    const __m128 p23 = _mm_fmadd_ps(f, _mm_set1_ps(0.0555041087f), _mm_set1_ps(0.240226507f));
    const __m128 p45 = _mm_fmadd_ps(f, _mm_set1_ps(0.00133335581f), _mm_set1_ps(0.00961812911f));
    const __m128 p67 = _mm_fmadd_ps(f, _mm_set1_ps(1.52527338e-5f), _mm_set1_ps(1.54035304e-4f));
    const __m128 p = _mm_fmadd_ps(f4, _mm_fmadd_ps(f2, p67, p45), _mm_fmadd_ps(f2, p23, p01));
    return _mm_mul_ps(p, _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(i, _mm_set1_epi32(127)), 23)));
}

__forceinline bool encodingX4(const RGBA& c, Encoding e, bool decoding, RGBA& out) {
    const __m128 signBit = _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>(0x80000000u)));
    const __m128 v = _mm_blend_ps(_mm_loadu_ps(&c.r), _mm_setzero_ps(), 8);
    const __m128 a = _mm_andnot_ps(signBit, v);
    if (_mm_movemask_ps(_mm_cmple_ps(a, _mm_set1_ps(3.4e38f))) != 0xF) return false;
    const __m128 tiny = _mm_set1_ps(1.17549435e-38f);
    __m128 r;
    if (e == Encoding::Gamma24) {
        const __m128 s = _mm_max_ps(a, tiny);
        if (decoding) r = _mm_mul_ps(_mm_mul_ps(a, a), exp2x4(_mm_mul_ps(_mm_set1_ps(0.4f), log2x4(s))));
        else r = _mm_and_ps(exp2x4(_mm_mul_ps(log2x4(s), _mm_set1_ps(1.f / 2.4f))), _mm_cmpge_ps(a, tiny));
        r = _mm_or_ps(r, _mm_and_ps(v, signBit));
    } else if (e == Encoding::DaVinciIntermediate) {
        if (decoding) {
            const __m128 y = _mm_sub_ps(_mm_div_ps(v, _mm_set1_ps(0.07329248f)), _mm_set1_ps(7.0f)), lin = _mm_cmple_ps(v, _mm_set1_ps(0.02740668f));
            if (_mm_movemask_ps(_mm_andnot_ps(lin, _mm_cmpgt_ps(y, _mm_set1_ps(127.f))))) return false;
            r = _mm_blendv_ps(_mm_sub_ps(exp2x4(y), _mm_set1_ps(0.0075f)), _mm_mul_ps(v, _mm_set1_ps(1.f / 10.44426855f)), lin);
        } else {
            const __m128 lin = _mm_cmple_ps(v, _mm_set1_ps(0.00262409f));
            const __m128 lg = log2x4(_mm_max_ps(_mm_add_ps(v, _mm_set1_ps(0.0075f)), tiny));
            r = _mm_blendv_ps(_mm_mul_ps(_mm_add_ps(lg, _mm_set1_ps(7.0f)), _mm_set1_ps(0.07329248f)), _mm_mul_ps(v, _mm_set1_ps(10.44426855f)), lin);
        }
    } else {
        if (decoding) {
            const __m128 y = _mm_sub_ps(_mm_mul_ps(v, _mm_set1_ps(17.52f)), _mm_set1_ps(9.72f)), lin = _mm_cmple_ps(v, _mm_set1_ps(0.155251141552511f));
            if (_mm_movemask_ps(_mm_andnot_ps(lin, _mm_cmpgt_ps(y, _mm_set1_ps(127.f))))) return false;
            r = _mm_blendv_ps(exp2x4(y), _mm_mul_ps(_mm_sub_ps(v, _mm_set1_ps(0.0729055341958355f)), _mm_set1_ps(1.f / 10.5402377416545f)), lin);
        } else {
            const __m128 lin = _mm_cmple_ps(v, _mm_set1_ps(0.0078125f));
            r = _mm_blendv_ps(_mm_div_ps(_mm_add_ps(log2x4(_mm_max_ps(v, tiny)), _mm_set1_ps(9.72f)), _mm_set1_ps(17.52f)),
                              _mm_add_ps(_mm_mul_ps(_mm_set1_ps(10.5402377416545f), v), _mm_set1_ps(0.0729055341958355f)), lin);
        }
    }
    _mm_storeu_ps(&out.r, r);
    out.a = c.a;
    return true;
}

__forceinline __m256 log2x8(__m256 x) {
    const __m256i b = _mm256_castps_si256(x);
    __m256i e = _mm256_sub_epi32(_mm256_srli_epi32(b, 23), _mm256_set1_epi32(127));
    __m256 m = _mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(b, _mm256_set1_epi32(0x007fffff)), _mm256_set1_epi32(0x3f800000)));
    const __m256 big = _mm256_cmp_ps(m, _mm256_set1_ps(1.41421356f), _CMP_GT_OQ);
    m = _mm256_blendv_ps(m, _mm256_mul_ps(m, _mm256_set1_ps(0.5f)), big);
    e = _mm256_sub_epi32(e, _mm256_castps_si256(big));
    const __m256 one = _mm256_set1_ps(1.f);
    const __m256 t = _mm256_div_ps(_mm256_sub_ps(m, one), _mm256_add_ps(m, one)), t2 = _mm256_mul_ps(t, t), t4 = _mm256_mul_ps(t2, t2);
    const __m256 lo = _mm256_fmadd_ps(t2, _mm256_set1_ps(0.961796694f), _mm256_set1_ps(2.88539008f));
    const __m256 hi = _mm256_fmadd_ps(t2, _mm256_set1_ps(0.412198583f), _mm256_set1_ps(0.577078016f));
    return _mm256_fmadd_ps(t, _mm256_fmadd_ps(t4, hi, lo), _mm256_cvtepi32_ps(e));
}

__forceinline __m256 exp2x8(__m256 y) {
    y = _mm256_min_ps(_mm256_max_ps(y, _mm256_set1_ps(-126.f)), _mm256_set1_ps(127.f));
    const __m256i i = _mm256_cvtps_epi32(y);
    const __m256 f = _mm256_sub_ps(y, _mm256_cvtepi32_ps(i)), f2 = _mm256_mul_ps(f, f), f4 = _mm256_mul_ps(f2, f2);
    const __m256 p01 = _mm256_fmadd_ps(f, _mm256_set1_ps(0.693147181f), _mm256_set1_ps(1.f));
    const __m256 p23 = _mm256_fmadd_ps(f, _mm256_set1_ps(0.0555041087f), _mm256_set1_ps(0.240226507f));
    const __m256 p45 = _mm256_fmadd_ps(f, _mm256_set1_ps(0.00133335581f), _mm256_set1_ps(0.00961812911f));
    const __m256 p67 = _mm256_fmadd_ps(f, _mm256_set1_ps(1.52527338e-5f), _mm256_set1_ps(1.54035304e-4f));
    const __m256 p = _mm256_fmadd_ps(f4, _mm256_fmadd_ps(f2, p67, p45), _mm256_fmadd_ps(f2, p23, p01));
    return _mm256_mul_ps(p, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(i, _mm256_set1_epi32(127)), 23)));
}

__forceinline bool encodingX8(const RGBA* c, Encoding e, bool decoding, RGBA* out) {
    const __m256 signBit = _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(0x80000000u)));
    const __m256 src = _mm256_loadu_ps(&c->r);
    const __m256 v = _mm256_blend_ps(src, _mm256_setzero_ps(), 0x88);
    const __m256 a = _mm256_andnot_ps(signBit, v);
    if (_mm256_movemask_ps(_mm256_cmp_ps(a, _mm256_set1_ps(3.4e38f), _CMP_LE_OQ)) != 0xFF) return false;
    const __m256 tiny = _mm256_set1_ps(1.17549435e-38f);
    __m256 r;
    if (e == Encoding::Gamma24) {
        const __m256 s = _mm256_max_ps(a, tiny);
        if (decoding) r = _mm256_mul_ps(_mm256_mul_ps(a, a), exp2x8(_mm256_mul_ps(_mm256_set1_ps(0.4f), log2x8(s))));
        else r = _mm256_and_ps(exp2x8(_mm256_mul_ps(log2x8(s), _mm256_set1_ps(1.f / 2.4f))), _mm256_cmp_ps(a, tiny, _CMP_GE_OQ));
        r = _mm256_or_ps(r, _mm256_and_ps(v, signBit));
    } else if (e == Encoding::DaVinciIntermediate) {
        if (decoding) {
            const __m256 y = _mm256_sub_ps(_mm256_div_ps(v, _mm256_set1_ps(0.07329248f)), _mm256_set1_ps(7.0f)), lin = _mm256_cmp_ps(v, _mm256_set1_ps(0.02740668f), _CMP_LE_OQ);
            if (_mm256_movemask_ps(_mm256_andnot_ps(lin, _mm256_cmp_ps(y, _mm256_set1_ps(127.f), _CMP_GT_OQ)))) return false;
            r = _mm256_blendv_ps(_mm256_sub_ps(exp2x8(y), _mm256_set1_ps(0.0075f)), _mm256_mul_ps(v, _mm256_set1_ps(1.f / 10.44426855f)), lin);
        } else {
            const __m256 lin = _mm256_cmp_ps(v, _mm256_set1_ps(0.00262409f), _CMP_LE_OQ);
            const __m256 lg = log2x8(_mm256_max_ps(_mm256_add_ps(v, _mm256_set1_ps(0.0075f)), tiny));
            r = _mm256_blendv_ps(_mm256_mul_ps(_mm256_add_ps(lg, _mm256_set1_ps(7.0f)), _mm256_set1_ps(0.07329248f)), _mm256_mul_ps(v, _mm256_set1_ps(10.44426855f)), lin);
        }
    } else {
        if (decoding) {
            const __m256 y = _mm256_sub_ps(_mm256_mul_ps(v, _mm256_set1_ps(17.52f)), _mm256_set1_ps(9.72f)), lin = _mm256_cmp_ps(v, _mm256_set1_ps(0.155251141552511f), _CMP_LE_OQ);
            if (_mm256_movemask_ps(_mm256_andnot_ps(lin, _mm256_cmp_ps(y, _mm256_set1_ps(127.f), _CMP_GT_OQ)))) return false;
            r = _mm256_blendv_ps(exp2x8(y), _mm256_mul_ps(_mm256_sub_ps(v, _mm256_set1_ps(0.0729055341958355f)), _mm256_set1_ps(1.f / 10.5402377416545f)), lin);
        } else {
            const __m256 lin = _mm256_cmp_ps(v, _mm256_set1_ps(0.0078125f), _CMP_LE_OQ);
            r = _mm256_blendv_ps(_mm256_div_ps(_mm256_add_ps(log2x8(_mm256_max_ps(v, tiny)), _mm256_set1_ps(9.72f)), _mm256_set1_ps(17.52f)),
                                 _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(10.5402377416545f), v), _mm256_set1_ps(0.0729055341958355f)), lin);
        }
    }
    _mm256_storeu_ps(&out->r, _mm256_blend_ps(r, src, 0x88));
    return true;
}
#endif

inline RGBA decode(const RGBA& c, Encoding e) {
    if (e == Encoding::Linear) return c;
#if defined(__AVX2__)
    RGBA o;
    if (encodingX4(c, e, true, o)) return o;
#endif
    return {decodeValue(c.r, e), decodeValue(c.g, e), decodeValue(c.b, e), c.a};
}
inline RGBA encode(const RGBA& c, Encoding e) {
    if (e == Encoding::Linear) return c;
#if defined(__AVX2__)
    RGBA o;
    if (encodingX4(c, e, false, o)) return o;
#endif
    return {encodeValue(c.r, e), encodeValue(c.g, e), encodeValue(c.b, e), c.a};
}

inline void encodingRow(const RGBA* src, RGBA* dst, int n, Encoding e, bool decoding) {
    if (e == Encoding::Linear) {
        if (src != dst) std::memmove(dst, src, static_cast<size_t>(n) * sizeof(RGBA));
        return;
    }
    int i = 0;
#if defined(__AVX2__)
    for (; i + 1 < n; i += 2) {
        if (encodingX8(src + i, e, decoding, dst + i)) continue;
        const RGBA p1 = src[i + 1];
        dst[i] = decoding ? decode(src[i], e) : encode(src[i], e);
        dst[i + 1] = decoding ? decode(p1, e) : encode(p1, e);
    }
#endif
    for (; i < n; ++i) dst[i] = decoding ? decode(src[i], e) : encode(src[i], e);
}
inline void decodeRow(const RGBA* src, RGBA* dst, int n, Encoding e) { encodingRow(src, dst, n, e, true); }
inline void encodeRow(const RGBA* src, RGBA* dst, int n, Encoding e) { encodingRow(src, dst, n, e, false); }

inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
inline uint32_t hashCombine(uint32_t a, uint32_t b) { return hash32(a ^ (hash32(b) + 0x9e3779b9U + (a << 6) + (a >> 2))); }
inline float hashFloat(uint32_t seed, int32_t a, int32_t b = 0, int32_t c = 0) {
    uint32_t h = hashCombine(hashCombine(hashCombine(seed, static_cast<uint32_t>(a)), static_cast<uint32_t>(b)), static_cast<uint32_t>(c));
    return static_cast<float>(h >> 8) * (1.0f / 16777216.0f);
}
inline float valueNoise1(uint32_t seed, double t) {
    double fl = std::floor(t);
    int32_t i = static_cast<int32_t>(fl);
    float f = static_cast<float>(t - fl);
    float p0 = hashFloat(seed, i - 1) * 2.f - 1.f, p1 = hashFloat(seed, i) * 2.f - 1.f;
    float p2 = hashFloat(seed, i + 1) * 2.f - 1.f, p3 = hashFloat(seed, i + 2) * 2.f - 1.f;
    float a = -0.5f * p0 + 1.5f * p1 - 1.5f * p2 + 0.5f * p3;
    float b = p0 - 2.5f * p1 + 2.f * p2 - 0.5f * p3;
    float c = -0.5f * p0 + 0.5f * p2;
    return ((a * f + b) * f + c) * f + p1;
}
inline float fbm1(uint32_t seed, double t, int octaves = 3) {
    float sum = 0.f, amp = 1.f, norm = 0.f;
    for (int o = 0; o < octaves; ++o) {
        sum += amp * valueNoise1(hashCombine(seed, static_cast<uint32_t>(o)), t);
        norm += amp; amp *= 0.5f; t *= 2.0;
    }
    return sum / norm;
}
inline float valueNoise2(uint32_t seed, float x, float y) {
    float fx = std::floor(x), fy = std::floor(y);
    int32_t ix = static_cast<int32_t>(fx), iy = static_cast<int32_t>(fy);
    float tx = x - fx, ty = y - fy;
    tx = tx * tx * (3.f - 2.f * tx); ty = ty * ty * (3.f - 2.f * ty);
    float a = hashFloat(seed, ix, iy), b = hashFloat(seed, ix + 1, iy);
    float c = hashFloat(seed, ix, iy + 1), d = hashFloat(seed, ix + 1, iy + 1);
    return lerpf(lerpf(a, b, tx), lerpf(c, d, tx), ty);
}

inline void boxBlurH(const ImageView& src, const ImageView& dst, int r) {
    const Rect& b = src.bounds;
#if defined(__AVX2__)
    if (src.valid() && dst.valid()) {
        const int W = b.w();
        const __m128 inv = _mm_set1_ps(1.f / static_cast<float>(2 * r + 1));
        parallelRows(b.y1, b.y2, [&](int y) {
            const float* s = src.px(b.x1, y);
            float* d = dst.px(b.x1, y);
            const auto ld = [&](int i) { return _mm_loadu_ps(s + 4 * static_cast<std::ptrdiff_t>(i < 0 ? 0 : (i >= W ? W - 1 : i))); };
            __m128 acc = _mm_setzero_ps();
            for (int k = -r; k <= r; ++k) acc = _mm_add_ps(acc, ld(k));
            for (int i = 0; i < W; ++i) {
                _mm_storeu_ps(d + 4 * static_cast<std::ptrdiff_t>(i), _mm_mul_ps(acc, inv));
                acc = _mm_add_ps(acc, _mm_sub_ps(ld(i + r + 1), ld(i - r)));
            }
        });
        return;
    }
#endif
    parallelRows(b.y1, b.y2, [&](int y) {
        RGBA acc;
        float inv = 1.f / static_cast<float>(2 * r + 1);
        for (int k = -r; k <= r; ++k) acc += src.at(b.x1 + k, y);
        for (int x = b.x1; x < b.x2; ++x) {
            dst.store(x, y, acc * inv);
            acc += src.at(x + r + 1, y) - src.at(x - r, y);
        }
    });
}

inline void boxBlurV(const ImageView& src, const ImageView& dst, int r) {
    const Rect& b = src.bounds;
    const int W = b.w();
    const int bands = static_cast<int>(workerCount()) * 4;
    const int bandW = std::max(16, (W + bands - 1) / bands);
    const int nb = (W + bandW - 1) / bandW;
#if defined(__AVX2__)
    if (src.valid() && dst.valid()) {
        const __m128 inv = _mm_set1_ps(1.f / static_cast<float>(2 * r + 1));
        parallelRows(0, nb, [&](int band) {
            const int xa = b.x1 + band * bandW, n = std::min(b.x2, xa + bandW) - xa;
            std::vector<__m128> acc(static_cast<size_t>(n), _mm_setzero_ps());
            const auto row = [&](int yy) { return src.px(xa, yy < b.y1 ? b.y1 : (yy >= b.y2 ? b.y2 - 1 : yy)); };
            for (int k = -r; k <= r; ++k) {
                const float* s = row(b.y1 + k);
                for (int i = 0; i < n; ++i) acc[static_cast<size_t>(i)] = _mm_add_ps(acc[static_cast<size_t>(i)], _mm_loadu_ps(s + 4 * static_cast<std::ptrdiff_t>(i)));
            }
            for (int y = b.y1; y < b.y2; ++y) {
                float* d = dst.px(xa, y);
                const float* in = row(y + r + 1);
                const float* out = row(y - r);
                for (int i = 0; i < n; ++i) {
                    __m128& a = acc[static_cast<size_t>(i)];
                    const std::ptrdiff_t o = 4 * static_cast<std::ptrdiff_t>(i);
                    _mm_storeu_ps(d + o, _mm_mul_ps(a, inv));
                    a = _mm_add_ps(a, _mm_sub_ps(_mm_loadu_ps(in + o), _mm_loadu_ps(out + o)));
                }
            }
        });
        return;
    }
#endif
    parallelRows(0, nb, [&](int band) {
        int xa = b.x1 + band * bandW, xb = std::min(b.x2, xa + bandW);
        std::vector<RGBA> acc(static_cast<size_t>(xb - xa));
        float inv = 1.f / static_cast<float>(2 * r + 1);
        for (int x = xa; x < xb; ++x)
            for (int k = -r; k <= r; ++k) acc[x - xa] += src.at(x, b.y1 + k);
        for (int y = b.y1; y < b.y2; ++y) {
            for (int x = xa; x < xb; ++x) {
                RGBA& a = acc[x - xa];
                dst.store(x, y, a * inv);
                a += src.at(x, y + r + 1) - src.at(x, y - r);
            }
        }
    });
}

inline void gaussianBoxRadii(float sigma, int radii[3]) {
    sigma = std::min(sigma, 2000.f);
    const int n = 3;
    float wIdeal = std::sqrt((12.f * sigma * sigma / n) + 1.f);
    int wl = static_cast<int>(std::floor(wIdeal));
    if (wl % 2 == 0) wl--;
    int wu = wl + 2;
    float mIdeal = (12.f * sigma * sigma - n * wl * wl - 4.f * n * wl - 3.f * n) / (-4.f * wl - 4.f);
    int m = static_cast<int>(std::round(mIdeal));
    for (int i = 0; i < n; ++i) radii[i] = ((i < m ? wl : wu) - 1) / 2;
}

inline void gaussianBlur(OwnedImage& img, float sigma) {
    if (!img.view.valid() || !(sigma > 0.35f)) return;
    int radii[3];
    gaussianBoxRadii(sigma, radii);
    OwnedImage tmp(img.view.bounds);
    for (int r : radii) {
        if (r < 1) continue;
        boxBlurH(img.view, tmp.view, r);
        boxBlurV(tmp.view, img.view, r);
    }
}

inline void gaussianBlurPlane(float* data, int w, int h, float sigma) {
    if (!data || w < 1 || h < 1 || !(sigma > 0.35f)) return;
    int radii[3];
    gaussianBoxRadii(sigma, radii);
    PixelBuffer tmp(static_cast<size_t>(w) * h);
    const int bandW = 64, nb = (w + bandW - 1) / bandW;
    for (int r : radii) {
        if (r < 1) continue;
        const float inv = 1.f / static_cast<float>(2 * r + 1);
        parallelRows(0, h, [&](int y) {
            const float* s = data + static_cast<size_t>(y) * w;
            float* d = tmp.data() + static_cast<size_t>(y) * w;
            const auto at = [&](int i) { return s[i < 0 ? 0 : (i >= w ? w - 1 : i)]; };
            float acc = 0.f;
            for (int k = -r; k <= r; ++k) acc += at(k);
            for (int i = 0; i < w; ++i) {
                d[i] = acc * inv;
                acc += at(i + r + 1) - at(i - r);
            }
        });
        parallelRows(0, nb, [&](int band) {
            const int xa = band * bandW, n = std::min(w, xa + bandW) - xa;
            float acc[bandW] = {};
            const auto row = [&](int yy) { return tmp.data() + static_cast<size_t>(yy < 0 ? 0 : (yy >= h ? h - 1 : yy)) * w + xa; };
            for (int k = -r; k <= r; ++k) {
                const float* s = row(k);
                for (int i = 0; i < n; ++i) acc[i] += s[i];
            }
            for (int y = 0; y < h; ++y) {
                float* d = data + static_cast<size_t>(y) * w + xa;
                const float* in = row(y + r + 1);
                const float* out = row(y - r);
                for (int i = 0; i < n; ++i) {
                    d[i] = acc[i] * inv;
                    acc[i] += in[i] - out[i];
                }
            }
        });
    }
}

struct Affine {
    double a = 1, b = 0, c = 0, d = 1, tx = 0, ty = 0;
    void apply(double x, double y, double& ox, double& oy) const {
        ox = a * x + b * y + tx;
        oy = c * x + d * y + ty;
    }
    static Affine translate(double x, double y) { Affine m; m.tx = x; m.ty = y; return m; }
    static Affine scale(double s) { Affine m; m.a = s; m.d = s; return m; }
    static Affine scale(double sx, double sy) { Affine m; m.a = sx; m.d = sy; return m; }
    static Affine rotate(double rad) {
        Affine m; double cs = std::cos(rad), sn = std::sin(rad);
        m.a = cs; m.b = -sn; m.c = sn; m.d = cs; return m;
    }
    Affine operator*(const Affine& o) const {
        Affine m;
        m.a = a * o.a + b * o.c; m.b = a * o.b + b * o.d;
        m.c = c * o.a + d * o.c; m.d = c * o.b + d * o.d;
        m.tx = a * o.tx + b * o.ty + tx; m.ty = c * o.tx + d * o.ty + ty;
        return m;
    }
    Affine inverse() const {
        double det = a * d - b * c;
        if (std::fabs(det) < 1e-12) return Affine{};
        Affine m;
        m.a = d / det; m.b = -b / det; m.c = -c / det; m.d = a / det;
        m.tx = -(m.a * tx + m.b * ty); m.ty = -(m.c * tx + m.d * ty);
        return m;
    }
    static Affine about(double cx, double cy, const Affine& m) { return translate(cx, cy) * m * translate(-cx, -cy); }
};

inline RGBA sampleAffine(const ImageView& src, const Affine& dstToSrc, double x, double y, SampleEdge e) {
    double sx, sy;
    dstToSrc.apply(x + 0.5, y + 0.5, sx, sy);
    return src.bilinear(static_cast<float>(sx), static_cast<float>(sy), e);
}

}
