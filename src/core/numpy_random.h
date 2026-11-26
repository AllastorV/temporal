#pragma once
#include <cstdint>
#include <vector>

namespace tp {

class NumpyPcg64 {
    struct U128 { uint64_t hi = 0, lo = 0; };
    static U128 mul64(uint64_t a, uint64_t b) {
        const uint64_t a0 = a & 0xffffffffu, a1 = a >> 32, b0 = b & 0xffffffffu, b1 = b >> 32;
        const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
        const uint64_t mid = (p00 >> 32) + (p01 & 0xffffffffu) + (p10 & 0xffffffffu);
        return {p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32), (mid << 32) | (p00 & 0xffffffffu)};
    }
    static U128 mul(U128 a, U128 b) {
        U128 r = mul64(a.lo, b.lo);
        r.hi += a.hi * b.lo + a.lo * b.hi;
        return r;
    }
    static U128 add(U128 a, U128 b) {
        U128 r{a.hi + b.hi, a.lo + b.lo};
        if (r.lo < a.lo) ++r.hi;
        return r;
    }
    U128 state, inc;
    bool hasHalf = false;
    uint32_t half = 0;
    void step() { state = add(mul(state, U128{2549297995355413924ull, 4865540595714422341ull}), inc); }

public:
    explicit NumpyPcg64(uint32_t seed) : NumpyPcg64(std::vector<uint32_t>{seed}) {}
    explicit NumpyPcg64(const std::vector<uint32_t>& entropy) {
        uint32_t hashA = 0x43b0d7e5u;
        const auto hashmix = [&](uint32_t v) {
            v ^= hashA;
            hashA *= 0x931e8875u;
            v *= hashA;
            return v ^ (v >> 16);
        };
        const auto mix = [](uint32_t x, uint32_t y) {
            const uint32_t r = 0xca01f9ddu * x - 0x4973f715u * y;
            return r ^ (r >> 16);
        };
        uint32_t pool[4];
        for (int i = 0; i < 4; ++i) pool[i] = hashmix(i < int(entropy.size()) ? entropy[i] : 0u);
        for (int s = 0; s < 4; ++s)
            for (int d = 0; d < 4; ++d)
                if (s != d) pool[d] = mix(pool[d], hashmix(pool[s]));
        for (size_t s = 4; s < entropy.size(); ++s)
            for (int d = 0; d < 4; ++d) pool[d] = mix(pool[d], hashmix(entropy[s]));
        uint32_t hashB = 0x8b51f9ddu, words[8];
        for (int i = 0; i < 8; ++i) {
            uint32_t v = pool[i % 4] ^ hashB;
            hashB *= 0x58f38dedu;
            v *= hashB;
            words[i] = v ^ (v >> 16);
        }
        uint64_t w[4];
        for (int i = 0; i < 4; ++i) w[i] = uint64_t(words[2 * i]) | (uint64_t(words[2 * i + 1]) << 32);
        inc = {(w[2] << 1) | (w[3] >> 63), (w[3] << 1) | 1u};
        step();
        state = add(state, U128{w[0], w[1]});
        step();
    }
    uint64_t next64() {
        step();
        const uint64_t v = state.hi ^ state.lo;
        const unsigned rot = unsigned(state.hi >> 58);
        return (v >> rot) | (v << ((64u - rot) & 63u));
    }
    uint32_t next32() {
        if (hasHalf) { hasHalf = false; return half; }
        const uint64_t n = next64();
        hasHalf = true;
        half = uint32_t(n >> 32);
        return uint32_t(n);
    }
    float nextFloat() { return float(next32() >> 8) * (1.0f / 16777216.0f); }
    double nextDouble() { return double(next64() >> 11) * (1.0 / 9007199254740992.0); }
    double uniform(double low, double high) { return low + (high - low) * nextDouble(); }
    uint32_t boundedLemire(uint32_t range) {
        const uint32_t excl = range + 1;
        uint64_t m = uint64_t(next32()) * excl;
        uint32_t leftover = uint32_t(m);
        if (leftover < excl) {
            const uint32_t threshold = (0xffffffffu - range) % excl;
            while (leftover < threshold) {
                m = uint64_t(next32()) * excl;
                leftover = uint32_t(m);
            }
        }
        return uint32_t(m >> 32);
    }
};

inline std::vector<float> numpyRandomFloat32(uint32_t seed, size_t count) {
    NumpyPcg64 rng(seed);
    std::vector<float> out(count);
    for (float& v : out) v = rng.nextFloat();
    return out;
}

}
