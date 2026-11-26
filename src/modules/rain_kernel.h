#pragma once
#include "print_lab_kernel.h"

namespace tp {

PL_HD inline float rainHash(uint32_t v) {
    v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16;
    return static_cast<float>(v >> 8) * (1.f / 16777216.f);
}

PL_HD inline uint32_t rainKey(uint32_t a, uint32_t b) { return a * 0x9e3779b1u ^ (b + 0x7f4a7c15u + (a << 6) + (a >> 2)); }

PL_HD inline float rainSmooth(float a, float b, float x) {
    const float t = plClamp((x - a) / (b - a), 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
}

PL_HD inline float rainLattice(int a, int b, uint32_t seed) {
    return rainHash(rainKey(rainKey(seed, static_cast<uint32_t>(a)), static_cast<uint32_t>(b)));
}

PL_HD inline float rainNoise2(float x, float y, uint32_t seed) {
    const float fx = floorf(x), fy = floorf(y), tx = x - fx, ty = y - fy;
    const int ix = static_cast<int>(fx), iy = static_cast<int>(fy);
    const float sx = tx * tx * (3.f - 2.f * tx), sy = ty * ty * (3.f - 2.f * ty);
    return plLerp(plLerp(rainLattice(ix, iy, seed), rainLattice(ix + 1, iy, seed), sx),
                  plLerp(rainLattice(ix, iy + 1, seed), rainLattice(ix + 1, iy + 1, seed), sx), sy);
}

struct RainDrop {
    static constexpr int kHarmonics = 8, kCuts = 2, kLobes = 3;
    uint32_t id = 0;
    float x = 0, y = 0, r = 0, stretch = 1, dirX = 0, dirY = -1;
    float aspect = 1, orientCos = 1, orientSin = 0, sag = 0, flat = 0, tail = 0;
    float ha[kHarmonics] = {}, hb[kHarmonics] = {};
    int cuts = 0;
    float cutX[kCuts] = {}, cutY[kCuts] = {}, cutD[kCuts] = {};
    int lobes = 1;
    float lobeX[kLobes] = {0, 0, 0}, lobeY[kLobes] = {0, 0, 0}, lobeR[kLobes] = {1, 0, 0};
    double birth = 0, death = 1e30;
    float full = 0;
    bool bead = false, willRun = false;
    int runner = -1;
};

constexpr int kRainHarmonicOrder[RainDrop::kHarmonics] = {2, 3, 4, 5, 6, 7, 11, 13};

PL_HD inline int rainHarmonicOrder(int j) { return j < 6 ? j + 2 : (j == 6 ? 11 : 13); }

struct RainWipe {
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0, r = 0, clear = 0;
    double pass = 0, death = 1e30;
    int runner = -1;
};

PL_HD inline float rainLobeHeight(const RainDrop& d, int lobe, float along, float across) {
    const float rl = d.r * d.lobeR[lobe];
    const float px = along - d.lobeX[lobe] * d.r, py = across - d.lobeY[lobe] * d.r;
    const float rr = px * px + py * py;
    if (rr >= 3.2f * rl * rl || rl <= 0.f) return 0.f;
    const float ex = (px * d.orientCos + py * d.orientSin) / sqrtf(d.aspect);
    const float ey = (-px * d.orientSin + py * d.orientCos) * sqrtf(d.aspect);
    const float rho = sqrtf(ex * ex + ey * ey);
    if (rho <= 1e-9f) return 0.3f * rl;
    const float c1 = ex / rho, s1 = ey / rho;
    float zc = c1, zs = s1, wave = 0.f;
    int j = 0;
    for (int k = 2; k <= 13 && j < RainDrop::kHarmonics; ++k) {
        const float nc = zc * c1 - zs * s1;
        zs = zc * s1 + zs * c1;
        zc = nc;
        if (k == rainHarmonicOrder(j)) {
            wave += d.ha[j] * zc + d.hb[j] * zs;
            ++j;
        }
    }
    const float g = px / sqrtf(rr);
    const float down = plMax(0.f, g), up = plMax(0.f, -g);
    const float up2 = up * up, up4 = up2 * up2;
    const float edge = rl * (1.f + wave) * (1.f + d.sag * down * down) * (1.f - d.flat * up4) * (1.f + d.tail * up4 * up2);
    float q = rho / plMax(1e-9f, edge);
    for (int i = 0; i < d.cuts; ++i) {
        const float cut = (px * d.cutX[i] + py * d.cutY[i]) / (d.cutD[i] * rl), gap = plMax(0.f, 0.12f - fabsf(q - cut));
        q = plMax(q, cut) + gap * gap / 0.48f;
    }
    return q < 1.f ? 0.3f * rl * sqrtf(1.f - q * q * q * q) : 0.f;
}

PL_HD inline float rainDropHeight(const RainDrop& d, float u, float v) {
    const float ox = u - d.x, oy = v - d.y;
    const float along = (ox * d.dirX + oy * d.dirY) / d.stretch, across = -ox * d.dirY + oy * d.dirX;
    float h = 0.f;
    for (int i = 0; i < d.lobes; ++i) {
        const float hi = rainLobeHeight(d, i, along, across);
        if (hi <= 0.f) continue;
        const float blend = 0.1f * d.r, gap = plMax(0.f, blend - fabsf(hi - h));
        h = plMax(h, hi) + (h > 0.f ? gap * gap / (4.f * blend) : 0.f);
    }
    return h;
}

struct RainSplat {
    const RainDrop* drops = nullptr;
    const RainWipe* wipes = nullptr;
    const int *dropStart = nullptr, *dropList = nullptr, *wipeStart = nullptr, *wipeList = nullptr;
    int dropCount = 0, wipeCount = 0, dropListCount = 0, wipeListCount = 0, bands = 0, band = 32;
    float fx1 = 0, fy1 = 0, fh = 1;
    int rx1 = 0, ry1 = 0, rx2 = 0, ry2 = 0;
    PL_HD float reach(const RainDrop& d) const { return 2.4f * d.r * d.stretch * fh + 2.f; }
};

PL_HD inline void rainAddDrop(const RainDrop& d, float h, float cx, float cy, float fh, float& height, float& wsum, float& wcx, float& wcy, float& wrad) {
    const float wgt = h * h;
    wsum += wgt; wcx += wgt * cx; wcy += wgt * cy; wrad += wgt * d.r * fh;
    const float blend = 0.22f * d.r, gap = plMax(0.f, blend - fabsf(h - height));
    height = plMax(height, h) + (height > 0.f ? gap * gap / (4.f * blend) : 0.f);
}

PL_HD inline float rainWipeClear(const RainWipe& w, float ax, float ay, float bx, float by, float rad, int x, int y) {
    const float ex = bx - ax, ey = by - ay, len2 = plMax(1e-6f, ex * ex + ey * ey);
    const float qx = x + 0.5f - ax, qy = y + 0.5f - ay;
    const float t = plClamp((qx * ex + qy * ey) / len2, 0.f, 1.f);
    const float ddx = qx - t * ex, ddy = qy - t * ey;
    return w.clear * (1.f - rainSmooth(0.85f * rad, rad, sqrtf(ddx * ddx + ddy * ddy)));
}

PL_HD inline void rainWipeSpan(const RainSplat& S, const RainWipe& w, float& ax, float& ay, float& bx, float& by, float& rad, int& x0, int& x1) {
    ax = S.fx1 + w.x0 * S.fh; ay = S.fy1 + w.y0 * S.fh;
    bx = S.fx1 + w.x1 * S.fh; by = S.fy1 + w.y1 * S.fh;
    rad = w.r * S.fh;
    const int lo = static_cast<int>(plMin(ax, bx) - rad - 1), hi = static_cast<int>(plMax(ax, bx) + rad) + 2;
    x0 = lo > S.rx1 ? lo : S.rx1;
    x1 = hi < S.rx2 ? hi : S.rx2;
}

PL_HD inline void rainSplatPixel(const RainSplat& S, int x, int y, float& height, float& wsum, float& wcx, float& wcy, float& wrad, float& wiped) {
    height = wsum = wcx = wcy = wrad = wiped = 0.f;
    const int b = (y - S.ry1) / S.band;
    const float v = (y + 0.5f - S.fy1) / S.fh, u = (x + 0.5f - S.fx1) / S.fh;
    for (int k = S.dropStart[b]; k < S.dropStart[b + 1]; ++k) {
        const RainDrop& d = S.drops[S.dropList[k]];
        const float cx = S.fx1 + d.x * S.fh, cy = S.fy1 + d.y * S.fh, reach = S.reach(d);
        if (fabsf(y + 0.5f - cy) > reach) continue;
        const int lo = static_cast<int>(cx - reach), hi = static_cast<int>(cx + reach) + 1;
        if (x < lo || x >= hi) continue;
        const float h = rainDropHeight(d, u, v);
        if (h <= 0.f) continue;
        rainAddDrop(d, h, cx, cy, S.fh, height, wsum, wcx, wcy, wrad);
    }
    for (int k = S.wipeStart[b]; k < S.wipeStart[b + 1]; ++k) {
        const RainWipe& w = S.wipes[S.wipeList[k]];
        float ax, ay, bx, by, rad;
        int x0, x1;
        rainWipeSpan(S, w, ax, ay, bx, by, rad, x0, x1);
        if (x < x0 || x >= x1) continue;
        const float clear = rainWipeClear(w, ax, ay, bx, by, rad, x, y);
        if (clear > wiped) wiped = clear;
    }
}

struct RainShade {
    float fh = 1, mist = 0, rim = 0, highlight = 0, caustic = 0, magnify = 1, dot = 1;
    float lx = 0, ly = 0, lz = 1, hz0 = 2, hn = 1, lxy = 1;
    uint32_t seed = 1;
    int matte = 0, blurDrops = 0;
};

struct RainMaps {
    const float *height = nullptr, *wiped = nullptr, *wsum = nullptr, *wcx = nullptr, *wcy = nullptr, *wrad = nullptr;
    const float *bead = nullptr, *glint = nullptr;
    int x1 = 0, y1 = 0, w = 0, h = 0;
    PL_HD float heightAt(int x, int y) const {
        x = x < x1 ? x1 : (x > x1 + w - 1 ? x1 + w - 1 : x);
        y = y < y1 ? y1 : (y > y1 + h - 1 ? y1 + h - 1 : y);
        return height[static_cast<size_t>(y - y1) * w + (x - x1)];
    }
};

PL_HD inline PL4 rainMix(const PL4& a, const PL4& b, float t) {
    return PL4{plLerp(a.r, b.r, t), plLerp(a.g, b.g, t), plLerp(a.b, b.b, t), plLerp(a.a, b.a, t)};
}

template <class BackAt, class FogAt, class InnerSample, class BackSample>
PL_HD inline PL4 rainShadePixel(const RainShade& P, const RainMaps& M, int x, int y, const BackAt& backAt, const FogAt& fogAt,
                                const InnerSample& inner, const BackSample& backSample, PL4& glass) {
    const size_t idx = static_cast<size_t>(y - M.y1) * M.w + (x - M.x1);
    const float h = M.height[idx], wn = M.wsum[idx];
    const float rpx = wn > 0.f ? M.wrad[idx] / wn : 1.f;
    const float water = rainSmooth(0.f, plMax(0.0002f, 0.08f * rpx / P.fh), h);
    const float level = plMin(1.f, h / plMax(1e-9f, 0.3f * rpx / P.fh));
    PL4 bg = backAt(x, y);
    if (P.mist > 0) {
        const float density = 0.6f + 0.4f * rainNoise2(x / (40.f * P.dot), y / (40.f * P.dot), P.seed + 7u);
        const float keep = P.mist * (1.f - M.wiped[idx]);
        const float bead = M.bead[idx], glint = M.glint[idx];
        const PL4 f = fogAt(x, y);
        const PL4 veil{f.r * 0.86f + 0.07f, f.g * 0.86f + 0.07f, f.b * 0.86f + 0.075f, f.a};
        bg = rainMix(bg, veil, plMin(1.f, keep * (0.45f + 0.35f * density)));
        const float speck = keep * bead, shine = 0.12f * glint * keep;
        bg = PL4{bg.r * (1.f - 0.1f * speck) + shine * bg.r + 0.03f * shine, bg.g * (1.f - 0.1f * speck) + shine * bg.g + 0.03f * shine,
                 bg.b * (1.f - 0.1f * speck) + shine * bg.b + 0.03f * shine, bg.a};
    }
    glass = bg;
    if (P.matte) return PL4{water, water, water, 1.f};
    PL4 drop{};
    float bnx = 0.f, bny = 0.f;
    if (water > 0 && wn > 0.f) {
        const float cx = M.wcx[idx] / wn, cy = M.wcy[idx] / wn;
        const float gx = (M.heightAt(x + 1, y) - M.heightAt(x - 1, y)) * 0.5f * P.fh;
        const float gy = (M.heightAt(x, y + 1) - M.heightAt(x, y - 1)) * 0.5f * P.fh;
        float nx = -gx, ny = -gy, nz = 1.f;
        const float nn = 1.f / sqrtf(nx * nx + ny * ny + nz * nz);
        nx *= nn; ny *= nn; nz *= nn;
        const float steep = sqrtf(nx * nx + ny * ny);
        bnx = nx;
        bny = ny;
        const float ox = x + 0.5f - cx, oy = y + 0.5f - cy;
        const float sx = cx - P.magnify * ox - 0.6f * P.magnify * rpx * nx;
        const float sy = cy - P.magnify * oy - 0.6f * P.magnify * rpx * ny;
        const PL4 seen = inner(sx, sy);
        const float shade = 1.04f * (1.f - P.rim * (1.f - rainSmooth(0.35f, 0.78f, level)));
        const float away = steep > 1e-3f ? plMax(0.f, -(nx * P.lx + ny * P.ly) / (steep * P.lxy)) : 0.f;
        const float glow = 1.f + P.caustic * 0.8f * away * rainSmooth(0.4f, 0.85f, steep);
        const float spec = P.highlight * powf(plMax(0.f, (nx * P.lx + ny * P.ly + nz * P.hz0) * P.hn), 120.f) * 2.f;
        drop = PL4{seen.r * shade * glow + spec, seen.g * shade * glow + spec, seen.b * shade * glow + spec, seen.a};
    }
    if (P.blurDrops) {
        const float ring = 1.f - rainSmooth(0.3f, 0.8f, level);
        const float edge = water * (0.06f + 0.94f * ring);
        const PL4 bent = backSample(x + 0.5f - 0.5f * rpx * bnx, y + 0.5f - 0.5f * rpx * bny);
        const float lift = 1.f + 0.6f * ring;
        return PL4{plLerp(bent.r, drop.r * lift, ring) * edge, plLerp(bent.g, drop.g * lift, ring) * edge,
                   plLerp(bent.b, drop.b * lift, ring) * edge, edge};
    }
    PL4 o = rainMix(bg, drop, water);
    o.a = bg.a;
    return o;
}

}
