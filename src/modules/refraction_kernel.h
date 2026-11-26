#pragma once
#include "print_lab_kernel.h"

namespace tp {

struct RLParams {
    int mode = 0, facets = 5, enc = 0, useBlur = 0;
    float cs = 1, sn = 0, s = 1, e = 0.01f, curv = 1, hscale = 1, cx = 0, cy = 0, T = 0, dist = 0, diag = 1;
    float nc[3] = {1.5f, 1.5f, 1.5f};
};

PL_HD inline float rlBoxK(float t, float e) { return 1.f - plSmooth(1.f - e, 1.f + e, fabsf(t)); }

PL_HD inline bool rlTraceK(const PLVec& n, float ior, float T, float dist, float& Dx, float& Dy) {
    const float eta = 1.f / ior;
    const float cosi = n.z;
    const float k = plMax(0.f, 1.f - eta * eta * (1.f - cosi * cosi));
    const float a = eta * cosi - sqrtf(k);
    const PLVec d1{a * n.x, a * n.y, -eta + a * n.z};
    const float z1 = plMax(fabsf(d1.z), 0.05f);
    const float cos2 = -d1.z;
    const float k2 = 1.f - ior * ior * (1.f - cos2 * cos2);
    if (k2 < 0.f) {
        Dx = -(T + dist) * d1.x / z1;
        Dy = -(T + dist) * d1.y / z1;
        return false;
    }
    const float b = ior * cos2 - sqrtf(k2);
    const PLVec d2{ior * d1.x, ior * d1.y, ior * d1.z + b};
    const float z2 = plMax(fabsf(d2.z), 0.05f);
    Dx = T * d1.x / z1 + dist * d2.x / z2;
    Dy = T * d1.y / z1 + dist * d2.y / z2;
    return true;
}

PL_HD inline float rlDecode(float v, int e) {
    switch (e) {
    case 1: return v < 0.f ? -powf(-v, 2.4f) : powf(v, 2.4f);
    case 2: return v <= 0.02740668f ? v / 10.44426855f : exp2f(v / 0.07329248f - 7.0f) - 0.0075f;
    case 3: return v <= 0.155251141552511f ? (v - 0.0729055341958355f) / 10.5402377416545f : exp2f(v * 17.52f - 9.72f);
    default: return v;
    }
}

PL_HD inline float rlGeom(const RLParams& P, float px, float py, PLVec& n) {
    const float dx = px - P.cx, dy = py - P.cy;
    const float u = (P.cs * dx + P.sn * dy) / P.s, v = (-P.sn * dx + P.cs * dy) / P.s;
    float mask, hu = 0.f, hv = 0.f;
    if (P.mode == 0) {
        mask = plSmooth(-P.e, P.e, u);
        const float rim = rlBoxK(sqrtf(u * u + v * v), P.e);
        hu = -2.f * u * P.curv * rim;
        hv = -2.f * v * P.curv * rim;
    } else {
        mask = rlBoxK(u, P.e) * rlBoxK(v, P.e);
        if (P.mode == 1) hu = P.curv;
        else if (P.mode == 2) hu = -2.f * plClamp(u, -1.f, 1.f) * P.curv;
        else {
            int i = static_cast<int>(floorf((u + 1.f) * 0.5f * P.facets));
            i = i < 0 ? 0 : (i > P.facets - 1 ? P.facets - 1 : i);
            hu = P.curv * (-1.f + 2.f * static_cast<float>(i) / static_cast<float>(P.facets - 1));
        }
    }
    const float gx = plClamp(P.hscale * (P.cs * hu - P.sn * hv), -1e4f, 1e4f);
    const float gy = plClamp(P.hscale * (P.sn * hu + P.cs * hv), -1e4f, 1e4f);
    const float nl = sqrtf(gx * gx + gy * gy + 1.f);
    n = PLVec{-gx / nl, -gy / nl, 1.f / nl};
    return mask;
}

template <class RawFn, class LinFn>
PL_HD inline PL4 rlPixel(const RLParams& P, int x, int y, const RawFn& rawAt, const LinFn& sampleLin) {
    const float dx = x + 0.5f - P.cx, dy = y + 0.5f - P.cy;
    const float u = (P.cs * dx + P.sn * dy) / P.s, v = (-P.sn * dx + P.cs * dy) / P.s;
    float mask, hu = 0.f, hv = 0.f;
    if (P.mode == 0) {
        mask = plSmooth(-P.e, P.e, u);
        const float rim = rlBoxK(sqrtf(u * u + v * v), P.e);
        hu = -2.f * u * P.curv * rim;
        hv = -2.f * v * P.curv * rim;
    } else {
        mask = rlBoxK(u, P.e) * rlBoxK(v, P.e);
        if (P.mode == 1) hu = P.curv;
        else if (P.mode == 2) hu = -2.f * plClamp(u, -1.f, 1.f) * P.curv;
        else {
            int i = static_cast<int>(floorf((u + 1.f) * 0.5f * P.facets));
            i = i < 0 ? 0 : (i > P.facets - 1 ? P.facets - 1 : i);
            hu = P.curv * (-1.f + 2.f * static_cast<float>(i) / static_cast<float>(P.facets - 1));
        }
    }
    const PL4 s0 = rawAt(x, y);
    if (!(mask > 0.f)) return s0;
    const float gx = plClamp(P.hscale * (P.cs * hu - P.sn * hv), -1e4f, 1e4f);
    const float gy = plClamp(P.hscale * (P.sn * hu + P.cs * hv), -1e4f, 1e4f);
    const float nl = sqrtf(gx * gx + gy * gy + 1.f);
    const PLVec n{-gx / nl, -gy / nl, 1.f / nl};
    float refr[4] = {0.f, 0.f, 0.f, 0.f};
    for (int ch = 0; ch < 3; ++ch) {
        float Dx, Dy;
        bool ok = rlTraceK(n, P.nc[ch], P.T, P.dist, Dx, Dy);
        float dl = sqrtf(Dx * Dx + Dy * Dy);
        if (dl > P.diag) { Dx *= P.diag / dl; Dy *= P.diag / dl; }
        PL4 smp = sampleLin(x + 0.5f + Dx, y + 0.5f + Dy);
        const float dark = ok ? 1.f : 0.35f;
        const float sv = ch == 0 ? smp.r : (ch == 1 ? smp.g : smp.b);
        refr[ch] = sv * dark;
        if (ch == 1) refr[3] = smp.a;
    }
    PL4 l0 = P.enc == 0 ? s0 : PL4{rlDecode(s0.r, P.enc), rlDecode(s0.g, P.enc), rlDecode(s0.b, P.enc), s0.a};
    PL4 o{plLerp(l0.r, refr[0], mask), plLerp(l0.g, refr[1], mask), plLerp(l0.b, refr[2], mask), plLerp(l0.a, refr[3], mask)};
    if (P.enc != 0) { o.r = plEncode(o.r, P.enc); o.g = plEncode(o.g, P.enc); o.b = plEncode(o.b, P.enc); }
    return o;
}

}
