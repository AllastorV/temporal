#pragma once
#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define PL_HD __host__ __device__
#else
#define PL_HD
#endif

namespace tp {

struct PrintLabRGB {
    float r = 0, g = 0, b = 0;
};

struct PL4 {
    float r = 0, g = 0, b = 0, a = 0;
};

struct PLVec { float x, y, z; };

struct PLPlate {
    float cs = 1, sn = 0, ox = 0, oy = 0;
    PrintLabRGB ink;
};

struct PLFold {
    float nx = 0, ny = 1, d0 = 0, depth = 1;
};

enum PrintLabEngine { kMono = 0, kCmyk = 1, kRiso = 2, kToner = 3, kSpot = 4, kCrt = 5, kLcd = 6 };

constexpr int kPrintLabLutN = 1024;

struct PLParams {
    float surfaceX = 0, surfaceY = 0, phase = 0;
    int engine = 0;
    PrintLabRGB lookInk, paperBase, inkA, inkB;
    float aging = 0, mottle = 0, rough = 0, tonerNoise = 0, tonerSoft = 0, cockle = 0, fiber = 0;
    int gutter = 0;
    PLPlate plates[4];
    int nplates = 0;
    PLFold folds[4];
    int nfolds = 0;
    float paperAmt = 1, inkGamma = 1, contrastP = 1, thr = 0.5f, s = 4, w = 1, u1 = 1;
    int PW = 2, PH = 2;
    uint32_t sFiberA = 0, sFiberB = 0, sGrain = 0, sMottle = 0, sStain = 0, sRough = 0, sStreak = 0, sCockle = 0, sFox = 0;
    float creaseBand = 1, creaseCore = 1, cockleS = 1;
    const float* lut = nullptr;
};

struct PLCamera {
    float sth = 0, cth = 1, crr = 1, srr = 0, f = 1, halfH = 1, dof = 0, zFocus = 1;
    PLVec Lw{0, 0, -1};
    PLVec Larea[5];
    float sheen = 0, alpha2 = 0.1f, vig = 0;
    PLVec n0{0, 0, -1}, ea{1, 0, 0}, eb{0, 1, 0};
    float flatDiffuse = 1, screenDiag = 1;
    uint32_t sTable = 0;
    float pageY0 = 0.5f, posX = 0.5f, W = 1, H = 1;
    int Fx1 = 0, Fy1 = 0;
    float ps = 1, scale = 1, paperAmt = 1;
    int enc = 0;
    int maxLod = 0;
};

PL_HD inline float plClamp(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
PL_HD inline float plSat(float v) { return plClamp(v, 0.f, 1.f); }
PL_HD inline float plLerp(float a, float b, float t) { return a + (b - a) * t; }
PL_HD inline float plMax(float a, float b) { return a > b ? a : b; }
PL_HD inline float plMin(float a, float b) { return a < b ? a : b; }
PL_HD inline float plSmooth(float e0, float e1, float x) {
    if (e1 == e0) return x < e0 ? 0.f : 1.f;
    float t = plSat((x - e0) / (e1 - e0));
    return t * t * (3.f - 2.f * t);
}
PL_HD inline bool plFinite(float v) { return v == v && fabsf(v) < 3.0e38f; }
PL_HD inline uint32_t plHash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
PL_HD inline uint32_t plHashCombine(uint32_t a, uint32_t b) { return plHash32(a ^ (plHash32(b) + 0x9e3779b9U + (a << 6) + (a >> 2))); }
PL_HD inline float plHashFloat(uint32_t seed, int32_t a, int32_t b = 0, int32_t c = 0) {
    uint32_t h = plHashCombine(plHashCombine(plHashCombine(seed, static_cast<uint32_t>(a)), static_cast<uint32_t>(b)), static_cast<uint32_t>(c));
    return static_cast<float>(h >> 8) * (1.0f / 16777216.0f);
}
PL_HD inline float plNoise(uint32_t seed, float x, float y) {
    float fx = floorf(x), fy = floorf(y);
    int32_t ix = static_cast<int32_t>(fx), iy = static_cast<int32_t>(fy);
    float tx = x - fx, ty = y - fy;
    tx = tx * tx * (3.f - 2.f * tx); ty = ty * ty * (3.f - 2.f * ty);
    float a = plHashFloat(seed, ix, iy), b = plHashFloat(seed, ix + 1, iy);
    float c = plHashFloat(seed, ix, iy + 1), d = plHashFloat(seed, ix + 1, iy + 1);
    return plLerp(plLerp(a, b, tx), plLerp(c, d, tx), ty);
}
PL_HD inline float plLuma(const PrintLabRGB& c) { return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b; }
PL_HD inline float plCurve(float d, float k) {
    float x = 2.f * plSat(d) - 1.f;
    float y = powf(fabsf(x), 1.f / k);
    return plSat(0.5f + 0.5f * (x < 0.f ? -y : y));
}
PL_HD inline float plDecodeGamma(float v) { return v < 0.f ? -powf(-v, 2.4f) : powf(v, 2.4f); }
PL_HD inline float plEncode(float v, int e) {
    switch (e) {
    case 1: return v < 0.f ? -powf(-v, 1.f / 2.4f) : powf(v, 1.f / 2.4f);
    case 2: return v <= 0.00262409f ? v * 10.44426855f : (log2f(v + 0.0075f) + 7.0f) * 0.07329248f;
    case 3: return v <= 0.0078125f ? 10.5402377416545f * v + 0.0729055341958355f : (log2f(v) + 9.72f) / 17.52f;
    default: return v;
    }
}
PL_HD inline PLVec plNorm(PLVec v) {
    float l = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return l > 1e-8f ? PLVec{v.x / l, v.y / l, v.z / l} : PLVec{0, 0, 1};
}
PL_HD inline float plDot(const PLVec& a, const PLVec& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

PL_HD inline float plHalftone(const float* lut, float fu, float fv, float s, float cov, float w, float rough) {
    if (cov < 0.035f) return 0.f;
    cov += (1.f - cov) * plSmooth(0.82f, 0.96f, cov);
    if (cov >= 1.f) return 1.f;
    float fi = cov * kPrintLabLutN;
    int i0 = static_cast<int>(fi);
    if (i0 > kPrintLabLutN - 1) i0 = kPrintLabLutN - 1;
    float r = s * plLerp(lut[i0], lut[i0 + 1], fi - i0);
    float d = sqrtf(fu * fu + fv * fv) * s;
    return plSmooth(-0.5f * w, 0.5f * w, r - d + rough) * plSat(2.f * r / w);
}

PL_HD inline float plShapeInk(const PLParams& P, float d) { return powf(plCurve(d, P.contrastP), P.inkGamma); }

PL_HD inline float plDensity(const PLParams& P, const PrintLabRGB& t, int plate) {
    if (P.engine == kCmyk) {
        float cc = plShapeInk(P, 1.f - t.r), mm = plShapeInk(P, 1.f - t.g), yy = plShapeInk(P, 1.f - t.b);
        float kk = plMin(cc, plMin(mm, yy)) * 0.9f;
        float v[4] = {cc - kk, mm - kk, yy - kk, kk};
        return plSat(v[plate]);
    }
    return plShapeInk(P, 1.f - plLuma(t));
}

PL_HD inline void plHeightGrad(const PLParams& P, float qx, float qy, float& gx, float& gy, float& crack) {
    gx = 0.f; gy = 0.f; crack = 0.f;
    for (int k = 0; k < P.nfolds; ++k) {
        const PLFold& f = P.folds[k];
        float d = f.nx * qx + f.ny * qy - f.d0;
        float e = expf(-fabsf(d) / P.creaseBand);
        float side = d >= 0.f ? 1.f : -1.f;
        float slope = f.depth * P.paperAmt * (0.55f * e + 0.09f) * side;
        gx += slope * f.nx; gy += slope * f.ny;
        crack = plMax(crack, P.paperAmt * expf(-(d * d) / (P.creaseCore * P.creaseCore)));
    }
    if (P.gutter) gx += -P.paperAmt * 0.9f * expf(-(qx / P.PW) / 0.12f);
    if (P.cockle > 0.f && P.paperAmt > 0.f) {
        float e = 0.35f;
        float cx = qx / P.cockleS, cy = qy / P.cockleS;
        float dx = plNoise(P.sCockle, cx + e, cy) - plNoise(P.sCockle, cx - e, cy);
        float dy = plNoise(P.sCockle, cx, cy + e) - plNoise(P.sCockle, cx, cy - e);
        float amp = P.cockle * P.paperAmt * 0.16f / (2.f * e);
        gx += dx * amp; gy += dy * amp;
    }
    float fb = plNoise(P.sFiberA, qx / (2.5f * P.u1), qy / (2.5f * P.u1)) - 0.5f;
    gx += P.paperAmt * P.fiber * 0.05f * fb;
    gy += P.paperAmt * P.fiber * 0.03f * fb;
}

template <class ToneFn>
PL_HD inline PL4 plPagePixel(const PLParams& P, int i, int j, const ToneFn& toneAt) {
    const float u1 = P.u1;
    const float qx = i + 0.5f, qy = j + 0.5f;
    const float PWf = static_cast<float>(P.PW), PHf = static_cast<float>(P.PH);
    if (P.engine == kCrt || P.engine == kLcd) {
        const float pitch = plMax(2.f, P.s);
        const float row = qy / pitch;
        const float col = qx / pitch;
        const float wobble = P.engine == kCrt ? 0.35f * sinf(row * 0.13f + P.phase * 4.f) : 0.f;
        PrintLabRGB signal = toneAt(qx + wobble * u1, qy);
        int subpixel = static_cast<int>(floorf(col * 3.f)) % 3;
        float scan = P.engine == kCrt ? 0.83f + 0.17f * cosf(6.2831853f * (row + 0.1f * P.phase))
                                      : 0.91f + 0.09f * cosf(6.2831853f * row);
        const float refresh = 0.97f + 0.03f * cosf(6.2831853f * (qy / PHf - P.phase));
        float gain = scan * refresh;
        return {plDecodeGamma(signal.r * gain * (subpixel == 0 ? 1.f : 0.69f)),
                plDecodeGamma(signal.g * gain * (subpixel == 1 ? 1.f : 0.69f)),
                plDecodeGamma(signal.b * gain * (subpixel == 2 ? 1.f : 0.69f)), 1.f};
    }
    const float fx = qx + P.surfaceX, fy = qy + P.surfaceY;
    float fiber = 0.6f * plNoise(P.sFiberA, fx / (2.5f * u1), fy / (2.5f * u1)) + 0.4f * plNoise(P.sFiberB, fx / (24.f * u1), fy / (5.f * u1));
    PrintLabRGB paper = P.paperBase;
    float tm = 1.f + P.paperAmt * P.fiber * 0.16f * (fiber - 0.5f);
    paper = {paper.r * tm, paper.g * tm, paper.b * tm};
    if (P.aging > 0.f && P.paperAmt > 0.f) {
        float st = 0.65f * plNoise(P.sStain, qx / (300.f * u1), qy / (300.f * u1)) + 0.35f * plNoise(P.sStain + 1u, qx / (90.f * u1), qy / (90.f * u1));
        float blot = plSmooth(0.55f, 0.85f, st);
        float tide = plSmooth(0.05f, 0.0f, fabsf(st - 0.55f)) * 0.6f;
        float fox = 0.f;
        const float cs = 40.f * u1;
        const int ci = static_cast<int>(floorf(qx / cs)), cj = static_cast<int>(floorf(qy / cs));
        for (int dj = -1; dj <= 1; ++dj)
            for (int di = -1; di <= 1; ++di) {
                int gi = ci + di, gj = cj + dj;
                if (plHashFloat(P.sFox, gi, gj, 1) > 0.22f) continue;
                float ox = (gi + plHashFloat(P.sFox, gi, gj, 2)) * cs, oy = (gj + plHashFloat(P.sFox, gi, gj, 3)) * cs;
                float dxs = qx - ox, dys = qy - oy;
                float wob = 1.f + 0.35f * (plNoise(P.sFox + 4u, qx / (2.f * u1), qy / (2.f * u1)) - 0.5f);
                float rad = (1.2f + 4.5f * plHashFloat(P.sFox, gi, gj, 4)) * u1 * wob;
                float dd = sqrtf(dxs * dxs + dys * dys);
                fox = plMax(fox, plSmooth(rad, rad * 0.35f, dd) * (0.45f + 0.55f * plHashFloat(P.sFox, gi, gj, 5)));
            }
        fox *= plSmooth(0.3f, 0.7f, plNoise(P.sFox + 2u, qx / (200.f * u1), qy / (200.f * u1)));
        float edgeAge = plSmooth(0.12f, 0.f, plMin(plMin(qx / PWf, 1.f - qx / PWf), plMin(qy / PHf, 1.f - qy / PHf)));
        float a = P.aging * P.paperAmt * plSat(0.22f * blot + 0.18f * tide + 0.45f * fox + 0.2f * edgeAge);
        paper = {paper.r * (1.f - a * 0.55f), paper.g * (1.f - a * 0.8f), paper.b * (1.f - a * 1.25f)};
    }
    float mottle = 1.f - P.mottle * (0.55f * plNoise(P.sMottle, fx / (9.f * u1), qy / (9.f * u1)) + 0.45f * plNoise(P.sMottle + 7u, qx / (60.f * u1), qy / (60.f * u1)) - 0.25f) * 0.5f;
    float rough = P.rough * (plNoise(P.sRough, fx / (1.3f * u1), qy / (1.3f * u1)) - 0.5f) * P.w * 1.6f;
    float gx, gy, crack;
    plHeightGrad(P, qx, qy, gx, gy, crack);

    PrintLabRGB acc = paper;
    if (P.engine == kMono || P.engine == kCmyk || P.engine == kRiso) {
        float band = P.engine == kRiso ? 0.84f + 0.16f * plNoise(P.sStreak, 0.37f, qy / (90.f * u1)) : 1.f;
        float grain = P.engine == kRiso ? plNoise(P.sGrain, qx / (1.1f * u1), qy / (1.1f * u1)) - 0.5f : 0.f;
        for (int k = 0; k < P.nplates; ++k) {
            const PLPlate& pl = P.plates[k];
            float ax = qx - pl.ox, ay = qy - pl.oy;
            float u = (ax * pl.cs + ay * pl.sn) / P.s, v = (-ax * pl.sn + ay * pl.cs) / P.s;
            float cu = floorf(u) + 0.5f, cv = floorf(v) + 0.5f;
            float cx = (cu * pl.cs - cv * pl.sn) * P.s, cy = (cu * pl.sn + cv * pl.cs) * P.s;
            float cov = plDensity(P, toneAt(cx, cy), k);
            cov = plSat(cov + 0.12f * cov * (1.f - cov) * 4.f);
            float ink = plHalftone(P.lut, u - cu, v - cv, P.s, cov, P.w, rough);
            ink *= plSat((mottle + 0.45f * grain) * band);
            ink *= 1.f - 0.7f * crack;
            acc.r *= 1.f - ink * (1.f - pl.ink.r);
            acc.g *= 1.f - ink * (1.f - pl.ink.g);
            acc.b *= 1.f - ink * (1.f - pl.ink.b);
        }
    } else if (P.engine == kToner) {
        const PLPlate& pl = P.plates[0];
        PrintLabRGB tc = toneAt(qx, qy);
        float copierLuma = 0.62f * tc.r + 0.3f * tc.g + 0.08f * tc.b;
        float dark = plShapeInk(P, 1.f - copierLuma);
        auto rn = [&](uint32_t sd, float sc, float ang, float stretch) {
            float ca = cosf(ang), sa = sinf(ang);
            float rx = qx * ca - qy * sa, ry = qx * sa + qy * ca;
            return plNoise(sd, rx / (sc * u1), ry / (sc * stretch * u1));
        };
        float g1 = rn(P.sGrain, 1.5f, 0.52f, 1.f);
        float g2 = rn(P.sGrain + 3u, 4.2f, 1.17f, 1.f);
        float g3 = rn(P.sGrain + 5u, 9.f, 0.08f, 4.5f);
        float G = 0.45f * g1 + 0.35f * g2 + 0.2f * g3;
        float mid = 4.f * dark * (1.f - dark);
        float toner = plSmooth(P.thr - P.tonerSoft, P.thr + P.tonerSoft, dark + P.tonerNoise * (G - 0.5f) * (0.25f + 0.75f * mid));
        toner *= 0.93f + 0.07f * rn(P.sGrain + 7u, 2.6f, 0.9f, 1.f);
        toner *= 0.86f + 0.14f * plNoise(P.sGrain + 9u, qx / (160.f * u1), qy / (160.f * u1));
        float dust = plSmooth(0.9f, 0.96f, rn(P.sGrain + 11u, 1.4f, 0.71f, 1.f)) * 0.55f * P.tonerNoise;
        float haze = 0.05f * P.tonerNoise * rn(P.sGrain + 13u, 40.f, 0.3f, 1.f);
        float streak = plSmooth(0.93f, 1.f, plNoise(P.sStreak, qx / (1.5f * u1), qy / (500.f * u1))) * 0.45f * P.tonerNoise;
        float edgeDist = plMin(plMin(qx, PWf - qx), plMin(qy, PHf - qy));
        float border = expf(-edgeDist / (18.f * u1)) * 0.55f;
        toner = plSat(plMax(plMax(toner, dust), plMax(haze, border)) + streak);
        toner *= 1.f - 0.6f * crack;
        acc.r *= 1.f - toner * (1.f - pl.ink.r);
        acc.g *= 1.f - toner * (1.f - pl.ink.g);
        acc.b *= 1.f - toner * (1.f - pl.ink.b);
    } else {
        for (int k = 0; k < 3; ++k) {
            const PLPlate& pl = P.plates[k];
            PrintLabRGB t = toneAt(qx - pl.ox, qy - pl.oy);
            float l0 = plLuma(t);
            float g = (1.f - plShapeInk(P, 1.f - l0)) - l0;
            PrintLabRGB tt{plSat(t.r + g), plSat(t.g + g), plSat(t.b + g)};
            const PrintLabRGB pal[4] = {P.paperBase, P.inkA, P.inkB, P.lookInk};
            float lt = plLuma(tt);
            float best = 1e9f, second = 1e9f;
            int bi = 0, si = 0;
            for (int q = 0; q < 4; ++q) {
                float lq = plLuma(pal[q]);
                float dl = lt - lq;
                float dr = (tt.r - lt) - (pal[q].r - lq), dg = (tt.g - lt) - (pal[q].g - lq), db = (tt.b - lt) - (pal[q].b - lq);
                float d2 = 2.f * dl * dl + 3.f * (dr * dr + dg * dg + db * db) + rough * 0.002f;
                if (d2 < best) { second = best; si = bi; best = d2; bi = q; }
                else if (d2 < second) { second = d2; si = q; }
            }
            float edge = plSmooth(0.f, 0.03f, second - best);
            float mk = bi == k + 1 ? (0.5f + 0.5f * edge) : ((k < 2 && bi == 3 && si == k + 1) ? 1.f : 0.f);
            mk *= plSat(mottle * 1.05f) * (1.f - 0.7f * crack);
            acc.r = plLerp(acc.r, pl.ink.r, mk);
            acc.g = plLerp(acc.g, pl.ink.g, mk);
            acc.b = plLerp(acc.b, pl.ink.b, mk);
        }
    }
    return PL4{plDecodeGamma(plSat(acc.r)), plDecodeGamma(plSat(acc.g)), plDecodeGamma(plSat(acc.b)), 1.f};
}

template <class SampleFn>
PL_HD inline PL4 plCameraPixel(const PLParams& P, const PLCamera& C, int x, int y, const SampleFn& samplePage) {
    const float W = C.W, H = C.H;
    const int PW = P.PW, PH = P.PH;
    const float ps = C.ps;
    float su = (x + 0.5f - (C.Fx1 + 0.5f * W)) / C.halfH, sv = (y + 0.5f - (C.Fy1 + 0.5f * H)) / C.halfH;
    float u = C.crr * su + C.srr * sv, v = -C.srr * su + C.crr * sv;
    float denom = C.cth - C.sth * v / C.f;
    float tableX = x + 0.5f, tableY = y + 0.5f, edgeD = -1e6f, mag = 1.f;
    PLVec pt{0.f, 0.f, 1.f};
    float qxF = 0.f, qyF = 0.f;
    if (denom > 0.02f) {
        float t = C.cth / denom;
        pt = PLVec{t * u / C.f, t * v / C.f, t};
        float a = pt.x, b = pt.y * C.cth + (pt.z - 1.f) * C.sth;
        float pX = C.posX * W + a * C.halfH;
        float pY = C.pageY0 * H + b * C.halfH;
        tableX = pX; tableY = pY;
        mag = C.f / t;
        float deckle = (plNoise(C.sTable + 5u, pX / 14.f, pY / 14.f) - 0.5f) * 0.9f * C.paperAmt;
        edgeD = plMin(plMin(pX, W - pX), plMin(pY, H - pY)) + deckle;
        qxF = pX * ps; qyF = pY * ps;
    }
    float wood = 0.8f + 0.4f * plNoise(C.sTable, tableX / 180.f, tableY / 180.f) * (0.8f + 0.2f * plNoise(C.sTable + 1u, tableX / 25.f, tableY / 25.f));
    float rx = (tableX - 0.5f * W) / W, ry = (tableY - 0.5f * H) / H;
    float tablePool = 1.f - 0.6f * plSmooth(0.2f, 1.2f, sqrtf(rx * rx + ry * ry) * 1.6f);
    float contact = edgeD < 0.f ? 1.f - 0.9f * expf(edgeD / plMax(1.f, 16.f * C.scale)) : 0.1f;
    const float tk = wood * tablePool * contact;
    PL4 table{0.0075f * tk, 0.0056f * tk, 0.0042f * tk, 1.f};
    PL4 outLin = table;
    float aa = 0.5f / plMax(0.2f, mag);
    float inside = denom > 0.02f ? plSmooth(-aa, aa, edgeD) : 0.f;
    if (inside > 0.f) {
        float magPage = mag / ps;
        float blurScreen = C.dof * 0.04f * H * fabsf(pt.z - C.zFocus) / plMax(0.05f, C.zFocus);
        float blurPage = blurScreen / plMax(1e-3f, magPage);
        float lod = plMax(log2f(plMax(1.f, 1.f / plMax(1e-3f, magPage))), log2f(plMax(1.f, blurPage * 1.3f)));
        PL4 col = samplePage(plClamp(qxF, 0.f, PW - 0.001f), plClamp(qyF, 0.f, PH - 0.001f), plClamp(lod, 0.f, static_cast<float>(C.maxLod)));
        float gx, gy, crack;
        plHeightGrad(P, plClamp(qxF, 0.f, static_cast<float>(PW)), plClamp(qyF, 0.f, static_cast<float>(PH)), gx, gy, crack);
        PLVec nrm = plNorm({C.n0.x - gx * C.ea.x - gy * C.eb.x, C.n0.y - gx * C.ea.y - gy * C.eb.y, C.n0.z - gx * C.ea.z - gy * C.eb.z});
        float ndl = plMax(0.f, plDot(nrm, C.Lw));
        float shade = plLerp(1.f, plClamp(ndl / C.flatDiffuse, 0.3f, 1.25f), 0.85f);
        PLVec V = plNorm({-pt.x, -pt.y, -pt.z});
        float spec = 0.f;
        for (int k = 0; k < 5; ++k) {
            const PLVec& La = C.Larea[k];
            PLVec Hh = plNorm({La.x + V.x, La.y + V.y, La.z + V.z});
            float nh = plMax(0.f, plDot(nrm, Hh));
            float dd = nh * nh * (C.alpha2 - 1.f) + 1.f;
            float D = C.alpha2 / (3.14159265358979323846f * dd * dd);
            float vh = plMax(0.f, plDot(V, Hh));
            float om = 1.f - vh;
            float fres = 0.04f + 0.96f * om * om * om * om * om;
            spec += D * plMax(0.f, plDot(nrm, La)) * (0.35f + fres);
        }
        spec *= C.sheen * 0.06f / 5.f;
        spec = spec / (1.f + spec * 1.2f);
        float rim = (1.f - plSmooth(0.f, 1.5f * C.scale, edgeD)) * 0.06f * C.paperAmt;
        PL4 lit{col.r * shade + spec + rim, col.g * shade * 0.985f + spec * 0.99f + rim, col.b * shade * 0.95f + spec * 0.96f + rim, 1.f};
        outLin = PL4{plLerp(table.r, lit.r, inside), plLerp(table.g, lit.g, inside), plLerp(table.b, lit.b, inside), 1.f};
    }
    float dxv = x + 0.5f - (C.Fx1 + 0.5f * W), dyv = y + 0.5f - (C.Fy1 + 0.5f * H);
    float rn = sqrtf(dxv * dxv + dyv * dyv) / plMax(1.f, C.screenDiag);
    float vg = 1.f - C.vig * 0.75f * plSmooth(0.35f, 1.1f, rn);
    float ch[3] = {outLin.r, outLin.g, outLin.b};
    for (int k = 0; k < 3; ++k) {
        float lin = ch[k] * vg;
        lin = plFinite(lin) ? plMax(0.f, lin) : 0.f;
        if (lin > 0.8f) lin = 0.8f + 0.2f * (1.f - expf(-(lin - 0.8f) / 0.2f));
        ch[k] = C.enc == 0 ? lin : plEncode(lin, C.enc);
    }
    return PL4{ch[0], ch[1], ch[2], 1.f};
}

}
