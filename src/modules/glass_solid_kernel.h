#pragma once
#include "print_lab_kernel.h"

namespace tp {

struct GlassSolid {
    int shape = 0, samples = 1, spectral = 0, taps = 0;
    float principalX = 0, principalY = 0, focal = 1;
    float center[3] = {0, 0, 1};
    float radius = 1;
    float rot[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float iorA = 1.5f, iorB = 0;
    float aperture = 0, focusZ = 1, sceneZ = 1, reflection = 1;
    float bevel = 0, light = 0, blur = 0, highlight = 0, environment = 1;
    float thickness = 0.12f, strength = 0.5f;
    float absorb[3] = {0.06f, 0.06f, 0.06f};
    float frost = 0, refocus = 0, lightCos = 1, lightSin = 0;
    int facets = 6, studio = 0;
    static constexpr int kMaxFaces = 72;
    int faces = 0;
    float planes[kMaxFaces][4] = {};
    int blurOn = 0;
    float center1[3] = {0, 0, 1};
    float rot1[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
};

struct GsVec { float x, y, z; };

PL_HD inline GsVec gsAdd(GsVec a, GsVec b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
PL_HD inline GsVec gsSub(GsVec a, GsVec b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
PL_HD inline GsVec gsMul(GsVec a, float k) { return {a.x * k, a.y * k, a.z * k}; }
PL_HD inline float gsDot(GsVec a, GsVec b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
PL_HD inline GsVec gsNorm(GsVec a) {
    const float l = sqrtf(gsDot(a, a));
    return l > 0 ? gsMul(a, 1.f / l) : a;
}

struct GsPose { float c[3]; float rot[9]; };

PL_HD inline GsVec gsToObject(const GsPose& s, GsVec v) {
    return {s.rot[0] * v.x + s.rot[3] * v.y + s.rot[6] * v.z,
            s.rot[1] * v.x + s.rot[4] * v.y + s.rot[7] * v.z,
            s.rot[2] * v.x + s.rot[5] * v.y + s.rot[8] * v.z};
}

PL_HD inline GsVec gsToWorld(const GsPose& s, GsVec v) {
    return {s.rot[0] * v.x + s.rot[1] * v.y + s.rot[2] * v.z,
            s.rot[3] * v.x + s.rot[4] * v.y + s.rot[5] * v.z,
            s.rot[6] * v.x + s.rot[7] * v.y + s.rot[8] * v.z};
}

PL_HD inline bool gsPolyhedron(int shape) { return (shape >= 2 && shape <= 6) || (shape >= 16 && shape <= 20); }

PL_HD inline bool gsPlate(int shape) { return (shape >= 7 && shape <= 15) || (shape >= 21 && shape <= 24); }

PL_HD inline bool gsRoundPlate(int shape) { return shape == 9 || (shape >= 11 && shape <= 14) || shape >= 21; }

inline int gsPlatonicFaces(int shape) {
    return shape == 2 ? 6 : shape == 3 ? 4 : shape == 4 ? 8 : shape == 5 ? 12 : shape == 6 ? 20 : 0;
}

inline float gsPlatonicInradius(int shape) {
    return shape == 2 ? 0.57735027f : shape == 3 ? 0.33333333f : shape == 4 ? 0.57735027f : 0.79465447f;
}

inline GsVec gsFace(int shape, int i) {
    const float phi = 1.6180340f, iphi = 0.6180340f;
    if (shape == 2) {
        const float s = (i & 1) ? -1.f : 1.f;
        return i < 2 ? GsVec{s, 0, 0} : i < 4 ? GsVec{0, s, 0} : GsVec{0, 0, s};
    }
    if (shape == 3) {
        const GsVec t[4] = {{1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}};
        return gsNorm(t[i]);
    }
    if (shape == 4) return gsNorm({(i & 1) ? -1.f : 1.f, (i & 2) ? -1.f : 1.f, (i & 4) ? -1.f : 1.f});
    if (shape == 5) {
        const float a = (i & 1) ? -1.f : 1.f, b = (i & 2) ? -phi : phi;
        const int g = i >> 2;
        return gsNorm(g == 0 ? GsVec{0, a, b} : g == 1 ? GsVec{a, b, 0} : GsVec{b, 0, a});
    }
    if (i < 8) return gsNorm({(i & 1) ? -1.f : 1.f, (i & 2) ? -1.f : 1.f, (i & 4) ? -1.f : 1.f});
    const int k = i - 8;
    const float a = (k & 1) ? -iphi : iphi, b = (k & 2) ? -phi : phi;
    const int g = k >> 2;
    return gsNorm(g == 0 ? GsVec{0, a, b} : g == 1 ? GsVec{a, b, 0} : GsVec{b, 0, a});
}

inline void gsBuildPlanes(GlassSolid& s) {
    s.faces = 0;
    const auto add = [&](GsVec n, float d) {
        if (s.faces >= GlassSolid::kMaxFaces) return;
        n = gsNorm(n);
        float* plane = s.planes[s.faces++];
        plane[0] = n.x; plane[1] = n.y; plane[2] = n.z; plane[3] = d;
    };
    const float deg = 3.14159265f / 180.f;
    const auto ring = [&](int count, float tilt, float start, float up, float d) {
        for (int k = 0; k < count; ++k) {
            const float a = (start + 360.f * k / count) * deg;
            add({sinf(tilt * deg) * cosf(a), up * cosf(tilt * deg), sinf(tilt * deg) * sinf(a)}, d);
        }
    };
    if (s.shape >= 2 && s.shape <= 6) {
        for (int i = 0; i < gsPlatonicFaces(s.shape); ++i) add(gsFace(s.shape, i), gsPlatonicInradius(s.shape));
    } else if (s.shape == 16) {
        ring(3, 90.f, 90.f, 0.f, 0.5f);
        add({0, 1, 0}, 1.2f);
        add({0, -1, 0}, 1.2f);
    } else if (s.shape == 17) {
        for (int i = 0; i < 60; ++i) {
            const float z = 1.f - 2.f * (i + 0.5f) / 60.f, r = sqrtf(1.f - z * z), a = 2.39996323f * i;
            add({r * cosf(a), z, r * sinf(a)}, 0.93f);
        }
    } else if (s.shape == 18) {
        add({0, 1, 0}, 0.33f);
        ring(8, 34.5f, 0.f, 1.f, sinf(34.5f * deg));
        ring(8, 42.f, 22.5f, 1.f, sinf(42.f * deg));
        ring(32, 90.f, 0.f, 0.f, 1.f);
        ring(8, 40.75f, 0.f, -1.f, sinf(40.75f * deg));
        ring(8, 44.f, 22.5f, -1.f, sinf(44.f * deg));
    } else if (s.shape == 19) {
        ring(6, 90.f, 0.f, 0.f, 0.5f);
        const float cap = 0.5f * sinf(52.f * deg) + 0.9f * cosf(52.f * deg);
        ring(6, 52.f, 0.f, 1.f, cap);
        ring(6, 52.f, 0.f, -1.f, cap);
    } else if (s.shape == 20) {
        ring(48, 90.f, 0.f, 0.f, 0.35f);
        add({0, 1, 0}, 1.f);
        add({0, -1, 0}, 1.f);
    }
}

PL_HD inline float gsHash(unsigned v) {
    v ^= v >> 16; v *= 0x7feb352dU; v ^= v >> 15; v *= 0x846ca68bU; v ^= v >> 16;
    return (v >> 8) * (1.f / 16777216.f);
}

PL_HD inline float gsPlateHeight(const GlassSolid& s, float u, float v, float& hu, float& hv) {
    const float k = s.strength;
    hu = hv = 0;
    if (s.shape == 7) {
        hu = -k * u;
        return 0.5f * k * (1.f - u * u);
    }
    if (s.shape == 8) {
        const float w = 2.f / s.facets;
        const float q = (plClamp(u, -1.f, 0.99999f) + 1.f) / w;
        const int strip = static_cast<int>(floorf(q));
        const float slope = k * 0.5f * static_cast<float>((strip * 7) % 5 - 2);
        hu = slope;
        return k * w + slope * (q - strip - 0.5f) * w;
    }
    if (s.shape == 9) {
        const float r = sqrtf(u * u + v * v), x = (r - 0.6f) / 0.25f;
        if (fabsf(x) >= 1.f || r < 1e-6f) return 0.f;
        const float slope = -k * 0.25f * 3.14159265f / 0.25f * sinf(3.14159265f * x);
        hu = slope * u / r;
        hv = slope * v / r;
        return k * 0.25f * (1.f + cosf(3.14159265f * x));
    }
    if (s.shape == 10) {
        hu = 0.5f * k;
        return 0.5f * k * (u + 1.f);
    }
    const float pi = 3.14159265f, n = static_cast<float>(s.facets);
    const float r = sqrtf(u * u + v * v);
    if (s.shape >= 21 && s.shape <= 23) {
        const float sides = s.shape == 21 ? 3.f : s.shape == 22 ? 4.f : 5.f, step = 2.f * pi / sides;
        const float apothem = s.shape == 21 ? 0.f : s.shape == 22 ? 0.42f : 0.35f;
        const float angle = (floorf(atan2f(v, u) / step + (s.shape == 22 ? 0.5f : 0.f)) + (s.shape == 22 ? 0.f : 0.5f)) * step;
        const float cs = cosf(angle), sn = sinf(angle), reach = u * cs + v * sn;
        if (reach <= apothem) return 0.5f * k * (1.f - apothem);
        hu = -0.5f * k * cs;
        hv = -0.5f * k * sn;
        return 0.5f * k * (1.f - reach);
    }
    if (s.shape == 24) {
        if (v <= 0.f) return 0.f;
        const unsigned strip = static_cast<unsigned>(floorf(v * 320.f));
        const float h = gsHash(strip * 0x9e3779b9u + 7u);
        hu = 0.5f * k * (h < 0.35f ? 0.3f : 0.3f + 0.7f * powf((h - 0.35f) / 0.65f, 1.5f));
        return hu * (u + 1.f);
    }
    if (s.shape == 11) {
        if (r < 1e-6f) return 0.12f * k;
        const float slope = -0.06f * k * pi * n * sinf(pi * n * r);
        hu = slope * u / r;
        hv = slope * v / r;
        return 0.06f * k * (1.f + cosf(pi * n * r));
    }
    if (s.shape == 12) {
        const float size = 1.f / n, sq3 = 1.7320508f;
        const float fq = (sq3 / 3.f * u - v / 3.f) / size, fr = (2.f / 3.f * v) / size, fs = -fq - fr;
        float rq = roundf(fq), rr = roundf(fr), rs = roundf(fs);
        const float dq = fabsf(rq - fq), dr = fabsf(rr - fr), ds = fabsf(rs - fs);
        if (dq > dr && dq > ds) rq = -rr - rs;
        else if (dr > ds) rr = -rq - rs;
        const float cx = size * sq3 * (rq + 0.5f * rr), cy = size * 1.5f * rr;
        const unsigned cell = static_cast<unsigned>(static_cast<int>(rq)) * 73856093u ^ static_cast<unsigned>(static_cast<int>(rr)) * 19349663u;
        const float turn = 6.2831853f * gsHash(cell), lean = 0.4f * k * (0.35f + 0.65f * gsHash(cell + 0x9e3779b9u));
        hu = lean * cosf(turn);
        hv = lean * sinf(turn);
        return 0.8f * k * size + hu * (u - cx) + hv * (v - cy);
    }
    if (s.shape == 13) {
        const float step = 2.f * pi / n;
        const float angle = (floorf(atan2f(v, u) / step) + 0.5f) * step;
        const float cs = cosf(angle), sn = sinf(angle);
        hu = -0.5f * k * cs;
        hv = -0.5f * k * sn;
        return 0.5f * k * (1.f - (u * cs + v * sn));
    }
    if (s.shape == 14) {
        if (r < 1e-6f) return 0.f;
        const float A = 0.1f * k, theta = atan2f(v, u), phase = n * theta + 2.f * pi * r;
        const float dr = A * (1.f + cosf(phase)) - A * r * sinf(phase) * 2.f * pi;
        const float dt = -A * r * sinf(phase) * n;
        hu = dr * u / r - dt * v / (r * r);
        hv = dr * v / r + dt * u / (r * r);
        return A * r * (1.f + cosf(phase));
    }
    const float f = 1.5f + 0.5f * n, A = 0.05f * k;
    const float p1 = f * (0.83f * u + 0.56f * v) + 1.3f, p2 = f * (-0.42f * u + 0.91f * v) * 1.7f + 4.1f, p3 = f * (0.97f * u - 0.24f * v) * 2.9f + 2.2f;
    hu = A * f * (0.83f * cosf(p1) - 0.6f * 0.42f * 1.7f * cosf(p2) + 0.35f * 0.97f * 2.9f * cosf(p3));
    hv = A * f * (0.56f * cosf(p1) + 0.6f * 0.91f * 1.7f * cosf(p2) - 0.35f * 0.24f * 2.9f * cosf(p3));
    return A * (2.f + sinf(p1) + 0.6f * sinf(p2) + 0.35f * sinf(p3));
}

PL_HD inline bool gsPlateFront(const GlassSolid& s, GsVec o, GsVec d, bool inside, float& t, GsVec& normal) {
    if (fabsf(d.z) < 1e-6f) return false;
    const float half = 0.5f * s.thickness;
    float hu, hv, h = 0;
    float tt = (-half - o.z) / d.z;
    for (int i = 0; i < 5; ++i) {
        h = gsPlateHeight(s, o.x + d.x * tt, o.y + d.y * tt, hu, hv);
        tt = (-half - h - o.z) / d.z;
    }
    const float x = o.x + d.x * tt, y = o.y + d.y * tt;
    if (tt <= 1e-5f || fabsf(x) > 1.f || fabsf(y) > 1.f || (gsRoundPlate(s.shape) && x * x + y * y > 1.f)) return false;
    gsPlateHeight(s, x, y, hu, hv);
    t = tt;
    normal = gsNorm({-hu, -hv, -1.f});
    return inside ? gsDot(normal, d) > 0 : gsDot(normal, d) < 0;
}

PL_HD inline bool gsPlateIntersect(const GlassSolid& s, GsVec o, GsVec d, bool inside, float& t, GsVec& normal) {
    if (!inside) return d.z > 0 && gsPlateFront(s, o, d, false, t, normal);
    const float half = 0.5f * s.thickness;
    float best = 1e30f;
    GsVec n{0, 0, 1};
    const GsVec walls[5] = {{0, 0, 1}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}};
    const float reach[5] = {half, 1.f, 1.f, 1.f, 1.f};
    const bool round = gsRoundPlate(s.shape);
    if (round) {
        const float a = d.x * d.x + d.y * d.y, b = o.x * d.x + o.y * d.y, c = o.x * o.x + o.y * o.y - 1.f;
        const float disc = b * b - a * c;
        if (a > 1e-9f && disc >= 0) {
            const float ti = (-b + sqrtf(disc)) / a;
            if (ti > 1e-5f) { best = ti; n = gsNorm({o.x + d.x * ti, o.y + d.y * ti, 0.f}); }
        }
    }
    for (int i = 0; i < (round ? 1 : 5); ++i) {
        const float denom = gsDot(walls[i], d);
        if (denom <= 1e-6f) continue;
        const float ti = (reach[i] - gsDot(walls[i], o)) / denom;
        if (ti > 1e-5f && ti < best) { best = ti; n = walls[i]; }
    }
    float tf;
    GsVec nf;
    if (d.z < 0 && gsPlateFront(s, o, d, true, tf, nf) && tf < best) { best = tf; n = nf; }
    if (best >= 1e30f) return false;
    t = best;
    normal = n;
    return true;
}

PL_HD inline GsVec gsBevel(const GlassSolid& s, GsVec p, GsVec d, int face, GsVec n, bool inside) {
    const float bevel = s.bevel;
    if (bevel <= 0) return n;
    GsVec bent = n;
    for (int j = 0; j < s.faces; ++j) {
        if (j == face) continue;
        const GsVec m{s.planes[j][0], s.planes[j][1], s.planes[j][2]};
        const float gap = s.planes[j][3] - gsDot(m, p);
        if (gap >= bevel) continue;
        const float w = 1.f - plMax(gap, 0.f) / bevel;
        bent = gsAdd(bent, gsMul(m, w * w));
    }
    bent = gsNorm(bent);
    const float facing = gsDot(bent, d);
    return (inside ? facing > 1e-3f : facing < -1e-3f) ? bent : n;
}

PL_HD inline bool gsIntersect(const GlassSolid& s, GsVec o, GsVec d, bool inside, float& t, GsVec& normal) {
    if (gsPlate(s.shape)) return gsPlateIntersect(s, o, d, inside, t, normal);
    if (s.shape == 1) {
        const float b = gsDot(o, d), c = gsDot(o, o) - 1.f, disc = b * b - c;
        if (disc < 0) return false;
        const float q = sqrtf(disc);
        t = inside ? -b + q : -b - q;
        if (t <= 1e-5f) return false;
        normal = gsNorm(gsAdd(o, gsMul(d, t)));
        return true;
    }
    float tNear = -1e30f, tFar = 1e30f;
    GsVec nNear{0, 0, 1}, nFar{0, 0, 1};
    int iNear = -1, iFar = -1;
    for (int i = 0; i < s.faces; ++i) {
        const GsVec n{s.planes[i][0], s.planes[i][1], s.planes[i][2]};
        const float denom = gsDot(n, d), dist = s.planes[i][3] - gsDot(n, o);
        if (fabsf(denom) < 1e-9f) {
            if (dist < 0) return false;
            continue;
        }
        const float ti = dist / denom;
        if (denom < 0) { if (ti > tNear) { tNear = ti; nNear = n; iNear = i; } }
        else if (ti < tFar) { tFar = ti; nFar = n; iFar = i; }
    }
    if (tNear > tFar) return false;
    if (inside) {
        if (tFar <= 1e-5f) return false;
        t = tFar;
        normal = gsBevel(s, gsAdd(o, gsMul(d, t)), d, iFar, nFar, true);
        return true;
    }
    if (tNear <= 1e-5f) return false;
    t = tNear;
    normal = gsBevel(s, gsAdd(o, gsMul(d, t)), d, iNear, nNear, false);
    return true;
}

PL_HD inline bool gsRefract(GsVec d, GsVec n, float eta, GsVec& out) {
    const float c = -gsDot(n, d), k = 1.f - eta * eta * (1.f - c * c);
    if (k < 0) return false;
    out = gsNorm(gsAdd(gsMul(d, eta), gsMul(n, eta * c - sqrtf(k))));
    return true;
}

PL_HD inline GsVec gsReflect(GsVec d, GsVec n) { return gsSub(d, gsMul(n, 2.f * gsDot(d, n))); }

PL_HD inline float gsSchlick(float cosine, float ior) {
    const float r0 = (ior - 1.f) / (ior + 1.f);
    const float m = 1.f - plClamp(cosine, 0.f, 1.f);
    return r0 * r0 + (1.f - r0 * r0) * m * m * m * m * m;
}


PL_HD inline void gsDisk(float u, float v, float& x, float& y) {
    const float a = 2.f * u - 1.f, b = 2.f * v - 1.f;
    if (a == 0 && b == 0) { x = y = 0; return; }
    const float quarter = 0.78539816f;
    float r, phi;
    if (fabsf(a) > fabsf(b)) { r = a; phi = quarter * (b / a); }
    else { r = b; phi = 2.f * quarter - quarter * (a / b); }
    x = r * cosf(phi);
    y = r * sinf(phi);
}

PL_HD inline float gsLobe(float l, float mu, float below, float above) {
    const float t = (l - mu) / (l < mu ? below : above);
    return expf(-0.5f * t * t);
}

PL_HD inline void gsSpectrum(float nm, float w[3]) {
    const float X = 1.056f * gsLobe(nm, 599.8f, 37.9f, 31.0f) + 0.362f * gsLobe(nm, 442.0f, 16.0f, 26.7f) - 0.065f * gsLobe(nm, 501.1f, 20.4f, 26.2f);
    const float Y = 0.821f * gsLobe(nm, 568.8f, 46.9f, 40.5f) + 0.286f * gsLobe(nm, 530.9f, 16.3f, 31.1f);
    const float Z = 1.217f * gsLobe(nm, 437.0f, 11.8f, 36.0f) + 0.681f * gsLobe(nm, 459.0f, 26.0f, 13.8f);
    w[0] = plMax(0.f, 3.2406f * X - 1.5372f * Y - 0.4986f * Z);
    w[1] = plMax(0.f, -0.9689f * X + 1.8758f * Y + 0.0415f * Z);
    w[2] = plMax(0.f, 0.0557f * X - 0.2040f * Y + 1.0570f * Z);
}

PL_HD inline void gsProject(const GlassSolid& s, GsVec origin, GsVec dir, float& px, float& py) {
    const float dz = fabsf(dir.z) > 1e-6f ? fabsf(dir.z) : 1e-6f;
    const float t = (s.sceneZ - (dir.z >= 0 ? origin.z : -origin.z)) / dz;
    px = s.principalX + s.focal * (origin.x + dir.x * t) / s.sceneZ;
    py = s.principalY + s.focal * (origin.y + dir.y * t) / s.sceneZ;
}

PL_HD inline float gsSoftbox(GsVec d, GsVec light) {
    const float c = gsDot(d, light);
    const float t = plClamp((c - 0.975f) * 50.f, 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
}

PL_HD inline PL4 gsStudio(const GlassSolid& s, GsVec dir, PL4 seen) {
    const float x = s.lightCos * dir.x + s.lightSin * dir.y, y = -s.lightSin * dir.x + s.lightCos * dir.y;
    const float side = sqrtf(x * x + y * y);
    const float lens = plClamp((side - 0.05f) * 120.f, 0.f, 1.f);
    const float dark = s.environment * lens;
    seen.r *= dark; seen.g *= dark; seen.b *= dark;
    if (s.light <= 0) return seen;
    const float az = atan2f(y, x);
    const float spoke = plClamp((cosf(6.f * az + 0.5f) + 0.35f) * 12.f, 0.f, 1.f);
    const float ring = lens * spoke * plClamp((0.20f - side) * 120.f, 0.f, 1.f);
    const float band = plClamp((side - 0.30f) * 12.f, 0.f, 1.f) * plClamp((0.88f - side) * 12.f, 0.f, 1.f);
    float glow = 0;
    if (band > 0) {
        for (int k = 0; k < 5; ++k) {
            const float gap = fabsf(remainderf(az - (0.21f + 1.2566371f * k), 6.2831853f));
            const float half = k == 0 ? 0.11f : 0.06f;
            glow += plClamp((1.4f * half - gap) / (0.4f * half), 0.f, 1.f);
        }
    }
    glow = s.light * (5.f * band * glow + 2.5f * ring);
    seen.r += glow; seen.g += glow; seen.b += glow;
    const float peak = 0.2126f * seen.r + 0.7152f * seen.g + 0.0722f * seen.b;
    if (peak > 4.f) {
        const float k = 4.f / peak;
        seen.r *= k; seen.g *= k; seen.b *= k;
    }
    return seen;
}

PL_HD inline PL4 gsSurround(const GlassSolid& s, GsVec dir, PL4 seen) {
    if (s.studio) return gsStudio(s, dir, seen);
    seen.r *= s.environment; seen.g *= s.environment; seen.b *= s.environment;
    if (s.light > 0) {
        const float c = s.lightCos, sn = s.lightSin;
        const GsVec key = gsNorm({-0.38f * c + 0.62f * sn, -0.38f * sn - 0.62f * c, -1.f});
        const GsVec fill = gsNorm({0.70f * c - 0.30f * sn, 0.70f * sn + 0.30f * c, -1.f});
        const float glow = s.light * (gsSoftbox(dir, key) + 0.45f * gsSoftbox(dir, fill));
        seen.r += glow; seen.g += glow; seen.b += glow;
    }
    const float peak = 0.2126f * seen.r + 0.7152f * seen.g + 0.0722f * seen.b;
    if (peak > 1.25f) {
        const float k = 1.25f / peak;
        seen.r *= k; seen.g *= k; seen.b *= k;
    }
    return seen;
}

PL_HD inline GsVec gsFrost(const GlassSolid& s, GsVec n, GsVec d, unsigned rng, bool inside) {
    if (s.frost <= 0) return n;
    const GsVec jitter{gsHash(rng) * 2.f - 1.f, gsHash(rng + 0x3c6ef372u) * 2.f - 1.f, gsHash(rng + 0xa54ff53au) * 2.f - 1.f};
    const GsVec rough = gsNorm(gsAdd(n, gsMul(jitter, 0.45f * s.frost)));
    const float facing = gsDot(rough, d);
    return (inside ? facing > 1e-3f : facing < -1e-3f) ? rough : n;
}

template <class Sample, class Look>
PL_HD inline PL4 gsTrace(const GlassSolid& s, GsVec origin, GsVec dir, float ior, float shiftX, float shiftY,
                         unsigned rng, const GsPose& pose, const Sample& sample, const Look& look, bool& hit) {
    const float scale = 1.f / s.radius;
    GsVec o = gsMul(gsToObject(pose, gsSub(origin, {pose.c[0], pose.c[1], pose.c[2]})), scale);
    GsVec d = gsToObject(pose, dir);
    float t;
    GsVec n;
    hit = gsIntersect(s, o, d, false, t, n);
    if (!hit) return {};
    n = gsFrost(s, n, d, rng, false);
    const float fresnel = plMin(1.f, s.reflection * gsSchlick(-gsDot(n, d), ior));
    const GsVec bounce = gsReflect(d, n);
    o = gsAdd(o, gsMul(d, t));
    const GsVec entry = gsAdd(gsMul(gsToWorld(pose, o), s.radius), {pose.c[0], pose.c[1], pose.c[2]});
    GsVec inner;
    if (!gsRefract(d, n, 1.f / ior, inner)) inner = bounce;
    d = inner;
    float path = 0, carry = 1.f;
    float px, py;
    PL4 through{0, 0, 0, 0};
    bool any = false;
    for (int k = 0; k < 8 && carry > 0.02f; ++k) {
        if (!gsIntersect(s, gsAdd(o, gsMul(d, 1e-4f)), d, true, t, n)) break;
        o = gsAdd(o, gsMul(d, t + 1e-4f));
        path += t;
        n = gsFrost(s, n, d, rng + 0x68e31da4u * (k + 1), true);
        GsVec out;
        if (!gsRefract(d, gsMul(n, -1.f), ior, out)) { d = gsReflect(d, n); continue; }
        const float leave = carry * (1.f - gsSchlick(gsDot(out, n), ior));
        const GsVec exitPoint = gsAdd(gsMul(gsToWorld(pose, o), s.radius), {pose.c[0], pose.c[1], pose.c[2]});
        const GsVec exitDir = gsToWorld(pose, out);
        gsProject(s, exitPoint, exitDir, px, py);
        PL4 seen = look(px + shiftX, py + shiftY);
        if (exitDir.z < 0) seen = gsSurround(s, exitDir, seen);
        through.r += seen.r * leave * expf(-s.absorb[0] * path);
        through.g += seen.g * leave * expf(-s.absorb[1] * path);
        through.b += seen.b * leave * expf(-s.absorb[2] * path);
        if (!any) { through.a = seen.a; any = true; }
        carry *= gsSchlick(gsDot(out, n), ior);
        d = gsReflect(d, n);
    }
    if (carry > 0.02f) {
        const GsVec endPoint = gsAdd(gsMul(gsToWorld(pose, o), s.radius), {pose.c[0], pose.c[1], pose.c[2]});
        const GsVec endDir = gsToWorld(pose, d);
        gsProject(s, endPoint, endDir, px, py);
        PL4 seen = look(px + shiftX, py + shiftY);
        if (endDir.z < 0) seen = gsSurround(s, endDir, seen);
        const float rest = 0.4f * carry;
        through.r += seen.r * rest * expf(-s.absorb[0] * path);
        through.g += seen.g * rest * expf(-s.absorb[1] * path);
        through.b += seen.b * rest * expf(-s.absorb[2] * path);
        if (!any) through.a = seen.a;
    }
    if (fresnel <= 0) return through;
    const GsVec mirrorDir = gsToWorld(pose, bounce);
    gsProject(s, entry, mirrorDir, px, py);
    const PL4 mirror = gsSurround(s, mirrorDir, sample(px + shiftX, py + shiftY));
    const float keep = 1.f - fresnel;
    return {through.r * keep + mirror.r * fresnel, through.g * keep + mirror.g * fresnel,
            through.b * keep + mirror.b * fresnel, through.a};
}

template <class Sample>
PL_HD inline PL4 gsBokehPixel(const Sample& sample, float cx, float cy, float radius, int taps, float highlight) {
    float r = 0, g = 0, b = 0, a = 0;
    const float ca = -0.73736887f, sa = 0.67549029f;
    float u = 1.f, v = 0.f;
    for (int k = 0; k < taps; ++k) {
        const float rr = radius * sqrtf((k + 0.5f) / taps);
        const PL4 tap = sample(cx + rr * u, cy + rr * v);
        const float lum = 0.2126f * tap.r + 0.7152f * tap.g + 0.0722f * tap.b;
        const float lift = plClamp((lum - 0.75f) * 4.f, 0.f, 1.f);
        const float gain = 1.f + highlight * lift * lift;
        r += tap.r * gain; g += tap.g * gain; b += tap.b * gain; a += tap.a;
        const float nu = u * ca - v * sa;
        v = u * sa + v * ca;
        u = nu;
    }
    const float w = 1.f / taps;
    return {r * w, g * w, b * w, a * w};
}

template <class Sample, class Sharp>
PL_HD inline PL4 glassSolidPixel(const GlassSolid& s, int encoding, int x, int y, const Sample& sample, const Sharp& sharp) {
    const auto look = [&](float lx, float ly) {
        if (s.refocus <= 0) return sample(lx, ly);
        const PL4 crisp = sharp(lx, ly);
        if (s.refocus >= 1) return crisp;
        const PL4 soft = sample(lx, ly);
        const float k = s.refocus;
        return PL4{soft.r + (crisp.r - soft.r) * k, soft.g + (crisp.g - soft.g) * k, soft.b + (crisp.b - soft.b) * k, soft.a + (crisp.a - soft.a) * k};
    };
    const unsigned seed = static_cast<unsigned>(x) * 73856093u ^ static_cast<unsigned>(y) * 19349663u;
    const float h1 = gsHash(seed), h2 = gsHash(seed + 0x9e3779b9u), h3 = gsHash(seed + 0x7f4a7c15u),
                h4 = gsHash(seed + 0x2545f491u), h5 = gsHash(seed + 0x1b873593u);
    const float cx = x + 0.5f, cy = y + 0.5f;
    const PL4 base = sample(cx, cy);
    float acc[3] = {0, 0, 0}, alpha = 0, sumL = 0, sumL2 = 0;
    int hits = 0, used = 0, target = s.samples;
    const int limit = s.samples * 8;
    for (int i = 0; i < limit; ++i) {
        if (i == target) {
            const float n = static_cast<float>(i), mean = sumL / n;
            const float error = sqrtf(plMax(0.f, sumL2 / n - mean * mean) / n);
            if (!hits || (hits == i && error < 0.0015f + 0.004f * mean)) break;
            target *= 2;
        }
        const float k = static_cast<float>(i);
        const float u = h1 + k * 0.75487767f, v = h2 + k * 0.56984029f;
        const float jx = h3 + k * 0.61803399f, jy = h4 + k * 0.41421356f;
        float lx, ly;
        gsDisk(u - floorf(u), v - floorf(v), lx, ly);
        const float px = x + (jx - floorf(jx)), py = y + (jy - floorf(jy));
        const GsVec pin{(px - s.principalX) / s.focal, (py - s.principalY) / s.focal, 1.f};
        const GsVec origin{lx * s.aperture, ly * s.aperture, 0.f};
        const GsVec dir = gsNorm(gsSub(gsMul(pin, s.focusZ), origin));
        float sx, sy;
        gsProject(s, origin, dir, sx, sy);
        GsPose pose;
        for (int a = 0; a < 3; ++a) pose.c[a] = s.center[a];
        for (int a = 0; a < 9; ++a) pose.rot[a] = s.rot[a];
        if (s.blurOn) {
            const float sh = gsHash(seed + 0x27d4eb2fu) + k * 0.85667489f, tau = sh - floorf(sh);
            for (int a = 0; a < 3; ++a) pose.c[a] += (s.center1[a] - s.center[a]) * tau;
            for (int a = 0; a < 9; ++a) pose.rot[a] += (s.rot1[a] - s.rot[a]) * tau;
            const GsVec r0 = gsNorm({pose.rot[0], pose.rot[1], pose.rot[2]});
            GsVec r1{pose.rot[3], pose.rot[4], pose.rot[5]};
            r1 = gsNorm(gsSub(r1, gsMul(r0, gsDot(r0, r1))));
            const GsVec r2{r0.y * r1.z - r0.z * r1.y, r0.z * r1.x - r0.x * r1.z, r0.x * r1.y - r0.y * r1.x};
            const float rows[9] = {r0.x, r0.y, r0.z, r1.x, r1.y, r1.z, r2.x, r2.y, r2.z};
            for (int a = 0; a < 9; ++a) pose.rot[a] = rows[a];
        }
        const float q = h5 + k * 0.73205081f;
        const int heroes = s.spectral ? 3 : 1;
        PL4 c = base;
        bool hit = false;
        float part[3] = {0, 0, 0}, norm[3] = {0, 0, 0};
        for (int j = 0; j < heroes; ++j) {
            float w[3] = {1.f, 1.f, 1.f}, lambda = 0.5876f;
            if (s.spectral) {
                const float f = q + j * 0.33333333f;
                lambda = 0.40f + 0.30f * (f - floorf(f));
                gsSpectrum(lambda * 1000.f, w);
            }
            const PL4 t = gsTrace(s, origin, dir, s.iorA + s.iorB / (lambda * lambda), cx - sx, cy - sy,
                                  seed + 0x85ebca6bu * static_cast<unsigned>(i + 1), pose, sample, look, hit);
            if (!hit) break;
            part[0] += t.r * w[0]; part[1] += t.g * w[1]; part[2] += t.b * w[2];
            norm[0] += w[0]; norm[1] += w[1]; norm[2] += w[2];
            c.a = t.a;
        }
        if (hit) {
            ++hits;
            c.r = norm[0] > 0 ? part[0] / norm[0] : 0.f;
            c.g = norm[1] > 0 ? part[1] / norm[1] : 0.f;
            c.b = norm[2] > 0 ? part[2] / norm[2] : 0.f;
            acc[0] += c.r; acc[1] += c.g; acc[2] += c.b;
        }
        alpha += c.a;
        const float lum = 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
        sumL += lum;
        sumL2 += lum * lum;
        ++used;
    }
    if (!hits) return {plEncode(base.r, encoding), plEncode(base.g, encoding), plEncode(base.b, encoding), base.a};
    const float w = 1.f / used, rest = 1.f - static_cast<float>(hits) * w;
    const float r = acc[0] * w + rest * base.r;
    const float g = acc[1] * w + rest * base.g;
    const float b = acc[2] * w + rest * base.b;
    return {plEncode(r, encoding), plEncode(g, encoding), plEncode(b, encoding), alpha / used};
}

}
