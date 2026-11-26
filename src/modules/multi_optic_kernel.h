#pragma once
#include "refraction_kernel.h"
#include "glass_solid_kernel.h"

namespace tp {

struct OpticElement {
    RLParams glass;
    float aspect = 1, tiltX = 0, tiltY = 0, ringWidth = 0.30f;
    float repeatScale = 1, repeatOffset = 0, facetFeather = 18, facetDefocus = 5;
    float blur = 0;
    bool enabled = false;
};

struct OpticStack {
    OpticElement elements[3];
    int order[3] = {0, 1, 2};
    int mask = 0, encoding = 0, channels = 3, edge = 0;
    float diagonal = 1, reflection = 0, roughness = 0, highlight = 0;
    float frameCenterX = 0, frameCenterY = 0, invHalfWidth = 0, invHalfHeight = 0, arrayFringe = 0;
    bool hasArray = false;
    GlassSolid solid;
};

struct OpticArrayPoint { float u, v, projection; };

PL_HD inline OpticArrayPoint opticArrayPoint(const OpticElement& element, float x, float y) {
    const RLParams& glass = element.glass;
    const float dx = x - glass.cx, dy = y - glass.cy;
    const float projection = sqrtf(1.f + element.tiltX * element.tiltX) / glass.s;
    return {(glass.cs * dx + glass.sn * dy) * projection,
            (-glass.sn * dx + glass.cs * dy) * sqrtf(1.f + element.tiltY * element.tiltY) / (glass.s * element.aspect),
            projection};
}

PL_HD inline float opticArrayMask(const RLParams& glass, OpticArrayPoint point) {
    return rlBoxK(point.u, glass.e) * rlBoxK(point.v, glass.e);
}

PL_HD inline PLVec opticTiltNormal(const OpticElement& element, PLVec normal) {
    const RLParams& glass = element.glass;
    normal.x += glass.cs * element.tiltX - glass.sn * element.tiltY;
    normal.y += glass.sn * element.tiltX + glass.cs * element.tiltY;
    const float norm = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    return {normal.x / norm, normal.y / norm, normal.z / norm};
}

PL_HD inline float opticGeometry(const OpticElement& element, float x, float y, PLVec& normal) {
    const RLParams& glass = element.glass;
    if (glass.mode == 7) {
        normal = {0.f, 0.f, 1.f};
        return opticArrayMask(glass, opticArrayPoint(element, x, y));
    }
    if (glass.mode < 4 && element.aspect == 1 && element.tiltX == 0 && element.tiltY == 0)
        return rlGeom(glass, x, y, normal);

    const auto point = opticArrayPoint(element, x, y);
    const float u = point.u, v = point.v;
    float mask;
    if (glass.mode < 4) {
        mask = rlGeom(glass, glass.cx + glass.s * (glass.cs * u - glass.sn * v),
                      glass.cy + glass.s * (glass.sn * u + glass.cs * v), normal);
    } else {
        float slopeU = 0, slopeV = 0;
        if (glass.mode == 6) {
            const float halfHeight = 0.5f * (1.f + u);
            mask = rlBoxK(u, glass.e) * (1.f - plSmooth(-glass.e, glass.e, fabsf(v) - halfHeight));
            slopeU = glass.curv;
            slopeV = 0.35f * glass.curv * v;
        } else {
            const float r = sqrtf(u * u + v * v);
            mask = rlBoxK(r, glass.e);
            float slope = -2.f * glass.curv * r;
            if (glass.mode == 5) {
                const float inner = 1.f - element.ringWidth;
                const float feather = plMin(glass.e, element.ringWidth * 0.45f);
                mask = rlBoxK(r, feather) * plSmooth(inner - feather, inner + feather, r);
                slope = -2.f * glass.curv * plClamp((r - inner) / element.ringWidth * 2.f - 1.f, -1.f, 1.f);
            }
            const float scale = slope / plMax(r, 0.001f);
            slopeU = scale * u;
            slopeV = scale * v;
        }
        normal = {-glass.hscale * (glass.cs * slopeU - glass.sn * slopeV),
                  -glass.hscale * (glass.sn * slopeU + glass.cs * slopeV), 1.f};
        const float norm = sqrtf(normal.x * normal.x + normal.y * normal.y + 1.f);
        normal = {normal.x / norm, normal.y / norm, 1.f / norm};
    }
    normal = opticTiltNormal(element, normal);
    return mask;
}

PL_HD inline float opticChannel(PL4 pixel, int channel) {
    return channel == 0 ? pixel.r : (channel == 1 ? pixel.g : pixel.b);
}

template <class Sample>
PL_HD inline PL4 apertureSample(const Sample& sample, float x, float y, float radius, int channels) {
    PL4 center = sample(x, y);
    if (radius < 0.5f) return center;
    const int taps = channels == 3 ? 8 : 4;
    const float offsets[8][2] = {
        {1, 0}, {0, 1}, {-1, 0}, {0, -1},
        {0.5f, 0.5f}, {-0.5f, 0.5f}, {-0.5f, -0.5f}, {0.5f, -0.5f}
    };
    PL4 total{2.f * center.r, 2.f * center.g, 2.f * center.b, 2.f * center.a};
    for (int i = 0; i < taps; ++i) {
        const PL4 tap = sample(x + radius * offsets[i][0], y + radius * offsets[i][1]);
        total.r += tap.r; total.g += tap.g; total.b += tap.b; total.a += tap.a;
    }
    const float weight = 1.f / (taps + 2.f);
    return {total.r * weight, total.g * weight, total.b * weight, total.a * weight};
}

struct OpticRay {
    float x, y, transmission, blur, reflection, coverage, weight;
    float dxX, dxY, dyX, dyY, facetFringe;
    int nextLayer;
};

PL_HD inline void advanceOpticRay(const OpticStack& stack, const OpticElement& element, OpticRay& ray,
                                 PLVec normal, float index, float coverage, float arrayOffset = 0) {
    const RLParams& glass = element.glass;
    float bx, by;
    const bool transmitted = rlTraceK(normal, index, glass.T, glass.dist, bx, by);
    bx += arrayOffset * glass.cs;
    by += arrayOffset * glass.sn;
    const float length = sqrtf(bx * bx + by * by);
    const float scale = coverage * (length > stack.diagonal ? stack.diagonal / length : 1.f);
    ray.x += bx * scale;
    ray.y += by * scale;
    ray.transmission *= plLerp(1.f, transmitted ? 1.f : 0.35f, coverage);
    ray.blur = plMax(ray.blur, element.blur * coverage);
    ray.coverage = plMax(ray.coverage, coverage);
    if (stack.reflection > 0) {
        const float grazing = 1.f - normal.z;
        ray.reflection = plMax(ray.reflection, coverage * (0.15f + 0.85f * grazing * grazing));
    }
}

PL_HD inline void advanceArrayFacet(const OpticStack& stack, const OpticElement& element, OpticRay& ray,
                                    OpticArrayPoint point, float width, int facet) {
    const RLParams& glass = element.glass;
    const float center = -1.f + (facet + 0.5f) * width;
    const float localU = point.u - center;
    const float target = localU * element.repeatScale + element.repeatOffset;
    const float index = glass.nc[1];
    const float strength = plSat((index - 1.f) * 2.f);
    const float offset = (target - point.u) / point.projection * strength;
    const float stretch = (element.repeatScale - 1.f) * strength;
    const float alongDx = (ray.dxX * glass.cs + ray.dxY * glass.sn) * stretch;
    const float alongDy = (ray.dyX * glass.cs + ray.dyY * glass.sn) * stretch;
    ray.dxX += alongDx * glass.cs;
    ray.dxY += alongDx * glass.sn;
    ray.dyX += alongDy * glass.cs;
    ray.dyY += alongDy * glass.sn;
    const float rim = plMax(2.f * fabsf(localU) / width, fabsf(point.v));
    ray.facetFringe = plMax(ray.facetFringe, plSmooth(0.55f, 1.f, rim) * strength);
    const float slope = glass.hscale * glass.curv * (-1.f + 2.f * facet / (glass.facets - 1));
    const float invNorm = 1.f / sqrtf(1.f + slope * slope);
    const PLVec normal{-glass.cs * slope * invNorm, -glass.sn * slope * invNorm, invNorm};
    advanceOpticRay(stack, element, ray, opticTiltNormal(element, normal), index, 1.f, offset);
}

template <class Sample>
PL_HD inline PL4 sampleArrayRay(const Sample& sample, const OpticRay& ray, float x, float y, int channels) {
    const float spanX2 = ray.dxX * ray.dxX + ray.dxY * ray.dxY;
    const float spanY2 = ray.dyX * ray.dyX + ray.dyY * ray.dyY;
    const float span2 = plMax(spanX2, spanY2);
    if (span2 <= 1.001f || (ray.blur >= 1.f && 4.f * ray.blur * ray.blur >= span2))
        return apertureSample(sample, x, y, ray.blur, channels);

    // Bilinear sampling already reconstructs a single pixel. Integrate the extra
    // footprint along each mapped axis only where a facet is shrinking detail.
    const int tapsX = spanX2 > 1.001f ? static_cast<int>(plMin(6.f, ceilf(sqrtf(spanX2)))) : 1;
    const int tapsY = spanY2 > 1.001f ? static_cast<int>(plMin(6.f, ceilf(sqrtf(spanY2)))) : 1;
    const float spreadX = tapsX > 1 ? sqrtf(1.f - 1.f / spanX2) : 0.f;
    const float spreadY = tapsY > 1 ? sqrtf(1.f - 1.f / spanY2) : 0.f;
    PL4 total;
    for (int j = 0; j < tapsY; ++j) {
        const float fy = ((j + 0.5f) / tapsY - 0.5f) * spreadY;
        for (int i = 0; i < tapsX; ++i) {
            const float fx = ((i + 0.5f) / tapsX - 0.5f) * spreadX;
            const PL4 tap = apertureSample(sample, x + fx * ray.dxX + fy * ray.dyX,
                                           y + fx * ray.dxY + fy * ray.dyY, ray.blur, channels);
            total.r += tap.r; total.g += tap.g; total.b += tap.b; total.a += tap.a;
        }
    }
    const float weight = 1.f / (tapsX * tapsY);
    return {total.r * weight, total.g * weight, total.b * weight, total.a * weight};
}

template <bool WithArrays, class Sample>
PL_HD inline PL4 shadeOpticRay(const OpticStack& stack, const OpticRay& ray,
                               float originX, float originY, const Sample& sample) {
    PL4 light;
    if constexpr (WithArrays) {
        light = sampleArrayRay(sample, ray, ray.x, ray.y, stack.channels);
        const float dx = originX - stack.frameCenterX, dy = originY - stack.frameCenterY;
        const float nx = dx * stack.invHalfWidth, ny = dy * stack.invHalfHeight;
        const float field = plSmooth(0.20f, 1.65f, nx * nx + ny * ny);
        const float fringe = stack.arrayFringe * field * ray.facetFringe;
        if (fringe > 0.01f) {
            const float factor = fringe / sqrtf(dx * dx + dy * dy);
            const float shiftX = factor * dx, shiftY = factor * dy;
            light.r = sampleArrayRay(sample, ray, ray.x - 0.62f * shiftX, ray.y - 0.62f * shiftY, stack.channels).r;
            light.b = sampleArrayRay(sample, ray, ray.x + shiftX, ray.y + shiftY, stack.channels).b;
        }
    } else {
        light = apertureSample(sample, ray.x, ray.y, ray.blur, stack.channels);
    }
    light.r *= ray.transmission; light.g *= ray.transmission; light.b *= ray.transmission;
    if (ray.reflection > 0) {
        const PL4 ghost = apertureSample(sample, 2.f * originX - ray.x + stack.diagonal * 0.025f,
                                        2.f * originY - ray.y - stack.diagonal * 0.014f,
                                        stack.roughness, stack.channels);
        const float lum = 0.2126f * ghost.r + 0.7152f * ghost.g + 0.0722f * ghost.b;
        const float weight = stack.reflection * ray.reflection * plSmooth(0.15f, 1.f, lum) * ghost.a;
        light.r += weight * ghost.r; light.g += weight * ghost.g; light.b += weight * ghost.b;
    }
    if (stack.highlight > 0) {
        const float lum = 0.2126f * light.r + 0.7152f * light.g + 0.0722f * light.b;
        const float gain = 1.f + stack.highlight * ray.coverage * plSmooth(0.3f, 1.f, lum);
        light.r *= gain; light.g *= gain; light.b *= gain;
    }
    return light;
}

template <bool WithArrays, class Sample>
PL_HD inline bool traceOpticChannel(const OpticStack& stack, float x, float y, int channel,
                                    const Sample& sample, PL4& light) {
    OpticRay ray{x, y, 1.f, 0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0};
    // Each of the three elements can defer an uncovered ray and a neighboring facet.
    OpticRay remaining[WithArrays ? 6 : 1];
    int pending = 0;
    while (true) {
        for (int layer = ray.nextLayer; layer < 3; ++layer) {
            const OpticElement& element = stack.elements[stack.order[layer]];
            if (!element.enabled) continue;
            const RLParams& glass = element.glass;
            const float index = glass.nc[channel];
            if constexpr (WithArrays) {
                if (glass.mode == 7) {
                    const auto point = opticArrayPoint(element, ray.x, ray.y);
                    const float mask = opticArrayMask(glass, point);
                    if (mask <= 0) continue;
                    // Blend independently traced colors at the rim. Scaling the displacement
                    // here compresses the source into bands along the edge.
                    if (mask < 1.f) {
                        OpticRay uncovered = ray;
                        uncovered.weight *= 1.f - mask;
                        uncovered.nextLayer = layer + 1;
                        remaining[pending++] = uncovered;
                        ray.weight *= mask;
                    }
                    const float width = 2.f / glass.facets;
                    const float position = (point.u + 1.f) / width;
                    const int seam = static_cast<int>(floorf(position + 0.5f));
                    const float distance = (position - seam) * width;
                    const float pixelHalf = 0.5f * (fabsf(glass.cs) + fabsf(glass.sn));
                    const float feather = plMin(0.25f * width, plMax(pixelHalf, element.facetFeather) * point.projection);
                    int facet = static_cast<int>(plClamp(floorf(position), 0.f, glass.facets - 1.f));
                    if (seam > 0 && seam < glass.facets && fabsf(distance) < feather) {
                        const float rightWeight = plSmooth(-feather, feather, distance);
                        const float defocus = element.facetDefocus * (1.f - plSmooth(0.f, feather, fabsf(distance)));
                        ray.blur = plMax(ray.blur, defocus);
                        OpticRay left = ray;
                        left.weight *= 1.f - rightWeight;
                        left.nextLayer = layer + 1;
                        advanceArrayFacet(stack, element, left, point, width, seam - 1);
                        remaining[pending++] = left;
                        ray.weight *= rightWeight;
                        facet = seam;
                    }
                    advanceArrayFacet(stack, element, ray, point, width, facet);
                    continue;
                }
            }
            PLVec normal;
            const float mask = opticGeometry(element, ray.x, ray.y, normal);
            if (mask > 0) advanceOpticRay(stack, element, ray, normal, index, mask);
        }
        if (ray.coverage <= 0 && ray.nextLayer == 0 && pending == 0) return false;
        const PL4 shaded = shadeOpticRay<WithArrays>(stack, ray, x, y, sample);
        light.r += ray.weight * shaded.r; light.g += ray.weight * shaded.g;
        light.b += ray.weight * shaded.b; light.a += ray.weight * shaded.a;
        if (pending == 0) return true;
        ray = remaining[--pending];
    }
}

template <class Raw, class Sample, class Sharp>
PL_HD inline PL4 opticStackPixel(const OpticStack& stack, int x, int y, const Raw& raw, const Sample& sample, const Sharp& sharp) {
    const float originX = x + 0.5f, originY = y + 0.5f;
    if (stack.mask) {
        const OpticElement& element = stack.elements[stack.mask - 1];
        PLVec normal{};
        const float mask = element.enabled ? opticGeometry(element, originX, originY, normal) : 0;
        return {mask, mask, mask, 1.f};
    }
    if (stack.solid.shape > 0) return glassSolidPixel(stack.solid, stack.encoding, x, y, sample, sharp);
    PL4 result = raw(x, y);
    if (stack.hasArray || stack.channels == 1) {
        PL4 light;
        const bool touched = stack.hasArray
            ? traceOpticChannel<true>(stack, originX, originY, 1, sample, light)
            : traceOpticChannel<false>(stack, originX, originY, 1, sample, light);
        return touched ? PL4{plEncode(light.r, stack.encoding), plEncode(light.g, stack.encoding),
                             plEncode(light.b, stack.encoding), light.a} : result;
    }
    for (int ch = 0; ch < stack.channels; ++ch) {
        PL4 light;
        if (!traceOpticChannel<false>(stack, originX, originY, ch, sample, light)) continue;
        const float value = plEncode(opticChannel(light, ch), stack.encoding);
        if (ch == 0) result.r = value;
        else if (ch == 1) { result.g = value; result.a = light.a; }
        else result.b = value;
    }
    return result;
}

}
