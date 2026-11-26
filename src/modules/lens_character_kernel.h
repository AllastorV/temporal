#pragma once
#include "print_lab_kernel.h"

namespace tp {

constexpr int kLensApertureTaps = 32;

struct LensCharacterParams {
    float centerX = 0, centerY = 0, radius = 1, coverage = 1;
    float distortion = 0, lateralColor = 0, cornerStops = 0, edgeBlur = 0;
    float transmission[3] = {1, 1, 1};
    float focusMeters = 3, cocInfinity = 0, maximumBlur = 0, depthNear = 0.5f, depthFar = 30.f;
    float fallbackBlur = 0, catEye = 0;
    float disk[kLensApertureTaps][2]{};
    float weights[kLensApertureTaps]{};
    int encoding = 0, hasDepth = 0;
};

PL_HD inline float lensDistance(const LensCharacterParams& lens, float depth) {
    return plLerp(lens.depthNear, lens.depthFar, plSat(depth));
}

PL_HD inline float lensBlurRadius(const LensCharacterParams& lens, float distance) {
    return plMin(lens.maximumBlur, lens.cocInfinity * fabsf(1.f - lens.focusMeters / distance));
}

template <class Source, class Depth>
PL_HD inline PL4 lensCharacterPixel(const LensCharacterParams& lens, int x, int y, const Source& source, const Depth& depth) {
    const float dx = x + 0.5f - lens.centerX, dy = y + 0.5f - lens.centerY;
    const float field2 = (dx * dx + dy * dy) / (lens.radius * lens.radius);
    const float field = sqrtf(field2);
    const float scale = 1.f + lens.distortion * field2;
    const float sx = lens.centerX + dx * scale, sy = lens.centerY + dy * scale;
    const float distance = lens.hasDepth ? lensDistance(lens, depth(sx, sy).r) : lens.focusMeters;
    const float defocus = lens.hasDepth ? lensBlurRadius(lens, distance) : lens.fallbackBlur;
    const float fieldBlur = lens.edgeBlur * field2;
    const float blur = sqrtf(defocus * defocus + fieldBlur * fieldBlur);
    const float radialX = field > 0 ? dx / (field * lens.radius) : 0.f;
    const float radialY = field > 0 ? dy / (field * lens.radius) : 0.f;
    const float fringe = lens.lateralColor * field2 * field;
    const float flatten = 1.f - plMin(0.90f, lens.catEye * field2);
    PL4 color;
    float sum = 0.f;
    const int count = blur > 1.25f ? kLensApertureTaps : (blur > 0.35f ? 4 : 1);
    for (int tap = 0; tap < count; ++tap) {
        const int point = count == 4 ? tap * 8 + 3 : tap;
        const float ox = count == 1 ? 0.f : lens.disk[point][0] * blur;
        const float oy = count == 1 ? 0.f : lens.disk[point][1] * blur;
        const float radial = (ox * radialX + oy * radialY) * (1.f - flatten);
        const float tx = sx + ox - radial * radialX, ty = sy + oy - radial * radialY;
        float weight = count == 1 ? 1.f : lens.weights[point];
        if (lens.hasDepth && count > 1) {
            const float sampleDistance = lensDistance(lens, depth(tx, ty).r);
            if (sampleDistance < distance) {
                const float separation = plSmooth(0.04f, 0.18f, 1.f - sampleDistance / distance);
                const float reach = lensBlurRadius(lens, sampleDistance);
                const float offset = sqrtf((tx - sx) * (tx - sx) + (ty - sy) * (ty - sy));
                weight *= 1.f - separation * plSmooth(reach, reach + 1.f, offset);
            }
        }
        if (weight <= 0) continue;
        const PL4 sample = source(tx, ty);
        color.r += sample.r * weight; color.g += sample.g * weight;
        color.b += sample.b * weight; color.a += sample.a * weight;
        sum += weight;
    }
    if (sum <= 0) { color = source(sx, sy); sum = 1.f; }
    if (fabsf(fringe) > 0.01f) {
        const PL4 center = source(sx, sy);
        const float attenuation = sum / (1.f + blur);
        color.r += (source(sx - 0.62f * fringe * radialX, sy - 0.62f * fringe * radialY).r - center.r) * attenuation;
        color.b += (source(sx + fringe * radialX, sy + fringe * radialY).b - center.b) * attenuation;
    }
    const float illumination = exp2f(-lens.cornerStops * field2 * field2) *
        (1.f - plSmooth(1.f, 1.05f, field * lens.coverage));
    return {plEncode(color.r * illumination * lens.transmission[0] / sum, lens.encoding),
            plEncode(color.g * illumination * lens.transmission[1] / sum, lens.encoding),
            plEncode(color.b * illumination * lens.transmission[2] / sum, lens.encoding), color.a / sum};
}

}
