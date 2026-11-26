#include "modules.h"
#include "../gpu/gpu.h"

namespace tp {
namespace {

struct LensProfile {
    const char* label;
    // Published focal length, aperture marking, image circle or supported diagonal,
    // minimum focus and diaphragm blades (zero means the count was not published).
    float focalMin, focalMax, open, closed, circle, nearFocus;
    int blades;
    // Look estimates at the full image circle. These are not calibration data.
    float wideDistortion, teleDistortion, lateralColor, cornerStops, edgeBlur, rounding, feather, catEye, warmth;
};

constexpr LensProfile kProfiles[] = {
    {"Large-Format Cine Prime 47mm T1.8", 47, 47, 1.8f, 22, 46, 0.45f, 0,
     -0.0020f, -0.0020f, 0.00045f, 0.33f, 1.10f, 1.f, 0.85f, 0.28f, 0.012f},
    {"Cine Prime 50mm T1.4", 50, 50, 1.4f, 22, 33, 0.60f, 0,
      0.0015f,  0.0015f, 0.00030f, 0.24f, 0.95f, 1.f, 0.73f, 0.20f, 0.007f},
    {"Large-Format Cine Zoom 28-100mm T2.9", 28, 100, 2.9f, 22, 46.3f, 0.80f, 13,
     -0.0100f,  0.0060f, 0.00038f, 0.25f, 0.80f, 0.65f, 0.42f, 0.17f, 0.0f},
    {"Mirrorless Prime 25mm F1.2", 25, 25, 1.2f, 16, 21.64f, 0.30f, 9,
     -0.0060f, -0.0060f, 0.00060f, 0.60f, 2.60f, 0.84f, 1.45f, 0.38f, 0.0f},
};

void setLensAperture(LensCharacterParams& lens, const LensProfile& profile, float stop) {
    const float stopped = saturate(std::log2(stop / profile.open) / 2.f);
    const float roundness = lerpf(profile.blades ? 0.92f : 1.f, profile.rounding, stopped);
    const float sector = profile.blades ? static_cast<float>(2 * kPi / profile.blades) : 0.f;
    for (int i = 0; i < kLensApertureTaps; ++i) {
        const float angle = i * 2.39996323f;
        const float r = std::sqrt((i + 0.5f) / kLensApertureTaps);
        float boundary = 1.f;
        if (profile.blades) {
            const float local = std::remainder(angle, sector);
            boundary = lerpf(std::cos(0.5f * sector) / std::cos(local), 1.f, roundness);
        }
        lens.disk[i][0] = std::cos(angle) * r * boundary;
        lens.disk[i][1] = std::sin(angle) * r * boundary;
        lens.weights[i] = std::exp(-profile.feather * r * r);
    }
}

}

ModuleDef makeLensCharacter() {
    ModuleDef m;
    m.id = "lens_character";
    m.cuda = CudaMode::Device;
    m.name = "Lens Character";
    m.family = "Optics";
    m.description = "Datasheet-informed cinema and photo lens profiles with field distortion, shading, edge color and aperture-shaped image blur. An optional linear Depth Map enables distance-aware defocus.";
    m.usesEncoding = true;
    std::vector<std::string> models;
    for (const LensProfile& profile : kProfiles) models.emplace_back(profile.label);
    m.params = {
        Choice("lens_model", "Lens", std::move(models), 0, "Published lens specifications inform these adjustable look approximations."),
        Dbl("lens_stop", "Aperture (T/F)", 0, 0, 22, 0, 22, Unit::None, "Zero selects the lens's published maximum aperture. Cinema T markings approximate the geometric aperture."),
        Dbl("lens_focal", "Zoom Focal Length (mm)", 28, 28, 100, 28, 100, Unit::None, "Sets the zoom focal length and interpolates its geometric character. Primes use their fixed focal length."),
        Dbl("sensor_diagonal", "Active Sensor Diagonal (mm)", 0, 0, 65, 0, 50, Unit::None, "Zero uses the selected lens's supported format. A larger value samples farther out in its image circle."),
        Point("optical_center", "Optical Center", 0.5, 0.5, "Position of the lens axis in the image."),
        Dbl("lens_distortion", "Distortion Strength", 1, -3, 3, -2, 2, Unit::None, "Scales or reverses the selected profile's mild radial distortion.", "Optical Character"),
        Dbl("lens_chroma", "Edge Color Strength", 1, 0, 4, 0, 2, Unit::None, "Scales lateral color. Wavelengths coincide on the optical axis.", "Optical Character"),
        Dbl("lens_shading", "Corner Shading", 1, 0, 4, 0, 2, Unit::None, "Scales the profile's estimated illumination falloff.", "Optical Character"),
        Dbl("lens_edge_blur", "Edge Softness", 1, 0, 4, 0, 2, Unit::None, "Scales softness away from the optical axis.", "Optical Character"),
        Dbl("lens_tint", "Lens Tint Strength", 1, 0, 4, 0, 2, Unit::None, "Scales the profile's subtle estimated optical tint.", "Optical Character"),
        Bool("lens_blur", "Lens Blur", true, "Enables distance-based or manual aperture blur while preserving the optical character.", "Depth and Defocus"),
        Dbl("lens_blur_amount", "Blur Amount", 1, 0, 4, 0, 2, Unit::None, "Scales the selected lens's aperture blur.", "Depth and Defocus"),
        Dbl("lens_focus", "Focus Distance (m)", 3, 0.10, 1000, 0.3, 20, Unit::None, "Distance in focus. Clamped to the published minimum focus distance.", "Depth and Defocus"),
        Dbl("depth_near", "Depth Black Distance (m)", 0.5, 0.10, 1000, 0.2, 10, Unit::None, "Black in the optional linear Depth Map represents this distance.", "Depth and Defocus"),
        Dbl("depth_far", "Depth White Distance (m)", 30, 0.10, 10000, 1, 100, Unit::None, "White in the optional linear Depth Map represents this distance.", "Depth and Defocus"),
        Dbl("maximum_blur", "Maximum Defocus", 32, 0, 120, 0, 60, Unit::Pixels, "Caps the aperture blur radius in the supplied depth map mode.", "Depth and Defocus"),
        Dbl("fallback_blur", "Manual Defocus", 0, 0, 120, 0, 60, Unit::Pixels, "Uniform aperture blur at the model's open stop when a depth map is absent.", "Depth and Defocus"),
        Choice("lens_outside", "Outside Frame", {"Clamp", "Mirror", "Transparent"}, 1, "Source sampling beyond the image boundary.", "Optical Character"),
        Bool("lens_fusion_controls", "Fusion Focus Controls", false, "Set by the Temporal Fusion macro."),
        Dbl("focus_distance", "AI Focus Distance", 0.75, 0, 1, 0, 1, Unit::Normalized, "Relative distance from near to far.", "Fusion Focus"),
        Dbl("focus_range", "In-Focus Range", 0.12, 0, 1, 0, 1, Unit::Normalized, "Width of the sharp band.", "Fusion Focus"),
        Dbl("focus_tilt_x", "Focus Tilt X", 0, -0.5, 0.5, -0.5, 0.5, Unit::None, "Left to right focus-plane slope.", "Fusion Focus"),
        Dbl("focus_tilt_y", "Focus Tilt Y", 0, -0.5, 0.5, -0.5, 0.5, Unit::None, "Bottom to top focus-plane slope.", "Fusion Focus"),
        Bool("blur_foreground", "Blur Foreground", false, "Blur the near side of the focus band.", "Fusion Focus"),
        Bool("use_focus_pull", "Use A/B/C Focus Pull", false, "Use the three focus marks.", "Focus Pull"),
        Dbl("focus_a", "Focus A", 0.78, 0, 1, 0, 1, Unit::Normalized, "First focus mark.", "Focus Pull"),
        Dbl("focus_b", "Focus B", 0.52, 0, 1, 0, 1, Unit::Normalized, "Second focus mark.", "Focus Pull"),
        Dbl("focus_c", "Focus C", 0.28, 0, 1, 0, 1, Unit::Normalized, "Third focus mark.", "Focus Pull"),
        Dbl("pull_position", "Pull Position (0 A / 1 B / 2 C)", 0, 0, 2, 0, 2, Unit::None, "Interpolate between focus marks.", "Focus Pull"),
        Bool("focus_color", "Focus Color Preview", false, "Show the green focus guide.", "Fusion Focus"),
        Dbl("matte_strength", "Subject Matte Strength", 0, 0, 1, 0, 1, Unit::Normalized, "Weight of the connected matte.", "Fusion Focus"),
        Dbl("matte_feather", "Matte Feather (px)", 0.8, 0, 4, 0, 4, Unit::Pixels, "Soften the mask boundary.", "Fusion Focus"),
    };
    m.framesNeeded = [](const ParamValues&, double) { return FrameWindow{}; };
    m.render = [](const RenderContext& c) {
        const auto src = c.source();
        if (!src.valid()) {
            requireHost(c);
            copyRect(src, c.dst, c.window);
            return;
        }
        const ParamValues& p = c.p();
        const LensProfile& profile = kProfiles[std::clamp(p.i("lens_model"), 0, static_cast<int>(std::size(kProfiles)) - 1)];
        const float focal = clampf(p.f("lens_focal"), profile.focalMin, profile.focalMax);
        const float zoom = profile.focalMax > profile.focalMin ? (focal - profile.focalMin) / (profile.focalMax - profile.focalMin) : 0.f;
        const float stop = p.f("lens_stop") > 0 ? clampf(p.f("lens_stop"), profile.open, profile.closed) : profile.open;
        const float openRatio = profile.open / stop;
        const float sensor = p.f("sensor_diagonal") > 0 ? p.f("sensor_diagonal") : profile.circle;
        const float diagonal = std::hypot(static_cast<float>(c.frame.w()), static_cast<float>(c.frame.h()));
        LensCharacterParams lens;
        lens.encoding = static_cast<int>(c.encoding);
        lens.coverage = sensor / profile.circle;
        lens.radius = std::max(1.f, 0.5f * diagonal);
        const float fieldScale = lens.coverage * lens.coverage;
        double x, y;
        c.pointToPixels(p.vec("optical_center"), x, y);
        lens.centerX = static_cast<float>(x);
        lens.centerY = static_cast<float>(y);
        lens.distortion = lerpf(profile.wideDistortion, profile.teleDistortion, zoom) * fieldScale * p.f("lens_distortion");
        lens.lateralColor = lens.radius * profile.lateralColor * fieldScale * p.f("lens_chroma");
        lens.cornerStops = profile.cornerStops * fieldScale * fieldScale * (0.20f + 0.80f * openRatio) * p.f("lens_shading");
        lens.edgeBlur = c.px(profile.edgeBlur) * fieldScale * (0.35f + 0.65f * openRatio) * p.f("lens_edge_blur");
        lens.transmission[0] = 1.f + profile.warmth * p.f("lens_tint");
        lens.transmission[2] = 1.f - profile.warmth * p.f("lens_tint");
        lens.focusMeters = std::max(profile.nearFocus, p.f("lens_focus"));
        const float pixelsPerMm = diagonal / sensor;
        const float blurAmount = p.b("lens_blur") ? p.f("lens_blur_amount") : 0.f;
        lens.cocInfinity = blurAmount * 0.5f * pixelsPerMm * focal * focal / (stop * (1000.f * lens.focusMeters - focal));
        lens.maximumBlur = c.px(p.f("maximum_blur"));
        lens.fallbackBlur = c.px(p.f("fallback_blur")) * openRatio * blurAmount;
        lens.depthNear = p.f("depth_near");
        lens.depthFar = p.f("depth_far");
        lens.hasDepth = blurAmount > 0 && c.secondSource.valid();
        lens.catEye = std::min(0.8f, profile.catEye * openRatio * fieldScale);
        setLensAperture(lens, profile, stop);

        const auto edge = static_cast<SampleEdge>(std::clamp(p.i("lens_outside"), 0, 2));
        if (gpu::lensCharacter(lens, src, c.secondSource, c.dst, c.window, edge)) return;
        requireHost(c);
        OwnedImage decoded;
        ImageView linear = src;
        if (c.encoding != Encoding::Linear) {
            decoded.reset(src.bounds);
            parallelRows(src.bounds.y1, src.bounds.y2, [&](int y) {
                for (int x = src.bounds.x1; x < src.bounds.x2; ++x) decoded.view.store(x, y, decode(src.load(x, y), c.encoding));
            });
            linear = decoded.view;
        }
        const auto sample = [&](float x, float y) {
            const RGBA v = linear.bilinear(x, y, edge);
            return PL4{v.r, v.g, v.b, v.a};
        };
        const auto depth = [&](float x, float y) {
            const Rect& d = c.secondSource.bounds;
            const float u = (x - src.bounds.x1) * d.w() / src.bounds.w();
            const float v = (y - src.bounds.y1) * d.h() / src.bounds.h();
            return PL4{c.secondSource.bilinear(d.x1 + u, d.y1 + v).r, 0, 0, 1};
        };
        parallelRows(c.window.y1, c.window.y2, [&](int y) {
            for (int x = c.window.x1; x < c.window.x2; ++x) {
                const PL4 v = lensCharacterPixel(lens, x, y, sample, depth);
                c.dst.store(x, y, {v.r, v.g, v.b, v.a});
            }
        });
    };
    return m;
}

}
