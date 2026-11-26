#include "harness.h"
#include "gpu/gpu.h"

using namespace tpt;

TP_TEST(lens_character_is_neutral_with_all_optical_controls_zero) {
    const ModuleDef& m = find("lens_character");
    Clip clip = patternClip(101, 81, 2);
    auto p = with(defaultValues(m), {{"lens_distortion", 0}, {"lens_chroma", 0}, {"lens_shading", 0},
        {"lens_edge_blur", 0}, {"lens_tint", 0}, {"encoding", 0}});
    const auto result = renderAt(m, clip, 1, p);
    CHECK(maxDiff(result.out.view, clip.frames[1].view) < 1e-6);
}

TP_TEST(lens_character_controls_field_color_sensor_and_aperture) {
    const ModuleDef& m = find("lens_character");
    Clip clip = makeClip(161, 101, 2, [](int x, int, int) {
        const float v = x & 1 ? 0.85f : 0.15f;
        return RGBA{v, v, v, 1};
    });
    auto p = with(defaultValues(m), {{"lens_model", 3}, {"lens_chroma", 4}, {"lens_edge_blur", 0},
        {"lens_distortion", 0}, {"lens_tint", 0}, {"encoding", 0}});
    const auto open = renderAt(m, clip, 1, p);
    const auto stopped = renderAt(m, clip, 1, with(p, {{"lens_stop", 8}}));
    const auto cropped = renderAt(m, clip, 1, with(p, {{"sensor_diagonal", 12}}));
    const auto outside = renderAt(m, clip, 1, with(p, {{"sensor_diagonal", 40}}));
    const RGBA center = open.out.view.load(80, 50);
    CHECK(std::fabs(center.r - center.b) < 1e-6f);
    float edgeColor = 0;
    for (int x = 140; x < 160; ++x) {
        const RGBA pixel = open.out.view.load(x, 50);
        edgeColor = std::max(edgeColor, std::fabs(pixel.r - pixel.b));
    }
    CHECK(edgeColor > 0.025f);
    CHECK(stopped.out.view.load(153, 50).g > open.out.view.load(153, 50).g);
    CHECK(cropped.out.view.load(153, 50).g > open.out.view.load(153, 50).g);
    CHECK(outside.out.view.load(160, 100).g == 0.f);
}

TP_TEST(lens_character_blur_switch_preserves_the_lens_look) {
    const ModuleDef& m = find("lens_character");
    Clip clip = patternClip(128, 96, 2);
    auto p = with(defaultValues(m), {{"lens_model", 3}, {"fallback_blur", 9}});
    const auto active = renderAt(m, clip, 1, p);
    const auto disabled = renderAt(m, clip, 1, with(p, {{"lens_blur", 0}}));
    const auto zero = renderAt(m, clip, 1, with(p, {{"lens_blur_amount", 0}}));
    CHECK(maxDiff(disabled.out.view, zero.out.view) < 1e-6);
    CHECK(maxDiff(disabled.out.view, clip.frames[1].view) > 0.05);
    CHECK(maxDiff(active.out.view, disabled.out.view) > 0.1);
}

TP_TEST(lens_character_depth_focus_matches_cpu_and_cuda) {
    const ModuleDef& m = find("lens_character");
    Clip clip = makeClip(160, 100, 2, [](int x, int, int) {
        const float v = x & 1 ? 0.8f : 0.2f;
        return RGBA{v, v, v, 1};
    });
    OwnedImage depth({5, 7, 85, 57}), cpu(clip.rect), gpuRender(clip.rect);
    for (int y = 7; y < 57; ++y) for (int x = 5; x < 85; ++x) {
        const float v = x < 45 ? 0.f : 1.f;
        depth.view.store(x, y, {v, v, v, 1});
    }
    auto p = with(defaultValues(m), {{"lens_model", 1}, {"lens_distortion", 0}, {"lens_chroma", 0},
        {"lens_shading", 0}, {"lens_edge_blur", 0}, {"lens_tint", 0}, {"lens_focus", 0.62},
        {"depth_near", 0.62}, {"depth_far", 25}, {"encoding", 0}});
    RenderContext ctx;
    ctx.params = &p;
    ctx.frame = ctx.window = clip.rect;
    ctx.secondSource = depth.view;
    ctx.fetchRaw = [&](double) { return clip.frames[0].view; };
    const bool enabled = gpu::enabled();
    struct RestoreGpu { bool enabled; ~RestoreGpu(){ gpu::setEnabled(enabled); } } restore{enabled};
    gpu::setEnabled(false);
    ctx.dst = cpu.view;
    runModule(m, ctx);
    CHECK(std::fabs(cpu.view.load(40, 50).r - 0.2f) < 1e-6f);
    CHECK(std::fabs(cpu.view.load(120, 50).r - 0.5f) < 0.15f);
    gpu::setEnabled(enabled);
    if (gpu::available()) {
        ctx.dst = gpuRender.view;
        runModule(m, ctx);
        CHECK(maxDiff(cpu.view, gpuRender.view) < 0.01);
    }
}

TP_TEST(lens_character_preview_keeps_resized_depth_aligned) {
    const ModuleDef& m = find("lens_character");
    const Clip clip = makeClip(1920, 160, 1, [](int x, int, int) {
        const float v = (x / 6) % 2 ? 0.85f : 0.15f;
        return RGBA{v, v, v, 1};
    });
    OwnedImage fullDepth(clip.rect), smallDepth({19, 7, 499, 47});
    for (ImageView view : {fullDepth.view, smallDepth.view}) {
        for (int y = view.bounds.y1; y < view.bounds.y2; ++y) {
            for (int x = view.bounds.x1; x < view.bounds.x2; ++x) {
                const float depth = 0.1f + 0.8f * (x + 0.5f - view.bounds.x1) / view.bounds.w();
                view.store(x, y, {depth, depth, depth, 1});
            }
        }
    }
    const auto p = with(defaultValues(m), {{"lens_distortion", 0}, {"lens_chroma", 0},
        {"lens_shading", 0}, {"lens_edge_blur", 0}, {"lens_tint", 0}, {"encoding", 0}});
    OwnedImage fullResult(clip.rect), smallResult(clip.rect);
    RenderContext ctx;
    ctx.params = &p;
    ctx.frame = ctx.window = clip.rect;
    ctx.preview = true;
    ctx.fetchRaw = [&](double) { return clip.frames[0].view; };
    ctx.secondSource = fullDepth.view;
    ctx.dst = fullResult.view;
    runModule(m, ctx);
    ctx.secondSource = smallDepth.view;
    ctx.dst = smallResult.view;
    runModule(m, ctx);
    CHECK(maxDiff(fullResult.view, smallResult.view) < 0.005f);
}

TP_TEST(lens_character_warps_matte_alpha_like_image_geometry) {
    const ModuleDef& m = find("lens_character");
    const Clip matte = makeClip(256, 128, 1, [](int x, int, int) {
        return RGBA{0, 0, 0, x >= 240 ? 1.f : 0.f};
    });
    const Clip picture = makeClip(256, 128, 1, [](int x, int, int) {
        const float v = x >= 240 ? 1.f : 0.f;
        return RGBA{v, v, v, 1};
    });
    const auto p = with(defaultValues(m), {{"lens_model", 2}, {"lens_focal", 28},
        {"lens_chroma", 0}, {"lens_shading", 0}, {"lens_edge_blur", 0},
        {"lens_tint", 0}, {"lens_blur", 0}, {"encoding", 0}});
    const auto warpedMatte = renderAt(m, matte, 0, p);
    const auto warpedPicture = renderAt(m, picture, 0, p);
    CHECK(warpedMatte.out.view.load(240, 64).a < 0.5f);
    for (int x = 230; x < 256; ++x)
        CHECK(std::fabs(warpedMatte.out.view.load(x, 64).a - warpedPicture.out.view.load(x, 64).r) < 1e-6f);
}
