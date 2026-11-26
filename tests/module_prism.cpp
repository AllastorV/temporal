#include "harness.h"

using namespace tpt;

static Clip prismRamp() {
    return makeClip(64, 48, 40, [](int x, int y, int) { return RGBA{x / 64.f, y / 48.f, x / 64.f, 1}; });
}

TP_TEST(prism_static_clip_without_spread_is_identity_in_all_modes) {
    const ModuleDef& m = find("prism");
    Clip c = prismRamp();
    for (int mode = 0; mode < 3; ++mode)
        for (double decay : {0.0, 0.6}) {
            ParamValues p = with(defaultValues(m), {{"mode", mode}, {"spread", 0}, {"decay", decay}, {"encoding", 0}});
            RenderResult r = renderAt(m, c, 20, p);
            CHECK(maxDiff(r.out.view, c.frames[20].view) < 1e-6);
        }
    Clip moving = patternClip();
    ParamValues zero = with(defaultValues(m), {{"mode", 0}, {"red_offset", 0}, {"green_offset", 0}, {"blue_offset", 0}, {"blur", 10}, {"encoding", 0}});
    CHECK(maxDiff(renderAt(m, moving, 20, zero).out.view, moving.frames[20].view) < 1e-6);
}

TP_TEST(prism_temporal_rgb_reads_each_channel_from_its_offset) {
    const ModuleDef& m = find("prism");
    Clip c = makeClip(8, 8, 40, [](int, int, int f) { return RGBA{f / 100.f, f / 100.f, f / 100.f, 1}; });
    ParamValues p = with(defaultValues(m), {{"mode", 0}, {"encoding", 0}, {"red_offset", -4}, {"green_offset", 0}, {"blue_offset", 4}});
    RenderResult r = renderAt(m, c, 20, p);
    RGBA o = r.out.view.load(3, 3);
    CHECK_NEAR(o.r, 0.16, 1e-6);
    CHECK_NEAR(o.g, 0.20, 1e-6);
    CHECK_NEAR(o.b, 0.24, 1e-6);
    CHECK_NEAR(o.a, 1.0, 1e-6);
    CHECK(r.window.before == 4 && r.window.after == 4);
    ParamValues spatialOnly = with(p, {{"mode", 1}});
    CHECK(effectiveWindow(m, spatialOnly, 24).before == 0 && effectiveWindow(m, spatialOnly, 24).after == 0);
}

TP_TEST(prism_decay_averages_toward_the_offset_frame) {
    const ModuleDef& m = find("prism");
    Clip c = makeClip(8, 8, 40, [](int, int, int f) { return RGBA{f / 100.f, f / 100.f, f / 100.f, 1}; });
    ParamValues p = with(defaultValues(m), {{"mode", 0}, {"encoding", 0}, {"red_offset", -4}, {"green_offset", 0}, {"blue_offset", 0}, {"decay", 0.5}});
    RenderResult r = renderAt(m, c, 20, p);
    double num = 0, den = 0;
    for (int j = 0; j <= 4; ++j) { double w = std::pow(0.5, 4 - j); num += w * (20 - j) / 100.0; den += w; }
    CHECK_NEAR(r.out.view.load(3, 3).r, num / den, 1e-6);
    CHECK_NEAR(r.out.view.load(3, 3).g, 0.20, 1e-6);
}

TP_TEST(prism_spatial_dispersion_pushes_red_out_and_blue_in_symmetrically) {
    const ModuleDef& m = find("prism");
    Clip c = prismRamp();
    ParamValues p = with(defaultValues(m), {{"mode", 1}, {"encoding", 0}, {"spread", 8}});
    RenderResult r = renderAt(m, c, 20, p);
    const double k = 8.0 / 40.0;
    for (int x : {10, 50}) {
        RGBA o = r.out.view.load(x, 23);
        double v = x + 0.5 - 32.0;
        CHECK_NEAR(o.r, (x - k * v) / 64.0, 1e-5);
        CHECK_NEAR(o.b, (x + k * v) / 64.0, 1e-5);
        CHECK_NEAR(o.g, c.frames[20].view.load(x, 23).g, 1e-6);
    }
    CHECK(r.out.view.load(50, 23).r < r.out.view.load(50, 23).b);
}

TP_TEST(prism_blur_follows_channel_offset_magnitude) {
    const ModuleDef& m = find("prism");
    Clip c = makeClip(32, 32, 30, [](int x, int y, int) { float v = ((x / 2 + y / 2) % 2) ? 1.f : 0.f; return RGBA{v, v, v, 1}; });
    ParamValues p = with(defaultValues(m), {{"mode", 0}, {"encoding", 0}, {"red_offset", -2}, {"green_offset", 0}, {"blue_offset", 4}, {"blur", 3}});
    RenderResult r = renderAt(m, c, 15, p);
    const ImageView& s = c.frames[15].view;
    double dr = 0, dg = 0, db = 0;
    for (int y = 8; y < 24; ++y)
        for (int x = 8; x < 24; ++x) {
            RGBA o = r.out.view.load(x, y), q = s.load(x, y);
            dr += std::fabs(o.r - q.r); dg += std::fabs(o.g - q.g); db += std::fabs(o.b - q.b);
        }
    CHECK(dg == 0.0);
    CHECK(dr > 1.0);
    CHECK(db > dr);
}

TP_TEST(prism_alpha_is_max_of_channel_alphas) {
    const ModuleDef& m = find("prism");
    Clip c = makeClip(8, 8, 40, [](int, int, int f) { return f >= 20 ? RGBA{1, 1, 1, 1} : RGBA{0, 0, 0, 0}; });
    ParamValues p = with(defaultValues(m), {{"mode", 0}, {"encoding", 0}, {"red_offset", -4}, {"green_offset", 0}, {"blue_offset", 4}});
    RGBA o = renderAt(m, c, 17, p).out.view.load(3, 3);
    CHECK_NEAR(o.r, 0.0, 1e-6);
    CHECK_NEAR(o.g, 0.0, 1e-6);
    CHECK_NEAR(o.b, 1.0, 1e-6);
    CHECK_NEAR(o.a, 1.0, 1e-6);
    RGBA none = renderAt(m, c, 10, p).out.view.load(3, 3);
    CHECK_NEAR(none.a, 0.0, 1e-6);
}

TP_TEST(prism_decay_trails_combine_with_dispersion_in_linear_light) {
    const ModuleDef& m = find("prism");
    Clip c = makeClip(64, 48, 40, [](int x, int, int f) { return RGBA{x / 64.f + f / 100.f, 0.5f, x / 64.f, 1}; });
    ParamValues p = with(defaultValues(m), {{"mode", 2}, {"encoding", 0}, {"red_offset", -4}, {"green_offset", 0}, {"blue_offset", 0}, {"spread", 8}, {"decay", 0.5}});
    RenderResult r = renderAt(m, c, 20, p);
    double num = 0, den = 0;
    for (int j = 0; j <= 4; ++j) { double w = std::pow(0.5, 4 - j); num += w * (20 - j) / 100.0; den += w; }
    const double k = 8.0 / 40.0;
    for (int x : {12, 50}) {
        double v = x + 0.5 - 32.0;
        CHECK_NEAR(r.out.view.load(x, 23).r, (x - k * v) / 64.0 + num / den, 1e-5);
        CHECK_NEAR(r.out.view.load(x, 23).b, (x + k * v) / 64.0, 1e-5);
    }
}
