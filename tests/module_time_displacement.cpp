#include "harness.h"

using namespace tpt;

static Clip timeDisplacementIndexClip() {
    return makeClip(64, 16, 50, [](int, int, int f) { return RGBA{f / 100.f, 0.5f, 0.25f, 1.f}; });
}

TP_TEST(time_displacement_zero_map_returns_current_frame) {
    const ModuleDef& m = find("time_displacement");
    Clip c = makeClip(32, 16, 40, [](int x, int, int f) { return f == 20 ? RGBA{0, 0, 0, 1} : RGBA{0.8f, 0.6f + 0.01f * x, 0.4f, 1}; });
    ParamValues p = with(defaultValues(m), {{"map_source", 0}, {"encoding", 0}, {"range", 16}});
    RenderResult r = renderAt(m, c, 20, p);
    CHECK(maxDiff(r.out.view, c.frames[20].view) == 0.0);
    ParamValues g = with(defaultValues(m), {{"map_source", 1}, {"contrast", 0}, {"invert", 0}, {"direction", 2}});
    Clip pc = patternClip();
    RenderResult rg = renderAt(m, pc, 20, g);
    CHECK(maxDiff(rg.out.view, pc.frames[20].view) == 0.0);
}

TP_TEST(time_displacement_horizontal_gradient_reads_older_frames_across_x) {
    const ModuleDef& m = find("time_displacement");
    Clip c = timeDisplacementIndexClip();
    ParamValues p = with(defaultValues(m), {{"map_source", 1}, {"map_blur", 0}, {"gradient_angle", 0}, {"radial", 0}, {"direction", 0},
                                            {"range", 16}, {"interpolation", 0}, {"encoding", 0}});
    RenderResult r = renderAt(m, c, 30, p);
    int prev = 1000;
    for (int x = 0; x < 64; ++x) {
        int f = static_cast<int>(std::lround(r.out.view.load(x, 8).r * 100.f));
        int expect = 30 - static_cast<int>(std::lround((x + 0.5) / 64.0 * 16.0));
        CHECK(f == expect);
        CHECK(f <= prev);
        prev = f;
    }
    CHECK(std::lround(r.out.view.load(0, 8).r * 100.f) == 30);
    CHECK(std::lround(r.out.view.load(63, 8).r * 100.f) == 14);
    CHECK(r.window.before == 16 && r.window.after == 0);
}

TP_TEST(time_displacement_blend_is_linear_in_time) {
    const ModuleDef& m = find("time_displacement");
    Clip c = timeDisplacementIndexClip();
    ParamValues p = with(defaultValues(m), {{"map_source", 1}, {"map_blur", 0}, {"gradient_angle", 0}, {"direction", 1},
                                            {"range", 10}, {"interpolation", 1}, {"encoding", 0}});
    RenderResult r = renderAt(m, c, 20, p);
    for (int x = 0; x < 64; ++x) CHECK_NEAR(r.out.view.load(x, 3).r * 100.0, 20.0 + (x + 0.5) / 64.0 * 10.0, 1e-3);
    CHECK(r.window.after == 10 && r.window.before == 0);
}

TP_TEST(time_displacement_both_is_centered) {
    const ModuleDef& m = find("time_displacement");
    Clip c = timeDisplacementIndexClip();
    ParamValues p = with(defaultValues(m), {{"map_source", 1}, {"map_blur", 0}, {"gradient_angle", 0}, {"direction", 2},
                                            {"range", 16}, {"interpolation", 1}, {"encoding", 0}});
    RenderResult r = renderAt(m, c, 25, p);
    CHECK(r.window.before == 8 && r.window.after == 8);
    for (int x = 0; x < 64; ++x) CHECK_NEAR(r.out.view.load(x, 3).r * 100.0, 25.0 + ((x + 0.5) / 64.0 - 0.5) * 16.0, 1e-3);
}

TP_TEST(time_displacement_static_noise_shifts_with_time) {
    const ModuleDef& m = find("time_displacement");
    Clip c = timeDisplacementIndexClip();
    ParamValues p = with(defaultValues(m), {{"map_source", 2}, {"noise_scale", 12}, {"noise_evolution", 0}, {"encoding", 0}, {"range", 12}});
    RenderResult a = renderAt(m, c, 30, p), b = renderAt(m, c, 31, p);
    double spread = 0;
    for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 64; ++x) {
            CHECK_NEAR(b.out.view.load(x, y).r - a.out.view.load(x, y).r, 0.01, 1e-5);
            spread = std::max(spread, static_cast<double>(std::fabs(a.out.view.load(x, y).r - a.out.view.load(0, 0).r)));
        }
    CHECK(spread > 0.01);
    ParamValues s2 = with(p, {{"seed", 77}});
    CHECK(maxDiff(renderAt(m, c, 30, s2).out.view, a.out.view) > 0.005);
    ParamValues blurred = with(p, {{"map_blur", 40}});
    RenderResult bl = renderAt(m, c, 30, blurred);
    RGBA ma = mean(a.out.view), mb = mean(bl.out.view);
    CHECK(std::fabs(ma.r - mb.r) < 0.03);
    auto variation = [](const ImageView& v) {
        double s = 0;
        for (int y = 0; y < 16; ++y)
            for (int x = 1; x < 64; ++x) s += std::fabs(v.load(x, y).r - v.load(x - 1, y).r);
        return s;
    };
    CHECK(maxDiff(bl.out.view, a.out.view) > 0.005);
    CHECK(variation(bl.out.view) < 0.5 * variation(a.out.view));
    Clip small = makeClip(32, 8, 50, [](int, int, int f) { return RGBA{f / 100.f, 0.5f, 0.25f, 1.f}; });
    ParamValues nearest = with(p, {{"interpolation", 0}});
    RenderResult half = renderAt(m, small, 30, nearest, 0.5);
    RenderResult fullRes = renderAt(m, c, 30, nearest);
    int mismatch = 0;
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 32; ++x)
            mismatch += std::fabs(half.out.view.load(x, y).r - fullRes.out.view.load(2 * x, 2 * y).r) > 0.025f ? 1 : 0;
    CHECK(mismatch < 26);
}
