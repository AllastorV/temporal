#include "harness.h"

using namespace tpt;

TP_TEST(time_shift_static_constant_clip_is_identity_in_all_modes) {
    const ModuleDef& m = find("time_shift");
    Clip c = constantClip(RGBA{0.3f, 0.5f, 0.7f, 1.f});
    for (int mode = 0; mode < 3; ++mode)
        for (int comb = 0; comb < 2; ++comb) {
            ParamValues p = with(defaultValues(m), {{"mode", mode}, {"combine", comb}, {"encoding", 0}, {"coverage", 1}});
            RenderResult r = renderAt(m, c, 20, p);
            CHECK(maxDiff(r.out.view, c.frames[20].view) < 1e-6);
        }
}

TP_TEST(time_shift_light_trails_impulse_matches_analytic_weight) {
    const ModuleDef& m = find("time_shift");
    Clip c = makeClip(8, 8, 30, [](int, int, int f) { return f == 12 ? RGBA{1, 1, 1, 1} : RGBA{0, 0, 0, 1}; });
    ParamValues p = with(defaultValues(m), {{"encoding", 0}, {"offset_frames", 8}, {"detail", 8}, {"light_response", 0}, {"streak_length", 0}});
    RenderResult r = renderAt(m, c, 20, p);
    auto fade = [](double f) { return std::pow(std::max(0.0, 1.0 - 0.92 * f), 1.3); };
    double norm = 0;
    for (int j = 1; j <= 8; ++j) norm += fade(j / 8.0);
    CHECK_NEAR(r.out.view.load(3, 3).r, 1.0 - std::exp(-2.5 * fade(1.0) / norm), 1e-4);
    CHECK_NEAR(r.out.view.load(3, 3).a, 1.0, 1e-6);
    CHECK(r.window.before == 8 && r.window.after == 0);
    CHECK_NEAR(renderAt(m, c, 21, p).out.view.load(3, 3).r, 0.0, 1e-6);
}

TP_TEST(time_shift_average_impulse_is_normalized) {
    const ModuleDef& m = find("time_shift");
    Clip c = makeClip(8, 8, 30, [](int, int, int f) { return f == 12 ? RGBA{1, 1, 1, 1} : RGBA{0, 0, 0, 1}; });
    ParamValues p = with(defaultValues(m), {{"combine", 1}, {"coverage", 1}, {"encoding", 0}, {"offset_frames", 8}, {"detail", 8}, {"light_response", 0}, {"streak_length", 0}});
    auto fade = [](double f) { return std::pow(std::max(0.0, 1.0 - 0.92 * f), 1.3); };
    double wsum = 0;
    for (int j = 1; j <= 8; ++j) wsum += fade(j / 8.0);
    CHECK_NEAR(renderAt(m, c, 20, p).out.view.load(3, 3).r, fade(1.0) / (1.0 + wsum), 1e-4);
}

TP_TEST(time_shift_streak_is_continuous_and_angle_flips_it) {
    const ModuleDef& m = find("time_shift");
    Clip c = makeClip(48, 8, 20, [](int x, int, int) { return x == 10 ? RGBA{1, 1, 1, 1} : RGBA{0, 0, 0, 1}; });
    ParamValues p = with(defaultValues(m), {{"encoding", 0}, {"offset_frames", 8}, {"detail", 2}, {"streak_length", 16}, {"angle", 0}, {"light_response", 0}, {"coverage", 1}});
    RenderResult r = renderAt(m, c, 10, p);
    CHECK_NEAR(r.out.view.load(10, 4).r, 1.0, 1e-6);
    for (int x = 11; x <= 22; ++x) CHECK(r.out.view.load(x, 4).r > 0.02f);
    for (int x = 0; x < 8; ++x) CHECK(r.out.view.load(x, 4).r < 1e-6f);
    CHECK(r.out.view.load(31, 4).r < 1e-6f);
    CHECK(r.out.view.load(12, 4).r > r.out.view.load(24, 4).r);
    RenderResult flip = renderAt(m, c, 10, with(p, {{"angle", 180}}));
    for (int x = 1; x <= 8; ++x) CHECK(flip.out.view.load(x, 4).r > 0.02f);
    CHECK(flip.out.view.load(15, 4).r < 1e-6f);
}

TP_TEST(time_shift_light_trails_never_darken_current_frame) {
    const ModuleDef& m = find("time_shift");
    Clip c = patternClip();
    for (int mode = 0; mode < 3; ++mode) {
        RenderResult r = renderAt(m, c, 20, with(defaultValues(m), {{"mode", mode}, {"encoding", 0}}));
        const ImageView& s = c.frames[20].view;
        for (int y = 0; y < c.rect.y2; ++y)
            for (int x = 0; x < c.rect.x2; ++x)
                for (int ch = 0; ch < 3; ++ch) CHECK(r.out.view.load(x, y)[ch] >= s.load(x, y)[ch] - 1e-6f);
    }
}

TP_TEST(time_shift_light_response_favors_highlights) {
    const ModuleDef& m = find("time_shift");
    Clip c = makeClip(8, 8, 30, [](int x, int, int f) { return f == 12 ? (x < 4 ? RGBA{0.25f, 0.25f, 0.25f, 1} : RGBA{1, 1, 1, 1}) : RGBA{0, 0, 0, 1}; });
    ParamValues p = with(defaultValues(m), {{"encoding", 0}, {"offset_frames", 8}, {"detail", 8}, {"streak_length", 0}, {"coverage", 1}});
    double prev = -1;
    for (double lr : {0.0, 1.0, 2.0, 4.0}) {
        RenderResult r = renderAt(m, c, 20, with(p, {{"light_response", lr}}));
        double ratio = r.out.view.load(6, 3).r / std::max(1e-9f, r.out.view.load(1, 3).r);
        CHECK(ratio > prev);
        prev = ratio;
    }
}

TP_TEST(time_shift_radial_drag_leaves_center_pixel_in_place) {
    const ModuleDef& m = find("time_shift");
    Clip c = makeClip(33, 33, 20, [](int x, int y, int) { return RGBA{x / 33.f, y / 33.f, 0.25f, 1}; });
    ParamValues p = with(defaultValues(m), {{"mode", 2}, {"combine", 1}, {"coverage", 1}, {"encoding", 0}, {"streak_length", 200}});
    RenderResult r = renderAt(m, c, 10, p);
    CHECK(maxDiff(r.out.view, c.frames[10].view) > 0.01);
    RGBA o = r.out.view.load(16, 16), s = c.frames[10].view.load(16, 16);
    CHECK_NEAR(o.r, s.r, 2e-3);
    CHECK_NEAR(o.g, s.g, 2e-3);
}

TP_TEST(time_shift_pull_positive_moves_past_down_negative_up) {
    const ModuleDef& m = find("time_shift");
    Clip c = makeClip(8, 32, 20, [](int, int y, int) { return y == 16 ? RGBA{1, 1, 1, 1} : RGBA{0, 0, 0, 1}; });
    ParamValues p = with(defaultValues(m), {{"mode", 1}, {"encoding", 0}, {"offset_frames", 8}, {"detail", 2}, {"streak_length", 8}, {"light_response", 0}, {"coverage", 1}});
    RenderResult down = renderAt(m, c, 10, p);
    CHECK(down.out.view.load(3, 12).r > 0.02f);
    CHECK(down.out.view.load(3, 9).r > 0.02f);
    CHECK(down.out.view.load(3, 20).r < 1e-6f);
    RenderResult up = renderAt(m, c, 10, with(p, {{"streak_length", -8}}));
    CHECK(up.out.view.load(3, 20).r > 0.02f);
    CHECK(up.out.view.load(3, 23).r > 0.02f);
    CHECK(up.out.view.load(3, 12).r < 1e-6f);
}

TP_TEST(time_shift_streak_length_is_in_full_resolution_pixels) {
    const ModuleDef& m = find("time_shift");
    Clip c = makeClip(48, 8, 20, [](int x, int, int) { return x == 10 ? RGBA{1, 1, 1, 1} : RGBA{0, 0, 0, 1}; });
    ParamValues p = with(defaultValues(m), {{"encoding", 0}, {"offset_frames", 8}, {"detail", 2}, {"streak_length", 32}, {"light_response", 0}, {"coverage", 1}});
    RenderResult half = renderAt(m, c, 10, p, 0.5);
    CHECK(half.out.view.load(24, 4).r > 0.02f);
    CHECK(half.out.view.load(31, 4).r < 1e-6f);
    RenderResult full = renderAt(m, c, 10, p, 1.0);
    auto extent = [](const RenderResult& rr) { int e = 0; for (int x = 0; x < 48; ++x) if (rr.out.view.load(x, 4).r > 1e-3f) e = x; return e; };
    CHECK(extent(full) >= extent(half) + 10);
}

TP_TEST(time_shift_zero_coverage_is_identity_for_sdr_content) {
    const ModuleDef& m = find("time_shift");
    Clip c = makeClip(32, 24, 30, [](int x, int y, int f) { return RGBA{0.1f + 0.02f * ((x + f) % 20), 0.3f + 0.01f * y, 0.2f, 1}; });
    for (int mode = 0; mode < 3; ++mode)
        for (int comb = 0; comb < 2; ++comb) {
            ParamValues p = with(defaultValues(m), {{"mode", mode}, {"combine", comb}, {"encoding", 0}, {"coverage", 0}});
            CHECK(maxDiff(renderAt(m, c, 20, p).out.view, c.frames[20].view) < 1e-6);
            CHECK(maxDiff(renderAt(m, c, 20, with(p, {{"coverage", 1}})).out.view, c.frames[20].view) > 0.01);
        }
}

TP_TEST(time_shift_shadow_trails_drag_dark_subject_and_never_brighten) {
    const ModuleDef& m = find("time_shift");
    Clip c = makeClip(64, 8, 30, [](int x, int, int f) { return (x >= f && x < f + 6) ? RGBA{0.05f, 0.05f, 0.05f, 1} : RGBA{0.85f, 0.85f, 0.85f, 1}; });
    ParamValues p = with(defaultValues(m), {{"combine", 2}, {"encoding", 0}, {"offset_frames", 8}, {"streak_length", 0}});
    RenderResult r = renderAt(m, c, 20, p);
    CHECK(r.out.view.load(16, 4).r < 0.8f);
    CHECK(r.out.view.load(16, 4).r > r.out.view.load(23, 4).r - 1.0f);
    for (int x = 0; x < 64; ++x) CHECK(r.out.view.load(x, 4).r <= c.frames[20].view.load(x, 4).r + 1e-6f);
    CHECK_NEAR(r.out.view.load(40, 4).r, 0.85, 1e-6);
}
