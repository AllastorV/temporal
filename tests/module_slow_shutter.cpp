#include "harness.h"

using namespace tpt;

static Clip slowShutterRamp() {
    return makeClip(8, 8, 60, [](int, int, int f) { return RGBA{f / 100.f, f / 200.f, 0.5f, 1}; });
}

TP_TEST(slow_shutter_static_clip_is_identity_for_all_syncs_and_modes) {
    const ModuleDef& m = find("slow_shutter");
    Clip c = constantClip(RGBA{0.3f, 0.5f, 0.7f, 1.f});
    for (int mode = 0; mode < 2; ++mode)
        for (int sync = 0; sync < 3; ++sync)
            for (int stepped = 0; stepped < 2; ++stepped) {
                ParamValues p = with(defaultValues(m), {{"mode", mode}, {"sync", sync}, {"stepped", stepped}, {"encoding", 0}});
                RenderResult r = renderAt(m, c, 15, p);
                CHECK(maxDiff(r.out.view, c.frames[15].view) < 1e-6);
            }
}

TP_TEST(slow_shutter_ramp_mean_follows_sync) {
    const ModuleDef& m = find("slow_shutter");
    Clip c = slowShutterRamp();
    ParamValues p = with(defaultValues(m), {{"encoding", 0}, {"softness", 0}, {"virtual_fps", 6}, {"shutter_angle", 360}});
    const double expect[] = {0.18, 0.20, 0.22};
    for (int sync = 0; sync < 3; ++sync) {
        RenderResult r = renderAt(m, c, 20, with(p, {{"sync", sync}}));
        CHECK_NEAR(r.out.view.load(4, 4).r, expect[sync], 1e-5);
        CHECK_NEAR(r.out.view.load(4, 4).a, 1.0, 1e-6);
    }
    RenderResult half = renderAt(m, c, 20, with(p, {{"shutter_angle", 180}}));
    CHECK_NEAR(half.out.view.load(4, 4).r, 0.19, 1e-5);
}

TP_TEST(slow_shutter_exposure_is_a_gain_in_stops) {
    const ModuleDef& m = find("slow_shutter");
    Clip c = constantClip(RGBA{0.2f, 0.1f, 0.05f, 1.f});
    RenderResult r = renderAt(m, c, 10, with(defaultValues(m), {{"encoding", 0}, {"exposure", 1}}));
    CHECK_NEAR(r.out.view.load(2, 2).r, 0.4, 1e-6);
    CHECK_NEAR(r.out.view.load(2, 2).b, 0.1, 1e-6);
    CHECK_NEAR(r.out.view.load(2, 2).a, 1.0, 1e-6);
}

TP_TEST(slow_shutter_stepped_holds_virtual_frames) {
    const ModuleDef& m = find("slow_shutter");
    Clip c = slowShutterRamp();
    ParamValues p = with(defaultValues(m), {{"encoding", 0}, {"stepped", 1}, {"virtual_fps", 6}});
    RenderResult a = renderAt(m, c, 8, p);
    for (int t = 9; t <= 11; ++t) CHECK(maxDiff(renderAt(m, c, t, p).out.view, a.out.view) == 0.0);
    CHECK(maxDiff(renderAt(m, c, 12, p).out.view, a.out.view) > 0.03);
    RenderResult free = renderAt(m, c, 9, with(p, {{"stepped", 0}}));
    CHECK(maxDiff(free.out.view, a.out.view) > 0.005);
}

TP_TEST(slow_shutter_hard_trails_keep_ghosts_bright) {
    const ModuleDef& m = find("slow_shutter");
    Clip c = makeClip(8, 8, 40, [](int, int, int f) { return f == 18 ? RGBA{1, 1, 1, 1} : RGBA{0, 0, 0, 1}; });
    ParamValues p = with(defaultValues(m), {{"encoding", 0}, {"softness", 0}, {"decay", 0.85}});
    RenderResult smooth = renderAt(m, c, 20, p);
    CHECK_NEAR(smooth.out.view.load(4, 4).r, 0.25, 1e-5);
    RenderResult hard = renderAt(m, c, 20, with(p, {{"mode", 1}}));
    CHECK_NEAR(hard.out.view.load(4, 4).r, 0.85 * 0.85, 1e-5);
    RenderResult none = renderAt(m, c, 20, with(p, {{"mode", 1}, {"decay", 0}}));
    CHECK_NEAR(none.out.view.load(4, 4).r, 0.25, 1e-5);
}

TP_TEST(slow_shutter_window_matches_exposure_and_is_capped) {
    const ModuleDef& m = find("slow_shutter");
    ParamValues p = defaultValues(m);
    FrameWindow w = effectiveWindow(m, with(p, {{"virtual_fps", 1}}), 24);
    CHECK(w.before == 24 && w.after == 0);
    w = effectiveWindow(m, with(p, {{"virtual_fps", 1}, {"sync", 1}}), 24);
    CHECK(w.before == 12 && w.after == 12);
    w = effectiveWindow(m, with(p, {{"virtual_fps", 1}, {"stepped", 1}}), 24);
    CHECK(w.before == 48);
    w = effectiveWindow(m, with(p, {{"virtual_fps", 1}, {"stepped", 1}}), 120);
    CHECK(w.before == 64);
}
