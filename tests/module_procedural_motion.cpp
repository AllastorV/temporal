#include "harness.h"

using namespace tpt;

static Clip procedural_motion_static_clip(int n = 80) {
    return makeClip(64, 48, n, [](int x, int y, int) {
        return RGBA{0.1f + 0.01f * x, 0.2f + 0.015f * y, ((x / 4 + y / 4) % 2) ? 0.9f : 0.1f, 1.f};
    });
}

TP_TEST(procedural_motion_zero_amounts_is_identity) {
    const ModuleDef& m = find("procedural_motion");
    Clip c = patternClip(64, 48, 40);
    for (int mode = 0; mode < 7; ++mode) {
        ParamValues p = with(defaultValues(m), {{"mode", mode}, {"encoding", 0}, {"amplitude", 0}, {"rotation_amount", 0}, {"zoom_amount", 0},
                                                {"axis_rx", 1}, {"axis_ry", 1}});
        for (double t : {0.0, 9.0, 23.0}) {
            RenderResult r = renderAt(m, c, t, p);
            CHECK(maxDiff(r.out.view, c.frames[static_cast<size_t>(t)].view) < 1e-5);
        }
    }
}

TP_TEST(procedural_motion_circle_matches_analytic_position) {
    const ModuleDef& m = find("procedural_motion");
    Clip c = procedural_motion_static_clip(40);
    ParamValues p = with(defaultValues(m), {{"mode", 0}, {"encoding", 0}, {"shutter", 0}, {"amplitude", 8}, {"frequency", 1},
                                            {"axis_z", 0}, {"axis_rz", 0}, {"axis_rx", 0}, {"axis_ry", 0}, {"phase", 0}});
    struct Case { double t; int dx, dy; };
    for (Case k : {Case{0, 8, 0}, Case{6, 0, 8}, Case{12, -8, 0}, Case{18, 0, -8}}) {
        RenderResult r = renderAt(m, c, k.t, p);
        const ImageView& s = c.frames[static_cast<size_t>(k.t)].view;
        for (int y = 10; y < 38; ++y)
            for (int x = 10; x < 54; ++x) {
                RGBA e = s.load(x - k.dx, y - k.dy), g = r.out.view.load(x, y);
                for (int ch = 0; ch < 4; ++ch) CHECK_NEAR(g[ch], e[ch], 1e-4);
            }
    }
}

TP_TEST(procedural_motion_loop_is_periodic) {
    const ModuleDef& m = find("procedural_motion");
    Clip c = procedural_motion_static_clip(80);
    for (int mode : {0, 1, 2, 3, 4, 5, 6}) {
        ParamValues p = with(defaultValues(m), {{"mode", mode}, {"loop", 2}, {"frequency", 0.7}, {"rotation_amount", 20}, {"axis_rx", 0.5}, {"axis_ry", 0.5}});
        for (double t : {3.0, 17.0}) {
            RenderResult a = renderAt(m, c, t, p);
            RenderResult b = renderAt(m, c, t + 48, p);
            CHECK(maxDiff(a.out.view, b.out.view) < 1e-5);
        }
        RenderResult h = renderAt(m, c, 3, p), q = renderAt(m, c, 27, p);
        CHECK(maxDiff(h.out.view, q.out.view) > 1e-3);
    }
}

TP_TEST(procedural_motion_loop_is_seamless) {
    const ModuleDef& m = find("procedural_motion");
    Clip c = procedural_motion_static_clip(80);
    for (int mode : {0, 1, 2, 3, 4, 5}) {
        ParamValues p = with(defaultValues(m), {{"mode", mode}, {"encoding", 0}, {"shutter", 0}, {"loop", 2}, {"frequency", 0.7},
                                                {"rotation_amount", 20}, {"axis_z", 1}, {"axis_rz", 1}});
        RenderResult a = renderAt(m, c, 48.0 - 1e-3, p), b = renderAt(m, c, 48.0, p);
        CHECK(maxDiff(a.out.view, b.out.view) < 0.02);
        RenderResult x = renderAt(m, c, 24.0 - 1e-3, p), y = renderAt(m, c, 24.0, p);
        CHECK(maxDiff(x.out.view, y.out.view) < 0.02);
    }
}

TP_TEST(procedural_motion_deterministic_and_seeded) {
    const ModuleDef& m = find("procedural_motion");
    Clip c = patternClip(64, 48, 40);
    for (int mode : {3, 4, 5}) {
        ParamValues p = with(defaultValues(m), {{"mode", mode}, {"amplitude", 20}});
        RenderResult a = renderAt(m, c, 14, p);
        renderAt(m, c, 30, p);
        RenderResult b = renderAt(m, c, 14, p);
        CHECK(maxDiff(a.out.view, b.out.view) == 0.0);
        RenderResult o = renderAt(m, c, 14, with(p, {{"seed", 9}}));
        CHECK(maxDiff(a.out.view, o.out.view) > 1e-3);
    }
}

TP_TEST(procedural_motion_spring_is_zero_before_start_and_decays) {
    const ModuleDef& m = find("procedural_motion");
    Clip c = procedural_motion_static_clip(80);
    ParamValues p = with(defaultValues(m), {{"mode", 6}, {"encoding", 0}, {"start_frame", 20}, {"amplitude", 10}, {"damping", 3}, {"frequency", 1.3}});
    RenderResult before = renderAt(m, c, 19, p);
    CHECK(maxDiff(before.out.view, c.frames[19].view) < 1e-5);
    RenderResult early = renderAt(m, c, 21, with(p, {{"shutter", 0}}));
    RenderResult late = renderAt(m, c, 75, with(p, {{"shutter", 0}}));
    double de = maxDiff(early.out.view, c.frames[21].view), dl = maxDiff(late.out.view, c.frames[75].view);
    CHECK(de > 0.05);
    CHECK(dl < de * 0.5);
}

TP_TEST(procedural_motion_door_hinge_stays_put) {
    const ModuleDef& m = find("procedural_motion");
    Clip c = procedural_motion_static_clip(40);
    ParamValues p = with(defaultValues(m), {{"mode", 2}, {"encoding", 0}, {"shutter", 0}, {"rotation_amount", 60}, {"frequency", 1}, {"outside", 2}});
    RenderResult r = renderAt(m, c, 12, p);
    const ImageView& s = c.frames[12].view;
    for (int y = 0; y < 48; ++y) {
        CHECK_NEAR(r.out.view.load(0, y).r, s.load(0, y).r, 0.02);
        CHECK_NEAR(r.out.view.load(60, y).a, 0.0, 1e-6);
    }
    RenderResult rr = renderAt(m, c, 12, with(p, {{"hinge", 1}}));
    for (int y = 0; y < 48; ++y) {
        CHECK_NEAR(rr.out.view.load(63, y).r, s.load(63, y).r, 0.02);
        CHECK_NEAR(rr.out.view.load(3, y).a, 0.0, 1e-6);
    }
}
