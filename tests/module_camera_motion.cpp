#include "harness.h"

using namespace tpt;

TP_TEST(camera_motion_before_start_is_identity) {
    const ModuleDef& m = find("camera_motion");
    Clip c = patternClip(64, 48, 40);
    ParamValues p = with(defaultValues(m), {{"mode", 0}, {"encoding", 0}, {"start_frame", 20}, {"breathing", 0}});
    for (double t : {0.0, 10.0, 19.0}) {
        RenderResult r = renderAt(m, c, t, p);
        CHECK(maxDiff(r.out.view, c.frames[static_cast<size_t>(t)].view) < 1e-5);
    }
    RenderResult moving = renderAt(m, c, 26, p);
    CHECK(maxDiff(moving.out.view, c.frames[26].view) > 0.05);
}

TP_TEST(camera_motion_shutter_zero_matches_manual_affine) {
    const ModuleDef& m = find("camera_motion");
    Clip c = patternClip(64, 48, 40);
    ParamValues p = with(defaultValues(m), {{"mode", 0}, {"encoding", 0}, {"shutter", 0}, {"easing", 0}, {"rebound", 0},
                                            {"rotation", 0}, {"zoom", 2}, {"duration", 10}, {"start_frame", 0}});
    RenderResult r = renderAt(m, c, 5, p);
    double s = std::exp(std::log(2.0) * 0.5);
    Affine inv = Affine::about(32, 24, Affine::scale(s)).inverse();
    for (auto xy : {std::pair<int, int>{10, 7}, {40, 30}, {5, 44}}) {
        RGBA e = sampleAffine(c.frames[5].view, inv, xy.first, xy.second, SampleEdge::Clamp);
        RGBA g = r.out.view.load(xy.first, xy.second);
        for (int ch = 0; ch < 4; ++ch) CHECK_NEAR(g[ch], e[ch], 1e-5);
    }
    RenderResult done = renderAt(m, c, 30, p);
    RGBA e = sampleAffine(c.frames[30].view, Affine::about(32, 24, Affine::scale(2.0)).inverse(), 12, 9, SampleEdge::Clamp);
    CHECK_NEAR(done.out.view.load(12, 9).g, e.g, 1e-5);
}

TP_TEST(camera_motion_handheld_zero_amplitude_is_identity) {
    const ModuleDef& m = find("camera_motion");
    Clip c = patternClip(64, 48, 40);
    ParamValues p = with(defaultValues(m), {{"mode", 3}, {"encoding", 0}, {"amplitude", 0}, {"rotation", 0}, {"breathing", 0}});
    RenderResult r = renderAt(m, c, 17, p);
    CHECK(maxDiff(r.out.view, c.frames[17].view) < 1e-5);
    ParamValues shake = with(p, {{"amplitude", 6}});
    RenderResult s = renderAt(m, c, 17, shake);
    CHECK(maxDiff(s.out.view, c.frames[17].view) > 0.01);
}

TP_TEST(camera_motion_deterministic_and_seeded) {
    const ModuleDef& m = find("camera_motion");
    Clip c = patternClip(64, 48, 40);
    ParamValues p = with(defaultValues(m), {{"mode", 3}});
    RenderResult a = renderAt(m, c, 13, p);
    renderAt(m, c, 2, p);
    RenderResult b = renderAt(m, c, 13, p);
    CHECK(maxDiff(a.out.view, b.out.view) == 0.0);
    RenderResult other = renderAt(m, c, 13, with(p, {{"seed", 7}}));
    CHECK(maxDiff(a.out.view, other.out.view) > 1e-3);
}

TP_TEST(camera_motion_whip_ends_on_offset_framing) {
    const ModuleDef& m = find("camera_motion");
    Clip c = patternClip(64, 48, 40);
    ParamValues p = with(defaultValues(m), {{"mode", 2}, {"encoding", 0}, {"distance", 10}, {"angle", 0}, {"rotation", 0},
                                            {"rebound", 0}, {"duration", 8}, {"outside", 2}});
    RenderResult r = renderAt(m, c, 30, p);
    for (int y = 0; y < 48; ++y) {
        CHECK_NEAR(r.out.view.load(3, y).a, 0.0, 1e-6);
        RGBA e = c.frames[30].view.load(20, y);
        CHECK_NEAR(r.out.view.load(30, y).r, e.r, 1e-5);
    }
}

TP_TEST(camera_motion_blur_grows_with_shutter) {
    const ModuleDef& m = find("camera_motion");
    Clip c = patternClip(64, 48, 40);
    ParamValues base = with(defaultValues(m), {{"mode", 2}, {"encoding", 0}, {"distance", 40}, {"duration", 8}, {"start_frame", 10}, {"rotation", 0}});
    RenderResult sharp = renderAt(m, c, 14, with(base, {{"shutter", 0}}));
    RenderResult mid = renderAt(m, c, 14, with(base, {{"shutter", 180}}));
    RenderResult wide = renderAt(m, c, 14, with(base, {{"shutter", 540}}));
    double d1 = maxDiff(mid.out.view, sharp.out.view), d2 = maxDiff(wide.out.view, sharp.out.view);
    CHECK(d1 > 1e-3);
    CHECK(d2 > d1);
}

TP_TEST(camera_motion_dolly_zoom_keeps_center) {
    const ModuleDef& m = find("camera_motion");
    Clip c = patternClip(64, 48, 40);
    ParamValues p = with(defaultValues(m), {{"mode", 4}, {"encoding", 0}, {"shutter", 0}, {"rotation", 0}, {"rebound", 0}, {"zoom", 2}});
    RenderResult r = renderAt(m, c, 30, p);
    RGBA ctr = r.out.view.load(32, 24), src = c.frames[30].view.load(32, 24);
    CHECK_NEAR(ctr.r, src.r, 1e-5);
    CHECK(maxDiff(r.out.view, c.frames[30].view) > 0.02);
}

TP_TEST(camera_motion_punch_in_is_a_hard_cut_and_reveal_ends_on_source) {
    const ModuleDef& m = find("camera_motion");
    Clip c = makeClip(64, 48, 40, [](int x, int y, int) { return RGBA{x / 64.f, y / 48.f, 0.3f, 1}; });
    ParamValues punch = with(defaultValues(m), {{"mode", 7}, {"start_frame", 10}, {"shutter", 0}, {"rebound", 0}, {"rotation", 0}, {"encoding", 0}});
    CHECK(maxDiff(renderAt(m, c, 9, punch).out.view, c.frames[9].view) < 1e-6);
    RenderResult after = renderAt(m, c, 10, punch);
    CHECK(maxDiff(after.out.view, c.frames[10].view) > 0.05);
    CHECK(maxDiff(after.out.view, renderAt(m, c, 11, punch).out.view) < 1e-6);
    ParamValues reveal = with(punch, {{"mode", 9}, {"duration", 12}});
    CHECK(maxDiff(renderAt(m, c, 30, reveal).out.view, c.frames[30].view) < 1e-5);
    CHECK(maxDiff(renderAt(m, c, 10, reveal).out.view, c.frames[10].view) > 0.05);
}

TP_TEST(camera_motion_snap_and_back_returns_and_pulse_repeats) {
    const ModuleDef& m = find("camera_motion");
    Clip c = makeClip(64, 48, 60, [](int x, int y, int) { return RGBA{x / 64.f, y / 48.f, 0.3f, 1}; });
    ParamValues snap = with(defaultValues(m), {{"mode", 6}, {"start_frame", 0}, {"duration", 20}, {"shutter", 0}, {"rotation", 0}, {"encoding", 0}});
    CHECK(maxDiff(renderAt(m, c, 10, snap).out.view, c.frames[10].view) > 0.05);
    CHECK(maxDiff(renderAt(m, c, 20, snap).out.view, c.frames[20].view) < 1e-5);
    ParamValues pulse = with(snap, {{"mode", 8}, {"duration", 10}});
    CHECK(maxDiff(renderAt(m, c, 12, pulse).out.view, renderAt(m, c, 22, pulse).out.view) < 1e-6);
}
