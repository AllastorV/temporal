#include "harness.h"

using namespace tpt;

static RGBA dmTex(float x, float y) {
    float v = 0.5f + 0.25f * std::sin(x * 0.21f) * std::cos(y * 0.17f) + 0.2f * std::sin((x + y) * 0.073f);
    return RGBA{v, 1.f - v, 0.5f * v, 1.f};
}

TP_TEST(datamosh_i_frame_is_clean_and_static_clip_is_identity) {
    const ModuleDef& m = find("datamosh");
    Clip mov = makeClip(128, 96, 40, [](int x, int y, int f) { return dmTex(x - 3.f * f, static_cast<float>(y)); });
    ParamValues p = with(defaultValues(m), {{"corruption", 0}, {"reset_interval", 8}});
    CHECK(maxDiff(renderAt(m, mov, 16, p).out.view, mov.frames[16].view) == 0.0);
    Clip still = makeClip(128, 96, 40, [](int x, int y, int) { return dmTex(static_cast<float>(x), static_cast<float>(y)); });
    CHECK(maxDiff(renderAt(m, still, 20, p).out.view, still.frames[20].view) < 0.02);
}

TP_TEST(datamosh_smear_carries_old_pixels_with_new_motion) {
    const ModuleDef& m = find("datamosh");
    Clip c = makeClip(128, 96, 40, [](int x, int y, int f) {
        RGBA t = dmTex(x - 3.f * f, static_cast<float>(y));
        if (f >= 17) t = RGBA{t.g, t.r, t.b, 1.f};
        return t;
    });
    ParamValues p = with(defaultValues(m), {{"corruption", 0}, {"reset_interval", 8}, {"persistence", 1}});
    RenderResult r = renderAt(m, c, 20, p);
    double dCur = maxDiff(r.out.view, c.frames[20].view);
    CHECK(dCur > 0.2);
    RenderResult a = renderAt(m, c, 20, p);
    renderAt(m, c, 5, p);
    CHECK(maxDiff(a.out.view, renderAt(m, c, 20, p).out.view) == 0.0);
    CHECK(r.window.before == 8);
}

TP_TEST(datamosh_corruption_is_seeded_and_scene_transition_window) {
    const ModuleDef& m = find("datamosh");
    Clip c = makeClip(128, 96, 40, [](int x, int y, int f) { return dmTex(x - 2.f * f, static_cast<float>(y)); });
    ParamValues p = with(defaultValues(m), {{"mode", 1}, {"corruption", 0.5}});
    RenderResult a = renderAt(m, c, 10, p), b = renderAt(m, c, 10, with(p, {{"seed", 9}}));
    CHECK(maxDiff(a.out.view, c.frames[10].view) > 0.05);
    CHECK(maxDiff(a.out.view, b.out.view) > 0.05);
    ParamValues tr = with(defaultValues(m), {{"mode", 2}, {"reference_frame", 20}, {"reset_interval", 6}, {"corruption", 0}});
    CHECK(maxDiff(renderAt(m, c, 18, tr).out.view, c.frames[18].view) == 0.0);
    CHECK(maxDiff(renderAt(m, c, 30, tr).out.view, c.frames[30].view) == 0.0);
}
