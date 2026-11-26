#include "harness.h"

using namespace tpt;

TP_TEST(echo_static_clip_is_identity_in_ghosts_average) {
    const ModuleDef& m = find("echo");
    Clip c = constantClip(RGBA{0.3f, 0.5f, 0.7f, 1.f});
    ParamValues p = with(defaultValues(m), {{"mode", 0}, {"blend", 0}, {"encoding", 0}});
    RenderResult r = renderAt(m, c, 20, p);
    CHECK(maxDiff(r.out.view, c.frames[20].view) < 1e-6);
}

TP_TEST(echo_impulse_weights_are_normalized) {
    const ModuleDef& m = find("echo");
    Clip c = makeClip(8, 8, 20, [](int, int, int f) { return f == 5 ? RGBA{1, 1, 1, 1} : RGBA{0, 0, 0, 1}; });
    ParamValues p = with(defaultValues(m), {{"mode", 0}, {"blend", 0}, {"encoding", 0}, {"count", 1}, {"spacing", 2}, {"decay", 1}});
    RenderResult r = renderAt(m, c, 7, p);
    CHECK_NEAR(r.out.view.load(3, 3).r, 0.5, 1e-6);
    CHECK_NEAR(r.out.view.load(3, 3).a, 1.0, 1e-6);
    RenderResult far = renderAt(m, c, 12, p);
    CHECK_NEAR(far.out.view.load(3, 3).r, 0.0, 1e-6);
}

TP_TEST(echo_future_direction_reads_later_frames) {
    const ModuleDef& m = find("echo");
    Clip c = makeClip(8, 8, 20, [](int, int, int f) { return f == 9 ? RGBA{1, 1, 1, 1} : RGBA{0, 0, 0, 1}; });
    ParamValues p = with(defaultValues(m), {{"direction", 1}, {"blend", 0}, {"encoding", 0}, {"count", 1}, {"spacing", 2}, {"decay", 1}});
    RenderResult r = renderAt(m, c, 7, p);
    CHECK_NEAR(r.out.view.load(3, 3).r, 0.5, 1e-6);
    CHECK(r.window.after >= 2 && r.window.before == 0);
}

TP_TEST(echo_darken_stacks_distinct_dark_silhouettes_on_bright_background) {
    const ModuleDef& m = find("echo");
    Clip c = makeClip(64, 8, 30, [](int x, int, int f) { return (x >= f * 2 && x < f * 2 + 4) ? RGBA{0.05f, 0.05f, 0.05f, 1} : RGBA{0.9f, 0.9f, 0.9f, 1}; });
    ParamValues p = with(defaultValues(m), {{"blend", 3}, {"encoding", 0}, {"count", 3}, {"spacing", 2}, {"decay", 0.6}});
    RenderResult r = renderAt(m, c, 20, p);
    CHECK_NEAR(r.out.view.load(41, 4).r, 0.05, 1e-5);
    float e1 = r.out.view.load(37, 4).r, e2 = r.out.view.load(33, 4).r, e3 = r.out.view.load(29, 4).r;
    CHECK(e1 < e2 && e2 < e3 && e3 < 0.9f);
    CHECK_NEAR(e1, 0.9 + (0.05 - 0.9) * 0.6, 1e-4);
    CHECK_NEAR(r.out.view.load(10, 4).r, 0.9, 1e-6);
}
