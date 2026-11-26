#include "harness.h"
#include "core/flow.h"

using namespace tpt;

static RGBA texturedAt(float x, float y) {
    float v = 0.5f + 0.25f * std::sin(x * 0.21f) * std::cos(y * 0.17f) + 0.2f * std::sin((x + y) * 0.073f);
    return RGBA{v, v * 0.9f, v * 0.8f, 1.f};
}

TP_TEST(motion_tear_flow_measures_translation) {
    Clip c = makeClip(256, 160, 3, [](int x, int y, int f) { return texturedAt(x - 3.f * f, y + 1.5f * f); });
    FlowField fl = computeFlow(c.frames[0].view, c.frames[1].view, c.rect, 1.f, Encoding::Linear, 2);
    double su = 0, sv = 0;
    int n = 0;
    for (int y = 40; y < 120; y += 8)
        for (int x = 64; x < 192; x += 8) {
            float u, v;
            fl.sample(x + 0.5f, y + 0.5f, u, v);
            su += u; sv += v; ++n;
        }
    CHECK_NEAR(su / n, 3.0, 0.5);
    CHECK_NEAR(sv / n, -1.5, 0.5);
}

TP_TEST(motion_tear_static_clip_is_identity) {
    const ModuleDef& m = find("motion_tear");
    Clip c = makeClip(96, 64, 10, [](int x, int y, int) { return texturedAt(static_cast<float>(x), static_cast<float>(y)); });
    RenderResult r = renderAt(m, c, 5, with(defaultValues(m), {{"encoding", 0}}));
    CHECK(maxDiff(r.out.view, c.frames[5].view) < 1e-5);
}

TP_TEST(motion_tear_trails_behind_moving_subject_and_leads_when_sharp_at_start) {
    const ModuleDef& m = find("motion_tear");
    Clip c = makeClip(160, 64, 12, [](int x, int y, int f) {
        bool sq = x >= 40 + f * 4 && x < 60 + f * 4 && y >= 22 && y < 42;
        return sq ? RGBA{1, 1, 1, 1} : RGBA{0.05f, 0.05f, 0.05f, 1};
    });
    ParamValues p = with(defaultValues(m), {{"encoding", 0}, {"shutter", 4}, {"amount", 1}, {"highlights", 0}, {"softness", 0}});
    RenderResult r = renderAt(m, c, 6, p);
    const int left = 40 + 6 * 4;
    double behind = 0, ahead = 0;
    for (int y = 26; y < 38; ++y) { behind += r.out.view.load(left - 4, y).r; ahead += r.out.view.load(left + 24, y).r; }
    CHECK(behind / 12 > 0.15);
    CHECK(ahead / 12 < 0.1);
    RenderResult lead = renderAt(m, c, 6, with(p, {{"sharp_at", 2}}));
    double ahead2 = 0;
    for (int y = 26; y < 38; ++y) ahead2 += lead.out.view.load(left + 24, y).r;
    CHECK(ahead2 / 12 > 0.15);
}

TP_TEST(motion_tear_fixed_direction_follows_angle) {
    const ModuleDef& m = find("motion_tear");
    Clip c = makeClip(96, 64, 4, [](int x, int y, int) { return (x >= 40 && x < 48 && y >= 28 && y < 36) ? RGBA{1, 1, 1, 1} : RGBA{0, 0, 0, 1}; });
    ParamValues p = with(defaultValues(m), {{"mode", 1}, {"encoding", 0}, {"speed", 3}, {"shutter", 4}, {"amount", 1}, {"angle", 0}, {"threshold", 0}, {"highlights", 0}});
    RenderResult r = renderAt(m, c, 2, p);
    CHECK(r.out.view.load(36, 31).r > 0.05f);
    CHECK(r.out.view.load(52, 31).r < 1e-4f);
    CHECK(r.window.before == 0);
}
