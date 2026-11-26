#include "harness.h"

using namespace tpt;

static Clip doubleExposureLevelClip() {
    return makeClip(16, 8, 40, [](int, int, int f) { float v = f / 50.f; return RGBA{v, v * 0.5f, 0.1f, 1.f}; });
}

TP_TEST(double_exposure_neutral_average_is_identity) {
    const ModuleDef& m = find("double_exposure");
    Clip c = patternClip();
    ParamValues p = with(defaultValues(m), {{"frame_offset", 0}, {"key_source", 0}, {"blend", 4}, {"balance", 0.5}, {"exposure", 0}});
    for (int enc = 0; enc < 4; ++enc) {
        RenderResult r = renderAt(m, c, 17, with(p, {{"encoding", static_cast<double>(enc)}}));
        CHECK(maxDiff(r.out.view, c.frames[17].view) < 1e-4);
    }
}

TP_TEST(double_exposure_add_sums_linear_light_with_gain) {
    const ModuleDef& m = find("double_exposure");
    Clip c = doubleExposureLevelClip();
    ParamValues p = with(defaultValues(m), {{"frame_offset", -5}, {"blend", 0}, {"balance", 0.5}, {"exposure", 1}, {"encoding", 0}});
    RenderResult r = renderAt(m, c, 20, p);
    RGBA o = r.out.view.load(4, 4);
    CHECK_NEAR(o.r, 20 / 50.0 + 2 * 15 / 50.0, 1e-5);
    CHECK_NEAR(o.g, 0.5 * (20 / 50.0 + 2 * 15 / 50.0), 1e-5);
    CHECK_NEAR(o.a, 1.0, 1e-6);
    CHECK(r.window.before == 5 && r.window.after == 0);
    ParamValues half = with(p, {{"balance", 0.25}, {"exposure", 0}});
    CHECK_NEAR(renderAt(m, c, 20, half).out.view.load(4, 4).r, 20 / 50.0 + 0.5 * 15 / 50.0, 1e-5);
}

TP_TEST(double_exposure_screen_matches_formula_and_stays_sane_in_hdr) {
    const ModuleDef& m = find("double_exposure");
    Clip c = makeClip(8, 8, 30, [](int, int, int f) { float v = f == 10 ? 0.6f : (f == 14 ? 0.3f : (f == 20 ? 3.f : 2.f)); return RGBA{v, v, v, 1.f}; });
    ParamValues p = with(defaultValues(m), {{"frame_offset", 4}, {"blend", 1}, {"balance", 0.5}, {"encoding", 0}});
    CHECK_NEAR(renderAt(m, c, 10, p).out.view.load(2, 2).r, 0.6 + 0.3 - 0.18, 1e-5);
    float hdr = renderAt(m, c, 20, p).out.view.load(2, 2).r;
    CHECK(std::isfinite(hdr) && hdr >= 3.f - 1e-5f && hdr <= 5.f);
    ParamValues lighten = with(p, {{"blend", 2}});
    CHECK_NEAR(renderAt(m, c, 10, lighten).out.view.load(2, 2).r, 0.6, 1e-5);
    ParamValues mult = with(p, {{"blend", 3}});
    CHECK_NEAR(renderAt(m, c, 10, mult).out.view.load(2, 2).r, 0.18, 1e-5);
}

TP_TEST(double_exposure_current_luma_key_masks_second) {
    const ModuleDef& m = find("double_exposure");
    Clip c = makeClip(16, 8, 30, [](int x, int, int f) {
        if (f == 20) { float v = x < 8 ? 0.05f : 0.8f; return RGBA{v, v, v, 1.f}; }
        return RGBA{0.5f, 0.5f, 0.5f, 1.f};
    });
    ParamValues p = with(defaultValues(m), {{"frame_offset", -3}, {"blend", 0}, {"key_source", 1}, {"key_threshold", 0.25}, {"feather", 0}, {"encoding", 0}});
    RenderResult r = renderAt(m, c, 20, p);
    CHECK_NEAR(r.out.view.load(2, 3).r, 0.55, 1e-5);
    CHECK_NEAR(r.out.view.load(12, 3).r, 0.8, 1e-5);
    RenderResult inv = renderAt(m, c, 20, with(p, {{"invert_key", 1}}));
    CHECK_NEAR(inv.out.view.load(2, 3).r, 0.05, 1e-5);
    CHECK_NEAR(inv.out.view.load(12, 3).r, 1.3, 1e-5);
    RenderResult soft = renderAt(m, c, 20, with(p, {{"key_blur", 6}}));
    float edge = soft.out.view.load(8, 3).r - 0.8f;
    CHECK(edge > 0.02f && edge < 0.48f);
}

TP_TEST(double_exposure_missing_second_keeps_first_and_flip_mirrors) {
    const ModuleDef& m = find("double_exposure");
    Clip c = patternClip();
    for (int blend = 0; blend < 5; ++blend) {
        ParamValues p = with(defaultValues(m), {{"frame_offset", -12}, {"blend", static_cast<double>(blend)}, {"edge", 2}, {"encoding", 0}});
        RenderResult r = renderAt(m, c, 3, p);
        CHECK(maxDiff(r.out.view, c.frames[3].view) < 1e-6);
    }
    Clip flat = constantClip(RGBA{0.6f, 0.4f, 0.2f, 1.f}, 32, 24, 30);
    ParamValues tr = with(defaultValues(m), {{"frame_offset", -3}, {"blend", 4}, {"balance", 0.5}, {"second_scale", 0.43}, {"second_rotation", 33}, {"encoding", 0}});
    CHECK(maxDiff(renderAt(m, flat, 10, tr).out.view, flat.frames[10].view) < 1e-5);
    ParamValues f = with(defaultValues(m), {{"frame_offset", 0}, {"blend", 4}, {"balance", 1}, {"flip_horizontal", 1}, {"encoding", 0}});
    RenderResult r = renderAt(m, c, 9, f);
    const ImageView& s = c.frames[9].view;
    for (int y = 0; y < 48; y += 7)
        for (int x = 0; x < 64; ++x) CHECK_NEAR(r.out.view.load(x, y).g, s.load(63 - x, y).g, 1e-5);
}
