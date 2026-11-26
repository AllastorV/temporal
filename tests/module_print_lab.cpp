#include "harness.h"

using namespace tpt;

static ParamValues neutralCamera(const ModuleDef& m, int look) {
    ParamValues p = with(defaultValues(m), {{"look", look}, {"encoding", 1}, {"zoom", 1}, {"angle", 0}, {"roll", 0}, {"depth_of_field", 0},
                                            {"vignette", 0}, {"sheen", 0}, {"paper", 0}, {"registration", 0}});
    p.v["position"] = {0.5, 0.5, 0};
    return p;
}

TP_TEST(print_lab_white_input_prints_as_paper_through_neutral_camera) {
    const ModuleDef& m = find("print_lab");
    Clip c = constantClip(RGBA{1, 1, 1, 1}, 96, 64, 3);
    const float paper[3] = {0.89f, 0.85f, 0.78f};
    const float tint[3] = {1.f, 0.985f, 0.95f};
    RenderResult r = renderAt(m, c, 1, neutralCamera(m, 1));
    RGBA o = r.out.view.load(48, 32);
    for (int ch = 0; ch < 3; ++ch) {
        double expect = encodeValue(decodeValue(paper[ch], Encoding::Gamma24) * tint[ch], Encoding::Gamma24);
        CHECK_NEAR(o[ch], expect, 2e-3);
    }
}

TP_TEST(print_lab_black_input_is_mostly_ink_for_every_look) {
    const ModuleDef& m = find("print_lab");
    Clip c = constantClip(RGBA{0, 0, 0, 1}, 96, 64, 3);
    Clip w = constantClip(RGBA{1, 1, 1, 1}, 96, 64, 3);
    for (int look = 0; look < 10; ++look) {
        ParamValues p = neutralCamera(m, look);
        double dark = luma(mean(renderAt(m, c, 1, p).out.view));
        double light = luma(mean(renderAt(m, w, 1, p).out.view));
        CHECK(dark < 0.6 * light);
    }
}

TP_TEST(print_lab_mean_is_monotonic_in_gray_level) {
    const ModuleDef& m = find("print_lab");
    for (int look : {0, 1, 2, 6, 8}) {
        double prev = -1;
        for (float g : {0.05f, 0.3f, 0.6f, 0.95f}) {
            Clip c = constantClip(RGBA{g, g, g, 1}, 128, 96, 3);
            double v = luma(mean(renderAt(m, c, 1, neutralCamera(m, look)).out.view));
            CHECK(v > prev);
            prev = v;
        }
    }
}

TP_TEST(print_lab_finite_for_all_looks_and_camera_moves) {
    const ModuleDef& m = find("print_lab");
    Clip c = patternClip(64, 48, 6);
    for (int look = 0; look < 10; ++look)
        for (int move = 0; move < 5; ++move) {
            ParamValues p = with(defaultValues(m), {{"look", look}, {"camera_move", move}});
            RenderResult r = renderAt(m, c, 4, p);
            for (int y = 0; y < 48; ++y)
                for (int x = 0; x < 64; ++x)
                    for (int ch = 0; ch < 4; ++ch) CHECK(std::isfinite(r.out.view.load(x, y)[ch]));
        }
}

TP_TEST(print_lab_locked_off_is_time_invariant_and_push_in_moves) {
    const ModuleDef& m = find("print_lab");
    Clip c = makeClip(96, 64, 60, [](int x, int y, int) { return RGBA{x / 96.f, y / 64.f, 0.5f, 1}; });
    ParamValues animated = defaultValues(m);
    CHECK(maxDiff(renderAt(m,c,5,animated).out.view,renderAt(m,c,50,animated).out.view) > 1e-5);
    ParamValues p = with(animated, {{"texture_motion", 0}});
    CHECK(maxDiff(renderAt(m, c, 5, p).out.view, renderAt(m, c, 50, p).out.view) == 0.0);
    ParamValues push = with(p, {{"camera_move", 3}});
    CHECK(maxDiff(renderAt(m, c, 5, push).out.view, renderAt(m, c, 50, push).out.view) > 0.01);
}

TP_TEST(print_lab_depth_of_field_blurs_away_from_focus) {
    const ModuleDef& m = find("print_lab");
    Clip c = constantClip(RGBA{0.5f, 0.5f, 0.5f, 1}, 256, 192, 3);
    ParamValues p = with(neutralCamera(m, 0), {{"angle", 45}, {"depth_of_field", 1}, {"focus", 0.5}, {"zoom", 1.5}});
    RenderResult r = renderAt(m, c, 1, p);
    auto localVar = [&](int y0) {
        double s = 0, s2 = 0; int n = 0;
        for (int y = y0; y < y0 + 16; ++y)
            for (int x = 96; x < 160; ++x) { double v = r.out.view.load(x, y).g; s += v; s2 += v * v; ++n; }
        double mu = s / n;
        return s2 / n - mu * mu;
    };
    double focusVar = localVar(88);
    double farVar = localVar(170);
    CHECK(focusVar > 4.0 * farVar);
}

TP_TEST(print_lab_show_whole_picture_reveals_table_at_corners) {
    const ModuleDef& m = find("print_lab");
    Clip c = constantClip(RGBA{1, 1, 1, 1}, 128, 96, 3);
    ParamValues p = with(neutralCamera(m, 1), {{"show_whole_picture", 1}});
    RenderResult r = renderAt(m, c, 1, p);
    CHECK(luma(r.out.view.load(1, 1)) < 0.15f);
    CHECK(luma(r.out.view.load(64, 48)) > 0.7f);
}

TP_TEST(print_lab_fold_relief_changes_shading_only_when_paper_on) {
    const ModuleDef& m = find("print_lab");
    Clip c = constantClip(RGBA{1, 1, 1, 1}, 160, 120, 3);
    ParamValues flat = neutralCamera(m, 1);
    ParamValues folded = with(flat, {{"paper", 1}});
    double dFlat = 0, dFold = 0;
    RenderResult a = renderAt(m, c, 1, flat), b = renderAt(m, c, 1, folded);
    for (int y = 0; y < 120; ++y)
        for (int x = 1; x < 160; ++x) {
            dFlat = std::max(dFlat, static_cast<double>(std::fabs(a.out.view.load(x, y).g - a.out.view.load(x - 1, y).g)));
            dFold = std::max(dFold, static_cast<double>(std::fabs(b.out.view.load(x, y).g - b.out.view.load(x - 1, y).g)));
        }
    CHECK(dFlat < 1e-4);
    CHECK(dFold > 5e-3);
}
