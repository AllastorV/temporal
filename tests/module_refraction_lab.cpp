#include "harness.h"

using namespace tpt;

static Clip rlGrayRamp() {
    return makeClip(96, 64, 1, [](int x, int y, int) {
        float v = 0.05f + 0.009f * x + 0.002f * y;
        return RGBA{v, v, v, 1.f};
    });
}

static Clip rlColorRamp() {
    return makeClip(96, 64, 1, [](int x, int y, int) {
        return RGBA{0.1f + 0.008f * x, 0.2f + 0.006f * y, 0.5f - 0.003f * x, 1.f};
    });
}

TP_TEST(refraction_lab_ior_one_returns_source) {
    const ModuleDef& m = find("refraction_lab");
    Clip c = rlColorRamp();
    for (int mode = 0; mode < 4; ++mode) {
        ParamValues p = with(defaultValues(m), {{"mode", static_cast<double>(mode)}, {"ior", 1.0}, {"dispersion", 0.0}, {"curvature", 3.0}, {"encoding", 0}});
        RenderResult r = renderAt(m, c, 0, p);
        CHECK(maxDiff(r.out.view, c.frames[0].view) < 1e-4);
    }
}

TP_TEST(refraction_lab_flat_prism_has_no_deviation) {
    const ModuleDef& m = find("refraction_lab");
    Clip c = rlColorRamp();
    ParamValues p = with(defaultValues(m), {{"mode", 1}, {"curvature", 0.0}, {"ior", 1.8}, {"dispersion", 0.05}, {"encoding", 0}});
    RenderResult r = renderAt(m, c, 0, p);
    CHECK(maxDiff(r.out.view, c.frames[0].view) < 1e-5);
}

TP_TEST(refraction_lab_prism_shift_is_constant) {
    const ModuleDef& m = find("refraction_lab");
    Clip c = rlGrayRamp();
    ParamValues p = with(defaultValues(m), {{"mode", 1}, {"curvature", 0.3}, {"thickness", 20}, {"dispersion", 0.0}, {"size", 0.8}, {"encoding", 0}});
    RenderResult r = renderAt(m, c, 0, p);
    double d0 = r.out.view.load(40, 30).r - c.frames[0].view.load(40, 30).r;
    double d1 = r.out.view.load(56, 34).r - c.frames[0].view.load(56, 34).r;
    CHECK(std::fabs(d0) > 1e-3);
    CHECK_NEAR(d0, d1, 1e-5);
    CHECK(maxDiff(r.out.view, c.frames[0].view) > 1e-3);
    CHECK(r.out.view.load(2, 2).r == c.frames[0].view.load(2, 2).r);
}

TP_TEST(refraction_lab_no_dispersion_no_fringes) {
    const ModuleDef& m = find("refraction_lab");
    Clip c = rlGrayRamp();
    for (int mode = 0; mode < 4; ++mode) {
        ParamValues p = with(defaultValues(m), {{"mode", static_cast<double>(mode)}, {"dispersion", 0.0}, {"encoding", 0}});
        RenderResult r = renderAt(m, c, 0, p);
        double fr = 0;
        for (int y = 0; y < 64; ++y)
            for (int x = 0; x < 96; ++x) {
                RGBA o = r.out.view.load(x, y);
                fr = std::max(fr, static_cast<double>(std::max(std::fabs(o.r - o.g), std::fabs(o.b - o.g))));
            }
        CHECK(fr == 0.0);
        ParamValues q = with(p, {{"dispersion", 0.05}});
        RenderResult d = renderAt(m, c, 0, q);
        double fd = 0;
        for (int y = 0; y < 64; ++y)
            for (int x = 0; x < 96; ++x) {
                RGBA o = d.out.view.load(x, y);
                fd = std::max(fd, static_cast<double>(std::fabs(o.r - o.b)));
            }
        CHECK(fd > 1e-3);
    }
}

TP_TEST(refraction_lab_total_internal_reflection_is_finite_and_darkened) {
    const ModuleDef& m = find("refraction_lab");
    Clip c = constantClip(RGBA{0.5f, 0.5f, 0.5f, 1.f}, 64, 48, 1);
    for (int mode = 0; mode < 4; ++mode) {
        ParamValues p = with(defaultValues(m), {{"mode", static_cast<double>(mode)}, {"ior", 2.5}, {"thickness", 2000}, {"curvature", 4}, {"size", 0.6}, {"encoding", 0}});
        RenderResult r = renderAt(m, c, 0, p);
        double lo = 1;
        for (int y = 0; y < 48; ++y)
            for (int x = 0; x < 64; ++x) {
                RGBA o = r.out.view.load(x, y);
                for (int ch = 0; ch < 4; ++ch) CHECK(std::isfinite(o[ch]));
                lo = std::min(lo, static_cast<double>(o.g));
            }
        CHECK(lo < 0.3);
    }
}

TP_TEST(refraction_lab_focus_shift_blurs_only_inside) {
    const ModuleDef& m = find("refraction_lab");
    Clip c = makeClip(96, 64, 1, [](int x, int y, int) {
        float v = ((x / 4 + y / 4) % 2) ? 0.9f : 0.1f;
        return RGBA{v, v, v, 1.f};
    });
    ParamValues p = with(defaultValues(m), {{"mode", 1}, {"curvature", 0.0}, {"focus_shift", 1.0}, {"f_stop", 1.0}, {"thickness", 40}, {"encoding", 0}});
    RenderResult r = renderAt(m, c, 0, p);
    CHECK(r.out.view.load(2, 2).r == c.frames[0].view.load(2, 2).r);
    CHECK(std::fabs(r.out.view.load(48, 32).r - 0.5) < 0.2);
}

TP_TEST(refraction_lab_diopter_magnifies_one_half_only) {
    const ModuleDef& m = find("refraction_lab");
    Clip c = rlGrayRamp();
    ParamValues p = with(defaultValues(m), {{"mode", 0}, {"dispersion", 0.0}, {"curvature", 1.0}, {"thickness", 20}, {"encoding", 0}});
    RenderResult r = renderAt(m, c, 0, p);
    const ImageView& src = c.frames[0].view;
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 44; ++x) CHECK(r.out.view.load(x, y).r == src.load(x, y).r);
    CHECK(r.out.view.load(56, 32).r < src.load(56, 32).r - 1e-3);
    ParamValues neg = with(p, {{"curvature", -1.0}});
    CHECK(renderAt(m, c, 0, neg).out.view.load(56, 32).r > src.load(56, 32).r + 1e-3);
}
