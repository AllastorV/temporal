#include "harness.h"

using namespace tpt;

static Clip dlSpotClip() {
    return makeClip(64, 48, 2, [](int x, int y, int) {
        bool hot = x >= 28 && x < 36 && y >= 20 && y < 28;
        bool dark = x < 8;
        if (hot) return RGBA{4.f, 4.f, 4.f, 1.f};
        if (dark) return RGBA{0.f, 0.f, 0.f, 1.f};
        return RGBA{0.2f, 0.2f, 0.2f, 1.f};
    });
}

TP_TEST(diffusion_lab_strength_zero_is_source) {
    const ModuleDef& m = find("diffusion_lab");
    Clip c = patternClip();
    for (int mode = 0; mode < 5; ++mode) {
        ParamValues p = with(defaultValues(m), {{"mode", static_cast<double>(mode)}, {"strength", 0.0}});
        RenderResult r = renderAt(m, c, 10, p);
        CHECK(maxDiff(r.out.view, c.frames[10].view) == 0.0);
    }
}

TP_TEST(diffusion_lab_white_mist_preserves_uniform_image) {
    const ModuleDef& m = find("diffusion_lab");
    Clip c = constantClip(RGBA{0.45f, 0.3f, 0.6f, 1.f});
    for (double s : {0.3, 1.0}) {
        ParamValues p = with(defaultValues(m), {{"mode", 0}, {"strength", s}, {"scatter_radius", 40}, {"encoding", 0}});
        RenderResult r = renderAt(m, c, 5, p);
        CHECK(maxDiff(r.out.view, c.frames[5].view) < 1e-5);
    }
}

TP_TEST(diffusion_lab_halo_grows_with_strength) {
    const ModuleDef& m = find("diffusion_lab");
    Clip c = dlSpotClip();
    for (int mode : {0, 1, 2, 4}) {
        double prev = -1;
        for (double s : {0.1, 0.3, 0.6, 1.0}) {
            ParamValues p = with(defaultValues(m), {{"mode", static_cast<double>(mode)}, {"strength", s}, {"scatter_radius", 10}, {"streak_length", 20}, {"encoding", 0}});
            RenderResult r = renderAt(m, c, 0, p);
            double halo = r.out.view.load(40, 24).r - 0.2;
            CHECK(halo > prev);
            prev = halo;
        }
        CHECK(prev > 0.0);
    }
}

TP_TEST(diffusion_lab_black_mist_keeps_black_with_full_protection) {
    const ModuleDef& m = find("diffusion_lab");
    Clip c = dlSpotClip();
    for (int e = 0; e < 3; ++e) {
        ParamValues p = with(defaultValues(m), {{"mode", 1}, {"strength", 1.0}, {"scatter_radius", 200}, {"threshold", 0.1}, {"shadow_protection", 1.0}, {"encoding", static_cast<double>(e)}});
        RenderResult r = renderAt(m, c, 0, p);
        for (int y = 0; y < 48; ++y)
            for (int x = 0; x < 8; ++x) {
                RGBA o = r.out.view.load(x, y);
                CHECK(o.r == 0.f && o.g == 0.f && o.b == 0.f);
            }
        CHECK(r.out.view.load(40, 24).r > 0.2f);
    }
    ParamValues open = with(defaultValues(m), {{"mode", 1}, {"strength", 1.0}, {"scatter_radius", 200}, {"threshold", 0.1}, {"shadow_protection", 0.0}, {"encoding", 0}});
    CHECK(renderAt(m, c, 0, open).out.view.load(4, 24).r > 0.f);
}

TP_TEST(diffusion_lab_streak_follows_angle) {
    const ModuleDef& m = find("diffusion_lab");
    Clip c = makeClip(64, 64, 1, [](int x, int y, int) {
        bool hot = x >= 30 && x < 34 && y >= 30 && y < 34;
        return hot ? RGBA{8.f, 8.f, 8.f, 1.f} : RGBA{0.f, 0.f, 0.f, 1.f};
    });
    ParamValues h = with(defaultValues(m), {{"mode", 4}, {"strength", 1.0}, {"streak_length", 24}, {"streak_angle", 0}, {"encoding", 0}});
    RenderResult a = renderAt(m, c, 0, h);
    CHECK(a.out.view.load(44, 32).r > 0.f);
    CHECK_NEAR(a.out.view.load(32, 44).r, 0.0, 1e-6);
    RenderResult b = renderAt(m, c, 0, with(h, {{"streak_angle", 90}}));
    CHECK(b.out.view.load(32, 20).r > 0.f);
    CHECK_NEAR(b.out.view.load(44, 32).r, 0.0, 1e-6);
}

TP_TEST(diffusion_lab_lens_scales_radius) {
    const ModuleDef& m = find("diffusion_lab");
    Clip c = dlSpotClip();
    ParamValues base = with(defaultValues(m), {{"mode", 1}, {"strength", 1.0}, {"scatter_radius", 4}, {"shadow_protection", 0}, {"encoding", 0}});
    double nearLens = renderAt(m, c, 0, base).out.view.load(50, 24).r;
    double longLens = renderAt(m, c, 0, with(base, {{"focal_length", 200}})).out.view.load(50, 24).r;
    CHECK(longLens > nearLens);
}

TP_TEST(diffusion_lab_glimmer_makes_orthogonal_star) {
    const ModuleDef& m = find("diffusion_lab");
    Clip c = makeClip(64, 64, 1, [](int x, int y, int) {
        bool hot = x >= 30 && x < 34 && y >= 30 && y < 34;
        return hot ? RGBA{8.f, 8.f, 8.f, 1.f} : RGBA{0.f, 0.f, 0.f, 1.f};
    });
    ParamValues p = with(defaultValues(m), {{"mode", 2}, {"strength", 1.0}, {"scatter_radius", 0}, {"streak_length", 24}, {"streak_angle", 0}, {"encoding", 0}});
    RenderResult r = renderAt(m, c, 0, p);
    const ImageView& o = r.out.view;
    CHECK(o.load(44, 31).r > 0.f);
    CHECK(o.load(20, 31).r > 0.f);
    CHECK(o.load(31, 44).r > 0.f);
    CHECK(o.load(31, 20).r > 0.f);
    CHECK_NEAR(o.load(31, 44).r, o.load(44, 31).r, 1e-4);
    CHECK_NEAR(o.load(44, 44).r, 0.0, 1e-6);
    CHECK_NEAR(o.load(20, 20).r, 0.0, 1e-6);
}

TP_TEST(diffusion_lab_fog_uniform_gray_matches_formula) {
    const ModuleDef& m = find("diffusion_lab");
    Clip c = constantClip(RGBA{0.4f, 0.4f, 0.4f, 1.f});
    for (double s : {0.5, 1.0}) {
        ParamValues p = with(defaultValues(m), {{"mode", 3}, {"strength", s}, {"encoding", 0}});
        RenderResult q = renderAt(m, c, 3, p);
        for (int y = 0; y < 24; ++y)
            for (int x = 0; x < 32; ++x) {
                RGBA o = q.out.view.load(x, y);
                CHECK_NEAR(o.g, 0.4 * (1.0 + 0.4 * s), 1e-5);
                CHECK_NEAR(o.a, 1.0, 1e-6);
            }
    }
    ParamValues dark = with(defaultValues(m), {{"mode", 3}, {"strength", 1.0}, {"encoding", 0}});
    dark.set("tint", 0.0, 0.0, 0.0);
    RenderResult r = renderAt(m, c, 3, dark);
    CHECK_NEAR(r.out.view.load(5, 5).r, 0.4 * 0.4 + 0.4 * 0.7, 1e-5);
}

TP_TEST(diffusion_lab_new_modes_behave) {
    const ModuleDef& m = find("diffusion_lab");
    Clip flat = constantClip(RGBA{0.4f, 0.4f, 0.4f, 1.f}, 48, 32, 3);
    ParamValues pearlP = with(defaultValues(m), {{"mode", 5}, {"encoding", 0}, {"strength", 0.5}});
    RGBA pr = mean(renderAt(m, flat, 1, pearlP).out.view);
    CHECK_NEAR(luma(pr), 0.4, 0.02);
    ParamValues soft = with(defaultValues(m), {{"mode", 7}, {"encoding", 0}, {"strength", 1}});
    Clip blk = constantClip(RGBA{0.f, 0.f, 0.f, 1.f}, 48, 32, 3);
    CHECK(maxDiff(renderAt(m, blk, 1, soft).out.view, blk.frames[1].view) < 1e-6);
    Clip spot = makeClip(64, 48, 3, [](int x, int y, int) { return (std::abs(x - 32) < 3 && std::abs(y - 24) < 3) ? RGBA{4, 4, 4, 1} : RGBA{0.02f, 0.02f, 0.02f, 1}; });
    ParamValues bloom = with(defaultValues(m), {{"mode", 6}, {"encoding", 0}, {"strength", 0.5}, {"scatter_radius", 6}});
    RenderResult b = renderAt(m, spot, 1, bloom);
    CHECK(b.out.view.load(40, 24).r > spot.frames[1].view.load(40, 24).r + 0.01f);
    ParamValues bronze = with(defaultValues(m), {{"mode", 8}, {"encoding", 0}, {"strength", 0.8}});
    RGBA g = renderAt(m, spot, 1, bronze).out.view.load(32, 30);
    CHECK(g.r >= g.b);
    ParamValues jelly = with(defaultValues(m), {{"mode", 9}, {"encoding", 0}, {"strength", 1}});
    Clip grad = makeClip(64, 48, 3, [](int x, int, int) { return RGBA{(x % 4) < 2 ? 0.8f : 0.1f, 0.3f, 0.3f, 1}; });
    RenderResult j = renderAt(m, grad, 1, jelly);
    CHECK(std::fabs(j.out.view.load(32, 24).r - grad.frames[1].view.load(32, 24).r) < 0.02f);
    CHECK(std::fabs(j.out.view.load(1, 1).r - grad.frames[1].view.load(1, 1).r) > 0.05f);
}
