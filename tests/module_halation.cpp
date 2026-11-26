#include "harness.h"

using namespace tpt;

static Clip halImpulseClip(int at) {
    return makeClip(48, 48, 24, [at](int x, int y, int f) {
        bool hot = f == at && x >= 22 && x < 26 && y >= 22 && y < 26;
        return hot ? RGBA{6.f, 6.f, 6.f, 1.f} : RGBA{0.f, 0.f, 0.f, 1.f};
    });
}

TP_TEST(halation_below_threshold_is_unchanged) {
    const ModuleDef& m = find("halation");
    Clip c = makeClip(40, 30, 12, [](int x, int y, int f) {
        float v = 0.05f + 0.5f * x / 40.f + 0.01f * (f % 3) + 0.001f * y;
        return RGBA{v, v * 0.9f, v * 0.8f, 1.f};
    });
    for (int e = 0; e < 2; ++e) {
        ParamValues p = with(defaultValues(m), {{"encoding", static_cast<double>(e)}, {"threshold", 0.8}, {"knee", 0.25}});
        RenderResult r = renderAt(m, c, 9, p);
        CHECK(maxDiff(r.out.view, c.frames[9].view) == 0.0);
    }
}

TP_TEST(halation_impulse_leaves_decaying_trail) {
    const ModuleDef& m = find("halation");
    Clip c = halImpulseClip(5);
    ParamValues p = with(defaultValues(m), {{"encoding", 0}, {"radius", 8}, {"decay", 0.7}, {"duration", 8}, {"threshold", 0.8}});
    double before = renderAt(m, c, 4, p).out.view.load(24, 24).r;
    CHECK_NEAR(before, 0.0, 1e-7);
    double prev = 1e9;
    for (int t = 6; t <= 12; ++t) {
        double v = renderAt(m, c, t, p).out.view.load(24, 24).r;
        CHECK(v > 0.0);
        CHECK(v < prev);
        prev = v;
    }
    double gone = renderAt(m, c, 13, p).out.view.load(24, 24).r;
    CHECK_NEAR(gone, 0.0, 1e-7);
}

TP_TEST(halation_trail_ratio_matches_decay) {
    const ModuleDef& m = find("halation");
    Clip c = halImpulseClip(5);
    ParamValues p = with(defaultValues(m), {{"encoding", 0}, {"radius", 8}, {"decay", 0.5}, {"duration", 6}});
    double a = renderAt(m, c, 7, p).out.view.load(24, 24).g;
    double b = renderAt(m, c, 8, p).out.view.load(24, 24).g;
    CHECK_NEAR(b / a, 0.5, 1e-4);
}

TP_TEST(halation_energy_grows_with_intensity) {
    const ModuleDef& m = find("halation");
    Clip c = patternClip();
    double prev = -1;
    for (double k : {0.0, 0.5, 1.0, 2.0, 4.0}) {
        ParamValues p = with(defaultValues(m), {{"encoding", 0}, {"intensity", k}});
        RenderResult r = renderAt(m, c, 20, p);
        RGBA o = mean(r.out.view), s = mean(c.frames[20].view);
        double energy = (o.r - s.r) + (o.g - s.g) + (o.b - s.b);
        CHECK(energy >= 0.0);
        if (k > 0) CHECK(energy > prev);
        prev = energy;
    }
}

TP_TEST(halation_duration_does_not_change_static_brightness) {
    const ModuleDef& m = find("halation");
    Clip c = makeClip(32, 32, 40, [](int x, int y, int) {
        bool hot = x >= 12 && x < 20 && y >= 12 && y < 20;
        return hot ? RGBA{3.f, 2.f, 1.f, 1.f} : RGBA{0.1f, 0.1f, 0.1f, 1.f};
    });
    ParamValues p1 = with(defaultValues(m), {{"encoding", 0}, {"duration", 1}});
    ParamValues p2 = with(defaultValues(m), {{"encoding", 0}, {"duration", 16}, {"decay", 0.9}});
    RenderResult a = renderAt(m, c, 30, p1), b = renderAt(m, c, 30, p2);
    CHECK(maxDiff(a.out.view, b.out.view) < 1e-4);
}

TP_TEST(halation_halo_only_is_the_added_layer) {
    const ModuleDef& m = find("halation");
    Clip c = patternClip();
    ParamValues base = with(defaultValues(m), {{"encoding", 0}});
    RenderResult full = renderAt(m, c, 20, base);
    RenderResult halo = renderAt(m, c, 20, with(base, {{"halo_only", 1}}));
    double err = 0;
    for (int y = 0; y < c.rect.y2; ++y)
        for (int x = 0; x < c.rect.x2; ++x) {
            RGBA s = c.frames[20].view.load(x, y), f = full.out.view.load(x, y), h = halo.out.view.load(x, y);
            for (int ch = 0; ch < 3; ++ch) err = std::max(err, static_cast<double>(std::fabs(s[ch] + h[ch] - f[ch])));
        }
    CHECK(err < 1e-5);
}
