#include "harness.h"
#include <algorithm>
#include <tuple>

using namespace tpt;

TP_TEST(pixel_grid1_pixelate_is_identity) {
    const ModuleDef& m = find("pixel");
    Clip c = patternClip();
    RenderResult r = renderAt(m, c, 10, with(defaultValues(m), {{"mode", 0}, {"grid", 1}}));
    CHECK(maxDiff(r.out.view, c.frames[10].view) == 0.0);
}

TP_TEST(pixel_constant_unchanged_by_pixelate_mosaic_sort) {
    const ModuleDef& m = find("pixel");
    Clip c = constantClip(RGBA{0.4f, 0.55f, 0.3f, 1.f}, 61, 37, 3);
    for (int mode : {0, 1, 2}) {
        for (double g : {1.0, 5.0, 12.0, 200.0}) {
            ParamValues p = with(defaultValues(m), {{"mode", mode}, {"grid", g}, {"variance_threshold", 0.0}, {"sort_low", 0.0}, {"sort_high", 1.0}});
            CHECK(maxDiff(renderAt(m, c, 1, p).out.view, c.frames[1].view) < 1e-6);
        }
    }
}

TP_TEST(pixel_pixelate_cell_is_box_average) {
    const ModuleDef& m = find("pixel");
    Clip c = makeClip(10, 6, 1, [](int x, int y, int) { return RGBA{static_cast<float>(x), static_cast<float>(y), 0.5f, 1.f}; });
    RenderResult r = renderAt(m, c, 0, with(defaultValues(m), {{"mode", 0}, {"grid", 4}}));
    CHECK_NEAR(r.out.view.load(0, 0).r, 1.5, 1e-6);
    CHECK_NEAR(r.out.view.load(3, 3).g, 1.5, 1e-6);
    CHECK_NEAR(r.out.view.load(5, 1).r, 5.5, 1e-6);
    CHECK_NEAR(r.out.view.load(9, 5).r, 8.5, 1e-6);
    CHECK_NEAR(r.out.view.load(9, 5).g, 4.5, 1e-6);
    RenderResult half = renderAt(m, c, 0, with(defaultValues(m), {{"mode", 0}, {"grid", 8}}), 0.5);
    CHECK_NEAR(half.out.view.load(5, 1).r, 5.5, 1e-6);
}

TP_TEST(pixel_mosaic_splits_detail_and_keeps_flat_cells) {
    const ModuleDef& m = find("pixel");
    Clip c = makeClip(64, 64, 1, [](int x, int y, int) { float v = (x >= 48 && y >= 48 && ((x + y) & 1)) ? 1.f : 0.f; return RGBA{v, v, v, 1.f}; });
    RenderResult r = renderAt(m, c, 0, with(defaultValues(m), {{"mode", 1}, {"grid", 8}, {"variance_threshold", 0.01}}));
    CHECK(maxDiff(r.out.view, renderAt(m, c, 0, with(defaultValues(m), {{"mode", 0}, {"grid", 1}})).out.view) > 0.4);
    CHECK_NEAR(r.out.view.load(5, 5).r, 0.0, 1e-7);
    CHECK_NEAR(r.out.view.load(50, 50).r, 0.5, 1e-6);
    CHECK_NEAR(r.out.view.load(33, 33).r, 0.0, 1e-7);
}

static void pixelCheckSortedPermutation(const ImageView& in, const ImageView& out, bool vertical, bool desc, float lo, float hi) {
    const int L = vertical ? in.bounds.h() : in.bounds.w();
    const int N = vertical ? in.bounds.w() : in.bounds.h();
    for (int k = 0; k < N; ++k) {
        auto get = [&](const ImageView& v, int i) { return vertical ? v.load(k, i) : v.load(i, k); };
        std::vector<std::tuple<float, float, float, float>> a, b;
        for (int i = 0; i < L; ++i) {
            RGBA p = get(in, i), q = get(out, i);
            a.emplace_back(p.r, p.g, p.b, p.a);
            b.emplace_back(q.r, q.g, q.b, q.a);
        }
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        CHECK(a == b);
        for (int i = 0; i < L; ++i) {
            float li = luma(get(in, i));
            bool inRun = li >= lo && li <= hi;
            if (!inRun) { RGBA p = get(in, i), q = get(out, i); CHECK(p.r == q.r && p.g == q.g && p.b == q.b && p.a == q.a); continue; }
            if (i + 1 < L) {
                float ln = luma(get(in, i + 1));
                if (ln >= lo && ln <= hi) {
                    float o0 = luma(get(out, i)), o1 = luma(get(out, i + 1));
                    CHECK(desc ? o0 >= o1 : o0 <= o1);
                }
            }
        }
    }
}

TP_TEST(pixel_sort_rows_are_sorted_permutations) {
    const ModuleDef& m = find("pixel");
    Clip c = makeClip(50, 30, 1, [](int x, int y, int) {
        float h = hashFloat(7, x, y);
        return RGBA{h, hashFloat(8, x, y) * 0.8f, 0.3f + 0.5f * h * h, 0.5f + 0.5f * hashFloat(9, x, y)};
    });
    const float lo = 0.2f, hi = 0.7f;
    for (int dir : {0, 1})
        for (int ord : {0, 1}) {
            ParamValues p = with(defaultValues(m), {{"mode", 2}, {"direction", dir}, {"sort_order", ord}, {"sort_low", lo}, {"sort_high", hi}});
            RenderResult r = renderAt(m, c, 0, p);
            pixelCheckSortedPermutation(c.frames[0].view, r.out.view, dir == 1, ord == 1, lo, hi);
            CHECK(maxDiff(r.out.view, c.frames[0].view) > 0.0);
        }
}

TP_TEST(pixel_dither_depth256_amount0_near_identity) {
    const ModuleDef& m = find("pixel");
    Clip c = makeClip(40, 30, 1, [](int x, int y, int) { return RGBA{x / 40.f, y / 30.f, 0.5f * (x + y) / 70.f, 1.f}; });
    RenderResult r = renderAt(m, c, 0, with(defaultValues(m), {{"mode", 4}, {"grid", 1}, {"color_depth", 256}, {"dither_amount", 0}}));
    CHECK(maxDiff(r.out.view, c.frames[0].view) <= 0.5 / 255.0 + 1e-6);
    RenderResult two = renderAt(m, c, 0, with(defaultValues(m), {{"mode", 4}, {"grid", 1}, {"color_depth", 2}, {"dither_amount", 1}}));
    for (int y = 0; y < 30; ++y)
        for (int x = 0; x < 40; ++x) {
            RGBA v = two.out.view.load(x, y);
            CHECK(v.r == 0.f || v.r == 1.f);
        }
    CHECK_NEAR(mean(two.out.view).r, mean(c.frames[0].view).r, 0.03);
}

TP_TEST(pixel_melt_zero_amount_identity_and_frozen_persistence) {
    const ModuleDef& m = find("pixel");
    Clip c = patternClip();
    RenderResult zero = renderAt(m, c, 10, with(defaultValues(m), {{"mode", 5}, {"melt_amount", 0}}));
    CHECK(maxDiff(zero.out.view, c.frames[10].view) == 0.0);
    Clip still = makeClip(64, 48, 30, [](int x, int y, int) { return RGBA{x / 64.f, y / 48.f, 0.5f, 1.f}; });
    ParamValues frozen = with(defaultValues(m), {{"mode", 5}, {"melt_amount", 30}, {"persistence", 1}});
    CHECK(maxDiff(renderAt(m, still, 3, frozen).out.view, renderAt(m, still, 25, frozen).out.view) == 0.0);
    ParamValues live = with(frozen, {{"persistence", 0}});
    CHECK(maxDiff(renderAt(m, still, 3, live).out.view, renderAt(m, still, 25, live).out.view) > 0.0);
    CHECK(maxDiff(renderAt(m, still, 3, frozen).out.view, still.frames[3].view) > 0.0);
}

TP_TEST(pixel_ascii_draws_glyphs_on_black) {
    const ModuleDef& m = find("pixel");
    Clip white = constantClip(RGBA{1, 1, 1, 1}, 48, 32, 1);
    RenderResult r = renderAt(m, white, 0, with(defaultValues(m), {{"mode", 3}, {"grid", 16}, {"ascii_color", 1}}));
    RGBA avg = mean(r.out.view);
    CHECK(avg.r > 0.2f && avg.r < 0.8f);
    CHECK_NEAR(avg.a, 1.0, 1e-6);
    Clip black = constantClip(RGBA{0, 0, 0, 1}, 48, 32, 1);
    CHECK_NEAR(mean(renderAt(m, black, 0, with(defaultValues(m), {{"mode", 3}})).out.view).r, 0.0, 1e-7);
}

TP_TEST(pixel_ascii_glyph_is_upright_at_ramp_end) {
    const ModuleDef& m = find("pixel");
    Clip white = constantClip(RGBA{1, 1, 1, 1}, 24, 24, 1);
    RenderResult r = renderAt(m, white, 0, with(defaultValues(m), {{"mode", 3}, {"grid", 12}, {"ascii_color", 1}}));
    const ImageView& v = r.out.view;
    CHECK(v.load(0, 11).r == 0.f);
    CHECK(v.load(2, 11).r == 1.f);
    CHECK(v.load(6, 11).r == 1.f);
    CHECK(v.load(8, 11).r == 0.f);
    for (int x = 0; x < 12; ++x) CHECK(v.load(x, 0).r == 0.f && v.load(x, 0).a == 1.f);
    CHECK(v.load(0, 9).r == 1.f);
    CHECK(v.load(14, 23).r == 1.f);
}
