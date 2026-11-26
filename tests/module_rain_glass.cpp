#include "harness.h"
#include "modules/rain_engine.h"
#include "gpu/gpu.h"

using namespace tpt;

TP_TEST(rain_glass_without_rain_blur_or_mist_is_identity) {
    const ModuleDef& m = find("rain_glass");
    Clip c = patternClip(96, 64, 3);
    ParamValues p = with(defaultValues(m), {{"amount", 0.0}, {"defocus", 0.0}, {"mist", 0.0}, {"encoding", 0}});
    CHECK(maxDiff(renderAt(m, c, 1, p).out.view, c.frames[1].view) < 1e-6);
}

TP_TEST(rain_glass_simulation_is_a_pure_function_of_time) {
    RainSettings s;
    s.amount = 1.f;
    s.slide = 1.f;
    std::vector<RainDrop> a, b;
    std::vector<RainWipe> wa, wb;
    s.seconds = 7.25;
    rainSimulate(s, a, wa);
    s.seconds = 3.0;
    rainSimulate(s, b, wb);
    s.seconds = 7.25;
    rainSimulate(s, b, wb);
    CHECK(a.size() == b.size() && wa.size() == wb.size() && !a.empty() && !wa.empty());
    for (size_t i = 0; i < a.size(); ++i) CHECK(a[i].x == b[i].x && a[i].y == b[i].y && a[i].r == b[i].r);
}

TP_TEST(rain_glass_running_drops_move_down_smoothly) {
    RainSettings s;
    s.amount = 1.f / (700.f * s.aspect) + 1e-4f;
    s.slide = 1.f;
    s.maxR = 0.03f;
    s.minR = 0.012f;
    s.slideR = 0.02f;
    int runs = 0;
    for (uint32_t seed = 1; seed <= 300 && runs < 5; ++seed) {
        s.seed = seed;
        std::vector<RainDrop> d;
        std::vector<RainWipe> w;
        double prevY = 0, prevX = 0, drop = 0, jump = 0;
        int frames = 0;
        for (int f = 0; f < 300; ++f) {
            s.seconds = f / 30.0;
            rainSimulate(s, d, w);
            if (w.empty() || d.empty()) { frames = 0; continue; }
            const RainDrop& head = d.back();
            if (frames > 0) {
                drop += head.y - prevY;
                jump = std::max(jump, std::hypot(head.x - prevX, head.y - prevY));
            }
            prevX = head.x;
            prevY = head.y;
            if (++frames == 20) break;
        }
        if (frames < 20) continue;
        ++runs;
        CHECK(drop < -0.005);
        CHECK(jump < 0.05);
    }
    CHECK(runs >= 3);
}

TP_TEST(rain_glass_draws_water_and_refracts_inside_drops) {
    const ModuleDef& m = find("rain_glass");
    Clip c = patternClip(160, 96, 3);
    ParamValues p = with(defaultValues(m), {{"amount", 0.3}, {"min_size", 4.0}, {"max_size", 24.0}, {"slide_size", 40.0}, {"defocus", 0.0}, {"encoding", 0}});
    const RenderResult matte = renderAt(m, c, 1, with(p, {{"output", 1}}));
    double water = 0;
    for (int y = 0; y < 96; ++y)
        for (int x = 0; x < 160; ++x) water += matte.out.view.load(x, y).r;
    water /= 160.0 * 96.0;
    CHECK(water > 0.02 && water < 0.9);
    const RenderResult look = renderAt(m, c, 1, p);
    CHECK(maxDiff(look.out.view, c.frames[1].view) > 0.05);
}

TP_TEST(rain_glass_focus_on_the_scene_keeps_it_sharp) {
    const ModuleDef& m = find("rain_glass");
    Clip c = patternClip(96, 64, 3);
    ParamValues p = with(defaultValues(m), {{"amount", 0.0}, {"defocus", 40.0}, {"mist", 0.0}, {"encoding", 0}});
    CHECK(maxDiff(renderAt(m, c, 1, with(p, {{"focus", 1.0}})).out.view, c.frames[1].view) < 1e-6);
    CHECK(maxDiff(renderAt(m, c, 1, with(p, {{"focus", 0.0}})).out.view, c.frames[1].view) > 0.05);
}

TP_TEST(rain_glass_large_drops_are_not_circles_and_small_ones_nearly_are) {
    const auto spread = [](float r, float rMax, uint32_t id) {
        RainDrop d;
        d.r = r;
        rainShape(d, id, rMax);
        float lo = 1e9f, hi = 0.f;
        for (int a = 0; a < 90; ++a) {
            const float t = a * 6.2831853f / 90.f, c = std::cos(t), s = std::sin(t);
            float in = 0.f, out = 3.f * r;
            for (int i = 0; i < 40; ++i) {
                const float mid = 0.5f * (in + out);
                (rainDropHeight(d, mid * c, mid * s) > 0.f ? in : out) = mid;
            }
            lo = std::min(lo, in);
            hi = std::max(hi, in);
        }
        return hi / lo;
    };
    double big = 0, small = 0;
    for (uint32_t id = 1; id <= 40; ++id) {
        big += spread(0.03f, 0.03f, id * 7919u);
        small += spread(0.002f, 0.03f, id * 7919u);
    }
    big /= 40;
    small /= 40;
    CHECK(big > 1.12);
    CHECK(small < 1.08);
}

TP_TEST(rain_glass_drops_above_the_slide_size_never_stay_put) {
    RainSettings s;
    s.amount = 1.5f;
    s.maxR = 0.03f;
    s.minR = 0.002f;
    s.slideR = 0.012f;
    s.slide = 0.f;
    std::vector<RainDrop> a, b;
    std::vector<RainWipe> w;
    int stuck = 0, big = 0;
    for (int f = 0; f < 6; ++f) {
        s.seconds = 6.0 + f * 1.3;
        rainSimulate(s, a, w);
        s.seconds += 1.0;
        rainSimulate(s, b, w);
        for (const RainDrop& x : b) {
            if (x.r <= 1.05f * s.slideR || x.runner != -1) continue;
            ++big;
            for (const RainDrop& y : a)
                if (y.id == x.id && y.r > 1.05f * s.slideR && std::fabs(y.x - x.x) < 1e-6f && std::fabs(y.y - x.y) < 1e-6f) ++stuck;
        }
    }
    CHECK(stuck == 0);
}
TP_TEST(rain_glass_gpu_matches_cpu) {
    const ModuleDef& m = find("rain_glass");
    Clip c = patternClip(320, 180, 3);
    const ParamValues p = with(defaultValues(m), {{"amount", 0.6}, {"min_size", 6.0}, {"max_size", 30.0}, {"defocus", 40.0}, {"mist", 0.5}, {"encoding", 0}});
    const std::vector<std::pair<const char*, ParamValues>> cases = {
        {"base", p},
        {"focus", with(p, {{"focus", 1.0}})},
        {"rack gamma", with(p, {{"focus", 0.5}, {"encoding", 1}})},
        {"dry", with(p, {{"mist", 0.0}, {"focus", 0.3}})},
        {"sharp", with(p, {{"defocus", 0.0}})},
        {"matte", with(p, {{"output", 1}})},
    };
    const bool wasEnabled = gpu::enabled();
    struct RestoreGpu { bool enabled; ~RestoreGpu() { gpu::setEnabled(enabled); } } restore{wasEnabled};
    std::vector<RenderResult> cpu;
    gpu::setEnabled(false);
    for (const auto& k : cases) cpu.push_back(renderAt(m, c, 1, k.second));
    gpu::setEnabled(wasEnabled);
    if (!gpu::available()) return;
    for (size_t i = 0; i < cases.size(); ++i) {
        const RenderResult g = renderAt(m, c, 1, cases[i].second);
        const double d = maxDiff(cpu[i].out.view, g.out.view);
        std::printf("  rain CPU vs CUDA %s: max %.2e\n", cases[i].first, d);
        CHECK(d < 2e-3);
    }
}
