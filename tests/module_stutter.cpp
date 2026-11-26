#include "harness.h"

using namespace tpt;

static Clip stutterIndexClip(int n = 60) {
    return makeClip(4, 4, n, [](int, int, int f) { return RGBA{f / 100.f, 0.5f, 0.25f, 1.f}; });
}

static int stutterShown(const ModuleDef& m, const Clip& c, int t, const ParamValues& p) {
    RenderResult r = renderAt(m, c, t, p);
    return static_cast<int>(std::lround(r.out.view.load(1, 1).r * 100.f));
}

TP_TEST(stutter_hold_one_is_identity) {
    const ModuleDef& m = find("stutter");
    Clip c = patternClip();
    ParamValues p = with(defaultValues(m), {{"mode", 0}, {"hold", 1}, {"reference_frame", 7}});
    for (int t : {0, 5, 13, 39}) {
        RenderResult r = renderAt(m, c, t, p);
        CHECK(maxDiff(r.out.view, c.frames[t].view) == 0.0);
    }
}

TP_TEST(stutter_frame_hold_snaps_to_block_start) {
    const ModuleDef& m = find("stutter");
    Clip c = stutterIndexClip();
    ParamValues p = with(defaultValues(m), {{"mode", 0}, {"hold", 4}, {"reference_frame", 2}});
    for (int t = 2; t < 30; ++t) CHECK(stutterShown(m, c, t, p) == 2 + ((t - 2) / 4) * 4);
    CHECK(stutterShown(m, c, 9, with(p, {{"reference_frame", 10}})) == 6);
    CHECK(renderAt(m, c, 9, p).window.before == 3);
}

TP_TEST(stutter_reverse_loop_plays_segments_backwards) {
    const ModuleDef& m = find("stutter");
    Clip c = stutterIndexClip();
    ParamValues p = with(defaultValues(m), {{"mode", 1}, {"loop_length", 4}, {"repeat", 1}, {"reference_frame", 0}});
    const int expect[] = {3, 2, 1, 0, 7, 6, 5, 4};
    for (int t = 0; t < 8; ++t) CHECK(stutterShown(m, c, t + 8, p) == expect[t] + 8);
    ParamValues pp = with(p, {{"repeat", 2}});
    const int ping[] = {3, 2, 1, 0, 0, 1, 2, 3, 11, 10};
    for (int t = 0; t < 10; ++t) CHECK(stutterShown(m, c, t + 8, pp) == ping[t] + 8);
}

TP_TEST(stutter_rhythmic_repeat_repeats_slice_then_plays) {
    const ModuleDef& m = find("stutter");
    Clip c = stutterIndexClip();
    ParamValues p = with(defaultValues(m), {{"mode", 2}, {"hold", 2}, {"repeat", 3}, {"bpm", 120}, {"chance", 1}, {"reference_frame", 0}});
    const int expect[] = {12, 13, 12, 13, 12, 13, 18, 19, 20, 21, 22, 23, 24, 25, 24};
    for (int t = 0; t < 15; ++t) CHECK(stutterShown(m, c, 12 + t, p) == expect[t]);
    ParamValues off = with(p, {{"chance", 0}});
    for (int t = 12; t < 40; ++t) CHECK(stutterShown(m, c, t, off) == t);
}

TP_TEST(stutter_chance_is_seeded_and_window_capped) {
    const ModuleDef& m = find("stutter");
    Clip c = stutterIndexClip();
    ParamValues p = with(defaultValues(m), {{"mode", 2}, {"hold", 3}, {"repeat", 4}, {"chance", 0.5}, {"seed", 42}});
    for (int b = 0; b < 4; ++b) {
        int t = 12 * b + 4;
        int a1 = stutterShown(m, c, t, p), a2 = stutterShown(m, c, t, p);
        CHECK(a1 == a2);
        CHECK(a1 == t || a1 == 12 * b + 1);
    }
    Clip longClip = stutterIndexClip(100);
    auto stutters = [&](const ParamValues& q, int b) { int t = 12 * b + 4; return stutterShown(m, longClip, t, q) != t; };
    int hits = 0, seedDiff = 0;
    for (int b = 0; b < 8; ++b) {
        bool lo = stutters(with(p, {{"chance", 0.3}}), b), mid = stutters(p, b), hi = stutters(with(p, {{"chance", 0.8}}), b);
        CHECK(!lo || mid);
        CHECK(!mid || hi);
        hits += mid ? 1 : 0;
        seedDiff += mid != stutters(with(p, {{"seed", 43}}), b) ? 1 : 0;
    }
    CHECK(hits > 0 && hits < 8);
    CHECK(seedDiff > 0);
    ParamValues big = with(defaultValues(m), {{"mode", 1}, {"loop_length", 64}, {"repeat", 16}});
    RenderResult r = renderAt(m, c, 30, big);
    CHECK(r.window.before <= 64 && r.window.after <= 64);
    for (int f : r.fetched) CHECK(f >= 30 - r.window.before && f <= 30 + r.window.after);
}
