#include "harness.h"
#include "gpu/gpu.h"
#include "modules/glass_motion.h"

using namespace tpt;

TP_TEST(multi_optic_all_off_or_ior_one_is_identity) {
    const ModuleDef& m = find("multi_optic");
    Clip c = patternClip(96, 64, 3);
    ParamValues off = with(defaultValues(m), {{"e1_type", 0}, {"e2_type", 0}, {"e3_type", 0}});
    CHECK(maxDiff(renderAt(m, c, 1, off).out.view, c.frames[1].view) == 0.0);
    ParamValues flat = with(defaultValues(m), {{"e1_ior", 1}, {"e2_ior", 1}, {"dispersion", 0}, {"encoding", 0}});
    CHECK(maxDiff(renderAt(m, c, 1, flat).out.view, c.frames[1].view) < 1e-4);
}

TP_TEST(multi_optic_single_element_matches_refraction_lab) {
    const ModuleDef& mo = find("multi_optic");
    const ModuleDef& rl = find("refraction_lab");
    Clip c = patternClip(96, 64, 3);
    ParamValues a = with(defaultValues(mo), {{"e1_type", 2}, {"e2_type", 0}, {"e3_type", 0}, {"e1_ior", 1.6}, {"e1_curvature", 1}, {"e1_distance", 40}, {"e1_size", 0.5}, {"e1_rotation", 0}, {"dispersion", 0.01}, {"encoding", 0}});
    a.v["e1_position"] = {0.5, 0.5, 0};
    ParamValues b = with(defaultValues(rl), {{"mode", 1}, {"ior", 1.6}, {"curvature", 1}, {"thickness", 40}, {"size", 0.5}, {"rotation", 0}, {"dispersion", 0.01}, {"encoding", 0}});
    RenderResult ra = renderAt(mo, c, 1, a), rb = renderAt(rl, c, 1, b);
    double d = 0;
    for (int y = 12; y < 52; ++y)
        for (int x = 28; x < 68; ++x)
            for (int ch = 0; ch < 3; ++ch) d = std::max(d, static_cast<double>(std::fabs(ra.out.view.load(x, y)[ch] - rb.out.view.load(x, y)[ch])));
    CHECK(d < 3e-3);
    CHECK(maxDiff(ra.out.view, c.frames[1].view) > 0.01);
}

TP_TEST(multi_optic_order_matters_for_overlapping_elements) {
    const ModuleDef& m = find("multi_optic");
    Clip c = patternClip(96, 64, 3);
    ParamValues p = with(defaultValues(m), {{"e1_type", 2}, {"e2_type", 3}, {"e3_type", 0}, {"e1_size", 0.9}, {"e2_size", 0.9}, {"encoding", 0}});
    p.v["e1_position"] = {0.5, 0.5, 0};
    p.v["e2_position"] = {0.5, 0.5, 0};
    RenderResult a = renderAt(m, c, 1, p), b = renderAt(m, c, 1, with(p, {{"element_order", 2}}));
    CHECK(maxDiff(a.out.view, c.frames[1].view) > 0.02);
    CHECK(maxDiff(a.out.view, b.out.view) > 1e-3);
}

TP_TEST(multi_optic_halo_mask_leaves_a_clear_center) {
    const ModuleDef& m = find("multi_optic");
    Clip c = patternClip(128, 96, 3);
    auto p = with(defaultValues(m), {{"e1_type", 6}, {"e1_size", 0.625}, {"e1_ring_width", 0.30}, {"preview", 1}});
    p.set("e1_position", 0.5, 0.5);
    const auto result = renderAt(m, c, 1, p);
    CHECK(result.out.view.load(64, 48).r == 0);
    CHECK(result.out.view.load(98, 48).r > 0.99);
    CHECK(result.out.view.load(120, 48).r == 0);
}

TP_TEST(multi_optic_motion_uses_time_and_focus_respects_the_footprint) {
    const ModuleDef& m = find("multi_optic");
    Clip c = makeClip(128, 96, 3, [](int x, int, int) {
        const float level = x < 64 ? 0.12f : 0.85f;
        return RGBA{level, level, level, 1.f};
    });
    auto p = with(defaultValues(m), {{"e1_type", 5}, {"e2_type", 0}, {"e1_curvature", 0}, {"f_stop", 1}, {"encoding", 0}});
    p.set("e1_position", 0.5, 0.5);
    auto focused = renderAt(m, c, 1, p);
    auto defocused = renderAt(m, c, 1, with(p, {{"focus_shift", 1}}));
    CHECK(maxDiff(focused.out.view, defocused.out.view) > 0.05);
    CHECK(focused.out.view.load(0, 0).r == defocused.out.view.load(0, 0).r);
    auto motion = with(p, {{"preview", 1}, {"motion_mode", 3}, {"motion_amount", 0.28}, {"motion_rate", 2}});
    auto before = renderAt(m, c, 0, motion, 1, 24);
    auto after = renderAt(m, c, 2, motion, 1, 24);
    CHECK(maxDiff(before.out.view, after.out.view) > 0.5);
    auto sameSeconds = renderAt(m, c, 1, motion, 1, 12);
    CHECK(maxDiff(sameSeconds.out.view, after.out.view) < 1e-5);
}

TP_TEST(multi_optic_prism_array_repeats_the_same_source_in_each_facet) {
    const ModuleDef& m = find("multi_optic");
    Clip c = makeClip(100, 64, 2, [](int x, int, int) {
        const float level = x >= 44 && x < 56 ? 1.f : 0.f;
        return RGBA{level, level, level, 1.f};
    });
    auto p = with(defaultValues(m), {{"e1_type", 8}, {"e1_size", 0.8}, {"e1_curvature", 0},
                                    {"e2_type", 0}, {"facets", 4}, {"dispersion", 0}, {"encoding", 0}});
    p.set("e1_position", 0.5, 0.5);
    const auto result = renderAt(m, c, 1, p);
    for (int center : {20, 40, 60, 80}) CHECK(result.out.view.load(center, 32).r > 0.99f);
    CHECK(result.out.view.load(0, 32).r == 0);
}

TP_TEST(multi_optic_prism_array_corners_blend_coverage_without_warping_the_source) {
    const ModuleDef& m = find("multi_optic");
    Clip c = makeClip(100, 64, 2, [](int x, int, int) {
        const float bar = x >= 38 && x < 44 ? 1.f : 0.f;
        return RGBA{bar, bar, bar, 1.f};
    });
    auto p = with(defaultValues(m), {{"e1_type", 8}, {"e1_size", 0.8}, {"e1_aspect", 0.5},
        {"e1_curvature", 0}, {"e2_type", 0}, {"facets", 4}, {"edge_softness", 0.09}, {"dispersion", 0}, {"encoding", 0}});
    p.set("e1_position", 0.5, 0.5);
    const auto mask = renderAt(m, c, 1, with(p, {{"preview", 1}}));
    const auto color = renderAt(m, c, 1, p);
    const float coverage = mask.out.view.load(10, 12).r;
    CHECK(coverage > 0.2f && coverage < 0.8f);
    CHECK(std::fabs(color.out.view.load(10, 12).r - coverage) < 0.001f);
    CHECK(color.out.view.load(2, 12).r == 0);
}

TP_TEST(multi_optic_prism_array_blends_the_two_sides_of_a_subpixel_seam) {
    const ModuleDef& m = find("multi_optic");
    Clip c = makeClip(100, 64, 2, [](int x, int, int) {
        const float level = x >= 50 ? 1.f : 0.f;
        return RGBA{level, level, level, 1.f};
    });
    auto p = with(defaultValues(m), {{"e1_type", 8}, {"e1_size", 0.8}, {"e1_curvature", 0},
        {"e2_type", 0}, {"facets", 4}, {"e1_facet_feather", 2.5}, {"e1_facet_defocus", 0},
        {"dispersion", 0}, {"encoding", 0}});
    p.set("e1_position", 0.505, 0.5);
    const auto mixed = renderAt(m, c, 1, p);
    CHECK(std::fabs(mixed.out.view.load(50, 32).r - 0.5f) < 0.001f);
    CHECK(mixed.out.view.load(49, 32).r > 0.5f && mixed.out.view.load(49, 32).r < 1.f);
    CHECK(mixed.out.view.load(51, 32).r > 0.f && mixed.out.view.load(51, 32).r < 0.5f);
    const auto minFeather = renderAt(m, c, 1, with(p, {{"e1_facet_feather", 0}}));
    CHECK(std::fabs(minFeather.out.view.load(50, 32).r - 0.5f) < 0.001f);
    CHECK(minFeather.out.view.load(49, 32).r > 0.99f);
    CHECK(minFeather.out.view.load(51, 32).r < 0.01f);
}

TP_TEST(multi_optic_prism_array_defocus_is_local_to_the_seam) {
    const ModuleDef& m = find("multi_optic");
    Clip c = makeClip(100, 64, 2, [](int, int y, int) {
        const float level = y >= 32 ? 1.f : 0.f;
        return RGBA{level, level, level, 1.f};
    });
    auto p = with(defaultValues(m), {{"e1_type", 8}, {"e1_size", 0.8}, {"e1_curvature", 0},
        {"e2_type", 0}, {"facets", 4}, {"e1_facet_feather", 4}, {"e1_facet_defocus", 5}, {"dispersion", 0}, {"encoding", 0}});
    p.set("e1_position", 0.505, 0.5);
    const auto soft = renderAt(m, c, 1, p);
    const auto sharp = renderAt(m, c, 1, with(p, {{"e1_facet_defocus", 0}}));
    CHECK(sharp.out.view.load(50, 32).r > 0.99f);
    CHECK(soft.out.view.load(50, 32).r > 0.5f && soft.out.view.load(50, 32).r < 0.9f);
    CHECK(soft.out.view.load(60, 32).r == sharp.out.view.load(60, 32).r);
}

TP_TEST(multi_optic_prism_array_keeps_the_field_and_facet_centers_neutral) {
    const ModuleDef& m = find("multi_optic");
    Clip c = makeClip(320, 180, 2, [](int x, int, int) {
        const float level = (x & 1) ? 0.9f : 0.1f;
        return RGBA{level, level, level, 1.f};
    });
    auto p = with(defaultValues(m), {{"e1_type", 8}, {"e1_size", 0.96}, {"e1_curvature", 0},
        {"e2_type", 0}, {"e1_facet_defocus", 0}, {"facets", 4}, {"dispersion", 0.012}, {"encoding", 0}});
    p.set("e1_position", 0.5, 0.5);
    const auto result = renderAt(m, c, 1, p);
    auto colorSpread = [&](int x, int y) {
        const RGBA pixel = result.out.view.load(x, y);
        return std::max(pixel.r, pixel.b) - std::min(pixel.r, pixel.b);
    };
    for (int x = 138; x <= 182; ++x) CHECK(colorSpread(x, 89) < 1e-6f);
    CHECK(colorSpread(44, 89) < 1e-6f);
    float outer = 0;
    for (int x = 291; x < 314; ++x) outer = std::max(outer, colorSpread(x, 89));
    CHECK(outer > 0.03f && outer < 0.25f);
}

TP_TEST(multi_optic_prism_array_prefilters_minified_texture_along_the_facet_axis) {
    const ModuleDef& m = find("multi_optic");
    Clip c = makeClip(100, 64, 2, [](int x, int y, int) {
        return RGBA{(x & 1) ? 1.f : 0.f, (y & 1) ? 1.f : 0.f, 0.f, 1.f};
    });
    auto p = with(defaultValues(m), {{"e1_type", 8}, {"e1_size", 0.8}, {"e1_curvature", 0}, {"e1_repeat_scale", 2},
        {"e2_type", 0}, {"facets", 4}, {"e1_facet_defocus", 0}, {"e1_facet_feather", 0}, {"dispersion", 0}, {"encoding", 0}});
    p.set("e1_position", 0.505, 0.5);
    const auto result = renderAt(m, c, 1, p);
    for (int x = 15; x <= 25; ++x) {
        const RGBA even = result.out.view.load(x, 32), odd = result.out.view.load(x, 33);
        CHECK(std::fabs(even.r - 0.5f) < 0.1f);
        CHECK(even.g == 0.f && odd.g > 0.999f);
    }
}

TP_TEST(multi_optic_looks_render_differently_on_cpu_and_match_cuda) {
    const ModuleDef& m = find("multi_optic");
    Clip c = patternClip(256, 144, 3);
    const bool wasEnabled = gpu::enabled();
    struct RestoreGpu { bool enabled; ~RestoreGpu() { gpu::setEnabled(enabled); } } restore{wasEnabled};
    auto previous = renderAt(m, c, 1, defaultValues(m));
    for (int look = 1; look <= 3; ++look) {
        auto p = defaultValues(m);
        for (const auto& [name, value] : m.loadLook(m, look).v) p.v[name] = value;
        gpu::setEnabled(false);
        auto cpu = renderAt(m, c, 1, p);
        CHECK(maxDiff(cpu.out.view, previous.out.view) > 0.04);
        gpu::setEnabled(wasEnabled);
        if (gpu::available()) {
            auto cuda = renderAt(m, c, 1, p);
            CHECK(maxDiff(cpu.out.view, cuda.out.view) < 0.015);
        }
        previous = std::move(cpu);
    }
}

static double regionDiff(const ImageView& a, const ImageView& b, int x0, int y0, int x1, int y1) {
    double d = 0;
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x)
            for (int ch = 0; ch < 3; ++ch) d = std::max(d, static_cast<double>(std::fabs(a.load(x, y)[ch] - b.load(x, y)[ch])));
    return d;
}

TP_TEST(multi_optic_glass_solid_bends_only_its_footprint) {
    const ModuleDef& m = find("multi_optic");
    Clip c = patternClip(160, 96, 3);
    for (int shape : {1, 2, 6, 16, 17, 18, 19, 20}) {
        for (int quality : {0, 1}) {
            ParamValues p = with(defaultValues(m), {{"solid_shape", shape}, {"solid_size", 0.35}, {"quality", quality}, {"encoding", 0}, {"scene_defocus", 0.0}});
            p.v["solid_position"] = {0.5, 0.5, 0};
            const RenderResult r = renderAt(m, c, 1, p);
            CHECK(regionDiff(r.out.view, c.frames[1].view, 0, 0, 30, 96) < 1e-5);
            CHECK(regionDiff(r.out.view, c.frames[1].view, 65, 35, 95, 61) > 0.02);
            p.v["scene_defocus"] = {1.0, 0, 0};
            const RenderResult soft = renderAt(m, c, 1, p);
            CHECK(regionDiff(soft.out.view, c.frames[1].view, 0, 0, 30, 96) > 0.02);
        }
    }
}

TP_TEST(multi_optic_diopter_plates_bend_only_their_footprint) {
    const ModuleDef& m = find("multi_optic");
    Clip c = patternClip(160, 96, 3);
    for (int shape : {7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 22, 23, 24}) {
        ParamValues p = with(defaultValues(m), {{"solid_shape", shape}, {"solid_size", 0.4}, {"encoding", 0}, {"scene_defocus", 0.0},
                                                {"solid_rot_x", 0.0}, {"solid_rot_y", 0.0}, {"solid_strength", 0.8}});
        p.v["solid_position"] = {0.5, 0.5, 0};
        const RenderResult r = renderAt(m, c, 1, p);
        CHECK(regionDiff(r.out.view, c.frames[1].view, 0, 0, 40, 96) < 1e-5);
        CHECK(regionDiff(r.out.view, c.frames[1].view, 120, 0, 160, 96) < 1e-5);
        CHECK(regionDiff(r.out.view, c.frames[1].view, 62, 30, 98, 66) > 0.02);
    }
}

TP_TEST(multi_optic_clear_centre_and_smear_plates_keep_their_clear_part) {
    const ModuleDef& m = find("multi_optic");
    Clip c = patternClip(160, 160, 3);
    for (int shape : {22, 23, 24}) {
        ParamValues p = with(defaultValues(m), {{"solid_shape", shape}, {"solid_size", 0.9}, {"encoding", 0}, {"scene_defocus", 0.0},
                                                {"solid_rot_x", 0.0}, {"solid_rot_y", 0.0}, {"solid_strength", 0.8}, {"f_stop", 22.0},
                                                {"solid_reflection", 0.0}});
        p.v["solid_position"] = {0.5, 0.5, 0};
        const RenderResult r = renderAt(m, c, 1, p);
        const double clear = shape == 24 ? regionDiff(r.out.view, c.frames[1].view, 50, 20, 110, 70)
                                         : regionDiff(r.out.view, c.frames[1].view, 74, 74, 86, 86);
        const double bent = shape == 24 ? regionDiff(r.out.view, c.frames[1].view, 50, 90, 110, 140)
                                        : regionDiff(r.out.view, c.frames[1].view, 20, 74, 40, 86);
        CHECK(bent > 3.0 * clear);
        CHECK(bent > 0.02);
    }
}

TP_TEST(multi_optic_streak_filter_draws_lines_from_highlights_only) {
    const ModuleDef& m = find("multi_optic");
    Clip c = makeClip(160, 96, 3, [](int x, int y, int) {
        const float v = x == 80 && y == 48 ? 20.f : 0.1f;
        return RGBA{v, v, v, 1.f};
    });
    ParamValues p = with(defaultValues(m), {{"e1_type", 0}, {"e2_type", 0}, {"e3_type", 0}, {"encoding", 0},
                                            {"streak_amount", 1.0}, {"streak_length", 60.0}, {"streak_angle", 0.0}});
    const RenderResult off = renderAt(m, c, 1, with(p, {{"streak_amount", 0.0}}));
    CHECK(maxDiff(off.out.view, c.frames[1].view) == 0.0);
    const RenderResult on = renderAt(m, c, 1, p);
    const RGBA along = on.out.view.load(110, 48), across = on.out.view.load(80, 78);
    CHECK(along.b > 0.1f + 1e-3f);
    CHECK(along.b > along.r);
    CHECK(std::fabs(across.b - 0.1f) < 1e-4f);
}

TP_TEST(multi_optic_glass_motion_is_deterministic_continuous_and_moves) {
    for (int mode = 1; mode <= 9; ++mode) {
        double travel = 0, jump = 0;
        GlassPose prev = glassMotion(mode, 0.0, 0.5, 0.1, 1.0);
        for (int i = 1; i <= 2400; ++i) {
            const double t = i / 600.0;
            const GlassPose a = glassMotion(mode, t, 0.5, 0.1, 1.0), b = glassMotion(mode, t, 0.5, 0.1, 1.0);
            CHECK(a.x == b.x && a.rx == b.rx && a.depth == b.depth);
            jump = std::max({jump, std::fabs(a.x - prev.x) / 0.1, std::fabs(a.y - prev.y) / 0.1, std::fabs(a.rx - prev.rx) / 10.0,
                             std::fabs(a.ry - prev.ry) / 10.0, std::fabs(a.rz - prev.rz) / 10.0});
            travel = std::max({travel, std::fabs(a.x), std::fabs(a.rx), std::fabs(a.ry)});
            prev = a;
        }
        CHECK(travel > 0.02);
        CHECK(jump < 0.08);
    }
    const GlassPose eased = glassMotionEased(4, 0.0, 0.5, 0.1, 1.0, 2.0);
    CHECK(eased.x == 0 && eased.ry == 0);
}

TP_TEST(multi_optic_glass_motion_blur_only_when_moving) {
    const ModuleDef& m = find("multi_optic");
    Clip c = patternClip(160, 96, 6);
    ParamValues still = with(defaultValues(m), {{"solid_shape", 2}, {"solid_size", 0.5}, {"encoding", 0}, {"motion_shutter", 0.0}});
    ParamValues stillBlur = with(still, {{"motion_shutter", 1.0}});
    CHECK(maxDiff(renderAt(m, c, 2, still).out.view, renderAt(m, c, 2, stillBlur).out.view) < 1e-6);
    ParamValues moving = with(still, {{"motion_mode", 6}, {"motion_amount", 0.2}, {"motion_rate", 2.0}});
    ParamValues movingBlur = with(moving, {{"motion_shutter", 1.0}});
    CHECK(regionDiff(renderAt(m, c, 2, moving).out.view, renderAt(m, c, 2, movingBlur).out.view, 0, 0, 160, 96) > 0.02);
    CHECK(maxDiff(renderAt(m, c, 2, movingBlur).out.view, renderAt(m, c, 2, movingBlur).out.view) < 1e-6);
}

static double regionDetail(const ImageView& v, int x0, int y0, int x1, int y1) {
    double d = 0;
    for (int y = y0; y < y1; ++y)
        for (int x = x0 + 1; x < x1; ++x) d += std::fabs(v.load(x, y)[1] - v.load(x - 1, y)[1]);
    return d / ((x1 - x0 - 1) * (y1 - y0));
}

TP_TEST(multi_optic_diopter_focus_sharpens_the_view_through_the_plate) {
    const ModuleDef& m = find("multi_optic");
    Clip c = makeClip(160, 96, 3, [](int x, int y, int) {
        const float v = ((x / 4 + y / 4) & 1) ? 0.8f : 0.2f;
        return RGBA{v, v, v, 1.f};
    });
    double detail[2];
    for (int k = 0; k < 2; ++k) {
        ParamValues p = with(defaultValues(m), {{"solid_shape", 10}, {"solid_size", 0.6}, {"encoding", 0}, {"f_stop", 1.4},
                                                {"solid_rot_x", 0.0}, {"solid_rot_y", 0.0}, {"solid_strength", 0.2}, {"solid_refocus", k ? 1.0 : 0.0}});
        p.v["solid_position"] = {0.5, 0.5, 0};
        detail[k] = regionDetail(renderAt(m, c, 1, p).out.view, 65, 35, 95, 61);
    }
    CHECK(detail[1] > 2.0 * detail[0]);
}

TP_TEST(multi_optic_glass_material_and_tint) {
    const ModuleDef& m = find("multi_optic");
    Clip grey = constantClip({0.5f, 0.5f, 0.5f, 1.f}, 96, 64, 3);
    ParamValues p = with(defaultValues(m), {{"solid_shape", 2}, {"solid_size", 0.6}, {"encoding", 0}, {"solid_light", 0.0},
                                            {"solid_reflection", 0.0}, {"solid_density", 2.0}});
    p.v["solid_tint"] = {1.0, 0.2, 0.2};
    const RGBA mid = renderAt(m, grey, 1, p).out.view.load(48, 32);
    CHECK(mid.r > mid.g + 0.05f);
    Clip c = patternClip(160, 96, 3);
    ParamValues crown = with(defaultValues(m), {{"solid_shape", 1}, {"solid_size", 0.5}, {"encoding", 0}, {"solid_material", 1}, {"scene_defocus", 0.0}});
    ParamValues diamond = crown;
    diamond.v["solid_material"] = {5, 0, 0};
    CHECK(regionDiff(renderAt(m, c, 1, crown).out.view, renderAt(m, c, 1, diamond).out.view, 60, 30, 100, 66) > 0.05);
}

TP_TEST(multi_optic_light_returned_through_the_crown_sees_the_environment) {
    const ModuleDef& m = find("multi_optic");
    Clip grey = constantClip({0.5f, 0.5f, 0.5f, 1.f}, 96, 64, 3);
    ParamValues lit = with(defaultValues(m), {{"solid_shape", 18}, {"solid_size", 0.7}, {"encoding", 0}, {"solid_material", 5},
                                              {"solid_rot_x", -90.0}, {"solid_rot_y", 0.0}, {"solid_light", 0.0},
                                              {"solid_reflection", 0.0}, {"solid_environment", 1.0}, {"scene_defocus", 0.0}});
    ParamValues dark = lit;
    dark.v["solid_environment"] = {0.0, 0, 0};
    const RGBA a = renderAt(m, grey, 1, lit).out.view.load(48, 32), b = renderAt(m, grey, 1, dark).out.view.load(48, 32);
    CHECK(a.g > 0.3f);
    CHECK(b.g < 0.5f * a.g);
}

TP_TEST(multi_optic_glass_sphere_spectrum_keeps_grey_neutral) {
    const ModuleDef& m = find("multi_optic");
    Clip c = constantClip({0.4f, 0.4f, 0.4f, 1.f}, 96, 64, 3);
    ParamValues p = with(defaultValues(m), {{"solid_shape", 1}, {"solid_size", 0.5}, {"quality", 1}, {"encoding", 0},
                                            {"dispersion", 0.0005}, {"solid_light", 0.0}, {"solid_reflection", 0.0}});
    const RenderResult r = renderAt(m, c, 1, p);
    double tint = 0;
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 96; ++x) {
            const auto v = r.out.view.load(x, y);
            tint = std::max(tint, static_cast<double>(std::max(std::fabs(v[0] - v[1]), std::fabs(v[1] - v[2]))));
        }
    CHECK(tint < 1e-3);
    CHECK(regionDiff(r.out.view, c.frames[1].view, 0, 0, 96, 64) > 0.002);
}

TP_TEST(multi_optic_glass_solid_gpu_matches_cpu) {
    if (!gpu::available()) { std::printf("  (no CUDA device: glass solid parity skipped)\n"); return; }
    const ModuleDef& m = find("multi_optic");
    Clip c = patternClip(160, 96, 3);
    for (int shape : {5, 8, 9, 18}) {
        ParamValues p = with(defaultValues(m), {{"solid_shape", shape}, {"solid_size", 0.45}, {"dispersion", 0.03}, {"f_stop", 2.0}, {"focus_shift", 0.4}, {"encoding", 0},
                                                {"motion_mode", shape == 9 ? 6 : 0}, {"motion_amount", 0.15}, {"motion_shutter", 1.0}});
        p.v["solid_position"] = {0.45, 0.55, 0};
        gpu::setEnabled(false);
        const RenderResult cpu = renderAt(m, c, 1, p);
        gpu::setEnabled(true);
        const RenderResult cuda = renderAt(m, c, 1, p);
        double sum = 0;
        for (int y = 0; y < 96; ++y)
            for (int x = 0; x < 160; ++x)
                for (int ch = 0; ch < 3; ++ch) sum += std::fabs(cpu.out.view.load(x, y)[ch] - cuda.out.view.load(x, y)[ch]);
        const double maxd = maxDiff(cpu.out.view, cuda.out.view), mean = sum / (160.0 * 96 * 3);
        std::printf("  glass solid %d CPU vs CUDA: max %.5f, mean %.6f\n", shape, maxd, mean);
        CHECK(mean < 1e-3);
        CHECK(maxd < 0.05);
    }
}
