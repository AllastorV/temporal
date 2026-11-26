#pragma once
#include "module.h"

namespace tp {
inline bool controlEnabled(const std::string& module, const std::string& id, const ParamValues& p) {
    const int mode = p.v.count("mode") ? p.i("mode") : 0;
    if (id == "history" || id == "edge") {
        if (module == "cutwright_art") return p.i("preset") <= 1;
        if (module == "geometric") return p.i("preset") == 0 || p.i("preset") == 1 || p.i("preset") == 3;
    }
    if (module == "stutter") {
        if (id == "hold") return mode != 1;
        if (id == "loop_length") return mode == 1;
        if (id == "repeat") return mode != 0;
        if (id == "bpm" || id == "chance" || id == "seed") return mode == 2;
    }
    if (module == "pixel") {
        if (id == "grid") return mode != 2 && mode != 5;
        if (id == "color_depth" || id == "dither_amount") return mode == 4;
        if (id == "variance_threshold") return mode == 1;
        if (id == "sort_low" || id == "sort_high" || id == "direction" || id == "sort_order") return mode == 2;
        if (id == "sort_angle") return mode == 2 && p.i("direction") == 2;
        if (id == "ascii_color") return mode == 3;
        if (id == "melt_amount" || id == "persistence" || id == "seed") return mode == 5;
    }
    if (module == "camera_motion") {
        if (id == "end_point" || id == "end_frame") return p.b("use_path");
        if (id == "duration") return !p.b("use_path") && mode != 3;
        if (id == "start_frame") return mode != 3 || p.b("use_path");
        if (id == "distance" || id == "angle") return mode == 2;
        if (id == "amplitude" || id == "frequency" || id == "seed") return mode == 3;
        if (id == "zoom") return mode != 2 && mode != 3;
        if (id == "easing") return p.b("use_path") || (mode != 1 && mode != 3 && mode != 5 && mode != 7 && mode != 8);
        if (id == "rebound") return mode == 0 || mode == 2 || mode == 4 || mode == 7 || mode >= 9;
        if (id == "motion_samples") return p.d("shutter") > 0;
    }
    if (module == "echo") {
        if (id == "scale" || id == "rotation") return mode != 0;
        if (id == "decay") return mode != 3;
        if (id == "feedback") return mode == 3;
    }
    if (module == "procedural_motion") {
        if (id == "hinge") return mode == 2;
        if (id == "smoothing") return mode == 4;
        if (id == "seed") return mode >= 3 && mode <= 5;
        if (id == "start_frame" || id == "damping") return mode == 6;
        if (id == "amplitude" || id == "axis_x" || id == "axis_y" || id == "axis_z" || id == "zoom_amount") return mode != 2;
        if (id == "axis_rx" || id == "axis_ry") return mode != 2 && mode != 3 && mode != 4;
        if (id == "axis_rz") return mode != 2 && mode != 3;
        if (id == "motion_samples") return p.d("shutter") > 0;
    }
    if (module == "prism") {
        if (id == "red_offset" || id == "green_offset" || id == "blue_offset" || id == "decay" || id == "edge") return mode != 1;
        if (id == "spread" || id == "center") return mode != 0;
    }
    if (module == "slow_shutter") {
        if (id == "decay") return mode == 1;
    }
    if (module == "time_shift") {
        if (id == "angle") return mode == 0;
        if (id == "center") return mode == 2;
    }
    if (module == "diffusion_lab") {
        if (id == "threshold") return mode == 1 || mode == 2 || mode == 4 || mode == 6 || mode == 8;
        if (id == "shadow_protection") return mode == 1 || mode == 7;
        if (id == "streak_angle" || id == "streak_length") return mode == 2 || mode == 4 || mode == 8 || mode == 9;
    }
    if (module == "datamosh") {
        if (id == "reference_frame" || id == "reset_interval" || id == "persistence" || id == "flow_scale" || id == "feedback" || id == "quality" || id == "edge") return mode != 1;
    }
    if (module == "motion_tear") {
        if (id == "angle" || id == "speed") return mode == 1;
        if (id == "quality") return mode == 0;
    }
    if (module == "double_exposure") {
        if (id == "key_threshold" || id == "feather" || id == "invert_key" || id == "key_blur") return p.i("key_source") != 0;
    }
    if (module == "refraction_lab") {
        if (id == "facets") return mode == 3;
        if (id == "f_stop") return p.d("focus_shift") != 0;
    }
    if (module == "time_displacement") {
        if (id == "center" || id == "radial") return p.i("map_source") == 1;
        if (id == "gradient_angle") return p.i("map_source") == 1 && !p.b("radial");
        if (id == "noise_scale" || id == "noise_evolution" || id == "seed") return p.i("map_source") == 2;
    }
    if (module == "blob_tracking") {
        if (id == "threshold") return p.i("source") == 0;
        if (id == "dark_threshold") return p.i("source") == 1;
        if (id == "motion_threshold") return p.i("source") == 2;
        if (id == "negative_boxes") return p.i("style") == 0 || p.i("style") == 1 || p.i("style") == 6;
        if (id == "link_curve") return p.i("style") == 6 && p.i("connections") > 0;
        if (id == "connections") return p.i("style") == 6;
        if (id == "link_distance") return p.i("style") == 6 && p.i("connections") > 0;
        if (id == "color" || id == "dim" || id == "line_width") return p.i("style") != 5;
    }
    if (module == "print_lab") {
        if (id == "ink_a") return p.i("look") == 8 || p.i("look") == 9;
        if (id == "ink_b") return p.i("look") == 9;
        if (id == "threshold") return p.i("look") == 6 || p.i("look") == 7;
        if (id == "paper" || id == "paper_colour" || id == "print" || id == "contrast") return p.i("look") < 10;
        if (id == "focus") return p.d("depth_of_field") > 0;
        if (id == "zoom" || id == "position") return !p.b("show_whole_picture");
    }
    if (module == "multi_optic") {
        const int solid = p.i("solid_shape");
        if (id == "dispersion") return p.i("quality") == 1 && (solid == 0 || p.i("solid_material") == 0);
        if (id == "facets") {
            const auto faceted = [](int shape) { return shape == 4 || shape == 8; };
            return (solid == 8 || (solid >= 11 && solid <= 15)) || faceted(p.i("e1_type")) || faceted(p.i("e2_type")) || faceted(p.i("e3_type"));
        }
        if (id == "solid_strength" || id == "solid_thickness" || id == "solid_refocus") return (solid >= 7 && solid <= 15) || solid >= 21;
        if (id == "solid_bevel") return (solid >= 2 && solid <= 6) || (solid >= 16 && solid <= 20);
        if (id == "streak_angle" || id == "streak_length" || id == "streak_threshold" || id == "streak_tint") return p.d("streak_amount") > 0;
        if (id == "solid_ior") return solid > 0 && p.i("solid_material") == 0;
        if (id == "solid_tint") return solid > 0 && p.d("solid_density") > 0;
        if (id.rfind("solid_", 0) == 0 && id != "solid_shape") return solid > 0;
        if (id == "motion_amount" || id == "motion_rate") return p.i("motion_mode") != 0;
        if (id == "motion_turn" || id == "motion_depth" || id == "motion_offset" || id == "motion_ease" || id == "motion_shutter")
            return p.i("motion_mode") != 0 && solid > 0;
        if (id == "reflection_roughness") return p.d("reflection") > 0;
        if (id.size() > 3 && id[0] == 'e' && id[1] >= '1' && id[1] <= '3' && id[2] == '_') {
            const std::string control = id.substr(3);
            if (control == "type") return true;
            const int shape = p.i(id.substr(0, 3) + "type");
            if (control == "ring_width") return shape == 6;
            if (control == "repeat_offset" || control == "repeat_scale" || control == "facet_feather" ||
                control == "facet_defocus") return shape == 8;
            return shape != 0;
        }
    }
    if (module == "lens_character" && id == "lens_focal") return p.i("lens_model") == 2;
    if (module == "halation") {
        if (id == "duration") return p.d("decay") > 0;
    }
    return true;
}
}
