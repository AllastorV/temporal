#include "modules.h"
#include "multi_optic_kernel.h"
#include "glass_motion.h"
#include "scatter_ops.h"
#include "../gpu/gpu.h"

namespace tp {

namespace {

const int kMoOrders[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

ParamValues studioLook(const ModuleDef& module, int look) {
    if (look < 1 || look > 8) return {};
    ParamValues p = defaultValues(module);
    for (const char* id : {"studio_look", "encoding", "mix", "quality"}) p.v.erase(id);
    if (look >= 4) {
        p.set("e1_type", 0);
        p.set("e2_type", 0);
        p.set("e3_type", 0);
        if (look == 8) {
            p.set("streak_amount", 1.0);
            p.set("streak_threshold", 0.9);
            return p;
        }
        static const int shapes[4] = {21, 22, 23, 24};
        p.set("solid_shape", shapes[look - 4]);
        p.set("solid_size", 2.2);
        p.set("solid_distance", 0.06);
        p.set("solid_rot_x", 0.0);
        p.set("solid_rot_y", 0.0);
        p.set("solid_strength", 0.3);
        p.set("solid_reflection", 0.0);
        p.set("solid_light", 0.0);
        p.set("solid_bevel", 0.0);
        p.set("solid_environment", 0.0);
        p.set("dispersion", 0.005);
        p.set("solid_refocus", 0.0);
        p.set("focus_shift", 1.0);
        p.set("f_stop", 2.8);
        return p;
    }
    p.set("e2_type", 0);
    p.set("off_frame", 1);
    p.set("motion_amount", 0.045);
    p.set("motion_rate", 0.32);
    if (look == 1) {
        p.set("e1_type", 6);  // Halo
        p.set("e1_position", 0.48, 0.50);
        p.set("e1_size", 0.78);
        p.set("e1_aspect", 0.76);
        p.set("e1_ring_width", 0.34);
        p.set("e1_curvature", 0.92);
        p.set("e1_distance", 120);
        p.set("e1_depth", 0.18);
        p.set("e1_tilt_y", -8);
        p.set("dispersion", 0.044);
        p.set("edge_softness", 0.040);
        p.set("focus_shift", 0.05);
        p.set("f_stop", 2.8);
        p.set("reflection", 0.46);
        p.set("highlight_boost", 0.22);
        p.set("motion_mode", 2);
    } else if (look == 2) {
        p.set("e1_type", 7);  // Wedge
        p.set("e1_position", 0.48, 0.54);
        p.set("e1_size", 0.83);
        p.set("e1_rotation", -24);
        p.set("e1_tilt_x", 12);
        p.set("e1_curvature", 0.54);
        p.set("e1_distance", 115);
        p.set("e1_depth", 0.42);
        p.set("dispersion", 0.020);
        p.set("edge_softness", 0.16);
        p.set("focus_shift", -0.17);
        p.set("f_stop", 2.0);
        p.set("reflection", 0.55);
        p.set("reflection_roughness", 7);
        p.set("motion_mode", 3);
        p.set("motion_amount", 0.23);
    } else {
        p.set("e1_type", 8);  // Prism Array
        p.set("e1_position", 0.50, 0.52);
        p.set("e1_size", 0.91);
        p.set("e1_aspect", 0.60);
        p.set("e1_repeat_offset", -0.24);
        p.set("e1_repeat_scale", 1.20);
        p.set("e1_curvature", -0.36);
        p.set("e1_distance", 93);
        p.set("e1_rotation", 5);
        p.set("e2_type", 5);  // Convex
        p.set("e2_position", 0.50, 0.49);
        p.set("e2_size", 0.48);
        p.set("e2_curvature", 0.33);
        p.set("e2_distance", 55);
        p.set("dispersion", 0.008);
        p.set("facets", 5);
        p.set("edge_softness", 0.078);
        p.set("motion_mode", 1);
        p.set("motion_amount", 0.042);
    }
    return p;
}

void addStreak(const RenderContext& c) {
    const ParamValues& p = c.p();
    const float k = clampf(p.f("streak_amount"), 0.f, 2.f);
    const ImageView src = c.source();
    if (k <= 0.f || p.i("preview") != 0 || !src.valid()) return;
    requireHost(c);
    Rect R = intersect(c.frame, src.bounds);
    if (R.empty()) R = src.bounds;
    const Encoding enc = c.encoding;
    const float thr = std::max(0.f, p.f("streak_threshold"));
    OwnedImage sel(R);
    parallelRows(R.y1, R.y2, [&](int y) {
        for (int x = R.x1; x < R.x2; ++x) sel.view.store(x, y, dlHighlight(decode(src.load(x, y), enc), thr));
    });
    const float diag = std::sqrt(static_cast<float>(c.frame.w()) * c.frame.w() + static_cast<float>(c.frame.h()) * c.frame.h());
    const float len = std::min(std::max(0.f, c.px(p.f("streak_length"))), diag);
    const float angle = -p.f("streak_angle") * kPi / 180.f;
    const DlLevel core = dlDirBlur(sel.view, angle, 0.1f * len, 96, 3.f), tail = dlDirBlur(sel.view, angle, len, 96, 6.f);
    const float coreGain = std::max(1.f, 0.125f * len / 24.f), tailGain = std::max(1.f, len / 24.f);
    const auto tv = p.vec("streak_tint");
    const float tint[3] = {static_cast<float>(tv[0]), static_cast<float>(tv[1]), static_cast<float>(tv[2])};
    parallelRows(c.window.y1, c.window.y2, [&](int y) {
        for (int x = c.window.x1; x < c.window.x2; ++x) {
            RGBA lin = decode(c.dst.load(x, y), enc);
            const RGBA a = core.sample(R, x, y, SampleEdge::Black), b = tail.sample(R, x, y, SampleEdge::Black);
            lin.r += k * (coreGain * a.r + tailGain * tint[0] * b.r);
            lin.g += k * (coreGain * a.g + tailGain * tint[1] * b.g);
            lin.b += k * (coreGain * a.b + tailGain * tint[2] * b.b);
            c.dst.store(x, y, encode(lin, enc));
        }
    });
}

}

ModuleDef makeMultiOptic() {
    ModuleDef m;
    m.id = "multi_optic";
    m.cuda = CudaMode::Device;
    m.name = "Multi-Optic";
    m.family = "Optics";
    m.description = "Stack three virtual glass elements with independent shapes, tilt and depth. Adjust sampled aperture defocus, color dispersion, source reflections and procedural glass movement. Start from a look or build the stack directly.";
    m.temporal = false;
    m.usesEncoding = true;
    std::vector<ParamSpec> ps = {
        Choice("studio_look", "Load Look", {"Choose a Look", "Spectral Halo", "Soft Prism Sweep", "Facet Echo",
                                            "Triad Filter", "Clear-Centre Quad Filter", "Clear-Centre Penta Filter", "Motion Smear Filter", "Blue Streak Filter"}, 0,
               "Loads an editable starting point into the current glass controls. The Filter looks screw a faceted glass onto the lens: it sits a few centimetres away, fills the frame and is defocused, while the scene stays in focus."),
        Choice("preview", "Element Mask", {"Off", "Element 1", "Element 2", "Element 3"}, 0,
               "Inspect the actual footprint as a matte. Select Off for the optical result."),
        Choice("element_order", "Element Order", {"1 > 2 > 3", "1 > 3 > 2", "2 > 1 > 3", "2 > 3 > 1", "3 > 1 > 2", "3 > 2 > 1"}, 0, "Order in which the ray passes the elements."),
        Dbl("dispersion", "Dispersion", 0.012, 0.0, 0.05, 0.0, 0.05, Unit::None,
            "Controls wavelength separation. Prism Array confines lateral color to off-axis facet rims."),
        Choice("quality", "Quality", {"Draft", "Final"}, 1,
               "Draft disables dispersion. Final separates wavelength samples, with a shared geometric ray for Prism Array stacks."),
        Dbl("edge_softness", "Edge Softness", 0.01, 0.0, 0.2, 0.0, 0.2, Unit::None, "Anti-aliased edge width of every element footprint."),
        Int("facets", "Facets", 5, 2, 16, 2, 16, Unit::None, "Strip count for Replicator and Prism Array elements, the strip, ring, cell, sector or wave count of the diopter plates."),
        Dbl("f_stop", "F-Stop", 5.6, 1.0, 22.0, 1.0, 22.0, Unit::None, "Open the sampled aperture to soften out-of-focus glass.", "Camera"),
        Dbl("focus_shift", "Focus Plane", 0.0, -1.0, 1.0, -1.0, 1.0, Unit::None, "Focus depth. Glass at this depth is sharp, with defocus increasing away from it.", "Camera"),
        Choice("off_frame", "Outside Frame", {"Clamp", "Mirror", "Transparent"}, 0, "How rays sample outside the source image.", "Camera"),
        Choice("motion_mode", "Motion", {"Manual", "Handheld", "Orbit", "Sweep", "Drift", "Swing", "Figure Eight", "Fold", "Handheld Light", "Handheld Strong"}, 0,
               "Procedural movement added to the keyframed glass. The glass solid and diopter plates use every mode; the element stack treats the later modes as Orbit.", "Glass Motion"),
        Dbl("motion_amount", "Travel", 0.0, 0.0, 0.5, 0.0, 0.3, Unit::Normalized, "Motion amplitude as a fraction of image width.", "Glass Motion"),
        Dbl("motion_rate", "Cycles per Second", 0.45, 0.05, 6.0, 0.05, 2.0, Unit::None, "Playback rate of procedural glass motion.", "Glass Motion"),
        Dbl("motion_turn", "Rotation Gain", 1.0, 0.0, 3.0, 0.0, 2.0, Unit::None, "How much the glass solid tilts and turns as it moves. At 1 the turn is a few degrees, as from a hand or wrist.", "Glass Motion"),
        Dbl("motion_depth", "Depth Travel", 0.3, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "How far the glass solid moves toward and away from the lens, in and out of focus.", "Glass Motion"),
        Dbl("motion_offset", "Time Offset", 0.0, -60.0, 60.0, -10.0, 10.0, Unit::Seconds, "Shifts the motion in time, so two copies can move differently.", "Glass Motion"),
        Dbl("motion_ease", "Ease In", 0.0, 0.0, 10.0, 0.0, 3.0, Unit::Seconds, "Fades the motion in from the start of the clip.", "Glass Motion"),
        Dbl("motion_shutter", "Motion Blur", 0.5, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Shutter as a fraction of a frame (0.5 is a 180 degree shutter). Blurs the moving glass solid.", "Glass Motion"),
        Dbl("reflection", "Reflection Strength", 0.0, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Adds a displaced reflection of bright source detail inside the glass.", "Surface Light"),
        Dbl("reflection_roughness", "Reflection Roughness", 4.0, 0.0, 48.0, 0.0, 24.0, Unit::Pixels, "Softens the reflected detail.", "Surface Light"),
        Dbl("highlight_boost", "Highlight Lift", 0.0, 0.0, 2.0, 0.0, 1.0, Unit::None, "Boosts linear-light highlights within the optical footprint.", "Surface Light"),
        Dbl("focal_length", "Focal Length (mm)", 35.0, 12.0, 200.0, 18.0, 135.0, Unit::None, "Virtual lens focal length on a 36 mm wide sensor. Used by the glass solid.", "Camera"),
        Dbl("scene_distance", "Scene Distance (m)", 4.0, 0.3, 100.0, 0.5, 20.0, Unit::None, "Distance of the filmed scene. The source stays sharp here unless Focus Plane moves the focus.", "Camera"),
        Choice("solid_shape", "Glass Solid", {"Off", "Sphere", "Cube", "Tetrahedron", "Octahedron", "Dodecahedron", "Icosahedron",
                                              "Stretch Diopter", "Facet Repeater", "Halo Ring", "Wedge Prism",
                                              "Ripple Lens", "Honeycomb Array", "Kaleido Prism", "Spiral Twist", "Liquid Wobble",
                                              "Triangular Prism", "Faceted Gem", "Brilliant Diamond", "Quartz Crystal", "Glass Rod",
                                              "Triad Prism", "Clear-Centre Quad", "Clear-Centre Penta", "Motion Smear"}, 0,
               "Ray-traces a 3D glass solid held in front of the lens: refraction through both faces, internal reflection and Fresnel edges. The footage outside the glass is untouched. Focus Plane 0 keeps the glass sharp; positive values move focus toward the scene. When on, it replaces the element stack. Triad Prism splits the view into three facets that meet in the middle. Clear-Centre Quad and Penta keep a flat clear centre and ring it with four or five tilted facets, each repeating the subject. Motion Smear leaves one half clear and fills the other with fine prism strips that drag a ghost and a fading trail to one side.", "Glass Solid"),
        Point("solid_position", "Position", 0.5, 0.5, "Where the centre of the solid appears in the frame.", "Glass Solid"),
        Dbl("solid_size", "Size", 0.55, 0.05, 3.0, 0.1, 1.5, Unit::None, "Apparent radius as a fraction of the frame height.", "Glass Solid"),
        Dbl("solid_distance", "Distance (m)", 0.35, 0.03, 5.0, 0.05, 2.0, Unit::None, "Distance of the solid from the lens.", "Glass Solid"),
        Dbl("solid_rot_x", "Rotation X", 20.0, -360.0, 360.0, -180.0, 180.0, Unit::Degrees, "Rotation about the horizontal axis.", "Glass Solid"),
        Dbl("solid_rot_y", "Rotation Y", 30.0, -360.0, 360.0, -180.0, 180.0, Unit::Degrees, "Rotation about the vertical axis.", "Glass Solid"),
        Dbl("solid_rot_z", "Rotation Z", 0.0, -360.0, 360.0, -180.0, 180.0, Unit::Degrees, "Rotation about the lens axis.", "Glass Solid"),
        Dbl("solid_ior", "IOR", 1.52, 1.0, 2.5, 1.0, 2.0, Unit::None, "Refractive index of the solid. Dispersion spreads it per wavelength in Final quality.", "Glass Solid"),
        Dbl("solid_reflection", "Surface Reflection", 1.0, 0.0, 2.0, 0.0, 2.0, Unit::None, "Scales Fresnel reflection at the entry face. 1 is uncoated glass.", "Glass Solid"),
        Dbl("solid_strength", "Diopter Strength", 0.5, 0.0, 2.0, 0.0, 1.5, Unit::None, "Curvature, ripple depth or facet angle of the diopter plates. Facets sets the strip, ring, cell, sector or wave count.", "Glass Solid"),
        Choice("solid_material", "Material", {"Custom", "Crown Glass", "Dense Flint", "Fused Silica", "Sapphire", "Diamond", "Acrylic", "Water"}, 0,
               "Optical material of the solid. Presets set the refractive index and the dispersion from the material's Abbe number; Custom uses IOR and Dispersion.", "Glass Material"),
        Color("solid_tint", "Tint", 0.39, 0.71, 0.60, "Colour the glass lets through when Density is above zero.", "Glass Material"),
        Dbl("solid_density", "Density", 0.0, 0.0, 4.0, 0.0, 2.0, Unit::None, "Absorption through the body of the glass. Thick paths take on the tint.", "Glass Material"),
        Dbl("solid_frost", "Frost", 0.0, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Surface roughness that scatters refraction like frosted or sandblasted glass.", "Glass Material"),
        Dbl("solid_refocus", "Diopter Focus", 1.0, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Close-up power of the diopter plates: at 1 the scene seen through a plate is brought back into focus while the footage around it stays defocused.", "Glass Solid"),
        Dbl("solid_thickness", "Plate Thickness", 0.12, 0.02, 0.6, 0.02, 0.4, Unit::None, "Thickness of the diopter plates relative to their half width.", "Glass Solid"),
        Dbl("solid_bevel", "Bevel", 0.035, 0.0, 0.25, 0.0, 0.12, Unit::None, "Rounds the edges of a polyhedron so they catch light and bend it more strongly. Faceted Gem and Brilliant Diamond use a tenth of it so their small facets stay crisp.", "Glass Solid"),
        Dbl("solid_light_angle", "Light Angle", 0.0, -180.0, 180.0, -180.0, 180.0, Unit::Degrees, "Turns the two virtual lights around the lens axis, moving the highlights across the glass.", "Glass Solid"),
        Dbl("solid_light", "Studio Light", 0.6, 0.0, 4.0, 0.0, 2.0, Unit::None, "Two soft virtual lights reflected in the glass faces and bevels. Faceted Gem and Brilliant Diamond sit in a jeweller's studio instead: a dark room, a ring light around the lens and narrow strip lights.", "Glass Solid"),
        Dbl("solid_environment", "Reflection Environment", 0.25, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Brightness of the surroundings reflected in the glass faces. The footage stands in for them; low values keep blacks deep.", "Glass Solid"),
        Dbl("scene_defocus", "Scene Defocus", 1.0, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "How much the lens defocuses the filmed scene behind the glass solid. 1 follows the lens, 0 keeps the footage sharp.", "Camera"),
        Dbl("streak_amount", "Streak", 0.0, 0.0, 2.0, 0.0, 1.0, Unit::None, "Etched-line streak filter on the lens: bright points throw long thin streaks with a white core and a coloured tail. 0 turns it off.", "Streak Filter"),
        Dbl("streak_angle", "Streak Angle", 0.0, -360.0, 360.0, -180.0, 180.0, Unit::Degrees, "Direction of the streaks, counter-clockwise from horizontal, as when the filter ring is turned.", "Streak Filter"),
        Dbl("streak_length", "Streak Length", 240.0, 0.0, 2000.0, 0.0, 1000.0, Unit::Pixels, "Length over which the coloured tail fades out, in full-resolution pixels. The white core is a tenth of it.", "Streak Filter"),
        Dbl("streak_threshold", "Streak Threshold", 0.9, 0.0, 8.0, 0.0, 2.0, Unit::None, "Linear luma above which light streaks, with a soft knee of half the threshold.", "Streak Filter"),
        Color("streak_tint", "Tail Colour", 0.12, 0.42, 1.0, "Colour of the streak tail; the default is the blue of a blue-coated filter.", "Streak Filter"),
    };
    const char* types[3] = {"Prism", "Cylindrical", "Off"};
    const double posx[3] = {0.35, 0.65, 0.5}, rotd[3] = {0.0, 90.0, 0.0};
    for (int e = 0; e < 3; ++e) {
        std::string g = "Element " + std::to_string(e + 1), k = "e" + std::to_string(e + 1) + "_";
        int defType = std::string(types[e]) == "Prism" ? 2 : (std::string(types[e]) == "Cylindrical" ? 3 : 0);
        ps.push_back(Choice(k + "type", "Type", {"Off", "Diopter", "Prism", "Cylindrical", "Replicator", "Convex", "Halo", "Wedge", "Prism Array"}, defType, "Shape of this element.", g));
        ps.push_back(Dbl(k + "ior", "IOR", 1.5, 1.0, 2.5, 1.0, 2.5, Unit::None, "Refractive index of this element.", g));
        ps.push_back(Dbl(k + "curvature", "Curvature", 0.6, -4.0, 4.0, -4.0, 4.0, Unit::None, "Surface strength; negative inverts.", g));
        ps.push_back(Dbl(k + "distance", "Thickness", 120.0, 0.0, 2000.0, 0.0, 600.0, Unit::Pixels, "Glass thickness in full-resolution pixels; scales the bending.", g));
        ps.push_back(Point(k + "position", "Position", posx[e], 0.5, "Centre of the element (normalized frame coordinates).", g));
        ps.push_back(Dbl(k + "size", "Size", 0.4, 0.05, 2.0, 0.05, 2.0, Unit::None, "Footprint width as a fraction of the frame width.", g));
        ps.push_back(Dbl(k + "rotation", "Rotation", rotd[e], -360.0, 360.0, -180.0, 180.0, Unit::Degrees, "Rotation of the element.", g));
        ps.push_back(Dbl(k + "aspect", "Aspect", 1.0, 0.15, 3.0, 0.3, 2.0, Unit::None, "Height of the footprint relative to its width.", g));
        ps.push_back(Dbl(k + "tilt_x", "Tilt Horizontal", 0.0, -65.0, 65.0, -50.0, 50.0, Unit::Degrees, "Local horizontal tilt and refraction angle.", g));
        ps.push_back(Dbl(k + "tilt_y", "Tilt Vertical", 0.0, -65.0, 65.0, -50.0, 50.0, Unit::Degrees, "Local vertical tilt and refraction angle.", g));
        ps.push_back(Dbl(k + "depth", "Depth", 0.0, -0.75, 1.0, -0.75, 1.0, Unit::None, "Adjusts apparent footprint size, ray travel and focus distance.", g));
        ps.push_back(Dbl(k + "ring_width", "Ring Width", 0.30, 0.06, 0.9, 0.06, 0.9, Unit::Normalized, "Fraction of the outer radius occupied by the Halo glass.", g));
        ps.push_back(Dbl(k + "repeat_offset", "Array Source Offset", 0.0, -2.0, 2.0, -1.0, 1.0, Unit::None, "Shifts the source sampled by each Prism Array facet along the glass axis.", g));
        ps.push_back(Dbl(k + "repeat_scale", "Array Source Scale", 1.0, 0.25, 4.0, 0.5, 2.0, Unit::None, "Controls how much of the source fits inside each Prism Array facet.", g));
        ps.push_back(Dbl(k + "facet_feather", "Facet Feather", 18.0, 0.0, 60.0, 0.0, 36.0, Unit::Pixels,
                         "Half-width of the blend between Prism Array facets. At zero, a pixel-wide antialiasing filter remains active.", g));
        ps.push_back(Dbl(k + "facet_defocus", "Facet Defocus", 5.0, 0.0, 24.0, 0.0, 12.0, Unit::Pixels,
                         "Softens the image near Prism Array seams. The blur fades toward the clear center of each facet.", g));
    }
    m.params = ps;
    m.loadLook = studioLook;
    m.framesNeeded = [](const ParamValues&, double) { return FrameWindow{}; };
    const auto optics = [](const RenderContext& c) {
        const ParamValues& p = c.p();
        const ImageView src = c.source();
        if (!src.valid()) {
            requireHost(c);
            parallelRows(c.window.y1, c.window.y2, [&](int y) { for (int x = c.window.x1; x < c.window.x2; ++x) c.dst.store(x, y, RGBA{}); });
            return;
        }
        const Encoding enc = c.encoding;
        const float diag = std::sqrt(static_cast<float>(c.frame.w()) * c.frame.w() + static_cast<float>(c.frame.h()) * c.frame.h());
        const float B = clampf(p.f("dispersion"), 0.f, 0.05f);
        const bool fine = p.i("quality") == 1;
        const float soft = clampf(p.f("edge_softness"), 0.f, 0.2f);
        const int facets = std::clamp(p.i("facets"), 2, 16);
        OpticStack stack;
        stack.diagonal = diag;
        stack.channels = fine ? 3 : 1;
        stack.encoding = static_cast<int>(enc);
        stack.mask = std::clamp(p.i("preview"), 0, 3);
        stack.edge = std::clamp(p.i("off_frame"), 0, 2);
        stack.reflection = p.f("reflection");
        stack.roughness = c.px(p.f("reflection_roughness"));
        stack.highlight = p.f("highlight_boost");
        stack.frameCenterX = c.frame.x1 + 0.5f * c.frame.w();
        stack.frameCenterY = c.frame.y1 + 0.5f * c.frame.h();
        stack.invHalfWidth = 2.f / std::max(1, c.frame.w());
        stack.invHalfHeight = 2.f / std::max(1, c.frame.h());
        stack.arrayFringe = fine ? 0.06f * c.frame.w() * B : 0.f;
        const float focus = p.f("focus_shift");
        const float fstop = clampf(p.f("f_stop"), 1.f, 22.f);
        const int motion = p.i("motion_mode");
        const float travel = motion ? p.f("motion_amount") : 0.f;
        const double phase = 2.0 * kPi * p.d("motion_rate") * (c.time - c.first) / std::max(1.0, c.fps);
        const int shape = std::clamp(p.i("solid_shape"), 0, 24);
        if (shape > 0) {
            GlassSolid& s = stack.solid;
            s.shape = shape;
            if (stack.edge == 0) stack.edge = 1;
            static const float kMaterials[8][2] = {{0, 0}, {1.5168f, 64.17f}, {1.7847f, 25.68f}, {1.4585f, 67.82f},
                                                   {1.7682f, 72.2f}, {2.4175f, 55.3f}, {1.4917f, 57.2f}, {1.3330f, 55.7f}};
            const int material = std::clamp(p.i("solid_material"), 0, 7);
            const float nd = material ? kMaterials[material][0] : clampf(p.f("solid_ior"), 1.f, 2.5f);
            const float spread = material ? (nd - 1.f) / (kMaterials[material][1] * 1.9105f) : B;
            s.spectral = fine && spread > 0 ? 1 : 0;
            s.samples = fine ? 48 : 8;
            s.principalX = stack.frameCenterX;
            s.principalY = stack.frameCenterY;
            const float focalMm = clampf(p.f("focal_length"), 12.f, 200.f);
            s.focal = focalMm / 36.f * std::max(1, c.frame.w());
            const float glassZ = 1000.f * clampf(p.f("solid_distance"), 0.03f, 5.f);
            s.sceneZ = std::max(1000.f * clampf(p.f("scene_distance"), 0.3f, 100.f), 1.5f * glassZ);
            const float shift = clampf(focus, -1.f, 1.f);
            const float invGlass = 1.f / glassZ, invScene = 1.f / s.sceneZ;
            const float inv = shift >= 0 ? invGlass + shift * (invScene - invGlass) : invGlass * (1.f - 0.9f * shift);
            s.focusZ = 1.f / std::max(inv, 1e-6f);
            s.aperture = focalMm / (2.f * fstop);
            double baseX, baseY;
            c.pointToPixels(p.vec("solid_position"), baseX, baseY);
            s.radius = clampf(p.f("solid_size"), 0.05f, 3.f) * 0.5f * c.frame.h() * glassZ / s.focal;
            const double rate = p.d("motion_rate"), turn = std::clamp(p.d("motion_turn"), 0.0, 3.0);
            const double ease = std::max(0.0, p.d("motion_ease")), depthTravel = std::clamp(p.d("motion_depth"), 0.0, 1.0);
            const double seconds = (c.time - c.first) / std::max(1.0, c.fps) + p.d("motion_offset");
            const auto place = [&](double at, float* center, float* rot) {
                const GlassPose g = glassMotionEased(motion, at, rate, travel, turn, ease);
                const double px = baseX + c.frame.w() * g.x, py = baseY + c.frame.w() * g.y;
                const float z = glassZ * static_cast<float>(std::max(0.3, 1.0 + 0.5 * depthTravel * g.depth));
                center[0] = static_cast<float>((px - s.principalX) / s.focal * z);
                center[1] = static_cast<float>((py - s.principalY) / s.focal * z);
                center[2] = z;
                const float ax = static_cast<float>((p.d("solid_rot_x") + g.rx) * kPi / 180.0);
                const float ay = static_cast<float>((p.d("solid_rot_y") + g.ry) * kPi / 180.0);
                const float az = static_cast<float>((p.d("solid_rot_z") + g.rz) * kPi / 180.0);
                const float cxr = std::cos(ax), sxr = std::sin(ax), cyr = std::cos(ay), syr = std::sin(ay), czr = std::cos(az), szr = std::sin(az);
                const float R[9] = {czr * cyr, czr * syr * sxr - szr * cxr, czr * syr * cxr + szr * sxr,
                                    szr * cyr, szr * syr * sxr + czr * cxr, szr * syr * cxr - czr * sxr,
                                    -syr, cyr * sxr, cyr * cxr};
                std::copy(R, R + 9, rot);
            };
            const double shutter = std::clamp(p.d("motion_shutter"), 0.0, 1.0) / std::max(1.0, c.fps);
            if (travel > 0 && shutter > 0) {
                place(seconds - 0.5 * shutter, s.center, s.rot);
                place(seconds + 0.5 * shutter, s.center1, s.rot1);
                s.blurOn = 1;
                if (fine) s.samples = 96;
            } else {
                place(seconds, s.center, s.rot);
            }
            s.iorB = s.spectral ? spread : 0.f;
            s.iorA = nd - s.iorB / (0.5876f * 0.5876f);
            const auto tint = p.vec("solid_tint");
            const float density = clampf(p.f("solid_density"), 0.f, 4.f);
            for (int ch = 0; ch < 3; ++ch) s.absorb[ch] = 0.06f + density * (1.f - clampf(static_cast<float>(tint[ch]), 0.f, 1.f));
            s.frost = clampf(p.f("solid_frost"), 0.f, 1.f);
            s.refocus = gsPlate(shape) ? clampf(p.f("solid_refocus"), 0.f, 1.f) : 0.f;
            const float lightAngle = p.f("solid_light_angle") * kPi / 180.f;
            s.lightCos = std::cos(lightAngle);
            s.lightSin = std::sin(lightAngle);
            gsBuildPlanes(s);
            s.reflection = clampf(p.f("solid_reflection"), 0.f, 2.f);
            s.bevel = clampf(p.f("solid_bevel"), 0.f, 0.25f) * (shape == 17 || shape == 18 ? 0.1f : 1.f);
            s.studio = shape == 17 || shape == 18 ? 1 : 0;
            s.light = clampf(p.f("solid_light"), 0.f, 4.f);
            s.environment = clampf(p.f("solid_environment"), 0.f, 1.f);
            s.strength = clampf(p.f("solid_strength"), 0.f, 2.f);
            s.thickness = clampf(p.f("solid_thickness"), 0.02f, 0.6f);
            s.facets = facets;
            s.highlight = clampf(p.f("highlight_boost"), 0.f, 2.f);
            const float coc = s.focal * s.aperture * std::fabs(1.f / s.focusZ - 1.f / s.sceneZ) * clampf(p.f("scene_defocus"), 0.f, 1.f);
            s.blur = std::min(coc, 0.05f * diag);
            s.taps = s.blur > 0.5f ? std::clamp(static_cast<int>(std::ceil(1.2566f * s.blur * s.blur)), 12, fine ? 512 : 96) : 0;
        }
        const int* order = kMoOrders[std::clamp(p.i("element_order"), 0, 5)];
        std::copy(order, order + 3, stack.order);
        for (int e = 0; e < 3; ++e) {
            std::string k = "e" + std::to_string(e + 1) + "_";
            int type = p.i(k + "type");
            OpticElement& element = stack.elements[e];
            element.enabled = type > 0;
            if (!element.enabled) continue;
            RLParams& P = element.glass;
            P.mode = type - 1;
            stack.hasArray |= P.mode == 7;
            P.facets = facets;
            const float depth = p.f(k + "depth");
            double cx, cy;
            c.pointToPixels(p.vec(k + "position"), cx, cy);
            float rotation = p.f(k + "rotation"), tiltX = p.f(k + "tilt_x"), tiltY = p.f(k + "tilt_y");
            if (travel > 0) {
                const double t = phase + 1.79 * e;
                const double mx = motion == 1 ? 0.68 * std::sin(t) + 0.20 * std::sin(2.37 * t) + 0.12 * std::sin(5.13 * t) : std::sin(t);
                const double my = motion == 1 ? 0.58 * std::sin(0.83 * t + 1.4) + 0.27 * std::cos(2.91 * t) : std::cos(t);
                cx += c.frame.w() * travel * mx;
                cy += c.frame.h() * travel * my * (motion == 3 ? 0.24 : 0.74);
                rotation += travel * 95.f * static_cast<float>(my);
                tiltX += travel * 60.f * static_cast<float>(mx);
                tiltY += travel * 35.f * static_cast<float>(my);
            }
            const float rot = rotation * kPi / 180.f;
            P.cs = std::cos(rot); P.sn = std::sin(rot);
            float sizeF = clampf(p.f(k + "size"), 0.05f, 2.f);
            P.s = std::max(1.f, sizeF * (1.f + 0.6f * depth) * (P.mode == 0 ? 2.f * diag : c.frame.w() * 0.5f));
            P.e = std::max(soft, 0.75f / P.s);
            P.curv = clampf(p.f(k + "curvature"), -4.f, 4.f);
            P.T = std::min(std::max(0.f, c.px(p.f(k + "distance"))), 4.f * diag);
            P.dist = std::max(0.f, 2.f * P.T + depth * 0.15f * diag);
            P.hscale = P.T / P.s;
            P.diag = diag;
            P.cx = static_cast<float>(cx); P.cy = static_cast<float>(cy);
            element.aspect = clampf(p.f(k + "aspect"), 0.15f, 3.f);
            element.ringWidth = clampf(p.f(k + "ring_width"), 0.06f, 0.9f);
            element.repeatOffset = p.f(k + "repeat_offset");
            element.repeatScale = p.f(k + "repeat_scale");
            element.facetFeather = c.px(p.f(k + "facet_feather"));
            element.facetDefocus = c.px(p.f(k + "facet_defocus"));
            element.tiltX = std::tan(clampf(tiltX, -70.f, 70.f) * kPi / 180.f);
            element.tiltY = std::tan(clampf(tiltY, -70.f, 70.f) * kPi / 180.f);
            element.blur = std::min(0.12f * diag * std::fabs(depth - focus) / fstop, c.px(48.f));
            float ior = clampf(p.f(k + "ior"), 1.f, 2.5f);
            float A = ior - B / (0.55f * 0.55f);
            const float lambda[3] = {0.65f, 0.55f, 0.45f};
            for (int i = 0; i < 3; ++i) P.nc[i] = clampf(fine ? A + B / (lambda[i] * lambda[i]) : ior, 1.f, 3.f);
        }
        if (gpu::multiOptic(stack, src, c.dst, c.window)) return;
        requireHost(c);
        Rect R = intersect(c.frame, src.bounds);
        if (R.empty()) R = src.bounds;
        OwnedImage decoded;
        ImageView linear = src;
        if (!stack.mask && enc != Encoding::Linear) {
            decoded.reset(R);
            parallelRows(R.y1, R.y2, [&](int y) {
                for (int x = R.x1; x < R.x2; ++x) decoded.view.store(x, y, decode(src.load(x, y), enc));
            });
            linear = decoded.view;
        }
        const auto raw = [&](int x, int y) {
            const RGBA v = src.at(x, y, SampleEdge::Black);
            return PL4{v.r, v.g, v.b, v.a};
        };
        OwnedImage defocused;
        const ImageView crisp = linear;
        if (!stack.mask && stack.solid.shape > 0 && stack.solid.taps > 0) {
            const ImageView sharp = linear;
            const auto tap = [&](float x, float y) {
                const RGBA v = sharp.bilinear(x, y, static_cast<SampleEdge>(stack.edge));
                return PL4{v.r, v.g, v.b, v.a};
            };
            defocused.reset(sharp.bounds);
            parallelRows(sharp.bounds.y1, sharp.bounds.y2, [&](int y) {
                for (int x = sharp.bounds.x1; x < sharp.bounds.x2; ++x) {
                    const PL4 v = gsBokehPixel(tap, x + 0.5f, y + 0.5f, stack.solid.blur, stack.solid.taps, stack.solid.highlight);
                    defocused.view.store(x, y, {v.r, v.g, v.b, v.a});
                }
            });
            linear = defocused.view;
        }
        const auto sample = [&](float x, float y) {
            const RGBA v = linear.bilinear(x, y, static_cast<SampleEdge>(stack.edge));
            return PL4{v.r, v.g, v.b, v.a};
        };
        const auto sharp = [&](float x, float y) {
            const RGBA v = crisp.bilinear(x, y, static_cast<SampleEdge>(stack.edge));
            return PL4{v.r, v.g, v.b, v.a};
        };
        parallelRows(c.window.y1, c.window.y2, [&](int y) {
            for (int x = c.window.x1; x < c.window.x2; ++x) {
                const PL4 out = opticStackPixel(stack, x, y, raw, sample, sharp);
                c.dst.store(x, y, {out.r, out.g, out.b, out.a});
            }
        });
    };
    m.render = [optics](const RenderContext& c) {
        optics(c);
        addStreak(c);
    };
    return m;
}

}
