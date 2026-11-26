#include "modules.h"
#include "print_lab_kernel.h"
#include "../gpu/gpu.h"

namespace tp {
namespace {

double printLabClippedDiskArea(double r) {
    if (r <= 0.5) return 3.14159265358979323846 * r * r;
    if (r >= 0.70710678118654752) return 1.0;
    double seg = r * r * std::acos(0.5 / r) - 0.5 * std::sqrt(r * r - 0.25);
    return 3.14159265358979323846 * r * r - 4.0 * seg;
}

const std::vector<float>& printLabRadiusLut() {
    static const std::vector<float> lut = [] {
        std::vector<float> l(kPrintLabLutN + 1);
        for (int i = 0; i <= kPrintLabLutN; ++i) {
            double target = static_cast<double>(i) / kPrintLabLutN, lo = 0.0, hi = 0.70710678118654752;
            for (int it = 0; it < 40; ++it) {
                double mid = 0.5 * (lo + hi);
                if (printLabClippedDiskArea(mid) < target) lo = mid; else hi = mid;
            }
            l[static_cast<size_t>(i)] = static_cast<float>(0.5 * (lo + hi));
        }
        return l;
    }();
    return lut;
}

struct PrintLabLook {
    int engine;
    PrintLabRGB paper;
    PrintLabRGB ink;
    float cell;
    float angleDeg;
    int folds;
    float aging;
    float gloss;
    float roughness;
    float mottle;
    float rough;
    float inkFade;
    float tonerNoise;
    float tonerSoft;
    float cockle;
    float fiber;
    bool gutter;
};

const PrintLabLook& printLabLook(int i) {
    static const PrintLabLook looks[] = {
        {kMono, {0.84f, 0.78f, 0.64f}, {0.14f, 0.13f, 0.12f}, 1.6f, 45.f, 1, 1.0f, 0.5f, 0.5f, 0.55f, 0.8f, 0.18f, 0.f, 0.f, 1.0f, 1.0f, false},
        {kCmyk, {0.89f, 0.85f, 0.78f}, {0.25f, 0.24f, 0.23f}, 2.2f, 45.f, 2, 0.35f, 0.6f, 0.42f, 0.4f, 0.7f, 0.25f, 0.f, 0.f, 1.0f, 0.9f, false},
        {kCmyk, {0.97f, 0.97f, 0.96f}, {0.06f, 0.06f, 0.07f}, 0.75f, 45.f, 0, 0.f, 1.0f, 0.16f, 0.06f, 0.15f, 0.f, 0.f, 0.f, 0.7f, 0.15f, false},
        {kCmyk, {0.94f, 0.93f, 0.90f}, {0.09f, 0.09f, 0.10f}, 0.9f, 45.f, 0, 0.f, 0.6f, 0.32f, 0.15f, 0.3f, 0.03f, 0.f, 0.f, 0.5f, 0.45f, false},
        {kCmyk, {0.90f, 0.84f, 0.70f}, {0.16f, 0.14f, 0.13f}, 1.1f, 45.f, 1, 0.8f, 0.8f, 0.2f, 0.3f, 0.5f, 0.32f, 0.f, 0.f, 0.9f, 0.6f, false},
        {kMono, {0.93f, 0.90f, 0.81f}, {0.11f, 0.10f, 0.10f}, 0.9f, 45.f, 0, 0.3f, 0.4f, 0.45f, 0.25f, 0.45f, 0.05f, 0.f, 0.f, 0.6f, 0.8f, true},
        {kToner, {0.93f, 0.93f, 0.92f}, {0.06f, 0.06f, 0.07f}, 1.f, 0.f, 0, 0.f, 0.5f, 0.4f, 0.3f, 0.f, 0.f, 0.6f, 0.10f, 0.4f, 0.5f, false},
        {kToner, {0.91f, 0.75f, 0.60f}, {0.19f, 0.13f, 0.10f}, 1.f, 0.f, 0, 0.15f, 0.4f, 0.45f, 0.4f, 0.f, 0.f, 1.0f, 0.035f, 0.6f, 0.7f, false},
        {kRiso, {0.95f, 0.93f, 0.88f}, {0.93f, 0.26f, 0.22f}, 1.6f, 45.f, 0, 0.f, 0.4f, 0.5f, 0.5f, 0.9f, 0.f, 0.f, 0.f, 0.5f, 0.9f, false},
        {kSpot, {0.94f, 0.91f, 0.84f}, {0.09f, 0.08f, 0.09f}, 1.f, 0.f, 0, 0.1f, 0.6f, 0.35f, 0.35f, 0.f, 0.f, 0.f, 0.f, 0.6f, 0.8f, false},
    };
    return looks[std::clamp(i, 0, 9)];
}

PrintLabRGB printLabPaperOverride(int choice, const PrintLabRGB& lookPaper) {
    switch (choice) {
    case 1: return {0.96f, 0.96f, 0.95f};
    case 2: return {0.94f, 0.90f, 0.80f};
    case 3: return {0.88f, 0.86f, 0.80f};
    case 4: return {0.92f, 0.77f, 0.63f};
    case 5: return {0.78f, 0.78f, 0.77f};
    case 6: return {0.74f, 0.60f, 0.43f};
    default: return lookPaper;
    }
}

}

ModuleDef makePrintLab() {
    ModuleDef m;
    m.id = "print_lab";
    m.cuda = CudaMode::Device;
    m.name = "Print Lab";
    m.family = "Print";
    m.description = "Prints the image onto a simulated physical sheet (halftone, CMYK, stencil print, photocopy toner or spot inks, with ink mottle, rough dot edges, paper fibre, cockle, ageing and folds) and re-photographs it with a virtual camera: zoom, position, tilt, depth of field, a moving light sheen, vignette and a table under the sheet. Stylized, not calibrated to press measurements.";
    m.temporal = false;
    m.usesEncoding = true;
    m.params = {
        Choice("look", "Look", {"Aged Newspaper", "Folded Broadsheet", "Glossy Magazine", "Magazine Matte", "Old Magazine", "Book Page", "Photocopy", "Hard Toner Copy", "Stencil Print One-Colour", "Spot-Colour Poster", "CRT Phosphor", "LCD Subpixels"}, 1,
               "Print process and paper stock: screen type, paper colour, ink behaviour, ageing, folds and gloss.", "Look"),
        Choice("preview_quality", "Preview Quality", {"Auto (Fast Preview)", "Full", "Fast"}, 0, "Auto uses Fast for host-reported preview renders. Fast forces the reduced sheet resolution when the host does not flag playback. Full always uses the detailed sheet.", "Look"),
        Dbl("texture_motion", "Texture Motion", 0.6, 0, 1, 0, 1, Unit::Normalized, "Animate the paper feed, surface grain and display refresh. Zero locks the texture.", "Look"),
        Choice("paper_colour", "Paper Colour", {"Look Default", "White", "Cream", "Newsprint", "Peach", "Grey", "Kraft"}, 0, "Overrides the paper stock colour of the look.", "Look"),
        Dbl("paper", "Paper", 1.0, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Amount of physical paper character: fibre, cockle, ageing stains, folds and surface relief.", "Look"),
        Dbl("print", "Print", 0.5, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Ink weight: below 0.5 prints lighter and thinner, above 0.5 heavier and darker.", "Look"),
        Dbl("black_point", "Black Point", 0.0, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Input level that prints as solid ink (display values).", "Look"),
        Dbl("white_point", "White Point", 1.0, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Input level that prints as bare paper (display values).", "Look"),
        Dbl("contrast", "Contrast", 1.0, 0.1, 3.0, 0.1, 3.0, Unit::None, "Ink density contrast around the mid-tone.", "Look"),
        Dbl("threshold", "Threshold", 0.5, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Photocopy and Hard Toner Copy: darkness level where toner starts to stick.", "Look"),
        Dbl("halftone_size", "Halftone Size", 2.0, 0.5, 64.0, 0.5, 20.0, Unit::Pixels, "Base halftone cell in full-resolution source pixels; each look scales it (glossy finer, newsprint coarser).", "Look"),
        Dbl("registration", "Misregistration", 1.0, 0.0, 40.0, 0.0, 10.0, Unit::Pixels, "Plate offset in full-resolution source pixels in seeded directions.", "Look"),
        Color("ink_a", "Ink A", 0.93, 0.26, 0.22, "Stencil Print and Spot-Colour Poster: first spot ink (display values).", "Look"),
        Color("ink_b", "Ink B", 0.10, 0.36, 0.62, "Spot-Colour Poster: second spot ink (display values).", "Look"),
        Choice("camera_move", "Camera Move", {"Locked Off", "Slow Drift", "Handheld", "Push In", "Pan Across"}, 0,
               "Virtual re-photography camera motion over time; the light sheen moves across the paper with it.", "Camera"),
        Dbl("zoom", "Zoom", 2.2, 0.5, 8.0, 0.5, 6.0, Unit::None, "Camera magnification of the printed sheet (1 frames the whole page when not tilted).", "Camera"),
        Point("position", "Position", 0.5, 0.6, "Point of the printed page at the centre of the camera view, measured from the top-left of the page (0,0) to the bottom-right (1,1).", "Camera"),
        Dbl("angle", "Angle", 20.0, -70.0, 70.0, -60.0, 60.0, Unit::Degrees, "Tilt of the sheet away from the camera; drives perspective and depth of field.", "Camera"),
        Dbl("roll", "Roll", -2.0, -180.0, 180.0, -20.0, 20.0, Unit::Degrees, "Camera roll.", "Camera"),
        Dbl("depth_of_field", "Depth of Field", 0.6, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Lens blur strength away from the focus plane (macro lens look).", "Camera"),
        Dbl("focus", "Focus", 0.5, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Vertical screen position of the focus plane (0 bottom, 1 top).", "Camera"),
        Dbl("light_angle", "Light Angle", 35.0, -180.0, 180.0, -180.0, 180.0, Unit::Degrees, "Direction of the key light over the sheet.", "Camera"),
        Dbl("sheen", "Sheen", 1.0, 0.0, 3.0, 0.0, 3.0, Unit::None, "Strength of the specular light reflection on the paper (shape set by the look's surface roughness).", "Camera"),
        Dbl("vignette", "Vignette", 0.35, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Lens light falloff toward the frame corners.", "Camera"),
        Bool("show_whole_picture", "Show Whole Picture", false, "Pull the camera back to see the whole printed sheet lying on the table.", "Camera"),
        Int("seed", "Seed", 1, 0, 1000000, 0, 1000, Unit::None, "Seed for registration, paper fibre, stains, folds, toner and handheld motion."),
    };
    m.framesNeeded = [](const ParamValues&, double) { return FrameWindow{}; };
    m.render = [](const RenderContext& c) {
        const ParamValues& p = c.p();
        const ImageView src = c.source();
        const Rect& win = c.window;
        if (!src.valid() || win.empty() || c.frame.empty()) {
            requireHost(c);
            parallelRows(win.y1, win.y2, [&](int y) {
                for (int x = win.x1; x < win.x2; ++x) c.dst.store(x, y, RGBA{});
            });
            return;
        }
        const int lookIdx = p.i("look");
        const PrintLabLook& L = printLabLook(lookIdx);
        const Encoding enc = c.encoding;
        const uint32_t seed = hashCombine(static_cast<uint32_t>(p.i("seed")), static_cast<uint32_t>(lookIdx + 1));
        const Rect F = c.frame;
        const float W = static_cast<float>(F.w()), H = static_cast<float>(F.h());

        const double tsec = c.time / std::max(1.0, c.fps);
        const int move = p.i("camera_move");
        const bool whole = p.b("show_whole_picture");
        double zoom = clampf(p.f("zoom"), 0.5f, 8.f);
        auto pos = p.vec("position");
        double posX = pos[0], posY = pos[1];
        double angle = clampf(p.f("angle"), -70.f, 70.f);
        double roll = p.f("roll");
        if (move == 1) {
            posX += 0.025 * std::sin(tsec * 0.41 + 0.7) + 0.012 * std::sin(tsec * 0.97);
            posY += 0.02 * std::sin(tsec * 0.33 + 1.9);
            angle += 2.5 * std::sin(tsec * 0.27 + 0.4);
            zoom *= 1.0 + 0.03 * std::sin(tsec * 0.21);
            roll += 0.8 * std::sin(tsec * 0.19);
        } else if (move == 2) {
            uint32_t hs = hashCombine(seed, 41u);
            posX += 0.012 * fbm1(hs, tsec * 1.3, 3);
            posY += 0.012 * fbm1(hs + 1u, tsec * 1.3, 3);
            roll += 0.9 * fbm1(hs + 2u, tsec * 1.1, 3);
            angle += 1.5 * fbm1(hs + 3u, tsec * 0.9, 2);
        } else if (move == 3) {
            zoom *= std::min(3.0, 1.0 + 0.08 * std::max(0.0, tsec));
        } else if (move == 4) {
            posX += 0.035 * tsec;
            angle += 1.0 * std::sin(tsec * 0.5);
        }
        if (whole) { zoom = 0.62 * std::cos(angle * kPi / 180.0); posX = 0.5; posY = 0.5; }

        float ps = whole ? 2.f : clampf(static_cast<float>(zoom), 1.f, 3.f);
        ps = std::min(ps, std::sqrt(10.0e6f / std::max(1.f, W * H)));
        ps = std::max(ps, 0.5f);
        if (p.i("preview_quality") == 2 || (p.i("preview_quality") == 0 && c.preview)) ps = std::min(ps, 0.5f);
        const int PW = std::max(2, static_cast<int>(std::ceil(W * ps))), PH = std::max(2, static_cast<int>(std::ceil(H * ps)));
        const float u1 = c.px(1.f) * ps;

        auto colorParam = [&](const char* id) {
            auto v = p.vec(id);
            return PrintLabRGB{saturate(static_cast<float>(v[0])), saturate(static_cast<float>(v[1])), saturate(static_cast<float>(v[2]))};
        };
        PLParams P;
        P.engine = lookIdx == 10 ? kCrt : lookIdx == 11 ? kLcd : L.engine;
        const float motion = p.f("texture_motion");
        P.phase = static_cast<float>(tsec) * motion;
        P.surfaceX = motion * u1 * (2.f * std::sin(tsec * 2.3) + std::sin(tsec * 0.71));
        P.surfaceY = motion * u1 * (3.f * std::sin(tsec * 1.7) + std::sin(tsec * 3.13));
        P.lookInk = L.ink;
        P.paperBase = printLabPaperOverride(p.i("paper_colour"), L.paper);
        P.inkA = colorParam("ink_a");
        P.inkB = colorParam("ink_b");
        P.aging = L.aging; P.mottle = L.mottle; P.rough = L.rough; P.tonerNoise = L.tonerNoise; P.tonerSoft = L.tonerSoft;
        P.cockle = L.cockle; P.fiber = L.fiber; P.gutter = L.gutter ? 1 : 0;
        P.paperAmt = saturate(p.f("paper"));
        P.inkGamma = std::exp2((0.5f - saturate(p.f("print"))) * 2.f);
        const float bp = saturate(p.f("black_point")), wp = saturate(p.f("white_point"));
        const float range = std::max(1e-3f, wp - bp);
        P.contrastP = clampf(finiteOr(p.f("contrast"), 1.f), 0.1f, 3.f);
        P.thr = saturate(p.f("threshold"));
        P.s = clampf(finiteOr(static_cast<float>(p.d("halftone_size")), 2.f) * u1 * L.cell, 2.f, static_cast<float>(std::max(PW, PH)));
        const float reg = clampf(finiteOr(p.f("registration"), 0.f) * u1, 0.f, static_cast<float>(std::max(PW, PH)));
        P.w = std::max(1.f, 1.f + 0.04f * P.s);
        P.u1 = u1;
        P.PW = PW; P.PH = PH;
        P.lut = printLabRadiusLut().data();
        const float baseAngle = L.angleDeg * kPi / 180.f;
        auto addPlate = [&](float angleOffDeg, PrintLabRGB ink) {
            PLPlate& pl = P.plates[P.nplates];
            float a = baseAngle + angleOffDeg * kPi / 180.f;
            pl.cs = std::cos(a); pl.sn = std::sin(a);
            float dir = 2.f * kPi * hashFloat(seed, P.nplates, 77);
            float rr = P.nplates == 0 ? 0.f : reg;
            pl.ox = rr * std::cos(dir); pl.oy = rr * std::sin(dir);
            pl.ink = PrintLabRGB{lerpf(ink.r, P.paperBase.r, L.inkFade), lerpf(ink.g, P.paperBase.g, L.inkFade), lerpf(ink.b, P.paperBase.b, L.inkFade)};
            ++P.nplates;
        };
        if (L.engine == kMono || L.engine == kToner) addPlate(0.f, L.ink);
        else if (L.engine == kCmyk) {
            addPlate(-30.f, {0.f, 0.62f, 0.90f});
            addPlate(30.f, {0.92f, 0.08f, 0.48f});
            addPlate(-45.f, {1.f, 0.88f, 0.f});
            addPlate(0.f, L.ink);
        } else if (L.engine == kRiso) addPlate(0.f, P.inkA);
        else {
            addPlate(0.f, P.inkA);
            addPlate(0.f, P.inkB);
            addPlate(0.f, L.ink);
        }
        for (int i = 0; i < std::min(L.folds, 4); ++i) {
            bool vertical = (i % 2 == 0) == (hashFloat(seed, i, 501) < 0.5f);
            float tilt = (hashFloat(seed, i, 502) - 0.5f) * 0.06f;
            float a = (vertical ? 0.f : 0.5f * kPi) + tilt;
            PLFold& f = P.folds[P.nfolds++];
            f.nx = std::cos(a); f.ny = std::sin(a);
            float fp = 0.3f + 0.4f * hashFloat(seed, i, 503);
            float qx = vertical ? fp * PW : 0.5f * PW;
            float qy = vertical ? 0.5f * PH : fp * PH;
            f.d0 = f.nx * qx + f.ny * qy;
            f.depth = (hashFloat(seed, i, 504) < 0.5f ? -1.f : 1.f) * (0.7f + 0.3f * hashFloat(seed, i, 505));
        }
        P.sFiberA = hashCombine(seed, 11u); P.sFiberB = hashCombine(seed, 12u); P.sGrain = hashCombine(seed, 13u);
        P.sMottle = hashCombine(seed, 16u); P.sStain = hashCombine(seed, 17u); P.sRough = hashCombine(seed, 18u);
        P.sStreak = hashCombine(seed, 14u); P.sCockle = hashCombine(seed, 19u); P.sFox = hashCombine(seed, 20u);
        P.creaseBand = std::max(1.f, 110.f * u1); P.creaseCore = std::max(0.6f, 1.4f * u1);
        P.cockleS = std::max(1.f, 230.f * u1);

        const int margin = static_cast<int>(std::ceil((P.s + reg) / ps + 4.f));
        const Rect R{F.x1 - margin, F.y1 - margin, F.x2 + margin, F.y2 + margin};
        const float toneSigma = L.engine == kToner ? c.px(0.7f) : (L.engine == kSpot ? c.px(0.9f) : std::max(0.5f, 0.3f * P.s / ps));

        PLCamera C;
        const float th = static_cast<float>(angle * kPi / 180.0);
        C.sth = std::sin(th); C.cth = std::cos(th);
        const float rr = static_cast<float>(roll * kPi / 180.0);
        C.crr = std::cos(rr); C.srr = std::sin(rr);
        C.f = static_cast<float>(zoom);
        C.halfH = 0.5f * H;
        C.dof = saturate(p.f("depth_of_field"));
        const float focusV = saturate(p.f("focus")) * 2.f - 1.f;
        C.zFocus = C.cth / std::max(0.05f, C.cth - C.sth * focusV / C.f);
        const float la = p.f("light_angle") * kPi / 180.f;
        C.Lw = plNorm({0.55f * std::cos(la), 0.55f * std::sin(la) + 0.35f, -0.75f});
        for (int i = 0; i < 5; ++i) {
            float o = (i - 2) * 0.22f;
            C.Larea[i] = plNorm({0.55f * std::cos(la) - o * std::sin(la), 0.55f * std::sin(la) + 0.35f + o * std::cos(la), -0.75f});
        }
        C.sheen = std::max(0.f, p.f("sheen")) * L.gloss;
        C.alpha2 = L.roughness * L.roughness;
        C.vig = saturate(p.f("vignette"));
        C.n0 = PLVec{0.f, C.sth, -C.cth};
        C.ea = PLVec{1.f, 0.f, 0.f};
        C.eb = PLVec{0.f, C.cth, C.sth};
        C.flatDiffuse = std::max(0.05f, plDot(C.n0, C.Lw));
        C.screenDiag = std::sqrt(W * W + H * H) * 0.5f;
        C.sTable = hashCombine(seed, 31u);
        C.pageY0 = static_cast<float>(1.0 - posY);
        C.posX = static_cast<float>(posX);
        C.W = W; C.H = H; C.Fx1 = F.x1; C.Fy1 = F.y1;
        C.ps = ps;
        C.scale = static_cast<float>(c.scale);
        C.paperAmt = P.paperAmt;
        C.enc = static_cast<int>(enc);

        if (gpu::available() && gpu::printLab(P, C, gpu::PrintLabTone{intersect(F, src.bounds), R, bp, range, toneSigma, static_cast<int>(enc)}, src, c.dst, win)) return;
        requireHost(c);

        OwnedImage tone(R);
        parallelRows(R.y1, R.y2, [&](int y) {
            for (int x = R.x1; x < R.x2; ++x) {
                RGBA v = src.at(std::clamp(x, F.x1, F.x2 - 1), std::clamp(y, F.y1, F.y2 - 1), SampleEdge::Clamp);
                float a = finiteOr(v.a, 0.f);
                RGBA o{1.f, 1.f, 1.f, 1.f};
                if (a > 1e-5f)
                    for (int ch = 0; ch < 3; ++ch) {
                        float d = saturate(finiteOr(enc == Encoding::Gamma24 ? v[ch] / a : encodeValue(decodeValue(v[ch] / a, enc), Encoding::Gamma24), 0.f));
                        o[ch] = saturate((d - bp) / range);
                    }
                tone.view.store(x, y, o);
            }
        });
        gaussianBlur(tone, toneSigma);

        const ImageView tv = tone.view;
        auto toneAt = [&](float qx, float qy) {
            RGBA t = tv.bilinear(F.x1 + qx / ps, F.y1 + qy / ps, SampleEdge::Clamp);
            return PrintLabRGB{saturate(t.r), saturate(t.g), saturate(t.b)};
        };
        std::vector<OwnedImage> mips;
        mips.emplace_back(Rect{0, 0, PW, PH});
        {
            const ImageView pv = mips[0].view;
            parallelRows(0, PH, [&](int j) {
                for (int i = 0; i < PW; ++i) {
                    PL4 o = plPagePixel(P, i, j, toneAt);
                    pv.store(i, j, RGBA{o.r, o.g, o.b, o.a});
                }
            });
        }
        while (mips.size() < 9) {
            const OwnedImage& prev = mips.back();
            const Rect pb = prev.view.bounds;
            if (pb.w() <= 2 || pb.h() <= 2) break;
            Rect nb{0, 0, (pb.w() + 1) / 2, (pb.h() + 1) / 2};
            OwnedImage next(nb);
            const ImageView prv = prev.view;
            parallelRows(0, nb.h(), [&](int j) {
                for (int i = 0; i < nb.w(); ++i) {
                    RGBA a = prv.at(2 * i, 2 * j, SampleEdge::Clamp) + prv.at(2 * i + 1, 2 * j, SampleEdge::Clamp)
                           + prv.at(2 * i, 2 * j + 1, SampleEdge::Clamp) + prv.at(2 * i + 1, 2 * j + 1, SampleEdge::Clamp);
                    next.view.store(i, j, a * 0.25f);
                }
            });
            mips.push_back(std::move(next));
        }
        C.maxLod = static_cast<int>(mips.size()) - 1;
        auto samplePage = [&](float qx, float qy, float lod) {
            int l0 = static_cast<int>(std::floor(lod));
            float t = lod - l0;
            float sc0 = 1.f / static_cast<float>(1 << l0);
            RGBA a = mips[static_cast<size_t>(l0)].view.bilinear(qx * sc0, qy * sc0, SampleEdge::Clamp);
            if (t > 1e-3f && l0 < C.maxLod) a = lerp(a, mips[static_cast<size_t>(l0) + 1].view.bilinear(qx * sc0 * 0.5f, qy * sc0 * 0.5f, SampleEdge::Clamp), t);
            return PL4{a.r, a.g, a.b, a.a};
        };
        parallelRows(win.y1, win.y2, [&](int y) {
            for (int x = win.x1; x < win.x2; ++x) {
                PL4 o = plCameraPixel(P, C, x, y, samplePage);
                float alpha = saturate(finiteOr(src.at(x, y, SampleEdge::Black).a, 0.f));
                c.dst.store(x, y, RGBA{o.r * alpha, o.g * alpha, o.b * alpha, alpha});
            }
        });
    };
    return m;
}

}
