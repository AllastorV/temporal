#include "modules.h"
#include "refraction_kernel.h"
#include "../gpu/gpu.h"

namespace tp {

namespace {

struct RlLevel {
    OwnedImage img;
    int f = 1;
};

RlLevel rlBlurred(const ImageView& lin, float sigma) {
    RlLevel L;
    L.f = std::clamp(static_cast<int>(std::floor(sigma / 2.5f)), 1, 64);
    const Rect& R = lin.bounds;
    const int f = L.f, lw = (R.w() + f - 1) / f, lh = (R.h() + f - 1) / f;
    L.img.reset(Rect{0, 0, lw, lh});
    parallelRows(0, lh, [&](int j) {
        const int ya = R.y1 + j * f, yb = std::min(R.y2, ya + f);
        for (int i = 0; i < lw; ++i) {
            const int xa = R.x1 + i * f, xb = std::min(R.x2, xa + f);
            RGBA s;
            for (int y = ya; y < yb; ++y)
                for (int x = xa; x < xb; ++x) s += lin.load(x, y);
            L.img.view.store(i, j, s * (1.f / static_cast<float>((xb - xa) * (yb - ya))));
        }
    });
    gaussianBlur(L.img, sigma / static_cast<float>(f));
    return L;
}

}

ModuleDef makeRefractionLab() {
    ModuleDef m;
    m.id = "refraction_lab";
    m.cuda = CudaMode::Device;
    m.name = "Refraction Lab";
    m.family = "Optics";
    m.description = "Analytic virtual glass in front of the image. The source is treated as a flat 2D plane at a fixed distance (twice the glass thickness) behind the glass; rays are refracted with Snell's law through a shaped front face and a flat back face, with per-channel dispersion and total internal reflection.";
    m.temporal = false;
    m.usesEncoding = true;
    m.params = {
        Choice("mode", "Mode", {"Diopter", "Prism", "Cylindrical", "Replicator"}, 0,
               "Diopter: spherical lens on one side of a split line (the other half untouched). Prism: flat wedge giving a constant shift with a hard edge. Cylindrical: lens curved along one axis. Replicator: strips of wedges with different slopes that repeat the image."),
        Dbl("ior", "Index of Refraction", 1.5, 1.0, 2.5, 1.0, 2.5, Unit::None, "Refractive index of the glass at 550 nm. 1.0 means no refraction; 1.5 is crown glass."),
        Dbl("thickness", "Thickness", 200.0, 0.0, 2000.0, 0.0, 600.0, Unit::Pixels, "Glass thickness in full-resolution pixels. Scales the surface height and the image-plane distance, so it sets how strong the bending is."),
        Dbl("curvature", "Curvature", 1.0, -4.0, 4.0, -4.0, 4.0, Unit::None, "Surface shape strength. Negative values invert the lens or wedge. 0 is flat glass (no deviation)."),
        Dbl("dispersion", "Dispersion", 0.01, 0.0, 0.05, 0.0, 0.05, Unit::None, "Cauchy B coefficient (um^2): red, green and blue are traced with their own index n = A + B / lambda^2, producing color fringes."),
        Dbl("f_stop", "F-Stop", 5.6, 1.0, 22.0, 1.0, 22.0, Unit::None, "Aperture for the defocus of the refracted region. Smaller numbers blur more."),
        Dbl("focus_shift", "Focus Shift", 0.0, -1.0, 1.0, -1.0, 1.0, Unit::None, "Defocus of the image seen through the glass. Blur radius is proportional to |focus shift| x thickness / f-stop (capped at 150 px)."),
        Point("position", "Position", 0.5, 0.5, "Center of the glass (and point on the split line in Diopter), normalized to the frame."),
        Dbl("rotation", "Rotation", 0.0, -360.0, 360.0, -180.0, 180.0, Unit::Degrees, "Rotation of the glass and of the split line."),
        Dbl("size", "Size", 0.5, 0.05, 2.0, 0.05, 2.0, Unit::None, "Glass footprint width as a fraction of the frame width. In Diopter it sets the lens radius as a multiple of twice the frame diagonal, so at 0.5 the half-lens covers its whole side of the frame like a real split diopter."),
        Int("facets", "Facets", 5, 2, 16, 2, 16, Unit::None, "Replicator only: number of strips; their slopes spread evenly from -curvature to +curvature."),
        Dbl("edge_softness", "Edge Softness", 0.01, 0.0, 0.2, 0.0, 0.2, Unit::None, "Width of the anti-aliased footprint edge as a fraction of the glass size."),
    };
    m.framesNeeded = [](const ParamValues&, double) { return FrameWindow{}; };
    m.render = [](const RenderContext& c) {
        const ParamValues& p = c.p();
        const ImageView src = c.source();
        if (!src.valid()) {
            requireHost(c);
            parallelRows(c.window.y1, c.window.y2, [&](int y) {
                for (int x = c.window.x1; x < c.window.x2; ++x) c.dst.store(x, y, RGBA{});
            });
            return;
        }
        const int mode = p.i("mode");
        const Encoding enc = c.encoding;
        const float diag = std::sqrt(static_cast<float>(c.frame.w()) * c.frame.w() + static_cast<float>(c.frame.h()) * c.frame.h());
        const float ior = clampf(p.f("ior"), 1.f, 2.5f);
        const float T = std::min(std::max(0.f, c.px(p.f("thickness"))), 4.f * diag);
        const float dist = 2.f * T;
        const float curv = clampf(p.f("curvature"), -4.f, 4.f);
        const float B = clampf(p.f("dispersion"), 0.f, 0.05f);
        const float fstop = clampf(p.f("f_stop"), 1.f, 22.f);
        const float fshift = clampf(p.f("focus_shift"), -1.f, 1.f);
        const float rot = p.f("rotation") * kPi / 180.f;
        const float cs = std::cos(rot), sn = std::sin(rot);
        const float sizeF = clampf(p.f("size"), 0.05f, 2.f);
        const float s = std::max(1.f, mode == 0 ? sizeF * 2.f * diag : sizeF * c.frame.w() * 0.5f);
        const int facets = std::clamp(p.i("facets"), 2, 16);
        const float e = std::max(clampf(p.f("edge_softness"), 0.f, 0.2f), 0.75f / s);
        double cxd, cyd;
        c.pointToPixels(p.vec("position"), cxd, cyd);
        const float cx = static_cast<float>(cxd), cy = static_cast<float>(cyd);
        const float A = ior - B / (0.55f * 0.55f);
        const float lambda[3] = {0.65f, 0.55f, 0.45f};
        float nc[3];
        for (int i = 0; i < 3; ++i) nc[i] = clampf(A + B / (lambda[i] * lambda[i]), 1.f, 3.f);
        const float hscale = T / s;

        const float blurR = std::min({std::fabs(fshift) * T / fstop * 0.5f, c.px(150.f), diag});
        const bool useBlur = blurR > 0.5f;
        RLParams RP;
        RP.mode = mode; RP.facets = facets; RP.enc = static_cast<int>(enc); RP.useBlur = useBlur ? 1 : 0;
        RP.cs = cs; RP.sn = sn; RP.s = s; RP.e = e; RP.curv = curv; RP.hscale = hscale; RP.cx = cx; RP.cy = cy;
        RP.T = T; RP.dist = dist; RP.diag = diag;
        for (int i = 0; i < 3; ++i) RP.nc[i] = nc[i];
        if (!useBlur && gpu::available() && gpu::refraction(RP, src, c.dst, c.window)) return;
        requireHost(c);
        Rect R = intersect(c.frame, src.bounds);
        if (R.empty()) R = src.bounds;
        OwnedImage decoded;
        ImageView lin = src;
        if (enc != Encoding::Linear) {
            decoded.reset(R);
            parallelRows(R.y1, R.y2, [&](int y) {
                for (int x = R.x1; x < R.x2; ++x) decoded.view.store(x, y, decode(src.load(x, y), enc));
            });
            lin = decoded.view;
        }
        RlLevel blurred;
        if (useBlur) blurred = rlBlurred(lin, blurR * 0.5f);
        const float invf = 1.f / static_cast<float>(blurred.f);

        auto rawAt = [&](int x, int y) { RGBA v = src.at(x, y, SampleEdge::Black); return PL4{v.r, v.g, v.b, v.a}; };
        auto sampleLin = [&](float sx, float sy) {
            RGBA smp = useBlur ? blurred.img.view.bilinear((sx - lin.bounds.x1) * invf, (sy - lin.bounds.y1) * invf, SampleEdge::Clamp)
                               : lin.bilinear(sx, sy, SampleEdge::Clamp);
            return PL4{smp.r, smp.g, smp.b, smp.a};
        };
        parallelRows(c.window.y1, c.window.y2, [&](int y) {
            for (int x = c.window.x1; x < c.window.x2; ++x) {
                PL4 o = rlPixel(RP, x, y, rawAt, sampleLin);
                c.dst.store(x, y, RGBA{o.r, o.g, o.b, o.a});
            }
        });
    };
    return m;
}

}
