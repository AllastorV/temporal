#include "modules.h"

namespace tp {
namespace {

int pixelFloorDiv(int a, int b) {
    int q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

Rect pixelCellRect(const Rect& frame, int g, int cx, int cy) {
    return intersect(Rect{frame.x1 + cx * g, frame.y1 + cy * g, frame.x1 + (cx + 1) * g, frame.y1 + (cy + 1) * g}, frame);
}

struct PixelStats {
    RGBA avg;
    double var = 0;
};

PixelStats pixelStats(const ImageView& src, const Rect& r, bool wantVar) {
    PixelStats s;
    if (r.empty() || !src.valid()) return s;
    Rect q = intersect(r, src.bounds);
    double acc[4] = {0, 0, 0, 0}, sl = 0, sl2 = 0;
    for (int y = q.y1; y < q.y2; ++y) {
        const float* p = src.px(q.x1, y);
        for (int x = q.x1; x < q.x2; ++x, p += 4) {
            acc[0] += p[0]; acc[1] += p[1]; acc[2] += p[2]; acc[3] += p[3];
            if (wantVar) {
                double l = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
                sl += l; sl2 += l * l;
            }
        }
    }
    double n = static_cast<double>(r.w()) * r.h();
    s.avg = RGBA{static_cast<float>(acc[0] / n), static_cast<float>(acc[1] / n), static_cast<float>(acc[2] / n), static_cast<float>(acc[3] / n)};
    if (wantVar) {
        double m = sl / n;
        s.var = std::max(0.0, sl2 / n - m * m);
        if (!std::isfinite(s.var)) s.var = 0;
    }
    return s;
}

struct PixelCells {
    int g = 1, cx0 = 0, cy0 = 0, nx = 0, ny = 0;
    std::vector<RGBA> avg;
    const RGBA& at(int cx, int cy) const { return avg[static_cast<size_t>(cy - cy0) * nx + (cx - cx0)]; }
};

PixelCells pixelComputeCells(const RenderContext& c, const ImageView& src, int g) {
    PixelCells cells;
    cells.g = g;
    cells.cx0 = pixelFloorDiv(c.window.x1 - c.frame.x1, g);
    cells.cy0 = pixelFloorDiv(c.window.y1 - c.frame.y1, g);
    cells.nx = pixelFloorDiv(c.window.x2 - 1 - c.frame.x1, g) - cells.cx0 + 1;
    cells.ny = pixelFloorDiv(c.window.y2 - 1 - c.frame.y1, g) - cells.cy0 + 1;
    cells.avg.assign(static_cast<size_t>(cells.nx) * cells.ny, RGBA{});
    parallelRows(0, cells.ny, [&](int j) {
        for (int i = 0; i < cells.nx; ++i)
            cells.avg[static_cast<size_t>(j) * cells.nx + i] = pixelStats(src, pixelCellRect(c.frame, g, cells.cx0 + i, cells.cy0 + j), false).avg;
    });
    return cells;
}

void pixelFill(const ImageView& dst, const Rect& r, const RGBA& v) {
    for (int y = r.y1; y < r.y2; ++y)
        for (int x = r.x1; x < r.x2; ++x) dst.store(x, y, v);
}

void pixelMosaicNode(const ImageView& src, const ImageView& dst, const Rect& win, const Rect& r, int minS, double thr) {
    if (r.empty() || intersect(r, win).empty()) return;
    bool sx = r.w() >= 2 * minS, sy = r.h() >= 2 * minS;
    PixelStats s = pixelStats(src, r, sx || sy);
    if ((sx || sy) && s.var > thr) {
        int mx = sx ? r.x1 + r.w() / 2 : r.x2;
        int my = sy ? r.y1 + r.h() / 2 : r.y2;
        pixelMosaicNode(src, dst, win, Rect{r.x1, r.y1, mx, my}, minS, thr);
        pixelMosaicNode(src, dst, win, Rect{mx, r.y1, r.x2, my}, minS, thr);
        pixelMosaicNode(src, dst, win, Rect{r.x1, my, mx, r.y2}, minS, thr);
        pixelMosaicNode(src, dst, win, Rect{mx, my, r.x2, r.y2}, minS, thr);
        return;
    }
    pixelFill(dst, intersect(r, win), s.avg);
}

struct PixelSortItem {
    float l;
    RGBA c;
};

void pixelSortLine(std::vector<PixelSortItem>& line, float lo, float hi, bool desc) {
    const size_t n = line.size();
    size_t i = 0;
    while (i < n) {
        if (!(line[i].l >= lo && line[i].l <= hi)) { ++i; continue; }
        size_t j = i;
        while (j < n && line[j].l >= lo && line[j].l <= hi) ++j;
        if (desc) std::stable_sort(line.begin() + i, line.begin() + j, [](const PixelSortItem& a, const PixelSortItem& b) { return a.l > b.l; });
        else std::stable_sort(line.begin() + i, line.begin() + j, [](const PixelSortItem& a, const PixelSortItem& b) { return a.l < b.l; });
        i = j;
    }
}

constexpr uint8_t kPixelGlyphs[10][7] = {
    {0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0x0C, 0x0C},
    {0, 0x0C, 0x0C, 0, 0x0C, 0x0C, 0},
    {0, 0, 0, 0x1F, 0, 0, 0},
    {0, 0, 0x1F, 0, 0x1F, 0, 0},
    {0, 0x04, 0x04, 0x1F, 0x04, 0x04, 0},
    {0, 0x04, 0x15, 0x0E, 0x15, 0x04, 0},
    {0x0A, 0x0A, 0x1F, 0x0A, 0x1F, 0x0A, 0x0A},
    {0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03},
    {0x0E, 0x11, 0x17, 0x15, 0x17, 0x10, 0x0F},
};

constexpr int kPixelBayer[8][8] = {
    {0, 32, 8, 40, 2, 34, 10, 42},
    {48, 16, 56, 24, 50, 18, 58, 26},
    {12, 44, 4, 36, 14, 46, 6, 38},
    {60, 28, 52, 20, 62, 30, 54, 22},
    {3, 35, 11, 43, 1, 33, 9, 41},
    {51, 19, 59, 27, 49, 17, 57, 25},
    {15, 47, 7, 39, 13, 45, 5, 37},
    {63, 31, 55, 23, 61, 29, 53, 21},
};

}

ModuleDef makePixel() {
    ModuleDef m;
    m.id = "pixel";
    m.name = "Pixel";
    m.family = "Digital";
    m.description = "Digital degradation on display-referred values: pixelate, adaptive mosaic, threshold pixel sort, ASCII glyphs, ordered dither and column melt.";
    m.temporal = false;
    m.usesEncoding = false;
    m.params = {
        Choice("mode", "Mode", {"Pixelate", "Adaptive Mosaic", "Pixel Sort", "ASCII", "Dither", "Melt"}, 0,
               "Pixelate: box-averaged grid cells. Adaptive Mosaic: quadtree cells that split where detail is high. Pixel Sort: sorts runs of pixels by luma. ASCII: cells drawn as text glyphs. Dither: ordered Bayer quantization. Melt: columns drip downward."),
        Dbl("grid", "Grid Size", 12.0, 1.0, 512.0, 1.0, 64.0, Unit::Pixels, "Cell size in full-resolution pixels (Pixelate, Mosaic top cells are 4x this, ASCII, Dither pre-pixelation). 1 = no pixelation."),
        Int("color_depth", "Color Depth", 8, 2, 256, 2, 32, Unit::None, "Dither: number of levels per channel."),
        Dbl("dither_amount", "Dither Amount", 1.0, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Dither: strength of the Bayer threshold pattern (0 = plain rounding)."),
        Dbl("variance_threshold", "Detail Threshold", 0.01, 0.0, 0.25, 0.0, 0.05, Unit::None, "Adaptive Mosaic: a cell splits while its luma variance exceeds this. Lower = more small cells."),
        Dbl("sort_low", "Sort Low", 0.25, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Pixel Sort: lowest luma included in a sorted run."),
        Dbl("sort_high", "Sort High", 0.85, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Pixel Sort: highest luma included in a sorted run."),
        Dbl("sort_angle", "Sort Angle", 45, -180, 180, -180, 180, Unit::Degrees, "Pixel Sort direction in degrees when Custom Angle is selected."),
        Choice("direction", "Sort Direction", {"Horizontal", "Vertical", "Custom Angle"}, 0, "Pixel Sort: sort along rows or columns."),
        Choice("sort_order", "Sort Order", {"Ascending", "Descending"}, 0, "Pixel Sort: dark-to-bright or bright-to-dark along the run, in increasing x (rows) or increasing y (columns, bottom-up in OFX)."),
        Choice("ascii_color", "ASCII Color", {"Cell Color", "Mono"}, 0, "ASCII: glyphs take the average cell color, or are white."),
        Dbl("melt_amount", "Melt Amount", 120.0, 0.0, 2000.0, 0.0, 400.0, Unit::Pixels, "Melt: maximum drip length in full-resolution pixels."),
        Dbl("persistence", "Persistence", 0.8, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Melt: how slowly the drip pattern changes frame to frame (1 = frozen)."),
        Int("seed", "Seed", 1, 0, 1000000, 0, 1000, Unit::None, "Melt: random pattern seed."),
    };
    m.framesNeeded = [](const ParamValues&, double) { return FrameWindow{}; };
    m.render = [](const RenderContext& c) {
        const ParamValues& p = c.p();
        const int mode = p.i("mode");
        const ImageView src = c.source();
        const Rect& win = c.window;
        if (!src.valid()) {
            parallelRows(win.y1, win.y2, [&](int y) {
                for (int x = win.x1; x < win.x2; ++x) c.dst.store(x, y, RGBA{});
            });
            return;
        }
        const double maxCell = std::max(1, std::max(c.frame.w(), c.frame.h()));
        const int g = static_cast<int>(std::clamp(std::lround(finiteOr(p.f("grid"), 1.f) * c.scale), 1L, static_cast<long>(maxCell)));

        if (mode == 0 || mode == 3 || mode == 4) {
            if (mode == 4 && g <= 1) {
                const int levels = std::clamp(p.i("color_depth"), 2, 256);
                const float q = static_cast<float>(levels - 1), amt = saturate(p.f("dither_amount"));
                parallelRows(win.y1, win.y2, [&](int y) {
                    for (int x = win.x1; x < win.x2; ++x) {
                        RGBA v = src.at(x, y, SampleEdge::Black);
                        float t = ((kPixelBayer[(y - c.frame.y1) & 7][(x - c.frame.x1) & 7] + 0.5f) / 64.f - 0.5f) * amt;
                        for (int ch = 0; ch < 3; ++ch) v[ch] = std::floor(v[ch] * q + 0.5f + t) / q;
                        c.dst.store(x, y, v);
                    }
                });
                return;
            }
            const PixelCells cells = pixelComputeCells(c, src, g);
            const int levels = std::clamp(p.i("color_depth"), 2, 256);
            const float q = static_cast<float>(levels - 1), amt = saturate(p.f("dither_amount"));
            const bool mono = p.i("ascii_color") == 1;
            parallelRows(win.y1, win.y2, [&](int y) {
                const int cy = pixelFloorDiv(y - c.frame.y1, g);
                const int ly = y - c.frame.y1 - cy * g;
                for (int x = win.x1; x < win.x2; ++x) {
                    const int cx = pixelFloorDiv(x - c.frame.x1, g);
                    RGBA v = cells.at(cx, cy);
                    if (mode == 3) {
                        const int lx = x - c.frame.x1 - cx * g;
                        int idx = std::clamp(static_cast<int>(finiteOr(luma(v), 0.f) * 10.f), 0, 9);
                        int gx = static_cast<int>((lx + 0.5f) / g * 6.f), gr = 7 - static_cast<int>((ly + 0.5f) / g * 8.f);
                        bool on = gx < 5 && gr >= 0 && gr < 7 && ((kPixelGlyphs[idx][gr] >> (4 - gx)) & 1);
                        float a = v.a;
                        if (!on) v = RGBA{0, 0, 0, a};
                        else if (mono) v = RGBA{a, a, a, a};
                    } else if (mode == 4) {
                        float t = ((kPixelBayer[(y - c.frame.y1) & 7][(x - c.frame.x1) & 7] + 0.5f) / 64.f - 0.5f) * amt;
                        for (int ch = 0; ch < 3; ++ch) v[ch] = std::floor(v[ch] * q + 0.5f + t) / q;
                    }
                    c.dst.store(x, y, v);
                }
            });
            return;
        }

        if (mode == 1) {
            const int top = static_cast<int>(std::min<double>(static_cast<double>(g) * 4.0, maxCell));
            const int minS = std::max(1, g / 4);
            const double thr = std::max(0.0, p.d("variance_threshold"));
            const int tx0 = pixelFloorDiv(win.x1 - c.frame.x1, top), tx1 = pixelFloorDiv(win.x2 - 1 - c.frame.x1, top);
            const int ty0 = pixelFloorDiv(win.y1 - c.frame.y1, top), ty1 = pixelFloorDiv(win.y2 - 1 - c.frame.y1, top);
            parallelRows(win.y1, win.y2, [&](int y) {
                for (int x = win.x1; x < win.x2; ++x) c.dst.store(x, y, RGBA{});
            });
            parallelRows(ty0, ty1 + 1, [&](int ty) {
                for (int tx = tx0; tx <= tx1; ++tx)
                    pixelMosaicNode(src, c.dst, win, pixelCellRect(c.frame, top, tx, ty), minS, thr);
            });
            return;
        }

        if (mode == 2) {
            const float lo = p.f("sort_low"), hi = p.f("sort_high");
            const bool desc = p.i("sort_order") == 1;
            if (p.i("direction") == 2) {
                const double radians = p.d("sort_angle") * kPi / 180.0;
                const double dx = std::cos(radians), dy = std::sin(radians);
                const bool alongX = std::abs(dx) >= std::abs(dy);
                const double slope = alongX ? dy / dx : dx / dy;
                const int length = alongX ? c.frame.w() : c.frame.h();
                const int breadth = alongX ? c.frame.h() : c.frame.w();
                const int drift = static_cast<int>(std::round((length - 1) * slope));
                parallelRows(-std::max(0, drift), breadth - std::min(0, drift), [&](int base) {
                    std::vector<PixelSortItem> line;
                    std::vector<std::pair<int,int>> points;
                    for (int i = 0; i < length; ++i) {
                        const int j = base + static_cast<int>(std::round(i * slope));
                        if (j < 0 || j >= breadth) continue;
                        const int x = c.frame.x1 + (alongX ? i : j);
                        const int y = c.frame.y1 + (alongX ? j : i);
                        const RGBA color = src.at(x, y, SampleEdge::Black);
                        line.push_back({finiteOr(luma(color), 0.f), color});
                        points.emplace_back(x, y);
                    }
                    pixelSortLine(line, lo, hi, desc != ((alongX ? dx : dy) < 0));
                    for (size_t i = 0; i < line.size(); ++i) {
                        const auto [x,y] = points[i];
                        if (win.contains(x,y)) c.dst.store(x,y,line[i].c);
                    }
                });
            } else if (p.i("direction") == 0) {
                const int a = std::min(c.frame.x1, win.x1), b = std::max(c.frame.x2, win.x2);
                parallelRows(win.y1, win.y2, [&](int y) {
                    std::vector<PixelSortItem> line(static_cast<size_t>(b - a));
                    for (int x = a; x < b; ++x) {
                        RGBA v = src.at(x, y, SampleEdge::Black);
                        line[x - a] = {finiteOr(luma(v), 0.f), v};
                    }
                    pixelSortLine(line, lo, hi, desc);
                    for (int x = win.x1; x < win.x2; ++x) c.dst.store(x, y, line[x - a].c);
                });
            } else {
                const int a = std::min(c.frame.y1, win.y1), b = std::max(c.frame.y2, win.y2);
                parallelRows(win.x1, win.x2, [&](int x) {
                    std::vector<PixelSortItem> line(static_cast<size_t>(b - a));
                    for (int y = a; y < b; ++y) {
                        RGBA v = src.at(x, y, SampleEdge::Black);
                        line[y - a] = {finiteOr(luma(v), 0.f), v};
                    }
                    pixelSortLine(line, lo, hi, desc);
                    for (int y = win.y1; y < win.y2; ++y) c.dst.store(x, y, line[y - a].c);
                });
            }
            return;
        }

        const float amount = std::clamp(static_cast<float>(finiteOr(p.f("melt_amount"), 0.f) * c.scale), 0.f, static_cast<float>(std::max(1, c.frame.h())));
        const double persistence = std::clamp(p.d("persistence"), 0.0, 1.0);
        const uint32_t seed = static_cast<uint32_t>(p.i("seed"));
        const float tau = static_cast<float>(std::fmod(c.time * (1.0 - persistence) * 0.35, 65536.0));
        const float colW = std::max(1e-3f, c.px(48.f)), fineW = std::max(1e-3f, c.px(7.f));
        std::vector<float> drip(static_cast<size_t>(win.w()));
        for (int x = win.x1; x < win.x2; ++x) {
            float xf = static_cast<float>(x - c.frame.x1) + 0.5f;
            float n = 0.65f * valueNoise2(seed, xf / colW, tau) + 0.35f * valueNoise2(hashCombine(seed, 0x51u), xf / fineW, tau * 1.7f);
            drip[x - win.x1] = saturate(n) * std::sqrt(saturate(n));
        }
        parallelRows(win.y1, win.y2, [&](int y) {
            for (int x = win.x1; x < win.x2; ++x) {
                RGBA s = src.at(x, y, SampleEdge::Black);
                float off = amount * (0.2f + 0.8f * saturate(finiteOr(luma(s), 0.f))) * drip[x - win.x1];
                c.dst.store(x, y, off > 0.f ? src.bilinear(x + 0.5f, y + 0.5f + off, SampleEdge::Clamp) : s);
            }
        });
    };
    return m;
}

}
