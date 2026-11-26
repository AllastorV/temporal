#pragma once
#include "cutwright_art.h"
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>

namespace tp::cwgeo {

using cwart::TopDown;
using cwart::pyRound;

inline cw::Plane fastBlur(const cw::Plane& p, float sigma) {
    cw::Plane out;
    if (sigma >= 0.3f && gpu::planeBlur(p, sigma, out)) return out;
    return cw::blur(p, sigma);
}

inline double phaseOf(int64_t key, long long seed, int salt) {
    uint64_t k = uint64_t(key);
    k = k * 0x9E3779B97F4A7C15ull + uint64_t((seed * 1000003LL + salt) & 0xFFFFFFFFLL);
    k ^= k >> 31;
    k *= 0xBF58476D1CE4E5B9ull;
    k ^= k >> 29;
    return double(k >> 11) / 9007199254740992.0;
}

struct Cells {
    int hh = 0, ww = 0, s = 0;
    std::vector<float> rgb[3];
    std::vector<float> luma;
};

inline Cells cellMeans(TopDown& img, int s) {
    Cells c;
    c.s = s;
    c.hh = std::max(1, img.h / s);
    c.ww = std::max(1, img.w / s);
    for (auto& v : c.rgb) v.assign(size_t(c.hh) * c.ww, 0.f);
    c.luma.assign(size_t(c.hh) * c.ww, 0.f);
    parallelRows(0, c.hh, [&](int cy) {
        for (int cx = 0; cx < c.ww; ++cx) {
            for (int k = 0; k < 3; ++k) {
                double sum = 0;
                for (int dy = 0; dy < s; ++dy)
                    for (int dx = 0; dx < s; ++dx) sum += img.at(k, std::min(cy * s + dy, img.h - 1), std::min(cx * s + dx, img.w - 1));
                c.rgb[k][size_t(cy) * c.ww + cx] = float(sum / (double(s) * s));
            }
            const size_t i = size_t(cy) * c.ww + cx;
            c.luma[i] = 0.2126f * c.rgb[0][i] + 0.7152f * c.rgb[1][i] + 0.0722f * c.rgb[2][i];
        }
    });
    return c;
}

inline void sobel(const std::vector<float>& y, int h, int w, std::vector<float>& gx, std::vector<float>& gy) {
    gx.assign(y.size(), 0.f);
    gy.assign(y.size(), 0.f);
    const auto p = [&](int r, int c) { return y[size_t(std::clamp(r, 0, h - 1)) * w + std::clamp(c, 0, w - 1)]; };
    parallelRows(0, h, [&](int r) {
        for (int c = 0; c < w; ++c) {
            const float x = p(r - 1, c + 1) + 2 * p(r, c + 1) + p(r + 1, c + 1) - p(r - 1, c - 1) - 2 * p(r, c - 1) - p(r + 1, c - 1);
            const float v = p(r + 1, c - 1) + 2 * p(r + 1, c) + p(r + 1, c + 1) - p(r - 1, c - 1) - 2 * p(r - 1, c) - p(r - 1, c + 1);
            gx[size_t(r) * w + c] = x / 8;
            gy[size_t(r) * w + c] = v / 8;
        }
    });
}

inline std::vector<float> cellSignal(const Cells& c, float edge) {
    double sum = 0;
    for (float v : c.luma) sum += v;
    const double mean = double(float(sum / double(c.luma.size())));
    std::vector<float> gx, gy;
    sobel(c.luma, c.hh, c.ww, gx, gy);
    std::vector<float> sig(c.luma.size());
    for (size_t i = 0; i < sig.size(); ++i) {
        const float base = mean < 0.5 ? c.luma[i] - float(mean) : float(mean) - c.luma[i];
        sig[i] = base + edge * std::hypot(gx[i], gy[i]) * 2;
    }
    return sig;
}

inline void horizontalEcho(const ImageView& exchange, long long seed) {
    TopDown img(exchange);
    const int w = img.w, h = img.h, shortSide = std::min(h, w);
    const int s = std::max(2, pyRound(0.012 * shortSide));
    const Cells cells = cellMeans(img, s);
    const int hh = cells.hh, ww = cells.ww;
    const std::vector<float> sig = cellSignal(cells, 0.3f);
    const float thr = 0.15f;
    std::vector<float> val(sig.size());
    for (size_t i = 0; i < sig.size(); ++i) val[i] = std::clamp((sig[i] - thr) / float(std::max(1e-6, 1 - 0.15)), 0.f, 1.f);
    std::vector<double> fb(hh, std::pow(0.92, 1.0 / std::max(1.0, 0.05)));
    for (int r = 0; r < hh; ++r)
        if (phaseOf(r, seed, 31) < 0.15 * 0.5) fb[r] = std::max(fb[r], 0.985);
    std::vector<float> tv = val;
    std::vector<int> source(val.size());
    for (size_t i = 0; i < source.size(); ++i) source[i] = int(i % ww);
    for (int span = 1; span < ww; span *= 2) {
        std::vector<float> nv = tv;
        std::vector<int> ns = source;
        for (int r = 0; r < hh; ++r) {
            const double k = std::pow(fb[r], double(span));
            for (int x = span; x < ww; ++x) {
                const float cand = float(double(tv[size_t(r) * ww + x - span]) * k);
                if (cand > tv[size_t(r) * ww + x]) {
                    nv[size_t(r) * ww + x] = cand;
                    ns[size_t(r) * ww + x] = source[size_t(r) * ww + x - span];
                }
            }
        }
        tv.swap(nv);
        source.swap(ns);
    }
    parallelRows(0, h, [&](int Y) {
        const int cy = std::min(Y / s, hh - 1);
        for (int X = 0; X < w; ++X) {
            const int cx = std::min(X / s, ww - 1);
            const size_t i = size_t(cy) * ww + cx;
            const float alpha = tv[i] > 0.f ? tv[i] : 0.f;
            const size_t from = size_t(cy) * ww + source[i];
            for (int k = 0; k < 3; ++k) {
                const float base = img.at(k, Y, X) * float(1 - 0.0);
                const float color = tv[i] > 0.f ? cells.rgb[k][from] : 0.f;
                img.at(k, Y, X) = std::clamp(base + (color - base) * alpha, 0.f, 1.f);
            }
        }
    });
    img.store(exchange);
}

using Rgb = std::array<double, 3>;

inline Rgb hexColor(const char* hex) { return cwart::hexRgb(hex); }

struct ComposeStore {
    std::vector<double> cov, sum;
};

struct Canvas {
    struct Sample { double x, y, wt; uint32_t color; };
    struct Block { double ax, ay, sx, sy, area; long long kx, ky; uint32_t color; };
    int h, w;
    std::vector<Sample> samples;
    std::vector<Block> blocks;
    std::vector<Rgb> palette;
    Canvas(int hh, int ww) : h(hh), w(ww) {}
    uint32_t colorOf(const Rgb& c) {
        for (size_t i = 0; i < palette.size(); ++i)
            if (palette[i] == c) return uint32_t(i);
        palette.push_back(c);
        return uint32_t(palette.size() - 1);
    }
    void sample(double x, double y, double weight, uint32_t c) {
        if (!(weight > 0)) return;
        samples.push_back({x, y, weight, c});
    }
    void append(const Canvas& other) {
        std::vector<uint32_t> map(other.palette.size());
        for (size_t i = 0; i < map.size(); ++i) map[i] = colorOf(other.palette[i]);
        const size_t at = samples.size();
        samples.insert(samples.end(), other.samples.begin(), other.samples.end());
        for (size_t i = at; i < samples.size(); ++i) samples[i].color = map[samples[i].color];
        const size_t from = blocks.size();
        blocks.insert(blocks.end(), other.blocks.begin(), other.blocks.end());
        for (size_t i = from; i < blocks.size(); ++i) blocks[i].color = map[blocks[i].color];
    }
    void line(double ax, double ay, double bx, double by, double width, const Rgb& rgb, double alpha) {
        const double dx = bx - ax, dy = by - ay, length = std::hypot(dx, dy);
        if (!(length > 1e-6) || !(alpha > 0) || !(width > 0)) return;
        const uint32_t c = colorOf(rgb);
        const long long ns = std::max<long long>(1, (long long)std::ceil(length / 0.5));
        const long long na = std::max<long long>(1, (long long)std::nearbyint(width / 0.5));
        const double nx = -dy / length, ny = dx / length;
        const double weight = length / double(ns) * width / double(na) * alpha;
        for (long long idx = 0; idx < ns * na; ++idx) {
            const double along = (double(idx / na) + 0.5) / double(ns);
            const double across = ((double(idx % na) + 0.5) / double(na) - 0.5) * std::max(width, 1.0);
            sample(ax + dx * along + nx * across, ay + dy * along + ny * across, weight, c);
        }
    }
    void polyline(std::vector<std::array<double, 2>> pts, double width, const Rgb& c, double alpha, bool closed) {
        if (closed && pts.size() > 2) pts.push_back(pts[0]);
        for (size_t i = 0; i + 1 < pts.size(); ++i) line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], width, c, alpha);
    }
    void rect(double ax, double ay, double bx, double by, const Rgb& rgb, double alpha) {
        const uint32_t c = colorOf(rgb);
        const double sx = std::max(bx - ax, 0.0), sy = std::max(by - ay, 0.0);
        const long long kx = std::max<long long>(1, (long long)std::ceil(sx / 0.5)), ky = std::max<long long>(1, (long long)std::ceil(sy / 0.5));
        const double area = sx * sy / double(kx * ky) * alpha;
        if (!(area > 0) || !std::isfinite(ax) || !std::isfinite(ay) || !std::isfinite(area)) return;
        blocks.push_back({ax, ay, sx, sy, area, kx, ky, c});
    }
    static void blockTaps(double a, double s, long long k, int& lo, std::vector<double>& taps) {
        const double first = std::floor(a + s * (0.5 / double(k)) - 0.5);
        const double last = std::floor(a + s * ((double(k - 1) + 0.5) / double(k)) - 0.5);
        lo = int(first);
        taps.assign(size_t(last - first) + 2, 0.0);
        for (long long i = 0; i < k; ++i) {
            const double p = a + s * ((double(i) + 0.5) / double(k)) - 0.5, p0 = std::floor(p), f = p - p0;
            const size_t at = size_t(p0 - first);
            taps[at] += 1 - f;
            taps[at + 1] += f;
        }
    }
    void disc(double cx, double cy, double r, const Rgb& rgb, double alpha) {
        if (!(r > 0) || !(alpha > 0)) return;
        const uint32_t c = colorOf(rgb);
        if (r < 0.5) { sample(cx, cy, 3.14159265358979323846 * r * r * alpha, c); return; }
        const long long k = 2 * (long long)std::ceil(r / 0.5);
        for (long long idx = 0; idx < k * k; ++idx) {
            const double ox = (double(idx % k) + 0.5) * 0.5 - double(k) * 0.5 / 2, oy = (double(idx / k) + 0.5) * 0.5 - double(k) * 0.5 / 2;
            sample(cx + ox, cy + oy, 0.5 * 0.5 * alpha * (ox * ox + oy * oy <= r * r ? 1.0 : 0.0), c);
        }
    }
    template <class Image>
    void compose(Image& img) const {
        const size_t n = size_t(h) * w;
        cw::Pooled<ComposeStore> store;
        if (store.item.cov.size() < n || store.item.cov.size() > 2 * n) {
            store.item.cov.assign(n, 0.0);
            store.item.sum.assign(3 * n, 0.0);
            store.item.cov.shrink_to_fit();
            store.item.sum.shrink_to_fit();
        }
        std::vector<double>& cov = store.item.cov;
        std::vector<double>& sum = store.item.sum;
        const int ddy[4] = {0, 0, 1, 1}, ddx[4] = {0, 1, 0, 1};
        const int band = 16, bands = (h + 1) / band + 2;
        std::vector<int> bandOf(samples.size());
        std::vector<size_t> start(bands + 1, 0);
        for (size_t s = 0; s < samples.size(); ++s) {
            const double y0 = std::floor(samples[s].y);
            bandOf[s] = y0 >= -1 && y0 < h ? (int(y0) + 1) / band : -1;
            if (bandOf[s] >= 0) ++start[bandOf[s] + 1];
        }
        for (int b = 0; b < bands; ++b) start[b + 1] += start[b];
        std::vector<size_t> order(start[bands]);
        std::vector<size_t> fillAt(start.begin(), start.end() - 1);
        for (size_t s = 0; s < samples.size(); ++s)
            if (bandOf[s] >= 0) order[fillAt[bandOf[s]]++] = s;
        for (int parity = 0; parity < 2; ++parity)
            parallelRows(0, (bands + 1) / 2, [&](int half) {
                const int b = half * 2 + parity;
                if (b >= bands) return;
                for (int tap = 0; tap < 4; ++tap)
                    for (size_t o = start[b]; o < start[b + 1]; ++o) {
                        const Sample& sm = samples[order[o]];
                        const double x0 = std::floor(sm.x), y0 = std::floor(sm.y), fx = sm.x - x0, fy = sm.y - y0;
                        const double k = tap == 0 ? (1 - fx) * (1 - fy) : tap == 1 ? fx * (1 - fy) : tap == 2 ? (1 - fx) * fy : fx * fy;
                        const double xi = x0 + ddx[tap], yi = y0 + ddy[tap];
                        if (!(xi >= 0 && xi < w && yi >= 0 && yi < h)) continue;
                        const size_t i = size_t(yi) * w + size_t(xi);
                        const double v = sm.wt * k;
                        const Rgb& rgb = palette[sm.color];
                        cov[i] += v;
                        for (int c = 0; c < 3; ++c) sum[c * n + i] += v * rgb[c];
                    }
            });
        const size_t nb = blocks.size();
        std::vector<int> bx(nb), by(nb);
        std::vector<std::vector<double>> wx(nb), wy(nb);
        parallelRows(0, int(nb), [&](int b) {
            blockTaps(blocks[b].ax, blocks[b].sx, blocks[b].kx, bx[b], wx[b]);
            blockTaps(blocks[b].ay, blocks[b].sy, blocks[b].ky, by[b], wy[b]);
        });
        std::vector<size_t> rowStart(size_t(h) + 1, 0);
        const auto rowsOf = [&](size_t b, auto&& fn) {
            for (int y = std::max(0, by[b]); y < std::min(h, by[b] + int(wy[b].size())); ++y) fn(y);
        };
        for (size_t b = 0; b < nb; ++b) rowsOf(b, [&](int y) { ++rowStart[size_t(y) + 1]; });
        for (int y = 0; y < h; ++y) rowStart[size_t(y) + 1] += rowStart[y];
        std::vector<int> rowBlocks(rowStart[h]);
        std::vector<size_t> rowFill(rowStart.begin(), rowStart.end() - 1);
        for (size_t b = 0; b < nb; ++b) rowsOf(b, [&](int y) { rowBlocks[rowFill[y]++] = int(b); });
        parallelRows(0, h, [&](int y) {
            for (size_t o = rowStart[y]; o < rowStart[size_t(y) + 1]; ++o) {
                const int b = rowBlocks[o];
                const double ty = wy[b][size_t(y - by[b])];
                if (ty == 0) continue;
                const Rgb& rgb = palette[blocks[b].color];
                for (size_t k = 0; k < wx[b].size(); ++k) {
                    const int x = bx[b] + int(k);
                    if (x < 0 || x >= w || wx[b][k] == 0) continue;
                    const size_t i = size_t(y) * w + size_t(x);
                    const double v = blocks[b].area * (wx[b][k] * ty);
                    cov[i] += v;
                    for (int c = 0; c < 3; ++c) sum[c * n + i] += v * rgb[c];
                }
            }
        });
        std::vector<char> touched(h, 0);
        for (int b = 0; b < bands; ++b)
            if (start[b + 1] > start[b])
                for (int y = std::max(0, b * band - 1); y < std::min(h, b * band + band); ++y) touched[y] = 1;
        for (int y = 0; y < h; ++y)
            if (rowStart[size_t(y) + 1] > rowStart[y]) touched[y] = 1;
        parallelRows(0, h, [&](int y) {
            if (!touched[y]) return;
            for (int x = 0; x < w; ++x) {
                const size_t i = size_t(y) * w + x;
                if (cov[i] == 0) continue;
                const double alpha = std::min(cov[i], 1.0);
                const float cover = float(alpha);
                for (int c = 0; c < 3; ++c) {
                    const float premul = float(sum[c * n + i] / std::max(cov[i], 1e-9) * alpha);
                    float& v = img.at(c, y, x);
                    v = std::clamp(premul + v * (1 - cover), 0.f, 1.f);
                    sum[c * n + i] = 0;
                }
                cov[i] = 0;
            }
        });
    }
};

inline void overAndStore(TopDown& img, const Canvas& canvas, const ImageView& exchange) {
    canvas.compose(img);
    img.store(exchange);
}

struct Box { double x0, y0, x1, y1; };

inline std::vector<Box> bsp(int w, int h, int depth, double fx, double fy, double avoid, long long seed, double minSide) {
    std::vector<Box> cells{{0, 0, double(w), double(h)}};
    for (int level = 0; level < depth; ++level) {
        std::vector<Box> next;
        for (size_t i = 0; i < cells.size(); ++i) {
            const Box b = cells[i];
            const bool wide = (b.x1 - b.x0) >= (b.y1 - b.y0);
            const double lo = wide ? b.x0 : b.y0, hi = wide ? b.x1 : b.y1;
            if (hi - lo < 2 * minSide) { next.push_back(b); continue; }
            const double r = 0.3 + 0.4 * phaseOf(int64_t(level) * 1000 + int64_t(i), seed, 41);
            double cut = lo + (hi - lo) * r;
            const double at = wide ? fx : fy;
            const bool inside = wide ? (b.y0 <= fy && fy <= b.y1) : (b.x0 <= fx && fx <= b.x1);
            if (inside && std::fabs(cut - at) < avoid) {
                const double r0 = lo + minSide, r1 = hi - minSide;
                std::vector<double> options;
                for (double c : {at - avoid, at + avoid})
                    if (r0 <= c && c <= r1) options.push_back(c);
                if (options.empty()) { next.push_back(b); continue; }
                double best = options[0];
                for (double c : options)
                    if (std::fabs(c - cut) < std::fabs(best - cut)) best = c;
                cut = best;
            }
            if (wide) { next.push_back({b.x0, b.y0, cut, b.y1}); next.push_back({cut, b.y0, b.x1, b.y1}); }
            else { next.push_back({b.x0, b.y0, b.x1, cut}); next.push_back({b.x0, cut, b.x1, b.y1}); }
        }
        cells = next;
    }
    return cells;
}

inline void gridSystem(const ImageView& exchange, long long seed) {
    TopDown img(exchange);
    const TopDown source = img;
    const int w = img.w, h = img.h, shortSide = std::min(h, w);
    const Rgb color = hexColor("#f4f1ea");
    const double line = std::max(0.5, 0.0015 * shortSide);
    const auto cells = bsp(w, h, 3, 0.5 * w, 0.45 * h, 0.12 * shortSide, seed, 0.06 * shortSide);
    const double shares[6] = {0.5, 0.0, 0.2, 0.0, 0.3, 0.0};
    double mix[6], acc = 0;
    for (int k = 0; k < 6; ++k) mix[k] = acc += shares[k];
    const double gutter = 0.006 * shortSide / 2;
    Canvas canvas(h, w);
    for (size_t i = 0; i < cells.size(); ++i) {
        const Box b = cells[i];
        const double r = phaseOf(int64_t(i) * 31, seed, 43);
        int kind = 0;
        while (kind < 6 && mix[kind] <= r * mix[5]) ++kind;
        const int ix0 = int(b.x0 + gutter), iy0 = int(b.y0 + gutter), ix1 = int(b.x1 - gutter), iy1 = int(b.y1 - gutter);
        if (ix1 <= ix0 || iy1 <= iy0) continue;
        if (kind == 2) {
            for (int c = 0; c < 3; ++c) {
                double sum = 0;
                for (int Y = iy0; Y < iy1; ++Y)
                    for (int X = ix0; X < ix1; ++X) sum += source.rgb[(size_t(c) * h + Y) * w + X];
                const float mean = float(sum / (double(iy1 - iy0) * (ix1 - ix0)));
                for (int Y = iy0; Y < iy1; ++Y)
                    for (int X = ix0; X < ix1; ++X) img.at(c, Y, X) = mean;
            }
        }
        canvas.polyline({{ix0 - 0.5, iy0 - 0.5}, {ix1 - 0.5, iy0 - 0.5}, {ix1 - 0.5, iy1 - 0.5}, {ix0 - 0.5, iy1 - 0.5}},
                        line * (kind == 4 ? 1.6 : 1.0), color, kind == 4 ? 0.9 : 0.5, true);
    }
    overAndStore(img, canvas, exchange);
}

inline const char* glyphBits(char ch) {
    switch (ch) {
    case '0': return "111101101101111"; case '1': return "010110010010111"; case '2': return "111001111100111";
    case '3': return "111001111001111"; case '4': return "101101111001001"; case '5': return "111100111001111";
    case '6': return "111100111101111"; case '7': return "111001001001001"; case '8': return "111101111101111";
    case '9': return "111101111001111"; case 'A': return "010101111101101"; case 'B': return "110101110101110";
    case 'C': return "111100100100111"; case 'D': return "110101101101110"; case 'E': return "111100111100111";
    case 'F': return "111100111100100"; case 'X': return "000000101010101"; case ':': return "000010000010000";
    case '.': return "000000000000010"; case '-': return "000000111000000"; case '+': return "000010111010000";
    case '/': return "001001010100100"; case '#': return "101111101111101"; case '=': return "000111000111000";
    default: return nullptr;
    }
}

inline void text(Canvas& canvas, const std::string& s, double x0, double y0, double px, const Rgb& c, double alpha) {
    for (size_t i = 0; i < s.size(); ++i) {
        const char* bits = glyphBits(char(std::toupper((unsigned char)s[i])));
        if (!bits) continue;
        for (int k = 0; k < 15; ++k)
            if (bits[k] == '1') {
                const double ax = x0 + double(4 * int(i) + k % 3) * px, ay = y0 + double(k / 3) * px;
                canvas.rect(ax, ay, ax + px, ay + px, c, alpha);
            }
    }
}

inline void portraitWindow(const ImageView& exchange) {
    TopDown img(exchange);
    const int w = img.w, h = img.h, shortSide = std::min(h, w);
    const double cx = 0.5 * w, cy = 0.42 * h, bw = 0.38 * shortSide, bh = 0.46 * shortSide;
    const int x0 = int(std::max(0.0, cx - bw / 2)), x1 = int(std::min(double(w), cx + bw / 2));
    const int y0 = int(std::max(0.0, cy - bh / 2)), y1 = int(std::min(double(h), cy + bh / 2));
    const float dimmed = float(1 - 0.3);
    for (int c = 0; c < 3; ++c) {
        cw::Plane plane(h, w);
        for (int Y = 0; Y < h; ++Y)
            for (int X = 0; X < w; ++X) plane.at(Y, X) = img.at(c, Y, X);
        const cw::Plane blurred = fastBlur(plane, float(0.02 * shortSide));
        parallelRows(0, h, [&](int Y) {
            for (int X = 0; X < w; ++X)
                if (!(x1 > x0 && y1 > y0 && Y >= y0 && Y < y1 && X >= x0 && X < x1)) img.at(c, Y, X) = blurred.at(Y, X) * dimmed;
        });
    }
    Canvas canvas(h, w);
    const Rgb color = hexColor("#f4f1ea");
    const double line = std::max(0.6, 0.0015 * shortSide);
    const std::array<double, 2> box[4] = {{x0 - 0.5, y0 - 0.5}, {x1 - 0.5, y0 - 0.5}, {x1 - 0.5, y1 - 0.5}, {x0 - 0.5, y1 - 0.5}};
    canvas.polyline({box[0], box[1], box[2], box[3]}, line, color, 0.9, true);
    const double tick = 0.03 * shortSide;
    const int signs[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    for (int k = 0; k < 4; ++k) {
        const double bx = box[k][0], by = box[k][1];
        canvas.line(bx, by, bx + signs[k][0] * tick, by, 2 * line, color, 1.0);
        canvas.line(bx, by, bx, by + signs[k][1] * tick, 2 * line, color, 1.0);
    }
    const double px = std::max(1.0, 0.005 * shortSide);
    char label[32];
    std::snprintf(label, sizeof(label), "%04d:%04d", x0, y0);
    text(canvas, label, double(x0), double(y1) + 2 * px, px, color, 0.9);
    overAndStore(img, canvas, exchange);
}

struct FieldMaps {
    int q = 1, fh = 0, fw = 0;
    cw::Plane y;
    std::vector<float> gx, gy;
};

inline cw::Plane lumaOf(TopDown& img) {
    cw::Plane luma(img.h, img.w);
    parallelRows(0, img.h, [&](int Y) {
        for (int X = 0; X < img.w; ++X) luma.at(Y, X) = 0.2126f * img.at(0, Y, X) + 0.7152f * img.at(1, Y, X) + 0.0722f * img.at(2, Y, X);
    });
    return luma;
}

inline cw::Plane lumaOf(const ImageView& exchange) {
    const int w = exchange.bounds.w(), h = exchange.bounds.h();
    cw::Plane luma(h, w);
    parallelRows(0, h, [&](int Y) {
        const float* row = exchange.px(0, h - 1 - Y);
        for (int X = 0; X < w; ++X) {
            const float* c = row + 4 * X;
            const float a = std::max(c[3], 1e-6f);
            luma.at(Y, X) = 0.2126f * std::clamp(c[0] / a, 0.f, 1.f) + 0.7152f * std::clamp(c[1] / a, 0.f, 1.f) + 0.0722f * std::clamp(c[2] / a, 0.f, 1.f);
        }
    });
    return luma;
}

inline FieldMaps fieldMaps(const cw::Plane& luma) {
    FieldMaps f;
    const int h = luma.h, w = luma.w, shortSide = std::min(h, w);
    f.q = std::max(1, pyRound(double(shortSide) / 540.0));
    cw::Plane down = luma;
    if (f.q > 1) {
        down = cw::Plane(h / f.q, w / f.q);
        parallelRows(0, down.h, [&](int y) {
            for (int x = 0; x < down.w; ++x) {
                double sum = 0;
                for (int dy = 0; dy < f.q; ++dy)
                    for (int dx = 0; dx < f.q; ++dx) sum += luma.at(y * f.q + dy, x * f.q + dx);
                down.at(y, x) = float(sum / (double(f.q) * f.q));
            }
        });
    }
    f.y = fastBlur(down, 0.7f);
    f.fh = f.y.h;
    f.fw = f.y.w;
    sobel(f.y.v, f.fh, f.fw, f.gx, f.gy);
    return f;
}

inline FieldMaps fieldMaps(TopDown& img) { return fieldMaps(lumaOf(img)); }

inline void curtain(const ImageView& exchange) {
    TopDown img(exchange);
    const int w = img.w, h = img.h, shortSide = std::min(h, w);
    const FieldMaps fld = fieldMaps(img);
    const int q = fld.q;
    const double length = 0.35 * h * (1 + 0.8 * 0.0);
    const int gap = std::max(1, pyRound((1.05 - 0.5) * 0.03 * shortSide));
    const int stripe = std::max(1, pyRound(gap * 0.8));
    std::vector<float> lum(size_t(w) * h);
    for (int Y = 0; Y < h; ++Y)
        for (int X = 0; X < w; ++X) lum[size_t(Y) * w + X] = 0.2126f * img.at(0, Y, X) + 0.7152f * img.at(1, Y, X) + 0.0722f * img.at(2, Y, X);
    std::vector<float> streak(size_t(3) * w * h, 0.f);
    const size_t n = size_t(w) * h;
    for (int x0 = gap / 2; x0 < w; x0 += gap) {
        const int fx = std::clamp(int(std::nearbyint((x0 + 0.5) / q - 0.5)), 0, fld.fw - 1);
        int row = 0;
        float best = std::fabs(fld.gy[size_t(0) * fld.fw + fx]);
        for (int r = 1; r < fld.fh; ++r) {
            const float v = std::fabs(fld.gy[size_t(r) * fld.fw + fx]);
            if (v > best) { best = v; row = r; }
        }
        if (!(best > 0.04f)) continue;
        const double ry = std::clamp((row + 0.5) * q - 0.5, 0.0, double(h - 1));
        const int above = std::clamp(int(std::nearbyint(ry - 1.5 * q)), 0, h - 1);
        const int under = std::clamp(int(std::nearbyint(ry + 1.5 * q)), 0, h - 1);
        const int lit = lum[size_t(above) * w + x0] >= lum[size_t(under) * w + x0] ? above : under;
        float col[3];
        for (int c = 0; c < 3; ++c) col[c] = img.at(c, lit, x0);
        const double inten = 0.8 * (1 - 0.35 * 0.0);
        const int c0 = std::clamp(x0 - stripe / 2, 0, w - 1), c1 = std::min(w, c0 + stripe);
        for (int Y = 0; Y < h; ++Y) {
            const double d = double(Y) - ry;
            const double fade = std::pow(std::max(1 - d / std::max(length, 1.0), 0.0), 1.5);
            const double prof = (d >= 0 && d <= length ? fade : 0.0) * inten;
            for (int c = 0; c < 3; ++c) {
                const float v = float(prof * double(col[c]));
                for (int X = c0; X < c1; ++X) {
                    float& s = streak[c * n + size_t(Y) * w + X];
                    s = std::max(s, v);
                }
            }
        }
    }
    parallelRows(0, h, [&](int Y) {
        for (int X = 0; X < w; ++X)
            for (int c = 0; c < 3; ++c) {
                float& v = img.at(c, Y, X);
                const float base = v * float(1 - 0.0);
                v = std::clamp(1 - (1 - base) * (1 - streak[c * n + size_t(Y) * w + X]), 0.f, 1.f);
            }
    });
    img.store(exchange);
}

inline void isoSegments(const cw::Plane& v, float level, int q, std::vector<std::array<double, 4>>& out) {
    const int ch = v.h - 1, cwid = v.w - 1;
    if (ch <= 0 || cwid <= 0) return;
    std::vector<std::vector<int>> bucket(30);
    for (int y = 0; y < ch; ++y)
        for (int x = 0; x < cwid; ++x) {
            const float v0 = v.at(y, x), v1 = v.at(y, x + 1), v2 = v.at(y + 1, x + 1), v3 = v.at(y + 1, x);
            const int c = (v0 > level) + (v1 > level) * 2 + (v2 > level) * 4 + (v3 > level) * 8;
            if (c == 0 || c == 15) continue;
            const bool centre = (v0 + v1 + v2 + v3) / 4.f > level;
            bucket[c * 2 + (centre ? 1 : 0)].push_back(y * cwid + x);
        }
    static const int cases[15][2][2] = {
        {}, {{3, 0}}, {{0, 1}}, {{3, 1}}, {{1, 2}}, {}, {{0, 2}}, {{3, 2}},
        {{2, 3}}, {{0, 2}}, {}, {{1, 2}}, {{1, 3}}, {{0, 1}}, {{3, 0}}};
    static const int saddle[2][2][2][2] = {
        {{{3, 0}, {1, 2}}, {{0, 1}, {2, 3}}},
        {{{0, 1}, {2, 3}}, {{3, 0}, {1, 2}}}};
    auto cross = [&](float a, float b) -> double {
        const float d = b - a;
        return std::fabs(d) > 1e-12f ? double((level - a) / d) : 0.5;
    };
    auto point = [&](int cell, int e, double& px, double& py) {
        const int y = cell / cwid, x = cell % cwid;
        const float v0 = v.at(y, x), v1 = v.at(y, x + 1), v2 = v.at(y + 1, x + 1), v3 = v.at(y + 1, x);
        if (e == 0) { px = x + cross(v0, v1); py = y; }
        else if (e == 1) { px = x + 1; py = y + cross(v1, v2); }
        else if (e == 2) { px = x + cross(v3, v2); py = y + 1; }
        else { px = x; py = y + cross(v0, v3); }
    };
    auto emit = [&](const std::vector<int>& cells, int e0, int e1) {
        for (int cell : cells) {
            double ax, ay, bx, by;
            point(cell, e0, ax, ay);
            point(cell, e1, bx, by);
            if (!(std::max(std::fabs(ax - bx), std::fabs(ay - by)) > 1e-12)) continue;
            out.push_back({(ax + 0.5) * q - 0.5, (ay + 0.5) * q - 0.5, (bx + 0.5) * q - 0.5, (by + 0.5) * q - 0.5});
        }
    };
    for (int c = 1; c < 15; ++c) {
        if (c == 5 || c == 10) {
            const int s = c == 5 ? 0 : 1;
            for (int above = 1; above >= 0; --above)
                for (int k = 0; k < 2; ++k) emit(bucket[c * 2 + above], saddle[s][above][k][0], saddle[s][above][k][1]);
            continue;
        }
        std::vector<int> cells = bucket[c * 2];
        cells.insert(cells.end(), bucket[c * 2 + 1].begin(), bucket[c * 2 + 1].end());
        std::sort(cells.begin(), cells.end());
        emit(cells, cases[c][0][0], cases[c][0][1]);
    }
}

inline void contours(const ImageView& exchange) {
    TopDown img(exchange);
    const int w = img.w, h = img.h, shortSide = std::min(h, w);
    const FieldMaps fld = fieldMaps(img);
    const int q = fld.q;
    const double width = std::max(0.35, 0.0015 * shortSide);
    const cw::Plane luma = fastBlur(fld.y, float(std::max(0.0, (1 - 0.5) * 0.012 * shortSide / q)));
    const Rgb color = hexColor("#f4f1ea");
    Canvas canvas(h, w);
    const int n = 12;
    std::vector<Canvas> levels(n, Canvas(h, w));
    parallelRows(0, n, [&](int k) {
        std::vector<std::array<double, 4>> segs;
        isoSegments(luma, float((k + 0.5) / n), q, segs);
        const double lw = width * (k % 4 == 3 ? 1.8 : 1.0);
        for (const auto& s : segs) levels[k].line(s[0], s[1], s[2], s[3], lw, color, 1.0);
    });
    for (const Canvas& level : levels) canvas.append(level);
    const float dimmed = float(1 - 0.2);
    parallelRows(0, h, [&](int Y) {
        for (int X = 0; X < w; ++X)
            for (int c = 0; c < 3; ++c) img.at(c, Y, X) *= dimmed;
    });
    overAndStore(img, canvas, exchange);
}

inline std::vector<int> spaced(const std::vector<std::array<double, 2>>& xy, double gap, int limit) {
    std::vector<int> out;
    const double g2 = gap * gap;
    for (int i = 0; i < int(xy.size()); ++i) {
        if (int(out.size()) >= limit) break;
        if (!out.empty()) {
            double best = INFINITY;
            for (int j : out) {
                const double dx = xy[j][0] - xy[i][0], dy = xy[j][1] - xy[i][1];
                best = std::min(best, dx * dx + dy * dy);
            }
            if (best < g2) continue;
        }
        out.push_back(i);
    }
    return out;
}

inline float parabolaPeak(float a, float b, float c) {
    const float d = a - 2 * b + c;
    if (!(d < -1e-9f)) return 0.f;
    return std::clamp(0.5f * (a - c) / d, -0.5f, 0.5f);
}

struct GeoField : FieldMaps {
    int h = 0, w = 0;
    double cell = 0;
    std::vector<float> mag, score;
    std::vector<std::array<double, 2>> xy;
    std::vector<double> s;
    std::vector<long long> keys;
    std::vector<int> anchors;
    void refine(int iy, int ix, double& sx, double& sy) const {
        const size_t i = size_t(iy) * fw + ix;
        const bool alongX = std::fabs(gx[i]) >= std::fabs(gy[i]);
        const float ox = alongX ? parabolaPeak(mag[i - 1], mag[i], mag[i + 1]) : 0.f;
        const float oy = alongX ? 0.f : parabolaPeak(mag[i - fw], mag[i], mag[i + fw]);
        sx = (double(ix) + double(ox) + 0.5) * q - 0.5;
        sy = (double(iy) + double(oy) + 0.5) * q - 0.5;
    }
};

inline GeoField geoField(const cw::Plane& luma, int maxNodes) {
    GeoField f;
    static_cast<FieldMaps&>(f) = fieldMaps(luma);
    f.h = luma.h;
    f.w = luma.w;
    const int shortSide = std::min(f.h, f.w);
    f.cell = shortSide * (0.12 - 0.09 * 0.5);
    const double sigma = std::max(1.0, f.cell / f.q / 3);
    const cw::Plane mean = fastBlur(f.y, float(sigma));
    cw::Plane yy = f.y;
    for (float& v : yy.v) v = v * v;
    const cw::Plane m2 = fastBlur(yy, float(sigma));
    const size_t n = size_t(f.fh) * f.fw;
    f.mag.resize(n);
    f.score.resize(n);
    parallelRows(0, f.fh, [&](int r) {
        for (size_t i = size_t(r) * f.fw; i < size_t(r + 1) * f.fw; ++i) {
            f.mag[i] = std::hypot(f.gx[i], f.gy[i]);
            const float contrast = std::sqrt(std::max(m2.v[i] - mean.v[i] * mean.v[i], 0.f));
            f.score[i] = f.mag[i] + 0.3f * contrast;
        }
    });
    struct Cand { long long key; double sc; int iy, ix; };
    const int q = f.q;
    std::vector<std::vector<Cand>> rowBest(f.fh);
    parallelRows(1, std::max(1, f.fh - 1), [&](int iy) {
        auto& out = rowBest[iy];
        for (int ix = 1; ix < f.fw - 1; ++ix) {
            const size_t i = size_t(iy) * f.fw + ix;
            if (!(f.mag[i] > 0.04f)) continue;
            const double gxs = (ix + 0.5) * q - 0.5, gys = (iy + 0.5) * q - 0.5;
            const long long cu = (long long)std::floor(gxs / f.cell), cv = (long long)std::floor(gys / f.cell);
            const long long key = (cv + (1LL << 20)) * (1LL << 21) + (cu + (1LL << 20));
            const double sc = double(f.score[i]) + std::fmod((gxs * 0.618034 + gys * 0.381966) / f.cell, 1.0) * 1e-6;
            if (!out.empty() && out.back().key == key) {
                if (sc > out.back().sc) out.back() = {key, sc, iy, ix};
            } else {
                out.push_back({key, sc, iy, ix});
            }
        }
    });
    std::map<long long, Cand> best;
    for (const auto& row : rowBest)
        for (const Cand& c : row) {
            const auto [it, added] = best.emplace(c.key, c);
            if (!added && c.sc > it->second.sc) it->second = c;
        }
    std::vector<Cand> win;
    for (const auto& kv : best) win.push_back(kv.second);
    std::vector<int> order(win.size());
    for (size_t i = 0; i < win.size(); ++i) order[i] = int(i);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return -f.score[size_t(win[a].iy) * f.fw + win[a].ix] < -f.score[size_t(win[b].iy) * f.fw + win[b].ix];
    });
    std::vector<std::array<double, 2>> xy(win.size());
    for (size_t k = 0; k < order.size(); ++k) {
        const Cand& c = win[order[k]];
        f.refine(c.iy, c.ix, xy[k][0], xy[k][1]);
    }
    for (int k : spaced(xy, 0.7 * f.cell, maxNodes)) {
        const Cand& c = win[order[k]];
        f.xy.push_back(xy[k]);
        f.s.push_back(double(f.score[size_t(c.iy) * f.fw + c.ix]));
        f.keys.push_back(c.key);
    }
    f.anchors = spaced(f.xy, 0.18 * shortSide, 7);
    return f;
}

inline double bilinearAt(const cw::Plane& a, double x, double y) {
    const int h = a.h, w = a.w;
    x = std::clamp(x, 0.0, double(w - 1));
    y = std::clamp(y, 0.0, double(h - 1));
    const int x0 = w > 1 ? std::min(int(std::floor(x)), w - 2) : 0, y0 = h > 1 ? std::min(int(std::floor(y)), h - 2) : 0;
    const double fx = x - x0, fy = y - y0;
    const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
    return (a.at(y0, x0) * (1 - fx) + a.at(y0, x1) * fx) * (1 - fy) + (a.at(y1, x0) * (1 - fx) + a.at(y1, x1) * fx) * fy;
}

struct GeoLayer {
    std::vector<std::array<double, 2>> xy;
    std::vector<double> s;
    std::vector<long long> keys;
    std::vector<int> anchors;
    double weight = 1;
    int born = 0;
    std::vector<std::array<int, 2>> pairs;
    bool linked = false;
};

struct GeoState {
    int index = -2147483647, h = 0, w = 0;
    bool hasLayers = false;
    GeoLayer layers[2];
};

inline uint64_t regionHash(const ImageView& view, const Rect& region) {
    const int w = region.w(), h = region.h();
    std::vector<uint64_t> rows(h);
    parallelRows(0, h, [&](int y) {
        const uint32_t* p = reinterpret_cast<const uint32_t*>(view.px(region.x1, region.y1 + y));
        uint64_t k = 1469598103934665603ull;
        for (size_t i = 0; i < size_t(w) * 4; ++i) {
            k ^= p[i];
            k *= 1099511628211ull;
        }
        rows[y] = k;
    });
    uint64_t k = 1469598103934665603ull ^ (uint64_t(uint32_t(w)) << 32 | uint32_t(h));
    for (uint64_t r : rows) {
        k ^= r;
        k *= 1099511628211ull;
        k ^= k >> 29;
    }
    return k;
}

inline uint64_t frameHash(const ImageView& exchange) { return regionHash(exchange, exchange.bounds); }

struct GeoFrame {
    int w = 0, h = 0, index = 0;
    uint64_t hash = 0, prevHash = 0;
    bool linked = false, cut = false;
    GeoField fld;
    cw::Plane fy, fx;
};

inline std::mutex& geoFramesLock() {
    static std::mutex m;
    return m;
}

inline std::vector<std::shared_ptr<const GeoFrame>>& geoFrames() {
    static std::vector<std::shared_ptr<const GeoFrame>> v;
    return v;
}

inline std::shared_ptr<const GeoFrame> findGeoFrame(int w, int h, int index, uint64_t hash) {
    std::lock_guard<std::mutex> guard(geoFramesLock());
    for (const auto& f : geoFrames())
        if (f->w == w && f->h == h && f->index == index && f->hash == hash) return f;
    return nullptr;
}

inline size_t geoFrameBytes(const GeoFrame& f) {
    return sizeof(float) * (f.fy.v.size() + f.fx.v.size() + f.fld.mag.size() + f.fld.gx.size() + f.fld.gy.size());
}

constexpr size_t geoFrameBudget = size_t(512) << 20;

inline void keepGeoFrame(const std::shared_ptr<const GeoFrame>& frame) {
    std::lock_guard<std::mutex> guard(geoFramesLock());
    auto& frames = geoFrames();
    frames.erase(std::remove_if(frames.begin(), frames.end(), [&](const std::shared_ptr<const GeoFrame>& f) {
        return f->w == frame->w && f->h == frame->h && f->index == frame->index;
    }), frames.end());
    frames.push_back(frame);
    size_t bytes = 0;
    for (const auto& f : frames) bytes += geoFrameBytes(*f);
    while (frames.size() > 16 || (frames.size() > 1 && bytes > geoFrameBudget)) {
        bytes -= geoFrameBytes(*frames.front());
        frames.erase(frames.begin());
    }
}

inline size_t geoFramesHeld() {
    std::lock_guard<std::mutex> guard(geoFramesLock());
    size_t bytes = 0;
    for (const auto& f : geoFrames()) bytes += geoFrameBytes(*f);
    return bytes;
}

struct LumaMemo {
    int w, h, index;
    uint64_t hash;
    cw::Plane luma;
};

inline std::vector<LumaMemo>& lumaMemo() {
    static std::vector<LumaMemo> v;
    return v;
}

inline bool findLuma(int w, int h, int index, uint64_t hash, cw::Plane& out) {
    std::lock_guard<std::mutex> guard(geoFramesLock());
    for (const LumaMemo& m : lumaMemo())
        if (m.w == w && m.h == h && m.index == index && m.hash == hash) {
            out = m.luma;
            return true;
        }
    return false;
}

inline void keepLuma(int w, int h, int index, uint64_t hash, const cw::Plane& luma) {
    std::lock_guard<std::mutex> guard(geoFramesLock());
    auto& memo = lumaMemo();
    memo.erase(std::remove_if(memo.begin(), memo.end(), [&](const LumaMemo& m) { return m.w == w && m.h == h && m.index == index; }), memo.end());
    memo.push_back({w, h, index, hash, luma});
    while (memo.size() > 3) memo.erase(memo.begin());
}

inline std::vector<std::array<int, 2>> delaunayEdges(const std::vector<std::array<double, 2>>& xy);

inline GeoLayer builtLayer(const GeoField& f, int index) { return {f.xy, f.s, f.keys, f.anchors, 1.0, index, {}, false}; }

inline void linkLayer(GeoLayer& layer, double reach) {
    layer.linked = true;
    if (layer.xy.size() < 2) return;
    for (const auto& e : delaunayEdges(layer.xy)) {
        const auto& a = layer.xy[e[0]];
        const auto& b = layer.xy[e[1]];
        if (std::hypot(a[0] - b[0], a[1] - b[1]) <= reach) layer.pairs.push_back(e);
    }
}

inline void carryLayer(GeoLayer& layer, const cw::Plane& fy, const cw::Plane& fx, const GeoField& fld) {
    for (auto& p : layer.xy) {
        const double px = p[0], py = p[1];
        double nx = px, ny = py;
        for (int k = 0; k < 2; ++k) {
            const double ax = px - bilinearAt(fx, nx, ny), ay = py - bilinearAt(fy, nx, ny);
            nx = ax;
            ny = ay;
        }
        const int ix = std::clamp(int(std::nearbyint((nx + 0.5) / fld.q - 0.5)), 1, fld.fw - 2);
        const int iy = std::clamp(int(std::nearbyint((ny + 0.5) / fld.q - 0.5)), 1, fld.fh - 2);
        double sx, sy;
        fld.refine(iy, ix, sx, sy);
        const size_t i = size_t(iy) * fld.fw + ix;
        const bool alongX = std::fabs(fld.gx[i]) >= std::fabs(fld.gy[i]);
        const bool onEdge = fld.mag[i] > 0.04f;
        if (onEdge && alongX) nx = nx + 0.5 * (sx - nx);
        if (onEdge && !alongX) ny = ny + 0.5 * (sy - ny);
        p = {nx, ny};
    }
}

struct GeoRun {
    bool links = false;
    GeoState st;
    cw::Plane luma;
    bool hasLuma = false;
    uint64_t hash = 0;
    std::shared_ptr<const GeoFrame> frame;
};

template <class Fill, class Previous>
inline bool geoAdvance(GeoRun& run, const ImageView& exchange, uint64_t hash, int index, Fill&& fillNow, Previous&& previousLuma) {
    const int w = exchange.bounds.w(), h = exchange.bounds.h();
    std::shared_ptr<const GeoFrame> fr = findGeoFrame(w, h, index, hash);
    std::shared_ptr<GeoFrame> made;
    cw::Plane luma;
    bool hasLuma = false;
    const auto ensureLuma = [&] {
        if (hasLuma) return;
        fillNow();
        luma = lumaOf(exchange);
        hasLuma = true;
        keepLuma(w, h, index, hash, luma);
    };
    if (!fr) {
        ensureLuma();
        made = std::make_shared<GeoFrame>();
        made->w = w;
        made->h = h;
        made->index = index;
        made->hash = hash;
        made->fld = geoField(luma, 160);
        made->fld.y = cw::Plane();
        made->fld.score = {};
        fr = made;
    }
    const bool consecutive = run.st.hasLayers && run.st.index == index - 1 && run.st.w == w && run.st.h == h;
    bool rebuild = !consecutive;
    if (consecutive) {
        if (!(fr->linked && fr->prevHash == run.hash)) {
            ensureLuma();
            cw::Plane fetched;
            const cw::Plane* prev = &run.luma;
            if (!run.hasLuma) {
                if (!findLuma(w, h, index - 1, run.hash, fetched) && !previousLuma(fetched)) return false;
                prev = &fetched;
            }
            std::shared_ptr<GeoFrame> linked = made ? made : std::make_shared<GeoFrame>(*fr);
            if (!gpu::cutwrightFlow(luma, *prev, linked->fy, linked->fx)) return false;
            std::vector<double> rows(h, 0.0);
            parallelRows(0, h, [&](int y) {
                double s = 0;
                for (int x = 0; x < w; ++x)
                    s += std::fabs(luma.at(y, x) - cw::sampleClamp(*prev, float(y) + linked->fy.at(y, x), float(x) + linked->fx.at(y, x)));
                rows[y] = s;
            });
            double sum = 0;
            for (double s : rows) sum += s;
            linked->cut = sum / (double(h) * w) > 0.1;
            linked->prevHash = run.hash;
            linked->linked = true;
            made = linked;
            fr = linked;
        }
        rebuild = fr->cut;
    }
    if (made) keepGeoFrame(made);
    GeoState& st = run.st;
    if (rebuild) {
        st.layers[0] = builtLayer(fr->fld, index);
        st.layers[1] = builtLayer(fr->fld, index);
    } else {
        for (auto& layer : st.layers) carryLayer(layer, fr->fy, fr->fx, fr->fld);
        if (((index % 12) + 12) % 12 == 0) st.layers[0] = builtLayer(fr->fld, index);
        if (((index % 12) + 12) % 12 == 6) st.layers[1] = builtLayer(fr->fld, index);
    }
    const int phase = ((index % 12) + 12) % 12;
    const double wa = 1 - std::fabs(phase - 6.0) / 6.0;
    st.layers[0].weight = wa;
    st.layers[1].weight = 1 - wa;
    if (run.links)
        for (auto& layer : st.layers)
            if (!layer.linked && layer.weight > 0 && !layer.xy.empty()) linkLayer(layer, 0.2 * std::min(w, h));
    st.hasLayers = true;
    st.index = index;
    st.w = w;
    st.h = h;
    run.hash = hash;
    run.frame = fr;
    run.luma = std::move(luma);
    run.hasLuma = hasLuma;
    return true;
}

inline void ring(Canvas& canvas, double cx, double cy, double r, double width, const Rgb& c, double alpha,
                 double a0 = 0, double a1 = 2 * 3.14159265358979323846) {
    const long long m = std::clamp((long long)std::ceil(std::fabs(a1 - a0) * r / 3), 8LL, 256LL);
    for (long long j = 0; j < m; ++j) {
        const double t0 = a0 + (a1 - a0) * double(j) / double(m), t1 = a0 + (a1 - a0) * double(j + 1) / double(m);
        canvas.line(cx + r * std::cos(t0), cy + r * std::sin(t0), cx + r * std::cos(t1), cy + r * std::sin(t1), width, c, alpha);
    }
}

inline void geoFieldDraw(const GeoRun& run, const ImageView& exchange) {
    TopDown img(exchange);
    const int h = img.h, w = img.w, shortSide = std::min(h, w);
    const Rgb color = hexColor("#3a4bff"), accent = hexColor("#ffffff");
    Canvas canvas(h, w);
    for (const GeoLayer& layer : run.st.layers) {
        if (layer.weight <= 0 || layer.xy.empty()) continue;
        for (const auto& p : layer.xy) canvas.disc(p[0], p[1], shortSide * 0.006 * 1.0, color, layer.weight);
        for (int k : layer.anchors)
            ring(canvas, layer.xy[k][0], layer.xy[k][1], shortSide * 0.03 * 1.0, std::max(1.0, shortSide * 0.002), accent, layer.weight);
    }
    const float dimmed = float(1 - 0.35);
    parallelRows(0, h, [&](int Y) {
        for (int X = 0; X < w; ++X)
            for (int c = 0; c < 3; ++c) img.at(c, Y, X) *= dimmed;
    });
    overAndStore(img, canvas, exchange);
}

inline double fieldAt(const std::vector<float>& a, const GeoField& f, double px, double py) {
    const int h = f.fh, w = f.fw;
    double x = std::clamp((px + 0.5) / f.q - 0.5, 0.0, double(w - 1));
    double y = std::clamp((py + 0.5) / f.q - 0.5, 0.0, double(h - 1));
    const int x0 = w > 1 ? std::min(int(std::floor(x)), w - 2) : 0, y0 = h > 1 ? std::min(int(std::floor(y)), h - 2) : 0;
    const double fx = x - x0, fy = y - y0;
    const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
    const auto at = [&](int yy, int xx) { return a[size_t(yy) * w + xx]; };
    return (at(y0, x0) * (1 - fx) + at(y0, x1) * fx) * (1 - fy) + (at(y1, x0) * (1 - fx) + at(y1, x1) * fx) * fy;
}

inline std::vector<std::array<int, 2>> delaunayEdges(const std::vector<std::array<double, 2>>& xy) {
    const int n = int(xy.size());
    if (n < 3) return n == 2 ? std::vector<std::array<int, 2>>{{0, 1}} : std::vector<std::array<int, 2>>{};
    double lo0 = xy[0][0], lo1 = xy[0][1], hi0 = xy[0][0], hi1 = xy[0][1];
    for (const auto& p : xy) {
        lo0 = std::min(lo0, p[0]);
        lo1 = std::min(lo1, p[1]);
        hi0 = std::max(hi0, p[0]);
        hi1 = std::max(hi1, p[1]);
    }
    const double span = std::max(hi0 - lo0, hi1 - lo1) + 1.0, mid0 = (lo0 + hi0) / 2, mid1 = (lo1 + hi1) / 2;
    std::vector<std::array<double, 2>> pts = xy;
    pts.push_back({mid0 - 20 * span, mid1 - span});
    pts.push_back({mid0 + 20 * span, mid1 - span});
    pts.push_back({mid0, mid1 + 20 * span});
    struct Tri { int a, b, c; double cx, cy, r2; };
    const auto make = [&](int ia, int ib, int ic) {
        const auto& a = pts[ia];
        const auto& b = pts[ib];
        const auto& c = pts[ic];
        double d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
        if (std::fabs(d) < 1e-12) d = 1e-12;
        const double sa = a[0] * a[0] + a[1] * a[1], sb = b[0] * b[0] + b[1] * b[1], sc = c[0] * c[0] + c[1] * c[1];
        const double cx = (sa * (b[1] - c[1]) + sb * (c[1] - a[1]) + sc * (a[1] - b[1])) / d;
        const double cy = (sa * (c[0] - b[0]) + sb * (a[0] - c[0]) + sc * (b[0] - a[0])) / d;
        return Tri{ia, ib, ic, cx, cy, (a[0] - cx) * (a[0] - cx) + (a[1] - cy) * (a[1] - cy)};
    };
    std::vector<Tri> tris{make(n, n + 1, n + 2)};
    for (int i = 0; i < n; ++i) {
        const double px = pts[i][0], py = pts[i][1];
        std::vector<Tri> keep;
        std::map<std::array<int, 2>, int> count;
        for (const Tri& t : tris) {
            if ((px - t.cx) * (px - t.cx) + (py - t.cy) * (py - t.cy) < t.r2 * (1 - 1e-12)) {
                for (const auto& e : {std::array<int, 2>{t.a, t.b}, std::array<int, 2>{t.b, t.c}, std::array<int, 2>{t.c, t.a}})
                    ++count[{std::min(e[0], e[1]), std::max(e[0], e[1])}];
            } else {
                keep.push_back(t);
            }
        }
        if (count.empty()) continue;
        for (const auto& [e, k] : count)
            if (k == 1) keep.push_back(make(e[0], e[1], i));
        tris = std::move(keep);
    }
    std::vector<std::array<int, 2>> edges;
    for (const Tri& t : tris) {
        if (t.a >= n || t.b >= n || t.c >= n) continue;
        for (const auto& e : {std::array<int, 2>{t.a, t.b}, std::array<int, 2>{t.b, t.c}, std::array<int, 2>{t.c, t.a}})
            edges.push_back({std::min(e[0], e[1]), std::max(e[0], e[1])});
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    return edges;
}

inline void constellationDraw(const GeoRun& run, const ImageView& exchange, long long seed) {
    TopDown img(exchange);
    const int h = img.h, w = img.w, shortSide = std::min(h, w);
    const GeoField& fld = run.frame->fld;
    const Rgb color = hexColor("#f4f1ea");
    const double width = std::max(0.35, 0.0015 * shortSide), size = 0.03 * shortSide;
    const double pi = 3.14159265358979323846;
    static const char* const kinds[] = {"arc", "bracket", "cross", "diamond", "tick"};
    static const double ladder[] = {1.0, 0.7, 0.45, 0.25};
    Canvas canvas(h, w);
    for (const GeoLayer& layer : run.st.layers) {
        const int n = int(layer.xy.size());
        if (layer.weight <= 0 || n == 0) continue;
        const auto& xy = layer.xy;
        const double la = layer.weight * std::min(1.0, 0.8 * 1.0) * 1.0;
        for (const auto& e : layer.pairs) {
            const auto& a = xy[e[0]];
            const auto& b = xy[e[1]];
            canvas.line(a[0], a[1], a[0] + (b[0] - a[0]) * 1.0, a[1] + (b[1] - a[1]) * 1.0, width, color, la);
        }
        std::vector<int> role(n, 3);
        std::vector<char> isAnchor(n, 0);
        for (int k : layer.anchors) isAnchor[k] = 1;
        std::vector<int> rest;
        for (int k = 0; k < n; ++k)
            if (!isAnchor[k]) rest.push_back(k);
        const size_t secondary = size_t(std::ceil(0.4 * double(rest.size())));
        for (size_t k = 0; k < secondary && k < rest.size(); ++k) role[rest[k]] = 2;
        for (size_t k = 2; k < layer.anchors.size(); ++k) role[layer.anchors[k]] = 1;
        for (size_t k = 0; k < 2 && k < layer.anchors.size(); ++k) role[layer.anchors[k]] = 0;
        for (int i = 0; i < n; ++i) {
            const int kind = int(phaseOf(layer.keys[i], seed, 11) * 5);
            const bool hot = role[i] == 0;
            const Rgb& rgb = color;
            const double lw = hot ? 1.5 * width : width, alpha = layer.weight * 1.0;
            const double r = size * ladder[role[i]] * 1.0, cx = xy[i][0], cy = xy[i][1];
            const std::string name = kinds[kind];
            if (name == "arc") {
                const double a0 = 2 * pi * phaseOf(layer.keys[i], seed, 13), a1 = a0 + 1.5 * pi;
                ring(canvas, cx, cy, r, lw, rgb, alpha, a0, a1);
            } else if (name == "tick") {
                const double gx = fieldAt(fld.gx, fld, cx, cy), gy = fieldAt(fld.gy, fld, cx, cy), norm = std::hypot(gx, gy) + 1e-9;
                const double tx = -gy / norm * r * 0.6, ty = gx / norm * r * 0.6;
                canvas.line(cx - tx, cy - ty, cx + tx, cy + ty, lw, rgb, alpha);
            } else if (name == "cross") {
                const double d = 0.7 * r;
                canvas.line(cx - 1.0 * d, cy - 0.0 * d, cx + 1.0 * d, cy + 0.0 * d, lw, rgb, alpha);
                canvas.line(cx - 0.0 * d, cy - 1.0 * d, cx + 0.0 * d, cy + 1.0 * d, lw, rgb, alpha);
            } else if (name == "diamond") {
                static const int tips[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
                for (int k = 0; k < 4; ++k) {
                    const int* t0 = tips[k];
                    const int* t1 = tips[(k + 1) % 4];
                    canvas.line(cx + t0[0] * r, cy + t0[1] * r, cx + t1[0] * r, cy + t1[1] * r, lw, rgb, alpha);
                }
            } else {
                for (int sx : {-1, 1})
                    for (int sy : {-1, 1}) {
                        const double ox = cx + sx * r, oy = cy + sy * r, arm = 0.45 * r;
                        canvas.line(ox, oy, ox + -sx * arm, oy + 0 * r, lw, rgb, alpha);
                        canvas.line(ox, oy, ox + 0 * r, oy + -sy * arm, lw, rgb, alpha);
                    }
            }
        }
    }
    const float dimmed = float(1 - 0.2);
    parallelRows(0, h, [&](int Y) {
        for (int X = 0; X < w; ++X)
            for (int c = 0; c < 3; ++c) img.at(c, Y, X) *= dimmed;
    });
    overAndStore(img, canvas, exchange);
}

inline std::array<double, 2> emptiest(const std::vector<std::array<double, 2>>& xy, const std::array<double, 2>& at, double reach, int h, int w) {
    const double pi = 3.14159265358979323846, limit = std::cos(pi / 8);
    double best = INFINITY;
    std::array<double, 2> out{1.0, 0.0};
    for (int k = 0; k < 16; ++k) {
        const double th = 2 * pi * k / 16, u0 = std::cos(th), u1 = std::sin(th);
        double count = 0;
        for (const auto& p : xy) {
            const double r0 = p[0] - at[0], r1 = p[1] - at[1], d = std::hypot(r0, r1);
            if (!(d > 1e-6 && d <= reach)) continue;
            if ((r0 * u0 + r1 * u1) / d > limit) count += 1;
        }
        const double e0 = at[0] + u0 * reach, e1 = at[1] + u1 * reach;
        if (!(0 <= e0 && e0 <= w - 1 && 0 <= e1 && e1 <= h - 1)) count += 1000;
        if (count < best) {
            best = count;
            out = {u0, u1};
        }
    }
    return out;
}

inline void pixelMemoryDraw(const GeoRun& run, const ImageView& exchange, long long seed) {
    TopDown img(exchange);
    const int h = img.h, w = img.w, shortSide = std::min(h, w);
    const std::vector<float> source = img.rgb;
    const int s = std::max(2, pyRound(0.02 * shortSide));
    const Cells cells = cellMeans(img, s);
    const std::vector<float> sig = cellSignal(cells, 0.5f);
    const double thr = std::max(0.0, 0.12 - 0.1 * 0.0);
    const float lit = float(thr), strongAt = float(thr + 0.6 * (1 - thr));
    const Rgb color = hexColor("#f4f1ea"), accent = hexColor("#3a4bff"), white{1.0, 1.0, 1.0};
    Canvas grid(h, w);
    const double g = 0.07 * s;
    std::vector<Canvas> rows(cells.hh, Canvas(h, w));
    parallelRows(0, cells.hh, [&](int iy) {
        for (int ix = 0; ix < cells.ww; ++ix) {
            const size_t i = size_t(iy) * cells.ww + ix;
            bool active = sig[i] > lit;
            active ^= phaseOf(int64_t(i) * 7 + 0 * 1000003, seed, 21) < 0.15 * 0.25;
            if (!active) continue;
            const double ax = ix * s + g, ay = iy * s + g;
            rows[iy].rect(ax, ay, ax + s - 2 * g, ay + s - 2 * g, sig[i] > strongAt ? accent : color, 1.0);
        }
    });
    for (const Canvas& row : rows) grid.append(row);
    const float dimmed = float(1 - 0.6);
    parallelRows(0, h, [&](int Y) {
        for (int X = 0; X < w; ++X)
            for (int c = 0; c < 3; ++c) img.at(c, Y, X) *= dimmed;
    });
    grid.compose(img);
    Canvas marks(h, w);
    const double ww = 0.26 * shortSide, wh = 0.17 * shortSide, line = std::max(0.6, 0.0015 * shortSide), zoom = 2.5;
    for (const GeoLayer& layer : run.st.layers) {
        if (!(layer.weight > 0)) continue;
        for (size_t n = 0; n < layer.anchors.size() && n < 2; ++n) {
            const int hub = layer.anchors[n];
            const auto at = layer.xy[hub];
            const auto u = emptiest(layer.xy, at, 0.35 * shortSide, h, w);
            const double cx = std::min(std::max(at[0] + u[0] * 0.24 * shortSide, ww / 2 + 2), w - ww / 2 - 3);
            const double cy = std::min(std::max(at[1] + u[1] * 0.24 * shortSide, wh / 2 + 2), h - wh / 2 - 3);
            const int x0 = int(cx - ww / 2), y0 = int(cy - wh / 2), x1 = x0 + int(ww), y1 = y0 + int(wh);
            const double k = layer.weight * 1.0;
            const float kf = float(k);
            parallelRows(std::max(0, y0), std::min(h, y1), [&](int y) {
                const int sy = std::clamp(int(std::nearbyint(at[1] + (y - cy) / zoom)), 0, h - 1);
                for (int x = std::max(0, x0); x < std::min(w, x1); ++x) {
                    const int sx = std::clamp(int(std::nearbyint(at[0] + (x - cx) / zoom)), 0, w - 1);
                    for (int c = 0; c < 3; ++c) {
                        float& v = img.at(c, y, x);
                        v += (source[(size_t(c) * h + sy) * w + sx] - v) * kf;
                    }
                }
            });
            const std::vector<std::array<double, 2>> box{{x0 - 0.5, y0 - 0.5}, {x1 - 0.5, y0 - 0.5}, {x1 - 0.5, y1 - 0.5}, {x0 - 0.5, y1 - 0.5}};
            marks.polyline(box, line, color, k, true);
            size_t nearest = 0;
            double close = INFINITY;
            for (size_t b = 0; b < box.size(); ++b) {
                const double d = std::hypot(box[b][0] - at[0], box[b][1] - at[1]);
                if (d < close) {
                    close = d;
                    nearest = b;
                }
            }
            marks.line(at[0], at[1], box[nearest][0], box[nearest][1], line, accent, k);
            ring(marks, at[0], at[1], 0.012 * shortSide, line, accent, k);
            const double px = std::max(1.0, 0.0055 * shortSide);
            marks.rect(x0 - 0.5, y0 - 7 * px, x0 - 0.5 + 34 * px, y0 - 0.5, accent, k);
            char label[32];
            std::snprintf(label, sizeof(label), "0X%03llX %d", (unsigned long long)(layer.keys[hub] % 4096), int(n + 1));
            text(marks, label, x0 + px, y0 - 6 * px, px, white, k);
        }
    }
    marks.compose(img);
    img.store(exchange);
}

inline bool native(int preset, const ImageView& exchange, long long seed) {
    if (preset == 2) { contours(exchange); return true; }
    if (preset == 4) { horizontalEcho(exchange, seed); return true; }
    if (preset == 6) { curtain(exchange); return true; }
    if (preset == 7) { portraitWindow(exchange); return true; }
    if (preset == 5) { gridSystem(exchange, seed); return true; }
    return false;
}

}
