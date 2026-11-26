#include "modules.h"

namespace tp {

namespace {

struct BtBlob { float cx, cy, x1, y1, x2, y2; int area; int label; };
struct BtTrack { int id; int last; std::vector<int> frames; std::vector<BtBlob> hist; };

const unsigned char kBtDigits[10][5] = {
    {7, 5, 5, 5, 7}, {2, 6, 2, 2, 7}, {7, 1, 7, 4, 7}, {7, 1, 7, 1, 7}, {5, 5, 7, 1, 1},
    {7, 4, 7, 1, 7}, {7, 4, 7, 5, 7}, {7, 1, 1, 1, 1}, {7, 5, 7, 5, 7}, {7, 5, 7, 1, 7}};

struct BtMask {
    int w = 0, h = 0, f = 1, x0 = 0, y0 = 0;
    std::vector<unsigned char> m;
    std::vector<int> lab;
};

int btFind(std::vector<int>& p, int a) {
    while (p[a] != a) { p[a] = p[p[a]]; a = p[a]; }
    return a;
}

BtMask btDetect(const ImageView& img, const ImageView& prev, const Rect& R, int f, int mode, float thr, Encoding enc) {
    BtMask M;
    M.f = f; M.x0 = R.x1; M.y0 = R.y1;
    M.w = std::max(1, (R.w() + f - 1) / f); M.h = std::max(1, (R.h() + f - 1) / f);
    M.m.assign(static_cast<size_t>(M.w) * M.h, 0);
    if (!img.valid()) {M.lab.assign(M.m.size(), -1); return M;}
    parallelRows(0, M.h, [&](int j) {
        for (int i = 0; i < M.w; ++i) {
            int x = R.x1 + i * f + f / 2, y = R.y1 + j * f + f / 2;
            const RGBA color = decode(img.at(x, y, SampleEdge::Clamp), enc);
            const float l = luma(color);
            bool on;
            if (mode == 0) on = l > thr;
            else if (mode == 1) on = encodeValue(std::max(0.f,l),Encoding::Gamma24) < thr;
            else {
                on = false;
                if (prev.valid()) {
                    const RGBA previous = decode(prev.at(x,y,SampleEdge::Clamp),enc);
                    float change = 0.f;
                    for (int ch=0; ch<3; ++ch) {
                        const float a=encodeValue(std::max(0.f,color[ch]),Encoding::Gamma24);
                        const float b=encodeValue(std::max(0.f,previous[ch]),Encoding::Gamma24);
                        change=std::max(change,std::abs(a-b)/(0.08f+std::max(a,b)));
                    }
                    on = change > thr;
                }
            }
            M.m[static_cast<size_t>(j) * M.w + i] = on && color.a > 1e-5f ? 1 : 0;
        }
    });
    std::vector<int> parent(M.m.size());
    for (size_t k = 0; k < parent.size(); ++k) parent[k] = static_cast<int>(k);
    for (int j = 0; j < M.h; ++j)
        for (int i = 0; i < M.w; ++i) {
            int k = j * M.w + i;
            if (!M.m[static_cast<size_t>(k)]) continue;
            if (i > 0 && M.m[static_cast<size_t>(k - 1)]) parent[btFind(parent, k)] = btFind(parent, k - 1);
            if (j > 0 && M.m[static_cast<size_t>(k - M.w)]) { int a = btFind(parent, k), b = btFind(parent, k - M.w); parent[std::max(a, b)] = std::min(a, b); }
        }
    M.lab.assign(M.m.size(), -1);
    for (size_t k = 0; k < M.m.size(); ++k) if (M.m[k]) M.lab[k] = btFind(parent, static_cast<int>(k));
    return M;
}

std::vector<BtBlob> btBlobs(const BtMask& M, int minA, int maxA) {
    std::vector<BtBlob> out;
    std::vector<int> idx(M.m.size(), -1);
    std::vector<BtBlob> all;
    for (int j = 0; j < M.h; ++j)
        for (int i = 0; i < M.w; ++i) {
            int l = M.lab[static_cast<size_t>(j) * M.w + i];
            if (l < 0) continue;
            if (idx[static_cast<size_t>(l)] < 0) { idx[static_cast<size_t>(l)] = static_cast<int>(all.size()); all.push_back({0, 0, 1e9f, 1e9f, -1e9f, -1e9f, 0, l}); }
            BtBlob& b = all[static_cast<size_t>(idx[static_cast<size_t>(l)])];
            b.cx += i; b.cy += j; b.area++;
            b.x1 = std::min(b.x1, static_cast<float>(i)); b.y1 = std::min(b.y1, static_cast<float>(j));
            b.x2 = std::max(b.x2, static_cast<float>(i + 1)); b.y2 = std::max(b.y2, static_cast<float>(j + 1));
        }
    for (BtBlob& b : all) {
        if (b.area < minA || b.area > maxA) continue;
        b.cx = M.x0 + (b.cx / b.area + 0.5f) * M.f; b.cy = M.y0 + (b.cy / b.area + 0.5f) * M.f;
        b.x1 = M.x0 + b.x1 * M.f; b.y1 = M.y0 + b.y1 * M.f; b.x2 = M.x0 + b.x2 * M.f; b.y2 = M.y0 + b.y2 * M.f;
        out.push_back(b);
    }
    std::sort(out.begin(), out.end(), [](const BtBlob& a, const BtBlob& b) { return a.area != b.area ? a.area > b.area : (a.cy != b.cy ? a.cy < b.cy : a.cx < b.cx); });
    if (out.size() > 64) out.resize(64);
    return out;
}

}

ModuleDef makeBlobTracking() {
    ModuleDef m;
    m.id = "blob_tracking";
    m.name = "Blob Tracking";
    m.family = "Tracking";
    m.description = "Finds bright, dark or moving regions with connected components and tracks them with persistent IDs, then draws boxes, IDs, contours, motion trails, centroids or a matte. Tracking restarts at fixed segment boundaries so every frame is reproducible on seek; IDs are stable within a segment. Overlays are burned into the image.";
    m.temporal = true;
    m.usesEncoding = true;
    m.params = {
        Choice("source", "Detect", {"Bright", "Dark", "Motion"}, 0, "What counts as a blob: bright areas, dark areas, or areas that changed since the previous frame."),
        Dbl("threshold", "Bright Threshold", 0.55, 0.0, 4.0, 0.0, 1.0, Unit::None, "Linear brightness required for Bright detection."),
        Dbl("dark_threshold", "Dark Threshold", 0.30, 0, 1, 0, 1, Unit::Normalized, "Select shadows below this display brightness."),
        Dbl("motion_threshold", "Motion Threshold", 0.15, 0, 1, 0, 1, Unit::Normalized, "Minimum relative channel change. Lower values detect more subtle movement in shadows."),
        Dbl("min_area", "Min Area", 0.05, 0.0, 50.0, 0.0, 5.0, Unit::Percent, "Smallest blob as a percentage of the frame area."),
        Dbl("max_area", "Max Area", 25.0, 0.0, 100.0, 0.0, 100.0, Unit::Percent, "Largest blob as a percentage of the frame area."),
        Int("persistence", "Segment", 24, 2, 48, 2, 48, Unit::Frames, "Tracking segment length: IDs persist within each segment of this many frames; also the longest trail."),
        Int("smoothing", "Smoothing", 3, 0, 12, 0, 12, Unit::Frames, "Boxes and centroids are averaged over this many recent frames."),
        Int("occlusion", "Occlusion", 4, 0, 24, 0, 24, Unit::Frames, "How many frames a track may be lost and still keep its ID."),
        Choice("style", "Overlay", {"Boxes + IDs", "Boxes", "Contours", "Trails", "Centroids", "Mask", "Nodes"}, 0, "What to draw. Mask outputs a white-on-black matte of the blobs. Nodes draws a network: boxes, IDs, a node at each centre and links to the nearest blobs."),
        Int("connections", "Connections", 2, 0, 8, 0, 8, Unit::None, "Nodes only: links drawn from each blob to its nearest neighbours."),
        Dbl("link_distance", "Link Distance", 40.0, 0.0, 100.0, 0.0, 100.0, Unit::Percent, "Nodes only: longest link as a percentage of the frame diagonal."),
        Bool("negative_boxes", "Negative Inside Boxes", false, "Invert the image inside tracked boxes. Labels, borders and connection lines retain their colour."),
        Dbl("link_curve", "Link Curvature", 0, -1, 1, -1, 1, Unit::Normalized, "Bend links smoothly to either side. Zero keeps straight links."),
        Color("color", "Colour", 1.0, 0.82, 0.2, "Overlay colour."),
        Dbl("line_width", "Line Width", 2.0, 0.5, 20.0, 0.5, 8.0, Unit::Pixels, "Line thickness in full-resolution pixels."),
        Dbl("dim", "Background Dim", 0.0, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Darkens the image under the overlay."),
    };
    m.framesNeeded = [](const ParamValues& p, double) {
        FrameWindow w;
        const int style = p.i("style");
        w.before = (style == 2 || style == 5 ? 0 : std::clamp(p.i("persistence"), 2, 48))
                   + (p.i("source") == 2 ? 1 : 0);
        return w;
    };
    m.render = [](const RenderContext& c) {
        const ParamValues& p = c.p();
        const ImageView cur = c.source();
        if (!cur.valid()) {
            parallelRows(c.window.y1, c.window.y2, [&](int y) { for (int x = c.window.x1; x < c.window.x2; ++x) c.dst.store(x, y, RGBA{}); });
            return;
        }
        Rect R = intersect(c.frame, cur.bounds);
        if (R.empty()) R = cur.bounds;
        const Encoding enc = c.encoding;
        const int src = p.i("source");
        const float thr = std::max(0.f,p.f(src == 2 ? "motion_threshold" : src == 1 ? "dark_threshold" : "threshold"));
        const int seg = std::clamp(p.i("persistence"), 2, 48);
        const int smooth = std::clamp(p.i("smoothing"), 0, 12);
        const int occl = std::clamp(p.i("occlusion"), 0, 24);
        const int style = p.i("style");
        const int f = c.preview ? std::max(1, (R.w() + 319) / 320) : std::max(1, R.w() / 480);
        const int t = static_cast<int>(std::lround(c.time));
        const int anchor = style == 5 || style == 2 ? t : std::max(static_cast<int>(std::round(c.first)),t - (((t % seg) + seg) % seg));
        const double cells = std::max(1.0, std::ceil(R.w() / static_cast<double>(f)) * std::ceil(R.h() / static_cast<double>(f)));
        const int minA = std::max(1, static_cast<int>(std::ceil(cells * std::max(0.f, p.f("min_area")) / 100.0)));
        const int maxA = std::max(minA, static_cast<int>(cells * clampf(p.f("max_area"), 0.f, 100.f) / 100.0));
        const float gate = 0.12f * std::hypot(static_cast<float>(R.w()), static_cast<float>(R.h()));

        std::vector<BtTrack> tracks;
        int nextId = 1;
        BtMask lastMask;
        std::vector<BtBlob> lastBlobs;
        for (int fr = anchor; fr <= t; ++fr) {
            if (c.isAborted()) return;
            ImageView img = fr == t ? cur : c.frameOffset(fr - t);
            ImageView prev = src == 2 ? c.frameOffset(fr - 1 - t) : ImageView{};
            BtMask M = btDetect(img, prev, R, f, src, thr, enc);
            if (c.releaseRaw) {
                if (src == 2 && fr-1 >= c.first && fr-1 != t) c.releaseRaw(fr-1);
                else if (src != 2 && fr != t) c.releaseRaw(fr);
            }
            std::vector<BtBlob> blobs = btBlobs(M, minA, maxA);
            std::vector<bool> used(blobs.size(), false);
            struct Cand { float d; int ti, bi; };
            std::vector<Cand> cands;
            for (size_t ti = 0; ti < tracks.size(); ++ti) {
                if (fr - tracks[ti].last > occl + 1) continue;
                const BtBlob& lb = tracks[ti].hist.back();
                for (size_t bi = 0; bi < blobs.size(); ++bi) {
                    float d = std::hypot(blobs[bi].cx - lb.cx, blobs[bi].cy - lb.cy);
                    if (d < gate) cands.push_back({d, static_cast<int>(ti), static_cast<int>(bi)});
                }
            }
            std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.d != b.d ? a.d < b.d : (a.ti != b.ti ? a.ti < b.ti : a.bi < b.bi); });
            std::vector<bool> tused(tracks.size(), false);
            for (const Cand& cd : cands) {
                if (tused[static_cast<size_t>(cd.ti)] || used[static_cast<size_t>(cd.bi)]) continue;
                tused[static_cast<size_t>(cd.ti)] = used[static_cast<size_t>(cd.bi)] = true;
                BtTrack& tr = tracks[static_cast<size_t>(cd.ti)];
                tr.last = fr; tr.frames.push_back(fr); tr.hist.push_back(blobs[static_cast<size_t>(cd.bi)]);
            }
            for (size_t bi = 0; bi < blobs.size(); ++bi)
                if (!used[bi] && tracks.size() < 256) tracks.push_back({nextId++, fr, {fr}, {blobs[bi]}});
            if (fr == t) { lastMask = std::move(M); lastBlobs = blobs; }
        }

        const float lw = std::max(1.f, c.px(p.f("line_width")));
        const int W = c.window.w(), H = c.window.h();
        std::vector<unsigned char> ov(static_cast<size_t>(W) * H, 0);
        auto stamp = [&](float x, float y, float r) {
            int xa = static_cast<int>(std::floor(x - r)), xb = static_cast<int>(std::ceil(x + r)), ya = static_cast<int>(std::floor(y - r)), yb = static_cast<int>(std::ceil(y + r));
            for (int yy = std::max(ya, c.window.y1); yy < std::min(yb, c.window.y2); ++yy)
                for (int xx = std::max(xa, c.window.x1); xx < std::min(xb, c.window.x2); ++xx) ov[static_cast<size_t>(yy - c.window.y1) * W + (xx - c.window.x1)] = 255;
        };
        auto line = [&](float x0, float y0, float x1, float y1) {
            float len = std::hypot(x1 - x0, y1 - y0);
            int n = std::max(1, static_cast<int>(std::ceil(len / std::max(0.5f, lw * 0.5f))));
            for (int k = 0; k <= n; ++k) { float s = static_cast<float>(k) / n; stamp(x0 + (x1 - x0) * s, y0 + (y1 - y0) * s, lw * 0.5f); }
        };
        auto digits = [&](int value, float x, float y, float cell) {
            std::string s = std::to_string(value);
            for (size_t d = 0; d < s.size(); ++d) {
                const unsigned char* g = kBtDigits[s[d] - '0'];
                for (int row = 0; row < 5; ++row)
                    for (int col = 0; col < 3; ++col)
                        if (g[row] & (4 >> col)) {
                            float gx = x + (d * 4 + col + 0.5f) * cell, gy = y - (row + 0.5f) * cell;
                            stamp(gx, gy, cell * 0.5f);
                        }
            }
        };
        const bool invertBoxes = p.b("negative_boxes") && (style == 0 || style == 1 || style == 6);
        std::vector<Rect> boxAreas;
        std::vector<std::pair<float, float>> nodes;
        if (style != 5) {
            for (const BtTrack& tr : tracks) {
                if (tr.last != t) continue;
                int n = static_cast<int>(tr.hist.size());
                int k0 = std::max(0, n - 1 - smooth);
                BtBlob s{0, 0, 0, 0, 0, 0, 0, 0};
                for (int k = k0; k < n; ++k) { const BtBlob& b = tr.hist[static_cast<size_t>(k)]; s.cx += b.cx; s.cy += b.cy; s.x1 += b.x1; s.y1 += b.y1; s.x2 += b.x2; s.y2 += b.y2; }
                float inv = 1.f / (n - k0);
                s.cx *= inv; s.cy *= inv; s.x1 *= inv; s.y1 *= inv; s.x2 *= inv; s.y2 *= inv;
                if (style == 0 || style == 1 || style == 6) {
                    if (invertBoxes) {
                        Rect box{int(std::ceil(s.x1-0.5f)),int(std::ceil(s.y1-0.5f)),
                                 int(std::ceil(s.x2-0.5f)),int(std::ceil(s.y2-0.5f))};
                        box=intersect(box,c.window);
                        if (!box.empty()) boxAreas.push_back(box);
                    }
                    line(s.x1, s.y1, s.x2, s.y1); line(s.x2, s.y1, s.x2, s.y2); line(s.x2, s.y2, s.x1, s.y2); line(s.x1, s.y2, s.x1, s.y1);
                    if (style != 1) digits(tr.id, s.x1, s.y2 + lw * 7.f, std::max(1.f, lw * 1.2f));
                    if (style == 6) { stamp(s.cx, s.cy, lw * 1.6f); nodes.push_back({s.cx, s.cy}); }
                } else if (style == 3) {
                    for (int k = 1; k < n; ++k) line(tr.hist[static_cast<size_t>(k - 1)].cx, tr.hist[static_cast<size_t>(k - 1)].cy, tr.hist[static_cast<size_t>(k)].cx, tr.hist[static_cast<size_t>(k)].cy);
                    stamp(s.cx, s.cy, lw * 1.5f);
                } else if (style == 4) {
                    stamp(s.cx, s.cy, lw * 2.f);
                }

            }
        }
        if (style == 6 && nodes.size() > 1) {
            const int kmax = std::clamp(p.i("connections"), 0, 8);
            const float maxD = clampf(p.f("link_distance"), 0.f, 100.f) / 100.f * std::hypot(static_cast<float>(R.w()), static_cast<float>(R.h()));
            std::vector<std::pair<int, int>> links;
            for (size_t i = 0; i < nodes.size(); ++i) {
                std::vector<std::pair<float, int>> nb;
                for (size_t j = 0; j < nodes.size(); ++j)
                    if (j != i) nb.push_back({std::hypot(nodes[i].first - nodes[j].first, nodes[i].second - nodes[j].second), static_cast<int>(j)});
                std::sort(nb.begin(), nb.end());
                for (int k = 0; k < std::min<int>(kmax, static_cast<int>(nb.size())); ++k) {
                    if (nb[static_cast<size_t>(k)].first > maxD) break;
                    int a = static_cast<int>(i), b = nb[static_cast<size_t>(k)].second;
                    links.push_back({std::min(a, b), std::max(a, b)});
                }
            }
            std::sort(links.begin(), links.end());
            links.erase(std::unique(links.begin(), links.end()), links.end());
            const float thin = lw;
            const float curvature = p.f("link_curve");
            for (const auto& l : links) {
                const auto& A = nodes[static_cast<size_t>(l.first)];
                const auto& B = nodes[static_cast<size_t>(l.second)];
                float len = std::hypot(B.first - A.first, B.second - A.second);
                int n = std::max(1, static_cast<int>(std::ceil(len / std::max(0.5f, thin * 0.5f))));
                for (int k = 0; k <= n; ++k) {
                    float s = static_cast<float>(k) / n;
                    const float bend = 0.8f * curvature * s * (1.f-s);
                    float dx=B.first-A.first,dy=B.second-A.second;
                    stamp(A.first + dx*s - dy*bend, A.second + dy*s + dx*bend, thin * 0.35f);
                }
            }
        }
        std::vector<char> keep;
        if (style == 2 || style == 5) {
            std::vector<char> accepted(lastMask.lab.size(), 0);
            for (const BtBlob& blob : lastBlobs) accepted[blob.label] = 1;
            keep.resize(lastMask.lab.size());
            for (size_t k = 0; k < keep.size(); ++k)
                keep[k] = lastMask.lab[k] >= 0 && accepted[lastMask.lab[k]];
        }
        const auto col = p.vec("color");
        const RGBA oc{static_cast<float>(col[0]), static_cast<float>(col[1]), static_cast<float>(col[2]), 1.f};
        const float dim = saturate(p.f("dim"));
        auto cellOn = [&](int x, int y) {
            int i = (x - lastMask.x0) / lastMask.f, j = (y - lastMask.y0) / lastMask.f;
            if (i < 0 || j < 0 || i >= lastMask.w || j >= lastMask.h) return false;
            return keep[static_cast<size_t>(j) * lastMask.w + i] != 0;
        };
        parallelRows(c.window.y1, c.window.y2, [&](int y) {
            std::vector<int> coverage;
            int inside=0;
            if (!boxAreas.empty()) {
                coverage.assign(W+1,0);
                for (const Rect& box : boxAreas) if (y>=box.y1 && y<box.y2) {
                    ++coverage[box.x1-c.window.x1]; --coverage[box.x2-c.window.x1];
                }
            }
            for (int x = c.window.x1; x < c.window.x2; ++x) {
                RGBA s = cur.at(x, y, SampleEdge::Black);
                if (!coverage.empty()) {
                    inside += coverage[x-c.window.x1];
                    if (inside>0 && s.a>1e-6f) {
                        for (int ch=0;ch<3;++ch) {
                            if (enc == Encoding::Gamma24) {
                                s[ch] = s.a * (1.f - saturate(s[ch] / s.a));
                            } else {
                                float display=encodeValue(decodeValue(s[ch]/s.a,enc),Encoding::Gamma24);
                                s[ch]=s.a*encodeValue(decodeValue(1.f-saturate(display),Encoding::Gamma24),enc);
                            }
                        }
                    }
                }
                if (style == 5) { float v = cellOn(x, y) ? 1.f : 0.f; c.dst.store(x, y, RGBA{v, v, v, 1.f}); continue; }
                float a = ov[static_cast<size_t>(y - c.window.y1) * W + (x - c.window.x1)] / 255.f;
                if (style == 2 && cellOn(x, y)) {
                    int r = std::max(1, static_cast<int>(std::lround(lw)));
                    if (!cellOn(x - r, y) || !cellOn(x + r, y) || !cellOn(x, y - r) || !cellOn(x, y + r)) a = 1.f;
                }
                if (a == 0.f && dim == 0.f) { c.dst.store(x, y, s); continue; }
                RGBA lin = decode(s, enc) * (1.f - dim);
                lin.a = s.a;
                RGBA o = lerp(lin, RGBA{oc.r, oc.g, oc.b, std::max(lin.a, 1.f)}, a);
                o.a = std::max(s.a, a);
                c.dst.store(x, y, encode(o, enc));
            }
        });
    };
    return m;
}

}
