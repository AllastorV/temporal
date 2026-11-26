#pragma once
#include "modules/modules.h"
#include <cstdio>
#include <set>
#include <sstream>
#include <string>
#include <thread>

namespace tpt {

using namespace tp;

struct TestCase { const char* name; void (*fn)(); };
inline std::vector<TestCase>& registry() { static std::vector<TestCase> r; return r; }

#define TP_CAT2(a, b) a##b
#define TP_CAT(a, b) TP_CAT2(a, b)
#define TP_TEST(name) \
    static void name(); \
    static const int TP_CAT(name, _registered) = (::tpt::registry().push_back({#name, &name}), 0); \
    static void name()

#define CHECK(cond) do { if (!(cond)) { std::ostringstream os_; os_ << __FILE__ << ":" << __LINE__ << " CHECK(" #cond ") failed"; throw std::runtime_error(os_.str()); } } while (0)
#define CHECK_NEAR(a, b, eps) do { double a_ = (a), b_ = (b); if (!(std::fabs(a_ - b_) <= (eps))) { std::ostringstream os_; os_ << __FILE__ << ":" << __LINE__ << " CHECK_NEAR(" #a ", " #b ") " << a_ << " vs " << b_; throw std::runtime_error(os_.str()); } } while (0)

struct Clip {
    int first = 0;
    Rect rect;
    std::vector<OwnedImage> frames;
    int last() const { return first + static_cast<int>(frames.size()) - 1; }
};

template <class Gen>
Clip makeClip(int w, int h, int n, Gen gen, int first = 0) {
    Clip c;
    c.first = first;
    c.rect = Rect{0, 0, w, h};
    for (int f = 0; f < n; ++f) {
        OwnedImage img(c.rect);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) img.view.store(x, y, gen(x, y, f + first));
        c.frames.push_back(std::move(img));
    }
    return c;
}

inline Clip patternClip(int w = 64, int h = 48, int n = 40, int first = 0) {
    return makeClip(w, h, n, [w, h](int x, int y, int f) {
        float gx = static_cast<float>(x) / w, gy = static_cast<float>(y) / h;
        int sx = (f * 3) % w;
        bool sq = x >= sx && x < sx + 8 && y >= h / 3 && y < h / 3 + 8;
        float v = sq ? 1.5f : 0.f;
        return RGBA{0.1f + 0.6f * gx + v, 0.2f + 0.5f * gy + v * 0.8f, 0.3f + 0.2f * (gx * gy) + v * 0.5f, 1.f};
    }, first);
}

inline Clip constantClip(RGBA c, int w = 32, int h = 24, int n = 30) {
    return makeClip(w, h, n, [c](int, int, int) { return c; });
}

inline const ModuleDef& find(const std::string& id) {
    for (auto* m : allModules()) if (m->id == id) return *m;
    throw std::runtime_error("module not found: " + id);
}

struct RenderResult {
    OwnedImage out;
    std::set<int> fetched;
    FrameWindow window;
};

inline RenderResult renderAt(const ModuleDef& m, const Clip& clip, double t, const ParamValues& pv, double scale = 1.0, double fps = 24.0) {
    RenderResult r;
    r.out.reset(clip.rect);
    r.window = effectiveWindow(m, pv, fps);
    const std::thread::id owner = std::this_thread::get_id();
    RenderContext ctx;
    ctx.params = &pv;
    ctx.time = t;
    ctx.fps = fps;
    ctx.first = clip.first;
    ctx.last = clip.last();
    ctx.scale = scale;
    ctx.frame = clip.rect;
    ctx.window = clip.rect;
    ctx.dst = r.out.view;
    if (pv.v.count("edge")) ctx.edge = static_cast<EdgeMode>(pv.i("edge"));
    if (pv.v.count("encoding")) ctx.encoding = static_cast<Encoding>(pv.i("encoding"));
    ctx.fetchRaw = [&](double ft) -> ImageView {
        if (std::this_thread::get_id() != owner) throw std::runtime_error("fetch called from a worker thread");
        int f = static_cast<int>(std::lround(ft));
        r.fetched.insert(f);
        if (f < clip.first || f > clip.last()) return {};
        return clip.frames[static_cast<size_t>(f - clip.first)].view;
    };
    runModule(m, ctx);
    return r;
}

inline double maxDiff(const ImageView& a, const ImageView& b) {
    double d = 0;
    for (int y = a.bounds.y1; y < a.bounds.y2; ++y)
        for (int x = a.bounds.x1; x < a.bounds.x2; ++x) {
            RGBA p = a.load(x, y), q = b.at(x, y, SampleEdge::Black);
            for (int c = 0; c < 4; ++c) d = std::max(d, static_cast<double>(std::fabs(p[c] - q[c])));
        }
    return d;
}

inline RGBA mean(const ImageView& a) {
    RGBA s;
    double n = 0;
    for (int y = a.bounds.y1; y < a.bounds.y2; ++y)
        for (int x = a.bounds.x1; x < a.bounds.x2; ++x) { s += a.load(x, y); n += 1; }
    return s * static_cast<float>(1.0 / std::max(1.0, n));
}

inline ParamValues with(ParamValues p, std::initializer_list<std::pair<const char*, double>> kv) {
    for (auto& e : kv) {
        if (!p.v.count(e.first)) throw std::runtime_error(std::string("unknown param in test: ") + e.first);
        p.v[e.first][0] = e.second;
    }
    return p;
}

struct SweepCase {
    std::string label;
    ParamValues params;
};

inline std::vector<SweepCase> sweepCases(const ModuleDef& m) {
    const ParamValues def = defaultValues(m);
    std::vector<SweepCase> cases{{"default", def}};
    const auto add = [&](const std::string& label, const ParamValues& p) { cases.push_back({label, p}); };
    for (const auto& s : m.params) {
        if (s.id == "mix") continue;
        if (s.id == "studio_look" && m.loadLook) {
            for (size_t k = 1; k < s.options.size(); ++k) {
                ParamValues p = def;
                for (const auto& [id, v] : m.loadLook(m, static_cast<int>(k)).v) p.v[id] = v;
                p.v[s.id][0] = static_cast<double>(k);
                add("look " + s.options[k], p);
            }
            continue;
        }
        switch (s.type) {
        case PType::Choice:
            for (size_t k = 0; k < s.options.size(); ++k) {
                ParamValues p = def;
                p.v[s.id][0] = static_cast<double>(k);
                add(s.id + "=" + s.options[k], p);
            }
            break;
        case PType::Bool: {
            ParamValues p = def;
            p.v[s.id][0] = def.v.at(s.id)[0] != 0 ? 0 : 1;
            add(s.id + " toggled", p);
            break;
        }
        case PType::Double:
        case PType::Int:
            for (double x : {s.min, s.max, s.dmin, s.dmax}) {
                ParamValues p = def;
                p.v[s.id][0] = s.type == PType::Int ? std::round(x) : x;
                add(s.id + "=" + std::to_string(x), p);
            }
            break;
        case PType::Color:
            for (double x : {0.0, 1.0}) {
                ParamValues p = def;
                p.v[s.id] = {x, x, x};
                add(s.id + "=" + std::to_string(x), p);
            }
            break;
        case PType::Point:
            for (double x : {0.0, 1.0}) {
                ParamValues p = def;
                p.v[s.id] = {x, 1.0 - x, 0};
                add(s.id + " corner " + std::to_string(x), p);
            }
            break;
        }
    }
    return cases;
}

}
