#include "harness.h"
#include <chrono>

using namespace tpt;

static ParamValues fuzzParams(const ModuleDef& m, uint32_t seed) {
    ParamValues p = defaultValues(m);
    int k = 0;
    for (const auto& s : m.params) {
        ++k;
        float r = hashFloat(seed, k, 1), r2 = hashFloat(seed, k, 2), wide = hashFloat(seed, k, 3);
        auto& v = p.v[s.id];
        switch (s.type) {
        case PType::Double:
        case PType::Int: {
            double lo = wide < 0.25f ? s.min : s.dmin, hi = wide < 0.25f ? s.max : s.dmax;
            lo = std::max(lo, s.min); hi = std::min(hi, s.max);
            double x = lo + (hi - lo) * r;
            v[0] = s.type == PType::Int ? std::round(x) : x;
            break;
        }
        case PType::Choice: v[0] = std::floor(r * s.options.size() * 0.9999); break;
        case PType::Bool: v[0] = r < 0.5f ? 0 : 1; break;
        case PType::Color: v = {r, r2, hashFloat(seed, k, 4)}; break;
        case PType::Point: v = {r, r2, 0}; break;
        }
    }
    if (p.v.count("mix")) p.v["mix"][0] = 1.0;
    return p;
}

static void checkFinite(const ImageView& v) {
    for (int y = v.bounds.y1; y < v.bounds.y2; ++y)
        for (int x = v.bounds.x1; x < v.bounds.x2; ++x) {
            RGBA c = v.load(x, y);
            for (int ch = 0; ch < 4; ++ch) CHECK(std::isfinite(c[ch]));
        }
}

static void checkWindow(const RenderResult& r, double t, const Clip& clip) {
    int tc = static_cast<int>(std::lround(t));
    for (int f : r.fetched) {
        bool inWindow = f >= tc - r.window.before && f <= tc + r.window.after;
        if (!inWindow) {
            std::ostringstream os;
            os << "fetched frame " << f << " outside declared window [" << tc - r.window.before << "," << tc + r.window.after << "] at t=" << t;
            throw std::runtime_error(os.str());
        }
        (void)clip;
    }
}

static void genericSuite(const ModuleDef& m) {
    Clip clip = patternClip();
    ParamValues def = defaultValues(m);

    {
        ParamValues p = with(def, {{"mix", 0.0}});
        RenderResult r = renderAt(m, clip, 12, p);
        CHECK(maxDiff(r.out.view, clip.frames[12].view) == 0.0);
    }
    {
        RenderResult a = renderAt(m, clip, 15, def);
        renderAt(m, clip, 3, def);
        renderAt(m, clip, 33, def);
        RenderResult b = renderAt(m, clip, 15, def);
        CHECK(maxDiff(a.out.view, b.out.view) == 0.0);
        checkFinite(a.out.view);
        checkWindow(a, 15, clip);
    }
    {
        ParamValues half = with(def, {{"mix", 0.5}});
        RenderResult full = renderAt(m, clip, 20, def);
        RenderResult h = renderAt(m, clip, 20, half);
        const ImageView& s = clip.frames[20].view;
        double err = 0;
        for (int y = 0; y < clip.rect.y2; ++y)
            for (int x = 0; x < clip.rect.x2; ++x) {
                RGBA e = lerp(s.load(x, y), full.out.view.load(x, y), 0.5f);
                RGBA g = h.out.view.load(x, y);
                for (int c = 0; c < 4; ++c) err = std::max(err, static_cast<double>(std::fabs(e[c] - g[c])));
            }
        CHECK(err < 1e-4);
    }
    {
        Clip big = patternClip(64, 48, 40, 100);
        double scale = 0.5;
        ParamValues p = def;
        RenderResult r = renderAt(m, big, 110, p, scale);
        checkFinite(r.out.view);
    }
    const double times[] = {0, 1, 19.6, 38, 39};
    for (uint32_t s = 1; s <= 14; ++s) {
        ParamValues p = fuzzParams(m, s * 7919u + static_cast<uint32_t>(m.id.size()));
        for (double t : times) {
            for (int e = 0; e < 3; ++e) {
                if (p.v.count("edge")) p.v["edge"][0] = e;
                else if (e > 0) break;
                RenderResult r = renderAt(m, clip, t, p);
                checkFinite(r.out.view);
                checkWindow(r, t, clip);
            }
        }
    }
    if (m.temporal) {
        Clip still = constantClip(RGBA{0.4f, 0.3f, 0.2f, 1.f});
        ParamValues p = with(def, {{"edge", 2.0}});
        RenderResult r = renderAt(m, still, 0, p);
        checkFinite(r.out.view);
    }
}

int main(int argc, char** argv) {
    std::string filter = argc > 1 ? argv[1] : "";
    int failed = 0, passed = 0;
    for (const ModuleDef* m : allModules()) {
        if (!filter.empty() && m->id.find(filter) == std::string::npos) continue;
        auto t0 = std::chrono::steady_clock::now();
        try {
            genericSuite(*m);
            ++passed;
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("[PASS] generic/%s (%.0f ms)\n", m->id.c_str(), ms);
        } catch (const std::exception& e) {
            ++failed;
            std::printf("[FAIL] generic/%s: %s\n", m->id.c_str(), e.what());
        }
    }
    for (const auto& tc : registry()) {
        if (!filter.empty() && std::string(tc.name).find(filter) == std::string::npos) continue;
        try {
            tc.fn();
            ++passed;
            std::printf("[PASS] %s\n", tc.name);
        } catch (const std::exception& e) {
            ++failed;
            std::printf("[FAIL] %s: %s\n", tc.name, e.what());
        }
    }
    std::printf("%d passed, %d failed, %zu modules\n", passed, failed, allModules().size());
    return failed == 0 ? 0 : 1;
}
