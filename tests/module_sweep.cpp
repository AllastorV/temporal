#include "harness.h"
#include <chrono>

using namespace tpt;

namespace {

bool finite(const ImageView& v) {
    for (int y = v.bounds.y1; y < v.bounds.y2; ++y)
        for (int x = v.bounds.x1; x < v.bounds.x2; ++x)
            for (int c = 0; c < 4; ++c)
                if (!std::isfinite(v.px(x, y)[c])) return false;
    return true;
}

}

TP_TEST(every_option_look_and_range_end_renders_finite) {
    const Clip clip = patternClip(96, 54, 40);
    std::vector<std::string> failures;
    size_t renders = 0;
    for (const ModuleDef* m : allModules()) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto cases = sweepCases(*m);
        for (const auto& c : cases) {
            for (double t : {0.0, 20.0, 39.0}) {
                for (double scale : {1.0, 0.5}) {
                    ++renders;
                    try {
                        RenderResult r = renderAt(*m, clip, t, c.params, scale);
                        if (!finite(r.out.view)) failures.push_back(m->id + " [" + c.label + "] t=" + std::to_string(t) + " scale=" + std::to_string(scale) + ": non-finite output");
                    } catch (const std::exception& e) {
                        failures.push_back(m->id + " [" + c.label + "] t=" + std::to_string(t) + " scale=" + std::to_string(scale) + ": " + e.what());
                    }
                }
            }
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  sweep %-20s %4zu cases %7.0f ms\n", m->id.c_str(), cases.size(), ms);
        std::fflush(stdout);
    }
    std::printf("  sweep: %zu renders, %zu failures\n", renders, failures.size());
    for (size_t i = 0; i < failures.size() && i < 60; ++i) std::printf("  FAIL %s\n", failures[i].c_str());
    CHECK(failures.empty());
}
