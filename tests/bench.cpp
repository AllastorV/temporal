#include "harness.h"
#include <chrono>
#include <map>

using namespace tpt;

int main(int argc, char** argv) {
    std::string filter = argc > 1 ? argv[1] : "";
    struct Res { int w, h; const char* name; };
    const Res sizes[] = {{1920, 1080, "1080p"}, {3840, 2160, "UHD"}};
    std::printf("threads=%u\n", workerCount());
    std::printf("%-20s %-6s %8s %8s %8s\n", "module", "res", "frames", "best_ms", "median_ms");
    for (const ModuleDef* m : allModules()) {
        if (!filter.empty() && m->id.find(filter) == std::string::npos) continue;
        ParamValues p = defaultValues(*m);
        for (int a = 2; a < argc; ++a) {
            const std::string kv = argv[a];
            const size_t eq = kv.find('=');
            if (eq != std::string::npos) p.set(kv.substr(0, eq), std::stod(kv.substr(eq + 1)));
        }
        for (const auto& s : sizes) {
            FrameWindow w = effectiveWindow(*m, p, 24.0);
            const int t = 100;
            std::map<int, OwnedImage> frames;
            Rect rect{0, 0, s.w, s.h};
            auto get = [&](int f) -> const OwnedImage& {
                auto it = frames.find(f);
                if (it != frames.end()) return it->second;
                OwnedImage img(rect);
                parallelRows(0, s.h, [&](int y) {
                    for (int x = 0; x < s.w; ++x) {
                        float gx = static_cast<float>(x) / s.w, gy = static_cast<float>(y) / s.h;
                        float sq = (x / 64 + y / 64 + f) % 7 == 0 ? 1.2f : 0.f;
                        img.view.store(x, y, RGBA{0.1f + 0.6f * gx + sq, 0.2f + 0.5f * gy + sq, 0.3f + 0.3f * gx * gy, 1.f});
                    }
                });
                return frames.emplace(f, std::move(img)).first->second;
            };
            for (int f = t - w.before; f <= t + w.after; ++f) get(f);
            OwnedImage out(rect);
            RenderContext ctx;
            ctx.params = &p;
            ctx.time = t;
            ctx.fps = 24;
            ctx.first = 0;
            ctx.last = 1000;
            ctx.frame = rect;
            ctx.window = rect;
            ctx.dst = out.view;
            if (p.v.count("edge")) ctx.edge = static_cast<EdgeMode>(p.i("edge"));
            if (p.v.count("encoding")) ctx.encoding = static_cast<Encoding>(p.i("encoding"));
            ctx.fetchRaw = [&](double ft) -> ImageView { return get(static_cast<int>(std::lround(ft))).view; };
            std::vector<double> ms;
            const int runs = s.w > 2000 ? 3 : 5;
            for (int i = 0; i < runs; ++i) {
                auto t0 = std::chrono::steady_clock::now();
                runModule(*m, ctx);
                ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            if (const char* d = std::getenv("TP_DUMP")) {
                std::string fn = std::string(d) + "_" + s.name + ".raw";
                FILE* fp = std::fopen(fn.c_str(), "wb");
                if (!fp) { std::fprintf(stderr, "cannot write %s\n", fn.c_str()); return 1; }
                for (int y = 0; y < s.h; ++y) for (int x = 0; x < s.w; ++x) { RGBA v = out.view.load(x, y); std::fwrite(&v, sizeof(float), 4, fp); }
                std::fclose(fp);
            }
            std::sort(ms.begin(), ms.end());
            std::printf("%-20s %-6s %8d %8.1f %8.1f\n", m->id.c_str(), s.name, w.before + w.after + 1, ms.front(), ms[ms.size() / 2]);
            std::fflush(stdout);
        }
    }
    return 0;
}
