#include "harness.h"
#include "gpu/gpu.h"
#include "core/flow.h"

using namespace tpt;

TP_TEST(gpu_flow_matches_reference_pyramid) {
    if (!gpu::available()) return;
    const auto clip=makeClip(337,227,2,[](int x,int y,int f) {
        const float l=0.45f+0.2f*std::sin((x-3*f)*0.13f)+0.18f*std::cos((y+2*f)*0.17f);
        return RGBA{l,l,l,1};
    });
    for (int quality : {0,1,2}) {
        gpu::setEnabled(false);
        auto cpu=computeFlow(clip.frames[0].view,clip.frames[1].view,clip.rect,1.f,Encoding::Linear,quality);
        gpu::setEnabled(true);
        auto gpuFlow=computeFlow(clip.frames[0].view,clip.frames[1].view,clip.rect,1.f,Encoding::Linear,quality);
        CHECK(cpu.scale==gpuFlow.scale && cpu.w==gpuFlow.w && cpu.h==gpuFlow.h);
        for (size_t k=0;k<cpu.u.size();++k) {
            CHECK_NEAR(cpu.u[k],gpuFlow.u[k],0.001);
            CHECK_NEAR(cpu.v[k],gpuFlow.v[k],0.001);
        }
    }
}

static double gpuCpuDiff(const ModuleDef& m, const Clip& c, double t, const ParamValues& p) {
    gpu::setEnabled(true);
    RenderResult g = renderAt(m, c, t, p);
    gpu::setEnabled(false);
    RenderResult cpu = renderAt(m, c, t, p);
    gpu::setEnabled(true);
    return maxDiff(g.out.view, cpu.out.view);
}

TP_TEST(gpu_motion_tear_matches_cpu_reference) {
    if (!gpu::available()) { std::printf("  (no CUDA device: GPU parity skipped)\n"); return; }
    const ModuleDef& m = find("motion_tear");
    Clip clip = patternClip(135, 83, 4);
    for (int mode : {0, 1}) {
        for (int encoding = 0; encoding < 4; ++encoding) {
            for (int sharp = 0; sharp < 3; ++sharp) {
                auto p = with(defaultValues(m), {{"mode", mode}, {"encoding", encoding},
                    {"sharp_at", sharp}, {"angle", -31}, {"warmth", -0.45}, {"amount", 1.3},
                    {"shutter", sharp ? 9 : 3}, {"samples", 11}, {"threshold", 0}});
                CHECK(gpuCpuDiff(m, clip, 2, p) < 3e-3);
            }
        }
    }
}

TP_TEST(gpu_motion_tear_writes_only_requested_window) {
    if (!gpu::available()) { std::printf("  (no CUDA device: GPU dispatch skipped)\n"); return; }
    const Rect bounds{-14, 9, 23, 38}, tile{-6, 16, 13, 27};
    OwnedImage src(bounds), out(bounds);
    for (int y = bounds.y1; y < bounds.y2; ++y)
        for (int x = bounds.x1; x < bounds.x2; ++x)
            src.view.store(x, y, {0.1f * x, 0.05f * y, 0.3f, 0.6f});
    gpu::TearExposure exposure{};
    exposure.fromMotion = true;
    // Missing previous frame gives zero flow, even with a nonzero shutter.
    exposure.shutter = 7;
    CHECK(gpu::motionTear(src.view, {}, out.view, tile, {}, {}, exposure));
    for (int y = bounds.y1; y < bounds.y2; ++y)
        for (int x = bounds.x1; x < bounds.x2; ++x) {
            const RGBA expected = tile.contains(x, y) ? src.view.load(x, y) : RGBA{};
            for (int ch = 0; ch < 4; ++ch)
                CHECK_NEAR(out.view.load(x, y)[ch], expected[ch], 1e-6);
        }
}

TP_TEST(gpu_time_shift_matches_cpu_reference) {
    if (!gpu::available()) { std::printf("  (no CUDA device: GPU parity skipped)\n"); return; }
    const ModuleDef& m = find("time_shift");
    Clip c = patternClip(96, 64, 40);
    ParamValues base = defaultValues(m);
    const std::initializer_list<std::pair<const char*, double>> sets[] = {
        {{"mode", 0}, {"combine", 0}},
        {{"mode", 1}, {"combine", 1}, {"streak_length", -40}},
        {{"mode", 2}, {"combine", 2}, {"streak_length", 120}, {"coverage", 0.8}},
        {{"mode", 0}, {"combine", 0}, {"encoding", 0}, {"streak_length", 4}},
        {{"mode", 0}, {"combine", 1}, {"encoding", 2}, {"offset_frames", 20}, {"detail", 20}},
    };
    for (const auto& s : sets) {
        ParamValues p = with(base, s);
        for (int e = 0; e < 3; ++e) {
            p.v["edge"][0] = e;
            double d = gpuCpuDiff(m, c, 3, p);
            if (!(d < 2e-3)) {
                std::ostringstream os;
                os << "GPU/CPU max diff " << d;
                throw std::runtime_error(os.str());
            }
        }
    }
}

TP_TEST(gpu_print_lab_matches_cpu_reference) {
    if (!gpu::available()) { std::printf("  (no CUDA device: GPU parity skipped)\n"); return; }
    const ModuleDef& m = find("print_lab");
    Clip c = patternClip(160, 96, 6);
    for (int look = 0; look < 12; ++look)
        for (int whole = 0; whole < 2; ++whole) {
            ParamValues p = with(defaultValues(m), {{"look", look}, {"show_whole_picture", whole}});
            gpu::setEnabled(true);
            RenderResult g = renderAt(m, c, 3, p);
            gpu::setEnabled(false);
            RenderResult r = renderAt(m, c, 3, p);
            gpu::setEnabled(true);
            int bad = 0, n = 0;
            double sum = 0;
            for (int y = 0; y < 96; ++y)
                for (int x = 0; x < 160; ++x) {
                    RGBA a = g.out.view.load(x, y), b = r.out.view.load(x, y);
                    for (int ch = 0; ch < 4; ++ch) {
                        double d = std::fabs(a[ch] - b[ch]);
                        CHECK(std::isfinite(a[ch]));
                        sum += d; ++n;
                        if (d > 0.02) ++bad;
                    }
                }
            if (!(sum / n < 1e-3 && bad <= n / 1000)) {
                std::ostringstream os;
                os << "look " << look << " whole " << whole << ": mean diff " << sum / n << ", " << bad << " of " << n << " samples > 0.02";
                throw std::runtime_error(os.str());
            }
        }
}

TP_TEST(gpu_motion_modules_match_cpu_reference) {
    if (!gpu::available()) { std::printf("  (no CUDA device: GPU parity skipped)\n"); return; }
    Clip c = patternClip(96, 64, 40);
    for (const char* id : {"camera_motion", "procedural_motion"}) {
        const ModuleDef& m = find(id);
        for (int mode = 0; mode < (std::string(id) == "camera_motion" ? 12 : 7); ++mode)
            for (int edge = 0; edge < 3; ++edge) {
                ParamValues p = with(defaultValues(m), {{"mode", mode}, {"outside", edge}});
                double d = gpuCpuDiff(m, c, 6, p);
                if (!(d < 3e-3)) {
                    std::ostringstream os;
                    os << id << " mode " << mode << " edge " << edge << ": GPU/CPU max diff " << d;
                    throw std::runtime_error(os.str());
                }
            }
    }
}

TP_TEST(gpu_refraction_matches_cpu_reference) {
    if (!gpu::available()) { std::printf("  (no CUDA device: GPU parity skipped)\n"); return; }
    const ModuleDef& m = find("refraction_lab");
    Clip c = patternClip(96, 64, 4);
    for (int mode = 0; mode < 4; ++mode)
        for (int enc = 0; enc < 4; ++enc) {
            ParamValues p = with(defaultValues(m), {{"mode", mode}, {"encoding", enc}, {"ior", 1.7}, {"dispersion", 0.03}});
            double d = gpuCpuDiff(m, c, 2, p);
            if (!(d < 3e-3)) {
                std::ostringstream os;
                os << "mode " << mode << " enc " << enc << ": GPU/CPU max diff " << d;
                throw std::runtime_error(os.str());
            }
        }
}

TP_TEST(gpu_halation_matches_cpu_reference) {
    if (!gpu::available()) { std::printf("  (no CUDA device: GPU parity skipped)\n"); return; }
    const ModuleDef& m = find("halation");
    Clip c = patternClip(160, 96, 12);
    const std::initializer_list<std::pair<const char*, double>> sets[] = {
        {{"threshold", 0.3}},
        {{"threshold", 0.2}, {"knee", 0}, {"radius", 60}, {"encoding", 0}},
        {{"threshold", 0.4}, {"duration", 1}, {"halo_only", 1}, {"encoding", 2}},
        {{"threshold", 0.1}, {"radius", 4}, {"decay", 0.9}, {"duration", 12}},
    };
    for (const auto& s : sets) {
        double d = gpuCpuDiff(m, c, 8, with(defaultValues(m), s));
        if (!(d < 2e-3)) {
            std::ostringstream os;
            os << "GPU/CPU max diff " << d;
            throw std::runtime_error(os.str());
        }
    }
}

TP_TEST(diffusion_lab_cuda_matches_all_ten_modes) {
    if (!gpu::available()) return;
    auto clip=patternClip(96,64,4);
    const auto& m=find("diffusion_lab");
    for (int mode=0; mode<10; ++mode) {
        auto p=with(defaultValues(m),{{"mode",double(mode)},{"strength",0.74},{"scatter_radius",8},{"streak_length",17}});
        CHECK(gpuCpuDiff(m,clip,2,p)<0.003);
    }
}
