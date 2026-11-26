#include "harness.h"
#include "gpu/gpu.h"
#include "core/numpy_random.h"
#include "modules/cutwright_geo.h"
#include <chrono>

using namespace tpt;

TP_TEST(numpy_pcg64_float32_matches_numpy) {
    const struct { uint32_t seed; float v[5]; } refs[] = {
        {0u, {0.850624204f, 0.636961639f, 0.511136472f, 0.269786656f, 0.30782938f}},
        {1u, {0.473188639f, 0.511821568f, 0.755167484f, 0.950463653f, 0.0348525047f}},
        {114u, {0.16096884f, 0.727888227f, 0.893970251f, 0.142159104f, 0.0698934197f}},
        {4294967295u, {0.190810978f, 0.251943529f, 0.724247456f, 0.219065785f, 0.909559011f}},
    };
    for (const auto& r : refs) {
        const auto got = tp::numpyRandomFloat32(r.seed, 5);
        for (int i = 0; i < 5; ++i) CHECK(got[i] == r.v[i]);
    }
}

struct Parity {
    double maxDiff = 0;
    int rowsOver = 0, rowsOverAwayFromCrossings = 0, changed = 0;
};

static Parity cutwrightGpuVsPython(const char* id, int preset, int w, int h, int crossings = 0) {
    const ModuleDef& m = find(id);
    const Clip clip = makeClip(w, h, 1, [w, h](int x, int y, int) {
        const float u = float(x) / w, v = float(y) / h;
        const float ring = 0.5f + 0.5f * std::sin(40.f * std::hypot(u - 0.4f, v - 0.55f));
        return RGBA{0.15f + 0.8f * u * ring, 0.1f + 0.85f * v, 0.9f - 0.7f * u * v + 0.08f * ((x / 7 + y / 5) % 2), 1};
    });
    const auto p = with(defaultValues(m), {{"preset", double(preset)}, {"resolution", 2}});
    gpu::setEnabled(false);
    const RenderResult python = renderAt(m, clip, 0, p);
    gpu::setEnabled(true);
    const RenderResult cuda = renderAt(m, clip, 0, p);
    Parity r;
    r.maxDiff = maxDiff(python.out.view, cuda.out.view);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const RGBA a = python.out.view.load(x, y), b = clip.frames[0].view.load(x, y);
            r.changed += std::fabs(a.r - b.r) + std::fabs(a.g - b.g) + std::fabs(a.b - b.b) > 0.02f;
        }
    for (int y = 0; y < h; ++y) {
        bool over = false;
        for (int x = 0; x < w && !over; ++x) {
            const RGBA a = python.out.view.load(x, y), b = cuda.out.view.load(x, y);
            over = std::fabs(a.r - b.r) > 1.0 / 255 || std::fabs(a.g - b.g) > 1.0 / 255 || std::fabs(a.b - b.b) > 1.0 / 255;
        }
        if (!over) continue;
        ++r.rowsOver;
        bool near = false;
        const int top = h - 1 - y;
        for (int k = 0; crossings && k <= crossings && !near; ++k) near = std::fabs(top - double(k) * h / crossings) <= 8;
        r.rowsOverAwayFromCrossings += !near;
    }
    std::printf("  %s preset %d CUDA vs Python max diff %.5f, rows over 1/255: %d (%d away from sin zero crossings), %d px changed by the effect\n",
                id, preset, r.maxDiff, r.rowsOver, r.rowsOverAwayFromCrossings, r.changed);
    return r;
}

TP_TEST(cutwright_classic_crt_cuda_matches_python) {
    if (!gpu::available()) { std::printf("  (no CUDA device: Cutwright parity skipped)\n"); return; }
    CHECK(cutwrightGpuVsPython("crt_analog", 0, 640, 360).maxDiff < 1.0 / 255);
    CHECK(cutwrightGpuVsPython("crt_analog", 0, 1920, 1080).maxDiff < 1.0 / 255);
}

TP_TEST(cutwright_analog_presets_cuda_match_python) {
    if (!gpu::available()) { std::printf("  (no CUDA device: Cutwright parity skipped)\n"); return; }
    for (int preset = 1; preset < 8; ++preset) {
        if (preset == 5) {
            const Parity r = cutwrightGpuVsPython("crt_analog", preset, 640, 360, 64);
            CHECK(r.rowsOverAwayFromCrossings == 0);
        } else {
            CHECK(cutwrightGpuVsPython("crt_analog", preset, 640, 360).maxDiff < 1.0 / 255);
        }
    }
}

TP_TEST(cutwright_mosh_presets_cuda_match_python) {
    if (!gpu::available()) { std::printf("  (no CUDA device: Cutwright parity skipped)\n"); return; }
    const ModuleDef& m = find("cutwright_mosh");
    const int w = 480, h = 270;
    const Clip clip = makeClip(w, h, 12, [w, h](int x, int y, int f) {
        const float u = float(x - 4 * f) / w, v = float(y - 2 * f) / h;
        const float ring = 0.5f + 0.5f * std::sin(30.f * std::hypot(u - 0.45f, v - 0.5f));
        return RGBA{0.1f + 0.8f * ring * (0.5f + 0.5f * u), 0.15f + 0.7f * v, 0.85f - 0.6f * ring, 1};
    });
    for (int preset = 0; preset < 4; ++preset) {
        const auto p = with(defaultValues(m), {{"preset", double(preset)}, {"resolution", 2}});
        gpu::setEnabled(false);
        const RenderResult python = renderAt(m, clip, 10, p);
        gpu::setEnabled(true);
        const auto started = std::chrono::steady_clock::now();
        const RenderResult cuda = renderAt(m, clip, 10, p);
        const double cudaMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        const double diff = maxDiff(python.out.view, cuda.out.view);
        int over = 0;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const RGBA a = python.out.view.load(x, y), b = cuda.out.view.load(x, y);
                over += std::fabs(a.r - b.r) > 1.0 / 255 || std::fabs(a.g - b.g) > 1.0 / 255 || std::fabs(a.b - b.b) > 1.0 / 255;
            }
        std::printf("  cutwright_mosh preset %d CUDA vs Python max diff %.5f, %d px over 1/255, CUDA render %.0f ms\n", preset, diff, over, cudaMs);
        CHECK(diff < 1.0 / 255);
    }
}

TP_TEST(cutwright_art_presets_match_python) {
    if (!gpu::available()) { std::printf("  (no CUDA device: Cutwright parity skipped)\n"); return; }
    for (int preset : {3, 4, 5, 6}) CHECK(cutwrightGpuVsPython("cutwright_art", preset, 640, 360).maxDiff < 1.0 / 255);
}

TP_TEST(cutwright_star_points_match_python) {
    if (!gpu::available()) { std::printf("  (no CUDA device: Cutwright parity skipped)\n"); return; }
    const ModuleDef& m = find("cutwright_art");
    const int w = 640, h = 360;
    const Clip clip = makeClip(w, h, 1, [](int x, int y, int) {
        const bool dot = ((x % 97) < 3 && (y % 61) < 3) || (std::abs(x - 300) < 2 && std::abs(y - 200) < 2);
        const float base = 0.08f + 0.1f * float(x) / 640;
        return dot ? RGBA{1.f, 0.95f, 0.8f, 1} : RGBA{base, base * 0.8f, base * 1.2f, 1};
    });
    const auto p = with(defaultValues(m), {{"preset", 3}, {"resolution", 2}});
    gpu::setEnabled(false);
    const RenderResult python = renderAt(m, clip, 0, p);
    gpu::setEnabled(true);
    const RenderResult native = renderAt(m, clip, 0, p);
    double armEnergy = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) armEnergy += python.out.view.load(x, y).r - clip.frames[0].view.load(x, y).r;
    const double diff = maxDiff(python.out.view, native.out.view);
    std::printf("  star points: arm energy %.1f, native vs Python max diff %.5f\n", armEnergy, diff);
    CHECK(armEnergy > 10);
    CHECK(diff < 1.0 / 255);
}

TP_TEST(numpy_generator_sequence_matches_numpy) {
    const struct { uint32_t seed, index; double v[6]; } refs[] = {
        {1, 0, {0.5118216247002567, 0.9009273926518706, 0.028603990317990845, 1.0, 1.0, 0.31183145201048545}},
        {1, 7, {0.3596767659247476, 0.23309273157079669, 0.04111953166170955, 1.0, -1.0, 0.8717288553088381}},
        {5, 123, {0.94125453667637, 0.6796803090560009, 0.04155825936539478, -1.0, 1.0, 0.25437402588808977}},
    };
    for (const auto& r : refs) {
        tp::NumpyPcg64 rng(std::vector<uint32_t>{r.seed, r.index});
        const double got[6] = {rng.nextDouble(), rng.uniform(-1, 1), rng.uniform(0.025, 0.05),
                               rng.boundedLemire(1) ? 1.0 : -1.0, rng.boundedLemire(1) ? 1.0 : -1.0, rng.nextDouble()};
        for (int i = 0; i < 6; ++i) CHECK(got[i] == r.v[i]);
    }
}

TP_TEST(cutwright_cell_vhs_matches_python_statistically) {
    if (!gpu::available()) { std::printf("  (no CUDA device: Cutwright parity skipped)\n"); return; }
    const ModuleDef& m = find("cutwright_art");
    const int w = 640, h = 360;
    const Clip clip = makeClip(w, h, 5, [w, h](int x, int y, int) {
        const float u = float(x) / w, v = float(y) / h;
        const float ring = 0.5f + 0.5f * std::sin(25.f * std::hypot(u - 0.4f, v - 0.55f));
        return RGBA{0.2f + 0.6f * u * ring, 0.15f + 0.7f * v, 0.8f - 0.5f * u * v, 1};
    });
    const auto p = with(defaultValues(m), {{"preset", 2}, {"resolution", 2}});
    gpu::setEnabled(false);
    const RenderResult python = renderAt(m, clip, 3, p);
    gpu::setEnabled(true);
    const RenderResult cuda = renderAt(m, clip, 3, p);
    const int r = 6;
    double worstBlock = 0, sumDiff = 0, sumSqPy = 0, sumSqCu = 0;
    std::vector<double> dpy(size_t(w) * h), dcu(size_t(w) * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const RGBA a = python.out.view.load(x, y), b = cuda.out.view.load(x, y);
            const double la = 0.2126 * a.r + 0.7152 * a.g + 0.0722 * a.b, lb = 0.2126 * b.r + 0.7152 * b.g + 0.0722 * b.b;
            dpy[size_t(y) * w + x] = la;
            dcu[size_t(y) * w + x] = lb;
            sumDiff += lb - la;
        }
    const int band = 14;
    int blocksOver = 0, blocks = 0;
    double bandPy = 0, bandCu = 0;
    for (int y = 0; y < band; ++y)
        for (int x = 0; x < w; ++x) { bandPy += dpy[size_t(y) * w + x]; bandCu += dcu[size_t(y) * w + x]; }
    for (int y = band + r; y < h - r; y += r)
        for (int x = r; x < w - r; x += r) {
            double sa = 0, sb = 0;
            for (int dy = -r; dy <= r; ++dy)
                for (int dx = -r; dx <= r; ++dx) { sa += dpy[size_t(y + dy) * w + x + dx]; sb += dcu[size_t(y + dy) * w + x + dx]; }
            const double k = double((2 * r + 1) * (2 * r + 1));
            worstBlock = std::max(worstBlock, std::fabs(sa - sb) / k);
            blocksOver += std::fabs(sa - sb) / k > 0.02;
            ++blocks;
            for (int dy = -r; dy <= r; ++dy)
                for (int dx = -r; dx <= r; ++dx) {
                    sumSqPy += std::pow(dpy[size_t(y + dy) * w + x + dx] - sa / k, 2);
                    sumSqCu += std::pow(dcu[size_t(y + dy) * w + x + dx] - sb / k, 2);
                }
        }
    const double bandDiff = (bandCu - bandPy) / (double(band) * w);
    std::printf("  cell vhs: mean luma diff %.5f, worst 13x13 block mean diff above the band %.5f, band mean diff %.5f, local texture ratio %.3f\n",
                sumDiff / (double(w) * h), worstBlock, bandDiff, std::sqrt(sumSqCu / sumSqPy));
    CHECK(std::fabs(sumDiff / (double(w) * h)) < 0.003);
    CHECK(std::fabs(bandDiff) < 0.03);
    std::printf("  cell vhs: %d of %d blocks over 0.02\n", blocksOver, blocks);
    CHECK(worstBlock < 0.06);
    CHECK(blocksOver <= 6);
    CHECK(std::fabs(std::sqrt(sumSqCu / sumSqPy) - 1) < 0.1);
}

TP_TEST(cutwright_mono_ink_matches_python) {
    if (!gpu::available()) { std::printf("  (no CUDA device: Cutwright parity skipped)\n"); return; }
    const ModuleDef& m = find("cutwright_art");
    const int w = 480, h = 270;
    const Clip clip = makeClip(w, h, 6, [w, h](int x, int y, int f) {
        const float u = float(x - 3 * f) / w, v = float(y - f) / h;
        const float ring = 0.5f + 0.5f * std::sin(24.f * std::hypot(u - 0.45f, v - 0.5f));
        return RGBA{0.1f + 0.8f * ring * (0.4f + 0.6f * u), 0.12f + 0.7f * v * ring, 0.85f - 0.6f * ring, 1};
    });
    const auto p = with(defaultValues(m), {{"preset", 0}, {"resolution", 2}});
    gpu::setEnabled(false);
    const RenderResult python = renderAt(m, clip, 4, p);
    gpu::setEnabled(true);
    const RenderResult cuda = renderAt(m, clip, 4, p);
    const double diff = maxDiff(python.out.view, cuda.out.view);
    int over = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const RGBA a = python.out.view.load(x, y), b = cuda.out.view.load(x, y);
            over += std::fabs(a.r - b.r) > 1.0 / 255 || std::fabs(a.g - b.g) > 1.0 / 255 || std::fabs(a.b - b.b) > 1.0 / 255;
        }
    std::printf("  mono ink: max diff %.5f, %d px over 1/255\n", diff, over);
    CHECK(diff < 1.0 / 255);
}

TP_TEST(cutwright_motion_ink_matches_python) {
    if (!gpu::available()) { std::printf("  (no CUDA device: Cutwright parity skipped)\n"); return; }
    const ModuleDef& m = find("cutwright_art");
    const int w = 640, h = 360;
    const Clip clip = makeClip(w, h, 10, [w, h](int x, int y, int f) {
        const float u = float(x) / w, v = float(y) / h;
        const bool box = std::abs(x - (200 + 14 * f)) < 40 && std::abs(y - (150 + 5 * f)) < 30;
        const bool disc = std::hypot(float(x - (450 - 10 * f)), float(y - 250)) < 25;
        if (box) return RGBA{0.9f, 0.85f, 0.2f, 1};
        if (disc) return RGBA{0.2f, 0.6f, 0.95f, 1};
        return RGBA{0.15f + 0.2f * u, 0.18f + 0.15f * v, 0.22f, 1};
    });
    const auto p = with(defaultValues(m), {{"preset", 1}, {"resolution", 2}});
    gpu::setEnabled(false);
    const RenderResult python = renderAt(m, clip, 8, p);
    gpu::setEnabled(true);
    const RenderResult native = renderAt(m, clip, 8, p);
    const double diff = maxDiff(python.out.view, native.out.view);
    int over = 0, lit = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const RGBA a = python.out.view.load(x, y), b = native.out.view.load(x, y);
            over += std::fabs(a.r - b.r) > 1.0 / 255 || std::fabs(a.g - b.g) > 1.0 / 255 || std::fabs(a.b - b.b) > 1.0 / 255;
            lit += a.r + a.g + a.b > 0.05f;
        }
    std::printf("  motion ink: max diff %.5f, %d px over 1/255, %d lit px in Python\n", diff, over, lit);
    CHECK(lit > 100);
    CHECK(diff < 1.0 / 255);
}

TP_TEST(cutwright_geometric_presets_match_python) {
    if (!gpu::available()) { std::printf("  (no CUDA device: Cutwright parity skipped)\n"); return; }
    for (int preset : {2, 4, 5, 6, 7}) CHECK(cutwrightGpuVsPython("geometric", preset, 640, 360).maxDiff < 1.0 / 255);
    const Parity field2 = cutwrightGpuVsPython("geometric", 2, 1920, 1080);
    std::printf("  contours 1080: %d px changed, max diff %.5f\n", field2.changed, field2.maxDiff);
    CHECK(field2.changed > 1000);
    CHECK(field2.maxDiff < 0.04);
    CHECK(field2.rowsOver < 150);
}

TP_TEST(cutwright_curtain_ridges_match_python) {
    if (!gpu::available()) { std::printf("  (no CUDA device: Cutwright parity skipped)\n"); return; }
    const ModuleDef& m = find("geometric");
    const int w = 640, h = 360;
    const Clip clip = makeClip(w, h, 1, [](int x, int y, int) {
        const bool band = (y > 120 && y < 170) || (y > 250 && y < 260 && x > 200);
        const float u = float(x) / 640;
        const float v = 0.05f + 0.25f * float(y) / 360;
        return band ? RGBA{0.9f, 0.7f * u + 0.2f, 0.3f, 1} : RGBA{v, v + 0.02f, v + 0.1f * u, 1};
    });
    const auto p = with(defaultValues(m), {{"preset", 6}, {"resolution", 2}});
    gpu::setEnabled(false);
    const RenderResult python = renderAt(m, clip, 0, p);
    gpu::setEnabled(true);
    const RenderResult native = renderAt(m, clip, 0, p);
    int changed = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const RGBA a = python.out.view.load(x, y), b = clip.frames[0].view.load(x, y);
            changed += std::fabs(a.r - b.r) + std::fabs(a.g - b.g) + std::fabs(a.b - b.b) > 0.02f;
        }
    const double diff = maxDiff(python.out.view, native.out.view);
    std::printf("  curtain ridges: %d px changed, native vs Python max diff %.5f\n", changed, diff);
    CHECK(changed > 1000);
    CHECK(diff < 1.0 / 255);
}

static void geoFieldParity(int w, int h, int preset = 0, float dim = 0.65f) {
    const ModuleDef& m = find("geometric");
    const Clip clip = makeClip(w, h, 14, [w, h](int x, int y, int f) {
        const float u = float(x) / w, v = float(y) / h;
        const bool box = std::abs(x - (180 + 9 * f) * w / 640) < 50 * w / 640 && std::abs(y - (160 + 3 * f) * h / 360) < 35 * h / 360;
        const bool disc = std::hypot(float(x - (470 - 6 * f) * w / 640), float(y - 240 * h / 360)) < 40.f * h / 360;
        const float ring = 0.5f + 0.5f * std::sin(25.f * std::hypot(u - 0.3f - 0.004f * f, v - 0.6f));
        if (box) return RGBA{0.9f, 0.8f, 0.25f, 1};
        if (disc) return RGBA{0.2f, 0.55f, 0.95f, 1};
        return RGBA{0.12f + 0.3f * u * ring, 0.15f + 0.2f * v, 0.25f + 0.1f * ring, 1};
    });
    const auto p = with(defaultValues(m), {{"preset", double(preset)}, {"resolution", 2}});
    gpu::setEnabled(false);
    const RenderResult python = renderAt(m, clip, 13, p);
    gpu::setEnabled(true);
    const RenderResult native = renderAt(m, clip, 13, p);
    const double diff = maxDiff(python.out.view, native.out.view);
    int over = 0, changed = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const RGBA a = python.out.view.load(x, y), b = native.out.view.load(x, y), s = clip.frames[13].view.load(x, y);
            over += std::fabs(a.r - b.r) > 1.0 / 255 || std::fabs(a.g - b.g) > 1.0 / 255 || std::fabs(a.b - b.b) > 1.0 / 255;
            changed += std::fabs(a.r - dim * s.r) + std::fabs(a.g - dim * s.g) + std::fabs(a.b - dim * s.b) > 0.05f;
        }
    std::printf("  geo preset %d %dx%d: max diff %.5f, %d px over 1/255, %d px drawn in Python\n", preset, w, h, diff, over, changed);
    CHECK(changed > 500);
    CHECK(diff < 1.0 / 255);
    renderAt(m, clip, 12, p);
    const RenderResult stepped = renderAt(m, clip, 13, p);
    const double carried = maxDiff(native.out.view, stepped.out.view);
    std::printf("  geo preset %d %dx%d: one-step continuation vs full replay max diff %.7f\n", preset, w, h, carried);
    CHECK(carried < 1e-6);
}

TP_TEST(cutwright_geo_field_tracks_like_python) {
    if (!gpu::available()) { std::printf("  (no CUDA device: Cutwright parity skipped)\n"); return; }
    geoFieldParity(640, 360);
    geoFieldParity(1920, 1080);
}

TP_TEST(cutwright_constellation_tracks_like_python) {
    if (!gpu::available()) { std::printf("  (no CUDA device: Cutwright parity skipped)\n"); return; }
    geoFieldParity(640, 360, 1, 0.8f);
    geoFieldParity(1920, 1080, 1, 0.8f);
}

TP_TEST(cutwright_pixel_memory_tracks_like_python) {
    if (!gpu::available()) { std::printf("  (no CUDA device: Cutwright parity skipped)\n"); return; }
    geoFieldParity(640, 360, 3, 0.4f);
    geoFieldParity(1920, 1080, 3, 0.4f);
}

TP_TEST(cutwright_buffers_and_frame_cache_stay_bounded) {
    {
        tp::cw::Pooled<std::vector<float>> a, b, c;
        a.item.resize(16);
        b.item.resize(16);
        c.item.resize(16);
    }
    CHECK(tp::cw::Pooled<std::vector<float>>::idleCount() <= 2);
    std::vector<float> v(4096);
    tp::cw::fit(v, 16);
    CHECK(v.size() == 16 && v.capacity() <= 32);
    const int w = 3840, h = 2160;
    for (int i = 0; i < 16; ++i) {
        auto f = std::make_shared<tp::cwgeo::GeoFrame>();
        f->w = w;
        f->h = h;
        f->index = 1000000 + i;
        f->fy = tp::cw::Plane(h, w);
        f->fx = tp::cw::Plane(h, w);
        tp::cwgeo::keepGeoFrame(f);
    }
    std::printf("  4K frame cache holds %.0f MB\n", tp::cwgeo::geoFramesHeld() / 1048576.0);
    CHECK(tp::cwgeo::geoFramesHeld() <= tp::cwgeo::geoFrameBudget);
    CHECK(tp::cwgeo::findGeoFrame(w, h, 1000015, 0) != nullptr);
    CHECK(tp::cwgeo::findGeoFrame(w, h, 1000009, 0) != nullptr);
    CHECK(tp::cwgeo::findGeoFrame(w, h, 1000000, 0) == nullptr);
}
