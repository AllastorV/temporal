#include "harness.h"
#include "gpu/gpu.h"
#include <cuda_runtime.h>
using namespace tpt;

TP_TEST(native_cuda_motion_matches_cpu_with_pitched_rows_and_mix) {
    if (!gpu::available()) return;
    const auto& m=find("camera_motion");
    auto clip=patternClip(75,49,8);
    const auto& input=clip.frames[7];
    auto p=with(defaultValues(m),{{"mode",4},{"start_frame",0},{"duration",7},{"mix",0.64},{"encoding",1}});
    gpu::setEnabled(false);
    auto expected=renderAt(m,clip,7,p);
    gpu::setEnabled(true);
    struct Buffer { void* ptr=nullptr; ~Buffer(){if(ptr)cudaFree(ptr);} } src,dst;
    struct Stream { cudaStream_t s=nullptr; ~Stream(){if(s)cudaStreamDestroy(s);} } stream;
    size_t pitch=0,outPitch=0,bytes=75*16;
    CHECK(cudaMallocPitch(&src.ptr,&pitch,bytes,49)==cudaSuccess);
    CHECK(cudaMallocPitch(&dst.ptr,&outPitch,bytes,49)==cudaSuccess);
    CHECK(cudaStreamCreate(&stream.s)==cudaSuccess);
    CHECK(cudaMemcpy2DAsync(src.ptr,pitch,input.view.data,input.view.rowBytes,bytes,49,cudaMemcpyHostToDevice,stream.s)==cudaSuccess);
    RenderContext c;
    c.params=&p;c.time=7;c.last=7;c.frame=c.window=clip.rect;c.cudaRender=true;c.cudaStream=stream.s;
    c.encoding=Encoding::Gamma24;c.dst={static_cast<float*>(dst.ptr),clip.rect,ptrdiff_t(outPitch)};
    c.fetchRaw=[&](double){return ImageView{static_cast<float*>(src.ptr),clip.rect,ptrdiff_t(pitch)};};
    runModule(m,c);
    OwnedImage actual(clip.rect);
    CHECK(cudaMemcpy2DAsync(actual.view.data,actual.view.rowBytes,dst.ptr,outPitch,bytes,49,cudaMemcpyDeviceToHost,stream.s)==cudaSuccess);
    CHECK(cudaStreamSynchronize(stream.s)==cudaSuccess);
    CHECK(maxDiff(actual.view,expected.out.view)<0.001);
}

namespace {

struct DeviceFrames {
    std::vector<void*> ptrs;
    size_t pitch = 0;
    Rect rect;
    explicit DeviceFrames(const Clip& clip) : rect(clip.rect) {
        const size_t bytes = size_t(rect.w()) * 16;
        for (const auto& f : clip.frames) {
            void* p = nullptr;
            CHECK(cudaMallocPitch(&p, &pitch, bytes, rect.h()) == cudaSuccess);
            ptrs.push_back(p);
            CHECK(cudaMemcpy2D(p, pitch, f.view.data, f.view.rowBytes, bytes, rect.h(), cudaMemcpyHostToDevice) == cudaSuccess);
        }
    }
    ~DeviceFrames() { for (void* p : ptrs) cudaFree(p); }
    ImageView view(size_t i) const { return {static_cast<float*>(ptrs[i]), rect, ptrdiff_t(pitch)}; }
};

bool allFinite(const ImageView& a) {
    for (int y = a.bounds.y1; y < a.bounds.y2; ++y)
        for (int x = a.bounds.x1; x < a.bounds.x2; ++x)
            for (int c = 0; c < 4; ++c) if (!std::isfinite(a.px(x, y)[c])) return false;
    return true;
}

OwnedImage renderMode(const ModuleDef& m, const Clip& clip, const DeviceFrames* dev, double t, const ParamValues& p, bool preview) {
    OwnedImage out(clip.rect);
    struct Buffer { void* ptr = nullptr; ~Buffer() { if (ptr) cudaFree(ptr); } } dst;
    struct Stream { cudaStream_t s = nullptr; ~Stream() { if (s) cudaStreamDestroy(s); } } stream;
    size_t pitch = 0;
    const size_t bytes = size_t(clip.rect.w()) * 16;
    RenderContext c;
    c.params = &p; c.time = t; c.fps = 24; c.first = clip.first; c.last = clip.last(); c.scale = 1;
    c.frame = c.window = clip.rect; c.preview = preview;
    if (p.v.count("edge")) c.edge = static_cast<EdgeMode>(p.i("edge"));
    if (p.v.count("encoding")) c.encoding = static_cast<Encoding>(p.i("encoding"));
    c.fetchRaw = [&](double ft) -> ImageView {
        const int f = static_cast<int>(std::lround(ft));
        if (f < clip.first || f > clip.last()) return {};
        const size_t i = size_t(f - clip.first);
        return dev ? dev->view(i) : clip.frames[i].view;
    };
    if (dev) {
        CHECK(cudaMallocPitch(&dst.ptr, &pitch, bytes, clip.rect.h()) == cudaSuccess);
        CHECK(cudaMemset2D(dst.ptr, pitch, 0xff, bytes, clip.rect.h()) == cudaSuccess);
        CHECK(cudaStreamCreate(&stream.s) == cudaSuccess);
        c.cudaRender = true; c.cudaStream = stream.s;
        c.dst = {static_cast<float*>(dst.ptr), clip.rect, ptrdiff_t(pitch)};
    } else c.dst = out.view;
    runModule(m, c);
    if (dev) CHECK(cudaMemcpy2D(out.view.data, out.view.rowBytes, dst.ptr, pitch, bytes, clip.rect.h(), cudaMemcpyDeviceToHost) == cudaSuccess);
    return out;
}

}

TP_TEST(cuda_render_matches_host_for_every_cuda_module) {
    if (!gpu::available()) return;
    auto clip = patternClip(83, 51, 12);
    const DeviceFrames dev(clip);
    int modules = 0;
    for (const ModuleDef* m : allModules()) {
        if (m->cuda == CudaMode::None) continue;
        ++modules;
        for (double mix : {1.0, 0.45, 0.0}) {
            for (bool preview : {false, true}) {
                auto p = defaultValues(*m);
                if (p.v.count("mix")) p.v["mix"][0] = mix;
                if (p.v.count("encoding")) p.v["encoding"][0] = 1;
                const auto expected = renderMode(*m, clip, nullptr, 10, p, preview);
                const auto actual = renderMode(*m, clip, &dev, 10, p, preview);
                const double d = maxDiff(actual.view, expected.view);
                if (!allFinite(actual.view)) throw std::runtime_error(m->id + " left unwritten device pixels");
                if (!(d <= (m->cuda == CudaMode::Native ? 1e-3 : 1e-5)))
                    throw std::runtime_error(m->id + " mix " + std::to_string(mix) + (preview ? " preview" : "") + " differs by " + std::to_string(d));
            }
        }
    }
    CHECK(modules == 15);
}

TP_TEST(cuda_render_keeps_device_modules_on_the_gpu) {
    if (!gpu::available()) return;
    auto clip = patternClip(83, 51, 12);
    const DeviceFrames dev(clip);
    for (const ModuleDef* m : allModules()) {
        if (m->cuda != CudaMode::Device && m->cuda != CudaMode::Bridged) continue;
        auto p = defaultValues(*m);
        const int before = cudaBridgeCount();
        renderMode(*m, clip, &dev, 10, p, false);
        if ((cudaBridgeCount() != before) != (m->cuda == CudaMode::Bridged))
            throw std::runtime_error(m->id + (m->cuda == CudaMode::Bridged ? " skipped" : " fell back to") + " the host bridge");
    }
}

TP_TEST(cuda_render_matches_host_for_every_option_and_look) {
    if (!gpu::available()) return;
    auto clip = patternClip(83, 51, 12);
    const DeviceFrames dev(clip);
    std::vector<std::string> failures;
    size_t renders = 0;
    for (const ModuleDef* m : allModules()) {
        if (m->cuda == CudaMode::None) continue;
        for (const auto& c : sweepCases(*m)) {
            ++renders;
            try {
                const auto expected = renderMode(*m, clip, nullptr, 10, c.params, false);
                const auto actual = renderMode(*m, clip, &dev, 10, c.params, false);
                const double d = maxDiff(actual.view, expected.view);
                if (!allFinite(actual.view) || !(d <= (m->cuda == CudaMode::Native ? 1e-3 : 1e-5)))
                    failures.push_back(m->id + " [" + c.label + "] differs by " + std::to_string(d));
            } catch (const std::exception& e) {
                failures.push_back(m->id + " [" + c.label + "]: " + e.what());
            }
        }
    }
    std::printf("  cuda sweep: %zu renders, %zu failures\n", renders, failures.size());
    for (size_t i = 0; i < failures.size() && i < 60; ++i) std::printf("  FAIL %s\n", failures[i].c_str());
    CHECK(failures.empty());
}
