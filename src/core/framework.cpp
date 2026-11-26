#include "module.h"
#include "../gpu/gpu.h"
#include <mutex>
#ifdef _WIN32
#include <windows.h>
#endif

namespace tp {

namespace {
struct BlockPool {
    std::mutex lock;
    std::vector<std::pair<size_t, void*>> blocks;
    size_t held = 0;
};
BlockPool& blockPool() {
    static BlockPool* pool = new BlockPool;
    return *pool;
}
constexpr size_t kPoolMinBytes = size_t(1) << 20, kPoolCapBytes = size_t(1) << 30;
}

void* poolAllocate(size_t bytes) {
    if (bytes >= kPoolMinBytes) {
        BlockPool& pool = blockPool();
        std::lock_guard<std::mutex> guard(pool.lock);
        for (size_t i = 0; i < pool.blocks.size(); ++i) {
            const size_t have = pool.blocks[i].first;
            if (have < bytes || have > bytes + bytes / 4) continue;
            void* p = pool.blocks[i].second;
            pool.held -= have;
            pool.blocks.erase(pool.blocks.begin() + static_cast<std::ptrdiff_t>(i));
            return p;
        }
    }
    return ::operator new(bytes);
}

void poolRelease(void* p, size_t bytes) {
    if (!p) return;
    if (bytes >= kPoolMinBytes && bytes <= kPoolCapBytes) {
        BlockPool& pool = blockPool();
        std::lock_guard<std::mutex> guard(pool.lock);
        while (!pool.blocks.empty() && pool.held + bytes > kPoolCapBytes) {
            ::operator delete(pool.blocks.front().second);
            pool.held -= pool.blocks.front().first;
            pool.blocks.erase(pool.blocks.begin());
        }
        pool.blocks.emplace_back(bytes, p);
        pool.held += bytes;
        return;
    }
    ::operator delete(p);
}

void runRowWorkers(unsigned count, void* context, void (*run)(void*)) {
#ifdef _WIN32
    struct Job { void* context; void (*run)(void*); };
    Job job{context, run};
    auto callback = [](PTP_CALLBACK_INSTANCE, void* data, PTP_WORK) {
        auto& item = *static_cast<Job*>(data);
        item.run(item.context);
    };
    PTP_WORK work = CreateThreadpoolWork(callback, &job, nullptr);
    if (!work) { run(context); return; }
    for (unsigned i = 1; i < count; ++i) SubmitThreadpoolWork(work);
    run(context);
    // The caller has exhausted the row queue. Cancel unstarted callbacks and join active ones.
    WaitForThreadpoolWorkCallbacks(work, TRUE);
    CloseThreadpoolWork(work);
#else
    std::vector<std::jthread> workers;
    for (unsigned i = 1; i < count; ++i) workers.emplace_back(run, context);
    run(context);
#endif
}

void addFrameworkParams(ModuleDef& m) {
    if (m.temporal)
        m.params.push_back(Choice("edge", "Frame Edges", {"Clamp", "Mirror", "Transparent"}, 0,
                                  "What to use when a sample falls before the first or after the last frame of the clip.", "Global"));
    if (m.usesEncoding)
        m.params.push_back(Choice("encoding", "Input Encoding", {"Linear", "Rec.709 (Gamma 2.4)", "DaVinci Intermediate", "ACEScct"}, 1,
                                  "Encoding of the incoming image. Exposure and optical math is done in linear light and re-encoded on output.", "Global"));
    m.params.push_back(Dbl("mix", "Mix", 1.0, 0.0, 1.0, 0.0, 1.0, Unit::Normalized, "Blend between the untouched source (0) and the full effect (1).", "Global"));
}

namespace {
int previewDivisor(const ModuleDef& m, const RenderContext& c) {
    int limit = 960;
    if (m.id == "stutter" || m.id == "camera_motion" || m.id == "procedural_motion" ||
        m.id == "blob_tracking" || m.id == "halation" || m.id == "diffusion_lab" || m.id == "refraction_lab" ||
        m.id == "crt_analog" || m.id == "geometric" || m.id == "cutwright_art" || m.id == "cutwright_mosh") return 1;
    if (m.id == "pixel" && c.p().i("mode") == 0) return 1;
    if (m.id == "print_lab" && c.p().i("preview_quality") == 1) return 1;
    if (m.id == "motion_tear" || m.id == "datamosh" || m.id == "echo" ||
        m.id == "slow_shutter" || m.id == "time_shift") limit = 640;
    if (m.id == "time_displacement" || (m.id == "echo" && c.p().i("mode") != 0)) limit = 480;
    return std::max(1, (std::max(c.frame.w(), c.frame.h()) + limit - 1) / limit);
}

OwnedImage reduceForPreview(const ImageView& src, Rect frame, Rect reducedBounds, int step, bool fitToFrame = false) {
    if (!src.valid()) return {};
    OwnedImage reduced(reducedBounds);
    const int a = step / 4, b = (3 * step) / 4;
    const bool full = src.bounds.contains(frame.x1, frame.y1) && src.bounds.contains(frame.x2-1, frame.y2-1);
    const bool remap = fitToFrame && (!full || src.bounds.w() != frame.w() || src.bounds.h() != frame.h());
    const float scaleX = static_cast<float>(src.bounds.w()) / frame.w();
    const float scaleY = static_cast<float>(src.bounds.h()) / frame.h();
    const auto get = [&](int sx, int sy) {
        if (remap) {
            return src.bilinear(src.bounds.x1 + (sx - frame.x1 + 0.5f) * scaleX,
                                src.bounds.y1 + (sy - frame.y1 + 0.5f) * scaleY);
        }
        return full ? src.load(sx, sy) : src.at(sx, sy, SampleEdge::Black);
    };
    parallelRows(0, reducedBounds.h(), [&](int y) {
        int ya = std::min(frame.y2-1, frame.y1 + y*step + a);
        int yb = std::min(frame.y2-1, frame.y1 + y*step + b);
        for (int x=0; x<reducedBounds.w(); ++x) {
            int xa = std::min(frame.x2-1, frame.x1 + x*step + a);
            int xb = std::min(frame.x2-1, frame.x1 + x*step + b);
            reduced.view.store(x,y,(get(xa,ya)+get(xb,ya)+get(xa,yb)+get(xb,yb))*0.25f);
        }
    });
    return reduced;
}

void renderPreview(const ModuleDef& module, const RenderContext& c, int step, float mix) {
    Rect reducedBounds{0,0,(c.frame.w()+step-1)/step,(c.frame.h()+step-1)/step};
    OwnedImage output(reducedBounds);
    auto second = reduceForPreview(c.secondSource,c.frame,reducedBounds,step,module.id == "lens_character");
    std::map<double,OwnedImage> sources;
    RenderContext draft = c;
    draft.frame = draft.window = reducedBounds;
    draft.dst = output.view;
    draft.secondSource = second.view;
    draft.scale /= step;
    draft.fetchRaw = [&](double time) {
        auto [it, inserted] = sources.try_emplace(time);
        if (inserted) it->second = reduceForPreview(c.fetchRaw(time),c.frame,reducedBounds,step);
        return it->second.view;
    };
    draft.releaseRaw = [&](double time) {
        sources.erase(time);
        if (c.releaseRaw) c.releaseRaw(time);
    };
    module.render(draft);
    if (c.isAborted()) return;
    const ImageView original = mix < 1 ? c.source() : ImageView{};
    expandPreview(c,output.view,1.f/step,1.f/step,original,mix);
}
}

void expandPreview(const RenderContext& c, const ImageView& source, float scaleX, float scaleY, const ImageView& original, float mix) {
    struct Column { int left, right; float fraction; };
    std::vector<Column> columns(c.window.w());
    for (int x=c.window.x1; x<c.window.x2; ++x) {
        const float u = (x-c.frame.x1+0.5f)*scaleX-0.5f;
        const int i = static_cast<int>(std::floor(u));
        columns[x-c.window.x1] = {std::clamp(i,0,source.bounds.w()-1),std::clamp(i+1,0,source.bounds.w()-1),u-i};
    }
    parallelRows(c.window.y1,c.window.y2,[&](int y) {
        const float v = (y-c.frame.y1+0.5f)*scaleY-0.5f;
        const int j = static_cast<int>(std::floor(v));
        const int top=std::clamp(j,0,source.bounds.h()-1), bottom=std::clamp(j+1,0,source.bounds.h()-1);
        for (int x=c.window.x1; x<c.window.x2; ++x) {
            auto col=columns[x-c.window.x1];
            RGBA a=lerp(source.load(col.left,top),source.load(col.right,top),col.fraction);
            RGBA b=lerp(source.load(col.left,bottom),source.load(col.right,bottom),col.fraction);
            RGBA result=lerp(a,b,v-j);
            if (mix < 1) result=lerp(original.at(x,y,SampleEdge::Black),result,mix);
            c.dst.store(x,y,result);
        }
    });
}

namespace {
std::atomic<int> gBridged{0};

void cudaCopy(const ImageView& from, const ImageView& to, const Rect& r, void* stream) {
    if (!gpu::copyImage(from, to, r, stream)) throw std::runtime_error("CUDA image copy failed");
}

void renderBridged(const ModuleDef& m, const RenderContext& c) {
    ++gBridged;
    std::mutex lock;
    std::map<double, OwnedImage> frames;
    RenderContext h = c;
    h.cudaRender = false;
    h.cudaStream = nullptr;
    OwnedImage out(c.dst.bounds), second;
    if (c.secondSource.valid()) {
        second.reset(c.secondSource.bounds);
        cudaCopy(c.secondSource, second.view, second.view.bounds, c.cudaStream);
        h.secondSource = second.view;
    }
    h.dst = out.view;
    h.fetchRaw = [&](double t) {
        std::lock_guard<std::mutex> guard(lock);
        auto it = frames.find(t);
        if (it == frames.end()) {
            const ImageView d = c.fetchRaw(t);
            OwnedImage copy;
            if (d.valid()) {
                copy.reset(d.bounds);
                cudaCopy(d, copy.view, d.bounds, c.cudaStream);
            }
            it = frames.emplace(t, std::move(copy)).first;
        }
        return it->second.view;
    };
    h.releaseRaw = [&](double t) {
        {
            std::lock_guard<std::mutex> guard(lock);
            frames.erase(t);
        }
        if (c.releaseRaw) c.releaseRaw(t);
    };
    runModule(m, h);
    if (c.isAborted()) return;
    cudaCopy(out.view, c.dst, c.window, c.cudaStream);
}

void runDevice(const ModuleDef& m, const RenderContext& ctx) {
    if (!gpu::syncStream(ctx.cudaStream)) throw std::runtime_error("CUDA stream synchronize failed");
    const float mix = ctx.p().f("mix");
    const ImageView src = ctx.source();
    const bool covered = src.valid() && src.bounds.contains(ctx.window.x1, ctx.window.y1) && src.bounds.contains(ctx.window.x2 - 1, ctx.window.y2 - 1);
    if (m.cuda != CudaMode::Device || (mix <= 0.f && !covered) ||
        (mix > 0.f && ctx.preview && !ctx.frame.empty() && previewDivisor(m, ctx) > 1)) {
        renderBridged(m, ctx);
        return;
    }
    if (mix <= 0.f) {
        cudaCopy(src, ctx.dst, ctx.window, ctx.cudaStream);
        return;
    }
    try {
        m.render(ctx);
    } catch (const NeedsHost&) {
        renderBridged(m, ctx);
        return;
    }
    if (ctx.isAborted()) return;
    if (!gpu::finishDevice(mix < 1.f ? ctx.source() : ImageView{}, ctx.dst, ctx.window, mix, ctx.cudaStream))
        throw std::runtime_error("CUDA finish failed");
}
}

int cudaBridgeCount() { return gBridged.load(); }

void runModule(const ModuleDef& m, const RenderContext& ctx) {
    if (ctx.cudaRender) {
        if (m.cuda == CudaMode::Native) m.render(ctx);
        else runDevice(m, ctx);
        return;
    }
    float mix = ctx.p().f("mix");
    if (mix <= 0.f) { copyRect(ctx.source(),ctx.dst,ctx.window); return; }
    const int divisor = ctx.preview && !ctx.frame.empty() ? previewDivisor(m,ctx) : 1;
    if (divisor > 1 && ctx.frame.contains(ctx.window.x1,ctx.window.y1) &&
        ctx.frame.contains(ctx.window.x2-1,ctx.window.y2-1)) {
        renderPreview(m,ctx,divisor,std::min(1.f,mix));
        if (!ctx.isAborted()) sanitize(ctx.dst,ctx.window);
        return;
    }
    m.render(ctx);
    if (ctx.isAborted()) return;
    if (mix < 1.f) applyMix(ctx.source(), ctx.dst, ctx.window, std::max(0.f, mix));
    sanitize(ctx.dst, ctx.window);
}

}
