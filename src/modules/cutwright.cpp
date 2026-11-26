#include "modules.h"
#include "../gpu/gpu.h"
#include "cutwright_geo.h"
#include "../core/pyexe.h"
#include <windows.h>
#include <filesystem>
#include <mutex>
#include <cstring>
#include <immintrin.h>

namespace tp {
namespace {
namespace fs = std::filesystem;

struct Handle {
    HANDLE h = nullptr;
    ~Handle() { if (h) CloseHandle(h); }
    Handle() = default;
    Handle(const Handle&) = delete;
    void take(HANDLE next) { if (h) CloseHandle(h); h = next; }
};
struct Header { int command, group, preset, w, h, index, seed, reset; double fps; };
static_assert(sizeof(Header) == 40);
constexpr size_t offset = 4096;

fs::path scriptPath() {
    HMODULE module;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&scriptPath), &module)) throw std::runtime_error("Cannot locate TEMPORAL");
    wchar_t path[32768];
    if (!GetModuleFileNameW(module,path,32768)) throw std::runtime_error("Cannot locate module path");
    auto dir = fs::path(path).parent_path();
    auto resources = dir.parent_path() / L"Resources";
    if (fs::path(path).extension() == L".exe") resources = dir / L"TEMPORAL.ofx.bundle/Contents/Resources";
    return resources / L"cutwright/worker.py";
}

// One bounded transfer buffer. Only the helper owns Python and its imported libraries.
struct Worker {
    Handle mapping, request, done, process;
    char* view = nullptr;
    size_t capacity = 0;
    ~Worker() { close(false); }
    void close(bool force) {
        if (force && process.h) TerminateProcess(process.h,1);
        else if (view && request.h) { reinterpret_cast<Header*>(view)->command = 0; SetEvent(request.h); }
        if (view) UnmapViewOfFile(view);
        view = nullptr; capacity = 0;
        process.take(nullptr); request.take(nullptr); done.take(nullptr); mapping.take(nullptr);
    }
    void ensure(size_t bytes, bool preview) {
        if (view && bytes <= capacity && WaitForSingleObject(process.h,0) == WAIT_TIMEOUT) return;
        close(false);
        static std::atomic<unsigned> serial{0};
        const std::wstring tag = L"Local\\TemporalFX_" + std::to_wstring(GetCurrentProcessId()) + L"_" + std::to_wstring(++serial);
        capacity = bytes;
        mapping.take(CreateFileMappingW(INVALID_HANDLE_VALUE,nullptr,PAGE_READWRITE,DWORD(bytes >> 32),DWORD(bytes),tag.c_str()));
        request.take(CreateEventW(nullptr,FALSE,FALSE,(tag + L"_request").c_str()));
        done.take(CreateEventW(nullptr,FALSE,FALSE,(tag + L"_done").c_str()));
        if (!mapping.h || !request.h || !done.h) throw std::runtime_error("Cannot allocate the effect exchange buffer");
        view = static_cast<char*>(MapViewOfFile(mapping.h,FILE_MAP_ALL_ACCESS,0,0,bytes));
        if (!view) throw std::runtime_error("Cannot map the effect exchange buffer");
        fs::path python = findPython();
        if (python.empty()) throw std::runtime_error("Python was not found. Install Python 3 with NumPy and Pillow, or set TEMPORAL_FX_PYTHON.");
        if (preview) {
            const wchar_t* custom = _wgetenv(L"TEMPORAL_PREVIEW_PYTHON");
            const fs::path candidate = custom ? fs::path(custom) : fs::path(u8"" TP_PREVIEW_PYTHON);
            if (fs::is_regular_file(candidate)) python = candidate;
        }
        const auto script = scriptPath();
        if (!fs::exists(script)) throw std::runtime_error("Missing effect worker in bundle Resources");
        std::wstring cmd = L"\"" + python.wstring() + L"\" -E -s -B \"" + script.wstring() + L"\" " + tag + L" " + std::to_wstring(bytes) + L" " + std::to_wstring(GetCurrentProcessId());
        if (preview) cmd += L" --preview";
        STARTUPINFOW si{}; si.cb = sizeof(si);
        PROCESS_INFORMATION pi{};
        if (!CreateProcessW(python.c_str(),cmd.data(),nullptr,nullptr,FALSE,CREATE_NO_WINDOW,nullptr,script.parent_path().c_str(),&si,&pi))
            throw std::runtime_error("Cannot start the effect worker. Set TEMPORAL_FX_PYTHON to Python with NumPy and Pillow.");
        CloseHandle(pi.hThread); process.take(pi.hProcess);
    }
    void execute(const Header& h, const RenderContext& c) {
        std::memcpy(view,&h,sizeof(h));
        SetEvent(request.h);
        HANDLE waits[] = {done.h, process.h};
        const ULONGLONG deadline = GetTickCount64() + 90000;
        for (;;) {
            DWORD result = WaitForMultipleObjects(2,waits,FALSE,100);
            if (result == WAIT_OBJECT_0) break;
            if (result != WAIT_TIMEOUT || c.isAborted() || GetTickCount64() >= deadline) {
                close(true);
                throw std::runtime_error("Effect worker exited, was cancelled or timed out");
            }
        }
        if (reinterpret_cast<Header*>(view)->command < 0) throw std::runtime_error(view + 64);
    }
};

const std::vector<std::string> presets[] = {
    {"Classic CRT","Analog CRT","Circuit Bender","Shutter Scan","Chroma Scan","Enders","Handicam","Bad Cable"},
    {"Geo Field","Constellation","Contours","Pixel Memory","Horizontal Echo","Grid System","Curtain","Portrait Window"},
    {"Mono Ink","Motion Ink","Cell VHS","Star","Particles","Pixel Art","Thermal"},
    {"Motion Trails","Liquid","Blocks","Chroma"}
};
int history(int group, int preset, int limit) {
    if (group == 1 && (preset == 0 || preset == 1 || preset == 3)) return std::min(limit,12);
    if (group == 2 && preset == 0) return 1;
    if ((group == 2 && preset == 1) || group == 3) return limit;
    return 0;
}
template <class SX, class SY>
void resample(const ImageView& src, const ImageView& dst, const Rect& out, bool black, SX&& sx, SY&& sy) {
    if (!src.valid()) {
        parallelRows(out.y1,out.y2,[&](int y) { std::memset(dst.px(out.x1,y),0,size_t(out.w())*16); });
        return;
    }
    const Rect& b = src.bounds;
    const auto index = [&](int i, int lo, int hi) { return black ? (i < lo || i >= hi ? -1 : i-lo) : std::clamp(i,lo,hi-1)-lo; };
    std::vector<int> cols(size_t(out.w())*2);
    std::vector<float> tx(size_t(out.w()));
    for (int x = out.x1; x < out.x2; ++x) {
        const float f = sx(x)-0.5f, fl = std::floor(f);
        const size_t k = size_t(x-out.x1);
        cols[2*k] = index(int(fl),b.x1,b.x2);
        cols[2*k+1] = index(int(fl)+1,b.x1,b.x2);
        tx[k] = f-fl;
    }
    parallelRows(out.y1,out.y2,[&](int y) {
        const float f = sy(y)-0.5f, fl = std::floor(f);
        const int r0 = index(int(fl),b.y1,b.y2), r1 = index(int(fl)+1,b.y1,b.y2);
        const float* p0 = r0 < 0 ? nullptr : src.px(b.x1,b.y1+r0);
        const float* p1 = r1 < 0 ? nullptr : src.px(b.x1,b.y1+r1);
        const __m128 ty = _mm_set1_ps(f-fl), zero = _mm_setzero_ps();
        const auto at = [&](const float* p, int i) { return p && i >= 0 ? _mm_loadu_ps(p+4*ptrdiff_t(i)) : zero; };
        float* d = dst.px(out.x1,y);
        const bool stream = (reinterpret_cast<uintptr_t>(d) & 15) == 0;
        for (size_t k = 0; k < tx.size(); ++k) {
            const int i0 = cols[2*k], i1 = cols[2*k+1];
            const __m128 t = _mm_set1_ps(tx[k]);
            const __m128 a = at(p0,i0), c = at(p1,i0);
            const __m128 top = _mm_add_ps(a,_mm_mul_ps(_mm_sub_ps(at(p0,i1),a),t));
            const __m128 bottom = _mm_add_ps(c,_mm_mul_ps(_mm_sub_ps(at(p1,i1),c),t));
            const __m128 v = _mm_add_ps(top,_mm_mul_ps(_mm_sub_ps(bottom,top),ty));
            if (stream) _mm_stream_ps(d+4*k,v);
            else _mm_storeu_ps(d+4*k,v);
        }
        if (stream) _mm_sfence();
    });
}
void renderPreset(const RenderContext& c, int group) {
    const auto& p = c.p();
    const int preset = p.i("preset");
    const int renderCap = p.i("resolution") == 0 ? 960 : p.i("resolution") == 1 ? 1920 : std::max(c.frame.w(),c.frame.h());
    const int cap = c.preview ? std::min(640,renderCap) : renderCap;
    const double factor = std::min(1.,double(cap)/std::max(c.frame.w(),c.frame.h()));
    int w = std::max(1,int(std::lround(c.frame.w()*factor))), h = std::max(1,int(std::lround(c.frame.h()*factor)));
    int t = int(std::lround(c.time));
    const int warm = history(group,preset,group ? p.i("history") : 0);
    const int start = std::max(int(std::round(c.first)), t-warm);
    const auto fill = [&](const ImageView& exchange, const ImageView& input) {
        if (w == c.frame.w() && h == c.frame.h() && input.bounds.contains(c.frame.x1,c.frame.y1) &&
            input.bounds.contains(c.frame.x2-1,c.frame.y2-1)) {
            parallelRows(0,h,[&](int y) { std::memcpy(exchange.px(0,y),input.px(c.frame.x1,c.frame.y1+y),size_t(w)*16); });
            return;
        }
        resample(input,exchange,exchange.bounds,true,[&](int x) { return c.frame.x1+(x+0.5f)*c.frame.w()/w; },
                 [&](int y) { return c.frame.y1+(y+0.5f)*c.frame.h()/h; });
    };
    const auto direct = [&](const ImageView& input) {
        if (w != c.frame.w() || h != c.frame.h() || !input.bounds.contains(c.frame.x1,c.frame.y1) ||
            !input.bounds.contains(c.frame.x2-1,c.frame.y2-1) || input.rowBytes != ptrdiff_t(w)*16) return ImageView{};
        return ImageView{input.px(c.frame.x1,c.frame.y1), {0,0,w,h}, ptrdiff_t(w)*16};
    };
    const auto finish = [&](const ImageView& exchange) {
        if (c.preview) {
            expandPreview(c,exchange,float(w)/c.frame.w(),float(h)/c.frame.h(),{},1.f);
            return;
        }
        if (w == c.frame.w() && h == c.frame.h() && c.frame.contains(c.window.x1,c.window.y1) && c.frame.contains(c.window.x2-1,c.window.y2-1)) {
            parallelRows(c.window.y1,c.window.y2,[&](int y) {
                std::memcpy(c.dst.px(c.window.x1,y),exchange.px(c.window.x1-c.frame.x1,y-c.frame.y1),size_t(c.window.w())*16);
            });
            return;
        }
        resample(exchange,c.dst,c.window,false,[&](int x) { return (x-c.frame.x1+0.5f)*w/c.frame.w(); },
                 [&](int y) { return (y-c.frame.y1+0.5f)*h/c.frame.h(); });
    };
    if (group == 0 && preset >= 0 && preset < 8 && gpu::enabled()) {
        cw::Pooled<std::vector<float>> pooled;
        auto& local = pooled.item;
        cw::fit(local,size_t(w)*h*4);
        const ImageView exchange{local.data(), {0,0,w,h}, ptrdiff_t(w)*16};
        fill(exchange,c.frameAt(t));
        if (c.isAborted()) return;
        bool done = false;
        if (preset == 0) done = gpu::cutwrightCrt({0.4f,0.4f,240.f,0.5f,0.35f,1},exchange);
        else {
            struct Row { int mode; float mask, curvature, chromatic, bleed, jitter, noise, dropout, bands; };
            static constexpr Row rows[] = {
                {0,0.35f,0.4f,0.25f,0.25f,0.15f,0.08f,0.1f,12}, {1,0.35f,0.4f,0.7f,0.25f,0.6f,0.08f,0.1f,12},
                {2,0.35f,0.4f,0.25f,0.25f,0.15f,0.08f,0.1f,16}, {3,0.35f,0.4f,0.65f,0.7f,0.15f,0.08f,0.1f,12},
                {4,0.8f,0.4f,0.25f,0.25f,0.15f,0.08f,0.1f,32}, {5,0.05f,0.05f,0.25f,0.6f,0.15f,0.08f,0.1f,12},
                {6,0.35f,0.4f,0.25f,0.25f,0.7f,0.4f,0.6f,12}};
            const Row& r = rows[preset - 1];
            gpu::AnalogParams a;
            a.crt = {r.curvature,0.4f,240.f,0.5f,r.mask,1};
            a.mode = r.mode; a.chromatic = r.chromatic; a.bleed = r.bleed; a.jitter = r.jitter;
            a.noise = r.noise; a.dropout = r.dropout; a.bands = r.bands;
            a.index = t; a.seed = p.i("seed"); a.fps = c.fps;
            done = gpu::cutwrightAnalog(a,exchange);
        }
        if (done) { finish(exchange); return; }
    }
    if (group == 2 && warm == 0 && gpu::enabled()) {
        cw::Pooled<std::vector<float>> pooled;
        auto& local = pooled.item;
        cw::fit(local,size_t(w)*h*4);
        const ImageView exchange{local.data(), {0,0,w,h}, ptrdiff_t(w)*16};
        fill(exchange,c.frameAt(t));
        if (c.isAborted()) return;
        if (preset == 2) {
            gpu::VhsParams v;
            v.seed = p.i("seed"); v.index = t; v.fps = c.fps;
            if (gpu::cutwrightVhs(v,exchange)) { finish(exchange); return; }
        } else if (cwart::native(preset,exchange,t,c.fps,p.i("seed"))) { finish(exchange); return; }
    }
    if (group == 1 && (preset == 0 || preset == 1 || preset == 3) && gpu::enabled()) {
        cw::Pooled<std::vector<float>> pooled;
        auto& local = pooled.item;
        cw::fit(local,size_t(w)*h*4);
        const ImageView exchange{local.data(), {0,0,w,h}, ptrdiff_t(w)*16};
        std::vector<float> before;
        cwgeo::GeoRun run;
        run.links = preset == 1;
        bool done = true;
        const int from = std::min(start,t);
        for (int i = from; i <= t && done; ++i) {
            if (c.isAborted()) return;
            const ImageView src = c.frameAt(i);
            bool filled = false;
            const auto fillNow = [&] { if (!filled) { fill(exchange,src); filled = true; } };
            uint64_t key;
            if (src.bounds.contains(c.frame.x1,c.frame.y1) && src.bounds.contains(c.frame.x2-1,c.frame.y2-1)) key = cwgeo::regionHash(src,c.frame);
            else { fillNow(); key = cwgeo::frameHash(exchange); }
            if (i == t) fillNow();
            const auto previous = [&](cw::Plane& out) {
                before.resize(size_t(w)*h*4);
                const ImageView prior{before.data(), {0,0,w,h}, ptrdiff_t(w)*16};
                fill(prior,c.frameAt(i-1));
                out = cwgeo::lumaOf(prior);
                return true;
            };
            done = cwgeo::geoAdvance(run,exchange,key,i,fillNow,previous);
            if (c.releaseRaw && i > from) c.releaseRaw(i-1);
        }
        if (done) {
            if (preset == 0) cwgeo::geoFieldDraw(run,exchange);
            else if (preset == 1) cwgeo::constellationDraw(run,exchange,p.i("seed"));
            else cwgeo::pixelMemoryDraw(run,exchange,p.i("seed"));
            finish(exchange);
            return;
        }
    }
    if (group == 1 && warm == 0 && gpu::enabled()) {
        cw::Pooled<std::vector<float>> pooled;
        auto& local = pooled.item;
        cw::fit(local,size_t(w)*h*4);
        const ImageView exchange{local.data(), {0,0,w,h}, ptrdiff_t(w)*16};
        fill(exchange,c.frameAt(t));
        if (c.isAborted()) return;
        if (cwgeo::native(preset,exchange,p.i("seed"))) { finish(exchange); return; }
    }
    if (group == 2 && preset == 1 && gpu::enabled()) {
        cw::Pooled<std::vector<float>> pooled;
        auto& local = pooled.item;
        cw::fit(local,size_t(w)*h*4);
        const ImageView exchange{local.data(), {0,0,w,h}, ptrdiff_t(w)*16};
        cwart::MotionInkState state;
        for (int i = std::min(start,t); i <= t; ++i) {
            if (c.isAborted()) return;
            fill(exchange,c.frameAt(i));
            if (c.releaseRaw && i != t) c.releaseRaw(i);
            cwart::motionInk(state,exchange,i,c.fps,i == t);
        }
        finish(exchange);
        return;
    }
    if (group == 2 && preset == 0 && gpu::enabled()) {
        gpu::InkParams ink;
        const auto pen = cwart::hexRgb("#1b1a1f"), sheet = cwart::hexRgb("#f3eee3");
        for (int k = 0; k < 3; ++k) { ink.ink[k] = float(pen[k]); ink.paper[k] = float(sheet[k]); }
        cw::Pooled<std::vector<float>> pooled;
        auto& local = pooled.item;
        cw::fit(local,size_t(w)*h*4);
        const ImageView exchange{local.data(), {0,0,w,h}, ptrdiff_t(w)*16};
        static std::mutex inkLock;
        std::lock_guard<std::mutex> inkGuard(inkLock);
        bool done = true;
        for (int i = std::min(start,t); i <= t && done; ++i) {
            if (c.isAborted()) return;
            const ImageView input = c.frameAt(i);
            const ImageView view = i == t ? ImageView{} : direct(input);
            if (!view.valid()) fill(exchange,input);
            ink.index = i;
            done = gpu::cutwrightInk(ink,view.valid() ? view : exchange,i == std::min(start,t),i == t);
            if (c.releaseRaw && i != t) c.releaseRaw(i);
        }
        if (done) { finish(exchange); return; }
    }
    if (group == 3 && preset >= 0 && preset < 4 && gpu::enabled()) {
        struct Row { int mode; float amount, motion, block, refresh, residual, decay, chromatic; };
        static constexpr Row rows[] = {
            {0,1.f,1.f,0.025f,0.f,1.f,0.94f,0.3f}, {1,0.7f,1.8f,0.f,0.025f,0.85f,0.96f,0.3f},
            {2,0.85f,1.8f,0.035f,0.035f,0.85f,0.96f,0.3f}, {3,0.75f,1.4f,0.025f,0.025f,1.f,0.96f,0.35f}};
        const Row& r = rows[preset];
        gpu::MoshParams m;
        m.mode = r.mode; m.amount = r.amount; m.motion = r.motion; m.block = r.block; m.refresh = r.refresh;
        m.residual = r.residual; m.decay = r.decay; m.chromatic = r.chromatic; m.seed = p.i("seed");
        cw::Pooled<std::vector<float>> pooled;
        auto& local = pooled.item;
        cw::fit(local,size_t(w)*h*4);
        const ImageView exchange{local.data(), {0,0,w,h}, ptrdiff_t(w)*16};
        static std::mutex moshLock;
        std::lock_guard<std::mutex> moshGuard(moshLock);
        bool done = true;
        for (int i = std::min(start,t); i <= t && done; ++i) {
            if (c.isAborted()) return;
            const ImageView input = c.frameAt(i);
            const ImageView view = i == t ? ImageView{} : direct(input);
            if (!view.valid()) fill(exchange,input);
            m.index = i;
            done = gpu::cutwrightMosh(m,view.valid() ? view : exchange,i == std::min(start,t),i == t);
            if (c.releaseRaw && i != t) c.releaseRaw(i);
        }
        if (done) { finish(exchange); return; }
    }
    static std::mutex lock;
    static Worker workers[2];
    Worker& worker = workers[c.preview ? 1 : 0];
    std::lock_guard<std::mutex> guard(lock);
    worker.ensure(offset + size_t(w)*h*16, c.preview);
    ImageView exchange{reinterpret_cast<float*>(worker.view + offset), {0,0,w,h}, ptrdiff_t(w)*16};
    for (int i = std::min(start,t); i <= t; ++i) {
        if (c.isAborted()) return;
        fill(exchange,c.frameAt(i));
        if (c.releaseRaw && i != t) c.releaseRaw(i);
        worker.execute({i == t ? 2 : 1,group,preset,w,h,i,p.i("seed"),i == std::min(start,t),c.fps},c);
    }
    finish(exchange);
}
ModuleDef makeFamily(int group, const char* id, const char* label) {
    ModuleDef m;
    m.id=id; m.name=label; m.family="Digital";
    m.cuda=CudaMode::Bridged;
    m.description="Original display-referred frame kernels. Processing resolution and temporal warm-up are explicit.";
    m.temporal=group != 0;
    m.params={Choice("preset","Preset",presets[group],0,"Effect preset."),
              Choice("resolution","Processing Resolution",{"960 px","1920 px","Full"},1,"Longest processed side. Full preserves source detail."),
              Int("seed","Seed",1,0,99999,0,1000,Unit::None,"Random seed.")};
    if (group) m.params.push_back(Int("history","History",6,1,64,1,24,Unit::Frames,"Previous frames reconstructed for a temporal preset."));
    m.framesNeeded=[group](const ParamValues& p,double) { return FrameWindow{history(group,p.i("preset"),group ? p.i("history") : 0),0}; };
    m.render=[group](const RenderContext& c){ renderPreset(c,group); };
    return m;
}
}
ModuleDef makeAnalog(){ return makeFamily(0,"crt_analog","CRT / Analog"); }
ModuleDef makeGeometric(){ return makeFamily(1,"geometric","Geometric"); }
ModuleDef makeCutwrightArt(){ return makeFamily(2,"cutwright_art","Art Styles"); }
ModuleDef makeCutwrightMosh(){ return makeFamily(3,"cutwright_mosh","Mosh Styles"); }
}
