#pragma once
#include "core.h"
#include <map>
#include <stdexcept>
#include <string>

namespace tp {

enum class CudaMode { None, Bridged, Device, Native };

struct NeedsHost {};

enum class PType { Double, Int, Choice, Bool, Color, Point };
enum class Unit { None, Frames, Pixels, Degrees, Percent, BPM, FPS, Normalized, Seconds };

struct ParamSpec {
    std::string id, label, hint, group;
    PType type = PType::Double;
    Unit unit = Unit::None;
    double def = 0, min = -1e7, max = 1e7, dmin = 0, dmax = 1;
    std::vector<std::string> options;
    std::array<double, 3> defv{0, 0, 0};
    bool animates = true;
};

inline ParamSpec Dbl(std::string id, std::string label, double def, double min, double max, double dmin, double dmax, Unit u, std::string hint, std::string group = "") {
    ParamSpec s; s.id = std::move(id); s.label = std::move(label); s.def = def; s.min = min; s.max = max; s.dmin = dmin; s.dmax = dmax;
    s.unit = u; s.hint = std::move(hint); s.group = std::move(group); s.type = PType::Double; return s;
}
inline ParamSpec Int(std::string id, std::string label, int def, int min, int max, int dmin, int dmax, Unit u, std::string hint, std::string group = "") {
    ParamSpec s = Dbl(std::move(id), std::move(label), def, min, max, dmin, dmax, u, std::move(hint), std::move(group));
    s.type = PType::Int; return s;
}
inline ParamSpec Choice(std::string id, std::string label, std::vector<std::string> options, int def, std::string hint, std::string group = "") {
    ParamSpec s; s.id = std::move(id); s.label = std::move(label); s.options = std::move(options); s.def = def;
    s.min = 0; s.max = static_cast<double>(s.options.size()) - 1; s.hint = std::move(hint); s.group = std::move(group);
    s.type = PType::Choice; s.animates = false; return s;
}
inline ParamSpec Bool(std::string id, std::string label, bool def, std::string hint, std::string group = "") {
    ParamSpec s; s.id = std::move(id); s.label = std::move(label); s.def = def ? 1 : 0; s.min = 0; s.max = 1;
    s.hint = std::move(hint); s.group = std::move(group); s.type = PType::Bool; s.animates = false; return s;
}
inline ParamSpec Color(std::string id, std::string label, double r, double g, double b, std::string hint, std::string group = "") {
    ParamSpec s; s.id = std::move(id); s.label = std::move(label); s.defv = {r, g, b}; s.hint = std::move(hint);
    s.group = std::move(group); s.type = PType::Color; return s;
}
inline ParamSpec Point(std::string id, std::string label, double x, double y, std::string hint, std::string group = "") {
    ParamSpec s; s.id = std::move(id); s.label = std::move(label); s.defv = {x, y, 0}; s.min = -2; s.max = 3; s.dmin = 0; s.dmax = 1;
    s.unit = Unit::Normalized; s.hint = std::move(hint); s.group = std::move(group); s.type = PType::Point; return s;
}

struct ParamValues {
    std::map<std::string, std::array<double, 3>> v;
    const std::array<double, 3>& raw(const std::string& id) const {
        auto it = v.find(id);
        if (it == v.end()) throw std::runtime_error("TEMPORAL: unknown parameter '" + id + "'");
        return it->second;
    }
    double d(const std::string& id) const { return raw(id)[0]; }
    float f(const std::string& id) const { return static_cast<float>(raw(id)[0]); }
    int i(const std::string& id) const { return static_cast<int>(std::lround(raw(id)[0])); }
    bool b(const std::string& id) const { return raw(id)[0] != 0.0; }
    std::array<double, 3> vec(const std::string& id) const { return raw(id); }
    void set(const std::string& id, double x, double y = 0, double z = 0) { v[id] = {x, y, z}; }
};

enum class EdgeMode { Clamp = 0, Mirror = 1, Transparent = 2 };

struct FrameWindow {
    int before = 0;
    int after = 0;
};

constexpr int kMaxTemporalWindow = 64;

struct RenderContext {
    const ParamValues* params = nullptr;
    double time = 0;
    double fps = 24;
    double first = 0, last = 0;
    double scale = 1;
    Rect frame;
    Rect window;
    ImageView dst;
    ImageView secondSource;
    bool preview = false;
    bool cudaRender = false;
    void* cudaStream = nullptr;
    EdgeMode edge = EdgeMode::Clamp;
    Encoding encoding = Encoding::Linear;
    std::function<ImageView(double)> fetchRaw;
    std::function<bool()> aborted;
    std::function<void(double)> releaseRaw;

    const ParamValues& p() const { return *params; }

    double resolveTime(double t, bool& transparent) const {
        transparent = false;
        double f = std::round(t);
        double a = std::round(first), b = std::round(last);
        if (b < a) std::swap(a, b);
        if (f >= a && f <= b) return f;
        if (edge == EdgeMode::Transparent) { transparent = true; return f; }
        if (edge == EdgeMode::Clamp || b == a) return std::clamp(f, a, b);
        double n = b - a;
        double k = std::fmod(f - a, 2 * n);
        if (k < 0) k += 2 * n;
        return a + (k <= n ? k : 2 * n - k);
    }
    ImageView frameAt(double t) const {
        bool transparent = false;
        double rt = resolveTime(t, transparent);
        if (transparent) return {};
        return fetchRaw(rt);
    }
    ImageView frameOffset(int offset) const { return frameAt(std::round(time) + offset); }
    ImageView source() const { return fetchRaw(std::round(time)); }
    float px(float fullResPixels) const { return static_cast<float>(fullResPixels * scale); }
    void pointToPixels(const std::array<double, 3>& n, double& x, double& y) const {
        x = frame.x1 + n[0] * frame.w();
        y = frame.y1 + n[1] * frame.h();
    }
    bool isAborted() const { return aborted && aborted(); }
};

struct ModuleDef {
    std::string id;
    std::string name;
    std::string family;
    std::string description;
    std::vector<ParamSpec> params;
    bool temporal = false;
    bool usesEncoding = false;
    CudaMode cuda = CudaMode::None;
    std::function<FrameWindow(const ParamValues&, double fps)> framesNeeded;
    std::function<void(const RenderContext&)> render;
    std::function<ParamValues(const ModuleDef&, int)> loadLook;
};

inline FrameWindow clampWindow(FrameWindow w) {
    w.before = std::clamp(w.before, 0, kMaxTemporalWindow);
    w.after = std::clamp(w.after, 0, kMaxTemporalWindow);
    return w;
}

inline FrameWindow effectiveWindow(const ModuleDef& m, const ParamValues& p, double fps) {
    FrameWindow w;
    if (m.framesNeeded) w = clampWindow(m.framesNeeded(p, fps));
    if (p.v.count("edge") && p.i("edge") == static_cast<int>(EdgeMode::Mirror)) {
        int r = std::max(w.before, w.after);
        w.before = w.after = r;
    }
    return w;
}

const std::vector<const ModuleDef*>& allModules();

void addFrameworkParams(ModuleDef& m);

inline ParamValues defaultValues(const ModuleDef& m) {
    ParamValues pv;
    for (const auto& s : m.params) {
        if (s.type == PType::Color || s.type == PType::Point) pv.v[s.id] = s.defv;
        else pv.v[s.id] = {s.def, 0, 0};
    }
    return pv;
}

inline void applyMix(const ImageView& src, const ImageView& dst, const Rect& r, float mix) {
    if (mix >= 1.f) return;
    parallelRows(r.y1, r.y2, [&](int y) {
        for (int x = r.x1; x < r.x2; ++x) {
            RGBA s = src.valid() ? src.at(x, y, SampleEdge::Black) : RGBA{};
            dst.store(x, y, mix <= 0.f ? s : lerp(s, dst.load(x, y), mix));
        }
    });
}

inline void sanitize(const ImageView& dst, const Rect& r) {
    parallelRows(r.y1, r.y2, [&](int y) {
        for (int x = r.x1; x < r.x2; ++x) {
            float* p = dst.px(x, y);
            for (int c = 0; c < 4; ++c) p[c] = finiteOr(p[c], 0.f);
        }
    });
}

void expandPreview(const RenderContext& ctx, const ImageView& source, float scaleX, float scaleY, const ImageView& original, float mix);

inline void requireHost(const RenderContext& c) {
    if (c.cudaRender) throw NeedsHost{};
}

void runModule(const ModuleDef& m, const RenderContext& ctx);
int cudaBridgeCount();

}
