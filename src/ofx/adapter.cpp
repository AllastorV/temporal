#include "../core/controls.h"
#include "../gpu/gpu.h"
#include "ofxsImageEffect.h"
#include "panel.h"
#include "ofxsSupportPrivate.h"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <set>

namespace {

using namespace tp;

bool isMacroControl(const ParamSpec& spec) {
    return spec.group == "Fusion Focus" || spec.group == "Focus Pull";
}

class TemporalEffect : public OFX::ImageEffect {
public:
    TemporalEffect(OfxImageEffectHandle h, const ModuleDef& m) : OFX::ImageEffect(h), m_(m) {
        dst_ = fetchClip(kOfxImageEffectOutputClipName);
        src_ = fetchClip(kOfxImageEffectSimpleSourceClipName);
        if (getContext() == OFX::eContextGeneral) {
            if (m.id == "double_exposure") second_ = fetchClip("SecondSource");
            if (m.id == "lens_character") second_ = fetchClip("DepthMap");
        }
        for (const auto& s : m_.params) {
            OFX::Param* p = nullptr;
            switch (s.type) {
            case PType::Double: p = fetchDoubleParam(s.id); break;
            case PType::Int: p = fetchIntParam(s.id); break;
            case PType::Choice: p = fetchChoiceParam(s.id); break;
            case PType::Bool: p = fetchBooleanParam(s.id); break;
            case PType::Color: p = fetchRGBParam(s.id); break;
            case PType::Point: p = fetchDouble2DParam(s.id); break;
            }
            params_.push_back({&s, p});
        }
        updateControls(0);

    }

    ~TemporalEffect() override {
        if (!panel_) return;
        std::lock_guard lock(panel_->mutex);
        panel_->read = nullptr;
        panel_->visible = nullptr;
        panel_->enabled = nullptr;
        panel_->write = nullptr;
    }

    ParamValues snapshot(double t) const {
        ParamValues pv;
        for (const auto& e : params_) {
            std::array<double, 3> v{0, 0, 0};
            switch (e.spec->type) {
            case PType::Double: static_cast<OFX::DoubleParam*>(e.param)->getValueAtTime(t, v[0]); break;
            case PType::Int: { int i = 0; static_cast<OFX::IntParam*>(e.param)->getValueAtTime(t, i); v[0] = i; break; }
            case PType::Choice: { int i = 0; static_cast<OFX::ChoiceParam*>(e.param)->getValueAtTime(t, i); v[0] = i; break; }
            case PType::Bool: { bool b = false; static_cast<OFX::BooleanParam*>(e.param)->getValueAtTime(t, b); v[0] = b ? 1 : 0; break; }
            case PType::Color: static_cast<OFX::RGBParam*>(e.param)->getValueAtTime(t, v[0], v[1], v[2]); break;
            case PType::Point: static_cast<OFX::Double2DParam*>(e.param)->getValueAtTime(t, v[0], v[1]); break;
            }
            if (e.spec->type == PType::Double || e.spec->type == PType::Int) v[0] = std::clamp(v[0], e.spec->min, e.spec->max);
            pv.v[e.spec->id] = v;
        }
        return pv;
    }

    void updateControls(double time) {
        const ParamValues values = snapshot(time);
        for (const auto& entry : params_) {
            bool enabled = controlEnabled(m_.id, entry.spec->id, values);
            if (second_ && entry.spec->id == "frame_offset") enabled = !second_->isConnected();
            if (m_.id == "lens_character") {
                const bool hasDepth = second_ && second_->isConnected();
                const bool blur = values.b("lens_blur");
                const std::string& id = entry.spec->id;
                if (id == "lens_focus" || id == "depth_near" || id == "depth_far" || id == "maximum_blur") enabled = blur && hasDepth;
                if (id == "fallback_blur") enabled = blur && !hasDepth;
                if (id == "lens_blur_amount") enabled = blur;
            }
            if (isMacroControl(*entry.spec)) entry.param->setIsSecret(!values.b("lens_fusion_controls"));
            entry.param->setEnabled(enabled);
        }
    }
    void changedParam(const OFX::InstanceChangedArgs& args, const std::string& name) override {
        if (name == kPanelButton) {
            if (args.reason == OFX::eChangeUserEdit) openPanel(args.time);
            return;
        }
        rememberPanelTime(args.time);
        if (m_.loadLook && name == "studio_look" && args.reason == OFX::eChangeUserEdit) {
            const auto look = m_.loadLook(m_, snapshot(args.time).i(name));
            if (!look.v.empty()) applyLook(look, args.time);
        }
        updateControls(args.time);
    }
    void changedClip(const OFX::InstanceChangedArgs& args, const std::string&) override { updateControls(args.time); }

    void getClipPreferences(OFX::ClipPreferencesSetter& preferences) override {
        if (m_.loadLook) preferences.setOutputFrameVarying(true);
    }

    void render(const OFX::RenderArguments& args) override {
        try {
            renderImpl(args);
            if (hadError_.exchange(false)) {
                try { clearPersistentMessage(); } catch (...) {}
            }
        } catch (const std::exception& ex) {
            hadError_ = true;
            std::string msg = std::string("TEMPORAL ") + m_.name + " render at t=" + std::to_string(args.time) + ": " + ex.what();
            logError(msg);
            try { setPersistentMessage(OFX::Message::eMessageError, "", msg); } catch (...) {}
            OFX::throwSuiteStatusException(kOfxStatFailed);
        }
    }

    bool isIdentity(const OFX::IsIdentityArguments& args, OFX::Clip*& identityClip, double& identityTime) override {
        double mix = 1.0;
        for (const auto& e : params_)
            if (e.spec->id == "mix") static_cast<OFX::DoubleParam*>(e.param)->getValueAtTime(args.time, mix);
        if (mix <= 0.0) {
            identityClip = src_;
            identityTime = args.time;
            return true;
        }
        return false;
    }

    void getFramesNeeded(const OFX::FramesNeededArguments& args, OFX::FramesNeededSetter& frames) override {
        try {
            FrameWindow w = effectiveWindow(m_, snapshot(args.time), frameRate());
            if (needsSecondSource(args.time)) {
                frames.setFramesNeeded(*second_, OfxRangeD{args.time,args.time});
                w = {};
            }
            OfxRangeD r;
            r.min = std::round(args.time) - w.before;
            r.max = std::round(args.time) + w.after;
            frames.setFramesNeeded(*src_, r);
        } catch (const std::exception& ex) {
            logError(std::string("TEMPORAL ") + m_.name + " getFramesNeeded: " + ex.what());
            throw;
        }
    }

private:
    struct Entry { const ParamSpec* spec; OFX::Param* param; };

    OFX::Param* panelParam(const ParamSpec& spec) const {
        const auto it = std::find_if(params_.begin(), params_.end(), [&](const Entry& entry) { return entry.spec == &spec; });
        return it == params_.end() ? nullptr : it->param;
    }

    void rememberPanelTime(double time) {
        if (std::isfinite(time) && std::fabs(time) < 1e8) panelTime_ = time;
    }

    double panelTime() {
        const double host = OFX::Private::gTimeLineSuite ? timeLineGetTime() : 0.0;
        return host != 0.0 ? host : panelTime_.load();
    }

    void openPanel(double time) {
        rememberPanelTime(time);
        if (!panel_) {
            panel_ = std::make_shared<PanelBinding>();
            panel_->module = &m_;
            panel_->read = [this] { return snapshot(panelTime()); };
            panel_->visible = [this](const ParamSpec& spec) {
                const auto* param = panelParam(spec);
                return spec.id != "lens_fusion_controls" && param && !param->getIsSecret();
            };
            panel_->enabled = [this](const ParamSpec& spec) {
                auto* param = panelParam(spec);
                return param && param->getIsEnable() && !param->getIsSecret();
            };
            panel_->write = [this](const ParamSpec& spec, const std::array<double, 3>& value) { writePanelParam(spec, value); };
            panel_->time = [this] { return panelTime(); };
            panel_->keys = [this](const ParamSpec& spec) {
                std::vector<double> out;
                if (auto* p = dynamic_cast<OFX::ValueParam*>(panelParam(spec)))
                    for (unsigned i = 0, n = p->getNumKeys(); i < n; ++i) out.push_back(p->getKeyTime(static_cast<int>(i)));
                return out;
            };
            panel_->keyframe = [this](const ParamSpec& spec, bool add) { panelKeyframe(spec, add); };
        }
        togglePanel(panel_);
    }

    void writePanelParam(const ParamSpec& spec, const std::array<double, 3>& value) {
        auto* param = panelParam(spec);
        if (!param || !param->getIsEnable()) return;
        const double time = panelTime();
        beginEditBlock("TEMPORAL Panel");
        try { writePanelValue(spec, param, value, time); }
        catch (const std::exception& ex) { endEditBlock(); logError("panel write failed " + spec.id + ": " + ex.what()); throw; }
        catch (...) { endEditBlock(); logError("panel write failed " + spec.id); throw; }
        endEditBlock();
        requestHostRefresh();
    }

    void panelKeyframe(const ParamSpec& spec, bool add) {
        auto* param = panelParam(spec);
        auto* vp = dynamic_cast<OFX::ValueParam*>(param);
        if (!vp || !param->getIsEnable() || !spec.animates) return;
        const double time = panelTime();
        beginEditBlock("TEMPORAL Keyframe");
        try {
            if (add) {
                const auto v = snapshot(time).raw(spec.id);
                switch (spec.type) {
                case PType::Double: static_cast<OFX::DoubleParam*>(param)->setValueAtTime(time, v[0]); break;
                case PType::Int: static_cast<OFX::IntParam*>(param)->setValueAtTime(time, static_cast<int>(v[0])); break;
                case PType::Color: static_cast<OFX::RGBParam*>(param)->setValueAtTime(time, v[0], v[1], v[2]); break;
                case PType::Point: static_cast<OFX::Double2DParam*>(param)->setValueAtTime(time, v[0], v[1]); break;
                default: break;
                }
            } else {
                for (unsigned i = 0, n = vp->getNumKeys(); i < n; ++i) {
                    const double k = vp->getKeyTime(static_cast<int>(i));
                    if (std::fabs(k - time) < 0.5) { vp->deleteKeyAtTime(k); break; }
                }
            }
        } catch (const std::exception& ex) { endEditBlock(); logError("panel keyframe failed " + spec.id + ": " + ex.what()); return; }
        catch (...) { endEditBlock(); logError("panel keyframe failed " + spec.id); return; }
        endEditBlock();
        requestHostRefresh();
    }

    void writePanelValue(const ParamSpec& spec, OFX::Param* param, const std::array<double, 3>& value, double time) {
        const auto set = [&]<class Param, class... Args>(Param* typed, Args... values) {
            if (spec.animates && typed->getNumKeys()) typed->setValueAtTime(time, values...);
            else typed->setValue(values...);
        };
        switch (spec.type) {
        case PType::Double: set(static_cast<OFX::DoubleParam*>(param), value[0]); break;
        case PType::Int: set(static_cast<OFX::IntParam*>(param), static_cast<int>(value[0])); break;
        case PType::Choice: static_cast<OFX::ChoiceParam*>(param)->setValue(static_cast<int>(value[0])); break;
        case PType::Bool: static_cast<OFX::BooleanParam*>(param)->setValue(value[0] != 0); break;
        case PType::Color: set(static_cast<OFX::RGBParam*>(param), value[0], value[1], value[2]); break;
        case PType::Point: set(static_cast<OFX::Double2DParam*>(param), value[0], value[1]); break;
        }
        if (m_.loadLook && spec.id == "studio_look") {
            const auto look = m_.loadLook(m_, static_cast<int>(value[0]));
            if (!look.v.empty()) applyLook(look, time);
        }
        updateControls(time);
    }

    std::shared_ptr<PanelBinding> panel_;
    std::atomic<double> panelTime_{0};

    bool needsSecondSource(double time) const {
        if (!second_ || !second_->isConnected()) return false;
        if (m_.id != "lens_character") return true;
        const auto values = snapshot(time);
        return values.b("lens_blur") && values.d("lens_blur_amount") > 0;
    }

    void applyLook(const ParamValues& look, double time) {
        beginEditBlock("Load Multi-Optic look");
        try {
            for (const auto& entry : params_) {
                const auto it = look.v.find(entry.spec->id);
                if (it == look.v.end()) continue;
                const auto& value = it->second;
                switch (entry.spec->type) {
                case PType::Choice: static_cast<OFX::ChoiceParam*>(entry.param)->setValue(static_cast<int>(value[0])); break;
                case PType::Int: {
                    auto* param = static_cast<OFX::IntParam*>(entry.param);
                    if (param->getNumKeys()) param->setValueAtTime(time, static_cast<int>(value[0]));
                    else param->setValue(static_cast<int>(value[0]));
                    break;
                }
                case PType::Double: {
                    auto* param = static_cast<OFX::DoubleParam*>(entry.param);
                    if (param->getNumKeys()) param->setValueAtTime(time, value[0]);
                    else param->setValue(value[0]);
                    break;
                }
                case PType::Point: {
                    auto* param = static_cast<OFX::Double2DParam*>(entry.param);
                    if (param->getNumKeys()) param->setValueAtTime(time, value[0], value[1]);
                    else param->setValue(value[0], value[1]);
                    break;
                }
                default: break;
                }
            }
        } catch (...) {
            endEditBlock();
            throw;
        }
        endEditBlock();
    }

    static void logError(const std::string& msg) {
        static std::mutex logMtx;
        std::lock_guard<std::mutex> lock(logMtx);
        const char* tmp = std::getenv("TEMP");
        std::string path = std::string(tmp ? tmp : ".") + "\\TEMPORAL_ofx.log";
        if (FILE* f = std::fopen(path.c_str(), "a")) {
            std::fprintf(f, "%s\n", msg.c_str());
            std::fclose(f);
        }
    }

    double frameRate() const {
        double fps = 0;
        static std::atomic<bool> warned{false};
        try { fps = src_->getFrameRate(); } catch (const std::exception& ex) {
            if (!warned.exchange(true)) logError(std::string("clip getFrameRate unsupported by host, using effect frame rate: ") + ex.what());
        }
        if (!(fps > 0)) {
            try { fps = const_cast<TemporalEffect*>(this)->getFrameRate(); } catch (const std::exception& ex) { logError(std::string("effect getFrameRate: ") + ex.what()); }
        }
        return fps > 0 ? fps : 24.0;
    }

    OfxRangeD frameRange() const {
        OfxRangeD r{0, 0};
        bool ok = false;
        try { r = src_->getFrameRange(); ok = true; } catch (const std::exception& ex) { logError(std::string("clip getFrameRange: ") + ex.what()); }
        if (!ok) {
            try { const_cast<TemporalEffect*>(this)->timeLineGetBounds(r.min, r.max); ok = true; } catch (const std::exception& ex) { logError(std::string("timeLineGetBounds: ") + ex.what()); }
        }
        if (!ok || r.max < r.min) r = OfxRangeD{-1e6, 1e6};
        return r;
    }

    static ImageView viewOf(OFX::Image* img) {
        ImageView v;
        if (!img) return v;
        const OfxRectI& b = img->getBounds();
        v.data = static_cast<float*>(img->getPixelData());
        v.bounds = Rect{b.x1, b.y1, b.x2, b.y2};
        v.rowBytes = img->getRowBytes();
        return v;
    }

    void renderImpl(const OFX::RenderArguments& args) {
        std::unique_ptr<OFX::Image> dst(dst_->fetchImage(args.time));
        if (!dst) OFX::throwSuiteStatusException(kOfxStatFailed);
        if (dst->getPixelDepth() != OFX::eBitDepthFloat || dst->getPixelComponents() != OFX::ePixelComponentRGBA)
            throw std::runtime_error("only 32-bit float RGBA is supported");

        const ParamValues pv = snapshot(args.time);
        std::mutex mtx;
        std::map<double, std::unique_ptr<OFX::Image>> cache;
        auto fetchRaw = [&](double t) -> ImageView {
            std::lock_guard<std::mutex> lock(mtx);
            auto it = cache.find(t);
            if (it == cache.end()) {
                std::unique_ptr<OFX::Image> img(src_->fetchImage(t));
                if (img && (img->getPixelDepth() != OFX::eBitDepthFloat || img->getPixelComponents() != OFX::ePixelComponentRGBA)) img.reset();
                it = cache.emplace(t, std::move(img)).first;
            }
            return viewOf(it->second.get());
        };

        std::unique_ptr<OFX::Image> secondary;
        RenderContext ctx;
        if (second_ && second_->isConnected() &&
            (m_.id != "lens_character" || (pv.b("lens_blur") && pv.d("lens_blur_amount") > 0))) {
            secondary.reset(second_->fetchImage(args.time));
            const bool valid = secondary && secondary->getPixelDepth() == OFX::eBitDepthFloat &&
                               secondary->getPixelComponents() == OFX::ePixelComponentRGBA;
            if (valid) ctx.secondSource = viewOf(secondary.get());
            else if (m_.id == "double_exposure") throw std::runtime_error("Second Source did not supply float RGBA at this time");
        }
        ctx.cudaRender = args.isEnabledCudaRender;
        ctx.cudaStream = args.pCudaStream;
        ctx.preview = args.interactiveRenderStatus || args.renderQualityDraft;
        ctx.params = &pv;
        ctx.time = args.time;
        if (!args.sequentialRenderStatus && !hostRefreshRecent()) rememberPanelTime(args.time);
        ctx.fps = frameRate();
        OfxRangeD range = frameRange();
        ctx.first = range.min;
        ctx.last = range.max;
        ctx.scale = args.renderScale.x > 0 ? args.renderScale.x : 1.0;
        OfxRectD rod{0, 0, 0, 0};
        try { rod = src_->getRegionOfDefinition(args.time); } catch (const std::exception& ex) { logError(std::string("getRegionOfDefinition: ") + ex.what()); }
        const double par = src_->getPixelAspectRatio();
        const double horizontalScale = args.renderScale.x / (std::isfinite(par) && par > 0 ? par : 1.0);
        ctx.frame = Rect{static_cast<int>(std::floor(rod.x1 * horizontalScale)), static_cast<int>(std::floor(rod.y1 * args.renderScale.y)),
                         static_cast<int>(std::ceil(rod.x2 * horizontalScale)), static_cast<int>(std::ceil(rod.y2 * args.renderScale.y))};
        fetchRaw(args.time);
        if (const auto* current = cache[args.time].get()) {
            const OfxRectI& r = current->getRegionOfDefinition();
            if (r.x2 > r.x1 && r.y2 > r.y1) ctx.frame = Rect{r.x1, r.y1, r.x2, r.y2};
        }
        ImageView dv = viewOf(dst.get());
        ctx.dst = dv;
        ctx.window = intersect(Rect{args.renderWindow.x1, args.renderWindow.y1, args.renderWindow.x2, args.renderWindow.y2}, dv.bounds);
        if (ctx.frame.empty()) ctx.frame = dv.bounds;
        if (pv.v.count("edge")) ctx.edge = static_cast<EdgeMode>(pv.i("edge"));
        if (pv.v.count("encoding")) ctx.encoding = static_cast<Encoding>(pv.i("encoding"));
        ctx.fetchRaw = fetchRaw;
        ctx.releaseRaw = [&](double time) {
            std::lock_guard<std::mutex> lock(mtx);
            cache.erase(time);
        };
        ctx.aborted = [this]() { return abort(); };
        if (ctx.window.empty()) return;
        const int bridged = cudaBridgeCount();
        runModule(m_, ctx);
        if (ctx.cudaRender && !loggedCuda_.exchange(true))
            logError("CUDA render: " + m_.id + (cudaBridgeCount() != bridged ? " via host bridge" : " on device"));
    }

    const ModuleDef& m_;
    OFX::Clip* second_ = nullptr;
    OFX::Clip* dst_ = nullptr;
    OFX::Clip* src_ = nullptr;
    std::vector<Entry> params_;
    std::atomic<bool> hadError_{false};
    std::atomic<bool> loggedCuda_{false};
};

void describeModule(const ModuleDef& m, OFX::ImageEffectDescriptor& d) {
#ifdef TP_OPTIC_STUDIO
    const std::string name = m.id == "multi_optic" ? m.name + " Studio" : m.name;
#else
    const std::string& name = m.name;
#endif
    d.setLabels("TP " + name, "TP " + name, "TEMPORAL " + name);
    d.setPluginGrouping("TEMPORAL/" + m.family);
    d.setPluginDescription(m.description);
    // Filter context only exposes the primary input in Fusion.
    if (m.id != "double_exposure" && m.id != "lens_character") d.addSupportedContext(OFX::eContextFilter);
    d.addSupportedContext(OFX::eContextGeneral);
    d.addSupportedBitDepth(OFX::eBitDepthFloat);
    d.setSingleInstance(false);
    d.setHostFrameThreading(false);
    d.setSupportsMultiResolution(true);
    d.setSupportsTiles(false);
    d.setTemporalClipAccess(m.temporal);
    if (m.cuda != CudaMode::None && gpu::available()) {
        d.setSupportsCudaRender(true);
        d.setSupportsCudaStream(true);
    }
    d.setRenderTwiceAlways(false);
    d.setSupportsMultipleClipPARs(false);
    d.setRenderThreadSafety(OFX::eRenderInstanceSafe);
}

void describeModuleInContext(const ModuleDef& m, OFX::ImageEffectDescriptor& d, OFX::ContextEnum context) {
    OFX::ClipDescriptor* src = d.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(OFX::ePixelComponentRGBA);
    src->setTemporalClipAccess(m.temporal);
    src->setSupportsTiles(false);
    src->setIsMask(false);
    if ((m.id == "double_exposure" || m.id == "lens_character") && context == OFX::eContextGeneral) {
        const bool depth = m.id == "lens_character";
        auto* second = d.defineClip(depth ? "DepthMap" : "SecondSource");
        const char* label = depth ? "Depth Map" : "Second Source";
        second->setLabels(label, label, label);
        second->addSupportedComponent(OFX::ePixelComponentRGBA);
        second->setOptional(true);
        second->setIsMask(false);
        second->setSupportsTiles(false);
    }
    OFX::ClipDescriptor* dst = d.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(OFX::ePixelComponentRGBA);
    dst->setSupportsTiles(false);

    OFX::PageParamDescriptor* page = d.definePageParam("Controls");
#ifdef _WIN32
    auto* panel = d.definePushButtonParam(kPanelButton);
    panel->setLabels("Temporal Panel", "Temporal Panel", "Temporal Panel");
    panel->setHint("Open or close the live TEMPORAL controls for this effect.");
    page->addChild(*panel);
#endif
    std::map<std::string, OFX::GroupParamDescriptor*> groups;
    auto groupFor = [&](const std::string& name) -> OFX::GroupParamDescriptor* {
        if (name.empty()) return nullptr;
        auto it = groups.find(name);
        if (it != groups.end()) return it->second;
        OFX::GroupParamDescriptor* g = d.defineGroupParam("grp_" + name);
        g->setLabels(name, name, name);
        g->setOpen(!m.loadLook || name == "Element 1" || name == "Global");
        page->addChild(*g);
        groups[name] = g;
        return g;
    };

    for (const auto& s : m.params) {
        OFX::ValueParamDescriptor* vp = nullptr;
        switch (s.type) {
        case PType::Double: {
            auto* p = d.defineDoubleParam(s.id);
            p->setDefault(s.def);
            p->setRange(s.min, s.max);
            p->setDisplayRange(s.dmin, s.dmax);
            p->setDoubleType(s.unit == Unit::Degrees ? OFX::eDoubleTypeAngle : OFX::eDoubleTypePlain);
            double span = s.dmax - s.dmin;
            p->setIncrement(span > 0 ? span / 100.0 : 0.01);
            p->setDigits(span <= 2.0 ? 3 : 2);
            vp = p;
            break;
        }
        case PType::Int: {
            auto* p = d.defineIntParam(s.id);
            p->setDefault(static_cast<int>(s.def));
            p->setRange(static_cast<int>(s.min), static_cast<int>(s.max));
            p->setDisplayRange(static_cast<int>(s.dmin), static_cast<int>(s.dmax));
            vp = p;
            break;
        }
        case PType::Choice: {
            auto* p = d.defineChoiceParam(s.id);
            for (const auto& o : s.options) p->appendOption(o);
            p->setDefault(static_cast<int>(s.def));
            vp = p;
            break;
        }
        case PType::Bool: {
            auto* p = d.defineBooleanParam(s.id);
            p->setDefault(s.def != 0);
            vp = p;
            break;
        }
        case PType::Color: {
            auto* p = d.defineRGBParam(s.id);
            p->setDefault(s.defv[0], s.defv[1], s.defv[2]);
            vp = p;
            break;
        }
        case PType::Point: {
            auto* p = d.defineDouble2DParam(s.id);
            p->setDefault(s.defv[0], s.defv[1]);
            p->setRange(s.min, s.min, s.max, s.max);
            p->setDisplayRange(s.dmin, s.dmin, s.dmax, s.dmax);
            vp = p;
            break;
        }
        }
        vp->setLabels(s.label, s.label, s.label);
        vp->setHint(s.hint);
        vp->setAnimates(s.animates);
        if (s.id == "lens_fusion_controls") vp->setIsSecret(true);
        if (auto* g = groupFor(s.group)) vp->setParent(*g);
        else page->addChild(*vp);
    }
}

template <int N>
class Factory : public OFX::PluginFactoryHelper<Factory<N>> {
public:
#ifdef TP_OPTIC_STUDIO
    Factory() : OFX::PluginFactoryHelper<Factory<N>>(allModules()[N]->id == "multi_optic"
                ? "com.temporal.multi_optic_studio" : "com.temporal.lens_character", 1,
                allModules()[N]->id == "multi_optic" ? 3 : 1) {}
#else
    Factory() : OFX::PluginFactoryHelper<Factory<N>>("com.temporal." + allModules()[N]->id, 1,
                allModules()[N]->id == "multi_optic" ? 5 : (allModules()[N]->id == "lens_character" ? 2 : 1)) {}
#endif
    void describe(OFX::ImageEffectDescriptor& d) override { describeModule(*allModules()[N], d); }
    void describeInContext(OFX::ImageEffectDescriptor& d, OFX::ContextEnum context) override { describeModuleInContext(*allModules()[N], d, context); }
    OFX::ImageEffect* createInstance(OfxImageEffectHandle h, OFX::ContextEnum) override { return new TemporalEffect(h, *allModules()[N]); }
};

template <int N>
void pushFactory(OFX::PluginFactoryArray& ids) {
    if constexpr (N > 0) pushFactory<N - 1>(ids);
    if (static_cast<size_t>(N) < allModules().size()
#ifdef TP_OPTIC_STUDIO
        && (allModules()[N]->id == "multi_optic" || allModules()[N]->id == "lens_character")
#endif
    ) {
        static Factory<N> f;
        ids.push_back(&f);
    }
}

}

namespace OFX {
namespace Plugin {
void getPluginIDs(OFX::PluginFactoryArray& ids) { pushFactory<23>(ids); }
}
}
