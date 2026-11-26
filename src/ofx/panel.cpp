#include "panel.h"
#include "../core/controls.h"
#ifdef _WIN32
#include <windows.h>
#include <windowsx.h>
#include <commctrl.h>
#include <dwmapi.h>
#include <d2d1.h>
#include <dwrite_1.h>
#include <wrl/client.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cwchar>
#include <filesystem>
#include <atomic>
#include <map>
#include <set>

namespace tp {
namespace {

using Microsoft::WRL::ComPtr;

constexpr UINT_PTR refreshTimer = 1, animTimer = 2;
constexpr float topBar = 44, gap = 10, sidePad = 16, topPad = 12, bottomPad = 18, cardPad = 12;

D2D1_COLOR_F rgb(unsigned hex, float a = 1.f) {
    return D2D1::ColorF(((hex >> 16) & 255) / 255.f, ((hex >> 8) & 255) / 255.f, (hex & 255) / 255.f, a);
}

D2D1_COLOR_F oklch(float l, float c, float hue) {
    const float h = hue * 3.14159265f / 180, a = c * std::cos(h), b = c * std::sin(h);
    const float l_ = l + 0.3963377774f * a + 0.2158037573f * b, m_ = l - 0.1055613458f * a - 0.0638541728f * b, s_ = l - 0.0894841775f * a - 1.2914855480f * b;
    const float L = l_ * l_ * l_, M = m_ * m_ * m_, S = s_ * s_ * s_;
    const auto enc = [](float x) { x = std::clamp(x, 0.f, 1.f); return x <= 0.0031308f ? 12.92f * x : 1.055f * std::pow(x, 1 / 2.4f) - 0.055f; };
    return D2D1::ColorF(enc(4.0767416621f * L - 3.3077115913f * M + 0.2309699292f * S), enc(-1.2684380046f * L + 2.6097574011f * M - 0.3413193965f * S),
                        enc(-0.0041960863f * L - 0.7034186147f * M + 1.7076926610f * S));
}

D2D1_COLOR_F hsl(float h, float s, float l) {
    h = std::fmod(std::fmod(h, 360.f) + 360.f, 360.f) / 60.f;
    const float c = (1 - std::fabs(2 * l - 1)) * s, x = c * (1 - std::fabs(std::fmod(h, 2.f) - 1)), m = l - c / 2;
    float r = 0, g = 0, b = 0;
    if (h < 1) { r = c; g = x; } else if (h < 2) { r = x; g = c; } else if (h < 3) { g = c; b = x; }
    else if (h < 4) { g = x; b = c; } else if (h < 5) { r = x; b = c; } else { r = c; b = x; }
    return D2D1::ColorF(r + m, g + m, b + m, 1.f);
}

std::wstring wide(const std::string& value) {
    const int size = MultiByteToWideChar(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), nullptr, 0);
    std::wstring out(size, L'\0');
    if (size) MultiByteToWideChar(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), out.data(), size);
    return out;
}

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool has(const std::string& s, std::initializer_list<const char*> words) {
    for (const char* w : words) if (s.find(w) != std::string::npos) return true;
    return false;
}

enum class Family { Space, Time, Light, Colour, Look, Global };

unsigned toneHex(Family f) {
    switch (f) {
    case Family::Space: return 0x6694d6;
    case Family::Time: return 0xd1a54a;
    case Family::Light: return 0xd67a5e;
    case Family::Global: return 0x58a8b4;
    case Family::Colour: return 0xc470c4;
    default: return 0x48b882;
    }
}

const wchar_t* familyName(Family f) {
    switch (f) {
    case Family::Space: return L"Space";
    case Family::Time: return L"Time";
    case Family::Light: return L"Light";
    case Family::Colour: return L"Colour";
    case Family::Global: return L"Global";
    default: return L"Look";
    }
}

Family familyOf(const ParamSpec& p) {
    const std::string s = lower(p.id + " " + p.label);
    if (p.type == PType::Color || has(s, {"tint", "color", "colour", "hue", "satur", "chroma"})) return Family::Colour;
    if (p.unit == Unit::Frames || p.unit == Unit::Seconds || p.unit == Unit::FPS || p.unit == Unit::BPM ||
        has(s, {"time", "tempo", "speed", "rate", "frame", "shutter", "offset", "duration", "history", "delay", "decay", "echo", "trail", "motion"}))
        return Family::Time;
    if (has(s, {"light", "highlight", "glow", "bright", "exposure", "threshold", "intensity", "gain", "halo", "caustic", "rim", "streak", "flare", "bloom", "strength", "amount", "rain"}))
        return Family::Light;
    if (p.unit == Unit::Degrees || p.unit == Unit::Pixels || p.type == PType::Point ||
        has(s, {"pos", "size", "scale", "rot", "angle", "dist", "curv", "radius", "width", "height", "length", "wind", "focus", "blur", "tilt", "depth", "aspect", "ior", "lens", "zoom"}))
        return Family::Space;
    return Family::Look;
}

std::wstring labelUnit(const std::string& label) {
    const auto open = label.rfind(" (");
    if (open == std::string::npos || label.back() != ')') return {};
    const std::string u = label.substr(open + 2, label.size() - open - 3);
    return u == "m" || u == "mm" || u == "px" ? L" " + std::wstring(u.begin(), u.end()) : std::wstring{};
}

std::wstring formatValue(const ParamSpec& s, double v) {
    wchar_t buf[64];
    if (s.type == PType::Int) swprintf_s(buf, L"%d", static_cast<int>(std::lround(v)));
    else {
        const double span = std::max(std::fabs(s.dmax - s.dmin), 1e-9);
        const int digits = span < 0.2 || (v != 0 && std::fabs(v) < 0.1) ? 3 : span < 10 ? 2 : span < 100 ? 1 : 0;
        swprintf_s(buf, L"%.*f", digits, v);
    }
    std::wstring out = buf;
    switch (s.unit) {
    case Unit::Degrees: out += L"°"; break;
    case Unit::Pixels: out += L" px"; break;
    case Unit::Frames: out += L" f"; break;
    case Unit::Seconds: out += L" s"; break;
    case Unit::Percent: out += L"%"; break;
    case Unit::FPS: out += L" fps"; break;
    case Unit::None: out += labelUnit(s.label); break;
    default: break;
    }
    return out;
}

enum class Kind { Bar, Seg, Drop, Field, Toggle };
enum class Widget { None, Dial, Pad, Wheel, Range };

struct Ctl {
    Kind kind = Kind::Bar;
    const ParamSpec* spec = nullptr;
    D2D1_RECT_F rect{};
    bool asDrop = false;
    float keyW = 0;
};

struct Card {
    std::string group;
    std::wstring title, count;
    Family tone = Family::Look;
    Widget widget = Widget::None;
    const ParamSpec* wa = nullptr;
    const ParamSpec* wb = nullptr;
    std::vector<Ctl> ctls;
    D2D1_RECT_F rect{}, wrect{};
    float flash = -1;
};

struct Drag {
    enum Mode { None, Bar, Dial, Pad, Wheel, RangeLo, RangeHi, Scroll } mode = None;
    const ParamSpec* spec = nullptr;
    const Card* card = nullptr;
    float startX = 0, startY = 0, width = 1;
    double startValue = 0;
    float startScroll = 0;
};

struct DropList {
    bool open = false;
    const ParamSpec* spec = nullptr;
    D2D1_RECT_F rect{};
    float itemH = 24;
    int first = 0, hover = -1, visible = 0;
};

float clampf(float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); }
bool inside(const D2D1_RECT_F& r, float x, float y) { return x >= r.left && x < r.right && y >= r.top && y < r.bottom; }

HWND resolveWindow() {
    struct Find { DWORD pid; HWND best; LONG area; } f{GetCurrentProcessId(), nullptr, 0};
    EnumWindows([](HWND h, LPARAM lp) -> BOOL {
        auto& f = *reinterpret_cast<Find*>(lp);
        DWORD pid = 0;
        GetWindowThreadProcessId(h, &pid);
        if (pid != f.pid || !IsWindowVisible(h) || GetWindow(h, GW_OWNER)) return TRUE;
        RECT r{};
        GetWindowRect(h, &r);
        const LONG area = (r.right - r.left) * (r.bottom - r.top);
        if (area > f.area) { f.area = area; f.best = h; }
        return TRUE;
    }, reinterpret_cast<LPARAM>(&f));
    return f.best;
}

class LivePanel {
public:
    void toggle(const std::shared_ptr<PanelBinding>& next) {
        if (window && IsWindowVisible(window) && binding.lock() == next) {
            savePlacement();
            ShowWindow(window, SW_HIDE);
            return;
        }
        if (!window && !create()) return;
        if (GetWindowThreadProcessId(window, nullptr) != GetCurrentThreadId()) return;
        binding = next;
        closeEditor(false);
        drop.open = false;
        scrollX = scrollTarget = 0;
        current.clear();
        signature.clear();
        moduleName = wide(next->module->name);
        moduleColor = nameColor(next->module->id);
        SetWindowTextW(window, (L"TEMPORAL  ·  " + moduleName).c_str());
        refresh(true);
        ShowWindow(window, SW_SHOWNOACTIVATE);
        SetForegroundWindow(window);
        InvalidateRect(window, nullptr, FALSE);
    }

private:
    HWND window = nullptr, editor = nullptr;
    HFONT editFont = nullptr;
    HBRUSH editBrush = nullptr;
    const ParamSpec* editSpec = nullptr;
    int editComponent = 0;
    UINT dpi = 96;
    std::weak_ptr<PanelBinding> binding;
    std::wstring moduleName;
    D2D1_COLOR_F moduleColor = rgb(0xe8e4dc);
    std::vector<Card> cards;
    std::vector<std::pair<std::string, D2D1_RECT_F>> chips;
    std::string current, signature;
    ParamValues values;
    std::map<std::string, bool> enabledMap;
    float scrollX = 0, scrollTarget = 0, scrollFrom = 0, contentW = 0, pitch = 1;
    double scrollStart = -1;
    bool navigating = false, failed = false;
    Drag drag;
    DropList drop;
    float hoverX = -1, hoverY = -1;
    const ParamSpec* tipSpec = nullptr;
    double tipSince = 0;
    bool tipShown = false;
    std::map<std::string, std::wstring> reasons;
    std::map<std::string, std::vector<double>> keyMap;
    double nowTime = 0;
    bool ungrouped = true;
    std::map<std::wstring, std::wstring> fitted;
    ComPtr<ID2D1Factory> d2d;
    ComPtr<ID2D1HwndRenderTarget> rt;
    ComPtr<ID2D1SolidColorBrush> brush;
    ComPtr<ID2D1StrokeStyle> dashed;
    ComPtr<IDWriteFactory> dw;
    ComPtr<IDWriteTextFormat> fLabel, fHead, fChip, fWord, fSmall, fValue;
    ComPtr<IDWriteTypography> tabular;
    ComPtr<IDWriteInlineObject> ellipsis;

    static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
    float dip(int px) const { return px * 96.f / dpi; }
    int px(float d) const { return static_cast<int>(std::lround(d * dpi / 96.f)); }

    static D2D1_COLOR_F nameColor(const std::string& id) {
        const auto& all = allModules();
        size_t index = 0;
        for (size_t i = 0; i < all.size(); ++i) if (all[i]->id == id) index = i;
        constexpr float centres[] = {290, 355};
        const float hue = centres[index % 2] + 5.f * (static_cast<float>(index / 2 % 3) - 1);
        return oklch(index / 6 % 2 ? 0.8f : 0.84f, 0.08f, hue);
    }

    bool create() {
        HMODULE module = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                               reinterpret_cast<LPCWSTR>(&proc), &module)) return false;
        const auto name = L"TemporalPanel2_" + std::to_wstring(reinterpret_cast<UINT_PTR>(module));
        WNDCLASSW wc{};
        wc.style = CS_DBLCLKS;
        wc.lpfnWndProc = proc;
        wc.hInstance = module;
        wc.lpszClassName = name.c_str();
        wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
        if (!RegisterClassW(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) return false;
        if (FAILED(D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, d2d.GetAddressOf()))) return false;
        if (FAILED(DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), reinterpret_cast<IUnknown**>(dw.GetAddressOf())))) return false;
        HWND owner = resolveWindow();
        const UINT sysDpi = owner ? GetDpiForWindow(owner) : GetDpiForSystem();
        RECT place{};
        if (!loadPlacement(place)) {
            RECT o{0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)};
            if (owner) GetWindowRect(owner, &o);
            const int w = std::min<int>(MulDiv(1280, sysDpi, 96), o.right - o.left - 40);
            const int h = MulDiv(420 + 32, sysDpi, 96);
            const int x = o.left + (o.right - o.left - w) / 2;
            const int y = o.top + (o.bottom - o.top) * 52 / 100 - h / 2;
            place = {x, y, x + w, y + h};
        }
        window = CreateWindowExW(0, name.c_str(), L"TEMPORAL", WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_THICKFRAME | WS_MINIMIZEBOX | WS_CLIPCHILDREN,
                                 place.left, place.top, place.right - place.left, place.bottom - place.top, owner, nullptr, module, this);
        if (!window) return false;
        wchar_t path[MAX_PATH]{};
        GetModuleFileNameW(module, path, MAX_PATH);
        const auto ico = std::filesystem::path(path).parent_path().parent_path() / L"Resources" / L"temporal.ico";
        for (const auto [kind, metric] : {std::pair{ICON_BIG, SM_CXICON}, std::pair{ICON_SMALL, SM_CXSMICON}}) {
            const int s = GetSystemMetricsForDpi(metric, sysDpi);
            if (auto* icon = static_cast<HICON>(LoadImageW(nullptr, ico.c_str(), IMAGE_ICON, s, s, LR_LOADFROMFILE)))
                SendMessageW(window, WM_SETICON, kind, reinterpret_cast<LPARAM>(icon));
        }
        return true;
    }

    static bool loadPlacement(RECT& r) {
        DWORD size = sizeof(r);
        return RegGetValueW(HKEY_CURRENT_USER, L"Software\\TEMPORAL\\Panel", L"Rect", RRF_RT_REG_BINARY, nullptr, &r, &size) == ERROR_SUCCESS &&
               size == sizeof(r) && r.right - r.left > 200 && r.bottom - r.top > 150;
    }

    void savePlacement() const {
        if (!window || IsIconic(window)) return;
        RECT r{};
        GetWindowRect(window, &r);
        HKEY key = nullptr;
        if (RegCreateKeyExW(HKEY_CURRENT_USER, L"Software\\TEMPORAL\\Panel", 0, nullptr, 0, KEY_SET_VALUE, nullptr, &key, nullptr) != ERROR_SUCCESS) return;
        RegSetValueExW(key, L"Rect", 0, REG_BINARY, reinterpret_cast<const BYTE*>(&r), sizeof(r));
        RegCloseKey(key);
    }

    ComPtr<IDWriteTextFormat> format(float size, DWRITE_FONT_WEIGHT weight) const {
        ComPtr<IDWriteTextFormat> f;
        dw->CreateTextFormat(L"Segoe UI Variable Text", nullptr, weight, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, size, L"en-us", f.GetAddressOf());
        if (f) {
            f->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_CENTER);
            f->SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP);
        }
        return f;
    }

    bool resources() {
        if (rt) return true;
        RECT r{};
        GetClientRect(window, &r);
        const auto props = D2D1::RenderTargetProperties(D2D1_RENDER_TARGET_TYPE_DEFAULT, D2D1::PixelFormat(), static_cast<float>(dpi), static_cast<float>(dpi));
        if (FAILED(d2d->CreateHwndRenderTarget(props, D2D1::HwndRenderTargetProperties(window, D2D1::SizeU(r.right, r.bottom)), rt.GetAddressOf()))) return false;
        rt->CreateSolidColorBrush(rgb(0xffffff), brush.GetAddressOf());
        const float dashes[] = {2.f, 2.f};
        d2d->CreateStrokeStyle(D2D1::StrokeStyleProperties(D2D1_CAP_STYLE_FLAT, D2D1_CAP_STYLE_FLAT, D2D1_CAP_STYLE_FLAT, D2D1_LINE_JOIN_MITER, 10.f, D2D1_DASH_STYLE_CUSTOM),
                               dashes, 2, dashed.GetAddressOf());
        fLabel = format(12, DWRITE_FONT_WEIGHT_NORMAL);
        fValue = format(12, DWRITE_FONT_WEIGHT_NORMAL);
        fHead = format(12, DWRITE_FONT_WEIGHT_SEMI_BOLD);
        fChip = format(12, DWRITE_FONT_WEIGHT_NORMAL);
        fWord = format(13, DWRITE_FONT_WEIGHT_BOLD);
        fSmall = format(10.5f, DWRITE_FONT_WEIGHT_NORMAL);
        if (!fLabel || !fValue || !fHead || !fChip || !fWord || !fSmall) return false;
        dw->CreateEllipsisTrimmingSign(fLabel.Get(), ellipsis.GetAddressOf());
        DWRITE_TRIMMING trim{DWRITE_TRIMMING_GRANULARITY_CHARACTER, 0, 0};
        for (auto* f : {fLabel.Get(), fHead.Get(), fChip.Get(), fValue.Get(), fSmall.Get()}) f->SetTrimming(&trim, ellipsis.Get());
        dw->CreateTypography(tabular.GetAddressOf());
        if (tabular) tabular->AddFontFeature({DWRITE_FONT_FEATURE_TAG_TABULAR_FIGURES, 1});
        return true;
    }

    float fade = 1.f;
    void color(unsigned hex, float a = 1.f) { brush->SetColor(rgb(hex, a * fade)); }
    void color(D2D1_COLOR_F c, float a = 1.f) { c.a *= a * fade; brush->SetColor(c); }

    void fillRound(const D2D1_RECT_F& r, float radius) { rt->FillRoundedRectangle(D2D1::RoundedRect(r, radius, radius), brush.Get()); }
    void strokeRound(const D2D1_RECT_F& r, float radius, float w, ID2D1StrokeStyle* style = nullptr) {
        rt->DrawRoundedRectangle(D2D1::RoundedRect(r, radius, radius), brush.Get(), w, style);
    }

    void text(const std::wstring& s, IDWriteTextFormat* f, const D2D1_RECT_F& r, DWRITE_TEXT_ALIGNMENT align = DWRITE_TEXT_ALIGNMENT_LEADING) {
        if (r.right <= r.left) return;
        f->SetTextAlignment(align);
        rt->DrawText(s.c_str(), static_cast<UINT32>(s.size()), f, r, brush.Get(), D2D1_DRAW_TEXT_OPTIONS_CLIP);
    }

    float measure(const std::wstring& s, IDWriteTextFormat* f, bool tab = false) const {
        ComPtr<IDWriteTextLayout> layout;
        if (FAILED(dw->CreateTextLayout(s.c_str(), static_cast<UINT32>(s.size()), f, 4000, 40, layout.GetAddressOf()))) return 0;
        if (tab && tabular) layout->SetTypography(tabular.Get(), {0, static_cast<UINT32>(s.size())});
        DWRITE_TEXT_METRICS m{};
        layout->GetMetrics(&m);
        return m.widthIncludingTrailingWhitespace;
    }

    void valueText(const std::wstring& s, const D2D1_RECT_F& r) {
        ComPtr<IDWriteTextLayout> layout;
        const float w = r.right - r.left;
        if (w <= 0 || FAILED(dw->CreateTextLayout(s.c_str(), static_cast<UINT32>(s.size()), fValue.Get(), w, r.bottom - r.top, layout.GetAddressOf()))) return;
        if (tabular) layout->SetTypography(tabular.Get(), {0, static_cast<UINT32>(s.size())});
        layout->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_TRAILING);
        layout->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_CENTER);
        rt->DrawTextLayout(D2D1::Point2F(r.left, r.top), layout.Get(), brush.Get(), D2D1_DRAW_TEXT_OPTIONS_CLIP);
    }

    void wordmark(float x, float cy) {
        const std::wstring s = L"TEMPORAL";
        ComPtr<IDWriteTextLayout> layout;
        if (FAILED(dw->CreateTextLayout(s.c_str(), static_cast<UINT32>(s.size()), fWord.Get(), 200, 20, layout.GetAddressOf()))) return;
        ComPtr<IDWriteTextLayout1> spaced;
        if (SUCCEEDED(layout.As(&spaced))) spaced->SetCharacterSpacing(0, 13 * 0.22f, 0, {0, static_cast<UINT32>(s.size())});
        layout->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_LEADING);
        color(0x5fb0bd);
        rt->DrawTextLayout(D2D1::Point2F(x, cy - 10), layout.Get(), brush.Get());
    }

    double value(const ParamSpec& s, int component = 0) const {
        auto it = values.v.find(s.id);
        return it == values.v.end() ? s.def : it->second[component];
    }
    bool enabled(const ParamSpec& s) const {
        auto it = enabledMap.find(s.id);
        return it == enabledMap.end() || it->second;
    }

    void refresh(bool force) {
        if (!window) return;
        auto b = binding.lock();
        if (!b) { if (IsWindowVisible(window)) ShowWindow(window, SW_HIDE); return; }
        std::lock_guard lock(b->mutex);
        if (!b->read || !b->visible || !b->enabled) { ShowWindow(window, SW_HIDE); return; }
        bool changed = force;
        try {
            ParamValues next = b->read();
            std::string sig;
            std::map<std::string, bool> en;
            for (const auto& s : b->module->params) {
                const bool v = b->visible(s);
                sig += v ? '1' : '0';
                if (v) en[s.id] = b->enabled(s);
            }
            if (sig != signature) { signature = sig; build(*b); fitHeight(); changed = true; }
            if (b->keys && b->time) {
                std::map<std::string, std::vector<double>> km;
                for (const auto& s : b->module->params) if (animatable(s) && b->visible(s)) km[s.id] = b->keys(s);
                const double t = b->time();
                if (km != keyMap || t != nowTime) changed = true;
                keyMap = std::move(km);
                nowTime = t;
            }
            if (next.v != values.v || en != enabledMap) changed = true;
            const bool enChanged = en != enabledMap;
            values = std::move(next);
            enabledMap = std::move(en);
            if (enChanged) explain(*b->module);
            failed = false;
        } catch (...) { failed = true; changed = true; }
        if (changed) InvalidateRect(window, nullptr, FALSE);
    }

    static bool animatable(const ParamSpec& s) {
        return s.animates && (s.type == PType::Double || s.type == PType::Int || s.type == PType::Color || s.type == PType::Point);
    }

    bool animated(const ParamSpec& s) const {
        auto it = keyMap.find(s.id);
        return it != keyMap.end() && !it->second.empty();
    }

    bool onKey(const ParamSpec& s) const {
        auto it = keyMap.find(s.id);
        if (it == keyMap.end()) return false;
        for (double k : it->second) if (std::fabs(k - nowTime) < 0.5) return true;
        return false;
    }

    static D2D1_RECT_F keyRect(const D2D1_RECT_F& r) {
        const float cx = r.right - 12, cy = (r.top + r.bottom) / 2;
        return D2D1::RectF(cx - 7, cy - 7, cx + 7, cy + 7);
    }

    D2D1_RECT_F headKeyRect(const Card& c) const {
        const float cx = c.rect.right - cardPad - 5, cy = c.rect.top + cardPad + 7;
        return D2D1::RectF(cx - 7, cy - 7, cx + 7, cy + 7);
    }

    void drawKey(const D2D1_RECT_F& box, const ParamSpec& s, const D2D1_COLOR_F& tone) {
        const float cx = (box.left + box.right) / 2, cy = (box.top + box.bottom) / 2, h = 4.5f;
        const bool hot = inside(box, hoverX, hoverY), key = onKey(s), anim = animated(s);
        ComPtr<ID2D1PathGeometry> g;
        d2d->CreatePathGeometry(g.GetAddressOf());
        ComPtr<ID2D1GeometrySink> sink;
        if (!g || FAILED(g->Open(sink.GetAddressOf()))) return;
        sink->BeginFigure(D2D1::Point2F(cx, cy - h), D2D1_FIGURE_BEGIN_FILLED);
        sink->AddLine(D2D1::Point2F(cx + h, cy));
        sink->AddLine(D2D1::Point2F(cx, cy + h));
        sink->AddLine(D2D1::Point2F(cx - h, cy));
        sink->EndFigure(D2D1_FIGURE_END_CLOSED);
        sink->Close();
        if (key) {
            color(0xf2eee6);
            rt->FillGeometry(g.Get(), brush.Get());
        } else {
            color(anim ? 0xf2eee6 : 0x8a9096, hot ? 1.f : anim ? 1.f : 0.55f);
            rt->DrawGeometry(g.Get(), brush.Get(), anim ? 1.5f : 1.1f);
            if (anim) rt->FillEllipse(D2D1::Ellipse(D2D1::Point2F(cx, cy), 1.4f, 1.4f), brush.Get());
        }
        (void)tone;
    }

    std::wstring needName(const ParamSpec& c, double t, double old) const {
        std::wstring n = wide(c.label);
        if (c.type == PType::Choice && old != 0 && t >= 0 && t < c.options.size()) n += L": " + wide(c.options[static_cast<size_t>(t)]);
        return n;
    }

    bool probe(const ModuleDef& m, const ParamSpec& s, const ParamSpec* skip, std::wstring& name, double* hit = nullptr) {
        for (const auto& c : m.params) {
            if (&c == &s || &c == skip || !values.v.count(c.id) || !(c.type == PType::Choice || c.type == PType::Bool || c.type == PType::Double || c.type == PType::Int)) continue;
            std::vector<double> tries;
            if (c.type == PType::Choice) for (size_t i = 0; i < c.options.size(); ++i) tries.push_back(static_cast<double>(i));
            else if (c.type == PType::Bool) tries.push_back(1 - value(c));
            else tries = {c.dmax, c.dmin, c.def, 1.0};
            double& slot = values.v[c.id][0];
            const double old = slot;
            for (double t : tries) {
                if (t == old) continue;
                slot = t;
                bool ok = false;
                try { ok = controlEnabled(m.id, s.id, values); } catch (...) {}
                if (ok) {
                    slot = old;
                    name = needName(c, t, old);
                    if (hit) *hit = t;
                    return true;
                }
            }
            slot = old;
        }
        return false;
    }

    void explain(const ModuleDef& m) {
        reasons.clear();
        std::map<const ParamSpec*, double> keys;
        std::vector<const ParamSpec*> hard;
        for (const auto& s : m.params) {
            if (enabled(s)) continue;
            try {
                if (controlEnabled(m.id, s.id, values)) continue;
                std::wstring name;
                double t = 0;
                if (probe(m, s, nullptr, name, &t)) {
                    reasons[s.id] = L"needs " + name;
                    for (const auto& c : m.params) if (name.rfind(wide(c.label), 0) == 0) keys[&c] = t;
                } else hard.push_back(&s);
            } catch (...) {}
        }
        for (const auto* s : hard) {
            for (const auto& [c, t] : keys) {
                double& slot = values.v[c->id][0];
                const double old = slot;
                slot = t;
                std::wstring name;
                const bool ok = probe(m, *s, c, name);
                slot = old;
                if (ok) { reasons[s->id] = L"needs " + needName(*c, t, old) + L" + " + name; break; }
            }
        }
    }

    std::wstring fitLabel(const std::string& label, const std::string& group, float width) {
        if (width <= 0) return wide(label);
        const std::wstring key = wide(label + "|" + group) + L"|" + std::to_wstring(static_cast<int>(width));
        if (auto it = fitted.find(key); it != fitted.end()) return it->second;
        std::string base = label;
        if (const auto open = base.rfind(" ("); open != std::string::npos && base.back() == ')') base = base.substr(0, open);
        std::vector<std::string> words;
        for (size_t i = 0; i < base.size();) {
            const size_t j = base.find(' ', i);
            words.push_back(base.substr(i, j == std::string::npos ? std::string::npos : j - i));
            if (j == std::string::npos) break;
            i = j + 1;
        }
        const auto join = [&] { std::string r; for (const auto& w : words) r += (r.empty() ? "" : " ") + w; return wide(r); };
        const auto fits = [&] { return measure(join(), fLabel.Get()) <= width; };
        {
            std::vector<std::string> g;
            for (size_t i = 0; i < group.size();) {
                const size_t j = group.find(' ', i);
                g.push_back(lower(group.substr(i, j == std::string::npos ? std::string::npos : j - i)));
                if (j == std::string::npos) break;
                i = j + 1;
            }
            if (!g.empty() && words.size() > 1 && lower(words.front()) == g.front() && lower(base) != lower(group)) words.erase(words.begin());
        }
        std::wstring out = join();
        if (!fits()) {
            static const std::map<std::string, std::string> abbr = {
                {"Reflection", "Refl."}, {"Environment", "Env."}, {"Strength", "Str."}, {"Threshold", "Thresh."}, {"Distance", "Dist."},
                {"Rotation", "Rot."}, {"Softness", "Soft."}, {"Thickness", "Thick."}, {"Horizontal", "Horiz."}, {"Vertical", "Vert."},
                {"Encoding", "Enc."}, {"Element", "El."}, {"Defocus", "Defoc."}, {"Diopter", "Diop."},
                {"Length", "Len."}, {"Position", "Pos."}, {"Background", "Bg"}, {"Foreground", "Fg"}, {"Maximum", "Max"}, {"Minimum", "Min"},
                {"Protection", "Prot."}, {"Evolution", "Evol."}, {"Curvature", "Curv."}, {"Registration", "Reg."}, {"Exposure", "Exp."},
                {"Direction", "Dir."}, {"Interval", "Int."}, {"Reference", "Ref."}, {"Samples", "Smpl."}, {"Roughness", "Rough."},
                {"Green", "Grn"}, {"Feather", "Feath."}, {"Amount", "Amt."}, {"Second", "Sec."}, {"Response", "Resp."}, {"Gradient", "Grad."}};
            if (!fits()) {
                if (words.size() >= 3 && words[words.size() - 2] == "per") { words[words.size() - 3] += "/" + words.back().substr(0, 1); words.resize(words.size() - 2); }
                for (size_t i = 0; i < words.size() && !fits(); ++i) if (auto it = abbr.find(words[i]); it != abbr.end()) words[i] = it->second;
            }
            while (words.size() > 1 && !fits()) words.erase(words.begin());
            out = join();
        }
        fitted[key] = out;
        return out;
    }

    void build(const PanelBinding& b) {
        cards.clear();
        std::vector<std::string> order;
        std::map<std::string, std::vector<const ParamSpec*>> groups;
        for (const auto& s : b.module->params) {
            if (!b.visible(s)) continue;
            if (!groups.count(s.group)) order.push_back(s.group);
            groups[s.group].push_back(&s);
        }
        ungrouped = std::all_of(order.begin(), order.end(), [](const std::string& g) { return g.empty() || lower(g) == "global"; });
        for (const auto& g : order) buildGroup(g, groups[g]);
        if (current.empty() && !cards.empty()) current = cards.front().group;
    }

    static std::string rangePartner(const std::string& id) {
        const auto swap = [&](const std::string& a, const std::string& b) -> std::string {
            if (id.rfind(a, 0) == 0) return b + id.substr(a.size());
            if (id.size() > a.size() && id.compare(id.size() - a.size(), a.size(), a) == 0) return id.substr(0, id.size() - a.size()) + b;
            return {};
        };
        for (auto [a, b] : {std::pair{"min_", "max_"}, std::pair{"_min", "_max"}, std::pair{"_near", "_far"}}) {
            auto r = swap(a, b);
            if (!r.empty()) return r;
        }
        return {};
    }

    static Kind kindOf(const ParamSpec& s) {
        if (s.type == PType::Bool) return Kind::Toggle;
        if (s.type == PType::Choice) return s.options.size() <= 4 ? Kind::Seg : Kind::Drop;
        if (s.type == PType::Int && lower(s.id).find("seed") != std::string::npos) return Kind::Field;
        return Kind::Bar;
    }

    void buildGroup(const std::string& group, const std::vector<const ParamSpec*>& list) {
        const std::wstring title = group.empty() ? L"Main" : wide(group);
        std::set<const ParamSpec*> used;
        struct Pending { size_t anchor; Card card; std::vector<const ParamSpec*> own; };
        std::vector<Pending> widgets;
        const auto find = [&](const std::string& id) -> const ParamSpec* {
            for (auto* p : list) if (p->id == id) return p;
            return nullptr;
        };
        for (size_t i = 0; i < list.size(); ++i) {
            const ParamSpec* p = list[i];
            if (used.count(p)) continue;
            Card c;
            c.group = group;
            c.title = title;
            c.tone = familyOf(*p);
            std::vector<const ParamSpec*> own;
            if (p->type == PType::Point) { c.widget = Widget::Pad; c.wa = p; c.tone = Family::Space; }
            else if (p->type == PType::Color) { c.widget = Widget::Wheel; c.wa = p; c.tone = Family::Colour; }
            else if (p->type == PType::Double && p->unit == Unit::Degrees && has(lower(p->id), {"angle", "direction"})) {
                c.widget = Widget::Dial;
                c.wa = p;
                own.push_back(p);
                const std::string prefix = lower(p->id).substr(0, lower(p->id).find("angle"));
                for (auto* q : list) {
                    if (q == p || used.count(q) || q->type != PType::Double || q->unit == Unit::Degrees) continue;
                    const std::string qi = lower(q->id);
                    if ((!prefix.empty() && qi.rfind(prefix, 0) == 0) || has(qi, {"length", "amount", "strength", "highlight", "light"})) { c.wb = q; own.push_back(q); break; }
                }
            } else if (p->type == PType::Double || p->type == PType::Int) {
                const std::string other = rangePartner(p->id);
                const ParamSpec* q = other.empty() ? nullptr : find(other);
                if (q && !used.count(q) && q->type == p->type) {
                    c.widget = Widget::Range;
                    c.wa = p;
                    c.wb = q;
                    own = {p, q};
                }
            }
            if (c.widget == Widget::None) continue;
            used.insert(p);
            if (c.wb) used.insert(c.wb);
            widgets.push_back({i, std::move(c), std::move(own)});
        }
        std::set<const ParamSpec*> taken;
        std::vector<std::pair<size_t, Card>> all;
        for (auto& w : widgets) {
            for (auto* p : w.own) w.card.ctls.push_back({kindOf(*p), p});
            if (w.card.widget != Widget::Range) {
                for (size_t j = w.anchor + 1; j < list.size() && w.card.ctls.size() < 4; ++j) {
                    const ParamSpec* q = list[j];
                    if (q->type == PType::Point || q->type == PType::Color) break;
                    if (used.count(q) || taken.count(q)) continue;
                    w.card.ctls.push_back({kindOf(*q), q});
                    taken.insert(q);
                }
            }
        }
        std::vector<const ParamSpec*> simple;
        for (auto* p : list) if (!used.count(p) && !taken.count(p)) simple.push_back(p);
        size_t room = 0;
        for (auto& w : widgets) if (w.card.widget != Widget::Range) room += 4 - std::min<size_t>(4, w.card.ctls.size());
        if (!simple.empty() && simple.size() <= room) {
            for (auto* q : simple) {
                Pending* best = nullptr;
                for (auto& w : widgets)
                    if (w.card.widget != Widget::Range && w.card.ctls.size() < 4 && (!best || w.card.ctls.size() < best->card.ctls.size())) best = &w;
                best->card.ctls.push_back({kindOf(*q), q});
            }
            simple.clear();
        }
        for (auto& w : widgets) {
            if (w.card.widget == Widget::Range || !w.card.ctls.empty() || simple.empty()) continue;
            const auto pos = [&](const ParamSpec* q) { return static_cast<long>(std::find(list.begin(), list.end(), q) - list.begin()); };
            for (int t = 0; t < 2 && simple.size() > 1; ++t) {
                auto best = std::min_element(simple.begin(), simple.end(), [&](auto* a, auto* b) {
                    return std::labs(pos(a) - static_cast<long>(w.anchor)) < std::labs(pos(b) - static_cast<long>(w.anchor));
                });
                w.card.ctls.push_back({kindOf(**best), *best});
                simple.erase(best);
            }
        }
        for (auto& w : widgets) {
            if (w.card.widget == Widget::Range || !w.card.ctls.empty()) continue;
            for (auto& d : widgets) {
                while (d.card.widget != Widget::Range && d.card.ctls.size() > std::max<size_t>(2, d.own.size()) && w.card.ctls.size() < 2) {
                    w.card.ctls.insert(w.card.ctls.begin(), d.card.ctls.back());
                    d.card.ctls.pop_back();
                }
            }
        }
        for (auto& w : widgets) all.emplace_back(w.anchor, std::move(w.card));
        size_t next = 0;
        while (next < simple.size()) {
            Card c;
            c.group = group;
            c.title = title;
            std::map<Family, int> votes;
            const size_t left = simple.size() - next, chunks = (left + 5) / 6, count = (left + chunks - 1) / chunks;
            const size_t anchor = std::find(list.begin(), list.end(), simple[next]) - list.begin();
            for (size_t k = 0; k < count; ++k, ++next) {
                c.ctls.push_back({kindOf(*simple[next]), simple[next]});
                ++votes[familyOf(*simple[next])];
            }
            c.tone = std::max_element(votes.begin(), votes.end(), [](auto& a, auto& b) { return a.second < b.second; })->first;
            all.emplace_back(anchor, std::move(c));
        }
        std::stable_sort(all.begin(), all.end(), [](auto& a, auto& b) { return a.first < b.first; });
        const size_t first = cards.size();
        for (auto& [anchor, c] : all) cards.push_back(std::move(c));
        const size_t n = cards.size() - first;
        if (lower(group) == "global") {
            for (size_t k = 0; k < n; ++k) cards[first + k].tone = Family::Global;
        } else if (group.empty() && ungrouped) {
            std::map<std::wstring, int> total, seen;
            for (size_t k = 0; k < n; ++k) {
                auto& c = cards[first + k];
                if (c.widget == Widget::None || c.widget == Widget::Range) {
                    std::map<Family, int> v;
                    for (const auto& ctl : c.ctls) ++v[familyOf(*ctl.spec)];
                    if (c.wa) v[familyOf(*c.wa)] += 2;
                    c.tone = std::max_element(v.begin(), v.end(), [](auto& a, auto& b) { return a.second < b.second; })->first;
                }
                c.title = familyName(c.tone);
                ++total[c.title];
            }
            for (size_t k = 0; k < n; ++k) {
                auto& c = cards[first + k];
                if (total[c.title] > 1) c.count = std::to_wstring(++seen[c.title]) + L"/" + std::to_wstring(total[c.title]);
            }
        } else {
            std::map<Family, int> votes;
            for (auto* p : list) ++votes[familyOf(*p)];
            const Family tone = std::max_element(votes.begin(), votes.end(), [](auto& a, auto& b) { return a.second < b.second; })->first;
            for (size_t k = 0; k < n; ++k) {
                cards[first + k].tone = tone;
                if (n > 1) cards[first + k].count = std::to_wstring(k + 1) + L"/" + std::to_wstring(n);
            }
        }
    }

    static int rowsFor(size_t n) { return static_cast<int>(std::clamp<size_t>((n + 2) / 3, 1, 3)); }

    void fitHeight() {
        if (!window || cards.empty()) return;
        const int rows = rowsFor(cards.size());
        const float want = topBar + topPad + rows * 108.f + (rows - 1) * gap + bottomPad;
        RECT wr{}, cr{};
        GetWindowRect(window, &wr);
        GetClientRect(window, &cr);
        const int h = px(want) + (wr.bottom - wr.top) - (cr.bottom - cr.top);
        if (h != wr.bottom - wr.top) SetWindowPos(window, nullptr, 0, 0, wr.right - wr.left, h, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
    }

    D2D1_SIZE_F size() const {
        RECT r{};
        GetClientRect(window, &r);
        return {dip(r.right), dip(r.bottom)};
    }

    void layout() {
        const auto sz = size();
        const int rows = rowsFor(cards.size());
        const int cols = static_cast<int>((cards.size() + rows - 1) / rows), shown = std::clamp(cols, 1, 3);
        const float rowH = std::max(60.f, (sz.height - topBar - topPad - bottomPad - (rows - 1) * gap) / rows);
        const float colW = std::max(300.f, (sz.width - 2 * sidePad - (shown - 1) * gap) / shown);
        pitch = colW + gap;
        contentW = 2 * sidePad + cols * colW + std::max(0, cols - 1) * gap;
        scrollX = clampf(scrollX, 0, std::max(0.f, contentW - sz.width));
        for (size_t i = 0; i < cards.size(); ++i) {
            auto& c = cards[i];
            const float x = sidePad + (i / rows) * (colW + gap) - scrollX, y = topBar + topPad + (i % rows) * (rowH + gap);
            c.rect = {x, y, x + colW, y + rowH};
            const float bodyTop = y + cardPad + 14 + 8, bodyBottom = c.rect.bottom - cardPad;
            const float left = x + cardPad, right = c.rect.right - cardPad, mid = (bodyTop + bodyBottom) / 2;
            if (c.widget == Widget::None) {
                const size_t n = c.ctls.size();
                const size_t perRow = n <= 2 ? n : (n + 1) / 2;
                const float h = 32.f, rowsH = n <= 2 ? h : 2 * h + 6;
                for (size_t k = 0; k < n; ++k) {
                    const size_t row = k / perRow, col = k % perRow, inRow = std::min(perRow, n - row * perRow);
                    const float w = (right - left - (inRow - 1) * 8.f) / inRow;
                    const float cx = left + col * (w + 8), cy = std::min(mid - rowsH / 2, bodyTop + 2) + row * (h + 6);
                    c.ctls[k].rect = {cx, cy, cx + w, cy + h};
                }
            } else if (c.widget == Widget::Range) {
                c.wrect = {left, mid - 21, right, mid - 5};
                const size_t n = c.ctls.size();
                const float w = (right - left - (n - 1) * 6.f) / std::max<size_t>(n, 1);
                for (size_t k = 0; k < n; ++k) c.ctls[k].rect = {left + k * (w + 6), mid + 3, left + k * (w + 6) + w, mid + 27};
            } else {
                c.wrect = {left, mid - 32, left + 64, mid + 32};
                const float gx = left + 76;
                const size_t n = c.ctls.size();
                const int colsN = n <= 2 ? 1 : 2, rowsN = static_cast<int>((n + colsN - 1) / colsN);
                const float w = (right - gx - (colsN - 1) * 6.f) / colsN, total = rowsN * 24.f + (rowsN - 1) * 6.f;
                for (size_t k = 0; k < n; ++k) {
                    const float cx = gx + (k % colsN) * (w + 6), cy = mid - total / 2 + (k / colsN) * 30.f;
                    c.ctls[k].rect = {cx, cy, cx + w, cy + 24};
                }
            }
            for (auto& k : c.ctls) fitSeg(k);
        }
    }

    void fitSeg(Ctl& k) const {
        if (k.kind != Kind::Seg || !fChip) return;
        float widest = 0;
        for (const auto& o : k.spec->options) widest = std::max(widest, measure(wide(o), fChip.Get()));
        const float need = 6 + 3.f * (k.spec->options.size() - 1) + k.spec->options.size() * (widest + 16);
        k.asDrop = need > k.rect.right - k.rect.left;
        const float key = fLabel ? measure(wide(k.spec->label), fLabel.Get()) + 14 : 0;
        k.keyW = !k.asDrop && need + key <= k.rect.right - k.rect.left ? key : 0;
        if (k.keyW == 0) k.asDrop = true;
    }

    void paint() {
        if (!resources()) return;
        layout();
        rt->BeginDraw();
        rt->Clear(rgb(0x141618));
        const auto sz = size();
        rt->PushAxisAlignedClip(D2D1::RectF(0, topBar, sz.width, sz.height), D2D1_ANTIALIAS_MODE_ALIASED);
        for (auto& c : cards) if (c.rect.right > 0 && c.rect.left < sz.width) drawCard(c);
        drawFades(sz);
        rt->PopAxisAlignedClip();
        drawBar(sz);
        drawScrollbar(sz);
        if (drop.open) drawDrop(sz);
        else drawTip(sz);
        if (rt->EndDraw() == D2DERR_RECREATE_TARGET) { rt.Reset(); InvalidateRect(window, nullptr, FALSE); }
    }

    void drawBar(const D2D1_SIZE_F& sz) {
        color(0x141618);
        rt->FillRectangle(D2D1::RectF(0, 0, sz.width, topBar), brush.Get());
        color(0x3f98a6, 0.45f);
        rt->FillRectangle(D2D1::RectF(0, topBar - 1, sz.width, topBar), brush.Get());
        const float cy = topBar / 2;
        color(0x3f98a6, 0.35f);
        fillRound(D2D1::RectF(16, cy - 7, 25, cy + 7), 2);
        color(0x3f98a6, 0.6f);
        fillRound(D2D1::RectF(19.5f, cy - 7, 28.5f, cy + 7), 2);
        color(0x3f98a6);
        fillRound(D2D1::RectF(23, cy - 7, 32, cy + 7), 2);
        wordmark(40, cy);
        float x = 40 + measure(L"TEMPORAL", fWord.Get()) + 8 * 0.22f * 13 + 12;
        color(0x33393e);
        rt->FillRectangle(D2D1::RectF(x, cy - 8, x + 1, cy + 8), brush.Get());
        x += 13;
        color(moduleColor);
        const float nameW = measure(moduleName, fHead.Get()) + 2;
        text(moduleName, fHead.Get(), D2D1::RectF(x, 0, x + nameW, topBar));
        x += nameW + 12;
        color(0x33393e);
        rt->FillRectangle(D2D1::RectF(x, cy - 8, x + 1, cy + 8), brush.Get());
        x += 9;
        chips.clear();
        std::set<std::string> seen;
        for (const auto& c : cards) {
            if (!seen.insert(c.group).second) continue;
            const std::wstring label = c.group.empty() ? L"Main" : wide(c.group);
            const float w = measure(label, fChip.Get()) + 18;
            if (x + w > sz.width - 12) break;
            const D2D1_RECT_F r = D2D1::RectF(x, cy - 13, x + w, cy + 13);
            chips.emplace_back(c.group, r);
            const bool cur = c.group == current, hot = inside(r, hoverX, hoverY);
            if (cur || hot) { color(0x1f2326); fillRound(r, 6); }
            color(cur || hot ? 0xe8e4dc : 0xa3a9a5);
            text(label, fChip.Get(), r, DWRITE_TEXT_ALIGNMENT_CENTER);
            x += w + 2;
        }
        if (failed) {
            color(0xc97a62);
            text(L"Edit failed", fSmall.Get(), D2D1::RectF(sz.width - 90, 0, sz.width - 12, topBar), DWRITE_TEXT_ALIGNMENT_TRAILING);
        }
    }

    void drawFades(const D2D1_SIZE_F& sz) {
        if (contentW <= sz.width + 1) return;
        const auto fade = [&](float x0, float x1, bool right) {
            ComPtr<ID2D1GradientStopCollection> stops;
            const D2D1_GRADIENT_STOP s[] = {{0.f, rgb(0x141618, right ? 0.f : 1.f)}, {1.f, rgb(0x141618, right ? 1.f : 0.f)}};
            rt->CreateGradientStopCollection(s, 2, stops.GetAddressOf());
            ComPtr<ID2D1LinearGradientBrush> g;
            rt->CreateLinearGradientBrush(D2D1::LinearGradientBrushProperties(D2D1::Point2F(x0, 0), D2D1::Point2F(x1, 0)), stops.Get(), g.GetAddressOf());
            if (g) rt->FillRectangle(D2D1::RectF(x0, topBar, x1, sz.height - 10), g.Get());
        };
        if (scrollX > 1) fade(0, 24, false);
        if (scrollX < contentW - sz.width - 1) fade(sz.width - 24, sz.width, true);
    }

    D2D1_RECT_F thumbRect(const D2D1_SIZE_F& sz) const {
        const float track = sz.width - 2 * sidePad, ratio = sz.width / contentW;
        const float w = std::max(40.f, track * ratio), maxScroll = contentW - sz.width;
        const float x = sidePad + (maxScroll > 0 ? scrollX / maxScroll : 0) * (track - w);
        return D2D1::RectF(x, sz.height - 8, x + w, sz.height - 2);
    }

    void drawScrollbar(const D2D1_SIZE_F& sz) {
        if (contentW <= sz.width + 1) return;
        auto r = thumbRect(sz);
        r.top += 1; r.bottom -= 1;
        color(drag.mode == Drag::Scroll || inside(thumbRect(sz), hoverX, hoverY) ? 0x6a7177 : 0x4a5157);
        fillRound(r, 2);
    }

    void drawCard(Card& c) {
        bool anyOn = c.wa && enabled(*c.wa);
        for (const auto& k : c.ctls) anyOn = anyOn || enabled(*k.spec);
        fade = anyOn ? 1.f : 0.55f;
        const D2D1_COLOR_F cardTone = rgb(toneHex(c.tone));
        color(0x1b1e21);
        fillRound(c.rect, 10);
        color(cardTone, 0.26f);
        fillRound(c.rect, 10);
        color(cardTone, 0.8f);
        strokeRound(D2D1::RectF(c.rect.left + 0.5f, c.rect.top + 0.5f, c.rect.right - 0.5f, c.rect.bottom - 0.5f), 10, 1);
        if (c.flash >= 0) {
            const float t = static_cast<float>(now() - c.flash);
            const float a = t < 0.25f ? 1.f : clampf(1.f - (t - 0.25f) / 1.f, 0.f, 1.f);
            if (a > 0) {
                color(0x3f98a6, 0.35f * a);
                strokeRound(D2D1::RectF(c.rect.left - 2, c.rect.top - 2, c.rect.right + 2, c.rect.bottom + 2), 12, 4);
                color(0x5fb6c4, a);
                strokeRound(D2D1::RectF(c.rect.left + 0.5f, c.rect.top + 0.5f, c.rect.right - 0.5f, c.rect.bottom - 0.5f), 10, 1.5f);
            } else c.flash = -1;
        }
        const D2D1_COLOR_F tone = rgb(toneHex(c.tone));
        const float hy = c.rect.top + cardPad + 7;
        color(0xf2eee6);
        text(c.title, fHead.Get(), D2D1::RectF(c.rect.left + cardPad, hy - 8, c.rect.right - cardPad - 90, hy + 8));
        float tx = c.rect.left + cardPad + measure(c.title, fHead.Get()) + 6;
        if (!c.count.empty()) {
            color(0xb4b8b8);
            text(c.count, fLabel.Get(), D2D1::RectF(tx, hy - 8, tx + 40, hy + 8));
            tx += measure(c.count, fLabel.Get()) + 10;
        }
        const bool headKey = (c.widget == Widget::Pad || c.widget == Widget::Wheel) && animatable(*c.wa);
        if (headKey) {
            const float outer = fade;
            if (!enabled(*c.wa) && fade > 0.6f) fade *= 0.5f;
            drawKey(headKeyRect(c), *c.wa, tone);
            fade = outer;
        }
        if ((c.widget == Widget::Pad || c.widget == Widget::Wheel) && enabled(*c.wa)) {
            wchar_t buf[48];
            if (c.widget == Widget::Dial) swprintf_s(buf, L"%.0f\u00b0", value(*c.wa));
            else if (c.widget == Widget::Pad) swprintf_s(buf, L"%.2f, %.2f", value(*c.wa, 0), value(*c.wa, 1));
            else swprintf_s(buf, L"#%02X%02X%02X", static_cast<int>(clampf(static_cast<float>(value(*c.wa, 0)), 0, 1) * 255 + 0.5f),
                            static_cast<int>(clampf(static_cast<float>(value(*c.wa, 1)), 0, 1) * 255 + 0.5f), static_cast<int>(clampf(static_cast<float>(value(*c.wa, 2)), 0, 1) * 255 + 0.5f));
            color(0xb4b8b8);
            valueText(buf, D2D1::RectF(c.rect.right - cardPad - 120, hy - 8, c.rect.right - cardPad - (headKey ? 18 : 0), hy + 8));
        }
        const bool won = !c.wa || enabled(*c.wa);
        const float cardFade = fade;
        if (!won && fade > 0.6f) fade *= 0.5f;
        switch (c.widget) {
        case Widget::Dial: drawDial(c, tone); break;
        case Widget::Pad: drawPad(c, tone); break;
        case Widget::Wheel: drawWheel(c); break;
        case Widget::Range: drawRange(c, tone); break;
        default: break;
        }
        fade = cardFade;
        for (const auto& k : c.ctls) drawCtl(k, tone, c.group);
        fade = 1.f;
    }

    float frac(const ParamSpec& s, double v) const {
        const double lo = s.dmin, hi = s.dmax;
        return hi > lo ? clampf(static_cast<float>((v - lo) / (hi - lo)), 0, 1) : 0.f;
    }

    void drawCtl(const Ctl& k, const D2D1_COLOR_F& tone, const std::string& group) {
        const auto& s = *k.spec;
        const auto& r = k.rect;
        const bool on = enabled(s);
        const float outer = fade;
        if (!on && fade > 0.6f) fade *= 0.5f;
        drawCtlBody(k, tone, group, on);
        fade = outer;
    }

    void drawCtlBody(const Ctl& k, const D2D1_COLOR_F& tone, const std::string& group, bool on) {
        const auto& s = *k.spec;
        const auto& r = k.rect;
        const bool hot = inside(r, hoverX, hoverY) && on;
        if (k.kind == Kind::Bar) {
            color(hot ? 0x2a2f33 : 0x24282c);
            fillRound(r, 6);
            const double v = value(s);
            const float p = frac(s, v), w = r.right - r.left;
            const bool bi = s.dmin < 0 && s.dmax > 0;
            const float z = bi ? frac(s, 0) : 0.f;
            rt->PushAxisAlignedClip(r, D2D1_ANTIALIAS_MODE_ALIASED);
            {
                color(tone, 0.36f);
                rt->FillRectangle(D2D1::RectF(r.left + std::min(z, p) * w, r.top, r.left + std::max(z, p) * w, r.bottom), brush.Get());
                const float tick = bi ? z : frac(s, s.def);
                if (bi || (s.def > s.dmin && s.def < s.dmax)) {
                    color(0xe8e4dc, 0.5f);
                    const float st = r.bottom - r.top < 30 ? 3.f : 4.f;
                    rt->FillRectangle(D2D1::RectF(r.left + tick * w, r.top, r.left + tick * w + 1, r.top + st), brush.Get());
                    rt->FillRectangle(D2D1::RectF(r.left + tick * w, r.bottom - st, r.left + tick * w + 1, r.bottom), brush.Get());
                }
                const float ex = clampf(r.left + p * w - 1, r.left, r.right - 2);
                color(tone);
                const float se = r.bottom - r.top < 30 ? 4.f : 6.f;
                rt->FillRectangle(D2D1::RectF(ex, r.top, ex + 2, r.top + se), brush.Get());
                rt->FillRectangle(D2D1::RectF(ex, r.bottom - se, ex + 2, r.bottom), brush.Get());
            }
            rt->PopAxisAlignedClip();
            const std::wstring val = formatValue(s, v);
            const bool keyed = animatable(s);
            const float vr = r.right - (keyed ? 24.f : 9.f);
            const float vw = measure(val, fValue.Get(), true) + 2;
            const float lw = vr - vw - 6 - (r.left + 9);
            color(0xd4d0c8);
            text(fitLabel(s.label, group, lw), fLabel.Get(), D2D1::RectF(r.left + 9, r.top, r.left + 9 + lw, r.bottom));
            color(0xe8e4dc);
            valueText(val, D2D1::RectF(r.left + 9, r.top, vr, r.bottom));
            if (keyed) drawKey(keyRect(r), s, tone);
        } else if (k.kind == Kind::Toggle) {
            color(hot ? 0x2a2f33 : 0x24282c);
            fillRound(r, 6);
            const bool v = value(s) != 0;
            const float cy = (r.top + r.bottom) / 2, sx = r.right - 9 - 24;
            color(v ? tone : rgb(0x3b4146), 1.f);
            fillRound(D2D1::RectF(sx, cy - 7, sx + 24, cy + 7), 7);
            color(v ? 0x141618 : 0xa3a9a5, 1.f);
            rt->FillEllipse(D2D1::Ellipse(D2D1::Point2F(v ? sx + 17 : sx + 7, cy), 5, 5), brush.Get());
            color(0xd4d0c8, 1.f);
            text(fitLabel(s.label, group, sx - 8 - (r.left + 9)), fLabel.Get(), D2D1::RectF(r.left + 9, r.top, sx - 8, r.bottom));
        } else if (k.kind == Kind::Seg && !k.asDrop) {
            color(0x24282c);
            fillRound(r, 6);
            const size_t n = s.options.size();
            const int sel = static_cast<int>(std::lround(value(s)));
            if (k.keyW > 0) {
                color(0xa4aaa4);
                text(wide(s.label), fLabel.Get(), D2D1::RectF(r.left + 9, r.top, r.left + k.keyW, r.bottom));
            }
            const float x0 = r.left + k.keyW;
            const float w = (r.right - x0 - 6 - (n - 1) * 3.f) / n;
            for (size_t i = 0; i < n; ++i) {
                const D2D1_RECT_F b = D2D1::RectF(x0 + 3 + i * (w + 3), r.top + 3, x0 + 3 + i * (w + 3) + w, r.bottom - 3);
                const bool selected = static_cast<int>(i) == sel;
                if (selected) { color(0x363c42); fillRound(b, 4); }
                else if (inside(b, hoverX, hoverY) && on) { color(0x2c3135); fillRound(b, 4); }
                color(selected ? 0xe8e4dc : 0xa3a9a5, 1.f);
                const std::wstring label = wide(s.options[i]);
                text(label, fChip.Get(), D2D1::RectF(b.left + 4, b.top, b.right - 4, b.bottom), DWRITE_TEXT_ALIGNMENT_CENTER);
            }
        } else if (k.kind == Kind::Drop || k.kind == Kind::Seg) {
            color(hot ? 0x2a2f33 : 0x24282c);
            fillRound(r, 6);
            const int sel = std::clamp(static_cast<int>(std::lround(value(s))), 0, static_cast<int>(s.options.size()) - 1);
            const std::wstring val = s.options.empty() ? L"" : wide(s.options[sel]);
            const float avail = r.right - 24 - (r.left + 9), vw = measure(val, fLabel.Get()) + 1, room = std::max(0.f, avail - vw - 8);
            const std::wstring key = fitLabel(s.label, group, room);
            const float kw = measure(key, fLabel.Get());
            const bool showKey = kw >= 20 && kw <= room;
            if (showKey) {
                color(0xa4aaa4, 1.f);
                text(key, fLabel.Get(), D2D1::RectF(r.left + 9, r.top, r.left + 9 + kw + 1, r.bottom));
            }
            color(0xe8e4dc, 1.f);
            text(val, fLabel.Get(), D2D1::RectF(showKey ? r.left + 17 + kw : r.left + 9, r.top, r.right - 24, r.bottom));
            const float cx = r.right - 13, cy = (r.top + r.bottom) / 2;
            color(0x9aa09c);
            rt->DrawLine(D2D1::Point2F(cx - 3.5f, cy - 1.5f), D2D1::Point2F(cx, cy + 2), brush.Get(), 1.4f);
            rt->DrawLine(D2D1::Point2F(cx, cy + 2), D2D1::Point2F(cx + 3.5f, cy - 1.5f), brush.Get(), 1.4f);
        } else {
            color(0x24282c);
            fillRound(r, 6);
            color(0x33393e);
            strokeRound(D2D1::RectF(r.left + 0.5f, r.top + 0.5f, r.right - 0.5f, r.bottom - 0.5f), 6, 1);
            color(0xd4d0c8);
            text(fitLabel(s.label, group, r.right - 100 - (r.left + 9)), fLabel.Get(), D2D1::RectF(r.left + 9, r.top, r.right - 100, r.bottom));
            color(0xe8e4dc);
            valueText(formatValue(s, value(s)), D2D1::RectF(r.left + 9, r.top, r.right - 52, r.bottom));
            for (int i = 0; i < 2; ++i) {
                const D2D1_RECT_F b = stepRect(r, i);
                if (inside(b, hoverX, hoverY)) { color(0x2c3135); fillRound(b, 4); }
                color(0xa3a9a5);
                const float cx = (b.left + b.right) / 2, cy = (b.top + b.bottom) / 2;
                rt->DrawLine(D2D1::Point2F(cx - 4, cy), D2D1::Point2F(cx + 4, cy), brush.Get(), 1.4f);
                if (i) rt->DrawLine(D2D1::Point2F(cx, cy - 4), D2D1::Point2F(cx, cy + 4), brush.Get(), 1.4f);
            }
        }
    }

    static D2D1_RECT_F stepRect(const D2D1_RECT_F& r, int i) {
        const float right = r.right - 3 - (1 - i) * 22;
        return D2D1::RectF(right - 20, r.top + 3, right, r.bottom - 3);
    }

    D2D1_POINT_2F center(const Card& c) const { return D2D1::Point2F((c.wrect.left + c.wrect.right) / 2, (c.wrect.top + c.wrect.bottom) / 2); }

    void drawDial(const Card& c, const D2D1_COLOR_F& tone) {
        const auto o = center(c);
        color(0x24282c);
        rt->FillEllipse(D2D1::Ellipse(o, 30, 30), brush.Get());
        color(0x33393e);
        rt->DrawEllipse(D2D1::Ellipse(o, 30, 30), brush.Get(), 1);
        color(0x2c3135);
        rt->DrawEllipse(D2D1::Ellipse(o, 18, 18), brush.Get(), 1, dashed.Get());
        color(0x5a6167);
        rt->DrawLine(D2D1::Point2F(o.x, o.y - 30), D2D1::Point2F(o.x, o.y - 25), brush.Get(), 1);
        color(0x8a9096);
        rt->DrawLine(D2D1::Point2F(o.x + 30, o.y), D2D1::Point2F(o.x + 24, o.y), brush.Get(), 1.5f);
        const float a = static_cast<float>(value(*c.wa) * 3.14159265 / 180.0);
        const float rr = 8 + 18 * (c.wb ? std::max(0.25f, frac(*c.wb, value(*c.wb))) : 0.75f);
        const D2D1_POINT_2F t = D2D1::Point2F(o.x + rr * std::cos(a), o.y - rr * std::sin(a));
        color(tone);
        rt->DrawLine(o, t, brush.Get(), 2);
        rt->FillEllipse(D2D1::Ellipse(t, 4, 4), brush.Get());
        color(0xe8e4dc);
        rt->FillEllipse(D2D1::Ellipse(o, 2.5f, 2.5f), brush.Get());
    }

    void drawPad(const Card& c, const D2D1_COLOR_F& tone) {
        const auto& r = c.wrect;
        color(0x24282c);
        fillRound(r, 6);
        color(0x33393e);
        strokeRound(D2D1::RectF(r.left + 0.5f, r.top + 0.5f, r.right - 0.5f, r.bottom - 0.5f), 6, 1);
        for (int i = 1; i < 4; ++i) {
            color(i == 2 ? 0x353b40 : 0x2a2f33);
            const float f = r.left + 4 + i * 14.f, g = r.top + 4 + i * 14.f;
            rt->DrawLine(D2D1::Point2F(f, r.top + 4), D2D1::Point2F(f, r.bottom - 4), brush.Get(), 1);
            rt->DrawLine(D2D1::Point2F(r.left + 4, g), D2D1::Point2F(r.right - 4, g), brush.Get(), 1);
        }
        const float x = r.left + 6 + 52 * clampf(static_cast<float>(value(*c.wa, 0)), 0, 1);
        const float y = r.bottom - 6 - 52 * clampf(static_cast<float>(value(*c.wa, 1)), 0, 1);
        color(tone, 0.4f);
        rt->DrawEllipse(D2D1::Ellipse(D2D1::Point2F(x, y), 9, 9), brush.Get(), 1);
        color(tone);
        rt->FillEllipse(D2D1::Ellipse(D2D1::Point2F(x, y), 4.5f, 4.5f), brush.Get());
    }

    void drawWheel(const Card& c) {
        const auto o = center(c);
        for (int i = 0; i < 36; ++i) {
            const float a0 = i * 10.f * 3.14159265f / 180, a1 = (i + 1) * 10.f * 3.14159265f / 180 + 0.01f;
            ComPtr<ID2D1PathGeometry> g;
            d2d->CreatePathGeometry(g.GetAddressOf());
            ComPtr<ID2D1GeometrySink> sink;
            if (!g || FAILED(g->Open(sink.GetAddressOf()))) continue;
            sink->BeginFigure(o, D2D1_FIGURE_BEGIN_FILLED);
            sink->AddLine(D2D1::Point2F(o.x + 30 * std::cos(a0), o.y - 30 * std::sin(a0)));
            sink->AddArc({D2D1::Point2F(o.x + 30 * std::cos(a1), o.y - 30 * std::sin(a1)), D2D1::SizeF(30, 30), 0, D2D1_SWEEP_DIRECTION_COUNTER_CLOCKWISE, D2D1_ARC_SIZE_SMALL});
            sink->EndFigure(D2D1_FIGURE_END_CLOSED);
            sink->Close();
            color(hsl(i * 10.f + 5, 0.45f, 0.55f));
            rt->FillGeometry(g.Get(), brush.Get());
        }
        ComPtr<ID2D1GradientStopCollection> stops;
        const D2D1_GRADIENT_STOP s[] = {{0.f, rgb(0x24282c, 0.85f)}, {0.7f, rgb(0x24282c, 0.f)}};
        rt->CreateGradientStopCollection(s, 2, stops.GetAddressOf());
        ComPtr<ID2D1RadialGradientBrush> rb;
        rt->CreateRadialGradientBrush(D2D1::RadialGradientBrushProperties(o, D2D1::Point2F(), 30, 30), stops.Get(), rb.GetAddressOf());
        if (rb) rt->FillEllipse(D2D1::Ellipse(o, 30, 30), rb.Get());
        float h = 0, sat = 0;
        toHsv(c, h, sat);
        const float a = h * 3.14159265f / 180, rr = 26 * sat;
        const D2D1_POINT_2F t = D2D1::Point2F(o.x + rr * std::cos(a), o.y - rr * std::sin(a));
        brush->SetColor(D2D1::ColorF(static_cast<float>(value(*c.wa, 0)), static_cast<float>(value(*c.wa, 1)), static_cast<float>(value(*c.wa, 2)), fade));
        rt->FillEllipse(D2D1::Ellipse(t, 5, 5), brush.Get());
        color(0xe8e4dc);
        rt->DrawEllipse(D2D1::Ellipse(t, 5, 5), brush.Get(), 1.5f);
    }

    void toHsv(const Card& c, float& h, float& s) const {
        const float r = clampf(static_cast<float>(value(*c.wa, 0)), 0, 1), g = clampf(static_cast<float>(value(*c.wa, 1)), 0, 1), b = clampf(static_cast<float>(value(*c.wa, 2)), 0, 1);
        const float mx = std::max({r, g, b}), mn = std::min({r, g, b}), d = mx - mn;
        s = mx > 0 ? d / mx : 0;
        if (d <= 0) { h = 0; return; }
        if (mx == r) h = 60 * std::fmod((g - b) / d, 6.f);
        else if (mx == g) h = 60 * ((b - r) / d + 2);
        else h = 60 * ((r - g) / d + 4);
        if (h < 0) h += 360;
    }

    void drawRange(const Card& c, const D2D1_COLOR_F& tone) {
        const auto& box = c.wrect;
        const float my = (box.top + box.bottom) / 2;
        const D2D1_RECT_F r = D2D1::RectF(box.left, my - 3, box.right, my + 3);
        color(0x24282c);
        fillRound(r, 3);
        const float w = r.right - r.left;
        const float lo = frac(*c.wa, value(*c.wa)), hi = frac(*c.wb, value(*c.wb));
        color(tone, 0.8f);
        rt->FillRectangle(D2D1::RectF(r.left + std::min(lo, hi) * w, r.top, r.left + std::max(lo, hi) * w, r.bottom), brush.Get());
        color(tone);
        for (float f : {lo, hi}) {
            const float x = clampf(r.left + f * w - 1.5f, r.left, r.right - 3);
            fillRound(D2D1::RectF(x, my - 7, x + 3, my + 7), 1.5f);
        }
    }

    void drawTip(const D2D1_SIZE_F& sz) {
        tipShown = false;
        if (!tipSpec || drag.mode != Drag::None || now() - tipSince < 0.5) return;
        const auto& s = *tipSpec;
        const std::wstring head = wide(s.label);
        std::wstring body = head;
        if (auto it = reasons.find(s.id); it != reasons.end() && !enabled(s)) body += L"  \u00b7  " + it->second;
        if (!s.hint.empty()) body += L"\n" + wide(s.hint);
        if (animatable(s) && enabled(s)) {
            const int frame = static_cast<int>(std::lround(nowTime));
            if (onKey(s)) body += L"\nKeyframe at frame " + std::to_wstring(frame) + L". Click \u25c6 to remove it.";
            else if (animated(s)) body += L"\nAnimated. Edits key frame " + std::to_wstring(frame) + L"; click \u25c7 to add a key.";
            else body += L"\nClick \u25c7 to add a keyframe at frame " + std::to_wstring(frame) + L".";
        }
        ComPtr<IDWriteTextLayout> layout;
        if (FAILED(dw->CreateTextLayout(body.c_str(), static_cast<UINT32>(body.size()), fSmall.Get(), 300, 200, layout.GetAddressOf()))) return;
        layout->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_LEADING);
        layout->SetWordWrapping(DWRITE_WORD_WRAPPING_WRAP);
        layout->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_NEAR);
        layout->SetFontWeight(DWRITE_FONT_WEIGHT_SEMI_BOLD, {0, static_cast<UINT32>(head.size())});
        DWRITE_TEXT_METRICS m{};
        layout->GetMetrics(&m);
        const float w = m.width + 20, h = m.height + 14;
        const float x = std::clamp(hoverX + 12, 4.f, std::max(4.f, sz.width - w - 4));
        float y = hoverY + 18;
        if (y + h > sz.height - 4) y = hoverY - h - 8;
        const D2D1_RECT_F r = D2D1::RectF(x, y, x + w, y + h);
        color(0x000000, 0.35f);
        fillRound(D2D1::RectF(r.left + 1, r.top + 3, r.right + 1, r.bottom + 4), 7);
        color(0x22262a);
        fillRound(r, 7);
        color(0x3b4146);
        strokeRound(D2D1::RectF(r.left + 0.5f, r.top + 0.5f, r.right - 0.5f, r.bottom - 0.5f), 7, 1);
        color(0xd4d0c8);
        rt->DrawTextLayout(D2D1::Point2F(r.left + 10, r.top + 7), layout.Get(), brush.Get());
        tipShown = true;
    }

    void drawDrop(const D2D1_SIZE_F& sz) {
        const auto& s = *drop.spec;
        color(0x000000, 0.35f);
        fillRound(D2D1::RectF(drop.rect.left + 2, drop.rect.top + 4, drop.rect.right + 2, drop.rect.bottom + 6), 8);
        color(0x22262a);
        fillRound(drop.rect, 8);
        color(0x3b4146);
        strokeRound(D2D1::RectF(drop.rect.left + 0.5f, drop.rect.top + 0.5f, drop.rect.right - 0.5f, drop.rect.bottom - 0.5f), 8, 1);
        const int sel = static_cast<int>(std::lround(value(s)));
        for (int i = 0; i < drop.visible; ++i) {
            const int idx = drop.first + i;
            if (idx >= static_cast<int>(s.options.size())) break;
            const D2D1_RECT_F r = D2D1::RectF(drop.rect.left + 4, drop.rect.top + 4 + i * drop.itemH, drop.rect.right - 4, drop.rect.top + 4 + (i + 1) * drop.itemH);
            if (idx == drop.hover) { color(0x2f353a); fillRound(r, 5); }
            if (idx == sel) { color(0x3f98a6); fillRound(D2D1::RectF(r.left + 4, r.top + 7, r.left + 6, r.bottom - 7), 1); }
            color(idx == sel ? 0xe8e4dc : 0xc9c5bd);
            text(wide(s.options[idx]), fLabel.Get(), D2D1::RectF(r.left + 12, r.top, r.right - 8, r.bottom));
        }
        (void)sz;
    }

    void write(const ParamSpec& s, std::array<double, 3> v) {
        auto b = binding.lock();
        if (!b) return;
        if (s.type == PType::Double || s.type == PType::Int || s.type == PType::Choice)
            v[0] = std::clamp(v[0], s.min, s.max);
        if (s.type == PType::Int || s.type == PType::Choice || s.type == PType::Bool) v[0] = std::round(v[0]);
        {
            std::lock_guard lock(b->mutex);
            if (!b->write || !b->enabled || !b->enabled(s)) return;
            try { b->write(s, v); failed = false; } catch (...) { failed = true; }
        }
        values.v[s.id] = v;
        InvalidateRect(window, nullptr, FALSE);
    }

    std::array<double, 3> vec(const ParamSpec& s) const {
        auto it = values.v.find(s.id);
        return it == values.v.end() ? std::array<double, 3>{s.def, s.def, s.def} : it->second;
    }

    void writeScalar(const ParamSpec& s, double v) { auto a = vec(s); a[0] = v; write(s, a); }

    const Card* cardAt(float x, float y) const {
        for (const auto& c : cards) if (inside(c.rect, x, y)) return &c;
        return nullptr;
    }

    const Ctl* ctlAt(const Card& c, float x, float y) const {
        for (const auto& k : c.ctls) if (inside(k.rect, x, y)) return &k;
        return nullptr;
    }

    void dialTo(const Card& c, float x, float y) {
        const auto o = center(c);
        double deg = std::atan2(o.y - y, x - o.x) * 180 / 3.14159265358979;
        const auto& s = *c.wa;
        while (deg < s.min) deg += 360;
        while (deg > s.max) deg -= 360;
        writeScalar(s, std::clamp(deg, s.min, s.max));
        if (c.wb) {
            const float d = std::hypot(x - o.x, y - o.y);
            const float f = clampf((d - 8) / 18, 0, 1);
            writeScalar(*c.wb, c.wb->dmin + f * (c.wb->dmax - c.wb->dmin));
        }
    }

    void padTo(const Card& c, float x, float y) {
        auto v = vec(*c.wa);
        v[0] = clampf((x - c.wrect.left - 6) / 52, 0, 1);
        v[1] = clampf((c.wrect.bottom - 6 - y) / 52, 0, 1);
        write(*c.wa, v);
    }

    void wheelTo(const Card& c, float x, float y) {
        const auto o = center(c);
        float h = std::atan2(o.y - y, x - o.x) * 180 / 3.14159265f;
        if (h < 0) h += 360;
        const float sat = clampf(std::hypot(x - o.x, y - o.y) / 26, 0, 1);
        auto v = vec(*c.wa);
        const float val = std::max({static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2]), 0.05f});
        const float cc = val * sat, hp = h / 60, xx = cc * (1 - std::fabs(std::fmod(hp, 2.f) - 1)), m = val - cc;
        float r = 0, g = 0, b = 0;
        if (hp < 1) { r = cc; g = xx; } else if (hp < 2) { r = xx; g = cc; } else if (hp < 3) { g = cc; b = xx; }
        else if (hp < 4) { g = xx; b = cc; } else if (hp < 5) { r = xx; b = cc; } else { r = cc; b = xx; }
        write(*c.wa, {r + m, g + m, b + m});
    }

    void rangeTo(const Card& c, float x, bool hi) {
        const ParamSpec& s = hi ? *c.wb : *c.wa;
        const float f = clampf((x - c.wrect.left) / (c.wrect.right - c.wrect.left), 0, 1);
        writeScalar(s, s.dmin + f * (s.dmax - s.dmin));
    }

    void reset(const ParamSpec& s) {
        if (s.type == PType::Color || s.type == PType::Point) write(s, {s.defv[0], s.defv[1], s.defv[2]});
        else writeScalar(s, s.def);
    }

    void openDrop(const Ctl& k) {
        const auto sz = size();
        drop = {};
        drop.open = true;
        drop.spec = k.spec;
        const float maxRows = std::floor((sz.height - 16) / drop.itemH);
        drop.visible = static_cast<int>(std::min<float>(maxRows, static_cast<float>(k.spec->options.size())));
        const float h = drop.visible * drop.itemH + 8;
        float w = k.rect.right - k.rect.left;
        for (const auto& o : k.spec->options) w = std::max(w, measure(wide(o), fLabel.Get()) + 40);
        float top = k.rect.bottom + 4;
        if (top + h > sz.height - 4) top = std::max(4.f, k.rect.top - 4 - h);
        const float left = clampf(k.rect.left, 4, sz.width - w - 4);
        drop.rect = D2D1::RectF(left, top, left + w, top + h);
        const int sel = static_cast<int>(std::lround(value(*k.spec)));
        drop.first = std::clamp(sel - drop.visible / 2, 0, std::max(0, static_cast<int>(k.spec->options.size()) - drop.visible));
        InvalidateRect(window, nullptr, FALSE);
    }

    void openEditor(const Ctl& k) {
        closeEditor(false);
        if (!enabled(*k.spec)) return;
        editSpec = k.spec;
        editComponent = 0;
        if (!editFont) editFont = CreateFontW(-px(12), 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, L"Segoe UI");
        if (!editBrush) editBrush = CreateSolidBrush(RGB(46, 52, 57));
        const auto& r = k.rect;
        editor = CreateWindowExW(0, L"EDIT", L"", WS_CHILD | WS_VISIBLE | ES_RIGHT | ES_AUTOHSCROLL, px(r.left + 6), px(r.top + (r.bottom - r.top) / 2 - 9),
                                 px(r.right - r.left - 12), px(18), window, nullptr, nullptr, nullptr);
        SendMessageW(editor, WM_SETFONT, reinterpret_cast<WPARAM>(editFont), FALSE);
        wchar_t buf[48];
        swprintf_s(buf, L"%.6g", value(*k.spec));
        SetWindowTextW(editor, buf);
        SendMessageW(editor, EM_SETSEL, 0, -1);
        SetWindowSubclass(editor, editProc, 1, reinterpret_cast<DWORD_PTR>(this));
        SetFocus(editor);
    }

    void closeEditor(bool commit) {
        if (!editor) return;
        HWND e = editor;
        editor = nullptr;
        if (commit && editSpec) {
            wchar_t buf[64]{};
            GetWindowTextW(e, buf, 64);
            wchar_t* end = nullptr;
            const double v = std::wcstod(buf, &end);
            if (end != buf && std::isfinite(v)) writeScalar(*editSpec, v);
        }
        editSpec = nullptr;
        DestroyWindow(e);
        SetFocus(window);
    }

    static LRESULT CALLBACK editProc(HWND child, UINT message, WPARAM wp, LPARAM lp, UINT_PTR, DWORD_PTR data) {
        auto& self = *reinterpret_cast<LivePanel*>(data);
        if (message == WM_KEYDOWN && wp == VK_RETURN) { self.closeEditor(true); return 0; }
        if (message == WM_KEYDOWN && wp == VK_ESCAPE) { self.closeEditor(false); return 0; }
        if (message == WM_KILLFOCUS && self.editor == child) { self.closeEditor(true); return 0; }
        if (message == WM_NCDESTROY) RemoveWindowSubclass(child, editProc, 1);
        return DefSubclassProc(child, message, wp, lp);
    }

    void go(const std::string& group) {
        const auto sz = size();
        for (auto& c : cards) {
            if (c.group != group) continue;
            const float base = c.rect.left + scrollX;
            scrollFrom = scrollX;
            scrollTarget = clampf(base - sidePad, 0, std::max(0.f, contentW - sz.width));
            scrollStart = now();
            current = group;
            navigating = true;
            c.flash = static_cast<float>(now());
            for (auto& o : cards) if (&o != &c) o.flash = -1;
            break;
        }
        SetTimer(window, animTimer, 16, nullptr);
        InvalidateRect(window, nullptr, FALSE);
    }

    void tick() {
        bool busy = false;
        if (scrollStart >= 0) {
            const double t = std::min(1.0, (now() - scrollStart) / 0.25);
            const double e = 1 - std::pow(1 - t, 3);
            scrollX = scrollFrom + static_cast<float>(e) * (scrollTarget - scrollFrom);
            if (t >= 1) { scrollStart = -1; navigating = false; } else busy = true;
            if (!navigating) followScroll();
        }
        for (const auto& c : cards) if (c.flash >= 0) busy = true;
        if (!busy) KillTimer(window, animTimer);
        InvalidateRect(window, nullptr, FALSE);
    }

    void followScroll() {
        if (navigating) return;
        for (const auto& c : cards) {
            if ((c.rect.left + c.rect.right) / 2 >= sidePad) {
                if (current != c.group) { current = c.group; }
                return;
            }
        }
    }

    void step(int dir) {
        const auto sz = size();
        const float base = scrollStart >= 0 ? scrollTarget : scrollX;
        scrollFrom = scrollX;
        scrollTarget = clampf((std::round(base / pitch) + dir) * pitch, 0, std::max(0.f, contentW - sz.width));
        if (scrollTarget == scrollX) return;
        scrollStart = now();
        navigating = false;
        SetTimer(window, animTimer, 16, nullptr);
    }

    void scrollBy(float d) {
        const auto sz = size();
        scrollStart = -1;
        navigating = false;
        scrollX = clampf(scrollX + d, 0, std::max(0.f, contentW - sz.width));
        layout();
        followScroll();
        InvalidateRect(window, nullptr, FALSE);
    }

    void toggleKey(const ParamSpec& s) {
        auto b = binding.lock();
        if (!b) return;
        {
            std::lock_guard lock(b->mutex);
            if (!b->keyframe) return;
            try { b->keyframe(s, !onKey(s)); failed = false; } catch (...) { failed = true; }
        }
        refresh(true);
    }

    void mouseDown(float x, float y, bool shift, bool ctrl) {
        (void)shift; (void)ctrl;
        closeEditor(true);
        if (drop.open) {
            if (inside(drop.rect, x, y)) {
                const int idx = drop.first + static_cast<int>((y - drop.rect.top - 4) / drop.itemH);
                if (idx >= 0 && idx < static_cast<int>(drop.spec->options.size())) writeScalar(*drop.spec, idx);
            }
            drop.open = false;
            InvalidateRect(window, nullptr, FALSE);
            return;
        }
        const auto sz = size();
        if (y < topBar) {
            for (const auto& [g, r] : chips) if (inside(r, x, y)) { go(g); return; }
            return;
        }
        if (contentW > sz.width + 1 && y >= sz.height - 10) {
            const auto t = thumbRect(sz);
            if (!inside(D2D1::RectF(t.left, sz.height - 10, t.right, sz.height), x, y)) {
                const float maxScroll = contentW - sz.width;
                scrollBy((x < t.left ? -1 : 1) * std::min(maxScroll, sz.width * 0.8f));
            }
            drag = {};
            drag.mode = Drag::Scroll;
            drag.startX = x;
            drag.startScroll = scrollX;
            SetCapture(window);
            return;
        }
        layout();
        const Card* c = cardAt(x, y);
        if (!c) return;
        if (c->wa && c->widget != Widget::Dial && c->widget != Widget::Range && animatable(*c->wa) && enabled(*c->wa) && inside(headKeyRect(*c), x, y)) {
            toggleKey(*c->wa);
            return;
        }
        for (const auto& k : c->ctls) {
            if (k.kind == Kind::Bar && animatable(*k.spec) && enabled(*k.spec) && inside(keyRect(k.rect), x, y)) {
                toggleKey(*k.spec);
                return;
            }
        }
        if (c->widget != Widget::None && inside(c->wrect, x, y) && enabled(*c->wa)) {
            drag = {};
            drag.card = c;
            if (c->widget == Widget::Dial) { drag.mode = Drag::Dial; dialTo(*c, x, y); }
            else if (c->widget == Widget::Pad) { drag.mode = Drag::Pad; padTo(*c, x, y); }
            else if (c->widget == Widget::Wheel) { drag.mode = Drag::Wheel; wheelTo(*c, x, y); }
            else {
                const float w = c->wrect.right - c->wrect.left;
                const float lo = c->wrect.left + frac(*c->wa, value(*c->wa)) * w, hi = c->wrect.left + frac(*c->wb, value(*c->wb)) * w;
                drag.mode = std::fabs(x - lo) <= std::fabs(x - hi) ? Drag::RangeLo : Drag::RangeHi;
                rangeTo(*c, x, drag.mode == Drag::RangeHi);
            }
            SetCapture(window);
            return;
        }
        const Ctl* k = ctlAt(*c, x, y);
        if (!k || !enabled(*k->spec)) return;
        const auto& s = *k->spec;
        if (k->kind == Kind::Bar) {
            drag = {};
            drag.mode = Drag::Bar;
            drag.spec = &s;
            drag.startX = x;
            drag.width = std::max(1.f, k->rect.right - k->rect.left);
            drag.startValue = value(s);
            SetCapture(window);
        } else if (k->kind == Kind::Toggle) {
            writeScalar(s, value(s) != 0 ? 0 : 1);
        } else if (k->kind == Kind::Seg && !k->asDrop) {
            const size_t n = s.options.size();
            const float x0 = k->rect.left + k->keyW;
            if (x < x0) return;
            const float w = (k->rect.right - x0 - 6) / n;
            const int idx = std::clamp(static_cast<int>((x - x0 - 3) / w), 0, static_cast<int>(n) - 1);
            writeScalar(s, idx);
        } else if (k->kind == Kind::Drop || k->kind == Kind::Seg) {
            openDrop(*k);
        } else {
            for (int i = 0; i < 2; ++i) if (inside(stepRect(k->rect, i), x, y)) writeScalar(s, value(s) + (i ? 1 : -1));
        }
    }

    void mouseMove(float x, float y, bool shift, bool ctrl) {
        hoverX = x;
        hoverY = y;
        if (drop.open) {
            const int idx = inside(drop.rect, x, y) ? drop.first + static_cast<int>((y - drop.rect.top - 4) / drop.itemH) : -1;
            if (idx != drop.hover) { drop.hover = idx; InvalidateRect(window, nullptr, FALSE); }
            return;
        }
        switch (drag.mode) {
        case Drag::Bar: {
            const auto& s = *drag.spec;
            const double span = s.dmax - s.dmin;
            double v = drag.startValue + (x - drag.startX) / drag.width * span * (shift ? 0.1 : 1.0);
            if (ctrl && span > 0) { const double step = span / 100; v = std::round(v / step) * step; }
            writeScalar(s, v);
            break;
        }
        case Drag::Dial: dialTo(*drag.card, x, y); break;
        case Drag::Pad: padTo(*drag.card, x, y); break;
        case Drag::Wheel: wheelTo(*drag.card, x, y); break;
        case Drag::RangeLo: rangeTo(*drag.card, x, false); break;
        case Drag::RangeHi: rangeTo(*drag.card, x, true); break;
        case Drag::Scroll: {
            const auto sz = size();
            const auto t = thumbRect(sz);
            const float track = sz.width - 2 * sidePad - (t.right - t.left);
            if (track > 0) scrollBy(drag.startScroll + (x - drag.startX) / track * (contentW - sz.width) - scrollX);
            break;
        }
        default: {
            const ParamSpec* hovered = nullptr;
            layout();
            if (const Card* c = cardAt(x, y)) if (const Ctl* k = ctlAt(*c, x, y)) hovered = k->spec;
            if (hovered != tipSpec) { tipSpec = hovered; tipSince = now(); }
            InvalidateRect(window, nullptr, FALSE);
            break;
        }
        }
    }

    void mouseUp() {
        if (drag.mode != Drag::None) ReleaseCapture();
        drag = {};
    }

    void doubleClick(float x, float y) {
        layout();
        const Card* c = cardAt(x, y);
        if (!c) return;
        const Ctl* k = ctlAt(*c, x, y);
        if (k && (k->kind == Kind::Bar || k->kind == Kind::Field)) openEditor(*k);
    }

    void rightClick(float x, float y) {
        layout();
        const Card* c = cardAt(x, y);
        if (!c) return;
        if (c->widget != Widget::None && inside(c->wrect, x, y)) {
            reset(*c->wa);
            if (c->wb) reset(*c->wb);
            return;
        }
        if (const Ctl* k = ctlAt(*c, x, y)) reset(*k->spec);
    }

    static LRESULT CALLBACK proc(HWND hwnd, UINT message, WPARAM wp, LPARAM lp) {
        if (message == WM_NCCREATE) {
            auto* self = static_cast<LivePanel*>(reinterpret_cast<CREATESTRUCTW*>(lp)->lpCreateParams);
            self->window = hwnd;
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
        }
        auto* self = reinterpret_cast<LivePanel*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        if (!self) return DefWindowProcW(hwnd, message, wp, lp);
        const auto mx = [&] { return self->dip(GET_X_LPARAM(lp)); };
        const auto my = [&] { return self->dip(GET_Y_LPARAM(lp)); };
        switch (message) {
        case WM_CREATE: {
            self->dpi = GetDpiForWindow(hwnd);
            if (!self->dpi) self->dpi = 96;
            const BOOL dark = TRUE;
            DwmSetWindowAttribute(hwnd, 20, &dark, sizeof(dark));
            const COLORREF caption = RGB(20, 22, 24);
            DwmSetWindowAttribute(hwnd, 35, &caption, sizeof(caption));
            SetTimer(hwnd, refreshTimer, 125, nullptr);
            return 0;
        }
        case WM_CLOSE: self->savePlacement(); ShowWindow(hwnd, SW_HIDE); return 0;
        case WM_EXITSIZEMOVE: self->savePlacement(); return 0;
        case WM_SIZE:
            if (self->rt) self->rt->Resize(D2D1::SizeU(LOWORD(lp), HIWORD(lp)));
            self->drop.open = false;
            InvalidateRect(hwnd, nullptr, FALSE);
            return 0;
        case WM_GETMINMAXINFO: {
            auto& m = *reinterpret_cast<MINMAXINFO*>(lp);
            m.ptMinTrackSize = {self->px(900), self->px(180)};
            return 0;
        }
        case WM_DPICHANGED: {
            self->dpi = HIWORD(wp);
            self->rt.Reset();
            const auto& r = *reinterpret_cast<RECT*>(lp);
            SetWindowPos(hwnd, nullptr, r.left, r.top, r.right - r.left, r.bottom - r.top, SWP_NOZORDER | SWP_NOACTIVATE);
            return 0;
        }
        case WM_ERASEBKGND: return 1;
        case WM_PAINT: {
            PAINTSTRUCT ps;
            BeginPaint(hwnd, &ps);
            self->paint();
            EndPaint(hwnd, &ps);
            return 0;
        }
        case WM_TIMER:
            if (wp == refreshTimer && IsWindowVisible(hwnd) && self->drag.mode == Drag::None) {
                self->refresh(false);
                if (self->tipSpec && !self->tipShown && now() - self->tipSince >= 0.5) InvalidateRect(hwnd, nullptr, FALSE);
            }
            if (wp == animTimer) self->tick();
            return 0;
        case WM_LBUTTONDOWN: self->mouseDown(mx(), my(), wp & MK_SHIFT, wp & MK_CONTROL); return 0;
        case WM_LBUTTONDBLCLK: self->doubleClick(mx(), my()); return 0;
        case WM_MOUSEMOVE: {
            TRACKMOUSEEVENT t{sizeof(t), TME_LEAVE, hwnd, 0};
            TrackMouseEvent(&t);
            self->mouseMove(mx(), my(), wp & MK_SHIFT, wp & MK_CONTROL);
            return 0;
        }
        case WM_MOUSELEAVE: self->hoverX = self->hoverY = -1; self->tipSpec = nullptr; InvalidateRect(hwnd, nullptr, FALSE); return 0;
        case WM_LBUTTONUP: self->mouseUp(); return 0;
        case WM_CAPTURECHANGED: self->drag = {}; return 0;
        case WM_RBUTTONUP: self->rightClick(mx(), my()); return 0;
        case WM_MOUSEWHEEL: case WM_MOUSEHWHEEL: {
            const float d = GET_WHEEL_DELTA_WPARAM(wp) / 120.f * 120.f;
            if (self->drop.open) {
                const int max = std::max(0, static_cast<int>(self->drop.spec->options.size()) - self->drop.visible);
                self->drop.first = std::clamp(self->drop.first - static_cast<int>(GET_WHEEL_DELTA_WPARAM(wp) / 40), 0, max);
                InvalidateRect(hwnd, nullptr, FALSE);
            } else self->step((message == WM_MOUSEHWHEEL ? d : -d) > 0 ? 1 : -1);
            return 0;
        }
        case WM_KEYDOWN:
            if (wp == VK_ESCAPE && self->drop.open) { self->drop.open = false; InvalidateRect(hwnd, nullptr, FALSE); }
            return 0;
        case WM_CTLCOLOREDIT: {
            const auto dc = reinterpret_cast<HDC>(wp);
            SetTextColor(dc, RGB(232, 228, 220));
            SetBkColor(dc, RGB(46, 52, 57));
            return reinterpret_cast<LRESULT>(self->editBrush);
        }
        case WM_DESTROY:
            KillTimer(hwnd, refreshTimer);
            KillTimer(hwnd, animTimer);
            self->rt.Reset();
            return 0;
        }
        return DefWindowProcW(hwnd, message, wp, lp);
    }
};

}

void togglePanel(const std::shared_ptr<PanelBinding>& binding) {
    static LivePanel panel;
    panel.toggle(binding);
}

namespace {
struct HostRefresher {
    HANDLE process = nullptr, input = nullptr;
    bool failed = false;
    ~HostRefresher() { stop(); }
    void stop() {
        if (input) CloseHandle(input);
        if (process) CloseHandle(process);
        input = process = nullptr;
    }
    bool start() {
        HMODULE module = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                reinterpret_cast<LPCWSTR>(&requestHostRefresh), &module)) return false;
        wchar_t path[32768];
        if (!GetModuleFileNameW(module, path, 32768)) return false;
        const auto script = std::filesystem::path(path).parent_path().parent_path() / L"Resources" / L"cutwright" / L"resolve_refresh.py";
        if (!std::filesystem::exists(script)) return false;
        const wchar_t* custom = _wgetenv(L"TEMPORAL_FX_PYTHON");
        const std::filesystem::path python = custom ? std::filesystem::path(custom) : std::filesystem::path(u8"" TP_PYTHON);
        if (!_wgetenv(L"RESOLVE_SCRIPT_LIB") && GetModuleFileNameW(nullptr, path, 32768)) {
            const auto library = std::filesystem::path(path).parent_path() / L"fusionscript.dll";
            if (std::filesystem::exists(library)) SetEnvironmentVariableW(L"RESOLVE_SCRIPT_LIB", library.c_str());
        }
        SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
        HANDLE read = nullptr, write = nullptr;
        if (!CreatePipe(&read, &write, &sa, 0)) return false;
        SetHandleInformation(write, HANDLE_FLAG_INHERIT, 0);
        STARTUPINFOW si{};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = read;
        PROCESS_INFORMATION pi{};
        std::wstring cmd = L"\"" + python.wstring() + L"\" -E -s -B \"" + script.wstring() + L"\"";
        const BOOL ok = CreateProcessW(python.c_str(), cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr,
                                       script.parent_path().c_str(), &si, &pi);
        CloseHandle(read);
        if (!ok) { CloseHandle(write); return false; }
        CloseHandle(pi.hThread);
        process = pi.hProcess;
        input = write;
        return true;
    }
    void poke() {
        if (process && WaitForSingleObject(process, 0) != WAIT_TIMEOUT) stop();
        if (!process) {
            if (failed) return;
            if (!start()) { failed = true; return; }
        }
        DWORD written = 0;
        if (!WriteFile(input, "\n", 1, &written, nullptr)) stop();
    }
};
}

std::atomic<double> gLastPoke{-1e9};

double steadyNow() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

void requestHostRefresh() {
    static std::mutex lock;
    static HostRefresher refresher;
    std::lock_guard guard(lock);
    gLastPoke = steadyNow();
    refresher.poke();
}

bool hostRefreshRecent() { return steadyNow() - gLastPoke.load() < 1.5; }

}
#else
namespace tp {
void togglePanel(const std::shared_ptr<PanelBinding>&) {}
void requestHostRefresh() {}
bool hostRefreshRecent() { return false; }
}
#endif
