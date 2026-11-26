#pragma once
#include "../core/module.h"
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace tp {

inline constexpr const char* kPanelButton = "temporal_open_panel";

struct PanelBinding {
    const ModuleDef* module = nullptr;
    std::recursive_mutex mutex;
    std::function<ParamValues()> read;
    std::function<bool(const ParamSpec&)> visible;
    std::function<bool(const ParamSpec&)> enabled;
    std::function<void(const ParamSpec&, const std::array<double, 3>&)> write;
    std::function<double()> time;
    std::function<std::vector<double>(const ParamSpec&)> keys;
    std::function<void(const ParamSpec&, bool)> keyframe;
};

void togglePanel(const std::shared_ptr<PanelBinding>& binding);
void requestHostRefresh();
bool hostRefreshRecent();

}
