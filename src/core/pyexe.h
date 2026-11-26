#pragma once
#include <cstdlib>
#include <filesystem>
#include <string>

namespace tp {

inline std::filesystem::path findPython() {
    namespace fs = std::filesystem;
    if (const wchar_t* custom = _wgetenv(L"TEMPORAL_FX_PYTHON")) return custom;
    const wchar_t* env = _wgetenv(L"PATH");
    const std::wstring list = env ? env : L"";
    for (size_t from = 0; from <= list.size();) {
        size_t to = list.find(L';', from);
        if (to == std::wstring::npos) to = list.size();
        const std::wstring dir = list.substr(from, to - from);
        std::error_code ec;
        if (!dir.empty() && dir.find(L"WindowsApps") == std::wstring::npos && fs::is_regular_file(fs::path(dir) / L"python.exe", ec)) return fs::path(dir) / L"python.exe";
        from = to + 1;
    }
    return {};
}

}
