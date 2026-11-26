#include "harness.h"
#include "core/pyexe.h"
#include <fstream>

namespace fs = std::filesystem;

TP_TEST(find_python_prefers_override_then_path_and_skips_store_stub) {
    const fs::path root = fs::temp_directory_path() / L"tp_findpython_test";
    fs::remove_all(root);
    fs::create_directories(root / L"empty");
    fs::create_directories(root / L"real");
    fs::create_directories(root / L"WindowsApps");
    std::ofstream(root / L"real" / L"python.exe").put('x');
    std::ofstream(root / L"WindowsApps" / L"python.exe").put('x');

    const std::wstring savedPath = _wgetenv(L"PATH") ? _wgetenv(L"PATH") : L"";
    const std::wstring savedOverride = _wgetenv(L"TEMPORAL_FX_PYTHON") ? _wgetenv(L"TEMPORAL_FX_PYTHON") : L"";
    _wputenv_s(L"TEMPORAL_FX_PYTHON", L"");

    _wputenv_s(L"PATH", (root / L"empty").c_str());
    CHECK(tp::findPython().empty());

    _wputenv_s(L"PATH", (root / L"WindowsApps").c_str());
    CHECK(tp::findPython().empty());

    _wputenv_s(L"PATH", ((root / L"empty").wstring() + L";" + (root / L"WindowsApps").wstring() + L";" + (root / L"real").wstring()).c_str());
    CHECK(tp::findPython() == root / L"real" / L"python.exe");

    _wputenv_s(L"TEMPORAL_FX_PYTHON", L"X:\\custom\\python.exe");
    CHECK(tp::findPython() == fs::path(L"X:\\custom\\python.exe"));

    _wputenv_s(L"PATH", savedPath.c_str());
    _wputenv_s(L"TEMPORAL_FX_PYTHON", savedOverride.c_str());
    fs::remove_all(root);
}
