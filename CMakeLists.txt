cmake_minimum_required(VERSION 3.24)
project(TEMPORAL VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set(OFX_ROOT ${CMAKE_SOURCE_DIR}/third_party/openfx)

if(MSVC)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8;/W3;/permissive-;/EHsc;/O2;/fp:precise;/arch:AVX2;/wd4244;/wd4267;/wd4305>")
  add_compile_definitions(NOMINMAX _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
endif()

set(TP_ONLY_MODULES "" CACHE STRING "Semicolon list of module ids to build (empty = all)")
if(TP_ONLY_MODULES)
  set(TP_MODULE_SOURCES "")
  foreach(m ${TP_ONLY_MODULES})
    list(APPEND TP_MODULE_SOURCES ${CMAKE_SOURCE_DIR}/src/modules/${m}.cpp)
  endforeach()
else()
  file(GLOB TP_MODULE_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/modules/*.cpp)
endif()
option(TP_WITH_CUDA "Build CUDA kernels" ON)
include(CheckLanguage)
check_language(CUDA)
if(TP_WITH_CUDA AND CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 86)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
  file(GLOB TP_GPU_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/gpu/*.cu)
  # Match the CPU sample positions and interpolation rounding near high-contrast log highlights.
  set_source_files_properties(src/gpu/motion_tear.cu src/gpu/flow.cu src/gpu/cutwright_blur.cu PROPERTIES
    COMPILE_OPTIONS "--fmad=false;--prec-div=true;--prec-sqrt=true")
  message(STATUS "TEMPORAL: CUDA kernels enabled")
else()
  set(TP_GPU_SOURCES ${CMAKE_SOURCE_DIR}/src/gpu/gpu_stub.cpp)
  message(STATUS "TEMPORAL: CUDA disabled, CPU only")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_custom_target(tp_cutwright_resources ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/resources/cutwright"
          "${CMAKE_BINARY_DIR}/TEMPORAL.ofx.bundle/Contents/Resources/cutwright")

add_library(tp_core STATIC
  src/core/framework.cpp
  src/core/registry.cpp
  ${TP_MODULE_SOURCES}
  ${TP_GPU_SOURCES})
set(TP_PREVIEW_PYTHON "" CACHE FILEPATH "Optional Python with CUDA PyTorch for interactive Cutwright renders")
target_compile_definitions(tp_core PRIVATE TP_PREVIEW_PYTHON="${TP_PREVIEW_PYTHON}")
add_dependencies(tp_core tp_cutwright_resources)
target_include_directories(tp_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
if(TP_WITH_CUDA AND CMAKE_CUDA_COMPILER)
  set_target_properties(tp_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)
  target_compile_options(tp_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
endif()

file(GLOB OFX_SUPPORT_SOURCES ${OFX_ROOT}/Support/Library/*.cpp)
add_library(TEMPORAL MODULE src/ofx/adapter.cpp src/ofx/panel.cpp ${OFX_SUPPORT_SOURCES})
target_include_directories(TEMPORAL PRIVATE ${OFX_ROOT}/include ${OFX_ROOT}/Support/include ${OFX_ROOT}/Support/Library)
target_link_libraries(TEMPORAL PRIVATE tp_core)
set(TP_BUNDLE ${CMAKE_BINARY_DIR}/TEMPORAL.ofx.bundle)
set_target_properties(TEMPORAL PROPERTIES
  PREFIX ""
  SUFFIX ".ofx"
  LIBRARY_OUTPUT_DIRECTORY ${TP_BUNDLE}/Contents/Win64
  LIBRARY_OUTPUT_DIRECTORY_RELEASE ${TP_BUNDLE}/Contents/Win64
  RUNTIME_OUTPUT_DIRECTORY ${TP_BUNDLE}/Contents/Win64)

add_library(TEMPORAL_Optic MODULE EXCLUDE_FROM_ALL src/ofx/adapter.cpp src/ofx/panel.cpp ${OFX_SUPPORT_SOURCES})
target_compile_definitions(TEMPORAL_Optic PRIVATE TP_OPTIC_STUDIO)
target_include_directories(TEMPORAL_Optic PRIVATE ${OFX_ROOT}/include ${OFX_ROOT}/Support/include ${OFX_ROOT}/Support/Library)
target_link_libraries(TEMPORAL_Optic PRIVATE tp_core)
add_custom_command(TARGET TEMPORAL_Optic POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/resources/cutwright/resolve_refresh.py"
          "${CMAKE_BINARY_DIR}/TEMPORAL-Optic.ofx.bundle/Contents/Resources/cutwright/resolve_refresh.py")
set(TP_ICON_IDS multi_optic_studio)
foreach(src ${TP_MODULE_SOURCES})
  file(STRINGS ${src} id_lines REGEX "(m\\.id = |makeFamily\\([0-9]+, *)\"[a-z_0-9]+\"")
  foreach(line ${id_lines})
    string(REGEX MATCH "\"([a-z_0-9]+)\"" _ "${line}")
    list(APPEND TP_ICON_IDS ${CMAKE_MATCH_1})
  endforeach()
endforeach()
list(REMOVE_DUPLICATES TP_ICON_IDS)
foreach(target_bundle TEMPORAL:TEMPORAL TEMPORAL_Optic:TEMPORAL-Optic)
  string(REPLACE ":" ";" pair ${target_bundle})
  list(GET pair 0 tgt)
  list(GET pair 1 bundle)
  set(res "${CMAKE_BINARY_DIR}/${bundle}.ofx.bundle/Contents/Resources")
  set(cmds COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/assets/brand/temporal.ico" "${res}/temporal.ico")
  foreach(id ${TP_ICON_IDS})
    list(APPEND cmds COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/assets/brand/ofx-icon.png" "${res}/com.temporal.${id}.png")
  endforeach()
  add_custom_command(TARGET ${tgt} POST_BUILD ${cmds})
endforeach()
if(WIN32)
  target_link_libraries(TEMPORAL PRIVATE comctl32 dwmapi d2d1 dwrite)
  target_link_libraries(TEMPORAL_Optic PRIVATE comctl32 dwmapi d2d1 dwrite)
endif()
set(TP_OPTIC_BUNDLE ${CMAKE_BINARY_DIR}/TEMPORAL-Optic.ofx.bundle)
set_target_properties(TEMPORAL_Optic PROPERTIES PREFIX "" SUFFIX ".ofx" OUTPUT_NAME "TEMPORAL-Optic"
  LIBRARY_OUTPUT_DIRECTORY ${TP_OPTIC_BUNDLE}/Contents/Win64
  LIBRARY_OUTPUT_DIRECTORY_RELEASE ${TP_OPTIC_BUNDLE}/Contents/Win64
  RUNTIME_OUTPUT_DIRECTORY ${TP_OPTIC_BUNDLE}/Contents/Win64)

enable_testing()
if(TP_ONLY_MODULES)
  set(TP_MODULE_TESTS "")
  foreach(m ${TP_ONLY_MODULES})
    if(EXISTS ${CMAKE_SOURCE_DIR}/tests/module_${m}.cpp)
      list(APPEND TP_MODULE_TESTS ${CMAKE_SOURCE_DIR}/tests/module_${m}.cpp)
    endif()
  endforeach()
else()
  file(GLOB TP_MODULE_TESTS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/module_*.cpp)
endif()
add_executable(tp_tests tests/tests.cpp ${TP_MODULE_TESTS})
target_include_directories(tp_tests PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(tp_tests PRIVATE tp_core)
if(TP_WITH_CUDA AND CMAKE_CUDA_COMPILER)
  target_sources(tp_tests PRIVATE tests/native_cuda.cu)
endif()
add_test(NAME tp_tests COMMAND tp_tests)

add_executable(tp_bench tests/bench.cpp)
target_link_libraries(tp_bench PRIVATE tp_core)
target_include_directories(tp_bench PRIVATE ${CMAKE_SOURCE_DIR}/tests)

add_executable(tp_preview_bench EXCLUDE_FROM_ALL tests/preview_bench.cpp)
target_link_libraries(tp_preview_bench PRIVATE tp_core)

add_executable(tp_docgen tests/docgen.cpp)
target_link_libraries(tp_docgen PRIVATE tp_core)
