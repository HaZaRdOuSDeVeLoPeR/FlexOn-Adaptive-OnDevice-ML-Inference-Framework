cmake_minimum_required(VERSION 3.20)

project(FlexOn VERSION 0.1.0 LANGUAGES CXX CUDA)

# ============================================================================
# Configuration
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

option(FLEXON_BUILD_TESTS "Build FlexOn tests" ON)
option(FLEXON_BUILD_APPS "Build FlexOn applications" ON)
option(FLEXON_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(FLEXON_ENABLE_WARNINGS "Enable strict compiler warnings" ON)

if(FLEXON_ENABLE_ASAN)
    add_compile_options(
        -fsanitize=address
        -fno-omit-frame-pointer
        -g
    )

    add_link_options(
        -fsanitize=address
    )
endif()

if(FLEXON_ENABLE_WARNINGS)
    add_compile_options(
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall>
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wextra>
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wpedantic>
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wunused-variable>
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wunused-function>
    )

    # Optional: Apply specific flags for CUDA files handled by NVCC
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    )
endif()

# Milestone 2 uses the CUDA runtime for explicit CPU<->CUDA boundary copies.
# ONNX Runtime remains responsible for executing CUDA segment kernels.
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# NVML provides the measured GPU utilization used by the closed-loop
# contention controller. WSL2 installations normally expose the driver
# library under /usr/lib/wsl/lib.
find_path(FLEXON_NVML_INCLUDE_DIR
    NAMES nvml.h
    PATHS
        /usr/include
        /usr/local/cuda/include
        /usr/include/nvidia
        /usr/include/nvidia-ml
)

find_library(FLEXON_NVML_LIBRARY
    NAMES nvidia-ml nvidia-ml.so.1
    PATHS
        /usr/lib/wsl/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib64
        /usr/local/cuda/lib64
)

if(NOT FLEXON_NVML_INCLUDE_DIR OR NOT FLEXON_NVML_LIBRARY)
    message(FATAL_ERROR
        "NVML headers/library not found. GPU contention monitoring requires "
        "nvml.h and libnvidia-ml.")
endif()

# User-facing applications are generated into the project apps/ directory.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_SOURCE_DIR}/apps"
)

# Test executable directory
set(FLEXON_TEST_RUNTIME_OUTPUT_DIR
    "${CMAKE_SOURCE_DIR}/tests"
)

# ============================================================================
# Project directories
# ============================================================================

set(FLEXON_INCLUDE_DIR
    "${CMAKE_SOURCE_DIR}/include"
)

set(FLEXON_SRC_DIR
    "${CMAKE_SOURCE_DIR}/src"
)

# ============================================================================
# yaml-cpp
# ============================================================================

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/yaml-cpp/CMakeLists.txt")
    message(FATAL_ERROR
        "yaml-cpp not found in third_party/yaml-cpp. Run setup.sh first."
    )
endif()

add_subdirectory(
    "${CMAKE_SOURCE_DIR}/third_party/yaml-cpp"
    "${CMAKE_BINARY_DIR}/yaml-cpp"
    EXCLUDE_FROM_ALL
)

# ============================================================================
# ONNX Runtime
# ============================================================================

set(FLEXON_ORT_ROOT
    "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-src"
)

set(FLEXON_ORT_BUILD_DIR
    "${FLEXON_ORT_ROOT}/build/Linux/Release"
)

set(FLEXON_ORT_INCLUDE_DIR
    "${FLEXON_ORT_ROOT}/include"
)

set(FLEXON_ORT_SESSION_INCLUDE_DIR
    "${FLEXON_ORT_ROOT}/include/onnxruntime/core/session"
)

find_library(
    FLEXON_ORT_LIBRARY
    NAMES onnxruntime
    PATHS "${FLEXON_ORT_BUILD_DIR}"
    NO_DEFAULT_PATH
)

if(NOT FLEXON_ORT_LIBRARY)
    message(FATAL_ERROR
        "ONNX Runtime library not found in:\n"
        "  ${FLEXON_ORT_BUILD_DIR}\n"
        "Run setup.sh first."
    )
endif()

if(NOT EXISTS "${FLEXON_ORT_INCLUDE_DIR}")
    message(FATAL_ERROR
        "ONNX Runtime headers not found in:\n"
        "  ${FLEXON_ORT_INCLUDE_DIR}\n"
        "Run setup.sh first."
    )
endif()

add_library(
    flexon_onnxruntime
    SHARED
    IMPORTED
)

set_target_properties(
    flexon_onnxruntime
    PROPERTIES
        IMPORTED_LOCATION
            "${FLEXON_ORT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES
            "${FLEXON_ORT_INCLUDE_DIR};${FLEXON_ORT_SESSION_INCLUDE_DIR}"
)

# ============================================================================
# ONNX
# ============================================================================

set(FLEXON_ONNX_BUILD_DIR
    "${FLEXON_ORT_BUILD_DIR}/_deps/onnx-build"
)

set(FLEXON_ONNX_SOURCE_DIR
    "${FLEXON_ORT_BUILD_DIR}/_deps/onnx-src"
)

set(FLEXON_ONNX_LIBRARY
    "${FLEXON_ONNX_BUILD_DIR}/libonnx.a"
)

set(FLEXON_ONNX_PROTO_LIBRARY
    "${FLEXON_ONNX_BUILD_DIR}/libonnx_proto.a"
)

set(FLEXON_PROTOBUF_BUILD_DIR
    "${FLEXON_ORT_BUILD_DIR}/_deps/protobuf-build"
)

set(FLEXON_PROTOBUF_LIBRARY
    "${FLEXON_PROTOBUF_BUILD_DIR}/libprotobuf-lite.a"
)

if(NOT EXISTS "${FLEXON_ONNX_LIBRARY}")
    message(FATAL_ERROR
        "ONNX library not found:\n"
        "  ${FLEXON_ONNX_LIBRARY}\n"
        "Run setup.sh first."
    )
endif()

if(NOT EXISTS "${FLEXON_ONNX_PROTO_LIBRARY}")
    message(FATAL_ERROR
        "ONNX protobuf library not found:\n"
        "  ${FLEXON_ONNX_PROTO_LIBRARY}\n"
        "Run setup.sh first."
    )
endif()

if(NOT EXISTS "${FLEXON_PROTOBUF_LIBRARY}")
    message(FATAL_ERROR
        "Protobuf lite library not found:\n"
        "  ${FLEXON_PROTOBUF_LIBRARY}\n"
        "Run setup.sh first."
    )
endif()

# ----------------------------------------------------------------------------
# ONNX protobuf
# ----------------------------------------------------------------------------

add_library(
    flexon_onnx_proto
    STATIC
    IMPORTED
)

set_target_properties(
    flexon_onnx_proto
    PROPERTIES
        IMPORTED_LOCATION
            "${FLEXON_ONNX_PROTO_LIBRARY}"

        INTERFACE_INCLUDE_DIRECTORIES
            "${FLEXON_ONNX_BUILD_DIR}"

        INTERFACE_COMPILE_DEFINITIONS
            "ONNX_NAMESPACE=onnx;ONNX_ML=1;ONNX_USE_LITE_PROTO=1"
)

# ----------------------------------------------------------------------------
# Protobuf
# ----------------------------------------------------------------------------

add_library(
    flexon_protobuf
    STATIC
    IMPORTED
)

set_target_properties(
    flexon_protobuf
    PROPERTIES
        IMPORTED_LOCATION
            "${FLEXON_PROTOBUF_LIBRARY}"

        INTERFACE_INCLUDE_DIRECTORIES
            "${FLEXON_ORT_BUILD_DIR}/_deps/protobuf-src/src"
)

# ----------------------------------------------------------------------------
# ONNX
# ----------------------------------------------------------------------------

add_library(
    flexon_onnx
    STATIC
    IMPORTED
)

set_target_properties(
    flexon_onnx
    PROPERTIES
        IMPORTED_LOCATION
            "${FLEXON_ONNX_LIBRARY}"

        INTERFACE_INCLUDE_DIRECTORIES
            "${FLEXON_ONNX_SOURCE_DIR};${FLEXON_ONNX_BUILD_DIR}"

        INTERFACE_COMPILE_DEFINITIONS
            "ONNX_NAMESPACE=onnx;ONNX_ML=1;ONNX_USE_LITE_PROTO=1;__ONNX_NO_DOC_STRINGS"
)

target_link_libraries(
    flexon_onnx
    INTERFACE
        flexon_onnx_proto
        flexon_protobuf
)

# ============================================================================
# Core
# ============================================================================

add_library(
    flexon_core
    INTERFACE
)

target_include_directories(
    flexon_core
    INTERFACE
        "${FLEXON_INCLUDE_DIR}"
)

# ============================================================================
# Offline library
# ============================================================================

add_library(
    flexon_offline_lib
    STATIC

    src/offline/model/onnx_model_loader.cpp
    src/offline/graph/graph_analyzer.cpp
    src/offline/shapes/runtime_shape_resolver.cpp
    src/offline/config/offline_config.cpp
    src/offline/config/model_config.cpp
    src/offline/segmentation/initial_segmenter.cpp
    src/offline/segmentation/multi_level_segmenter.cpp
    src/offline/model/segment_model_generator.cpp
    src/offline/profiling/segment_profiler.cpp
    src/offline/artifact/artifact_writer.cpp
    src/offline/artifact/artifact_validator.cpp
    src/offline/offline_helper.cpp
    src/offline/offline_engine.cpp
)

target_include_directories(
    flexon_offline_lib
    PUBLIC
        "${FLEXON_INCLUDE_DIR}"
)

target_link_libraries(
    flexon_offline_lib
    PUBLIC
        flexon_core
        flexon_onnx
        yaml-cpp
        flexon_onnxruntime
)

# ============================================================================
# Contention library
# ============================================================================

add_library(
    flexon_contention_lib
    STATIC

    src/contention/contention_generator.cpp
    src/contention/monitor.cpp
    src/contention/cpu_stressor.cpp
    src/contention/bandwidth_stressor.cpp
    src/contention/gpu_stressor.cu
)

target_include_directories(
    flexon_contention_lib
    PUBLIC
        "${FLEXON_INCLUDE_DIR}"
        "${FLEXON_NVML_INCLUDE_DIR}"
    INTERFACE
        "${FLEXON_INCLUDE_DIR}"
)

set_target_properties(
    flexon_contention_lib
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
)

# Suppress benign host-static library warnings inside nvlink
target_compile_options(
    flexon_contention_lib
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xnvlink=-w>
)

target_link_libraries(
    flexon_contention_lib
    PRIVATE
        $<$<LINK_LANGUAGE:CXX>:Threads::Threads>
)

target_link_libraries(
    flexon_contention_lib
    PUBLIC
        flexon_core
        CUDA::cudart
        "${FLEXON_NVML_LIBRARY}"
)

# ============================================================================
# Online library
# ============================================================================

# Online library
add_library(
    flexon_online_lib
    STATIC

    src/online/artifact/artifact_handler.cpp
    src/online/execution/buffer_arena.cpp
    src/online/execution/executor.cpp
    src/online/execution/helper.cpp
    src/online/resource/resource_monitor.cpp
    src/online/resource/resource_runtime.cpp
    src/online/scheduler/recovery_manager.cpp
    src/online/scheduler/scheduler.cpp
    src/online/scheduler/thread_priority.cpp
    src/online/runtime_helper.cpp
    src/online/runtime_engine.cpp
)

target_include_directories(
    flexon_online_lib
    PUBLIC
        "${FLEXON_INCLUDE_DIR}"
)

target_link_libraries(
    flexon_online_lib
    PUBLIC
        flexon_core
        flexon_onnxruntime
        yaml-cpp
        CUDA::cudart
)

# ============================================================================
# Applications
# ============================================================================

if(FLEXON_BUILD_APPS)

    # Offline phase: generation and validation are exposed through one CLI.
    add_executable(
        flexon_offline
        src/flexon_offline.cpp
        src/offline/offline_helper.cpp
    )

    target_link_libraries(
        flexon_offline
        PRIVATE
            flexon_core
            flexon_offline_lib
            flexon_contention_lib
    )

    # Online phase: runtime execution and validation behavior are exposed
    # through one experiment-friendly CLI.
    add_executable(
        flexon_online
        src/flexon_online.cpp
        src/online/runtime_helper.cpp
    )

    target_link_libraries(
        flexon_online
        PRIVATE
            flexon_core
            flexon_online_lib
    )

    # Standalone closed-loop contention generator. Targets are measured
    # resource load percentages rather than raw stressor duty cycles.
    add_executable(
        flexon_contention
        src/flexon_contention.cpp
    )

    target_link_libraries(
        flexon_contention
        PRIVATE
            flexon_contention_lib
    )

    # Tiny privileged helper used only to promote the already-running
    # scheduler/control thread. CAP_SYS_NICE is installed on this helper,
    # never on flexon_online, so CUDA/WSL initialization remains untouched.
    add_executable(
        flexon_scheduler_helper
        src/online/scheduler_helper.cpp
    )

endif()

# ============================================================================
# Tests
# ============================================================================

if(FLEXON_BUILD_TESTS)

    enable_testing()

    function(flexon_add_test target source)
        add_executable(${target} ${source})
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${FLEXON_TEST_RUNTIME_OUTPUT_DIR}"
        )
        add_test(NAME ${target} COMMAND ${target})
    endfunction()

    # ------------------------------------------------------------------------
    # Offline tests
    # ------------------------------------------------------------------------

    flexon_add_test(
        flexon_onnx_model_loader_test
        src/tests/offline/onnx_model_loader_test.cpp
    )
    target_link_libraries(flexon_onnx_model_loader_test PRIVATE flexon_offline_lib)

    flexon_add_test(
        flexon_graph_analyzer_test
        src/tests/offline/graph_analyzer_test.cpp
    )
    target_link_libraries(flexon_graph_analyzer_test PRIVATE flexon_offline_lib)

    flexon_add_test(
        flexon_segment_model_generator_test
        src/tests/offline/segment_model_generator_test.cpp
    )
    target_link_libraries(flexon_segment_model_generator_test PRIVATE flexon_offline_lib)

    flexon_add_test(
        flexon_runtime_shape_resolver_test
        src/tests/offline/runtime_shape_resolver_test.cpp
    )
    target_link_libraries(flexon_runtime_shape_resolver_test PRIVATE flexon_offline_lib)

    flexon_add_test(
        flexon_model_config_test
        src/tests/offline/model_config_test.cpp
    )
    target_link_libraries(flexon_model_config_test PRIVATE flexon_offline_lib)

    # ------------------------------------------------------------------------
    # Online tests
    # ------------------------------------------------------------------------

    flexon_add_test(
        flexon_scheduler_test
        src/tests/online/scheduler_test.cpp
    )
    target_link_libraries(flexon_scheduler_test PRIVATE flexon_online_lib)

    flexon_add_test(
        flexon_resource_runtime_test
        src/tests/online/resource_runtime_test.cpp
    )
    target_link_libraries(flexon_resource_runtime_test PRIVATE flexon_online_lib)

    flexon_add_test(
        flexon_buffer_arena_test
        src/tests/online/buffer_arena_test.cpp
    )
    target_link_libraries(flexon_buffer_arena_test PRIVATE flexon_online_lib)

    flexon_add_test(
        flexon_artifact_handler_test
        src/tests/online/artifact_handler_test.cpp
    )
    target_link_libraries(flexon_artifact_handler_test PRIVATE flexon_online_lib)

    flexon_add_test(
        flexon_runtime_helper_test
        src/tests/online/runtime_helper_test.cpp
    )
    target_link_libraries(flexon_runtime_helper_test PRIVATE flexon_online_lib)

    flexon_add_test(
        flexon_resource_monitor_test
        src/tests/online/resource_monitor_test.cpp
    )
    target_link_libraries(flexon_resource_monitor_test PRIVATE flexon_online_lib)

    flexon_add_test(
        flexon_runtime_engine_test
        src/tests/online/runtime_engine_test.cpp
    )
    target_link_libraries(flexon_runtime_engine_test PRIVATE flexon_online_lib)

    # ------------------------------------------------------------------------
    # General smoke test
    # ------------------------------------------------------------------------

    flexon_add_test(
        flexon_smoke_test
        src/tests/smoke_test.cpp
    )
    target_link_libraries(flexon_smoke_test PRIVATE flexon_core)

endif()

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "========== FlexOn ${PROJECT_VERSION} ==========")
message(STATUS "C++ standard:       C++${CMAKE_CXX_STANDARD}")
message(STATUS "Build tests:        ${FLEXON_BUILD_TESTS}")
message(STATUS "Build applications: ${FLEXON_BUILD_APPS}")
message(STATUS "Compiler warnings:  ${FLEXON_ENABLE_WARNINGS}")
message(STATUS "")
message(STATUS "Include directory:  ${FLEXON_INCLUDE_DIR}")
message(STATUS "Source directory:   ${FLEXON_SRC_DIR}")
message(STATUS "Executable output:  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "")
message(STATUS "ORT root:            ${FLEXON_ORT_ROOT}")
message(STATUS "ORT build:           ${FLEXON_ORT_BUILD_DIR}")
message(STATUS "ONNX build:          ${FLEXON_ONNX_BUILD_DIR}")
message(STATUS "")

add_custom_target(BuildFinishedMessage ALL
    COMMAND ${CMAKE_COMMAND} -E echo "==============================================="
    COMMAND ${CMAKE_COMMAND} -E echo "Build Completed successfully!"
    COMMAND ${CMAKE_COMMAND} -E echo "==============================================="
    COMMAND ${CMAKE_COMMAND} -E echo "To Elevate the Flexon Online Scheduler:" 
    COMMAND ${CMAKE_COMMAND} -E echo "Run ./setup_flexon_scheduler.sh"
    COMMAND ${CMAKE_COMMAND} -E echo "==============================================="
)

add_dependencies(BuildFinishedMessage
    flexon_offline
    flexon_online
    flexon_scheduler_helper
    flexon_artifact_handler_test
    flexon_onnx_model_loader_test
    flexon_runtime_helper_test
    flexon_smoke_test
    flexon_buffer_arena_test
    flexon_resource_monitor_test
    flexon_runtime_shape_resolver_test
    flexon_graph_analyzer_test
    flexon_resource_runtime_test
    flexon_scheduler_test
    flexon_model_config_test
    flexon_runtime_engine_test
    flexon_segment_model_generator_test
    flexon_contention
)
