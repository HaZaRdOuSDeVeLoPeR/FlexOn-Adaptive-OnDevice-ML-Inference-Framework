cmake_minimum_required(VERSION 3.20)

project(FlexOn VERSION 0.1.0 LANGUAGES CXX)

# ============================================================================
# Configuration
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FLEXON_BUILD_TESTS "Build FlexOn tests" ON)
option(FLEXON_BUILD_APPS "Build FlexOn applications" ON)

# All executables are generated into the project apps/ directory.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_SOURCE_DIR}/apps"
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
# Online library
# ============================================================================

# src/online/ is currently empty.
# Convert this to STATIC once online implementation sources are added.

add_library(
    flexon_online_lib
    INTERFACE
)

target_include_directories(
    flexon_online_lib
    INTERFACE
        "${FLEXON_INCLUDE_DIR}"
)

target_link_libraries(
    flexon_online_lib
    INTERFACE
        flexon_core
        flexon_onnxruntime
)

# ============================================================================
# Applications
# ============================================================================

if(FLEXON_BUILD_APPS)

    # Final FlexOn application
    add_executable(
        flexon
        src/flexon_online.cpp
    )

    target_link_libraries(
        flexon
        PRIVATE
            flexon_core
            flexon_online_lib
    )

    # Offline phase
    add_executable(
        flexon_offline
        src/flexon_offline.cpp
    )

    target_link_libraries(
        flexon_offline
        PRIVATE
            flexon_core
            flexon_offline_lib
    )

    # Offline tester
    add_executable(
        flexon_offline_tester
        src/flexon_offline_tester.cpp
    )

    target_link_libraries(
        flexon_offline_tester
        PRIVATE
            flexon_offline_lib
    )

    # Online tester
    add_executable(
        flexon_online_tester
        src/flexon_online_tester.cpp
    )

    target_link_libraries(
        flexon_online_tester
        PRIVATE
            flexon_online_lib
    )

endif()

# ============================================================================
# Tests
# ============================================================================

if(FLEXON_BUILD_TESTS)

    enable_testing()

    # ------------------------------------------------------------------------
    # Offline tests
    # ------------------------------------------------------------------------

    add_executable(
        flexon_onnx_model_loader_test
        tests/offline/onnx_model_loader_test.cpp
    )

    target_link_libraries(
        flexon_onnx_model_loader_test
        PRIVATE
            flexon_offline_lib
    )

    add_test(
        NAME flexon_onnx_model_loader_test
        COMMAND flexon_onnx_model_loader_test
    )

    add_executable(
        flexon_graph_analyzer_test
        tests/offline/graph_analyzer_test.cpp
    )

    target_link_libraries(
        flexon_graph_analyzer_test
        PRIVATE
            flexon_offline_lib
    )

    add_test(
        NAME flexon_graph_analyzer_test
        COMMAND flexon_graph_analyzer_test
    )

    add_executable(
        flexon_segment_model_generator_test
        tests/offline/segment_model_generator_test.cpp
    )

    add_executable(
        flexon_runtime_shape_resolver_test
        tests/offline/runtime_shape_resolver_test.cpp
    )

    target_link_libraries(
        flexon_runtime_shape_resolver_test
        PRIVATE
            flexon_offline_lib
    )

    add_test(
        NAME flexon_runtime_shape_resolver_test
        COMMAND flexon_runtime_shape_resolver_test
    )

    add_executable(
        flexon_model_config_test
        tests/offline/model_config_test.cpp
    )

    target_link_libraries(
        flexon_model_config_test
        PRIVATE
            flexon_offline_lib
    )

    add_test(
        NAME flexon_model_config_test
        COMMAND flexon_model_config_test
    )

    target_link_libraries(
        flexon_segment_model_generator_test
        PRIVATE
            flexon_offline_lib
    )

    add_test(
        NAME flexon_segment_model_generator_test
        COMMAND flexon_segment_model_generator_test
    )

    # ------------------------------------------------------------------------
    # General smoke test
    # ------------------------------------------------------------------------

    add_executable(
        flexon_smoke_test
        tests/smoke_test.cpp
    )

    target_link_libraries(
        flexon_smoke_test
        PRIVATE
            flexon_core
    )

    add_test(
        NAME flexon_smoke_test
        COMMAND flexon_smoke_test
    )

endif()

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "========== FlexOn ${PROJECT_VERSION} ==========")
message(STATUS "C++ standard:       C++${CMAKE_CXX_STANDARD}")
message(STATUS "Build tests:        ${FLEXON_BUILD_TESTS}")
message(STATUS "Build applications: ${FLEXON_BUILD_APPS}")
message(STATUS "")
message(STATUS "Include directory:  ${FLEXON_INCLUDE_DIR}")
message(STATUS "Source directory:   ${FLEXON_SRC_DIR}")
message(STATUS "Executable output:  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "")
message(STATUS "ORT root:            ${FLEXON_ORT_ROOT}")
message(STATUS "ORT build:           ${FLEXON_ORT_BUILD_DIR}")
message(STATUS "ONNX build:          ${FLEXON_ONNX_BUILD_DIR}")
message(STATUS "==============================================")
message(STATUS "")