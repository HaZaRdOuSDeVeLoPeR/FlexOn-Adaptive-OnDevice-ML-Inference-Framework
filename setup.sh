#!/usr/bin/env bash
set -euo pipefail

# FlexOn reproducible environment bootstrap.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BUILD_ENV="$ROOT/config/build.env"

if [[ ! -f "$BUILD_ENV" ]]; then
    echo "Missing build configuration: $BUILD_ENV" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$BUILD_ENV"

: "${ORT_VERSION:?ORT_VERSION is not set in config/build.env}"
: "${ONNX_VERSION:?ONNX_VERSION is not set in config/build.env}"
: "${YAML_CPP_VERSION:?YAML_CPP_VERSION is not set in config/build.env}"
: "${CUDNN_VERSION:?CUDNN_VERSION is not set in config/build.env}"
: "${ORT_BUILD_JOBS:?ORT_BUILD_JOBS is not set in config/build.env}"
: "${ORT_EIGEN_SHA1_OLD:?ORT_EIGEN_SHA1_OLD is not set in config/build.env}"
: "${ORT_EIGEN_SHA1_NEW:?ORT_EIGEN_SHA1_NEW is not set in config/build.env}"

MODEL_ENV="$ROOT/config/models.env"
if [[ ! -f "$MODEL_ENV" ]]; then
    echo "Missing model configuration: $MODEL_ENV" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$MODEL_ENV"

: "${RESNET18_URL:?RESNET18_URL is not set in config/models.env}"
: "${EFFICIENTNET_LITE4_URL:?EFFICIENTNET_LITE4_URL is not set in config/models.env}"
: "${YOLOV4_TINY_URL:?YOLOV4_TINY_URL is not set in config/models.env}"

echo "FlexOn setup"
echo "============"
echo "Project root: $ROOT"
echo
echo "Target: Linux/WSL2 + NVIDIA CUDA + CPU/CUDA execution providers"
echo

command -v git >/dev/null || { echo "git is required"; exit 1; }
command -v cmake >/dev/null || { echo "cmake is required"; exit 1; }
command -v c++ >/dev/null || { echo "a C++ compiler is required"; exit 1; }
command -v nvcc >/dev/null || { echo "nvcc is required for the CUDA backend"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required"; exit 1; }
command -v curl >/dev/null || { echo "curl is required for model downloads"; exit 1; }

mkdir -p third_party

clone_or_update() {
    local url="$1"
    local dir="$2"
    local ref="$3"

    if [[ ! -d "$dir/.git" ]]; then
        git -c http.version=HTTP/1.1 clone \
            --depth 1 \
            --branch "$ref" \
            --recursive \
            "$url" \
            "$dir"
        return 0
    fi

    git -c http.version=HTTP/1.1 \
        -C "$dir" fetch --depth 1 origin "$ref"

    git -C "$dir" checkout --detach FETCH_HEAD

    git -c http.version=HTTP/1.1 \
        -C "$dir" submodule update --init --recursive --depth 1
}

clone_or_update \
    "https://github.com/jbeder/yaml-cpp.git" \
    "third_party/yaml-cpp" \
    "$YAML_CPP_VERSION"

# ---------------------------------------------------------------------------
# CUDA toolkit discovery
# ---------------------------------------------------------------------------

detect_cuda_home() {
    if [[ -n "${CUDA_HOME:-}" ]]; then
        if [[ -x "${CUDA_HOME}/bin/nvcc" ]]; then
            echo "$CUDA_HOME"
            return 0
        fi

        echo "CUDA_HOME is set to '$CUDA_HOME', but '$CUDA_HOME/bin/nvcc' does not exist." >&2
        return 1
    fi

    if [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
        echo "/usr/local/cuda"
        return 0
    fi

    if [[ -x "/usr/lib/nvidia-cuda-toolkit/bin/nvcc" ]]; then
        echo "/usr/lib/nvidia-cuda-toolkit"
        return 0
    fi

    echo "CUDA toolkit not found." >&2
    echo "Expected either:" >&2
    echo "  /usr/local/cuda/bin/nvcc" >&2
    echo "  /usr/lib/nvidia-cuda-toolkit/bin/nvcc" >&2
    echo "or set CUDA_HOME explicitly." >&2
    return 1
}

CUDA_HOME="$(detect_cuda_home)"
export CUDA_HOME

echo "CUDA toolkit: $CUDA_HOME"
echo "CUDA compiler: $CUDA_HOME/bin/nvcc"
"$CUDA_HOME/bin/nvcc" --version | tail -1
echo

# ---------------------------------------------------------------------------
# cuDNN discovery
# ---------------------------------------------------------------------------

detect_cudnn_home() {
    # Explicit CUDNN_HOME supplied by the user.
    if [[ -n "${CUDNN_HOME:-}" ]]; then
        echo "$CUDNN_HOME"
        return 0
    fi

    # Debian/Ubuntu packaged cuDNN.
    #
    # Headers:
    #   /usr/include/x86_64-linux-gnu/cudnn.h
    #
    # Libraries:
    #   /usr/lib/x86_64-linux-gnu/libcudnn.so.9
    if dpkg-query -W -f='${Status}' libcudnn9-cuda-12 2>/dev/null |
           grep -q 'install ok installed'; then
        echo "/usr"
        return 0
    fi

    # Conventional NVIDIA CUDA installation.
    if [[ -f "/usr/local/cuda/include/cudnn.h" ]] &&
       compgen -G "/usr/local/cuda/lib64/libcudnn.so*" >/dev/null 2>&1; then
        echo "/usr/local/cuda"
        return 0
    fi

    echo "cuDNN not found." >&2
    echo "Install a compatible cuDNN package or set CUDNN_HOME explicitly." >&2
    return 1
}

CUDNN_HOME="$(detect_cudnn_home)"
export CUDNN_HOME

if [[ "$CUDNN_HOME" == "/usr" ]]; then
    CUDNN_INCLUDE_DIR="/usr/include/x86_64-linux-gnu"
    CUDNN_LIBRARY_DIR="/usr/lib/x86_64-linux-gnu"
elif [[ -d "$CUDNN_HOME/include" ]]; then
    CUDNN_INCLUDE_DIR="$CUDNN_HOME/include"

    if [[ -d "$CUDNN_HOME/lib64" ]]; then
        CUDNN_LIBRARY_DIR="$CUDNN_HOME/lib64"
    else
        CUDNN_LIBRARY_DIR="$CUDNN_HOME/lib"
    fi
else
    echo "Unable to determine cuDNN include directory under: $CUDNN_HOME" >&2
    exit 1
fi

if [[ ! -f "$CUDNN_INCLUDE_DIR/cudnn.h" ]]; then
    echo "cuDNN header not found: $CUDNN_INCLUDE_DIR/cudnn.h" >&2
    exit 1
fi

if ! compgen -G "$CUDNN_LIBRARY_DIR/libcudnn.so*" >/dev/null 2>&1; then
    echo "cuDNN runtime libraries not found under: $CUDNN_LIBRARY_DIR" >&2
    exit 1
fi

if [[ "$CUDNN_HOME" == "/usr" ]]; then
    CUDNN_LIBRARY_DIR="/usr/lib/x86_64-linux-gnu"
elif [[ -d "$CUDNN_HOME/lib64" ]]; then
    CUDNN_LIBRARY_DIR="$CUDNN_HOME/lib64"
else
    CUDNN_LIBRARY_DIR="$CUDNN_HOME/lib"
fi

if ! compgen -G "$CUDNN_LIBRARY_DIR/libcudnn.so*" >/dev/null 2>&1; then
    echo "cuDNN runtime libraries not found under: $CUDNN_LIBRARY_DIR" >&2
    exit 1
fi

installed_cudnn_version="$(
    dpkg-query -W -f='${Version}' libcudnn9-cuda-12 2>/dev/null || true
)"

if [[ -n "$installed_cudnn_version" ]]; then
    if [[ "$installed_cudnn_version" != "$CUDNN_VERSION" ]]; then
        echo "cuDNN version mismatch: expected $CUDNN_VERSION, found $installed_cudnn_version" >&2
        exit 1
    fi

    echo "cuDNN: $installed_cudnn_version"
else
    echo "cuDNN: non-Debian installation detected under $CUDNN_HOME"
fi

echo "cuDNN root: $CUDNN_HOME"
echo "cuDNN headers: $CUDNN_INCLUDE_DIR"
echo "cuDNN libraries: $CUDNN_LIBRARY_DIR"
echo

# ---------------------------------------------------------------------------
# ONNX Runtime
# ---------------------------------------------------------------------------

clone_or_update \
    "https://github.com/microsoft/onnxruntime.git" \
    "third_party/onnxruntime-src" \
    "$ORT_VERSION"

# ---------------------------------------------------------------------------
# ORT dependency corrections
# ---------------------------------------------------------------------------

ORT_DEPS_FILE="third_party/onnxruntime-src/cmake/deps.txt"

if [[ ! -f "$ORT_DEPS_FILE" ]]; then
    echo "ONNX Runtime dependency file not found: $ORT_DEPS_FILE" >&2
    exit 1
fi

if grep -q "$ORT_EIGEN_SHA1_OLD" "$ORT_DEPS_FILE"; then
    echo "Applying ORT Eigen dependency hash correction..."

    sed -i \
        "s/${ORT_EIGEN_SHA1_OLD}/${ORT_EIGEN_SHA1_NEW}/" \
        "$ORT_DEPS_FILE"
fi

if ! grep -q "$ORT_EIGEN_SHA1_NEW" "$ORT_DEPS_FILE"; then
    echo "Expected Eigen SHA1 not found after dependency correction." >&2
    exit 1
fi

pushd third_party/onnxruntime-src >/dev/null

./build.sh \
    --config Release \
    --update \
    --build \
    --parallel "$ORT_BUILD_JOBS" \
    --build_shared_lib \
    --skip_tests \
    --use_cuda \
    --cuda_home "$CUDA_HOME" \
    --cudnn_home "$CUDNN_HOME" \
    --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=native"

popd >/dev/null

echo
echo "ONNX Runtime CUDA build completed."
# ---------------------------------------------------------------------------
# Evaluation models
# ---------------------------------------------------------------------------
#
# The benchmark corpus is configured in config/models.env. The default models
# are the ONNX Model Zoo ResNet18, ONNX Model Zoo EfficientNet-Lite4, and the
# 416x416 YOLOv4-tiny ONNX model published by Kalray.

download_model() {
    local url="$1"
    local destination="$2"

    mkdir -p "$(dirname "$destination")"

    if [[ -s "$destination" ]]; then
        echo "Model already exists: $destination"
        return 0
    fi

    echo "Downloading $(basename "$destination")..."
    curl -L --fail --retry 3 --retry-delay 2 \
        "$url" \
        -o "$destination"
}

download_model "$RESNET18_URL" "$RESNET18_FILE"
download_model "$EFFICIENTNET_LITE4_URL" "$EFFICIENTNET_LITE4_FILE"
download_model "$YOLOV4_TINY_URL" "$YOLOV4_TINY_FILE"

echo
echo "Evaluation models are available under: $ROOT/models"

echo
echo "Environment bootstrap completed."
echo "Do not start profiling until the backend smoke test passes."