#!/usr/bin/env bash

set -euo pipefail

PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
PYBIND11_CMAKE_DIR="$("${PYTHON_BIN}" -c 'import pybind11; print(pybind11.get_cmake_dir())')"

echo "Building optional mycpp extension..."
cmake -S "${PROJ_ROOT}/mycpp" \
  -B "${PROJ_ROOT}/mycpp/build" \
  -DPython_EXECUTABLE="${PYTHON_BIN}" \
  -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}"
cmake --build "${PROJ_ROOT}/mycpp/build" --parallel "$(nproc)"

if [ -d "${PROJ_ROOT}/bundlesdf/mycuda" ]; then
  echo "Installing optional bundlesdf/mycuda extension..."
  cd "${PROJ_ROOT}/bundlesdf/mycuda"
  rm -rf build *egg* *.so
  "${PYTHON_BIN}" -m pip install -e .
else
  echo "Skipping bundlesdf/mycuda: directory not present in this checkout."
fi
