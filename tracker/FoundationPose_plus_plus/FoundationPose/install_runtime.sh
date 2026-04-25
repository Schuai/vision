#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
PYTORCH3D_REF="${PYTORCH3D_REF:-stable}"
ENV_PREFIX="$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)"
export PATH="${ENV_PREFIX}/bin:${PATH}"

if [ -z "${CUDA_HOME:-}" ] && [ -x "${ENV_PREFIX}/bin/nvcc" ]; then
  export CUDA_HOME="${ENV_PREFIX}"
  export CUDA_PATH="${ENV_PREFIX}"
  export CUDACXX="${ENV_PREFIX}/bin/nvcc"
elif [ -z "${CUDA_HOME:-}" ] && [ -d "/usr/local/cuda-12.6" ]; then
  export CUDA_HOME="/usr/local/cuda-12.6"
  export CUDA_PATH="/usr/local/cuda-12.6"
  export CUDACXX="/usr/local/cuda-12.6/bin/nvcc"
fi

echo "Installing FoundationPose runtime extras into ${PYTHON_BIN}..."
if [ -n "${CUDA_HOME:-}" ]; then
  echo "Using CUDA toolkit at ${CUDA_HOME}"
fi

if [ -z "${CC:-}" ]; then
  for candidate in \
    "${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-cc" \
    "${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
  do
    if [ -x "${candidate}" ]; then
      export CC="${candidate}"
      break
    fi
  done
fi

if [ -z "${CXX:-}" ]; then
  for candidate in \
    "${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-c++" \
    "${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
  do
    if [ -x "${candidate}" ]; then
      export CXX="${candidate}"
      break
    fi
  done
fi

if [ -n "${CXX:-}" ] && [ -z "${CUDAHOSTCXX:-}" ]; then
  export CUDAHOSTCXX="${CXX}"
fi

CUDA_INCLUDE_DIR="${CUDA_HOME:-}/targets/x86_64-linux/include"
CUDA_LIB_DIR="${CUDA_HOME:-}/targets/x86_64-linux/lib"
CUDA_STUB_DIR="${CUDA_HOME:-}/targets/x86_64-linux/lib/stubs"
if [ -d "${CUDA_INCLUDE_DIR}" ]; then
  export CPATH="${CUDA_INCLUDE_DIR}${CPATH:+:${CPATH}}"
  export C_INCLUDE_PATH="${CUDA_INCLUDE_DIR}${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
  export CPLUS_INCLUDE_PATH="${CUDA_INCLUDE_DIR}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
fi
if [ -d "${CUDA_LIB_DIR}" ]; then
  export LIBRARY_PATH="${CUDA_LIB_DIR}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
fi
if [ -d "${CUDA_STUB_DIR}" ]; then
  export LIBRARY_PATH="${CUDA_STUB_DIR}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
fi

if [ -n "${CC:-}" ]; then
  echo "Using host C compiler at ${CC}"
fi
if [ -n "${CXX:-}" ]; then
  echo "Using host C++ compiler at ${CXX}"
fi

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel ninja

NVDIFFRAST_SRC_DIR="$(mktemp -d)"
trap 'rm -rf "${NVDIFFRAST_SRC_DIR}"' EXIT
git clone --filter=blob:none https://github.com/NVlabs/nvdiffrast.git "${NVDIFFRAST_SRC_DIR}"
"${PYTHON_BIN}" - "${NVDIFFRAST_SRC_DIR}/setup.py" <<'PY'
from pathlib import Path
import sys

setup_py = Path(sys.argv[1])
text = setup_py.read_text()
old = '"nvcc": ["-DNVDR_TORCH", "-lineinfo"],'
new = '"nvcc": ["-DNVDR_TORCH", "-lineinfo", "-allow-unsupported-compiler"],'
if old not in text:
    raise SystemExit("Failed to patch nvdiffrast setup.py for the local compiler override.")
setup_py.write_text(text.replace(old, new))
PY
"${PYTHON_BIN}" -m pip install --no-build-isolation "${NVDIFFRAST_SRC_DIR}"
"${PYTHON_BIN}" -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@${PYTORCH3D_REF}"

echo "FoundationPose runtime extras installed successfully."
