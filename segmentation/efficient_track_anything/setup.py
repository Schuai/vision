# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Our codebase is built based on [SAM2](https://github.com/facebookresearch/sam2). Thanks to SAM 2 authors.

import os

from setuptools import setup

# Package metadata and dependencies are configured in pyproject.toml.

# Following SAM2, we also build the Efficient Track Anything CUDA extension.
# You may turn off CUDA build with `export Efficient_Track_Anything_BUILD_CUDA=0`.
BUILD_CUDA = os.getenv("Efficient_Track_Anything_BUILD_CUDA", "1") == "1"
# Following SAM2, we allow efficient track anything installation to proceed even with build errors.
# You may force stopping on errors with `export Efficient_Track_Anything_BUILD_ALLOW_ERRORS=0`.
BUILD_ALLOW_ERRORS = (
    os.getenv("Efficient_Track_Anything_BUILD_ALLOW_ERRORS", "1") == "1"
)

# Following SAM2, we also catch and skip errors during extension building and print a warning message
# (note that this message only shows up under verbose build mode
# such as "pip install -v -e .")
CUDA_ERROR_MSG = (
    "{}\n\n"
    "Failed to build the Efficient Track Anything CUDA extension due to the error above. "
    "You can still use Efficient Track Anything and it's OK to ignore the error above.\n"
)


def get_extensions():
    if not BUILD_CUDA:
        return []

    try:
        from torch.utils.cpp_extension import CUDAExtension

        srcs = ["efficient_track_anything/csrc/connected_components.cu"]
        compile_args = {
            "cxx": [],
            "nvcc": [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        }
        ext_modules = [
            CUDAExtension(
                "efficient_track_anything._C", srcs, extra_compile_args=compile_args
            )
        ]
    except Exception as e:
        if BUILD_ALLOW_ERRORS:
            print(CUDA_ERROR_MSG.format(e))
            ext_modules = []
        else:
            raise e

    return ext_modules


try:
    from torch.utils.cpp_extension import BuildExtension

    class BuildExtensionIgnoreErrors(BuildExtension):

        def finalize_options(self):
            try:
                super().finalize_options()
            except Exception as e:
                print(CUDA_ERROR_MSG.format(e))
                self.extensions = []

        def build_extensions(self):
            try:
                super().build_extensions()
            except Exception as e:
                print(CUDA_ERROR_MSG.format(e))
                self.extensions = []

        def get_ext_filename(self, ext_name):
            try:
                return super().get_ext_filename(ext_name)
            except Exception as e:
                print(CUDA_ERROR_MSG.format(e))
                self.extensions = []
                return "_C.so"

    cmdclass = {
        "build_ext": (
            BuildExtensionIgnoreErrors.with_options(no_python_abi_suffix=True)
            if BUILD_ALLOW_ERRORS
            else BuildExtension.with_options(no_python_abi_suffix=True)
        )
    }
except Exception as e:
    cmdclass = {}
    if BUILD_ALLOW_ERRORS:
        print(CUDA_ERROR_MSG.format(e))
    else:
        raise e


setup(
    ext_modules=get_extensions(),
    cmdclass=cmdclass,
)
