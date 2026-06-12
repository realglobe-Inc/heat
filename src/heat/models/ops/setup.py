# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

import glob
import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension

requirements = ["torch", "torchvision"]
EXTENSION_NAME = "heat.models.ops.MultiScaleDeformableAttention"


def should_build_cuda_extension() -> bool:
    if CUDA_HOME is None:
        return False
    return torch.cuda.is_available() or bool(os.environ.get("TORCH_CUDA_ARCH_LIST"))


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    if not should_build_cuda_extension():
        print("CUDA extension build skipped; inference will use the PyTorch fallback.")
        return []

    sources += source_cuda
    define_macros = [("WITH_CUDA", True)]
    extra_compile_args = {
        "cxx": ["-Wno-error", "-w"],
        "nvcc": [
            "-DCUDA_HAS_FP16=1",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-Xcompiler",
            "-fno-strict-aliasing",
            "-Xcompiler",
            "-Wno-error",
            "-Xcompiler",
            "-w",
            "-Xcompiler",
            "-fpermissive",
            "-diag-suppress=20011",  # Support for CUDA 13.x sinpi/cospi warnings
        ],
    }

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    print("sources:", sources)
    print("include_dirs:", include_dirs)
    print("extension", CUDAExtension)
    print("define_macros", define_macros)
    print("extra_compile_args", extra_compile_args)
    ext_modules = [
        CUDAExtension(
            EXTENSION_NAME,
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules


setup(
    name="MultiScaleDeformableAttention",
    version="1.0",
    package_dir={"heat.models.ops": "."},
    author="Weijie Su",
    url="https://github.com/fundamentalvision/Deformable-DETR",
    description="PyTorch Wrapper for CUDA Functions of Multi-Scale Deformable Attention",
    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": torch.utils.cpp_extension.BuildExtension.with_options(
            use_ninja=False
        )
    },
)
