import glob
import os
from pathlib import Path

EXTENSION_NAME = "heat.models.ops.MultiScaleDeformableAttention"


def should_build_cuda_extension(torch, cuda_home: str | None) -> bool:
    if cuda_home is None:
        return False
    return torch.cuda.is_available() or bool(os.environ.get("TORCH_CUDA_ARCH_LIST"))


def build(setup_kwargs):
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
    except ImportError:
        return

    project_root = Path(__file__).resolve().parents[4]
    extensions_dir = project_root / "src" / "heat" / "models" / "ops" / "src"

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))

    sources = main_file + source_cpu

    include_dirs = [extensions_dir.relative_to(project_root).as_posix()]

    if not should_build_cuda_extension(torch, CUDA_HOME):
        return

    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
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
            "-diag-suppress=20011",
        ],
    }

    extension = CUDAExtension(
        EXTENSION_NAME,
        [Path(s).relative_to(project_root).as_posix() for s in sources],
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )

    setup_kwargs.update(
        {
            "ext_modules": [extension],
            "cmdclass": {"build_ext": BuildExtension.with_options(use_ninja=False)},
        }
    )


if __name__ == "__main__":
    setup_kwargs = {}
    build(setup_kwargs)
    # This is mainly for manual testing or when called by poetry
