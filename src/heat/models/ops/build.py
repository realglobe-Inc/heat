import glob
import os


def build(setup_kwargs):
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
    except ImportError:
        # ビルド環境のセットアップ中などで torch がまだ利用できない場合はスキップ
        return

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))

    # Always include CPU sources
    sources = main_file + source_cpu

    include_dirs = [extensions_dir]
    define_macros = []
    extra_compile_args = {"cxx": ["-Wno-error", "-w"]}

    if torch.cuda.is_available() and CUDA_HOME is not None:
        source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
        sources += source_cuda
        define_macros += [("WITH_CUDA", True)]
        extra_compile_args["nvcc"] = [
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
        ]
        extension_class = CUDAExtension
    else:
        # Fallback to CPU only if CUDA is not available
        extension_class = torch.utils.cpp_extension.CppExtension

    extension = extension_class(
        "MultiScaleDeformableAttention",
        [os.path.join(extensions_dir, s) for s in sources],
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
