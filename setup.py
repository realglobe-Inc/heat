import importlib.util
from pathlib import Path

from setuptools import find_packages, setup


def load_ops_build_module():
    build_path = Path(__file__).parent / "src" / "heat" / "models" / "ops" / "build.py"
    spec = importlib.util.spec_from_file_location("heat_ops_build", build_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load extension build script: {build_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup_kwargs = {
    "package_dir": {"": "src"},
    "packages": find_packages("src"),
}

load_ops_build_module().build(setup_kwargs)

setup(**setup_kwargs)
