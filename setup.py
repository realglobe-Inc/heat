from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import os
import sys

def read_requirements():
    requirements = []
    dependency_links = []

    with open('requirements.txt') as req_file:
        for line in req_file:
            line = line.strip()
            if line.startswith('-f'):
                dependency_links.append(line[3:].strip())
            elif line and not line.startswith('#'):
                requirements.append(line)

    return requirements, dependency_links

requirements, dependency_links = read_requirements()

class CustomInstallCommand(install):
    def run(self):
        # setup.pyの実行
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        setup_script_path = os.path.join(current_dir, 'models', 'ops', 'setup.py')
        subprocess.check_call([sys.executable, setup_script_path, 'build', 'install', '--user'])
        
        # 継続してインストール
        install.run(self)

setup(
    name="heat",
    version="1.0",
    author="realglobe",
    url="https://github.com/realglobe-Inc/heat",
    description="PyTorch Wrapper for CUDA Functions of Multi-Scale Deformable Attention",
    packages=find_packages(where='.', exclude=("datasets", "models", "utils")),
    package_dir={'': '.'},
    dependency_links=dependency_links,
    # cmdclass={
    #     'install': CustomInstallCommand,
    # }
)
