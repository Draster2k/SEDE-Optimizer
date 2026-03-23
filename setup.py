import sys
from setuptools import setup, Extension
import pybind11
import os

extra_compile_args = ['-O3', '-std=c++11']
extra_link_args = []

if sys.platform.startswith("linux"):
    extra_compile_args += ['-fopenmp']
    extra_link_args += ['-fopenmp']
elif sys.platform == "darwin":
    # Defensive macOS OpenMP check for Apple Silicon
    if os.path.exists('/opt/homebrew/opt/libomp/include'):
        extra_compile_args += ['-Xpreprocessor', '-fopenmp', '-I/opt/homebrew/opt/libomp/include']
        extra_link_args += ['-L/opt/homebrew/opt/libomp/lib', '-lomp']
    # Intel Mac fallback check
    elif os.path.exists('/usr/local/opt/libomp/include'):
        extra_compile_args += ['-Xpreprocessor', '-fopenmp', '-I/usr/local/opt/libomp/include']
        extra_link_args += ['-L/usr/local/opt/libomp/lib', '-lomp']

ext_modules = [
    Extension(
        "sede_core",
        ["src/sede_core.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++"
    ),
]

setup(
    name="sede_core",
    version="1.0",
    ext_modules=ext_modules,
)
