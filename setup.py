import sys

from setuptools import setup, Extension
from Cython.Build import cythonize

# Optimize the compiled extension: MSVC (Windows) and GCC/Clang (POSIX) take
# different flags. -march=native is invalid on MSVC, so pick per-platform.
if sys.platform.startswith("win"):
    extra_compile_args = ["/O2"]
else:
    extra_compile_args = ["-O3", "-march=native"]

setup(
    name="evosim-evolution-core",
    version="1.0",
    description="Cython-accelerated CPU evolution core for evosim.",
    ext_modules=cythonize([
        Extension(
            "evolution_core",
            ["evolution_core.pyx"],
            extra_compile_args=extra_compile_args,
            language="c",
        )
    ], language_level=3),
)