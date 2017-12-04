from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize("_matutils.pyx", annotate=True),
    include_dirs = [np.get_include()],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)