from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("c_utils", ["c_utils.pyx"], include_dirs=[np.get_include()])
]

setup(
    name="c_utils",
    ext_modules=cythonize(extensions)
)

