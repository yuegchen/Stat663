from setuptools import setup
from Cython.Build import cythonize

setup(
	name="plsr_cython",
    ext_modules=cythonize("plsr_cython.pyx"),
    zip_safe=False
)