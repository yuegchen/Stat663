import setuptools
from distutils.extension import Extension
from Cython.Distutils import build_ext

with open("README.md", "r") as fh:
    long_description = fh.read()

plsr_cython = Extension(name='plsr.plsr_cy.plsr_cython', sources=['plsr/plsr_cy/plsr_cython.pyx'])

setuptools.setup(
    name="Stat_663_Plsr", # Replace with your own username
    version="0.0.1",
    author="Yuege Chen",
    author_email="yuege.chen@gmail.com",
    description="final project of stat 663. An implementation of least squre regression algorithm.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yuegchen/Stat663",
    packages=['plsr'],
    cmdclass={'build_ext':build_ext},
    ext_modules = [plsr_cython],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)