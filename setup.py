import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Stat-663-Plsr", # Replace with your own username
    version="0.0.1",
    author="Yuege Chen",
    author_email="yuege.chen@gmail.com",
    description="final project of stat 663. An implementation of least squre regression algorithm.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yuegchen/Stat663",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)