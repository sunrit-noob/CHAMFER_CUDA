import os
from setuptools import setup, find_packages
from chamferCuda import __version__

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="chamfer_cuda",
    version=__version__,
    author="Sunrit Sarkar",
    author_email="sunrit2022pers@gmail.com",
    description="A PyTorch extension for computing Chamfer Distance using CUDA.",
    packages=find_packages(),
    package_data={
        "chamferCuda": ["src/chamfer.cpp", "src/chamfer.cu"]
    },
    install_requires=requirements
)