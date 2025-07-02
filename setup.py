#!/usr/bin/env python
"""
Setup script for the EarthMind-TerraTorch integration package
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="earthmind-terratorch",
    version="0.1.0",
    author="Colm Keyes",
    author_email="colm.keyes@example.com",
    description="Integration of EarthMind with TerraTorch for geospatial backbone experimentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/colm-keyes/earthmind-terratorch",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "test-earthmind=scripts.test_earthmind:main",
            "benchmark-backbones=scripts.benchmark_backbones:main",
            "optimize-large-models=scripts.optimize_large_models:main",
        ],
    },
)
