import os
import setuptools

with open("README.md") as f:
    longDescription = f.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="hardware-tools",
    version="0.0.0",
    description="A library for automating hardware development and testing",
    long_description=longDescription,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=[
        "hardware_tools",
        "hardware_tools.equipment",
        "hardware_tools.measurement"
    ],
    package_data={"hardware_tools": []},
    install_requires=required,
    test_suite='tests',
    scripts=[],
    author="Bradley Davis",
    author_email="me@bradleydavis.tech",
    url="https://github.com/WattsUp/hardware-tools",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.6",
    include_package_data=True,
    zip_safe=True,
)
