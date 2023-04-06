
import setuptools
import re
import os
import sys


setuptools.setup(
    name="neural-diffeqs",
    version="0.2.1rc",
    python_requires=">3.6.0",
    author="Michael E. Vinyard - Harvard University - Broad Institute of MIT and Harvard - Massachussetts General Hospital",
    author_email="mvinyard@broadinstitute.org",
    url=None,
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    description="Neural differential equations made easy.",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy==1.22.4",
        "torch==1.13",
        "torch-nets>=0.0.2",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.6",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    license="MIT",
)
