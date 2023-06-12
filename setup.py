import re
import subprocess
import sys

import setuptools
from setuptools import find_packages

# groundingdino needs torch to be installed before it can be installed
# this is a hack but couldn't find any other way to make it work
try:
    import torch
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])

with open("./autodistill_sam_clip/__init__.py", "r") as f:
    content = f.read()
    # from https://www.py4u.net/discuss/139845
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content).group(1)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autodistill-sam-clip",
    version=version,
    author="Roboflow",
    author_email="support@roboflow.com",
    description="SAM-CLIP model for use with Autodistill",
    long_description_content_type="text/markdown",
    url="https://github.com/autodistill/autodistill-sam-clip",
    install_requires=[
        "torch",
        "autodistill",
        "numpy>=1.20.0",
        "opencv-python>=4.6.0",
        "rf_groundingdino",
        "rf_segment_anything",
        "supervision",
    ],
    packages=find_packages(exclude=("tests",)),
    extras_require={
        "dev": ["flake8", "black==22.3.0", "isort", "twine", "pytest", "wheel"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
