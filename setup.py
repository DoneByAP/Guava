from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README if it exists
readme_path = Path(__file__).parent / "README.md"
long_description = (
    readme_path.read_text(encoding="utf-8")
    if readme_path.exists()
    else "Guava: distributed neural network training across multiple GPUs / machines."
)

setup(
    name="guava",
    version="0.1.2",
    author="Azani",
    description="Distributed neural network training across multiple GPUs and machines (data parallel, model parallel, pipeline parallel, tensor parallel).",
    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=find_packages(),
    python_requires=">=3.8",

    install_requires=[
        "numpy",
        "tqdm",
        "psutil",
        # torch purposely not included — users install correct CUDA wheel from PyTorch site
    ],

    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
        ],
    },

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
